"""License-gated, resumable orchestration for the offline-verified v2 workflow.

The module contains no provider implementation and performs no request unless all
three provider stages are explicitly injected and a local written-authorization
record covers every declared use. Google details/reviews and exact model traffic are
kept in memory only.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from skills.wayback_agent import WaybackServiceUnavailable
from wayback_state import WaybackDeferred

from v2_classifier import (
    PROMPT_TEMPLATE_SHA256,
    PROMPT_VERSION,
    build_candidate_facts,
    build_review_evidence,
    is_target_match,
)
from v2_evidence_models import (
    ClassifierContractV2,
    ClassifierInputV2,
    LLMDecisionV2,
    QueryProvenanceV2,
    WaybackCollectionV2,
)
from v2_downstream_models import CandidateLocationV2, candidate_location_from_details
from v2_delivery import (
    DeliveryCandidateV2,
    PilotDeliveryInputsV2,
    _checkpoint_decision,
    _validate_archive_checkpoint,
    authenticate_pilot_ablation_package,
    run_pilot_v2_delivery,
)
from v2_eligibility import classification_decision_sha256
from v2_pilot import (
    run_ablation,
    select_representative_candidates,
    summarize_ablation,
)


RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
PLACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUTH_SCHEMA = "google-places-research-authorization-v2"
WORKFLOW_SCHEMA = "nets-ai-wayback-llm-v2"
COMPLETED_UNIT_SCHEMA = "nets-ai-v2-completed-candidate-v1"
PROMOTION_SCHEMA = "nets-ai-v2-pilot-promotion-v2"
AUTHORIZED_SERVICE_NAMES = (
    "places_api_new_text_search", "places_api_new_place_details",
)
AUTHORIZED_SERVICES = frozenset(AUTHORIZED_SERVICE_NAMES)
REQUIRED_PERMISSION_NAMES = (
    "batch_retrieval", "transmit_to_openai", "retain_place_ids",
    "retain_derived_classifications", "wayback_linkage", "reference_linkage",
    "spatial_analysis", "thesis_publication",
)
REQUIRED_PERMISSIONS = frozenset(REQUIRED_PERMISSION_NAMES)
DERIVED_OUTPUTS = (
    "classification_v2", "record_linkage", "spatial_analysis",
    "honors_thesis_repository_disclosure",
)
RETENTION_RULE = "place_id_and_authorized_derived_outputs_only"
RETENTION_EXPIRES_AT = "2027-12-31T23:59:59Z"
PUBLICATION_SCOPE = "honors_thesis_and_repository_outputs"
AUTHORIZATION_FIELDS = frozenset({
    "schema_version", "authorization_id", "project", "effective_at",
    "expires_at", "written_permission_file", "written_permission_sha256",
    "services", "third_party_processor", "retention_rule",
    "publication_scope", "openai_store_false", "permissions",
    "places_search_field_mask", "places_details_field_mask",
    "query_definition_ids", "candidate_schema_sha256",
    "prompt_template_sha256", "classifier_contract_sha256",
    "workflow_contract_sha256", "retention_expires_at", "derived_outputs",
})
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_PATHS = (
    "code/run_v2_workflow.py",
    "code/v2_evidence_models.py",
    "code/v2_classifier.py",
    "code/v2_workflow.py",
    "code/v2_delivery.py",
    "code/v2_downstream_models.py",
    "code/v2_eligibility.py",
    "code/v2_board_adapter.py",
    "code/v2_spatial_adapter.py",
    "code/v2_pilot.py",
    "code/wayback_runtime.py",
    "code/wayback_state.py",
    "code/outscraper_wayback.py",
    "skills/google_maps.py",
    "skills/wayback_agent.py",
    "docs/v2_candidate_schema.json",
    "docs/v2_candidate_occurrence_schema.json",
    "docs/v2_eligibility_schema.json",
    "requirements-analysis.txt",
)
PROVIDER_POLICY_URLS = (
    "https://cloud.google.com/maps-platform/terms",
    "https://developers.google.com/maps/documentation/places/web-service/policies",
    "https://developers.openai.com/api/docs/guides/your-data",
)
THESIS_STAGE_COMPLIANCE_WARNING = (
    "Thesis-stage live execution is user-authorized by AGENTS.md; this run does "
    "not claim provider-issued permission or authorize later publication/release."
)


class AuthorizationGateError(RuntimeError):
    """Raised before provider calls when written authorization is not authenticated."""


@dataclass(frozen=True, slots=True, repr=False)
class V2WorkflowConfig:
    run_id: str
    task: str
    city: str
    zips: tuple[str, ...]
    output_root: Path
    authorization_file: Path
    model: str
    dry_run: bool = False
    resume: bool = False
    limit: int | None = None
    run_mode: str = "smoke"
    pilot_size: int = 8
    promotion_file: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or RUN_ID_RE.fullmatch(self.run_id) is None:
            raise ValueError("run_id must use YYYYMMDDTHHMMSSZ")
        try:
            datetime.strptime(self.run_id, "%Y%m%dT%H%M%SZ")
        except ValueError as exc:
            raise ValueError("run_id is not a valid UTC timestamp") from exc
        if self.task != "pharmacy":
            raise ValueError("v2 currently authenticates only the pharmacy test case")
        if (not isinstance(self.city, str) or not self.city.strip()
                or len(self.city) > 200
                or any(ord(char) < 32 or ord(char) == 127 for char in self.city)):
            raise ValueError("city is required")
        if (not isinstance(self.zips, (tuple, list)) or not self.zips
                or any(not isinstance(item, str)
                       or re.fullmatch(r"\d{5}", item) is None for item in self.zips)
                or len(set(self.zips)) != len(self.zips)):
            raise ValueError("invalid ZIP frame")
        if not isinstance(self.output_root, Path) or not isinstance(self.authorization_file, Path):
            raise TypeError("workflow paths must be pathlib.Path values")
        if (not isinstance(self.model, str)
                or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", self.model) is None):
            raise ValueError("invalid model identifier")
        if type(self.dry_run) is not bool or type(self.resume) is not bool:
            raise TypeError("dry_run and resume must be booleans")
        if self.limit is not None and (type(self.limit) is not int or self.limit < 1):
            raise ValueError("limit must be a positive integer")
        if self.run_mode not in {"smoke", "pilot", "full"}:
            raise ValueError("run_mode must be smoke, pilot, or full")
        if self.run_mode == "smoke" and self.limit != 1:
            raise ValueError("smoke mode requires --limit 1")
        if self.run_mode == "full" and self.limit is not None:
            raise ValueError("full mode cannot use a candidate limit")
        if self.run_mode == "pilot" and (
            type(self.limit) is not int or not 8 <= self.limit <= 10
        ):
            raise ValueError("pilot mode requires an 8--10 candidate limit")
        if (type(self.pilot_size) is not int or not 8 <= self.pilot_size <= 10):
            raise ValueError("pilot_size must be 8--10")
        if self.run_mode == "pilot" and self.pilot_size > self.limit:
            raise ValueError("pilot_size cannot exceed the staged candidate limit")
        if self.run_mode == "pilot" and self.resume:
            raise ValueError("pilot mode is an atomic non-resumable run")
        if self.promotion_file is not None and not isinstance(
                self.promotion_file, Path):
            raise TypeError("promotion_file must be a pathlib.Path")
        object.__setattr__(self, "zips", tuple(self.zips))


def _iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> datetime:
    if not isinstance(value, str):
        raise AuthorizationGateError("authorization timestamp is missing")
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise AuthorizationGateError("authorization timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise AuthorizationGateError("authorization timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_authorization_contract() -> tuple[dict[str, Any], str]:
    """Return the exact code/data-use contract a permission must approve."""
    from skills.google_maps import GooglePlacesNewProvider

    query_ids = [
        query_id
        for definitions in GooglePlacesNewProvider.query_definitions.values()
        for query_id, _ in definitions
    ]
    payload = {
        "workflow_schema": WORKFLOW_SCHEMA,
        "services": list(AUTHORIZED_SERVICE_NAMES),
        "permissions": list(REQUIRED_PERMISSION_NAMES),
        "places_search_field_mask": GooglePlacesNewProvider.search_field_mask,
        "places_details_field_mask": GooglePlacesNewProvider.field_mask,
        "query_definition_ids": query_ids,
        "candidate_schema_sha256": _sha256_file(
            PROJECT_ROOT / "docs/v2_candidate_schema.json"),
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "classifier_contract_sha256": (
            ClassifierContractV2.from_project_constants().contract_hash),
        "third_party_processor": "OpenAI API",
        "retention_rule": RETENTION_RULE,
        "retention_expires_at": RETENTION_EXPIRES_AT,
        "derived_outputs": list(DERIVED_OUTPUTS),
        "publication_scope": PUBLICATION_SCOPE,
        "openai_store_false": True,
    }
    digest = _sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))
    return payload, digest


def _read_promotion(path: Path, approved_sha256: str, now: datetime,
                    config: V2WorkflowConfig, *, authorization_hash: str,
                    permission_hash: str, workflow_contract_hash: str,
                    implementation_hashes: dict[str, str]) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuthorizationGateError("smoke promotion file is unavailable") from exc
    if not raw or len(raw) > 100_000:
        raise AuthorizationGateError("smoke promotion file has invalid size")
    observed_hash = _sha256_bytes(raw)
    if observed_hash != approved_sha256:
        raise AuthorizationGateError("smoke promotion is not code-approved")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationGateError("smoke promotion file is invalid") from exc
    expected_fields = {
        "schema_version", "promotion_id", "reviewed_by", "reviewed_at",
        "smoke_manifest_file", "smoke_manifest_sha256", "smoke_run_id",
        "authorization_file_sha256", "written_permission_sha256",
        "workflow_contract_sha256", "deterministic_validation_passed",
        "privacy_check_passed", "pilot_manifest_file",
        "pilot_manifest_sha256", "pilot_semantic_manifest_sha256",
        "pilot_run_id",
    }
    if (type(value) is not dict or set(value) != expected_fields
            or value.get("schema_version") != PROMOTION_SCHEMA
            or value.get("deterministic_validation_passed") is not True
            or value.get("privacy_check_passed") is not True):
        raise AuthorizationGateError("smoke promotion review is incomplete")
    for name in ("promotion_id", "reviewed_by"):
        if (not isinstance(value.get(name), str)
                or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value[name]) is None):
            raise AuthorizationGateError("smoke promotion identity is invalid")
    reviewed_at = _parse_iso(value.get("reviewed_at"))
    if reviewed_at > now.astimezone(timezone.utc):
        raise AuthorizationGateError("smoke promotion review is future-dated")
    relative_name = value.get("smoke_manifest_file")
    if not isinstance(relative_name, str) or not relative_name:
        raise AuthorizationGateError("smoke manifest path is invalid")
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuthorizationGateError("smoke manifest must be local")
    parent = path.resolve().parent
    smoke_manifest_path = (parent / relative).resolve()
    if (not smoke_manifest_path.is_relative_to(parent)
            or smoke_manifest_path == path.resolve()
            or not smoke_manifest_path.is_file()):
        raise AuthorizationGateError("smoke manifest is unavailable")
    expected_smoke_hash = value.get("smoke_manifest_sha256")
    if (not isinstance(expected_smoke_hash, str)
            or SHA256_RE.fullmatch(expected_smoke_hash) is None
            or _sha256_file(smoke_manifest_path) != expected_smoke_hash):
        raise AuthorizationGateError("smoke manifest hash does not match")
    try:
        smoke = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationGateError("smoke manifest is invalid") from exc
    if (type(smoke) is not dict
            or smoke.get("schema_version") != WORKFLOW_SCHEMA
            or smoke.get("status") != "smoke_complete"
            or smoke.get("run_mode") != "smoke"
            or smoke.get("limit") != 1
            or smoke.get("run_id") != value.get("smoke_run_id")
            or smoke.get("task") != config.task
            or smoke.get("city") != config.city
            or smoke.get("zips") != list(config.zips)
            or smoke.get("model") != config.model
            or smoke.get("authorization_file_sha256") != authorization_hash
            or smoke.get("written_permission_sha256") != permission_hash
            or smoke.get("workflow_contract_sha256") != workflow_contract_hash
            or smoke.get("implementation_files") != implementation_hashes
            or value.get("authorization_file_sha256") != authorization_hash
            or value.get("written_permission_sha256") != permission_hash
            or value.get("workflow_contract_sha256") != workflow_contract_hash):
        raise AuthorizationGateError("reviewed smoke does not match the full run")
    finished = _parse_iso(smoke.get("finished_at"))
    if reviewed_at < finished:
        raise AuthorizationGateError("smoke review predates smoke completion")
    try:
        _verify_manifest_files(smoke_manifest_path.parent.resolve(), smoke)
    except ValueError as exc:
        raise AuthorizationGateError("reviewed smoke files failed authentication") from exc
    pilot_relative_name = value.get("pilot_manifest_file")
    if not isinstance(pilot_relative_name, str) or not pilot_relative_name:
        raise AuthorizationGateError("pilot manifest path is invalid")
    pilot_relative = Path(pilot_relative_name)
    if pilot_relative.is_absolute() or ".." in pilot_relative.parts:
        raise AuthorizationGateError("pilot manifest must be local")
    pilot_manifest_path = (parent / pilot_relative).resolve()
    if (
        not pilot_manifest_path.is_relative_to(parent)
        or pilot_manifest_path == path.resolve()
        or pilot_manifest_path.name != "run_manifest.json"
        or not pilot_manifest_path.is_file()
    ):
        raise AuthorizationGateError("pilot manifest is unavailable")
    pilot_manifest_hash = value.get("pilot_manifest_sha256")
    pilot_semantic_hash = value.get("pilot_semantic_manifest_sha256")
    pilot_run_id = value.get("pilot_run_id")
    if (
        not isinstance(pilot_manifest_hash, str)
        or SHA256_RE.fullmatch(pilot_manifest_hash) is None
        or not isinstance(pilot_semantic_hash, str)
        or SHA256_RE.fullmatch(pilot_semantic_hash) is None
        or not isinstance(pilot_run_id, str)
        or RUN_ID_RE.fullmatch(pilot_run_id) is None
    ):
        raise AuthorizationGateError("pilot promotion hashes are invalid")
    try:
        _, _, pilot = authenticate_pilot_ablation_package(
            pilot_manifest_path.parent,
            expected_authorization_file_sha256=authorization_hash,
            expected_written_permission_sha256=permission_hash,
            expected_model=config.model,
            expected_run_id=pilot_run_id,
            expected_manifest_sha256=pilot_manifest_hash,
            expected_semantic_sha256=pilot_semantic_hash,
        )
        pilot_executed = _parse_iso(pilot["run_metadata"]["executed_at"])
    except (KeyError, TypeError, ValueError, AuthorizationGateError) as exc:
        raise AuthorizationGateError(
            "reviewed pilot package failed authentication"
        ) from exc
    if reviewed_at < pilot_executed:
        raise AuthorizationGateError("pilot review predates pilot completion")
    return {
        "promotion_id": value["promotion_id"],
        "promotion_file_sha256": observed_hash,
        "smoke_manifest_sha256": expected_smoke_hash,
        "smoke_run_id": value["smoke_run_id"],
        "reviewed_at": _iso(reviewed_at),
        "pilot_manifest_sha256": pilot_manifest_hash,
        "pilot_semantic_manifest_sha256": pilot_semantic_hash,
        "pilot_run_id": pilot_run_id,
    }


def _read_authorization(path: Path, now: datetime,
                        approved_permission_sha256: str | None) -> tuple[
        dict[str, Any], str, str]:
    if (not isinstance(approved_permission_sha256, str)
            or SHA256_RE.fullmatch(approved_permission_sha256) is None):
        raise AuthorizationGateError(
            "no code-pinned written permission document is approved"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AuthorizationGateError("written authorization file is unavailable") from exc
    if not raw or len(raw) > 100_000:
        raise AuthorizationGateError("written authorization file has invalid size")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationGateError("written authorization file is invalid") from exc
    if (type(value) is not dict or set(value) != AUTHORIZATION_FIELDS
            or value.get("schema_version") != AUTH_SCHEMA):
        raise AuthorizationGateError("written authorization schema is invalid")
    if value.get("project") != "NETS-AI Fall 2026 honors thesis":
        raise AuthorizationGateError("written authorization names another project")
    if (not isinstance(value.get("authorization_id"), str)
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}",
                            value["authorization_id"]) is None):
        raise AuthorizationGateError("written authorization ID is missing")
    document_hash = value.get("written_permission_sha256")
    if not isinstance(document_hash, str) or SHA256_RE.fullmatch(document_hash) is None:
        raise AuthorizationGateError("written permission hash is invalid")
    permission_name = value.get("written_permission_file")
    if (not isinstance(permission_name, str) or not permission_name
            or "\x00" in permission_name):
        raise AuthorizationGateError("written permission file is invalid")
    permission_relative = Path(permission_name)
    if permission_relative.is_absolute() or ".." in permission_relative.parts:
        raise AuthorizationGateError("written permission file must be local")
    authorization_parent = path.resolve().parent
    permission_path = (authorization_parent / permission_relative).resolve()
    if (not permission_path.is_relative_to(authorization_parent)
            or permission_path == path.resolve()):
        raise AuthorizationGateError("written permission file escaped its directory")
    try:
        permission_size = permission_path.stat().st_size
    except OSError as exc:
        raise AuthorizationGateError("written permission document is unavailable") from exc
    if not 0 < permission_size <= 10_000_000:
        raise AuthorizationGateError("written permission document has invalid size")
    try:
        observed_document_hash = _sha256_file(permission_path)
    except OSError as exc:
        raise AuthorizationGateError("written permission document is unreadable") from exc
    if observed_document_hash != document_hash:
        raise AuthorizationGateError("written permission document hash does not match")
    if observed_document_hash != approved_permission_sha256:
        raise AuthorizationGateError("written permission document is not code-approved")
    effective = _parse_iso(value.get("effective_at"))
    expires = _parse_iso(value.get("expires_at"))
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise AuthorizationGateError("authorization clock lacks timezone")
    now_utc = now.astimezone(timezone.utc)
    if not effective <= now_utc < expires:
        raise AuthorizationGateError("written authorization is not currently effective")
    permissions = value.get("permissions")
    if (type(permissions) is not dict or set(permissions) != REQUIRED_PERMISSIONS
            or any(permissions[name] is not True for name in REQUIRED_PERMISSIONS)):
        raise AuthorizationGateError("written authorization does not cover the full workflow")
    services = value.get("services")
    if (not isinstance(services, list)
            or not all(isinstance(item, str) for item in services)
            or len(services) != len(set(services))
            or tuple(services) != AUTHORIZED_SERVICE_NAMES):
        raise AuthorizationGateError("written authorization names invalid services")
    contract, contract_hash = _runtime_authorization_contract()
    bound_fields = {
        "services": value.get("services"),
        "permissions": list(REQUIRED_PERMISSION_NAMES),
        "places_search_field_mask": value.get("places_search_field_mask"),
        "places_details_field_mask": value.get("places_details_field_mask"),
        "query_definition_ids": value.get("query_definition_ids"),
        "candidate_schema_sha256": value.get("candidate_schema_sha256"),
        "prompt_template_sha256": value.get("prompt_template_sha256"),
        "classifier_contract_sha256": value.get("classifier_contract_sha256"),
        "workflow_schema": WORKFLOW_SCHEMA,
        "third_party_processor": value.get("third_party_processor"),
        "retention_rule": value.get("retention_rule"),
        "retention_expires_at": value.get("retention_expires_at"),
        "derived_outputs": value.get("derived_outputs"),
        "publication_scope": value.get("publication_scope"),
        "openai_store_false": value.get("openai_store_false"),
    }
    if (bound_fields != contract
            or value.get("workflow_contract_sha256") != contract_hash):
        raise AuthorizationGateError("written authorization scope is incomplete")
    retention_expires = _parse_iso(value.get("retention_expires_at"))
    if retention_expires < expires:
        raise AuthorizationGateError("authorized retention ends before authorization")
    return value, _sha256_bytes(raw), observed_document_hash


def _read_thesis_stage_authorization(
    path: Path, now: datetime,
) -> tuple[dict[str, Any], str, str]:
    """Bind a live thesis run to the user's repository instruction, not provider permission."""
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise AuthorizationGateError("authorization clock lacks timezone")
    resolved = path.resolve()
    expected = (PROJECT_ROOT / "AGENTS.md").resolve()
    if resolved != expected:
        raise AuthorizationGateError(
            "thesis-stage execution must be bound to the repository AGENTS.md"
        )
    try:
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise AuthorizationGateError(
            "thesis-stage execution instruction is unavailable"
        ) from exc
    required_markers = (
        "the user explicitly authorized thesis-stage live execution of the v2 workflow",
        "The repository must not require APPROVED_PERMISSION_SHA256",
        "Do not fabricate provider permission",
    )
    if any(marker not in text for marker in required_markers):
        raise AuthorizationGateError(
            "AGENTS.md does not authorize the declared thesis-stage live run"
        )
    _, contract_hash = _runtime_authorization_contract()
    instruction_hash = _sha256_bytes(raw)
    return ({
        "authorization_id": "user-thesis-stage-live-2026-08-30",
        "workflow_contract_sha256": contract_hash,
        "authorization_mode": "user_explicit_thesis_stage",
        "provider_permission_claimed": False,
    }, instruction_hash, instruction_hash)


def _configuration_sha256(config: V2WorkflowConfig) -> str:
    payload = {
        "schema_version": WORKFLOW_SCHEMA,
        "run_id": config.run_id,
        "task": config.task,
        "city": config.city,
        "zips": list(config.zips),
        "model": config.model,
        "limit": config.limit,
        "run_mode": config.run_mode,
        "pilot_size": config.pilot_size,
    }
    return _sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))


def _implementation_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise ValueError(f"required implementation file is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _environment_record() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
    }


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".",
                                                  suffix=".tmp",
                                                  dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2,
                                   ensure_ascii=False) + "\n").encode("utf-8"))


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False) + "\n" for row in rows)
    _atomic_bytes(path, raw.encode("utf-8"))


def _decision_payload(decision: LLMDecisionV2) -> dict[str, Any]:
    return {
        "calculated_naics": decision.calculated_naics,
        "target_construct_status": decision.target_construct_status,
        "business_status": decision.business_status,
        "candidate_plausibility": decision.candidate_plausibility,
        "identity_consistency": decision.identity_consistency,
        "historical_year_value": decision.historical_year_value,
        "historical_year_type": decision.historical_year_type,
        "claim_confidence": dict(decision.claim_confidence),
        "claim_evidence_ids": {key: list(value)
                               for key, value in decision.claim_evidence_ids.items()},
        "reasoning_hashes": dict(decision.reasoning_hashes),
    }


def _safe_provider_metadata(metadata: dict[str, Any], decision: LLMDecisionV2,
                            expected_model: str) -> dict[str, Any]:
    if metadata.get("status") != "success":
        raise ValueError("classifier metadata does not report success")
    if metadata.get("requested_model") != expected_model:
        raise ValueError("classifier metadata changed the requested model")
    returned_model = metadata.get("returned_model")
    if (returned_model is not None
            and (not isinstance(returned_model, str)
                 or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}",
                                 returned_model) is None)):
        raise ValueError("classifier returned an unsafe model identifier")
    requested_at = metadata.get("requested_at")
    if not isinstance(requested_at, str):
        raise ValueError("classifier metadata lacks request time")
    raw_time = requested_at[:-1] + "+00:00" if requested_at.endswith("Z") else requested_at
    try:
        parsed_time = datetime.fromisoformat(raw_time)
    except ValueError as exc:
        raise ValueError("classifier metadata has invalid request time") from exc
    if parsed_time.tzinfo is None:
        raise ValueError("classifier request time lacks timezone")
    if (metadata.get("prompt_template_version") != PROMPT_VERSION
            or metadata.get("prompt_template_sha256")
            != PROMPT_TEMPLATE_SHA256):
        raise ValueError("classifier prompt provenance does not match this run")
    if metadata.get("schema_version") != ClassifierContractV2.from_project_constants().schema_version:
        raise ValueError("classifier schema provenance does not match this run")
    fingerprint = metadata.get("system_fingerprint")
    if (fingerprint is not None
            and (not isinstance(fingerprint, str)
                 or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", fingerprint) is None)):
        raise ValueError("classifier fingerprint is unsafe")
    usage = metadata.get("token_usage")
    token_keys = {"input_tokens", "output_tokens", "total_tokens"}
    if (type(usage) is not dict or set(usage) != token_keys
            or any(value is not None
                   and (type(value) is not int or value < 0)
                   for value in usage.values())):
        raise ValueError("classifier token usage is invalid")
    if all(type(usage[key]) is int for key in token_keys):
        if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
            raise ValueError("classifier token total is inconsistent")
    payload = _decision_payload(decision)
    expected_hash = _sha256_bytes(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))
    if metadata.get("decision_sha256") != expected_hash:
        raise ValueError("classifier decision hash does not match")
    target_match = is_target_match(decision)
    if metadata.get("is_target_match") is not target_match:
        raise ValueError("classifier target flag does not match conventional code")
    return {
        "status": "success",
        "requested_model": expected_model,
        "returned_model": returned_model,
        "requested_at": _iso(parsed_time),
        "schema_version": ClassifierContractV2.from_project_constants().schema_version,
        "prompt_template_version": PROMPT_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "system_fingerprint": fingerprint,
        "token_usage": dict(usage),
        "decision_sha256": expected_hash,
        "is_target_match": target_match,
    }


def _decision_record(place_id: str, candidate_id: str,
                     candidate_snapshot_sha256: str, decision: LLMDecisionV2,
                     metadata: dict[str, Any], expected_model: str) -> dict[str, Any]:
    if (not isinstance(candidate_snapshot_sha256, str)
            or SHA256_RE.fullmatch(candidate_snapshot_sha256) is None):
        raise ValueError("classification lacks an authenticated candidate snapshot")
    allowed_metadata = _safe_provider_metadata(metadata, decision, expected_model)
    return {
        "place_id": place_id,
        "candidate_id": candidate_id,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        **_decision_payload(decision),
        "provider_metadata": allowed_metadata,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                value = json.loads(line)
                if type(value) is not dict:
                    raise ValueError
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("existing v2 checkpoint is invalid") from exc
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing v2 JSON checkpoint is invalid") from exc
    if type(value) is not dict:
        raise ValueError("existing v2 JSON checkpoint is invalid")
    return value


def _validate_search_batch(value: Any) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if (type(value) is not dict
            or set(value) != {"status", "candidates", "occurrences", "queries"}
            or value.get("status") not in {"complete", "capped", "limit_reached"}
            or not isinstance(value.get("candidates"), list)
            or not isinstance(value.get("occurrences"), list)
            or not isinstance(value.get("queries"), list)
            or not value["queries"]):
        raise ValueError("candidate provider returned invalid search provenance")
    terminal_states: list[str] = []
    query_keys: set[tuple[str, str]] = set()
    page_counts: dict[tuple[str, str, int], int] = {}
    for query in value["queries"]:
        if (type(query) is not dict
                or set(query) != {"query_definition_id", "source_zip",
                                  "terminal_state", "unique_candidates_added",
                                  "pages"}):
            raise ValueError("candidate provider returned invalid query provenance")
        query_id = query.get("query_definition_id")
        source_zip = query.get("source_zip")
        QueryProvenanceV2(query_id, source_zip, "places_search")
        key = (query_id, source_zip)
        if key in query_keys:
            raise ValueError("candidate provider duplicated query provenance")
        query_keys.add(key)
        terminal = query.get("terminal_state")
        if terminal not in {"exhausted", "page_cap", "limit_reached"}:
            raise ValueError("candidate provider returned invalid terminal state")
        terminal_states.append(terminal)
        if (type(query.get("unique_candidates_added")) is not int
                or query["unique_candidates_added"] < 0
                or not isinstance(query.get("pages"), list)
                or not query["pages"]):
            raise ValueError("candidate provider returned invalid query counts")
        for expected_number, page in enumerate(query["pages"], 1):
            if (type(page) is not dict
                    or set(page) != {"page_number", "retrieved_at",
                                     "response_sha256",
                                     "returned_place_id_count",
                                     "next_page_token_present"}
                    or page.get("page_number") != expected_number
                    or not isinstance(page.get("retrieved_at"), str)
                    or not isinstance(page.get("response_sha256"), str)
                    or SHA256_RE.fullmatch(page["response_sha256"]) is None
                    or type(page.get("returned_place_id_count")) is not int
                    or page["returned_place_id_count"] < 0
                    or type(page.get("next_page_token_present")) is not bool):
                raise ValueError("candidate provider returned invalid page provenance")
            raw_time = (page["retrieved_at"][:-1] + "+00:00"
                        if page["retrieved_at"].endswith("Z")
                        else page["retrieved_at"])
            try:
                parsed = datetime.fromisoformat(raw_time)
            except ValueError as exc:
                raise ValueError("candidate provider returned invalid page time") from exc
            if parsed.tzinfo is None:
                raise ValueError("candidate provider returned naive page time")
            page_counts[(query_id, source_zip, expected_number)] = page[
                "returned_place_id_count"
            ]

    occurrence_counts: dict[tuple[str, str, int], int] = {}
    occurrence_ids: set[str] = set()
    for occurrence in value["occurrences"]:
        if (type(occurrence) is not dict
                or set(occurrence) != {
                    "place_id", "query_definition_id", "source_zip", "page_number",
                }):
            raise ValueError("candidate provider returned invalid occurrence fields")
        place_id = occurrence.get("place_id")
        query_id = occurrence.get("query_definition_id")
        source_zip = occurrence.get("source_zip")
        page_number = occurrence.get("page_number")
        key = (query_id, source_zip, page_number)
        if (not isinstance(place_id, str)
                or PLACE_ID_RE.fullmatch(place_id) is None
                or not isinstance(query_id, str)
                or not isinstance(source_zip, str)
                or type(page_number) is not int
                or key not in page_counts):
            raise ValueError("candidate provider returned invalid occurrence provenance")
        QueryProvenanceV2(query_id, source_zip, "places_search")
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1
        occurrence_ids.add(place_id)
    if any(occurrence_counts.get(key, 0) != count
           for key, count in page_counts.items()):
        raise ValueError("candidate occurrences do not reconcile with search pages")

    candidate_ids: set[str] = set()
    for candidate in value["candidates"]:
        if (type(candidate) is not dict
                or set(candidate) != {
                    "place_id", "source_zip", "query_definition_id",
                }):
            raise ValueError("candidate provider returned invalid candidate fields")
        place_id = candidate.get("place_id")
        source_zip = candidate.get("source_zip")
        query_id = candidate.get("query_definition_id")
        if (not isinstance(place_id, str)
                or PLACE_ID_RE.fullmatch(place_id) is None
                or not isinstance(source_zip, str)
                or not isinstance(query_id, str)
                or (query_id, source_zip) not in query_keys):
            raise ValueError("candidate provider returned invalid candidate provenance")
        candidate_ids.add(place_id)
    if not candidate_ids.issubset(occurrence_ids):
        raise ValueError("candidate and occurrence Place IDs disagree")

    status = value["status"]
    if ((status == "complete" and any(item != "exhausted" for item in terminal_states))
            or (status == "capped" and "page_cap" not in terminal_states)
            or (status == "limit_reached" and terminal_states[-1] != "limit_reached")):
        raise ValueError("candidate search status contradicts its query ledger")
    return list(value["candidates"]), list(value["occurrences"]), {
        "status": status, "queries": list(value["queries"]),
    }


def _verify_manifest_files(run_root: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if type(files) is not dict:
        raise ValueError("resume manifest lacks authenticated files")
    required = {
        "place_ids.jsonl", "candidate_occurrences.jsonl",
        "search_provenance.json", "failures.jsonl",
    }
    completed_count = manifest.get("completed_count")
    if type(completed_count) is int and completed_count > 0:
        required.update({"classification_v2.jsonl",
                         "private/archive_evidence.jsonl"})
    if manifest.get("status") in {
            "smoke_complete", "smoke_incomplete", "full_complete",
            "full_incomplete", "search_incomplete", "empty_search",
    } and not required <= set(files):
        raise ValueError("resume manifest omits required checkpoint files")
    for relative, expected_hash in files.items():
        if (not isinstance(relative, str) or not relative
                or not isinstance(expected_hash, str)
                or SHA256_RE.fullmatch(expected_hash) is None):
            raise ValueError("resume manifest has an invalid file record")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("resume manifest file escaped the run root")
        path = (run_root / relative_path).resolve()
        if not path.is_relative_to(run_root) or not path.is_file():
            raise ValueError("resume checkpoint file is unavailable")
        if _sha256_file(path) != expected_hash:
            raise ValueError("resume checkpoint hash does not match")


def _checkpoint_files(run_root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        path.relative_to(run_root).as_posix(): _sha256_file(path)
        for path in paths if path.is_file()
    }


def _validate_place_rows(rows: list[dict[str, Any]], config: V2WorkflowConfig) -> None:
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"place_id", "source_zip", "query_definition_id"}:
            raise ValueError("existing Place-ID checkpoint has invalid fields")
        place_id = row.get("place_id")
        source_zip = row.get("source_zip")
        query_id = row.get("query_definition_id")
        if (not isinstance(place_id, str)
                or PLACE_ID_RE.fullmatch(place_id) is None
                or place_id in seen
                or not isinstance(source_zip, str)
                or source_zip not in config.zips):
            raise ValueError("existing Place-ID checkpoint is invalid")
        QueryProvenanceV2(query_id, source_zip, "places_search")
        seen.add(place_id)
    if config.limit is not None and len(rows) > config.limit:
        raise ValueError("existing Place-ID checkpoint exceeds the run limit")


def _validate_completed_rows(classifications: list[dict[str, Any]],
                             archives: list[dict[str, Any]],
                             candidate_ids: set[str]) -> set[str]:
    expected_classification = {
        "place_id", "candidate_id", "candidate_snapshot_sha256",
        "calculated_naics", "target_construct_status", "business_status",
        "candidate_plausibility", "identity_consistency",
        "historical_year_value", "historical_year_type", "claim_confidence",
        "claim_evidence_ids", "reasoning_hashes", "provider_metadata",
    }
    completed: set[str] = set()
    for row in classifications:
        place_id = row.get("place_id")
        if (set(row) != expected_classification
                or not isinstance(place_id, str)
                or PLACE_ID_RE.fullmatch(place_id) is None
                or place_id not in candidate_ids or place_id in completed
                or not isinstance(row.get("candidate_id"), str)
                or re.fullmatch(r"candidate_[0-9a-f]{64}",
                                row["candidate_id"]) is None
                or not isinstance(row.get("candidate_snapshot_sha256"), str)
                or SHA256_RE.fullmatch(row["candidate_snapshot_sha256"]) is None):
            raise ValueError("existing classification checkpoint is invalid")
        completed.add(place_id)
    archive_ids: set[str] = set()
    for row in archives:
        place_id = row.get("place_id")
        if (set(row) != {"place_id", "archive_evidence"}
                or not isinstance(place_id, str) or place_id in archive_ids
                or place_id not in candidate_ids
                or type(row.get("archive_evidence")) is not dict):
            raise ValueError("existing archive checkpoint is invalid")
        archive_ids.add(place_id)
    if archive_ids != completed:
        raise ValueError("classification and archive checkpoints disagree")
    return completed


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8"))


def _location_record(location: CandidateLocationV2) -> dict[str, Any]:
    return {
        "place_id": location.place_id,
        "candidate_id": location.candidate_id,
        "name": location.name,
        "formatted_address": location.formatted_address,
        "returned_postal_code": location.returned_postal_code,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "retrieved_at": _iso(location.retrieved_at),
        "query_definition_id": location.query_provenance.query_definition_id,
        "source_zip": location.query_provenance.source_zip,
        "retrieval_channel": location.query_provenance.retrieval_channel,
    }


def _location_from_record(value: Any) -> CandidateLocationV2:
    expected = {
        "place_id", "candidate_id", "name", "formatted_address",
        "returned_postal_code", "latitude", "longitude", "retrieved_at",
        "query_definition_id", "source_zip", "retrieval_channel",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("completed candidate location checkpoint is invalid")
    try:
        retrieved_at = datetime.fromisoformat(
            value["retrieved_at"].replace("Z", "+00:00")
        )
        provenance = QueryProvenanceV2(
            value["query_definition_id"], value["source_zip"],
            value["retrieval_channel"],
        )
        return CandidateLocationV2(
            value["place_id"], value["candidate_id"], value["name"],
            value["formatted_address"], value["returned_postal_code"],
            value["latitude"], value["longitude"], retrieved_at, provenance,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "completed candidate location checkpoint is invalid"
        ) from exc


def _completed_unit_path(root: Path, place_id: str) -> Path:
    return root / f"{_sha256_bytes(place_id.encode('utf-8'))}.json"


def _completed_unit(
    location: CandidateLocationV2,
    classification: dict[str, Any],
    archive: dict[str, Any],
    *,
    model: str,
    implementation_binding_sha256: str,
    workflow_contract_sha256: str,
    search_frame_sha256: str,
) -> dict[str, Any]:
    decision = _checkpoint_decision(classification, model)
    _validate_archive_checkpoint(archive)
    if (
        classification["place_id"] != location.place_id
        or classification["candidate_id"] != location.candidate_id
        or classification["candidate_snapshot_sha256"]
        != location.candidate_snapshot_sha256
        or archive.get("place_id") != location.place_id
    ):
        raise ValueError("completed candidate checkpoint lineage disagrees")
    core = {
        "schema_version": COMPLETED_UNIT_SCHEMA,
        "place_id": location.place_id,
        "model": model,
        "prompt_template_version": PROMPT_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "implementation_binding_sha256": implementation_binding_sha256,
        "workflow_contract_sha256": workflow_contract_sha256,
        "search_frame_sha256": search_frame_sha256,
        "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
        "classification_decision_sha256":
        classification_decision_sha256(decision),
        "location": _location_record(location),
        "classification_record": classification,
        "archive_record": archive,
        "classification_record_sha256": _canonical_sha256(classification),
        "archive_record_sha256": _canonical_sha256(archive),
    }
    return {**core, "completed_unit_sha256": _canonical_sha256(core)}


def _load_completed_units(
    root: Path,
    candidates: list[dict[str, Any]],
    *,
    model: str,
    implementation_binding_sha256: str,
    workflow_contract_sha256: str,
    search_frame_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
           list[DeliveryCandidateV2]]:
    if not root.exists():
        return [], [], []
    if not root.is_dir():
        raise ValueError("completed candidate checkpoint root is invalid")
    expected_ids = {row["place_id"] for row in candidates}
    classifications: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    delivery: list[DeliveryCandidateV2] = []
    seen: set[str] = set()
    expected_fields = {
        "schema_version", "place_id", "model", "prompt_template_version",
        "prompt_template_sha256", "implementation_binding_sha256",
        "workflow_contract_sha256", "search_frame_sha256",
        "candidate_snapshot_sha256", "classification_decision_sha256",
        "location", "classification_record", "archive_record",
        "classification_record_sha256", "archive_record_sha256",
        "completed_unit_sha256",
    }
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            raise ValueError("completed candidate checkpoint contains an invalid file")
        unit = _load_json_object(path)
        place_id = unit.get("place_id")
        if (
            set(unit) != expected_fields
            or unit.get("schema_version") != COMPLETED_UNIT_SCHEMA
            or not isinstance(place_id, str)
            or place_id not in expected_ids
            or place_id in seen
            or path != _completed_unit_path(root, place_id)
            or unit.get("model") != model
            or unit.get("prompt_template_version") != PROMPT_VERSION
            or unit.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256
            or unit.get("implementation_binding_sha256")
            != implementation_binding_sha256
            or unit.get("workflow_contract_sha256") != workflow_contract_sha256
            or unit.get("search_frame_sha256") != search_frame_sha256
        ):
            raise ValueError("completed candidate checkpoint binding is invalid")
        core = {key: value for key, value in unit.items()
                if key != "completed_unit_sha256"}
        classification = unit["classification_record"]
        archive = unit["archive_record"]
        location = _location_from_record(unit["location"])
        decision = _checkpoint_decision(classification, model)
        _validate_archive_checkpoint(archive)
        if (
            unit["completed_unit_sha256"] != _canonical_sha256(core)
            or unit["classification_record_sha256"]
            != _canonical_sha256(classification)
            or unit["archive_record_sha256"] != _canonical_sha256(archive)
            or unit["candidate_snapshot_sha256"]
            != location.candidate_snapshot_sha256
            or unit["classification_decision_sha256"]
            != classification_decision_sha256(decision)
            or classification.get("place_id") != place_id
            or classification.get("candidate_id") != location.candidate_id
            or classification.get("candidate_snapshot_sha256")
            != location.candidate_snapshot_sha256
            or archive.get("place_id") != place_id
        ):
            raise ValueError("completed candidate checkpoint hash is invalid")
        classifications.append(classification)
        archives.append(archive)
        delivery.append(DeliveryCandidateV2(location, decision))
        seen.add(place_id)
    return classifications, archives, delivery


class V2Workflow:
    """Coordinate injected providers while enforcing transient/persisted boundaries."""

    def __init__(self, google_provider: Any, wayback_collector: Any,
                 classifier: Callable[[ClassifierInputV2, str],
                                       tuple[LLMDecisionV2, dict[str, Any]]],
                 *, clock: Callable[[], datetime] | None = None,
                 approved_permission_sha256: str | None = None,
                 live_preflight: Callable[[], None] | None = None,
                 approved_promotion_sha256: str | None = None,
                 allow_thesis_stage_live: bool = False,
                 ablation_evaluator: Callable[[list[dict[str, str]], str],
                                               dict[str, Any]] | None = None,
                 delivery_callback: Callable[[tuple[DeliveryCandidateV2, ...],
                                               V2WorkflowConfig, Path],
                                              dict[str, Any]] | None = None):
        self.google_provider = google_provider
        self.wayback_collector = wayback_collector
        self.classifier = classifier
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.approved_permission_sha256 = approved_permission_sha256
        self.live_preflight = live_preflight
        self.approved_promotion_sha256 = approved_promotion_sha256
        if type(allow_thesis_stage_live) is not bool:
            raise TypeError("allow_thesis_stage_live must be a boolean")
        self.allow_thesis_stage_live = allow_thesis_stage_live
        self.ablation_evaluator = ablation_evaluator
        self.delivery_callback = delivery_callback

    def run(self, config: V2WorkflowConfig) -> dict[str, Any]:
        if not isinstance(config, V2WorkflowConfig):
            raise TypeError("invalid workflow configuration")
        if config.dry_run:
            return {
                "schema_version": WORKFLOW_SCHEMA,
                "status": "dry_run", "run_id": config.run_id,
                "task": config.task, "city": config.city,
                "zip_count": len(config.zips), "limit": config.limit,
                "run_mode": config.run_mode,
                "pilot_size": config.pilot_size,
                "external_calls": 0,
                "planned_stages": ["places_search", "places_details",
                                   "wayback_cdx_and_replay", "structured_llm"],
                "planned_endpoints": [
                    "places_text_search_new", "places_details_new",
                    "wayback_cdx", "wayback_replay", "openai_responses",
                ],
                "authorization_required_for_live_mode":
                not self.allow_thesis_stage_live,
                "thesis_stage_compliance_warning": (
                    THESIS_STAGE_COMPLIANCE_WARNING
                    if self.allow_thesis_stage_live else None
                ),
            }

        now = self.clock()
        if self.allow_thesis_stage_live:
            authorization, authorization_hash, permission_hash = (
                _read_thesis_stage_authorization(config.authorization_file, now)
            )
        else:
            authorization, authorization_hash, permission_hash = _read_authorization(
                config.authorization_file, now, self.approved_permission_sha256,
            )
        configuration_hash = _configuration_sha256(config)
        implementation_hashes = _implementation_hashes()
        promotion_metadata: dict[str, str] = {}
        if config.run_mode == "full":
            if (config.promotion_file is None
                    or not isinstance(self.approved_promotion_sha256, str)
                    or SHA256_RE.fullmatch(self.approved_promotion_sha256) is None):
                raise AuthorizationGateError(
                    "full run lacks a code-pinned reviewed smoke promotion"
                )
            promotion_metadata = _read_promotion(
                config.promotion_file, self.approved_promotion_sha256, now,
                config, authorization_hash=authorization_hash,
                permission_hash=permission_hash,
                workflow_contract_hash=
                authorization["workflow_contract_sha256"],
                implementation_hashes=implementation_hashes,
            )
        if not callable(self.live_preflight):
            raise AuthorizationGateError("live dependency preflight is unavailable")
        if config.run_mode == "pilot" and not callable(self.ablation_evaluator):
            raise AuthorizationGateError("pilot ablation evaluator is unavailable")
        if config.run_mode == "full" and not callable(self.delivery_callback):
            raise AuthorizationGateError("full downstream delivery is unavailable")
        output_root = config.output_root.resolve()
        run_root = (output_root / config.run_id).resolve()
        if run_root.parent != output_root:
            raise ValueError("run root escaped its output root")
        manifest_path = run_root / "run_manifest.json"

        if run_root.exists() and not config.resume:
            raise FileExistsError("v2 run already exists")
        if config.resume:
            if not manifest_path.is_file():
                raise ValueError("resume requires an authenticated run manifest")
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("resume manifest is invalid") from exc
            if (type(existing) is not dict
                    or existing.get("schema_version") != WORKFLOW_SCHEMA
                    or existing.get("run_id") != config.run_id
                    or existing.get("authorization_file_sha256")
                    != authorization_hash
                    or existing.get("written_permission_sha256")
                    != permission_hash
                    or existing.get("configuration_sha256")
                    != configuration_hash
                    or existing.get("implementation_files")
                    != implementation_hashes
                    or any(existing.get(key) != expected
                           for key, expected in promotion_metadata.items())):
                raise ValueError("resume manifest does not match this run")
            _verify_manifest_files(run_root, existing)
            if existing.get("status") in {"smoke_complete", "full_complete"}:
                return existing
        try:
            self.live_preflight()
        except (WaybackDeferred, WaybackServiceUnavailable):
            raise
        except Exception:
            raise AuthorizationGateError("live dependency preflight failed") from None
        if not config.resume:
            run_root.mkdir(parents=True, exist_ok=False)
            (run_root / "private").mkdir()

        places_path = run_root / "place_ids.jsonl"
        occurrences_path = run_root / "candidate_occurrences.jsonl"
        search_path = run_root / "search_provenance.json"
        classification_path = run_root / "classification_v2.jsonl"
        failure_path = run_root / "failures.jsonl"
        archive_path = run_root / "private" / "archive_evidence.jsonl"
        failure_history_path = run_root / "private" / "failure_history.jsonl"
        completed_root = run_root / "private" / "completed_candidates"
        completed_root.mkdir(exist_ok=True)

        existing_places = _load_jsonl(places_path) if config.resume else []
        if existing_places:
            _validate_place_rows(existing_places, config)
            candidates = existing_places
            occurrences = _load_jsonl(occurrences_path)
            search_provenance = _load_json_object(search_path)
            _, occurrences, search_provenance = _validate_search_batch({
                **search_provenance, "candidates": candidates,
                "occurrences": occurrences,
            })
        else:
            raw_search = self.google_provider.list_candidates(
                task=config.task, city=config.city, zips=config.zips,
                limit=config.limit,
            )
            raw_candidates, occurrences, search_provenance = (
                _validate_search_batch(raw_search)
            )
            candidates = []
            seen: set[str] = set()
            for item in raw_candidates:
                if type(item) is not dict:
                    raise ValueError("candidate provider returned an invalid row")
                place_id = item.get("place_id")
                source_zip = item.get("source_zip")
                query_id = item.get("query_definition_id")
                if (not isinstance(place_id, str)
                        or PLACE_ID_RE.fullmatch(place_id) is None
                        or not isinstance(source_zip, str)
                        or re.fullmatch(r"\d{5}", source_zip) is None
                        or source_zip not in config.zips
                        or not isinstance(query_id, str)):
                    raise ValueError("candidate provider returned invalid provenance")
                QueryProvenanceV2(query_id, source_zip, "places_search")
                if place_id not in seen:
                    candidates.append({"place_id": place_id,
                                       "source_zip": source_zip,
                                       "query_definition_id": query_id})
                    seen.add(place_id)
                if config.limit is not None and len(candidates) >= config.limit:
                    break
            _atomic_jsonl(places_path, candidates)
            _atomic_jsonl(occurrences_path, occurrences)
            _atomic_json(search_path, search_provenance)

        _validate_place_rows(candidates, config)
        search_frame_sha256 = _canonical_sha256({
            "candidates": candidates,
            "occurrences": occurrences,
            "search_provenance": search_provenance,
        })
        implementation_binding_sha256 = _canonical_sha256(implementation_hashes)
        resume_attempt = (
            existing.get("resume_attempt", 0) + 1
            if config.resume else 0
        )

        running_manifest = {
            "schema_version": WORKFLOW_SCHEMA, "run_id": config.run_id,
            "status": "running", "task": config.task, "city": config.city,
            "zips": list(config.zips), "limit": config.limit,
            "run_mode": config.run_mode,
            "pilot_size": config.pilot_size,
            "occurrence_count": len(occurrences),
            "model": config.model, "authorization_file_sha256": authorization_hash,
            "authorization_id": authorization["authorization_id"],
            "written_permission_sha256": permission_hash,
            "workflow_contract_sha256":
            authorization["workflow_contract_sha256"],
            "configuration_sha256": configuration_hash,
            "implementation_files": implementation_hashes,
            "environment": _environment_record(),
            "provider_policy_urls": list(PROVIDER_POLICY_URLS),
            "authorization_mode": authorization.get(
                "authorization_mode", "authenticated_written_permission"
            ),
            "provider_permission_claimed": authorization.get(
                "provider_permission_claimed", True
            ),
            "compliance_warnings": (
                [THESIS_STAGE_COMPLIANCE_WARNING]
                if self.allow_thesis_stage_live else []
            ),
            "started_or_resumed_at": _iso(now), "files": {},
            "search_frame_sha256": search_frame_sha256,
            "implementation_binding_sha256": implementation_binding_sha256,
            "resume_supported": True,
            "resume_attempt": resume_attempt,
            **promotion_metadata,
        }
        _atomic_json(manifest_path, running_manifest)

        expected_search_status = (
            "limit_reached" if config.run_mode == "smoke" else "complete"
        )
        search_acceptable = search_provenance["status"] == expected_search_status
        if config.run_mode == "pilot":
            search_acceptable = (
                search_provenance["status"] == "limit_reached"
                or (
                    search_provenance["status"] == "complete"
                    and config.pilot_size <= len(candidates) <= 10
                )
            )
        if not candidates or not search_acceptable:
            _atomic_jsonl(failure_path, [])
            files = _checkpoint_files(
                run_root,
                (places_path, occurrences_path, search_path, failure_path),
            )
            terminal_status = "empty_search" if not candidates else "search_incomplete"
            manifest = {
                **running_manifest, "status": terminal_status,
                "finished_at": _iso(self.clock()),
                "candidate_count": len(candidates), "completed_count": 0,
                "failure_count": 0, "files": files,
            }
            _atomic_json(manifest_path, manifest)
            return manifest

        if config.run_mode == "pilot":
            failure_path = run_root / "failures.jsonl"
            archive_path = run_root / "private" / "archive_evidence.jsonl"
            packets: list[ClassifierInputV2] = []
            archives: list[dict[str, Any]] = []
            failures: list[dict[str, Any]] = []
            wayback_pause = None
            for item in candidates:
                place_id = item["place_id"]
                stage = "places_details"
                try:
                    details = self.google_provider.get_place_details(place_id)
                    retrieved = self.clock()
                    provenance = QueryProvenanceV2(
                        item["query_definition_id"], item["source_zip"],
                        "places_search",
                    )
                    # Both transient downstream lineage and classifier facts are
                    # created from this same Details object and retrieval timestamp.
                    location = candidate_location_from_details(
                        details, provenance, retrieved
                    )
                    candidate = build_candidate_facts(details, provenance, retrieved)
                    if (
                        candidate.source_place_id != place_id
                        or location.place_id != place_id
                        or candidate.candidate_id != location.candidate_id
                    ):
                        raise ValueError("Details response changed candidate identity")
                    reviews = build_review_evidence(details, retrieved)
                    stage = "wayback"
                    wayback = self.wayback_collector.collect(candidate)
                    if not isinstance(wayback, WaybackCollectionV2):
                        raise ValueError("Wayback collector returned an invalid packet")
                    packets.append(ClassifierInputV2(
                        candidate, reviews, wayback,
                        ClassifierContractV2.from_project_constants(),
                    ))
                    archives.append({
                        "place_id": place_id,
                        "candidate_snapshot_sha256":
                        location.candidate_snapshot_sha256,
                        "archive_evidence":
                        wayback.archive_evidence.to_private_dict(),
                    })
                except WaybackDeferred as exc:
                    wayback_pause = {"reason": exc.reason, "retry_at": exc.retry_at}
                    failures.append({
                        "place_id": place_id, "status": "wayback_deferred",
                        "retryable": True, "exception_type": type(exc).__name__,
                        **wayback_pause,
                    })
                    break
                except Exception as exc:
                    failures.append({
                        "place_id": place_id,
                        "status": f"{stage}_error",
                        "retryable": True,
                        "exception_type": type(exc).__name__,
                    })
            _atomic_jsonl(archive_path, archives)
            _atomic_jsonl(failure_path, failures)
            if failures or len(packets) != len(candidates):
                files = _checkpoint_files(
                    run_root,
                    (places_path, occurrences_path, search_path, failure_path,
                     archive_path),
                )
                manifest = {
                    **running_manifest,
                    "status": ("pilot_paused_wayback" if wayback_pause
                               else "pilot_incomplete"),
                    "finished_at": _iso(self.clock()),
                    "candidate_count": len(candidates),
                    "completed_count": len(packets),
                    "failure_count": len(failures),
                    "files": files,
                    **({"wayback_retry": wayback_pause} if wayback_pause else {}),
                }
                _atomic_json(manifest_path, manifest)
                return manifest

            selection_size = (
                len(candidates)
                if search_provenance["status"] == "complete"
                else config.pilot_size
            )
            selection = select_representative_candidates(
                packets, target_size=selection_size
            )
            ablation_rows = run_ablation(
                selection, self.ablation_evaluator, model=config.model
            )
            try:
                ablation_summary = summarize_ablation(
                    ablation_rows,
                    privacy_check_passed=True,
                    board_separation_passed=True,
                )
            except TypeError:
                # Compatibility with the immediately preceding v2.1 pilot contract;
                # its replacement is covered by the pilot module's own tests.
                ablation_summary = summarize_ablation(ablation_rows)
            pilot_root = run_root / "pilot_delivery"
            pilot_manifest = run_pilot_v2_delivery(
                PilotDeliveryInputsV2(
                    selection_ledger=selection.selection_ledger,
                    unavailable_roles=selection.unavailable_roles,
                    ablation_rows=ablation_rows,
                    ablation_summary=ablation_summary,
                    run_metadata={
                        "run_id": config.run_id,
                        "model": config.model,
                        "executed_at": _iso(self.clock()),
                        "authorization_file_sha256": authorization_hash,
                        "written_permission_sha256": permission_hash,
                    },
                    empirical_v2_results_exist=True,
                ),
                pilot_root,
            )
            pilot_files = tuple(
                path for path in pilot_root.rglob("*") if path.is_file()
            )
            files = _checkpoint_files(
                run_root,
                (places_path, occurrences_path, search_path, failure_path,
                 archive_path, *pilot_files),
            )
            manifest = {
                **running_manifest,
                "status": pilot_manifest["status"],
                "finished_at": _iso(self.clock()),
                "candidate_count": len(selection.packets),
                "staged_candidate_count": len(candidates),
                "completed_count": len(packets),
                "failure_count": pilot_manifest["failure_count"],
                "pilot_semantic_manifest_sha256":
                pilot_manifest["semantic_manifest_sha256"],
                "files": files,
            }
            _atomic_json(manifest_path, manifest)
            return manifest

        classifications, archives, delivery_candidates = _load_completed_units(
            completed_root, candidates, model=config.model,
            implementation_binding_sha256=implementation_binding_sha256,
            workflow_contract_sha256=authorization["workflow_contract_sha256"],
            search_frame_sha256=search_frame_sha256,
        )
        completed = _validate_completed_rows(
            classifications, archives,
            {row["place_id"] for row in candidates},
        )
        if config.resume:
            persisted_classifications = _load_jsonl(classification_path)
            persisted_archives = _load_jsonl(archive_path)
            persisted_completed = _validate_completed_rows(
                persisted_classifications, persisted_archives,
                {row["place_id"] for row in candidates},
            )
            if (
                not persisted_completed.issubset(completed)
                or not {
                    _canonical_sha256(row)
                    for row in persisted_classifications
                }.issubset({
                    _canonical_sha256(row) for row in classifications
                })
                or not {
                    _canonical_sha256(row) for row in persisted_archives
                }.issubset({_canonical_sha256(row) for row in archives})
            ):
                raise ValueError(
                    "completed units disagree with aggregate checkpoints"
                )
        prior_failures = _load_jsonl(failure_path) if config.resume else []
        failure_history = (
            _load_jsonl(failure_history_path) if config.resume else []
        )
        prior_attempt = max(0, resume_attempt - 1)
        archived_failure_keys = {
            (row.get("resume_attempt"), _canonical_sha256(row.get("record")))
            for row in failure_history
            if type(row) is dict and type(row.get("record")) is dict
        }
        for row in prior_failures:
            key = (prior_attempt, _canonical_sha256(row))
            if key not in archived_failure_keys:
                failure_history.append({
                    "resume_attempt": prior_attempt,
                    "record": row,
                })
                archived_failure_keys.add(key)
        _atomic_jsonl(failure_history_path, failure_history)
        failures: list[dict[str, Any]] = []
        _atomic_jsonl(failure_path, failures)

        def persist_running_checkpoint(*, status: str | None = None,
                                       wayback_retry=None) -> dict[str, Any]:
            paths = [
                places_path, occurrences_path, search_path,
                classification_path, failure_path, archive_path,
                failure_history_path,
            ]
            paths.extend(sorted(completed_root.glob("*.json")))
            checkpoint = {
                **running_manifest,
                "candidate_count": len(candidates),
                "completed_count": len(completed),
                "failure_count": len(failures),
                "files": _checkpoint_files(run_root, tuple(paths)),
            }
            if status is not None:
                checkpoint["status"] = status
            if wayback_retry is not None:
                checkpoint["wayback_retry"] = wayback_retry
            _atomic_json(manifest_path, checkpoint)
            return checkpoint

        _atomic_jsonl(classification_path, classifications)
        _atomic_jsonl(archive_path, archives)
        persist_running_checkpoint()

        for item in candidates:
            place_id = item["place_id"]
            if place_id in completed:
                continue
            stage = "places_details"
            try:
                details = self.google_provider.get_place_details(place_id)
                retrieved = self.clock()
                provenance = QueryProvenanceV2(
                    item["query_definition_id"], item["source_zip"],
                    "places_search",
                )
                location = candidate_location_from_details(
                    details, provenance, retrieved
                )
                candidate = build_candidate_facts(details, provenance, retrieved)
                if (
                    candidate.source_place_id != place_id
                    or location.place_id != place_id
                    or candidate.candidate_id != location.candidate_id
                ):
                    raise ValueError("Details response changed Place ID")
                review_evidence = build_review_evidence(details, retrieved)
                stage = "wayback"
                wayback = self.wayback_collector.collect(candidate)
                if not isinstance(wayback, WaybackCollectionV2):
                    raise ValueError("Wayback collector returned an invalid packet")
                packet = ClassifierInputV2(
                    candidate, review_evidence, wayback,
                    ClassifierContractV2.from_project_constants(),
                )
                stage = "classifier"
                decision, metadata = self.classifier(packet, config.model)
                if not isinstance(decision, LLMDecisionV2) or type(metadata) is not dict:
                    raise ValueError("classifier returned an invalid decision")
                classification_record = _decision_record(
                    place_id, candidate.candidate_id,
                    location.candidate_snapshot_sha256, decision, metadata,
                    config.model,
                )
                archive_record = {
                    "place_id": place_id,
                    "archive_evidence":
                    wayback.archive_evidence.to_private_dict(),
                }
                unit = _completed_unit(
                    location, classification_record, archive_record,
                    model=config.model,
                    implementation_binding_sha256=
                    implementation_binding_sha256,
                    workflow_contract_sha256=
                    authorization["workflow_contract_sha256"],
                    search_frame_sha256=search_frame_sha256,
                )
                unit_path = _completed_unit_path(completed_root, place_id)
                if unit_path.exists():
                    raise ValueError("completed candidate unit is immutable")
                _atomic_json(unit_path, unit)
                classifications.append(classification_record)
                archives.append(archive_record)
                delivery_candidates.append(DeliveryCandidateV2(location, decision))
                completed.add(place_id)
                _atomic_jsonl(classification_path, classifications)
                _atomic_jsonl(archive_path, archives)
                persist_running_checkpoint()
            except WaybackDeferred as exc:
                retry = {"reason": exc.reason, "retry_at": exc.retry_at}
                failure = {
                    "place_id": place_id, "status": "wayback_deferred",
                    "retryable": True, "exception_type": type(exc).__name__,
                    **retry,
                }
                failures.append(failure)
                failure_history.append({
                    "resume_attempt": resume_attempt, "record": failure,
                })
                _atomic_jsonl(failure_path, failures)
                _atomic_jsonl(failure_history_path, failure_history)
                return persist_running_checkpoint(
                    status=f"{config.run_mode}_paused_wayback", wayback_retry=retry,
                )
            except Exception as exc:  # stage boundary intentionally redacts messages
                failure = {
                    "place_id": place_id, "status": f"{stage}_error",
                    "retryable": True, "exception_type": type(exc).__name__,
                }
                failures.append(failure)
                failure_history.append({
                    "resume_attempt": resume_attempt,
                    "record": failure,
                })
                _atomic_jsonl(failure_path, failures)
                _atomic_jsonl(failure_history_path, failure_history)
                persist_running_checkpoint()

        _atomic_jsonl(failure_path, failures)

        delivery_metadata: dict[str, Any] = {}
        if (
            config.run_mode == "full"
            and len(completed) == len(candidates)
            and not failures
        ):
            delivery_checkpoint_paths = [
                places_path, occurrences_path, search_path,
                classification_path, failure_path, archive_path,
                failure_history_path,
            ]
            delivery_checkpoint_paths.extend(
                sorted(completed_root.glob("*.json"))
            )
            delivery_checkpoint = {
                **running_manifest,
                "candidate_count": len(candidates),
                "completed_count": len(completed),
                "failure_count": 0,
                "files": _checkpoint_files(
                    run_root, tuple(delivery_checkpoint_paths)
                ),
            }
            _atomic_json(manifest_path, delivery_checkpoint)
            try:
                delivered = self.delivery_callback(
                    tuple(delivery_candidates), config, run_root / "delivery"
                )
                if (
                    not isinstance(delivered, dict)
                    or delivered.get("status") != "full_delivery_complete"
                    or not isinstance(
                        delivered.get("semantic_manifest_sha256"), str
                    )
                    or SHA256_RE.fullmatch(
                        delivered["semantic_manifest_sha256"]
                    ) is None
                    or type(delivered.get("failure_count")) is not int
                    or delivered["failure_count"] != 0
                ):
                    raise ValueError(
                        "full delivery callback returned invalid metadata"
                    )
            except Exception as exc:
                failure_paths = [
                    places_path, occurrences_path, search_path,
                    classification_path, failure_path, archive_path,
                ]
                delivery_root = run_root / "delivery"
                if delivery_root.is_dir():
                    failure_paths.extend(
                        path for path in delivery_root.rglob("*")
                        if path.is_file()
                    )
                failed_manifest = {
                    **running_manifest,
                    "status": "full_delivery_failed",
                    "finished_at": _iso(self.clock()),
                    "candidate_count": len(candidates),
                    "completed_count": len(completed),
                    "acquisition_failure_count": len(failures),
                    "failure_count": 1,
                    "failure_stage": "delivery_callback",
                    "failure_type": type(exc).__name__,
                    "resume_supported": True,
                    "files": _checkpoint_files(
                        run_root, tuple(failure_paths)
                    ),
                }
                _atomic_json(manifest_path, failed_manifest)
                raise ValueError("full delivery failed closed") from None
            delivery_metadata = {
                "delivery_status": delivered.get("status"),
                "delivery_failure_count": delivered["failure_count"],
                "delivery_semantic_manifest_sha256":
                delivered["semantic_manifest_sha256"],
            }

        checkpoint_paths = [
            places_path, occurrences_path, search_path, classification_path,
            failure_path, archive_path, failure_history_path,
        ]
        checkpoint_paths.extend(sorted(completed_root.glob("*.json")))
        delivery_root = run_root / "delivery"
        if delivery_root.is_dir():
            checkpoint_paths.extend(
                path for path in delivery_root.rglob("*") if path.is_file()
            )
        files = _checkpoint_files(run_root, tuple(checkpoint_paths))
        final_status = (
            f"{config.run_mode}_complete"
            if len(completed) == len(candidates)
            else f"{config.run_mode}_incomplete"
        )
        manifest = {
            **running_manifest, "status": final_status,
            "finished_at": _iso(self.clock()),
            "candidate_count": len(candidates),
            "completed_count": len(completed),
            "failure_count": len(failures), "files": files,
            **delivery_metadata,
        }
        _atomic_json(manifest_path, manifest)
        return manifest
