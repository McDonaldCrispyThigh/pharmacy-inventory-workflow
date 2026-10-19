"""Atomic, privacy-safe delivery orchestration for v2.1 downstream outputs.

The live acquisition workflow may construct :class:`DeliveryCandidateV2` records
only after its authorization gate.  This module itself performs no provider calls,
reads no credentials, and accepts Board/spatial inputs only after every candidate has
passed through the Board-blind eligibility reducer.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from v2_board_adapter import (
    LINKAGE_ADAPTER_VERSION,
    LinkageConfigV2,
    prepare_v2_board_frames,
    prepare_v2_google_frame,
    run_v2_board_linkage,
    merge_v2_board_sources,
)
from v2_classifier import PROMPT_TEMPLATE_SHA256, PROMPT_VERSION, is_target_match
from v2_downstream_models import CandidateLocationV2
from v2_evidence_models import (
    ClassifierContractV2,
    LLMDecisionV2,
    QueryProvenanceV2,
    evidence_id,
)
from v2_eligibility import (
    ELIGIBILITY_CONTRACT_SHA256,
    ELIGIBILITY_CONTRACT_VERSION,
    EligibilityDecisionV2,
    classification_decision_sha256,
    reduce_eligibility,
)
from v2_pilot import PILOT_ROLES, summarize_ablation
from v2_spatial_adapter import (
    SPATIAL_ADAPTER_VERSION,
    SpatialContractV2,
    prepare_fixed_tract_origins,
    run_v2_spatial_analysis,
)


DELIVERY_CONTRACT_VERSION = "nets-ai-v2.1-delivery-v1"
FULL_STAGE_ORDER = (
    "candidate_lineage",
    "eligibility_reducer",
    "board_preparation",
    "board_linkage",
    "spatial_analysis",
    "safe_serialization",
)
PILOT_STAGE_ORDER = (
    "pilot_selection",
    "ablation_A_B_C",
    "privacy_and_board_separation_gate",
    "safe_serialization",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACE_ID = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
_RUN_METADATA_KEYS = frozenset(
    {
        "run_id",
        "model",
        "executed_at",
        "authorization_file_sha256",
        "written_permission_sha256",
        "promotion_file_sha256",
        "provider_version",
        "evidence_version",
    }
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


_DELIVERY_CONTRACT_PAYLOAD = {
    "contract_version": DELIVERY_CONTRACT_VERSION,
    "candidate_unit": "unique_google_place_id",
    "full_stage_order": FULL_STAGE_ORDER,
    "pilot_stage_order": PILOT_STAGE_ORDER,
    "public_restricted_fields": (
        "name",
        "address",
        "coordinates",
        "review_text",
        "prompt",
        "raw_model_output",
    ),
    "full_artifacts": (
        "candidate_inventory.csv",
        "eligibility.csv",
        "unresolved_cases.csv",
        "board_links.csv",
        "unmatched_google.csv",
        "unmatched_board.csv",
        "performance_metrics.json",
        "ablation_results.jsonl",
        "ablation_summary.json",
        "tract_spatial.csv",
        "coordinate_qa.csv",
        "coordinate_ledger.csv",
        "spatial_summary.json",
    ),
}
DELIVERY_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_bytes(_DELIVERY_CONTRACT_PAYLOAD)
).hexdigest()


def _json_domain(value: Any, label: str) -> Any:
    """Return a detached JSON-domain value, rejecting non-finite/opaque objects."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc


def _validated_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("run metadata must be a mapping")
    if not set(value) <= _RUN_METADATA_KEYS:
        raise ValueError("run metadata contains undeclared fields")
    result = _json_domain(dict(value), "run metadata")
    hashes = {
        "authorization_file_sha256", "written_permission_sha256",
        "promotion_file_sha256",
    }
    for key, item in result.items():
        if not isinstance(item, str) or not item or len(item) > 256:
            raise ValueError("run metadata values must be bounded strings")
        if any(ord(char) < 32 for char in item):
            raise ValueError("run metadata contains control characters")
        if key in hashes and _SHA256.fullmatch(item) is None:
            raise ValueError("run metadata hash is invalid")
        if key == "run_id" and (
            re.fullmatch(r"\d{8}T\d{6}Z", item) is None
            and re.fullmatch(r"[a-z][a-z0-9-]{2,63}", item) is None
        ):
            raise ValueError("run metadata ID is invalid")
        if key in {"model", "provider_version", "evidence_version"} and (
            re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", item) is None
        ):
            raise ValueError("run metadata identifier is invalid")
        if key == "executed_at":
            try:
                timestamp = pd.Timestamp(
                    item[:-1] + "+00:00" if item.endswith("Z") else item
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("run metadata time is invalid") from exc
            if timestamp.tzinfo is None:
                raise ValueError("run metadata time lacks timezone")
    return result


def _validated_stage_failures(value: Mapping[str, Any]) -> dict[str, int]:
    failures = dict(value)
    if any(
        not isinstance(key, str)
        or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key)
        or type(count) is not int
        or count < 0
        for key, count in failures.items()
    ):
        raise ValueError("invalid stage failure counts")
    return failures


def _validated_source_hashes(value: Mapping[str, Any]) -> dict[str, str]:
    hashes = dict(value)
    if any(
        not isinstance(key, str)
        or not re.fullmatch(r"[A-Za-z0-9._/() -]{1,256}", key)
        or Path(key).is_absolute()
        or ".." in Path(key).parts
        or Path(key).parts[0] not in {
            "analysis", "data", "docs", "pilot", "tests",
        }
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        for key, digest in hashes.items()
    ):
        raise ValueError("invalid delivery source hashes")
    return hashes


@dataclass(frozen=True, slots=True, repr=False)
class DeliveryCandidateV2:
    """One transient Details snapshot paired with its validated model decision."""

    location: CandidateLocationV2
    decision: LLMDecisionV2 | None
    failure_status: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.location, CandidateLocationV2):
            raise TypeError("delivery candidate requires a location envelope")
        if self.decision is not None and not isinstance(self.decision, LLMDecisionV2):
            raise TypeError("delivery candidate requires a validated decision")
        if self.decision is None:
            if (
                not isinstance(self.failure_status, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.failure_status)
            ):
                raise ValueError("missing decision requires a safe failure status")
        elif self.failure_status is not None:
            raise ValueError("successful decision cannot retain a failure status")

    @property
    def decision_sha256(self) -> str | None:
        return (
            classification_decision_sha256(self.decision)
            if self.decision is not None
            else None
        )

    def reduce(self) -> EligibilityDecisionV2:
        return reduce_eligibility(
            self.decision,
            place_id=self.location.place_id,
            candidate_id=self.location.candidate_id,
            classification_decision_sha256=self.decision_sha256,
        )

    def lineage_record(self) -> dict[str, Any]:
        return {
            "place_id": self.location.place_id,
            "candidate_id": self.location.candidate_id,
            "candidate_snapshot_sha256": self.location.candidate_snapshot_sha256,
            "classification_decision_sha256": self.decision_sha256,
        }


@dataclass(frozen=True, slots=True, repr=False)
class DeliveryInputsV2:
    candidates: tuple[DeliveryCandidateV2, ...]
    board: pd.DataFrame
    target_zips: tuple[str, ...]
    origins: gpd.GeoDataFrame
    ablation_rows: tuple[Mapping[str, Any], ...]
    run_metadata: Mapping[str, Any]
    empirical_v2_results_exist: bool
    source_label: str
    linkage_contract: LinkageConfigV2 = field(default_factory=LinkageConfigV2)
    spatial_contract: SpatialContractV2 = field(default_factory=SpatialContractV2)
    stage_failures: Mapping[str, int] = field(default_factory=dict)
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    live_provenance: LiveDeliveryProvenanceV2 | None = None

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if not candidates or any(
            not isinstance(item, DeliveryCandidateV2) for item in candidates
        ):
            raise TypeError("delivery candidates must be a nonempty typed tuple")
        place_ids = [item.location.place_id for item in candidates]
        if len(place_ids) != len(set(place_ids)):
            raise ValueError("delivery Place IDs must be unique")
        if not isinstance(self.board, pd.DataFrame):
            raise TypeError("delivery Board source must be a DataFrame")
        if not isinstance(self.origins, gpd.GeoDataFrame) or self.origins.crs is None:
            raise ValueError("delivery origins require a GeoDataFrame with CRS")
        zips = tuple(self.target_zips)
        if (
            not zips
            or len(zips) != len(set(zips))
            or any(not isinstance(item, str) or re.fullmatch(r"\d{5}", item) is None for item in zips)
        ):
            raise ValueError("delivery target ZIP frame is invalid")
        rows = tuple(self.ablation_rows)
        if not rows or any(not isinstance(item, Mapping) for item in rows):
            raise TypeError("delivery requires a safe A/B/C ablation ledger")
        if type(self.empirical_v2_results_exist) is not bool:
            raise TypeError("empirical result flag must be boolean")
        if self.empirical_v2_results_exist:
            if not isinstance(self.live_provenance, LiveDeliveryProvenanceV2):
                raise ValueError(
                    "empirical delivery requires gated live-run provenance"
                )
        elif self.live_provenance is not None:
            raise ValueError("offline delivery cannot carry live-run provenance")
        if (
            not isinstance(self.source_label, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", self.source_label)
        ):
            raise ValueError("invalid delivery source label")
        if not isinstance(self.linkage_contract, LinkageConfigV2):
            raise TypeError("invalid delivery linkage contract")
        if not isinstance(self.spatial_contract, SpatialContractV2):
            raise TypeError("invalid delivery spatial contract")
        failures = _validated_stage_failures(self.stage_failures)
        source_hashes = _validated_source_hashes(self.source_hashes)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "target_zips", zips)
        object.__setattr__(self, "ablation_rows", rows)
        object.__setattr__(self, "run_metadata", _validated_metadata(self.run_metadata))
        object.__setattr__(self, "stage_failures", failures)
        object.__setattr__(self, "source_hashes", source_hashes)


@dataclass(frozen=True, slots=True, repr=False)
class LocalDeliveryContextV2:
    """Authenticated local Board, origin, and completed-pilot inputs."""

    board: pd.DataFrame
    origins: gpd.GeoDataFrame
    ablation_rows: tuple[Mapping[str, Any], ...]
    source_hashes: Mapping[str, str]


_LIVE_PROVENANCE_SEAL_TOKEN = object()


class _LiveProvenanceSeal:
    __slots__ = ("binding_sha256",)

    def __init__(self, payload: Mapping[str, Any], token: object) -> None:
        if token is not _LIVE_PROVENANCE_SEAL_TOKEN:
            raise ValueError("live provenance seal is private")
        self.binding_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _live_provenance_binding(
    *,
    run_id: str,
    acquisition_checkpoint_sha256: str,
    authorization_file_sha256: str,
    written_permission_sha256: str,
    promotion_file_sha256: str,
    pilot_manifest_sha256: str,
    pilot_semantic_manifest_sha256: str,
    candidate_lineage_sha256: str,
    candidate_count: int,
    checkpoint_file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "acquisition_checkpoint_sha256": acquisition_checkpoint_sha256,
        "authorization_file_sha256": authorization_file_sha256,
        "written_permission_sha256": written_permission_sha256,
        "promotion_file_sha256": promotion_file_sha256,
        "pilot_manifest_sha256": pilot_manifest_sha256,
        "pilot_semantic_manifest_sha256": pilot_semantic_manifest_sha256,
        "candidate_lineage_sha256": candidate_lineage_sha256,
        "candidate_count": candidate_count,
        "checkpoint_file_hashes": dict(sorted(checkpoint_file_hashes.items())),
    }


@dataclass(frozen=True, slots=True, repr=False)
class LiveDeliveryProvenanceV2:
    """Sealed provenance derived from a gated, complete acquisition checkpoint."""

    run_id: str
    acquisition_checkpoint_sha256: str
    authorization_file_sha256: str
    written_permission_sha256: str
    promotion_file_sha256: str
    pilot_manifest_sha256: str
    pilot_semantic_manifest_sha256: str
    candidate_lineage_sha256: str
    candidate_count: int
    checkpoint_file_hashes: Mapping[str, str]
    _seal: object = field(repr=False, compare=False)

    def _binding_is_authentic(self) -> bool:
        binding = _live_provenance_binding(
            run_id=self.run_id,
            acquisition_checkpoint_sha256=self.acquisition_checkpoint_sha256,
            authorization_file_sha256=self.authorization_file_sha256,
            written_permission_sha256=self.written_permission_sha256,
            promotion_file_sha256=self.promotion_file_sha256,
            pilot_manifest_sha256=self.pilot_manifest_sha256,
            pilot_semantic_manifest_sha256=self.pilot_semantic_manifest_sha256,
            candidate_lineage_sha256=self.candidate_lineage_sha256,
            candidate_count=self.candidate_count,
            checkpoint_file_hashes=self.checkpoint_file_hashes,
        )
        return (
            isinstance(self._seal, _LiveProvenanceSeal)
            and self._seal.binding_sha256
            == hashlib.sha256(_canonical_bytes(binding)).hexdigest()
        )

    def __post_init__(self) -> None:
        if not self._binding_is_authentic():
            raise ValueError("live delivery provenance must come from the gated checkpoint")
        hashes = (
            self.acquisition_checkpoint_sha256,
            self.authorization_file_sha256,
            self.written_permission_sha256,
            self.promotion_file_sha256,
            self.pilot_manifest_sha256,
            self.pilot_semantic_manifest_sha256,
            self.candidate_lineage_sha256,
            *self.checkpoint_file_hashes.values(),
        )
        if (
            not isinstance(self.run_id, str)
            or not re.fullmatch(r"\d{8}T\d{6}Z", self.run_id)
            or type(self.candidate_count) is not int
            or self.candidate_count < 1
            or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None
                   for item in hashes)
        ):
            raise ValueError("live delivery provenance is invalid")

    def to_dict(self) -> dict[str, Any]:
        if not self._binding_is_authentic():
            raise ValueError("live delivery provenance binding changed")
        return {
            "run_id": self.run_id,
            "acquisition_checkpoint_sha256": self.acquisition_checkpoint_sha256,
            "authorization_file_sha256": self.authorization_file_sha256,
            "written_permission_sha256": self.written_permission_sha256,
            "promotion_file_sha256": self.promotion_file_sha256,
            "pilot_manifest_sha256": self.pilot_manifest_sha256,
            "pilot_semantic_manifest_sha256":
            self.pilot_semantic_manifest_sha256,
            "candidate_lineage_sha256": self.candidate_lineage_sha256,
            "candidate_count": self.candidate_count,
            "checkpoint_file_hashes": dict(
                sorted(self.checkpoint_file_hashes.items())
            ),
        }


_LOCAL_INPUT_PATHS = (
    "data/board_geocoded.csv",
    "data/Board/F.1  Pharmacy (PH).csv",
    "data/tl_2023_27_tract.zip",
    "data/tl_2023_us_zcta520.zip",
    "data/acs_tracts_2023.csv",
    "data/acs_tracts_vehicles_2023.csv",
)


@dataclass(frozen=True, slots=True, repr=False)
class PilotDeliveryInputsV2:
    selection_ledger: tuple[Mapping[str, Any], ...]
    unavailable_roles: tuple[str, ...]
    ablation_rows: tuple[Mapping[str, Any], ...]
    ablation_summary: Mapping[str, Any]
    run_metadata: Mapping[str, Any]
    empirical_v2_results_exist: bool

    def __post_init__(self) -> None:
        selection = _validated_pilot_selection(self.selection_ledger)
        rows = tuple(self.ablation_rows)
        if not 8 <= len(selection) <= 10:
            raise ValueError("pilot delivery requires 8--10 selected candidates")
        if any(not isinstance(item, Mapping) for item in rows):
            raise TypeError("pilot ledgers must contain mappings")
        summary = _json_domain(dict(self.ablation_summary), "ablation summary")
        if summary.get("candidate_count") != len(selection):
            raise ValueError("pilot summary and selection counts differ")
        if type(self.empirical_v2_results_exist) is not bool:
            raise TypeError("empirical result flag must be boolean")
        roles = tuple(self.unavailable_roles)
        if (
            len(roles) != len(set(roles))
            or any(item not in PILOT_ROLES for item in roles)
            or roles != tuple(role for role in PILOT_ROLES if role in roles)
        ):
            raise ValueError("invalid unavailable pilot roles")
        object.__setattr__(self, "selection_ledger", selection)
        object.__setattr__(self, "unavailable_roles", roles)
        object.__setattr__(self, "ablation_rows", rows)
        object.__setattr__(self, "ablation_summary", summary)
        object.__setattr__(self, "run_metadata", _validated_metadata(self.run_metadata))


_PILOT_SELECTION_FIELDS = frozenset({
    "selection_rank", "provider_order_index", "place_id", "candidate_id", "roles",
})


def _validated_pilot_selection(
    selection: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Project the exact public pilot-selection schema or fail before writing."""
    if not isinstance(selection, (tuple, list)) or not 8 <= len(selection) <= 10:
        raise ValueError("pilot selection requires 8--10 rows")
    projected: list[dict[str, Any]] = []
    for row in selection:
        if type(row) is not dict or set(row) != _PILOT_SELECTION_FIELDS:
            raise ValueError("pilot selection row has undeclared fields")
        rank = row["selection_rank"]
        provider_index = row["provider_order_index"]
        place_id = row["place_id"]
        candidate_id = row["candidate_id"]
        roles = row["roles"]
        if (
            type(rank) is not int
            or type(provider_index) is not int
            or provider_index < 0
            or not isinstance(place_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{10,256}", place_id)
            or candidate_id != evidence_id(
                "candidate", {"source_place_id": place_id}
            )
            or type(roles) is not list
            or any(not isinstance(role, str) for role in roles)
            or len(roles) != len(set(roles))
            or any(role not in PILOT_ROLES for role in roles)
            or roles != [role for role in PILOT_ROLES if role in roles]
        ):
            raise ValueError("pilot selection row violates its closed domain")
        projected.append({
            "selection_rank": rank,
            "provider_order_index": provider_index,
            "place_id": place_id,
            "candidate_id": candidate_id,
            "roles": list(roles),
        })
    if (
        {row["selection_rank"] for row in projected}
        != set(range(1, len(projected) + 1))
        or len({row["provider_order_index"] for row in projected})
        != len(projected)
        or len({row["place_id"] for row in projected}) != len(projected)
        or len({row["candidate_id"] for row in projected}) != len(projected)
    ):
        raise ValueError("pilot selection ranks and identities must be unique")
    return tuple(projected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_authenticated_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if not raw or len(raw) > 5_000_000:
        raise ValueError(f"{label} has invalid size")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


_CLASSIFICATION_FIELDS = frozenset({
    "place_id", "candidate_id", "candidate_snapshot_sha256",
    "calculated_naics", "target_construct_status", "business_status",
    "candidate_plausibility", "identity_consistency", "historical_year_value",
    "historical_year_type", "claim_confidence", "claim_evidence_ids",
    "reasoning_hashes", "provider_metadata",
})
_PROVIDER_METADATA_FIELDS = frozenset({
    "status", "requested_model", "returned_model", "requested_at",
    "schema_version", "prompt_template_version", "prompt_template_sha256",
    "system_fingerprint", "token_usage", "decision_sha256",
    "is_target_match",
})
_CLAIM_FIELDS = frozenset({
    "calculated_naics", "target_construct_status", "business_status",
    "candidate_plausibility", "identity_consistency", "historical_year",
})
_ARCHIVE_FIELDS = frozenset({
    "query_status", "page_status", "cap_status", "query_key_digest",
    "query_scope", "retrieved_at", "max_cdx_pages", "max_cdx_records",
    "cdx_response_chain_sha256", "cdx_pages", "annual_selections",
    "fetch_records", "source_snapshots", "opening_claims",
})
_ARCHIVE_MEMBER_FIELDS = {
    "cdx_pages": frozenset({
        "page_number", "status", "attempts", "retrieved_at", "http_status",
        "response_sha256", "response_length", "returned_rows",
        "returned_years", "incoming_resume_key_digest",
        "outgoing_resume_key_digest", "pagination_complete",
        "cache_provenance", "capture_rows",
    }),
    "annual_selections": frozenset({
        "year", "max_attempts", "attempted_capture_request_ids",
        "selected_source_evidence_id", "stop_reason",
    }),
    "fetch_records": frozenset({
        "capture_request_id", "source_evidence_id", "status", "attempts",
        "retrieved_at", "http_status", "response_sha256", "cache_provenance",
    }),
    "source_snapshots": frozenset({
        "evidence_id", "capture_request_id", "captured_at", "replay_url",
        "original_url", "archived_page_text", "page_text_sha256",
        "body_truncated", "text_truncated",
    }),
    "opening_claims": frozenset({
        "evidence_id", "source_evidence_id", "year", "excerpt",
        "page_text_sha256", "extractor_rule", "extractor_version",
    }),
}
_CAPTURE_ROW_FIELDS = frozenset({
    "evidence_id", "captured_at", "original_url", "mimetype",
    "status_code", "digest", "length",
})


def _candidate_lineage_sha256(
    candidates: Sequence[DeliveryCandidateV2],
) -> str:
    rows = sorted(
        (item.lineage_record() for item in candidates),
        key=lambda row: row["place_id"],
    )
    if not rows or len({row["place_id"] for row in rows}) != len(rows):
        raise ValueError("candidate lineage must be nonempty and unique")
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def _checkpoint_decision(row: dict[str, Any], expected_model: str) -> LLMDecisionV2:
    if type(row) is not dict or set(row) != _CLASSIFICATION_FIELDS:
        raise ValueError("live classification checkpoint has undeclared fields")
    place_id = row.get("place_id")
    candidate_id = row.get("candidate_id")
    if (
        not isinstance(place_id, str)
        or _PLACE_ID.fullmatch(place_id) is None
        or candidate_id != evidence_id("candidate", {"source_place_id": place_id})
        or not isinstance(row.get("candidate_snapshot_sha256"), str)
        or _SHA256.fullmatch(row["candidate_snapshot_sha256"]) is None
    ):
        raise ValueError("live classification identity is invalid")
    try:
        decision = LLMDecisionV2(
            row["calculated_naics"], row["target_construct_status"],
            row["business_status"], row["candidate_plausibility"],
            row["identity_consistency"], row["historical_year_value"],
            row["historical_year_type"], row["claim_confidence"],
            row["claim_evidence_ids"], row["reasoning_hashes"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("live classification decision is invalid") from exc
    if any(
        ref.startswith("candidate_") and ref != candidate_id
        for refs in decision.claim_evidence_ids.values()
        for ref in refs
    ):
        raise ValueError("live classification cites another candidate")
    provider = row["provider_metadata"]
    token_fields = {"input_tokens", "output_tokens", "total_tokens"}
    decision_hash = classification_decision_sha256(decision)
    if (
        type(provider) is not dict
        or set(provider) != _PROVIDER_METADATA_FIELDS
        or provider.get("status") != "success"
        or provider.get("requested_model") != expected_model
        or provider.get("schema_version")
        != ClassifierContractV2.from_project_constants().schema_version
        or provider.get("prompt_template_version") != PROMPT_VERSION
        or provider.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256
        or provider.get("decision_sha256") != decision_hash
        or provider.get("is_target_match") is not is_target_match(decision)
        or type(provider.get("token_usage")) is not dict
        or set(provider["token_usage"]) != token_fields
        or any(
            value is not None and (type(value) is not int or value < 0)
            for value in provider["token_usage"].values()
        )
    ):
        raise ValueError("live classifier provider metadata is invalid")
    for key in ("returned_model", "system_fingerprint"):
        value = provider[key]
        if value is not None and (
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) is None
        ):
            raise ValueError("live classifier provider identifier is invalid")
    if not isinstance(provider.get("requested_at"), str):
        raise ValueError("live classifier request time is invalid")
    raw_time = (
        provider["requested_at"][:-1] + "+00:00"
        if provider["requested_at"].endswith("Z")
        else provider["requested_at"]
    )
    try:
        requested_at = pd.Timestamp(raw_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("live classifier request time is invalid") from exc
    if requested_at.tzinfo is None:
        raise ValueError("live classifier request time lacks timezone")
    usage = provider["token_usage"]
    if all(type(usage[key]) is int for key in token_fields) and (
        usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
    ):
        raise ValueError("live classifier token total is inconsistent")
    return decision


def _validate_archive_checkpoint(row: dict[str, Any]) -> None:
    if set(row) != {"place_id", "archive_evidence"}:
        raise ValueError("live archive checkpoint has undeclared fields")
    archive = row.get("archive_evidence")
    if type(archive) is not dict or set(archive) != _ARCHIVE_FIELDS:
        raise ValueError("live archive evidence has undeclared fields")
    for container, fields in _ARCHIVE_MEMBER_FIELDS.items():
        members = archive.get(container)
        if type(members) is not list or any(
            type(member) is not dict or set(member) != fields
            for member in members
        ):
            raise ValueError("live archive member has undeclared fields")
    for page in archive["cdx_pages"]:
        captures = page.get("capture_rows")
        if type(captures) is not list or any(
            type(capture) is not dict or set(capture) != _CAPTURE_ROW_FIELDS
            for capture in captures
        ):
            raise ValueError("live archive capture row has undeclared fields")


def authenticate_live_delivery_provenance(
    run_root: Path,
    expected_run_id: str,
    *,
    expected_authorization_file_sha256: str,
    expected_written_permission_sha256: str,
    expected_promotion_file_sha256: str,
    expected_candidates: Sequence[DeliveryCandidateV2],
) -> LiveDeliveryProvenanceV2:
    """Seal provenance only from the workflow's gated pre-delivery checkpoint."""
    if not isinstance(run_root, Path):
        raise TypeError("live run root must be a pathlib.Path")
    root = run_root.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = _read_authenticated_json(manifest_path, "live acquisition manifest")
    expected_hashes = (
        expected_authorization_file_sha256,
        expected_written_permission_sha256,
        expected_promotion_file_sha256,
    )
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in expected_hashes
    ):
        raise ValueError("authenticated gate hashes are invalid")
    candidates = tuple(expected_candidates)
    if not candidates or any(
        not isinstance(item, DeliveryCandidateV2) for item in candidates
    ):
        raise TypeError("live provenance requires typed transient candidates")
    expected_lineage_sha256 = _candidate_lineage_sha256(candidates)
    required_manifest_hashes = (
        "authorization_file_sha256",
        "written_permission_sha256",
        "promotion_file_sha256",
        "pilot_manifest_sha256",
        "pilot_semantic_manifest_sha256",
    )
    if (
        manifest.get("schema_version") != "nets-ai-wayback-llm-v2"
        or manifest.get("status") != "running"
        or manifest.get("run_mode") != "full"
        or manifest.get("run_id") != expected_run_id
        or manifest.get("authorization_file_sha256")
        != expected_authorization_file_sha256
        or manifest.get("written_permission_sha256")
        != expected_written_permission_sha256
        or manifest.get("promotion_file_sha256")
        != expected_promotion_file_sha256
        or any(
            not isinstance(manifest.get(key), str)
            or _SHA256.fullmatch(manifest[key]) is None
            for key in required_manifest_hashes
        )
    ):
        raise ValueError("live acquisition checkpoint is not gated and complete")
    required_files = (
        "place_ids.jsonl",
        "candidate_occurrences.jsonl",
        "search_provenance.json",
        "classification_v2.jsonl",
        "failures.jsonl",
        "private/archive_evidence.jsonl",
    )
    checkpoint_hashes: dict[str, str] = {}
    for relative in required_files:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("live acquisition checkpoint file is unavailable")
        checkpoint_hashes[relative] = _sha256_file(path)

    def jsonl(relative: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            for line in (root / relative).read_text(encoding="utf-8").splitlines():
                if line:
                    value = json.loads(line)
                    if type(value) is not dict:
                        raise ValueError
                    rows.append(value)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("live acquisition checkpoint JSONL is invalid") from exc
        return rows

    places = jsonl("place_ids.jsonl")
    occurrences = jsonl("candidate_occurrences.jsonl")
    classifications = jsonl("classification_v2.jsonl")
    archives = jsonl("private/archive_evidence.jsonl")
    failures = jsonl("failures.jsonl")
    place_ids = [row.get("place_id") for row in places]
    if any(
        set(row) != {"place_id", "source_zip", "query_definition_id"}
        or not isinstance(row.get("place_id"), str)
        or _PLACE_ID.fullmatch(row["place_id"]) is None
        or not isinstance(row.get("source_zip"), str)
        or re.fullmatch(r"\d{5}", row["source_zip"]) is None
        for row in places
    ):
        raise ValueError("live Place-ID checkpoint has undeclared fields")
    for row in places:
        QueryProvenanceV2(
            row.get("query_definition_id"), row["source_zip"], "places_search"
        )
    occurrence_ids: set[str] = set()
    for row in occurrences:
        if (
            set(row) != {
                "place_id", "query_definition_id", "source_zip", "page_number",
            }
            or not isinstance(row.get("place_id"), str)
            or _PLACE_ID.fullmatch(row["place_id"]) is None
            or not isinstance(row.get("source_zip"), str)
            or re.fullmatch(r"\d{5}", row["source_zip"]) is None
            or type(row.get("page_number")) is not int
            or row["page_number"] < 1
            # The live search adapter follows provider-issued continuation
            # tokens through its fail-closed ten-page safety ceiling.
            or row["page_number"] > 10
        ):
            raise ValueError("live candidate occurrence ledger is invalid")
        QueryProvenanceV2(
            row.get("query_definition_id"), row["source_zip"], "places_search"
        )
        occurrence_ids.add(row["place_id"])
    if (
        not places
        or not occurrences
        or failures
        or len(place_ids) != len(set(place_ids))
        or not set(place_ids).issubset(occurrence_ids)
        or {row.get("place_id") for row in classifications} != set(place_ids)
        or {row.get("place_id") for row in archives} != set(place_ids)
        or len(classifications) != len(places)
        or len(archives) != len(places)
    ):
        raise ValueError("live acquisition records are incomplete or inconsistent")
    classification_lineage: list[dict[str, Any]] = []
    for row in classifications:
        decision = _checkpoint_decision(row, manifest.get("model"))
        classification_lineage.append({
            "place_id": row["place_id"],
            "candidate_id": row["candidate_id"],
            "candidate_snapshot_sha256": row["candidate_snapshot_sha256"],
            "classification_decision_sha256":
            classification_decision_sha256(decision),
        })
    for row in archives:
        _validate_archive_checkpoint(row)
    observed_lineage_sha256 = hashlib.sha256(_canonical_bytes(sorted(
        classification_lineage, key=lambda row: row["place_id"]
    ))).hexdigest()
    if observed_lineage_sha256 != expected_lineage_sha256:
        raise ValueError("live transient candidates differ from their checkpoint")
    checkpoint_payload = {
        "run_id": expected_run_id,
        "manifest_sha256": _sha256_file(manifest_path),
        "candidate_count": len(places),
        "checkpoint_file_hashes": checkpoint_hashes,
    }
    checkpoint_sha = hashlib.sha256(
        _canonical_bytes(checkpoint_payload)
    ).hexdigest()
    provenance_fields = {
        "run_id": expected_run_id,
        "acquisition_checkpoint_sha256": checkpoint_sha,
        "authorization_file_sha256": manifest["authorization_file_sha256"],
        "written_permission_sha256": manifest["written_permission_sha256"],
        "promotion_file_sha256": manifest["promotion_file_sha256"],
        "pilot_manifest_sha256": manifest["pilot_manifest_sha256"],
        "pilot_semantic_manifest_sha256":
        manifest["pilot_semantic_manifest_sha256"],
        "candidate_lineage_sha256": observed_lineage_sha256,
        "candidate_count": len(places),
        "checkpoint_file_hashes": checkpoint_hashes,
    }
    return LiveDeliveryProvenanceV2(
        **provenance_fields,
        _seal=_LiveProvenanceSeal(
            _live_provenance_binding(**provenance_fields),
            _LIVE_PROVENANCE_SEAL_TOKEN,
        ),
    )


def authenticate_pilot_ablation_package(
    pilot_root: Path,
    *,
    expected_authorization_file_sha256: str | None = None,
    expected_written_permission_sha256: str | None = None,
    expected_model: str | None = None,
    expected_run_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_semantic_sha256: str | None = None,
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, str], dict[str, Any]]:
    """Authenticate one completed, empirical pilot package and return safe rows."""
    if not isinstance(pilot_root, Path):
        raise TypeError("pilot delivery root must be a pathlib.Path")
    root = pilot_root.resolve()
    manifest_path = root / "run_manifest.json"
    manifest = _read_authenticated_json(manifest_path, "pilot manifest")
    observed_manifest_hash = _sha256_file(manifest_path)
    expected_artifacts = {
        "pilot_selection.json",
        "ablation_results.jsonl",
        "ablation_summary.json",
    }
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != DELIVERY_CONTRACT_VERSION
        or manifest.get("status") != "pilot_complete"
        or manifest.get("source_label") != "pilot_ablation"
        or manifest.get("empirical_v2_results_exist") is not True
        or manifest.get("failure_count") != 0
        or manifest.get("spatial_outputs_created") is not False
        or manifest.get("stage_order") != list(PILOT_STAGE_ORDER)
        or type(manifest.get("candidate_count")) is not int
        or not 8 <= manifest["candidate_count"] <= 10
        or type(artifacts) is not dict
        or set(artifacts) != expected_artifacts
    ):
        raise ValueError("pilot package is not a complete empirical A/B/C run")
    run_metadata = manifest.get("run_metadata")
    if type(run_metadata) is not dict:
        raise ValueError("pilot package lacks run metadata")
    expected_metadata = {
        "authorization_file_sha256": expected_authorization_file_sha256,
        "written_permission_sha256": expected_written_permission_sha256,
        "model": expected_model,
        "run_id": expected_run_id,
    }
    for key, expected in expected_metadata.items():
        if expected is not None and run_metadata.get(key) != expected:
            raise ValueError("pilot package does not match full-run provenance")
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_hash != expected_manifest_sha256
    ):
        raise ValueError("pilot manifest is not the promoted manifest")
    for relative, expected_hash in artifacts.items():
        artifact_path = (root / relative).resolve()
        if (
            not artifact_path.is_relative_to(root)
            or not artifact_path.is_file()
            or not isinstance(expected_hash, str)
            or _SHA256.fullmatch(expected_hash) is None
            or _sha256_file(artifact_path) != expected_hash
        ):
            raise ValueError("pilot artifact authentication failed")
    semantic = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_metadata", "artifacts", "semantic_manifest_sha256"}
    }
    semantic["artifacts"] = artifacts
    expected_semantic = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
    if manifest.get("semantic_manifest_sha256") != expected_semantic:
        raise ValueError("pilot semantic manifest hash does not match")
    if (
        expected_semantic_sha256 is not None
        and expected_semantic != expected_semantic_sha256
    ):
        raise ValueError("pilot semantic result is not the promoted result")
    rows_path = root / "ablation_results.jsonl"
    rows: list[Mapping[str, Any]] = []
    try:
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if type(row) is not dict:
                    raise ValueError
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("pilot ablation ledger is invalid") from exc
    summary = _read_authenticated_json(
        root / "ablation_summary.json", "pilot ablation summary"
    )
    recomputed = summarize_ablation(
        tuple(rows),
        privacy_check_passed=True,
        board_separation_passed=True,
    )
    if summary != recomputed or recomputed.get("full_run_recommendation") != "proceed":
        raise ValueError("pilot ablation summary is not clean or reproducible")
    return tuple(rows), {
        "pilot/run_manifest.json": observed_manifest_hash,
        "pilot/semantic_manifest": expected_semantic,
        "pilot/ablation_results.jsonl": artifacts["ablation_results.jsonl"],
        "pilot/ablation_summary.json": artifacts["ablation_summary.json"],
    }, manifest


def load_local_delivery_context(
    input_manifest_path: Path,
    pilot_delivery_root: Path,
    target_zips: Sequence[str],
    *,
    project_root: Path = _PROJECT_ROOT,
    pilot_expectations: Mapping[str, str] | None = None,
) -> LocalDeliveryContextV2:
    """Hash-verify and load the declared local full-delivery inputs.

    Callers must invoke this only after the workflow's written-permission gate.
    The function performs local reads only and never accesses credentials or network.
    """
    if not all(isinstance(item, Path) for item in (
        input_manifest_path, pilot_delivery_root, project_root
    )):
        raise TypeError("local delivery paths must be pathlib.Path values")
    root = project_root.resolve()
    manifest_path = input_manifest_path.resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("delivery input manifest must remain inside the project")
    manifest = _read_authenticated_json(manifest_path, "delivery input manifest")
    entries = manifest.get("inputs")
    if manifest.get("schema_version") != 1 or type(entries) is not list:
        raise ValueError("delivery input manifest contract is invalid")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if type(entry) is not dict or not isinstance(entry.get("path"), str):
            raise ValueError("delivery input manifest row is invalid")
        relative = entry["path"]
        if relative in by_path:
            raise ValueError("delivery input manifest duplicates a path")
        by_path[relative] = entry
    if any(relative not in by_path for relative in _LOCAL_INPUT_PATHS):
        raise ValueError("delivery input manifest omits a required local source")
    paths: dict[str, Path] = {}
    hashes = {"analysis/input_manifest.json": _sha256_file(manifest_path)}
    for relative in _LOCAL_INPUT_PATHS:
        relative_path = Path(relative)
        path = (root / relative_path).resolve()
        expected_hash = by_path[relative].get("sha256")
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not path.is_relative_to(root)
            or not path.is_file()
            or not isinstance(expected_hash, str)
            or _SHA256.fullmatch(expected_hash) is None
            or _sha256_file(path) != expected_hash
        ):
            raise ValueError(f"local delivery input failed authentication: {relative}")
        paths[relative] = path
        hashes[relative] = expected_hash

    expected = dict(pilot_expectations or {})
    allowed_expectations = {
        "authorization_file_sha256", "written_permission_sha256", "model",
        "run_id", "manifest_sha256", "semantic_sha256",
    }
    if not set(expected) <= allowed_expectations:
        raise ValueError("unknown pilot provenance expectation")
    ablation_rows, pilot_hashes, _ = authenticate_pilot_ablation_package(
        pilot_delivery_root,
        expected_authorization_file_sha256=expected.get(
            "authorization_file_sha256"
        ),
        expected_written_permission_sha256=expected.get(
            "written_permission_sha256"
        ),
        expected_model=expected.get("model"),
        expected_run_id=expected.get("run_id"),
        expected_manifest_sha256=expected.get("manifest_sha256"),
        expected_semantic_sha256=expected.get("semantic_sha256"),
    )
    hashes.update(pilot_hashes)
    board_raw = pd.read_csv(
        paths["data/Board/F.1  Pharmacy (PH).csv"],
        encoding="utf-8-sig",
        skiprows=3,
        low_memory=False,
    )
    board_cache = pd.read_csv(
        paths["data/board_geocoded.csv"], low_memory=False
    )
    board = merge_v2_board_sources(board_raw, board_cache)
    tracts = gpd.read_file(
        f"zip://{paths['data/tl_2023_27_tract.zip'].as_posix()}"
    )
    zctas = gpd.read_file(
        f"zip://{paths['data/tl_2023_us_zcta520.zip'].as_posix()}"
    )
    acs = pd.read_csv(
        paths["data/acs_tracts_2023.csv"], dtype={"GEOID": str}
    )
    vehicles = pd.read_csv(
        paths["data/acs_tracts_vehicles_2023.csv"], dtype={"GEOID": str}
    )
    origins = prepare_fixed_tract_origins(
        tracts, acs, vehicles, zctas, target_zips
    )
    return LocalDeliveryContextV2(
        board=board,
        origins=origins,
        ablation_rows=ablation_rows,
        source_hashes=hashes,
    )


def _contract_manifest(
    linkage: LinkageConfigV2 | None = None,
    spatial: SpatialContractV2 | None = None,
) -> dict[str, Any]:
    classifier = ClassifierContractV2.from_project_constants()
    result: dict[str, Any] = {
        "delivery_contract_version": DELIVERY_CONTRACT_VERSION,
        "delivery_contract_sha256": DELIVERY_CONTRACT_SHA256,
        "classifier_schema_version": classifier.schema_version,
        "classifier_contract_sha256": classifier.contract_hash,
        "prompt_template_version": PROMPT_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
        "board_adapter_version": LINKAGE_ADAPTER_VERSION,
        "spatial_adapter_version": SPATIAL_ADAPTER_VERSION,
        "candidate_schema_sha256": _sha256_file(
            _PROJECT_ROOT / "docs" / "v2_candidate_schema.json"
        ),
        "eligibility_schema_sha256": _sha256_file(
            _PROJECT_ROOT / "docs" / "v2_eligibility_schema.json"
        ),
    }
    if linkage is not None:
        linkage_payload = {
            "threshold": linkage.threshold,
            "name_weight": linkage.name_weight,
            "address_weight": linkage.address_weight,
            "primary_objective": linkage.primary_objective,
            "sensitivity_objective": linkage.sensitivity_objective,
            "threshold_sensitivity": linkage.threshold_sensitivity,
        }
        result["linkage_contract_sha256"] = hashlib.sha256(
            _canonical_bytes(linkage_payload)
        ).hexdigest()
    if spatial is not None:
        result["spatial_contract_sha256"] = spatial.contract_sha256
        result["spatial_thresholds_miles"] = list(spatial.thresholds_miles)
    return result


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_domain(value, "artifact"),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            _json_domain(dict(row), "JSONL row"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].map(
            lambda value: isinstance(value, (tuple, list, dict))
        ).any():
            result[column] = result[column].map(
                lambda value: json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                if isinstance(value, (tuple, list, dict))
                else value
            )
    return result


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _csv_ready(frame).to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        na_rep="",
        float_format="%.10g",
    )


def _safe_tract_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove geometry/source-coordinate fields while retaining derived distances."""
    forbidden_exact = {
        "geometry",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "intptlat",
        "intptlon",
        "x",
        "y",
    }
    keep = [
        column
        for column in frame.columns
        if str(column).casefold() not in forbidden_exact
    ]
    return frame.loc[:, keep].copy()


_SPATIAL_SURFACES = frozenset({
    "google_include",
    "google_include_unresolved",
    "board_screened",
    "board_screened_exact",
    "board_all_active",
})
_SPATIAL_SUMMARY_FIELDS = frozenset({
    "adapter_version", "analysis_scope", "estimand", "distance_crs",
    "distance_units", "supply_input_crs", "origin_count",
    "origin_id_column", "origin_sha256", "spatial_contract_sha256",
    "spatial_contract", "origin_status", "input_sha256",
    "surface_definitions", "thresholds", "coordinate_qa",
})
_THRESHOLD_FIELDS = frozenset({
    "threshold_miles", "threshold_meters", "origin_count", "surfaces",
    "primary_google_vs_board", "conditional_google_vs_board",
})
_SURFACE_SUMMARY_FIELDS = frozenset({
    "known_origins", "beyond_count", "beyond_pct",
})
_PAIRED_FIELDS = frozenset({
    "paired_origins", "both_beyond", "board_only_beyond",
    "google_only_beyond", "neither_beyond", "paired_disagreements",
})
_SPATIAL_SURFACE_DEFINITIONS = {
    "google_include": "Google candidates reduced to include",
    "google_include_unresolved":
    "conditional sensitivity: Google include plus unresolved",
    "board_screened":
    "automatically screened active Board operational reference",
    "board_screened_exact":
    "screened Board records with Exact cached coordinates only",
    "board_all_active": "all active Board records in the extended frame",
}


def _recomputed_paired(board: pd.Series, google: pd.Series) -> dict[str, int]:
    known = board.notna() & google.notna()
    left = board.loc[known].astype(bool)
    right = google.loc[known].astype(bool)
    return {
        "paired_origins": int(known.sum()),
        "both_beyond": int((left & right).sum()),
        "board_only_beyond": int((left & ~right).sum()),
        "google_only_beyond": int((~left & right).sum()),
        "neither_beyond": int((~left & ~right).sum()),
        "paired_disagreements": int((left != right).sum()),
    }


def _validated_spatial_summary(spatial: Any, contract: SpatialContractV2) -> dict[str, Any]:
    """Close and reconcile the adapter's public summary before persistence."""
    summary = _json_domain(spatial.summary, "spatial summary")
    if type(summary) is not dict or set(summary) != _SPATIAL_SUMMARY_FIELDS:
        raise ValueError("spatial summary has undeclared fields")
    if (
        summary["adapter_version"] != SPATIAL_ADAPTER_VERSION
        or summary["analysis_scope"] != "full"
        or summary["estimand"] != "projected straight-line potential proximity"
        or summary["distance_crs"] != contract.distance_crs
        or summary["distance_units"] != "meter"
        or summary["supply_input_crs"] != contract.supply_crs
        or summary["origin_count"] != len(spatial.tracts)
        or summary["origin_id_column"] != contract.origin_id_column
        or summary["origin_status"]
        != ("ok" if len(spatial.tracts) else "empty_origin_frame")
        or summary["spatial_contract_sha256"] != contract.contract_sha256
        or summary["spatial_contract"]
        != _json_domain(asdict(contract), "spatial contract")
        or not isinstance(summary["origin_sha256"], str)
        or _SHA256.fullmatch(summary["origin_sha256"]) is None
        or type(summary["input_sha256"]) is not dict
        or set(summary["input_sha256"]) != {
            "google", "board_screened_extended", "board_full_active_extended",
        }
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in summary["input_sha256"].values()
        )
        or summary["surface_definitions"] != _SPATIAL_SURFACE_DEFINITIONS
        or type(summary["coordinate_qa"]) is not list
        or sorted(summary["coordinate_qa"], key=lambda row: row.get("surface", ""))
        != sorted(
            _json_domain(spatial.coordinate_qa.to_dict("records"), "coordinate QA"),
            key=lambda row: row.get("surface", ""),
        )
        or type(summary["thresholds"]) is not list
        or len(summary["thresholds"]) != len(contract.thresholds_miles)
    ):
        raise ValueError("spatial summary does not match its adapter result")
    for expected_miles, row in zip(contract.thresholds_miles, summary["thresholds"]):
        if (
            type(row) is not dict
            or set(row) != _THRESHOLD_FIELDS
            or row["threshold_miles"] != expected_miles
            or row["threshold_meters"]
            != round(expected_miles * contract.meters_per_mile, 3)
            or row["origin_count"] != len(spatial.tracts)
            or type(row["surfaces"]) is not dict
            or set(row["surfaces"]) != _SPATIAL_SURFACES
        ):
            raise ValueError("spatial threshold summary violates its contract")
        suffix = str(expected_miles).replace(".", "_") + "mi"
        for surface, observed in row["surfaces"].items():
            column = f"{surface}_beyond_{suffix}"
            values = spatial.tracts[column]
            if (
                type(observed) is not dict
                or set(observed) != _SURFACE_SUMMARY_FIELDS
                or observed["known_origins"] != int(values.notna().sum())
                or observed["beyond_count"] != int(values.sum(skipna=True))
                or observed["beyond_pct"] != (
                    round(100.0 * float(values.dropna().astype(bool).mean()), 2)
                    if int(values.notna().sum()) else None
                )
            ):
                raise ValueError("spatial surface summary does not reconcile")
        board = spatial.tracts[f"board_screened_beyond_{suffix}"]
        primary = spatial.tracts[f"google_include_beyond_{suffix}"]
        conditional = spatial.tracts[
            f"google_include_unresolved_beyond_{suffix}"
        ]
        if (
            row["primary_google_vs_board"] != _recomputed_paired(board, primary)
            or row["conditional_google_vs_board"]
            != _recomputed_paired(board, conditional)
        ):
            raise ValueError("spatial paired summary does not reconcile")
    return summary


def _write_atomic_package(
    output_root: Path,
    artifacts: Mapping[str, tuple[str, Any]],
    manifest_base: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(output_root, Path):
        raise TypeError("delivery output root must be a pathlib.Path")
    target = output_root.resolve()
    if target.exists():
        raise FileExistsError("delivery output already exists")
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        artifact_hashes: dict[str, str] = {}
        for relative, (kind, value) in artifacts.items():
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("delivery artifact path escaped its package")
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "csv":
                if not isinstance(value, pd.DataFrame):
                    raise TypeError("CSV delivery artifact requires a DataFrame")
                _write_csv(path, value)
            elif kind == "json":
                path.write_text(_json_text(value), encoding="utf-8", newline="")
            elif kind == "jsonl":
                path.write_text(_jsonl_text(value), encoding="utf-8", newline="")
            else:
                raise ValueError("unknown delivery artifact kind")
            artifact_hashes[relative] = _sha256_file(path)

        semantic = {
            key: value
            for key, value in manifest_base.items()
            if key not in {"run_metadata"}
        }
        semantic["artifacts"] = artifact_hashes
        semantic_hash = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
        manifest = {
            **_json_domain(dict(manifest_base), "delivery manifest"),
            "artifacts": artifact_hashes,
            "semantic_manifest_sha256": semantic_hash,
        }
        (temporary / "run_manifest.json").write_text(
            _json_text(manifest), encoding="utf-8", newline=""
        )
        os.replace(temporary, target)
        return manifest
    except Exception:
        if temporary.exists() and temporary.parent == parent:
            shutil.rmtree(temporary)
        raise


def _candidate_inventory(
    candidates: Sequence[DeliveryCandidateV2],
    eligibility: Sequence[EligibilityDecisionV2],
) -> pd.DataFrame:
    reduced = {item.place_id: item for item in eligibility}
    rows = []
    for item in sorted(candidates, key=lambda row: row.location.place_id):
        location = item.location
        decision = reduced[location.place_id]
        rows.append(
            {
                "place_id": location.place_id,
                "candidate_id": location.candidate_id,
                "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
                "classification_decision_sha256": decision.classification_decision_sha256,
                "query_definition_id": location.query_provenance.query_definition_id,
                "query_zip": location.query_provenance.source_zip,
                "returned_postal_code": location.returned_postal_code,
                "retrieved_at": location.retrieved_at.isoformat().replace(
                    "+00:00", "Z"
                ),
                "coordinate_state": (
                    "complete"
                    if location.latitude is not None and location.longitude is not None
                    else "missing"
                ),
                "classification_status": (
                    "success" if item.decision is not None else item.failure_status
                ),
                "eligibility_state": decision.eligibility_state,
                "eligibility_reason_codes": decision.reason_codes,
            }
        )
    return pd.DataFrame(rows)


def _validated_safe_ablation(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Fail closed before serialization unless every row matches the safe schema."""
    frozen = tuple(
        _json_domain(dict(row), "ablation row")
        for row in rows
        if isinstance(row, Mapping)
    )
    if len(frozen) != len(rows):
        raise ValueError("ablation ledger contains a non-mapping row")
    summary = summarize_ablation(
        frozen,
        privacy_check_passed=True,
        board_separation_passed=True,
    )
    if (
        summary.get("unsafe_content_detected") is not False
        or summary.get("privacy_check_passed") is not True
        or summary.get("board_separation_passed") is not True
        or summary.get("validation_failure_count") != 0
    ):
        raise ValueError("ablation ledger failed the closed privacy-safe schema")
    return frozen, summary


def run_v2_delivery(
    fixtures: DeliveryInputsV2,
    output_root: Path,
) -> dict[str, Any]:
    """Run downstream delivery from typed inputs without performing acquisition."""
    if not isinstance(fixtures, DeliveryInputsV2):
        raise TypeError("v2 delivery requires DeliveryInputsV2")
    run_metadata = _validated_metadata(fixtures.run_metadata)
    source_hashes = _validated_source_hashes(fixtures.source_hashes)
    stage_failures = _validated_stage_failures(fixtures.stage_failures)
    if (
        fixtures.live_provenance is not None
        and not fixtures.live_provenance._binding_is_authentic()
    ):
        raise ValueError("live delivery provenance binding changed")
    if fixtures.empirical_v2_results_exist and (
        fixtures.live_provenance.candidate_count != len(fixtures.candidates)
        or run_metadata.get("run_id")
        != fixtures.live_provenance.run_id
        or _candidate_lineage_sha256(fixtures.candidates)
        != fixtures.live_provenance.candidate_lineage_sha256
    ):
        raise ValueError("live delivery inputs do not match acquisition provenance")

    ablation_rows, ablation_summary = _validated_safe_ablation(
        fixtures.ablation_rows
    )
    candidates = fixtures.candidates
    eligibility = tuple(item.reduce() for item in candidates)
    lineage = tuple(item.lineage_record() for item in candidates)
    locations = tuple(item.location for item in candidates)
    google = prepare_v2_google_frame(locations, eligibility, lineage)

    returned_zips = tuple(
        item.location.returned_postal_code
        for item in candidates
        if item.location.returned_postal_code is not None
    )
    boards = prepare_v2_board_frames(
        fixtures.board,
        fixtures.target_zips,
        observed_google_zips=returned_zips,
    )
    linkage = run_v2_board_linkage(
        google, boards, fixtures.linkage_contract
    )
    spatial = run_v2_spatial_analysis(
        google,
        boards,
        fixtures.origins,
        fixtures.spatial_contract,
        analysis_scope="full",
    )
    spatial_summary = _validated_spatial_summary(
        spatial, fixtures.spatial_contract
    )
    eligibility_frame = pd.DataFrame(
        [item.to_public_dict() for item in eligibility]
    ).sort_values("place_id").reset_index(drop=True)
    unresolved = eligibility_frame.loc[
        eligibility_frame["eligibility_state"].eq("unresolved")
    ].copy().reset_index(drop=True)
    artifacts: dict[str, tuple[str, Any]] = {
        "candidate_inventory.csv": (
            "csv",
            _candidate_inventory(candidates, eligibility),
        ),
        "eligibility.csv": ("csv", eligibility_frame),
        "unresolved_cases.csv": ("csv", unresolved),
        "board_links.csv": ("csv", linkage.links),
        "unmatched_google.csv": ("csv", linkage.unmatched_google),
        "unmatched_board.csv": ("csv", linkage.unmatched_board),
        "performance_metrics.json": ("json", linkage.summary),
        "ablation_results.jsonl": ("jsonl", ablation_rows),
        "ablation_summary.json": ("json", ablation_summary),
        "tract_spatial.csv": ("csv", _safe_tract_frame(spatial.tracts)),
        "coordinate_qa.csv": ("csv", spatial.coordinate_qa),
        "coordinate_ledger.csv": (
            "csv", spatial.coordinate_ledger
        ),
        "spatial_summary.json": ("json", spatial_summary),
    }
    contract_manifest = _contract_manifest(
        fixtures.linkage_contract, fixtures.spatial_contract
    )
    failure_counts = {
        **stage_failures,
        "classification_unavailable": sum(
            item.decision is None for item in candidates
        ),
        "ablation_condition_failures": int(
            ablation_summary.get("condition_failure_count", 0)
        ),
        "ablation_validation_failures": int(
            ablation_summary.get("validation_failure_count", 0)
        ),
        "ablation_safety_gate_failures": int(
            ablation_summary.get("privacy_check_passed") is not True
            or ablation_summary.get("board_separation_passed") is not True
            or ablation_summary.get("unsafe_content_detected") is True
        ),
    }
    failure_count = sum(failure_counts.values())
    status_prefix = (
        "full_delivery"
        if fixtures.empirical_v2_results_exist
        else "offline_validation"
    )
    manifest_base = {
        "schema_version": DELIVERY_CONTRACT_VERSION,
        "status": f"{status_prefix}_{'complete' if failure_count == 0 else 'incomplete'}",
        "source_label": fixtures.source_label,
        "empirical_v2_results_exist": fixtures.empirical_v2_results_exist,
        "stage_order": list(FULL_STAGE_ORDER),
        "candidate_count": len(candidates),
        "eligibility_counts": {
            str(key): int(value)
            for key, value in eligibility_frame["eligibility_state"]
            .value_counts()
            .sort_index()
            .items()
        },
        "unresolved_count": len(unresolved),
        "origin_count": len(spatial.tracts),
        "failure_counts": failure_counts,
        "failure_count": failure_count,
        "request_counts": {
            "ablation_A_B_C": int(ablation_summary["request_count"]),
            "ablation_by_condition": dict(
                ablation_summary["condition_request_counts"]
            ),
        },
        "ablation_returned_models": list(
            ablation_summary["returned_models"]
        ),
        "ablation_system_fingerprints": list(
            ablation_summary["system_fingerprints"]
        ),
        "run_metadata": run_metadata,
        "source_hashes": dict(sorted(source_hashes.items())),
        "live_provenance": (
            fixtures.live_provenance.to_dict()
            if fixtures.live_provenance is not None
            else None
        ),
        "contracts": contract_manifest,
        "privacy_contract": {
            "public_name_address_coordinates_retained": False,
            "review_text_retained": False,
            "prompt_retained": False,
            "raw_model_output_retained": False,
        },
        "board_separation": {
            "board_enters_classifier": False,
            "board_enters_eligibility_reducer": False,
            "board_first_stage": "board_preparation",
        },
    }
    return _write_atomic_package(output_root, artifacts, manifest_base)


def run_offline_v2_delivery(
    fixtures: DeliveryInputsV2,
    output_root: Path,
) -> dict[str, Any]:
    """Run a package that is explicitly labeled synthetic/non-empirical."""
    if not isinstance(fixtures, DeliveryInputsV2):
        raise TypeError("offline delivery requires DeliveryInputsV2")
    if fixtures.empirical_v2_results_exist:
        raise ValueError("offline delivery cannot be labeled as empirical v2 results")
    return run_v2_delivery(fixtures, output_root)


def run_pilot_v2_delivery(
    inputs: PilotDeliveryInputsV2,
    output_root: Path,
) -> dict[str, Any]:
    """Write one bounded A/B/C pilot package; Board and spatial inputs are absent."""
    if not isinstance(inputs, PilotDeliveryInputsV2):
        raise TypeError("pilot delivery requires PilotDeliveryInputsV2")
    selection = _validated_pilot_selection(inputs.selection_ledger)
    run_metadata = _validated_metadata(inputs.run_metadata)
    ablation_rows, recomputed_summary = _validated_safe_ablation(
        inputs.ablation_rows
    )
    if recomputed_summary != inputs.ablation_summary:
        raise ValueError("pilot ablation summary does not match its safe ledger")
    selection_payload = {
        "candidate_count": len(selection),
        "selection_ledger": [dict(item) for item in selection],
        "unavailable_roles": list(inputs.unavailable_roles),
    }
    artifacts = {
        "pilot_selection.json": ("json", selection_payload),
        "ablation_results.jsonl": ("jsonl", ablation_rows),
        "ablation_summary.json": ("json", inputs.ablation_summary),
    }
    classification_failures = int(
        inputs.ablation_summary.get("condition_failure_count", 0)
    )
    validation_failures = int(
        inputs.ablation_summary.get("validation_failure_count", 0)
    )
    safety_failure = int(
        inputs.ablation_summary.get("privacy_check_passed") is not True
        or inputs.ablation_summary.get("board_separation_passed") is not True
        or inputs.ablation_summary.get("unsafe_content_detected") is True
    )
    clean = (
        inputs.ablation_summary.get("pilot_status") == "clean"
        and inputs.ablation_summary.get("full_run_recommendation") == "proceed"
        and classification_failures == 0
        and validation_failures == 0
        and safety_failure == 0
    )
    manifest_base = {
        "schema_version": DELIVERY_CONTRACT_VERSION,
        "status": (
            "pilot_complete" if clean else "pilot_incomplete"
        ),
        "source_label": "pilot_ablation",
        "empirical_v2_results_exist": inputs.empirical_v2_results_exist,
        "stage_order": list(PILOT_STAGE_ORDER),
        "candidate_count": len(selection),
        "ablation_row_count": len(inputs.ablation_rows),
        "failure_counts": {
            "ablation_condition_failures": classification_failures,
            "ablation_validation_failures": validation_failures,
            "pilot_safety_gate_failures": safety_failure,
        },
        "failure_count": (
            classification_failures + validation_failures + safety_failure
        ),
        "request_counts": {
            "ablation_A_B_C": int(inputs.ablation_summary["request_count"]),
            "ablation_by_condition": dict(
                inputs.ablation_summary["condition_request_counts"]
            ),
        },
        "ablation_returned_models": list(
            inputs.ablation_summary["returned_models"]
        ),
        "ablation_system_fingerprints": list(
            inputs.ablation_summary["system_fingerprints"]
        ),
        "run_metadata": run_metadata,
        "contracts": _contract_manifest(),
        "privacy_contract": {
            "public_name_address_coordinates_retained": False,
            "review_text_retained": False,
            "prompt_retained": False,
            "raw_model_output_retained": False,
        },
        "board_separation": {
            "board_inputs_present": False,
            "board_enters_classifier": False,
            "board_enters_eligibility_reducer": False,
        },
        "spatial_outputs_created": False,
    }
    return _write_atomic_package(output_root, artifacts, manifest_base)


__all__ = [
    "DELIVERY_CONTRACT_SHA256",
    "DELIVERY_CONTRACT_VERSION",
    "LocalDeliveryContextV2",
    "LiveDeliveryProvenanceV2",
    "authenticate_live_delivery_provenance",
    "authenticate_pilot_ablation_package",
    "DeliveryCandidateV2",
    "DeliveryInputsV2",
    "FULL_STAGE_ORDER",
    "PILOT_STAGE_ORDER",
    "PilotDeliveryInputsV2",
    "run_offline_v2_delivery",
    "run_pilot_v2_delivery",
    "run_v2_delivery",
    "load_local_delivery_context",
]
