"""CLI for the fixed-frame, single-model NETS-AI forced-choice workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from forced_choice_classifier import (  # noqa: E402
    MODEL_ID,
    PROMPT_TEMPLATE_SHA256,
    PROMPT_VERSION,
    build_forced_choice_messages,
    prompt_sha256,
)
from forced_choice_delivery import (  # noqa: E402
    ForcedChoiceDeliveryCandidate,
    load_forced_choice_local_context,
    run_forced_choice_delivery,
)
from forced_choice_eligibility import ELIGIBILITY_CONTRACT_SHA256  # noqa: E402
from forced_choice_models import (  # noqa: E402
    FORCED_CHOICE_CONTRACT_SHA256,
    ForcedChoiceDecision,
)
from forced_choice_pilot import (  # noqa: E402
    _classify_with_technical_retries,
    run_forced_choice_ablation,
    write_forced_choice_pilot_package,
)
from run_v2_workflow import (  # noqa: E402
    _LazyGoogleProvider,
    _WaybackArchiveTransport,
)
from skills.wayback_agent import (  # noqa: E402
    WaybackServiceUnavailable,
)
from wayback_runtime import (  # noqa: E402
    build_wayback_collector, deferred_report, require_research_wayback_service,
)
from wayback_state import WaybackDeferred  # noqa: E402
from v2_classifier import build_candidate_facts, build_review_evidence  # noqa: E402
from v2_downstream_models import candidate_location_from_details  # noqa: E402
from v2_evidence_models import (  # noqa: E402
    ClassifierContractV2,
    ClassifierInputV2,
    QueryProvenanceV2,
    WaybackCollectionV2,
)


FIXED_RUN_ID = "20260831T230500Z"
FIXED_ROOT = ROOT / "data" / "v2" / FIXED_RUN_ID
CANONICAL_POINTER = ROOT / "data" / "v2" / "canonical_thesis_empirical_run.json"
PILOT_SELECTION = (
    ROOT / "data" / "v2" / "20260831T225700Z"
    / "pilot_delivery" / "pilot_selection.json"
)
WORKFLOW_SCHEMA = "nets-ai-v2.2-forced-choice-workflow-v1"
COMPLETED_UNIT_SCHEMA = "nets-ai-v2.2-forced-choice-completed-candidate-v2"
_RUN_ID = re.compile(r"^\d{8}T\d{6}Z$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EVIDENCE_ID = re.compile(
    r"^(?:candidate|review|archive_source_v2|opening_claim_v2)_[0-9a-f]{64}$"
)
_IMPLEMENTATION_PATHS = (
    "code/forced_choice_models.py",
    "code/forced_choice_classifier.py",
    "code/forced_choice_eligibility.py",
    "code/forced_choice_pilot.py",
    "code/forced_choice_delivery.py",
    "code/run_forced_choice_workflow.py",
    "code/run_v2_workflow.py",
    "code/wayback_runtime.py",
    "code/wayback_state.py",
    "code/outscraper_wayback.py",
    "code/v2_classifier.py",
    "code/v2_evidence_models.py",
    "code/v2_downstream_models.py",
    "code/v2_board_adapter.py",
    "code/v2_spatial_adapter.py",
    "code/study_design.py",
    "skills/google_maps.py",
    "skills/wayback_agent.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _implementation_hashes() -> dict[str, str]:
    result = {}
    for relative in _IMPLEMENTATION_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"forced-choice implementation file is missing: {relative}")
        result[relative] = _sha256_file(path)
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if type(value) is not dict:
            raise ValueError("fixed-frame JSONL contains a non-object")
        rows.append(value)
    return rows


def authenticate_fixed_frame() -> dict[str, Any]:
    pointer = _read_json(CANONICAL_POINTER)
    if (
        type(pointer) is not dict
        or pointer.get("canonical_run_id") != FIXED_RUN_ID
        or pointer.get("candidate_count") != 1226
        or pointer.get("completed_candidate_count") != 1226
    ):
        raise ValueError("canonical pointer does not authenticate the fixed frame")
    places_path = FIXED_ROOT / "place_ids.jsonl"
    occurrences_path = FIXED_ROOT / "candidate_occurrences.jsonl"
    provenance_path = FIXED_ROOT / "search_provenance.json"
    places = _read_jsonl(places_path)
    occurrences = _read_jsonl(occurrences_path)
    provenance = _read_json(provenance_path)
    if (
        len(places) != 1226
        or len({row.get("place_id") for row in places}) != 1226
        or len({row.get("source_zip") for row in places}) != 60
        or provenance.get("status") != "complete"
    ):
        raise ValueError("fixed candidate frame is structurally invalid")
    expected = pointer["run_provenance"]
    manifest = _read_json(FIXED_ROOT / "run_manifest.json")
    if (
        manifest["files"]["place_ids.jsonl"] != _sha256_file(places_path)
        or manifest["files"]["candidate_occurrences.jsonl"]
        != _sha256_file(occurrences_path)
        or manifest["files"]["search_provenance.json"]
        != _sha256_file(provenance_path)
        or expected["run_manifest_sha256"]
        != _sha256_file(FIXED_ROOT / "run_manifest.json")
    ):
        raise ValueError("fixed candidate frame hashes do not match canonical provenance")
    payload = {
        "canonical_source_run_id": FIXED_RUN_ID,
        "candidate_count": len(places),
        "occurrence_count": len(occurrences),
        "zip_count": len({row["source_zip"] for row in places}),
        "place_ids_sha256": _sha256_file(places_path),
        "candidate_occurrences_sha256": _sha256_file(occurrences_path),
        "search_provenance_sha256": _sha256_file(provenance_path),
        "canonical_pointer_sha256": _sha256_file(CANONICAL_POINTER),
    }
    return {
        **payload,
        "fixed_frame_sha256": _canonical_sha256(payload),
        "places": places,
        "occurrences": occurrences,
        "search_provenance": provenance,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="",
    )
    os.replace(temporary, path)


def _location_record(location) -> dict[str, Any]:
    return {
        "place_id": location.place_id,
        "candidate_id": location.candidate_id,
        "name": location.name,
        "formatted_address": location.formatted_address,
        "returned_postal_code": location.returned_postal_code,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "retrieved_at": location.retrieved_at.isoformat().replace("+00:00", "Z"),
        "query_provenance": {
            "query_definition_id": location.query_provenance.query_definition_id,
            "source_zip": location.query_provenance.source_zip,
            "retrieval_channel": location.query_provenance.retrieval_channel,
        },
    }


def _parse_utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("checkpoint timestamp is invalid")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None:
        raise ValueError("checkpoint timestamp lacks timezone")
    return parsed.astimezone(timezone.utc)


def _location_from_record(value: Any):
    from v2_downstream_models import CandidateLocationV2

    expected = {
        "place_id", "candidate_id", "name", "formatted_address",
        "returned_postal_code", "latitude", "longitude", "retrieved_at",
        "query_provenance",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("completed candidate location is invalid")
    provenance = value["query_provenance"]
    if type(provenance) is not dict or set(provenance) != {
        "query_definition_id", "source_zip", "retrieval_channel"
    }:
        raise ValueError("completed candidate provenance is invalid")
    return CandidateLocationV2(
        value["place_id"],
        value["candidate_id"],
        value["name"],
        value["formatted_address"],
        value["returned_postal_code"],
        value["latitude"],
        value["longitude"],
        _parse_utc(value["retrieved_at"]),
        QueryProvenanceV2(
            provenance["query_definition_id"],
            provenance["source_zip"],
            provenance["retrieval_channel"],
        ),
    )


def _completed_unit_path(root: Path, place_id: str) -> Path:
    return root / f"{hashlib.sha256(place_id.encode('utf-8')).hexdigest()}.json"


def _archive_checkpoint(place_id: str, wayback: WaybackCollectionV2) -> dict[str, Any]:
    """Retain derived archive provenance without archived page text or URLs."""
    archive = wayback.archive_evidence
    return {
        "place_id": place_id,
        "archive_evidence": {
            "query_status": archive.query_status,
            "page_status": archive.page_status,
            "cap_status": archive.cap_status,
            "query_scope": archive.query_scope,
            "retrieved_at": archive.retrieved_at.isoformat().replace("+00:00", "Z"),
            "max_cdx_pages": archive.max_cdx_pages,
            "max_cdx_records": archive.max_cdx_records,
            "cdx_response_chain_sha256": archive.cdx_response_chain_sha256,
            "cdx_pages": [item.to_public_dict() for item in archive.cdx_pages],
            "fetch_records": [item.to_public_dict() for item in archive.fetch_records],
            "source_snapshots": [
                item.to_public_dict() for item in archive.source_snapshots
            ],
            "opening_claims": [
                item.to_public_dict() for item in archive.opening_claims
            ],
            "archive_match_assessments": [
                {
                    "source_evidence_id": item.source_evidence_id,
                    "evidence_scope": item.evidence_scope,
                }
                for item in wayback.archive_match_assessments
            ],
            "archive_claim_assessments": [
                {
                    "claim_evidence_id": item.claim_evidence_id,
                    "evidence_scope": item.evidence_scope,
                }
                for item in wayback.archive_claim_assessments
            ],
        },
    }


def _load_promotion(path: Path, frame_sha256: str) -> dict[str, Any]:
    promotion = _read_json(path)
    if (
        type(promotion) is not dict
        or promotion.get("schema_version")
        != "nets-ai-v2.2-forced-choice-promotion-v1"
        or promotion.get("model") != MODEL_ID
        or promotion.get("fixed_frame_sha256") != frame_sha256
        or promotion.get("promotion_basis")
        != "24_of_24_structurally_valid_forced_choice_outputs"
    ):
        raise ValueError("forced-choice pilot promotion is invalid")
    relative = Path(promotion.get("pilot_manifest_file", ""))
    pilot_manifest_path = (path.resolve().parent / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not pilot_manifest_path.is_relative_to((ROOT / "data" / "v2").resolve())
        or _sha256_file(pilot_manifest_path) != promotion.get("pilot_manifest_sha256")
    ):
        raise ValueError("promoted pilot manifest failed authentication")
    pilot = _read_json(pilot_manifest_path)
    if (
        pilot.get("status") != "pilot_complete"
        or pilot.get("candidate_count") != 8
        or pilot.get("output_count") != 24
        or pilot.get("model") != MODEL_ID
        or pilot.get("fixed_candidate_frame_sha256") != frame_sha256
        or pilot.get("prompt_template_version") != PROMPT_VERSION
        or pilot.get("prompt_template_sha256") != PROMPT_TEMPLATE_SHA256
        or pilot.get("classifier_contract_sha256")
        != FORCED_CHOICE_CONTRACT_SHA256
        or pilot.get("eligibility_contract_sha256")
        != ELIGIBILITY_CONTRACT_SHA256
        or pilot.get("semantic_manifest_sha256")
        != promotion.get("pilot_semantic_manifest_sha256")
    ):
        raise ValueError("promoted pilot does not match the frozen forced-choice contract")
    pilot_root = pilot_manifest_path.parent
    rows_path = pilot_root / "ablation_results.jsonl"
    summary_path = pilot_root / "ablation_summary.json"
    if (
        _sha256_file(rows_path) != pilot["artifacts"]["ablation_results.jsonl"]
        or _sha256_file(summary_path) != pilot["artifacts"]["ablation_summary.json"]
    ):
        raise ValueError("promoted pilot artifacts failed authentication")
    rows = _read_jsonl(rows_path)
    summary = _read_json(summary_path)
    if (
        len(rows) != 24
        or summary.get("promotion_passed") is not True
        or summary.get("successful_output_count") != 24
        or summary.get("validation_pass_count") != 24
        or summary.get("candidate_binding_pass_count") != 24
        or summary.get("technical_failure_count") != 0
        or summary.get("board_separation_passed") is not True
    ):
        raise ValueError("promoted pilot gate is not clean")
    return {
        "promotion": promotion,
        "promotion_sha256": _sha256_file(path),
        "pilot_manifest": pilot,
        "pilot_manifest_path": pilot_manifest_path,
        "ablation_rows": rows,
        "ablation_summary": summary,
    }


def _build_completed_unit(
    *,
    location: Any,
    packet: ClassifierInputV2,
    decision: ForcedChoiceDecision,
    metadata: dict[str, Any],
    archive_record: dict[str, Any],
    fixed_frame_sha256: str,
    scientific_implementation_sha256: str,
) -> dict[str, Any]:
    location_value = _location_record(location)
    decision_value = decision.to_public_dict()
    if metadata.get("decision_sha256") != decision.decision_sha256:
        raise ValueError("provider metadata decision hash changed")
    archive = packet.wayback_collection.archive_evidence
    evidence_ledger = {
        "candidate_evidence_id": packet.candidate.candidate_id,
        "review_status": packet.review_evidence.status,
        "review_returned_count": packet.review_evidence.returned_count,
        "review_evidence_ids": [
            item.evidence_id for item in packet.review_evidence.snippets
        ],
        "archive_source_evidence_ids": [
            item.evidence_id for item in archive.source_snapshots
        ],
        "opening_claim_evidence_ids": [
            item.evidence_id for item in archive.opening_claims
        ],
    }
    input_prompt_sha256 = prompt_sha256(
        build_forced_choice_messages(packet, "C_metadata_reviews_wayback")
    )
    record = {
        "schema_version": COMPLETED_UNIT_SCHEMA,
        "place_id": location.place_id,
        "candidate_id": location.candidate_id,
        "model": MODEL_ID,
        "returned_model": metadata.get("returned_model"),
        "prompt_template_version": PROMPT_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
        "fixed_frame_sha256": fixed_frame_sha256,
        "scientific_implementation_sha256": scientific_implementation_sha256,
        "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
        "classification_decision_sha256": decision.decision_sha256,
        "location": location_value,
        "decision": decision_value,
        "provider_metadata": metadata,
        "archive_record": archive_record,
        "input_evidence_ledger": evidence_ledger,
        "input_evidence_ledger_sha256": _canonical_sha256(evidence_ledger),
        "input_prompt_sha256": input_prompt_sha256,
        "location_record_sha256": _canonical_sha256(location_value),
        "decision_record_sha256": _canonical_sha256(decision_value),
        "archive_record_sha256": _canonical_sha256(archive_record),
    }
    return record


def _validate_completed_unit(
    value: Any,
    *,
    fixed_frame_sha256: str,
    scientific_implementation_sha256: str,
) -> tuple[ForcedChoiceDeliveryCandidate, dict[str, Any], dict[str, Any]]:
    required = {
        "schema_version", "place_id", "candidate_id", "model", "returned_model",
        "prompt_template_version", "prompt_template_sha256",
        "classifier_contract_sha256", "eligibility_contract_sha256",
        "fixed_frame_sha256", "scientific_implementation_sha256",
        "candidate_snapshot_sha256", "classification_decision_sha256",
        "location", "decision", "provider_metadata", "archive_record",
        "input_evidence_ledger", "input_evidence_ledger_sha256",
        "input_prompt_sha256",
        "location_record_sha256", "decision_record_sha256", "archive_record_sha256",
    }
    if type(value) is not dict or set(value) != required:
        raise ValueError("completed candidate checkpoint fields are invalid")
    if (
        value["schema_version"] != COMPLETED_UNIT_SCHEMA
        or value["model"] != MODEL_ID
        or value["prompt_template_version"] != PROMPT_VERSION
        or value["prompt_template_sha256"] != PROMPT_TEMPLATE_SHA256
        or value["classifier_contract_sha256"] != FORCED_CHOICE_CONTRACT_SHA256
        or value["eligibility_contract_sha256"] != ELIGIBILITY_CONTRACT_SHA256
        or value["fixed_frame_sha256"] != fixed_frame_sha256
        or value["scientific_implementation_sha256"]
        != scientific_implementation_sha256
        or value["location_record_sha256"] != _canonical_sha256(value["location"])
        or value["decision_record_sha256"] != _canonical_sha256(value["decision"])
        or value["archive_record_sha256"]
        != _canonical_sha256(value["archive_record"])
        or value["input_evidence_ledger_sha256"]
        != _canonical_sha256(value["input_evidence_ledger"])
        or not isinstance(value["input_prompt_sha256"], str)
        or _SHA256.fullmatch(value["input_prompt_sha256"]) is None
    ):
        raise ValueError("completed candidate checkpoint authentication failed")
    location = _location_from_record(value["location"])
    decision = ForcedChoiceDecision.from_public_dict(value["decision"])
    provider_metadata = value["provider_metadata"]
    archive_record = value["archive_record"]
    evidence_ledger = value["input_evidence_ledger"]
    expected_ledger_keys = {
        "candidate_evidence_id", "review_status", "review_returned_count",
        "review_evidence_ids",
        "archive_source_evidence_ids", "opening_claim_evidence_ids",
    }
    archive_evidence = (
        archive_record.get("archive_evidence")
        if type(archive_record) is dict else None
    )
    cited = {
        item
        for values in decision.claim_evidence_ids.values()
        for item in values
    }
    known = set()
    if type(evidence_ledger) is dict and set(evidence_ledger) == expected_ledger_keys:
        known = {
            evidence_ledger.get("candidate_evidence_id"),
            *evidence_ledger.get("review_evidence_ids", []),
            *evidence_ledger.get("archive_source_evidence_ids", []),
            *evidence_ledger.get("opening_claim_evidence_ids", []),
        }
    if (
        location.place_id != value["place_id"]
        or location.candidate_id != value["candidate_id"]
        or decision.candidate_id != value["candidate_id"]
        or location.candidate_snapshot_sha256 != value["candidate_snapshot_sha256"]
        or decision.decision_sha256 != value["classification_decision_sha256"]
        or type(provider_metadata) is not dict
        or provider_metadata.get("decision_sha256")
        != decision.decision_sha256
        or provider_metadata.get("requested_model") != MODEL_ID
        or provider_metadata.get("prompt_template_sha256")
        != PROMPT_TEMPLATE_SHA256
        or type(archive_record) is not dict
        or archive_record.get("place_id") != location.place_id
        or set(archive_record) != {"place_id", "archive_evidence"}
        or type(evidence_ledger) is not dict
        or set(evidence_ledger) != expected_ledger_keys
        or evidence_ledger.get("candidate_evidence_id") != location.candidate_id
        or evidence_ledger.get("review_status") not in {
            "success", "missing", "timeout", "http_429", "http_4xx",
            "http_5xx", "decode_error", "other_error", "not_requested",
        }
        or type(evidence_ledger.get("review_returned_count")) is not int
        or evidence_ledger["review_returned_count"] < 0
        or evidence_ledger["review_returned_count"] > 5
        or evidence_ledger["review_returned_count"]
        != len(evidence_ledger.get("review_evidence_ids", []))
        or any(
            type(evidence_ledger.get(key)) is not list
            or len(evidence_ledger[key]) != len(set(evidence_ledger[key]))
            or any(
                not isinstance(item, str)
                or _SAFE_EVIDENCE_ID.fullmatch(item) is None
                for item in evidence_ledger[key]
            )
            for key in (
                "review_evidence_ids", "archive_source_evidence_ids",
                "opening_claim_evidence_ids",
            )
        )
        or not cited.issubset(known)
        or type(archive_evidence) is not dict
        or evidence_ledger["archive_source_evidence_ids"]
        != [item.get("evidence_id") for item in archive_evidence.get("source_snapshots", [])]
        or evidence_ledger["opening_claim_evidence_ids"]
        != [item.get("evidence_id") for item in archive_evidence.get("opening_claims", [])]
    ):
        raise ValueError("completed candidate checkpoint binding failed")
    source_scope = {
        item.get("source_evidence_id"): item.get("evidence_scope")
        for item in archive_evidence.get("archive_match_assessments", [])
        if type(item) is dict
    }
    claim_scope = {
        item.get("claim_evidence_id"): item.get("evidence_scope")
        for item in archive_evidence.get("archive_claim_assessments", [])
        if type(item) is dict
    }
    sources = {
        item.get("evidence_id"): item
        for item in archive_evidence.get("source_snapshots", [])
        if type(item) is dict
    }
    claims = {
        item.get("evidence_id"): item
        for item in archive_evidence.get("opening_claims", [])
        if type(item) is dict
    }
    outlet_sources = {
        item_id for item_id, scope in source_scope.items()
        if scope == "outlet_specific"
    }
    outlet_claims = {
        item_id for item_id, scope in claim_scope.items()
        if scope == "outlet_specific"
    }
    year_citations = set(decision.claim_evidence_ids["historical_year"])
    if decision.historical_year_type == "active_by_upper_bound":
        try:
            earliest = min(
                _parse_utc(sources[item_id]["captured_at"]).year
                for item_id in outlet_sources
            )
        except (KeyError, ValueError) as exc:
            raise ValueError("completed active-by evidence is invalid") from exc
        eligible = {
            item_id for item_id in outlet_sources
            if _parse_utc(sources[item_id]["captured_at"]).year == earliest
        }
        if (
            decision.historical_year_value != earliest
            or not (year_citations & eligible)
        ):
            raise ValueError("completed active-by evidence binding failed")
    elif decision.historical_year_type == "explicit_archived_opening_claim":
        outlet_years = {claims[item_id].get("year") for item_id in outlet_claims}
        eligible = {
            item_id for item_id in outlet_claims
            if claims[item_id].get("year") == decision.historical_year_value
        }
        if (
            outlet_years != {decision.historical_year_value}
            or not (year_citations & eligible)
        ):
            raise ValueError("completed opening-year evidence binding failed")
    return (
        ForcedChoiceDeliveryCandidate(location, decision),
        {
            "place_id": location.place_id,
            "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
            "classification_decision_sha256": decision.decision_sha256,
            **decision.to_public_dict(),
            "provider_metadata": provider_metadata,
        },
        archive_record,
    )


def _authenticate_completed_delivery(root: Path) -> dict[str, Any]:
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing delivery lacks a manifest")
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("status") != "full_delivery_complete"
        or manifest.get("candidate_count") != 1226
        or manifest.get("origin_count") != 327
        or manifest.get("runtime_failure_count") != 0
        or type(artifacts) is not dict
        or not artifacts
        or any(
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            for relative, digest in artifacts.items()
        )
    ):
        raise ValueError("existing delivery is not a complete forced-choice package")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != set(artifacts) | {"run_manifest.json"}:
        raise ValueError("existing delivery contains undeclared or missing files")
    if any(_sha256_file(root / relative) != digest for relative, digest in artifacts.items()):
        raise ValueError("existing delivery artifact authentication failed")
    semantic = {
        key: item
        for key, item in manifest.items()
        if key not in {"run_metadata", "semantic_manifest_sha256"}
    }
    if (
        not isinstance(manifest.get("semantic_manifest_sha256"), str)
        or _SHA256.fullmatch(manifest["semantic_manifest_sha256"]) is None
        or _canonical_sha256(semantic) != manifest["semantic_manifest_sha256"]
    ):
        raise ValueError("existing delivery semantic manifest authentication failed")
    return manifest


def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI credential is unavailable")
    from openai import OpenAI
    return OpenAI(api_key=key, timeout=180.0)


def run_pilot(run_id: str, output_root: Path, *, resume: bool = False) -> dict[str, Any]:
    frame = authenticate_fixed_frame()
    selection = _read_json(PILOT_SELECTION)
    ledger = selection.get("selection_ledger")
    if type(ledger) is not list or len(ledger) != 8:
        raise ValueError("preserved pilot selection is invalid")
    places = {row["place_id"]: row for row in frame["places"]}
    if any(row.get("place_id") not in places for row in ledger):
        raise ValueError("pilot selection is outside the fixed candidate frame")
    run_root = (output_root / run_id).resolve()
    if run_root.parent != output_root.resolve():
        raise ValueError("pilot run root escaped the output directory")
    implementation_files = _implementation_hashes()
    manifest_path = run_root / "run_manifest.json"
    previous = None
    if resume:
        previous = _read_json(manifest_path)
        if (
            previous.get("schema_version") != WORKFLOW_SCHEMA
            or previous.get("run_id") != run_id
            or previous.get("run_mode") != "pilot"
            or previous.get("model") != MODEL_ID
            or previous.get("fixed_frame_sha256") != frame["fixed_frame_sha256"]
            or previous.get("implementation_files") != implementation_files
            or previous.get("source_pilot_selection_sha256") != _sha256_file(PILOT_SELECTION)
            or previous.get("status") not in {"pilot_paused_wayback", "pilot_incomplete"}
        ):
            raise ValueError("pilot resume differs from the authenticated paused run")
    elif run_root.exists():
        raise FileExistsError("forced-choice run ID already exists")
    state_dir = output_root / "private" / "wayback_state"
    require_research_wayback_service(
        state_dir=state_dir, direct_transport=_WaybackArchiveTransport(),
        envfile=ROOT / ".env",
    )
    run_root.mkdir(parents=True, exist_ok=resume)
    private = run_root / "private"
    private.mkdir(exist_ok=resume)
    cache = private / "wayback_cache"
    started = datetime.now(timezone.utc)
    if previous is not None:
        history_path = private / "failure_history.jsonl"
        history = _read_jsonl(history_path) if history_path.exists() else []
        prior = run_root / "failures.jsonl"
        history.extend({"resume_attempt": previous.get("resume_attempt", 0), "record": row}
                       for row in (_read_jsonl(prior) if prior.exists() else []))
        _atomic_jsonl(history_path, history)
    running = {
        "schema_version": WORKFLOW_SCHEMA,
        "status": "pilot_running",
        "run_id": run_id,
        "run_mode": "pilot",
        "model": MODEL_ID,
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "fixed_frame_sha256": frame["fixed_frame_sha256"],
        "fixed_candidate_count": 1226,
        "pilot_candidate_count": 8,
        "text_search_requests": 0,
        "canonical_gpt41_run_untouched": True,
        "implementation_files": implementation_files,
        "source_pilot_selection_sha256": _sha256_file(PILOT_SELECTION),
        "resume_supported": True,
        "resume_attempt": int(previous.get("resume_attempt", 0)) + 1 if previous else 0,
    }
    _atomic_json(manifest_path, running)
    google = _LazyGoogleProvider()
    collector = build_wayback_collector(
        cache, state_dir=state_dir, direct_transport=_WaybackArchiveTransport(),
        envfile=ROOT / ".env",
    )
    packets: list[ClassifierInputV2] = []
    archives: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    wayback_pause = None
    for row in sorted(ledger, key=lambda item: item["selection_rank"]):
        place_id = row["place_id"]
        source = places[place_id]
        stage = "places_details"
        try:
            details = google.get_place_details(place_id)
            retrieved = datetime.now(timezone.utc)
            provenance = QueryProvenanceV2(
                source["query_definition_id"], source["source_zip"], "places_search"
            )
            location = candidate_location_from_details(details, provenance, retrieved)
            candidate = build_candidate_facts(details, provenance, retrieved)
            reviews = build_review_evidence(details, retrieved)
            if (
                location.place_id != place_id
                or candidate.source_place_id != place_id
                or row["candidate_id"] != candidate.candidate_id
            ):
                raise ValueError("pilot Details response changed candidate binding")
            stage = "wayback"
            wayback = collector.collect(candidate)
            if not isinstance(wayback, WaybackCollectionV2):
                raise ValueError("Wayback collector returned an invalid packet")
            packets.append(
                ClassifierInputV2(
                    candidate,
                    reviews,
                    wayback,
                    ClassifierContractV2.from_project_constants(),
                )
            )
            archives.append(
                {
                    "place_id": place_id,
                    "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
                    "archive_evidence": wayback.archive_evidence.to_private_dict(),
                }
            )
        except WaybackDeferred as exc:
            wayback_pause = {"reason": exc.reason, "retry_at": exc.retry_at}
            failures.append({
                "place_id": place_id, "stage": "wayback",
                "failure_type": type(exc).__name__, "retryable": True,
                **wayback_pause,
            })
            break
        except Exception as exc:
            failures.append(
                {
                    "place_id": place_id,
                    "stage": stage,
                    "failure_type": type(exc).__name__,
                    "retryable": True,
                }
            )
    _atomic_jsonl(private / "archive_evidence.jsonl", archives)
    _atomic_jsonl(run_root / "failures.jsonl", failures)
    if failures or len(packets) != 8:
        manifest = {
            **running,
            "status": "pilot_paused_wayback" if wayback_pause else "pilot_incomplete",
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "staged_candidate_count": len(packets),
            "failure_count": len(failures),
            **({"wayback_retry": wayback_pause} if wayback_pause else {}),
        }
        _atomic_json(manifest_path, manifest)
        return manifest

    rows, summary = run_forced_choice_ablation(
        packets, _openai_client(), model=MODEL_ID
    )
    package = write_forced_choice_pilot_package(
        run_root / "pilot_delivery",
        rows=rows,
        summary=summary,
        selection_ledger=ledger,
        run_metadata={
            "run_id": run_id,
            "model": MODEL_ID,
            "executed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_pilot_selection_sha256": _sha256_file(PILOT_SELECTION),
        },
        fixed_frame_sha256=frame["fixed_frame_sha256"],
    )
    final = {
        **running,
        "status": package["status"],
        "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "staged_candidate_count": 8,
        "output_count": package["output_count"],
        "failure_count": 0 if package["status"] == "pilot_complete" else 1,
        "returned_models": package["returned_models"],
        "pilot_manifest_sha256": _sha256_file(
            run_root / "pilot_delivery" / "run_manifest.json"
        ),
        "pilot_semantic_manifest_sha256": package["semantic_manifest_sha256"],
    }
    _atomic_json(manifest_path, final)
    if package["status"] == "pilot_complete":
        promotion = {
            "schema_version": "nets-ai-v2.2-forced-choice-promotion-v1",
            "pilot_run_id": run_id,
            "pilot_manifest_file": (
                f"{run_id}/pilot_delivery/run_manifest.json"
            ),
            "pilot_manifest_sha256": final["pilot_manifest_sha256"],
            "pilot_semantic_manifest_sha256": package["semantic_manifest_sha256"],
            "model": MODEL_ID,
            "returned_models": package["returned_models"],
            "fixed_frame_sha256": frame["fixed_frame_sha256"],
            "reviewed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "promotion_basis": "24_of_24_structurally_valid_forced_choice_outputs",
        }
        promotion_path = output_root / f"forced_choice_promotion_{run_id}.json"
        _atomic_json(promotion_path, promotion)
        final["promotion_file"] = promotion_path.relative_to(ROOT).as_posix()
        final["promotion_file_sha256"] = _sha256_file(promotion_path)
        _atomic_json(manifest_path, final)
    return final


def _checkpoint_hashes(run_root: Path, paths: list[Path]) -> dict[str, str]:
    hashes = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(run_root.resolve()) or not resolved.is_file():
            raise ValueError("checkpoint path escaped the run root")
        hashes[resolved.relative_to(run_root.resolve()).as_posix()] = _sha256_file(
            resolved
        )
    return dict(sorted(hashes.items()))


def run_full(
    run_id: str,
    output_root: Path,
    promotion_file: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    frame = authenticate_fixed_frame()
    promotion = _load_promotion(promotion_file, frame["fixed_frame_sha256"])
    allowed_returned_models = tuple(promotion["pilot_manifest"]["returned_models"])
    if allowed_returned_models != ("gpt-5-mini-2025-08-07",):
        raise ValueError("pilot returned-model version is not the promoted version")
    implementation_files = _implementation_hashes()
    scientific_implementation_sha256 = _canonical_sha256(implementation_files)
    authorization_sha256 = _sha256_file(ROOT / "AGENTS.md")
    configuration = {
        "schema_version": WORKFLOW_SCHEMA,
        "run_id": run_id,
        "run_mode": "full",
        "model": MODEL_ID,
        "fixed_frame_sha256": frame["fixed_frame_sha256"],
        "promotion_file_sha256": promotion["promotion_sha256"],
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
    }
    configuration_sha256 = _canonical_sha256(configuration)
    run_root = (output_root / run_id).resolve()
    if run_root.parent != output_root.resolve():
        raise ValueError("full run root escaped the output directory")
    manifest_path = run_root / "run_manifest.json"
    private = run_root / "private"
    completed_root = private / "completed_candidates"
    failures_path = run_root / "failures.jsonl"
    history_path = private / "failure_history.jsonl"
    classifications_path = run_root / "classification_forced_choice.jsonl"
    archives_path = private / "archive_evidence.jsonl"
    places_path = run_root / "place_ids.jsonl"
    occurrences_path = run_root / "candidate_occurrences.jsonl"
    provenance_path = run_root / "search_provenance.json"
    cache_path = private / "wayback_cache"
    state_dir = output_root / "private" / "wayback_state"

    if resume:
        if not manifest_path.is_file():
            raise ValueError("resume requires an existing authenticated manifest")
        existing = _read_json(manifest_path)
        if (
            existing.get("schema_version") != WORKFLOW_SCHEMA
            or existing.get("run_id") != run_id
            or existing.get("run_mode") != "full"
            or existing.get("model") != MODEL_ID
            or existing.get("configuration_sha256") != configuration_sha256
            or existing.get("implementation_files") != implementation_files
            or existing.get("scientific_implementation_sha256")
            != scientific_implementation_sha256
            or existing.get("fixed_frame_sha256") != frame["fixed_frame_sha256"]
            or existing.get("promotion_file_sha256")
            != promotion["promotion_sha256"]
        ):
            raise ValueError("resume manifest is incompatible with the frozen run")
        if existing.get("status") == "full_complete" and (
            existing.get("delivery_status") == "full_delivery_complete"
        ):
            return existing
        declared_unit_hashes = existing.get("completed_unit_hashes")
        if type(declared_unit_hashes) is not dict or any(
            not isinstance(name, str)
            or not name.endswith(".json")
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            for name, digest in declared_unit_hashes.items()
        ):
            raise ValueError("resume manifest lacks valid completed-unit hashes")
        resume_attempt = int(existing.get("resume_attempt", 0)) + 1
    else:
        if run_root.exists():
            raise FileExistsError("forced-choice full run ID already exists")
        require_research_wayback_service(
            state_dir=state_dir, direct_transport=_WaybackArchiveTransport(),
            envfile=ROOT / ".env",
        )
        run_root.mkdir(parents=True)
        private.mkdir()
        completed_root.mkdir()
        shutil.copyfile(FIXED_ROOT / "place_ids.jsonl", places_path)
        shutil.copyfile(FIXED_ROOT / "candidate_occurrences.jsonl", occurrences_path)
        shutil.copyfile(FIXED_ROOT / "search_provenance.json", provenance_path)
        resume_attempt = 0
        declared_unit_hashes = {}

    if not completed_root.is_dir():
        completed_root.mkdir(parents=True, exist_ok=True)
    if (
        _sha256_file(places_path) != frame["place_ids_sha256"]
        or _sha256_file(occurrences_path) != frame["candidate_occurrences_sha256"]
        or _sha256_file(provenance_path) != frame["search_provenance_sha256"]
    ):
        raise ValueError("run-local fixed frame differs from canonical source")

    expected_place_ids = {row["place_id"] for row in frame["places"]}
    candidates_by_id: dict[str, ForcedChoiceDeliveryCandidate] = {}
    classifications_by_id: dict[str, dict[str, Any]] = {}
    archives_by_id: dict[str, dict[str, Any]] = {}
    unit_hashes: dict[str, str] = {}
    for path in sorted(completed_root.glob("*.json")):
        value = _read_json(path)
        candidate, classification, archive = _validate_completed_unit(
            value,
            fixed_frame_sha256=frame["fixed_frame_sha256"],
            scientific_implementation_sha256=scientific_implementation_sha256,
        )
        place_id = candidate.location.place_id
        if (
            place_id not in expected_place_ids
            or place_id in candidates_by_id
            or path != _completed_unit_path(completed_root, place_id)
            or value["returned_model"] not in allowed_returned_models
        ):
            raise ValueError("completed candidate unit is outside the authenticated frame")
        candidates_by_id[place_id] = candidate
        classifications_by_id[place_id] = classification
        archives_by_id[place_id] = archive
        unit_hashes[path.name] = _sha256_file(path)
    if any(
        name not in unit_hashes or unit_hashes[name] != digest
        for name, digest in declared_unit_hashes.items()
    ):
        raise ValueError("completed candidate unit differs from the resume manifest")

    if resume and len(candidates_by_id) < len(expected_place_ids):
        require_research_wayback_service(
            state_dir=state_dir, direct_transport=_WaybackArchiveTransport(),
            envfile=ROOT / ".env",
        )

    prior_failures = _read_jsonl(failures_path) if failures_path.is_file() else []
    history = _read_jsonl(history_path) if history_path.is_file() else []
    if prior_failures:
        history.extend(
            {"resume_attempt": max(0, resume_attempt - 1), "record": row}
            for row in prior_failures
        )
    failures: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    base_manifest = {
        "schema_version": WORKFLOW_SCHEMA,
        "run_id": run_id,
        "run_mode": "full",
        "model": MODEL_ID,
        "promoted_returned_models": list(allowed_returned_models),
        "status": "full_running",
        "started_or_resumed_at": started.isoformat().replace("+00:00", "Z"),
        "resume_supported": True,
        "resume_attempt": resume_attempt,
        "configuration_sha256": configuration_sha256,
        "fixed_frame_sha256": frame["fixed_frame_sha256"],
        "fixed_candidate_count": 1226,
        "text_search_requests": 0,
        "promotion_file": promotion_file.resolve().relative_to(ROOT).as_posix(),
        "promotion_file_sha256": promotion["promotion_sha256"],
        "pilot_run_id": promotion["promotion"]["pilot_run_id"],
        "pilot_manifest_sha256": promotion["promotion"]["pilot_manifest_sha256"],
        "pilot_semantic_manifest_sha256": promotion["promotion"][
            "pilot_semantic_manifest_sha256"
        ],
        "prompt_template_version": PROMPT_VERSION,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
        "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
        "scientific_implementation_sha256": scientific_implementation_sha256,
        "implementation_files": implementation_files,
        "authorization_file_sha256": authorization_sha256,
        "canonical_gpt41_run_untouched": True,
    }

    def persist(status: str = "full_running") -> dict[str, Any]:
        ordered_ids = sorted(candidates_by_id)
        classification_rows = [classifications_by_id[item] for item in ordered_ids]
        archive_rows = [archives_by_id[item] for item in ordered_ids]
        _atomic_jsonl(classifications_path, classification_rows)
        _atomic_jsonl(archives_path, archive_rows)
        _atomic_jsonl(failures_path, failures)
        _atomic_jsonl(history_path, history)
        checkpoint_files = _checkpoint_hashes(
            run_root,
            [
                places_path,
                occurrences_path,
                provenance_path,
                classifications_path,
                archives_path,
                failures_path,
                history_path,
            ],
        )
        manifest = {
            **base_manifest,
            "status": status,
            "candidate_count": 1226,
            "completed_count": len(candidates_by_id),
            "failure_count": len(failures),
            "completed_unit_hashes": dict(sorted(unit_hashes.items())),
            "completed_units_sha256": _canonical_sha256(dict(sorted(unit_hashes.items()))),
            "checkpoint_file_hashes": checkpoint_files,
        }
        _atomic_json(manifest_path, manifest)
        return manifest

    persist()
    google = _LazyGoogleProvider()
    collector = build_wayback_collector(
        cache_path, state_dir=state_dir, direct_transport=_WaybackArchiveTransport(),
        envfile=ROOT / ".env",
    )
    client = _openai_client()
    for index, source in enumerate(frame["places"], 1):
        place_id = source["place_id"]
        if place_id in candidates_by_id:
            continue
        stage = "places_details"
        try:
            details = google.get_place_details(place_id)
            retrieved = datetime.now(timezone.utc)
            provenance = QueryProvenanceV2(
                source["query_definition_id"], source["source_zip"], "places_search"
            )
            location = candidate_location_from_details(details, provenance, retrieved)
            candidate = build_candidate_facts(details, provenance, retrieved)
            reviews = build_review_evidence(details, retrieved)
            if (
                location.place_id != place_id
                or candidate.source_place_id != place_id
                or location.candidate_id != candidate.candidate_id
            ):
                raise ValueError("Details response changed fixed candidate binding")
            stage = "wayback"
            wayback = collector.collect(candidate)
            if not isinstance(wayback, WaybackCollectionV2):
                raise ValueError("Wayback collector returned an invalid packet")
            packet = ClassifierInputV2(
                candidate,
                reviews,
                wayback,
                ClassifierContractV2.from_project_constants(),
            )
            stage = "classifier"
            decision, metadata, attempts = _classify_with_technical_retries(
                packet,
                client,
                "C_metadata_reviews_wayback",
                model=MODEL_ID,
            )
            if metadata.get("returned_model") not in allowed_returned_models:
                raise ValueError("full run returned an unpromoted model version")
            metadata = {**metadata, "status": "success", "attempt_count": attempts,
                        "decision_sha256": decision.decision_sha256}
            archive_record = _archive_checkpoint(place_id, wayback)
            unit = _build_completed_unit(
                location=location,
                packet=packet,
                decision=decision,
                metadata=metadata,
                archive_record=archive_record,
                fixed_frame_sha256=frame["fixed_frame_sha256"],
                scientific_implementation_sha256=scientific_implementation_sha256,
            )
            unit_path = _completed_unit_path(completed_root, place_id)
            if unit_path.exists():
                raise ValueError("completed candidate unit is immutable")
            _atomic_json(unit_path, unit)
            unit_hashes[unit_path.name] = _sha256_file(unit_path)
            delivery_candidate, classification_record, validated_archive = (
                _validate_completed_unit(
                    unit,
                    fixed_frame_sha256=frame["fixed_frame_sha256"],
                    scientific_implementation_sha256=scientific_implementation_sha256,
                )
            )
            candidates_by_id[place_id] = delivery_candidate
            classifications_by_id[place_id] = classification_record
            archives_by_id[place_id] = validated_archive
            persist()
            if len(candidates_by_id) % 10 == 0 or len(candidates_by_id) == 1226:
                print(
                    json.dumps(
                        {
                            "event": "forced_choice_progress",
                            "run_id": run_id,
                            "completed": len(candidates_by_id),
                            "total": 1226,
                            "failures_current_attempt": len(failures),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        except WaybackDeferred as exc:
            retry = {"reason": exc.reason, "retry_at": exc.retry_at}
            failures.append({
                "place_id": place_id, "stage": "wayback",
                "failure_type": type(exc).__name__, "retryable": True,
                "resume_attempt": resume_attempt, **retry,
            })
            manifest = persist("full_paused_wayback")
            manifest["wayback_retry"] = retry
            _atomic_json(manifest_path, manifest)
            return manifest
        except Exception as exc:
            failures.append(
                {
                    "place_id": place_id,
                    "stage": stage,
                    "failure_type": type(exc).__name__,
                    "retryable": True,
                    "resume_attempt": resume_attempt,
                }
            )
            persist()

    if len(candidates_by_id) != 1226 or failures:
        manifest = persist("full_incomplete")
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        _atomic_json(manifest_path, manifest)
        return manifest

    ordered_candidates = tuple(
        candidates_by_id[row["place_id"]] for row in frame["places"]
    )
    context = load_forced_choice_local_context(
        ROOT / "analysis" / "input_manifest.json",
        tuple(sorted({row["source_zip"] for row in frame["places"]})),
    )
    pilot_root = promotion["pilot_manifest_path"].parent
    source_hashes = {
        "data/v2/canonical_thesis_empirical_run.json": _sha256_file(CANONICAL_POINTER),
        f"data/v2/{FIXED_RUN_ID}/place_ids.jsonl": frame["place_ids_sha256"],
        f"data/v2/{FIXED_RUN_ID}/candidate_occurrences.jsonl": frame[
            "candidate_occurrences_sha256"
        ],
        promotion_file.resolve().relative_to(ROOT).as_posix(): promotion[
            "promotion_sha256"
        ],
        pilot_root.joinpath("run_manifest.json").resolve().relative_to(ROOT).as_posix():
        _sha256_file(pilot_root / "run_manifest.json"),
        pilot_root.joinpath("ablation_results.jsonl").resolve().relative_to(ROOT).as_posix():
        _sha256_file(pilot_root / "ablation_results.jsonl"),
        pilot_root.joinpath("ablation_summary.json").resolve().relative_to(ROOT).as_posix():
        _sha256_file(pilot_root / "ablation_summary.json"),
    }
    delivery_root = run_root / "delivery"
    try:
        if delivery_root.exists():
            if not resume:
                raise FileExistsError(
                    "delivery root already exists before complete delivery"
                )
            delivered = _authenticate_completed_delivery(delivery_root)
        else:
            delivered = run_forced_choice_delivery(
                ordered_candidates,
                context=context,
                target_zips=tuple(
                    sorted({row["source_zip"] for row in frame["places"]})
                ),
                ablation_rows=promotion["ablation_rows"],
                ablation_summary=promotion["ablation_summary"],
                run_metadata={
                    "run_id": run_id,
                    "model": MODEL_ID,
                    "returned_model": allowed_returned_models[0],
                    "executed_at": datetime.now(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    "fixed_frame_sha256": frame["fixed_frame_sha256"],
                },
                source_hashes=source_hashes,
                output_root=delivery_root,
            )
    except Exception as exc:
        manifest = persist("full_delivery_failed")
        manifest.update(
            {
                "finished_at": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "failure_count": 1,
                "failure_stage": "delivery",
                "failure_type": type(exc).__name__,
                "resume_supported": True,
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    final = persist("full_complete")
    final.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "failure_count": 0,
            "delivery_status": delivered["status"],
            "delivery_failure_count": 0,
            "delivery_semantic_manifest_sha256": delivered[
                "semantic_manifest_sha256"
            ],
            "delivery_manifest_sha256": _sha256_file(
                delivery_root / "run_manifest.json"
            ),
        }
    )
    _atomic_json(manifest_path, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed-frame forced-choice workflow")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "v2")
    parser.add_argument("--promotion-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def _run_command(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _RUN_ID.fullmatch(args.run_id) is None:
        raise SystemExit("run ID must use YYYYMMDDTHHMMSSZ")
    load_dotenv(ROOT / ".env", override=False)
    if args.run_mode == "pilot":
        if args.promotion_file is not None:
            raise SystemExit("pilot mode does not accept promotion arguments")
        result = run_pilot(args.run_id, args.output_root, resume=args.resume)
        success = result.get("status") == "pilot_complete"
    else:
        if args.promotion_file is None:
            raise SystemExit("full mode requires the authenticated pilot promotion file")
        result = run_full(
            args.run_id,
            args.output_root,
            args.promotion_file,
            resume=args.resume,
        )
        success = (
            result.get("status") == "full_complete"
            and result.get("delivery_status") == "full_delivery_complete"
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    if str(result.get("status", "")).endswith("_paused_wayback"):
        return 3
    return 0 if success else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return _run_command(argv)
    except WaybackDeferred as exc:
        print(json.dumps(deferred_report(exc), sort_keys=True, indent=2), file=sys.stderr)
        return 3
    except WaybackServiceUnavailable as exc:
        print(json.dumps({"status": "wayback_preflight_failed", "probe": exc.report},
                         sort_keys=True, indent=2), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
