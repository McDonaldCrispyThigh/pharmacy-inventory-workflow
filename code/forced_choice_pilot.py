"""Eight-candidate A/B/C execution gate for the forced-choice workflow."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forced_choice_classifier import (
    ABLATION_CONDITIONS,
    MODEL_ID,
    PROMPT_TEMPLATE_SHA256,
    PROMPT_VERSION,
    ForcedChoiceResponseError,
    build_forced_choice_messages,
    classify_forced_choice,
    contains_board_data,
    prompt_sha256,
)
from forced_choice_eligibility import (
    ELIGIBILITY_CONTRACT_SHA256,
    ELIGIBILITY_CONTRACT_VERSION,
    reduce_forced_choice_eligibility,
)
from forced_choice_models import (
    CORE_FIELDS,
    FORCED_CHOICE_CONTRACT_SHA256,
    FORCED_CHOICE_SCHEMA_VERSION,
)
from v2_evidence_models import ClassifierInputV2
from v2_pilot import build_ablation_packet


PILOT_CONTRACT_VERSION = "nets-ai-v2.2-forced-choice-pilot-v1"
_RETRYABLE = frozenset(
    {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"}
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_counts(packet: ClassifierInputV2, condition: str) -> dict[str, int]:
    archive = packet.wayback_collection.archive_evidence
    return {
        "candidate": 1,
        "reviews": len(packet.review_evidence.snippets)
        if condition != "A_metadata_only" else 0,
        "wayback_sources": len(archive.source_snapshots)
        if condition == "C_metadata_reviews_wayback" else 0,
        "wayback_opening_claims": len(archive.opening_claims)
        if condition == "C_metadata_reviews_wayback" else 0,
    }


def _classify_with_technical_retries(
    packet: ClassifierInputV2,
    client: Any,
    condition: str,
    *,
    model: str,
    max_attempts: int = 10,
) -> tuple[Any, dict[str, Any], int]:
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            decision, metadata = classify_forced_choice(
                packet, client, model, condition=condition
            )
            return decision, metadata, attempt
        except ForcedChoiceResponseError as exc:
            last = exc
        except Exception as exc:  # only named transport/rate-limit failures retry
            if type(exc).__name__ not in _RETRYABLE:
                raise
            last = exc
        if attempt < max_attempts:
            time.sleep(min(2 ** (attempt - 1), 30))
    raise RuntimeError("forced-choice technical retries exhausted") from last


def run_forced_choice_ablation(
    packets: Sequence[ClassifierInputV2],
    openai_client: Any,
    *,
    model: str = MODEL_ID,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Run exactly 8 x 3 requests; substantive judgments never fail the gate."""
    packets = tuple(packets)
    if model != MODEL_ID:
        raise ValueError("forced-choice pilot permits gpt-5-mini only")
    if len(packets) != 8 or len({p.candidate.source_place_id for p in packets}) != 8:
        raise ValueError("forced-choice pilot requires eight unique candidates")
    rows: list[dict[str, Any]] = []
    for rank, complete_packet in enumerate(packets, 1):
        candidate_rows: list[dict[str, Any]] = []
        for condition in ABLATION_CONDITIONS:
            condition_packet = build_ablation_packet(complete_packet, condition)
            messages = build_forced_choice_messages(condition_packet, condition)
            base = {
                "selection_rank": rank,
                "place_id": complete_packet.candidate.source_place_id,
                "candidate_id": complete_packet.candidate.candidate_id,
                "condition": condition,
                "model": model,
                "schema_version": FORCED_CHOICE_SCHEMA_VERSION,
                "prompt_template_version": PROMPT_VERSION,
                "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
                "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
                "eligibility_contract_version": ELIGIBILITY_CONTRACT_VERSION,
                "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
                "prompt_sha256": prompt_sha256(messages),
                "supplied_evidence_counts": _evidence_counts(
                    condition_packet, condition
                ),
                "board_leakage_pass": not contains_board_data(messages),
            }
            try:
                decision, metadata, attempts = _classify_with_technical_retries(
                    condition_packet,
                    openai_client,
                    condition,
                    model=model,
                )
                eligibility = reduce_forced_choice_eligibility(
                    decision,
                    place_id=condition_packet.candidate.source_place_id,
                    candidate_id=condition_packet.candidate.candidate_id,
                )
                row = {
                    **base,
                    "classification_status": "success",
                    "technical_failure": False,
                    "validation_pass": True,
                    "candidate_binding_pass": (
                        decision.candidate_id == condition_packet.candidate.candidate_id
                    ),
                    "attempt_count": attempts,
                    "returned_model": metadata["returned_model"],
                    "requested_at": metadata["requested_at"],
                    "system_fingerprint": metadata["system_fingerprint"],
                    "token_usage": metadata["token_usage"],
                    "classification_decision_sha256": decision.decision_sha256,
                    "decision": decision.to_public_dict(),
                    "eligibility_state": eligibility.eligibility_state,
                    "eligibility_reason_codes": list(eligibility.reason_codes),
                }
            except Exception as exc:
                row = {
                    **base,
                    "classification_status": "technical_failure",
                    "technical_failure": True,
                    "validation_pass": False,
                    "candidate_binding_pass": False,
                    "attempt_count": 10,
                    "returned_model": None,
                    "requested_at": None,
                    "system_fingerprint": None,
                    "token_usage": {
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                    },
                    "classification_decision_sha256": None,
                    "decision": None,
                    "eligibility_state": "unresolved",
                    "eligibility_reason_codes": ["classification_unavailable"],
                    "failure_type": type(exc).__name__,
                }
            candidate_rows.append(row)
            rows.append(row)
        baseline = candidate_rows[0].get("decision")
        previous = None
        for row in candidate_rows:
            current = row.get("decision")
            row["decision_changed_from_A"] = current != baseline
            row["decision_changed_from_previous"] = (
                previous is not None and current != previous
            )
            if current is not None:
                row["core_changes_from_A"] = [
                    field for field in CORE_FIELDS
                    if baseline is not None and current[field] != baseline[field]
                ]
            else:
                row["core_changes_from_A"] = []
            previous = current

    expected_count = 8 * len(ABLATION_CONDITIONS)
    successful = [row for row in rows if row["classification_status"] == "success"]
    returned_models = sorted(
        {row["returned_model"] for row in successful if row["returned_model"]}
    )
    structural_pass = (
        len(rows) == expected_count
        and len(successful) == expected_count
        and all(row["validation_pass"] for row in rows)
        and all(row["candidate_binding_pass"] for row in rows)
        and all(row["board_leakage_pass"] for row in rows)
        and not any(row["technical_failure"] for row in rows)
    )
    summary = {
        "schema_version": PILOT_CONTRACT_VERSION,
        "pilot_status": "clean" if structural_pass else "failed",
        "candidate_count": len(packets),
        "row_count": len(rows),
        "request_count": len(rows),
        "successful_output_count": len(successful),
        "required_successful_output_count": expected_count,
        "validation_pass_count": sum(row["validation_pass"] for row in rows),
        "candidate_binding_pass_count": sum(
            row["candidate_binding_pass"] for row in rows
        ),
        "board_separation_passed": all(
            row["board_leakage_pass"] for row in rows
        ),
        "technical_failure_count": sum(row["technical_failure"] for row in rows),
        "returned_models": returned_models,
        "condition_request_counts": {
            condition: sum(row["condition"] == condition for row in rows)
            for condition in ABLATION_CONDITIONS
        },
        "condition_eligibility_counts": {
            condition: {
                state: sum(
                    row["condition"] == condition
                    and row["eligibility_state"] == state
                    for row in rows
                )
                for state in ("include", "exclude", "unresolved")
            }
            for condition in ABLATION_CONDITIONS
        },
        "substantive_disagreement_is_not_gate_failure": True,
        "purposive_pilot_does_not_estimate_population_accuracy": True,
        "promotion_passed": structural_pass,
    }
    return tuple(rows), summary


def write_forced_choice_pilot_package(
    output_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    selection_ledger: Sequence[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
    fixed_frame_sha256: str,
) -> dict[str, Any]:
    target = output_root.resolve()
    if target.exists():
        raise FileExistsError("forced-choice pilot package already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        artifacts = {
            "ablation_results.jsonl": "".join(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            "ablation_summary.json": json.dumps(
                dict(summary), sort_keys=True, indent=2
            ) + "\n",
            "pilot_selection.json": json.dumps(
                {"candidate_count": len(selection_ledger),
                 "selection_ledger": list(selection_ledger)},
                sort_keys=True,
                indent=2,
            ) + "\n",
        }
        hashes = {}
        for name, content in artifacts.items():
            path = temporary / name
            path.write_text(content, encoding="utf-8", newline="")
            hashes[name] = _sha256_file(path)
        semantic = {
            "schema_version": PILOT_CONTRACT_VERSION,
            "status": "pilot_complete" if summary.get("promotion_passed") else "pilot_failed",
            "model": MODEL_ID,
            "returned_models": summary.get("returned_models", []),
            "prompt_template_version": PROMPT_VERSION,
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
            "eligibility_contract_sha256": ELIGIBILITY_CONTRACT_SHA256,
            "fixed_candidate_frame_sha256": fixed_frame_sha256,
            "candidate_count": 8,
            "output_count": len(rows),
            "artifacts": hashes,
        }
        semantic_hash = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()
        manifest = {
            **semantic,
            "run_metadata": dict(run_metadata),
            "semantic_manifest_sha256": semantic_hash,
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="",
        )
        os.replace(temporary, target)
        return manifest
    except Exception:
        if temporary.exists() and temporary.parent == target.parent:
            shutil.rmtree(temporary)
        raise


__all__ = [
    "PILOT_CONTRACT_VERSION",
    "run_forced_choice_ablation",
    "write_forced_choice_pilot_package",
]
