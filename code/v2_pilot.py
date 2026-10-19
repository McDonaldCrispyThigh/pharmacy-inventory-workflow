"""Pure offline helpers for the bounded v2 representative pilot.

This module performs no file, network, provider, or Board access.  It selects a
small pre-LLM set from already assembled Google/Wayback packets and builds three
prompt conditions from the same packet.  Exact provider content remains in memory;
the returned ledgers contain only allowed identifiers, categorical decisions, and
hashes.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from v2_classifier import build_classifier_messages
from v2_evidence_models import (
    ArchiveEvidenceV2,
    ClassifierContractV2,
    ClassifierInputV2,
    EvidenceValidationError,
    LLMDecisionV2,
    ReviewEvidenceV2,
    WaybackCollectionV2,
    evidence_id,
    validate_llm_decision,
)
from v2_eligibility import (
    classification_decision_sha256,
    reduce_eligibility,
)


ABLATION_CONDITIONS = (
    "A_metadata_only",
    "B_metadata_reviews",
    "C_metadata_reviews_wayback",
)

_ABLATION_OUTPUT_RULES = {
    "A_metadata_only": (
        "Reviews and Wayback evidence are deliberately withheld in this condition. "
        "Return identity_consistency NotAssessable. Return historical_year_value "
        "null, historical_year_type unknown, and no historical-year evidence IDs. "
        "If calculated_naics or another candidate-assessable field is non-null or "
        "assessed, cite the candidate evidence ID exactly."
    ),
    "B_metadata_reviews": (
        "Wayback evidence is deliberately withheld in this condition. Review dates "
        "cannot support a historical year, so return historical_year_value null, "
        "historical_year_type unknown, and no historical-year evidence IDs. Assess "
        "identity only when both the candidate and a supplied review support it; "
        "otherwise return NotAssessable."
    ),
    "C_metadata_reviews_wayback": (
        "Use only evidence displayed in the candidate, reviews, and Wayback blocks."
    ),
}

PILOT_ROLES = (
    "repeated_public_name",
    "single_public_name",
    "provider_closed",
    "institutional_keyword",
    "ambiguous_provider_type",
    "website_present",
    "website_absent",
    "reviews_present",
    "reviews_absent",
    "wayback_rich",
    "wayback_sparse",
)

_SPECIAL_ROLE_WEIGHT = {
    "repeated_public_name": 2,
    "provider_closed": 3,
    "institutional_keyword": 3,
    "ambiguous_provider_type": 3,
    "website_absent": 2,
    "reviews_absent": 2,
    "wayback_rich": 2,
}
_INSTITUTIONAL = re.compile(
    r"\b(?:hospital|clinic|medical|health\s*system|university|institutional)\b",
    re.I,
)


@dataclass(frozen=True, slots=True, repr=False)
class PilotSelectionV2:
    packets: tuple[ClassifierInputV2, ...]
    selection_ledger: tuple[dict[str, Any], ...]
    unavailable_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        packets = tuple(self.packets)
        ledger = tuple(self.selection_ledger)
        if not 8 <= len(packets) <= 10:
            raise ValueError("pilot selection must contain 8--10 candidates")
        if len(ledger) != len(packets):
            raise ValueError("pilot selection ledger is incomplete")
        place_ids = [item.candidate.source_place_id for item in packets]
        if len(place_ids) != len(set(place_ids)):
            raise ValueError("pilot candidates must be unique")
        if any(not isinstance(item, ClassifierInputV2) for item in packets):
            raise TypeError("pilot selection requires classifier packets")
        object.__setattr__(self, "packets", packets)
        object.__setattr__(self, "selection_ledger", ledger)
        object.__setattr__(self, "unavailable_roles", tuple(self.unavailable_roles))


def _normalized_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _roles(
    packet: ClassifierInputV2,
    name_counts: Counter[str],
) -> frozenset[str]:
    candidate = packet.candidate
    normalized_name = _normalized_name(candidate.name)
    roles = {
        "repeated_public_name"
        if name_counts[normalized_name] > 1
        else "single_public_name",
        "website_present" if candidate.website else "website_absent",
        "reviews_present"
        if packet.review_evidence.returned_count > 0
        else "reviews_absent",
        "wayback_rich"
        if len(packet.wayback_collection.archive_evidence.source_snapshots) >= 2
        else "wayback_sparse",
    }
    if candidate.provider_business_status in {
        "CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY"
    }:
        roles.add("provider_closed")
    name_and_types = " ".join((candidate.name, *candidate.types)).replace("_", " ")
    if _INSTITUTIONAL.search(name_and_types):
        roles.add("institutional_keyword")
    provider_types = {item.casefold() for item in candidate.types}
    if provider_types.isdisjoint({"pharmacy", "drugstore", "drug_store"}):
        roles.add("ambiguous_provider_type")
    return frozenset(roles)


def select_representative_candidates(
    packets: Sequence[ClassifierInputV2],
    *,
    target_size: int = 8,
) -> PilotSelectionV2:
    """Greedily cover predeclared pre-LLM roles in a capped 8--10 packet pool."""
    if type(target_size) is not int or not 8 <= target_size <= 10:
        raise ValueError("pilot target size must be 8--10")
    if not isinstance(packets, (tuple, list)):
        raise TypeError("pilot pool must be an ordered sequence")
    pool = tuple(packets)
    if len(pool) < target_size:
        raise ValueError("pilot candidate pool is smaller than target_size")
    if any(not isinstance(item, ClassifierInputV2) for item in pool):
        raise TypeError("pilot pool requires classifier packets")
    place_ids = [item.candidate.source_place_id for item in pool]
    if len(place_ids) != len(set(place_ids)):
        raise ValueError("pilot staging pool contains duplicate Place IDs")

    name_counts = Counter(_normalized_name(item.candidate.name) for item in pool)
    roles_by_index = [_roles(item, name_counts) for item in pool]
    available = set().union(*roles_by_index)
    covered: set[str] = set()
    remaining = list(range(len(pool)))
    selected_indices: list[int] = []

    while len(selected_indices) < target_size:
        def score(index: int) -> tuple[int, int, int]:
            new_roles = roles_by_index[index] - covered
            weighted = sum(_SPECIAL_ROLE_WEIGHT.get(role, 1)
                           for role in new_roles)
            return weighted, len(new_roles), -index

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected_indices.append(chosen)
        covered.update(roles_by_index[chosen])

    selected = tuple(pool[index] for index in selected_indices)
    ledger = tuple(
        {
            "selection_rank": rank,
            "provider_order_index": index,
            "place_id": pool[index].candidate.source_place_id,
            "candidate_id": pool[index].candidate.candidate_id,
            "roles": [role for role in PILOT_ROLES
                      if role in roles_by_index[index]],
        }
        for rank, index in enumerate(selected_indices, 1)
    )
    unavailable = tuple(role for role in PILOT_ROLES if role not in available)
    return PilotSelectionV2(selected, ledger, unavailable)


def _remove_block(content: str, tag: str) -> str:
    pattern = re.compile(
        rf"\n<{re.escape(tag)}>\n.*?\n</{re.escape(tag)}>",
        re.S,
    )
    stripped, count = pattern.subn("", content)
    if count != 1:
        raise EvidenceValidationError(f"classifier prompt lacks {tag} block")
    return stripped


def build_ablation_messages(
    packet: ClassifierInputV2,
    condition: str,
) -> list[dict[str, str]]:
    """Return one transient prompt containing only the condition's evidence blocks."""
    condition_packet = build_ablation_packet(packet, condition)
    messages = build_classifier_messages(condition_packet)
    content = messages[1]["content"]
    if condition == "A_metadata_only":
        content = _remove_block(content, "UNTRUSTED_REVIEWS_JSON")
        content = _remove_block(content, "UNTRUSTED_WAYBACK_JSON")
    elif condition == "B_metadata_reviews":
        content = _remove_block(content, "UNTRUSTED_WAYBACK_JSON")
    return [
        dict(messages[0]),
        {
            "role": "user",
            "content": (
                f"ABLATION_CONDITION={condition}\n"
                f"CONDITION_OUTPUT_RULE={_ABLATION_OUTPUT_RULES[condition]}\n"
                f"{content}"
            ),
        },
    ]


def _withheld_reviews(packet: ClassifierInputV2) -> ReviewEvidenceV2:
    """Represent a deliberately withheld review block without copying its state."""
    return ReviewEvidenceV2(
        "not_requested",
        packet.candidate.source_place_id,
        packet.candidate.retrieved_at,
        "provider_relevance",
        0,
        None,
        (),
    )


def _withheld_wayback(packet: ClassifierInputV2) -> WaybackCollectionV2:
    """Represent a deliberately withheld archive block with no original evidence."""
    original = packet.wayback_collection.archive_evidence
    empty = ArchiveEvidenceV2(
        "missing_url",
        "not_attempted",
        "not_applicable",
        None,
        original.query_scope,
        packet.candidate.retrieved_at,
        original.max_cdx_pages,
        original.max_cdx_records,
        None,
        (),
        (),
        (),
        (),
        (),
    )
    return WaybackCollectionV2(empty, (), ())


def build_ablation_packet(
    packet: ClassifierInputV2,
    condition: str,
) -> ClassifierInputV2:
    """Build the exact evidence packet exposed to one ablation condition.

    This packet is used for both prompt creation and output validation.  A and B
    therefore cannot authenticate a citation against evidence withheld by their
    experimental condition, even though the complete C packet remains in memory.
    """
    if not isinstance(packet, ClassifierInputV2):
        raise EvidenceValidationError("invalid pilot classifier packet")
    if condition not in ABLATION_CONDITIONS:
        raise ValueError("invalid ablation condition")
    if condition == "C_metadata_reviews_wayback":
        return packet
    reviews = (
        _withheld_reviews(packet)
        if condition == "A_metadata_only"
        else packet.review_evidence
    )
    return ClassifierInputV2(
        packet.candidate,
        reviews,
        _withheld_wayback(packet),
        packet.contract,
    )


def _allowed_evidence_ids(packet: ClassifierInputV2) -> frozenset[str]:
    allowed = {packet.candidate.candidate_id}
    allowed.update(item.evidence_id for item in packet.review_evidence.snippets)
    archive = packet.wayback_collection.archive_evidence
    allowed.update(item.evidence_id for item in archive.source_snapshots)
    allowed.update(item.evidence_id for item in archive.opening_claims)
    return frozenset(allowed)


def _decision_record(decision: LLMDecisionV2) -> dict[str, Any]:
    return {
        "calculated_naics": decision.calculated_naics,
        "target_construct_status": decision.target_construct_status,
        "business_status": decision.business_status,
        "candidate_plausibility": decision.candidate_plausibility,
        "identity_consistency": decision.identity_consistency,
        "historical_year_value": decision.historical_year_value,
        "historical_year_type": decision.historical_year_type,
        "claim_confidence": dict(decision.claim_confidence),
        "claim_evidence_ids": {
            key: list(value) for key, value in decision.claim_evidence_ids.items()
        },
        "reasoning_hashes": dict(decision.reasoning_hashes),
    }


def _prompt_sha256(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(
            messages,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _supplied_counts(packet: ClassifierInputV2) -> dict[str, int]:
    archive = packet.wayback_collection.archive_evidence
    return {
        "candidate": 1,
        "reviews": len(packet.review_evidence.snippets),
        "wayback_sources": len(archive.source_snapshots),
        "wayback_opening_claims": len(archive.opening_claims),
    }


def _base_row(
    packet: ClassifierInputV2,
    condition: str,
    model: str,
    selection_rank: int,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "selection_rank": selection_rank,
        "place_id": packet.candidate.source_place_id,
        "candidate_id": packet.candidate.candidate_id,
        "condition": condition,
        "model": model,
        "schema_version": packet.contract.schema_version,
        "prompt_template_version": packet.contract.prompt_template_version,
        "classifier_contract_hash": packet.contract.contract_hash,
        "prompt_sha256": _prompt_sha256(messages),
        "supplied_evidence_counts": _supplied_counts(packet),
    }


_EVALUATOR_METADATA_FIELDS = frozenset({
    "returned_model", "requested_at", "system_fingerprint", "token_usage",
    "request_count",
})
_TOKEN_USAGE_FIELDS = frozenset({
    "input_tokens", "output_tokens", "total_tokens",
})


def _default_evaluator_metadata() -> dict[str, Any]:
    return {
        "returned_model": None,
        "requested_at": None,
        "system_fingerprint": None,
        "token_usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        "request_count": 1,
    }


def _validated_evaluator_result(value: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if isinstance(value, Mapping):
        return value, _default_evaluator_metadata()
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not isinstance(value[0], Mapping)
        or type(value[1]) is not dict
        or set(value[1]) != _EVALUATOR_METADATA_FIELDS
    ):
        raise EvidenceValidationError("ablation evaluator returned invalid metadata")
    metadata = dict(value[1])
    for key in ("returned_model", "system_fingerprint"):
        item = metadata[key]
        if item is not None and (
            not isinstance(item, str)
            or re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", item) is None
        ):
            raise EvidenceValidationError("ablation provider identifier is invalid")
    requested_at = metadata["requested_at"]
    if requested_at is not None:
        if not isinstance(requested_at, str):
            raise EvidenceValidationError("ablation request time is invalid")
        raw = requested_at[:-1] + "+00:00" if requested_at.endswith("Z") else requested_at
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise EvidenceValidationError("ablation request time is invalid") from exc
        if parsed.tzinfo is None:
            raise EvidenceValidationError("ablation request time lacks timezone")
    usage = metadata["token_usage"]
    if (
        type(usage) is not dict
        or set(usage) != _TOKEN_USAGE_FIELDS
        or any(
            item is not None and (type(item) is not int or item < 0)
            for item in usage.values()
        )
        or metadata["request_count"] != 1
    ):
        raise EvidenceValidationError("ablation request accounting is invalid")
    if all(type(usage[key]) is int for key in _TOKEN_USAGE_FIELDS) and (
        usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
    ):
        raise EvidenceValidationError("ablation token total is inconsistent")
    return value[0], {
        "returned_model": metadata["returned_model"],
        "requested_at": metadata["requested_at"],
        "system_fingerprint": metadata["system_fingerprint"],
        "token_usage": dict(usage),
        "request_count": 1,
    }


_SEMANTIC_DECISION_FIELDS = (
    "calculated_naics",
    "target_construct_status",
    "business_status",
    "candidate_plausibility",
    "identity_consistency",
    "historical_year_value",
    "historical_year_type",
)


def _decision_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    decision = row.get("decision")
    if not isinstance(decision, Mapping):
        return (row.get("classification_status"), None)
    return (
        row.get("classification_status"),
        *(decision.get(field) for field in _SEMANTIC_DECISION_FIELDS),
    )


def _eligibility_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("eligibility_state"),
        tuple(row.get("eligibility_reason_codes", ())),
    )


def _add_change_flags(rows: list[dict[str, Any]]) -> None:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate_id"], {})[row["condition"]] = row
    for conditions in by_candidate.values():
        baseline = conditions[ABLATION_CONDITIONS[0]]
        previous = baseline
        for index, condition in enumerate(ABLATION_CONDITIONS):
            row = conditions[condition]
            row["decision_changed_from_A"] = (
                False if index == 0
                else _decision_signature(row) != _decision_signature(baseline)
            )
            row["eligibility_changed_from_A"] = (
                False if index == 0
                else _eligibility_signature(row) != _eligibility_signature(baseline)
            )
            row["decision_changed_from_previous"] = (
                False if index == 0
                else _decision_signature(row) != _decision_signature(previous)
            )
            row["eligibility_changed_from_previous"] = (
                False if index == 0
                else _eligibility_signature(row) != _eligibility_signature(previous)
            )
            previous = row


def run_ablation(
    selection: PilotSelectionV2,
    evaluator: Callable[[list[dict[str, str]], str], Mapping[str, Any]],
    *,
    model: str,
) -> tuple[dict[str, Any], ...]:
    """Run A/B/C through an injected evaluator and return a safe in-memory ledger.

    The evaluator receives messages and the model name only.  It never receives the
    full packet, so a condition cannot inspect evidence that its messages omit.
    """
    if not isinstance(selection, PilotSelectionV2):
        raise TypeError("ablation requires a bounded pilot selection")
    if not callable(evaluator):
        raise TypeError("ablation evaluator must be callable")
    if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", model):
        raise ValueError("invalid model identifier")

    rank_by_place = {
        row["place_id"]: row["selection_rank"]
        for row in selection.selection_ledger
    }
    ledger: list[dict[str, Any]] = []
    for complete_packet in selection.packets:
        for condition in ABLATION_CONDITIONS:
            condition_packet = build_ablation_packet(complete_packet, condition)
            messages = build_ablation_messages(complete_packet, condition)
            base = _base_row(
                condition_packet,
                condition,
                model,
                rank_by_place[complete_packet.candidate.source_place_id],
                messages,
            )
            try:
                raw, evaluator_metadata = _validated_evaluator_result(
                    evaluator(messages, model)
                )
            except Exception:  # provider failures become explicit pilot observations
                eligibility = reduce_eligibility(
                    None,
                    place_id=condition_packet.candidate.source_place_id,
                    candidate_id=condition_packet.candidate.candidate_id,
                    classification_decision_sha256=None,
                )
                ledger.append({
                    **base,
                    **_default_evaluator_metadata(),
                    "classification_status": "evaluator_error",
                    "classification_decision_sha256": None,
                    "decision": None,
                    "eligibility_contract_version": eligibility.contract_version,
                    "eligibility_contract_sha256": eligibility.contract_sha256,
                    "eligibility_state": eligibility.eligibility_state,
                    "eligibility_reason_codes": list(eligibility.reason_codes),
                })
                continue

            try:
                if not isinstance(raw, Mapping):
                    raise EvidenceValidationError(
                        "ablation evaluator returned invalid JSON"
                    )
                decision = validate_llm_decision(condition_packet, dict(raw))
                allowed = _allowed_evidence_ids(condition_packet)
                cited = {
                    evidence_id
                    for values in decision.claim_evidence_ids.values()
                    for evidence_id in values
                }
                if not cited.issubset(allowed):
                    raise EvidenceValidationError(
                        "ablation decision cited evidence withheld by its condition"
                    )
                decision_hash = classification_decision_sha256(decision)
                eligibility = reduce_eligibility(
                    decision,
                    place_id=condition_packet.candidate.source_place_id,
                    candidate_id=condition_packet.candidate.candidate_id,
                    classification_decision_sha256=decision_hash,
                )
            except (EvidenceValidationError, TypeError, ValueError):
                eligibility = reduce_eligibility(
                    None,
                    place_id=condition_packet.candidate.source_place_id,
                    candidate_id=condition_packet.candidate.candidate_id,
                    classification_decision_sha256=None,
                )
                ledger.append({
                    **base,
                    **evaluator_metadata,
                    "classification_status": "validation_error",
                    "classification_decision_sha256": None,
                    "decision": None,
                    "eligibility_contract_version": eligibility.contract_version,
                    "eligibility_contract_sha256": eligibility.contract_sha256,
                    "eligibility_state": eligibility.eligibility_state,
                    "eligibility_reason_codes": list(eligibility.reason_codes),
                })
                continue

            ledger.append({
                **base,
                **evaluator_metadata,
                "classification_status": "success",
                "classification_decision_sha256": decision_hash,
                "decision": _decision_record(decision),
                "eligibility_contract_version": eligibility.contract_version,
                "eligibility_contract_sha256": eligibility.contract_sha256,
                "eligibility_state": eligibility.eligibility_state,
                "eligibility_reason_codes": list(eligibility.reason_codes),
            })
    _add_change_flags(ledger)
    return tuple(ledger)


_ROW_FIELDS = frozenset({
    "selection_rank",
    "place_id",
    "candidate_id",
    "condition",
    "model",
    "schema_version",
    "prompt_template_version",
    "classifier_contract_hash",
    "prompt_sha256",
    "supplied_evidence_counts",
    "returned_model",
    "requested_at",
    "system_fingerprint",
    "token_usage",
    "request_count",
    "classification_status",
    "classification_decision_sha256",
    "decision",
    "eligibility_contract_version",
    "eligibility_contract_sha256",
    "eligibility_state",
    "eligibility_reason_codes",
    "decision_changed_from_A",
    "eligibility_changed_from_A",
    "decision_changed_from_previous",
    "eligibility_changed_from_previous",
})
_DECISION_FIELDS = frozenset({
    "calculated_naics",
    "target_construct_status",
    "business_status",
    "candidate_plausibility",
    "identity_consistency",
    "historical_year_value",
    "historical_year_type",
    "claim_confidence",
    "claim_evidence_ids",
    "reasoning_hashes",
})
_CLAIM_FIELDS = frozenset({
    "calculated_naics",
    "target_construct_status",
    "business_status",
    "candidate_plausibility",
    "identity_consistency",
    "historical_year",
})
_EVIDENCE_COUNT_FIELDS = frozenset({
    "candidate", "reviews", "wayback_sources", "wayback_opening_claims",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACE_ID = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_UNSAFE_KEY = re.compile(
    r"(?:^|_)(?:board|raw|text|content|address|phone|website|"
    r"model_response|provider_response|review_text|archived_page|"
    r"prompt_content|prompt_message)(?:_|$)",
    re.I,
)
_SAFE_KEY_EXCEPTIONS = frozenset({
    "prompt_sha256", "prompt_template_version", "reasoning_hashes",
})


def _contains_unsafe_field(value: Any) -> bool:
    """Detect forbidden field names without retaining or returning their values."""
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                return True
            if (key not in _SAFE_KEY_EXCEPTIONS
                    and _UNSAFE_KEY.search(key) is not None):
                return True
            if _contains_unsafe_field(nested):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_unsafe_field(item) for item in value)
    return False


def _decision_from_safe_record(value: Any) -> LLMDecisionV2:
    if type(value) is not dict or set(value) != _DECISION_FIELDS:
        raise ValueError("invalid safe decision projection")
    for key in ("claim_confidence", "claim_evidence_ids", "reasoning_hashes"):
        if type(value[key]) is not dict or set(value[key]) != _CLAIM_FIELDS:
            raise ValueError("invalid safe decision map")
    if any(type(items) is not list for items in value["claim_evidence_ids"].values()):
        raise ValueError("invalid safe citation projection")
    return LLMDecisionV2(
        value["calculated_naics"],
        value["target_construct_status"],
        value["business_status"],
        value["candidate_plausibility"],
        value["identity_consistency"],
        value["historical_year_value"],
        value["historical_year_type"],
        value["claim_confidence"],
        value["claim_evidence_ids"],
        value["reasoning_hashes"],
    )


def _valid_evidence_counts(value: Any, condition: str) -> bool:
    if type(value) is not dict or set(value) != _EVIDENCE_COUNT_FIELDS:
        return False
    if any(type(item) is not int or item < 0 for item in value.values()):
        return False
    if value["candidate"] != 1 or value["reviews"] > 5:
        return False
    if condition == "A_metadata_only":
        return all(value[key] == 0 for key in (
            "reviews", "wayback_sources", "wayback_opening_claims",
        ))
    if condition == "B_metadata_reviews":
        return value["wayback_sources"] == value["wayback_opening_claims"] == 0
    return True


def _valid_evaluator_provenance(row: Mapping[str, Any]) -> bool:
    for key in ("returned_model", "system_fingerprint"):
        value = row[key]
        if value is not None and (
            not isinstance(value, str) or _MODEL_ID.fullmatch(value) is None
        ):
            return False
    requested_at = row["requested_at"]
    if requested_at is not None:
        if not isinstance(requested_at, str):
            return False
        raw = requested_at[:-1] + "+00:00" if requested_at.endswith("Z") else requested_at
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            return False
    usage = row["token_usage"]
    if (
        type(usage) is not dict
        or set(usage) != _TOKEN_USAGE_FIELDS
        or any(
            value is not None and (type(value) is not int or value < 0)
            for value in usage.values()
        )
        or row["request_count"] != 1
    ):
        return False
    return not (
        all(type(usage[key]) is int for key in _TOKEN_USAGE_FIELDS)
        and usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
    )


def _row_is_valid(row: Any) -> bool:
    """Validate one closed safe row, including hashes and reducer semantics."""
    if type(row) is not dict or set(row) != _ROW_FIELDS:
        return False
    contract = ClassifierContractV2.from_project_constants()
    if (type(row["selection_rank"]) is not int
            or row["selection_rank"] < 1
            or not isinstance(row["place_id"], str)
            or _PLACE_ID.fullmatch(row["place_id"]) is None
            or row["candidate_id"] != evidence_id(
                "candidate", {"source_place_id": row["place_id"]}
            )
            or row["condition"] not in ABLATION_CONDITIONS
            or not isinstance(row["model"], str)
            or _MODEL_ID.fullmatch(row["model"]) is None
            or row["schema_version"] != contract.schema_version
            or row["prompt_template_version"]
            != contract.prompt_template_version
            or row["classifier_contract_hash"] != contract.contract_hash
            or not isinstance(row["prompt_sha256"], str)
            or _SHA256.fullmatch(row["prompt_sha256"]) is None
            or not _valid_evidence_counts(
                row["supplied_evidence_counts"], row["condition"]
            )
            or not _valid_evaluator_provenance(row)
            or row["classification_status"] not in {
                "success", "evaluator_error", "validation_error",
            }
            or any(type(row[key]) is not bool for key in (
                "decision_changed_from_A",
                "eligibility_changed_from_A",
                "decision_changed_from_previous",
                "eligibility_changed_from_previous",
            ))
            or type(row["eligibility_reason_codes"]) is not list):
        return False
    try:
        if row["classification_status"] == "success":
            decision = _decision_from_safe_record(row["decision"])
            candidate_references = {
                item
                for values in decision.claim_evidence_ids.values()
                for item in values
                if item.startswith("candidate_")
            }
            if not candidate_references.issubset({row["candidate_id"]}):
                return False
            decision_hash = classification_decision_sha256(decision)
            if row["classification_decision_sha256"] != decision_hash:
                return False
        else:
            if (row["decision"] is not None
                    or row["classification_decision_sha256"] is not None):
                return False
            decision = None
            decision_hash = None
        expected = reduce_eligibility(
            decision,
            place_id=row["place_id"],
            candidate_id=row["candidate_id"],
            classification_decision_sha256=decision_hash,
        )
    except (EvidenceValidationError, TypeError, ValueError):
        return False
    return (
        row["eligibility_contract_version"] == expected.contract_version
        and row["eligibility_contract_sha256"] == expected.contract_sha256
        and row["eligibility_state"] == expected.eligibility_state
        and row["eligibility_reason_codes"] == list(expected.reason_codes)
    )


def _validated_groups(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], int]:
    """Validate cross-row provenance and return groups plus failure count."""
    failures = 0
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        condition = row["condition"]
        slots = groups.setdefault(candidate_id, {})
        if condition in slots:
            failures += 1
        else:
            slots[condition] = row
    if not 8 <= len(groups) <= 10 or len(rows) != 3 * len(groups):
        failures += 1
    if any(set(slots) != set(ABLATION_CONDITIONS) for slots in groups.values()):
        failures += 1
    if failures:
        return groups, failures

    ranks: set[int] = set()
    place_ids: set[str] = set()
    fixed_contracts: set[tuple[str, str, str, str]] = set()
    returned_models: set[str] = set()
    system_fingerprints: set[str] = set()
    for slots in groups.values():
        ordered = [slots[item] for item in ABLATION_CONDITIONS]
        ranks_for_candidate = {row["selection_rank"] for row in ordered}
        places_for_candidate = {row["place_id"] for row in ordered}
        if len(ranks_for_candidate) != 1 or len(places_for_candidate) != 1:
            failures += 1
            continue
        rank = ordered[0]["selection_rank"]
        place_id = ordered[0]["place_id"]
        if rank in ranks or place_id in place_ids:
            failures += 1
        ranks.add(rank)
        place_ids.add(place_id)
        fixed_contracts.update((
            row["model"],
            row["schema_version"],
            row["prompt_template_version"],
            row["classifier_contract_hash"],
        ) for row in ordered)
        returned_models.update(
            row["returned_model"] for row in ordered
            if row["returned_model"] is not None
        )
        system_fingerprints.update(
            row["system_fingerprint"] for row in ordered
            if row["system_fingerprint"] is not None
        )
        if (ordered[1]["supplied_evidence_counts"]["reviews"]
                != ordered[2]["supplied_evidence_counts"]["reviews"]):
            failures += 1
        if len({row["prompt_sha256"] for row in ordered}) != 3:
            failures += 1

        baseline = ordered[0]
        previous = baseline
        for index, row in enumerate(ordered):
            expected_flags = (
                False if index == 0
                else _decision_signature(row) != _decision_signature(baseline),
                False if index == 0
                else _eligibility_signature(row)
                != _eligibility_signature(baseline),
                False if index == 0
                else _decision_signature(row) != _decision_signature(previous),
                False if index == 0
                else _eligibility_signature(row)
                != _eligibility_signature(previous),
            )
            observed_flags = (
                row["decision_changed_from_A"],
                row["eligibility_changed_from_A"],
                row["decision_changed_from_previous"],
                row["eligibility_changed_from_previous"],
            )
            if observed_flags != expected_flags:
                failures += 1
            previous = row
    if (
        ranks != set(range(1, len(groups) + 1))
        or len(fixed_contracts) != 1
        or len(returned_models) > 1
        or len(system_fingerprints) > 1
    ):
        failures += 1
    return groups, failures


def _safe_summary(
    *,
    candidate_count: int,
    row_count: int,
    condition_failure_count: int,
    validation_failure_count: int,
    unsafe_content_detected: bool,
    privacy_check_passed: bool,
    board_separation_passed: bool,
    pilot_status: str,
    full_run_recommendation: str,
    evidence_recommendation: str,
    request_count: int,
    condition_request_counts: Mapping[str, int],
    returned_models: Sequence[str],
    system_fingerprints: Sequence[str],
    condition_eligibility_counts: Mapping[str, Any] | None = None,
    eligibility_transitions: Mapping[str, Any] | None = None,
    changed_candidates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fixed aggregate summary that cannot echo row content."""
    return {
        "candidate_count": candidate_count,
        "row_count": row_count,
        "condition_failure_count": condition_failure_count,
        "validation_failure_count": validation_failure_count,
        "unsafe_content_detected": unsafe_content_detected,
        "privacy_check_passed": privacy_check_passed,
        "board_separation_passed": board_separation_passed,
        "pilot_status": pilot_status,
        "full_run_recommendation": full_run_recommendation,
        "evidence_recommendation": evidence_recommendation,
        "request_count": request_count,
        "condition_request_counts": dict(condition_request_counts),
        "returned_models": list(returned_models),
        "system_fingerprints": list(system_fingerprints),
        "condition_eligibility_counts": dict(
            condition_eligibility_counts or {}
        ),
        "eligibility_transitions": dict(eligibility_transitions or {}),
        "changed_candidates": dict(changed_candidates or {}),
    }


def summarize_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    privacy_check_passed: bool | None = None,
    board_separation_passed: bool | None = None,
) -> dict[str, Any]:
    """Validate and summarize one bounded A/B/C pilot without echoing content."""
    if not isinstance(rows, (tuple, list)):
        frozen: tuple[Any, ...] = ()
        input_shape_failure = 1
    else:
        frozen = tuple(rows)
        input_shape_failure = 0
    unsafe = _contains_unsafe_field(frozen)
    explicit_privacy = privacy_check_passed is True
    explicit_board_separation = board_separation_passed is True
    safe_candidate_ids = {
        row.get("candidate_id")
        for row in frozen
        if type(row) is dict
        and isinstance(row.get("candidate_id"), str)
        and re.fullmatch(r"candidate_[0-9a-f]{64}", row["candidate_id"])
    }
    condition_failure_count = sum(
        type(row) is dict
        and row.get("classification_status")
        in {"evaluator_error", "validation_error"}
        for row in frozen
    )

    invalid_rows = input_shape_failure + sum(
        not _row_is_valid(row) for row in frozen
    )
    valid_rows = tuple(
        row for row in frozen if type(row) is dict and _row_is_valid(row)
    )
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    group_failures = 0
    if not invalid_rows:
        groups, group_failures = _validated_groups(valid_rows)
    validation_failure_count = invalid_rows + group_failures
    request_count = sum(
        row.get("request_count", 0)
        for row in frozen if type(row) is dict
        and type(row.get("request_count")) is int
        and row["request_count"] >= 0
    )
    condition_request_counts = {
        condition: sum(
            row.get("request_count", 0)
            for row in frozen if type(row) is dict
            and row.get("condition") == condition
            and type(row.get("request_count")) is int
            and row["request_count"] >= 0
        )
        for condition in ABLATION_CONDITIONS
    }
    returned_models = sorted({
        row["returned_model"] for row in frozen
        if type(row) is dict and isinstance(row.get("returned_model"), str)
        and _MODEL_ID.fullmatch(row["returned_model"]) is not None
    })
    system_fingerprints = sorted({
        row["system_fingerprint"] for row in frozen
        if type(row) is dict and isinstance(row.get("system_fingerprint"), str)
        and _MODEL_ID.fullmatch(row["system_fingerprint"]) is not None
    })
    provenance_summary = {
        "request_count": request_count,
        "condition_request_counts": condition_request_counts,
        "returned_models": returned_models,
        "system_fingerprints": system_fingerprints,
    }

    if (not explicit_privacy or not explicit_board_separation or unsafe):
        return _safe_summary(
            candidate_count=len(safe_candidate_ids),
            row_count=len(frozen),
            condition_failure_count=condition_failure_count,
            validation_failure_count=validation_failure_count,
            unsafe_content_detected=unsafe,
            privacy_check_passed=explicit_privacy,
            board_separation_passed=explicit_board_separation,
            pilot_status="blocked",
            full_run_recommendation="stop_full_v2",
            evidence_recommendation="stop_until_safety_gates_pass",
            **provenance_summary,
        )
    if validation_failure_count:
        return _safe_summary(
            candidate_count=len(safe_candidate_ids),
            row_count=len(frozen),
            condition_failure_count=condition_failure_count,
            validation_failure_count=validation_failure_count,
            unsafe_content_detected=False,
            privacy_check_passed=True,
            board_separation_passed=True,
            pilot_status="has_failures",
            full_run_recommendation="revise_and_repeat_pilot",
            evidence_recommendation="repair_invalid_pilot_artifacts",
            **provenance_summary,
        )

    condition_state_counts = {
        condition: dict(Counter(
            slots[condition]["eligibility_state"]
            for slots in groups.values()
        ))
        for condition in ABLATION_CONDITIONS
    }
    transitions: dict[str, dict[str, int]] = {}
    changed_candidates: dict[str, int] = {}
    for label, left, right in (
        ("A_to_B", ABLATION_CONDITIONS[0], ABLATION_CONDITIONS[1]),
        ("B_to_C", ABLATION_CONDITIONS[1], ABLATION_CONDITIONS[2]),
    ):
        transition_counts = Counter(
            f"{slots[left]['eligibility_state']}->"
            f"{slots[right]['eligibility_state']}"
            for slots in groups.values()
        )
        transitions[label] = dict(sorted(transition_counts.items()))
        changed_candidates[label] = sum(
            _decision_signature(slots[left]) != _decision_signature(slots[right])
            or _eligibility_signature(slots[left])
            != _eligibility_signature(slots[right])
            for slots in groups.values()
        )

    if condition_failure_count:
        pilot_status = "has_failures"
        recommendation = "revise_and_repeat_pilot"
        evidence_recommendation = "repair_failed_conditions_before_promotion"
    else:
        pilot_status = "clean"
        recommendation = "proceed"
        if changed_candidates["B_to_C"]:
            evidence_recommendation = "wayback_changes_pilot_decisions"
        elif changed_candidates["A_to_B"]:
            evidence_recommendation = "reviews_change_pilot_decisions"
        else:
            evidence_recommendation = "no_observed_pilot_decision_change"
    return _safe_summary(
        candidate_count=len(groups),
        row_count=len(frozen),
        condition_failure_count=condition_failure_count,
        validation_failure_count=0,
        unsafe_content_detected=False,
        privacy_check_passed=True,
        board_separation_passed=True,
        pilot_status=pilot_status,
        full_run_recommendation=recommendation,
        evidence_recommendation=evidence_recommendation,
        condition_eligibility_counts=condition_state_counts,
        eligibility_transitions=transitions,
        changed_candidates=changed_candidates,
        **provenance_summary,
    )


__all__ = [
    "ABLATION_CONDITIONS",
    "PILOT_ROLES",
    "PilotSelectionV2",
    "build_ablation_packet",
    "build_ablation_messages",
    "run_ablation",
    "select_representative_candidates",
    "summarize_ablation",
]
