"""Validated forced-choice decisions for the fixed NETS-AI candidate frame.

The model must make four substantive binary judgments whenever an evidence packet
is structurally interpretable.  Confidence is retained separately and never changes
the substantive label.  Historical-year output remains evidence bounded.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from v2_evidence_models import ClassifierInputV2, evidence_id


FORCED_CHOICE_SCHEMA_VERSION = "v2.2-forced-choice"
FORCED_CHOICE_CONTRACT_VERSION = "nets-ai-v2.2-forced-choice-classifier-v1"
TARGET_DEFINITION = (
    "One current, public-facing prescription-dispensing pharmacy outlet or "
    "pharmacy department named by the listing."
)
CORE_FIELDS = (
    "target_construct_status",
    "business_status",
    "candidate_plausibility",
    "identity_consistency",
)
CLAIM_FIELDS = (*CORE_FIELDS, "historical_year")
LABELS = {
    "target_construct_status": frozenset({"target_pharmacy", "non_target"}),
    "business_status": frozenset({"active", "inactive"}),
    "candidate_plausibility": frozenset({"plausible", "implausible"}),
    "identity_consistency": frozenset({"consistent", "inconsistent"}),
}
CORE_CONFIDENCE = frozenset({"High", "Moderate", "Low"})
HISTORICAL_CONFIDENCE = frozenset({"High", "Moderate", "Low", "NotAssessable"})
YEAR_TYPES = frozenset(
    {"explicit_archived_opening_claim", "active_by_upper_bound", "unknown"}
)
_CANDIDATE = re.compile(r"^candidate_[0-9a-f]{64}$")
_EVIDENCE = re.compile(
    r"^(?:candidate|review|archive_source_v2|opening_claim_v2)_[0-9a-f]{64}$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


FORCED_CHOICE_CONTRACT = {
    "contract_version": FORCED_CHOICE_CONTRACT_VERSION,
    "schema_version": FORCED_CHOICE_SCHEMA_VERSION,
    "target_definition": TARGET_DEFINITION,
    "candidate_unit": "one fixed Google Place ID",
    "core_labels": {key: sorted(values) for key, values in LABELS.items()},
    "ordinary_uncertainty_rule": (
        "Sparse evidence, missing reviews, missing website, missing Wayback evidence, "
        "or low confidence does not permit abstention on a core judgment."
    ),
    "historical_year_rule": "Historical year remains evidence bounded and may be unknown.",
    "board_blind": True,
}
FORCED_CHOICE_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_bytes(FORCED_CHOICE_CONTRACT)
).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class ForcedChoiceDecision:
    candidate_id: str
    target_construct_status: str
    business_status: str
    candidate_plausibility: str
    identity_consistency: str
    historical_year_value: int | None
    historical_year_type: str
    claim_confidence: Mapping[str, str]
    claim_evidence_ids: Mapping[str, tuple[str, ...]]
    reasoning_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or _CANDIDATE.fullmatch(
            self.candidate_id
        ) is None:
            raise ValueError("invalid forced-choice candidate ID")
        for field in CORE_FIELDS:
            value = getattr(self, field)
            if not isinstance(value, str) or value not in LABELS[field]:
                raise ValueError(f"invalid forced-choice label: {field}")
        if self.historical_year_type not in YEAR_TYPES:
            raise ValueError("invalid historical-year type")
        if (self.historical_year_type == "unknown") != (
            self.historical_year_value is None
        ):
            raise ValueError("historical-year value/type disagree")
        if self.historical_year_value is not None and (
            type(self.historical_year_value) is not int
            or self.historical_year_value < 1600
        ):
            raise ValueError("invalid historical year")

        confidence = dict(self.claim_confidence)
        citations = {
            key: tuple(value) for key, value in dict(self.claim_evidence_ids).items()
        }
        hashes = dict(self.reasoning_hashes)
        if set(confidence) != set(CLAIM_FIELDS):
            raise ValueError("invalid confidence fields")
        if set(citations) != set(CLAIM_FIELDS):
            raise ValueError("invalid evidence-ID fields")
        if set(hashes) != set(CLAIM_FIELDS):
            raise ValueError("invalid reasoning-hash fields")
        if any(confidence[field] not in CORE_CONFIDENCE for field in CORE_FIELDS):
            raise ValueError("core confidence must be High, Moderate, or Low")
        if confidence["historical_year"] not in HISTORICAL_CONFIDENCE:
            raise ValueError("invalid historical-year confidence")
        if any(
            not isinstance(values, tuple)
            or len(values) != len(set(values))
            or any(
                not isinstance(item, str) or _EVIDENCE.fullmatch(item) is None
                for item in values
            )
            for values in citations.values()
        ):
            raise ValueError("invalid forced-choice evidence IDs")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in hashes.values()
        ):
            raise ValueError("invalid reasoning hashes")
        for field in CORE_FIELDS:
            if self.candidate_id not in citations[field]:
                raise ValueError("each core judgment must consider candidate evidence")
        if self.historical_year_type == "unknown":
            if citations["historical_year"] or confidence["historical_year"] != "NotAssessable":
                raise ValueError("unknown historical year must remain uncited/not assessable")
        elif not citations["historical_year"]:
            raise ValueError("historical-year claim requires evidence")
        object.__setattr__(self, "claim_confidence", confidence)
        object.__setattr__(self, "claim_evidence_ids", citations)
        object.__setattr__(self, "reasoning_hashes", hashes)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_construct_status": self.target_construct_status,
            "business_status": self.business_status,
            "candidate_plausibility": self.candidate_plausibility,
            "identity_consistency": self.identity_consistency,
            "historical_year_value": self.historical_year_value,
            "historical_year_type": self.historical_year_type,
            "claim_confidence": dict(self.claim_confidence),
            "claim_evidence_ids": {
                key: list(value) for key, value in self.claim_evidence_ids.items()
            },
            "reasoning_hashes": dict(self.reasoning_hashes),
        }

    @property
    def decision_sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_public_dict())).hexdigest()

    @classmethod
    def from_public_dict(cls, value: Mapping[str, Any]) -> "ForcedChoiceDecision":
        if not isinstance(value, Mapping):
            raise TypeError("forced-choice decision record must be a mapping")
        expected = {
            "candidate_id", *CORE_FIELDS, "historical_year_value",
            "historical_year_type", "claim_confidence", "claim_evidence_ids",
            "reasoning_hashes",
        }
        if set(value) != expected:
            raise ValueError("forced-choice decision record has undeclared fields")
        return cls(
            value["candidate_id"],
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


def validate_forced_choice_decision(
    packet: ClassifierInputV2,
    raw: Mapping[str, Any],
) -> ForcedChoiceDecision:
    """Validate structure, candidate binding, citations, and historical-year rules."""
    if not isinstance(packet, ClassifierInputV2) or type(raw) is not dict:
        raise ValueError("invalid forced-choice decision packet")
    expected = {
        "candidate_id", *CORE_FIELDS, "historical_year_value",
        "historical_year_type", "claim_confidence", "claim_evidence_ids",
        "reasoning",
    }
    if set(raw) != expected:
        raise ValueError("forced-choice output has undeclared fields")
    if raw["candidate_id"] != packet.candidate.candidate_id:
        raise ValueError("forced-choice output is bound to the wrong candidate")
    reasoning = raw["reasoning"]
    if (
        type(reasoning) is not dict
        or set(reasoning) != set(CLAIM_FIELDS)
        or any(
            not isinstance(value, str) or not value.strip() or len(value) > 500
            for value in reasoning.values()
        )
    ):
        raise ValueError("invalid forced-choice reasoning")
    reasoning_hashes = {
        key: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for key, value in reasoning.items()
    }
    decision = ForcedChoiceDecision(
        raw["candidate_id"],
        raw["target_construct_status"],
        raw["business_status"],
        raw["candidate_plausibility"],
        raw["identity_consistency"],
        raw["historical_year_value"],
        raw["historical_year_type"],
        raw["claim_confidence"],
        raw["claim_evidence_ids"],
        reasoning_hashes,
    )

    archive = packet.wayback_collection.archive_evidence
    sources = {item.evidence_id: item for item in archive.source_snapshots}
    claims = {item.evidence_id: item for item in archive.opening_claims}
    known = (
        {packet.candidate.candidate_id}
        | {item.evidence_id for item in packet.review_evidence.snippets}
        | set(sources)
        | set(claims)
    )
    if any(
        not set(values).issubset(known)
        for values in decision.claim_evidence_ids.values()
    ):
        raise ValueError("forced-choice decision cites unknown/withheld evidence")

    match_scope = {
        item.source_evidence_id: item.evidence_scope
        for item in packet.wayback_collection.archive_match_assessments
    }
    claim_scope = {
        item.claim_evidence_id: item.evidence_scope
        for item in packet.wayback_collection.archive_claim_assessments
    }
    outlet_sources = {
        item_id for item_id, scope in match_scope.items()
        if scope == "outlet_specific"
    }
    outlet_claims = {
        item_id for item_id, scope in claim_scope.items()
        if scope == "outlet_specific"
    }
    outlet_years = {claims[item_id].year for item_id in outlet_claims}
    year_type = decision.historical_year_type
    year = decision.historical_year_value
    year_ids = set(decision.claim_evidence_ids["historical_year"])
    if packet.wayback_collection.outlet_opening_year_state == "conflict" and year_type != "unknown":
        raise ValueError("conflicting outlet opening years must remain unknown")
    if year_type == "active_by_upper_bound":
        if not outlet_sources:
            raise ValueError("active-by year lacks an outlet-specific archive source")
        earliest = min(sources[item_id].captured_at.year for item_id in outlet_sources)
        eligible = {
            item_id for item_id in outlet_sources
            if sources[item_id].captured_at.year == earliest
        }
        if year != earliest or not (year_ids & eligible):
            raise ValueError("invalid active-by year/evidence")
    elif year_type == "explicit_archived_opening_claim":
        eligible = {
            item_id for item_id in outlet_claims if claims[item_id].year == year
        }
        if len(outlet_years) != 1 or year not in outlet_years or not (year_ids & eligible):
            raise ValueError("invalid explicit archived opening claim")
    return decision


def candidate_id_for_place_id(place_id: str) -> str:
    return evidence_id("candidate", {"source_place_id": place_id})


__all__ = [
    "CLAIM_FIELDS",
    "CORE_FIELDS",
    "FORCED_CHOICE_CONTRACT",
    "FORCED_CHOICE_CONTRACT_SHA256",
    "FORCED_CHOICE_CONTRACT_VERSION",
    "FORCED_CHOICE_SCHEMA_VERSION",
    "ForcedChoiceDecision",
    "candidate_id_for_place_id",
    "validate_forced_choice_decision",
]
