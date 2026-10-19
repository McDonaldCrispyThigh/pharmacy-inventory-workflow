"""Pure, Board-blind eligibility reduction for validated v2.1 decisions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from v2_evidence_models import LLMDecisionV2, evidence_id


ELIGIBILITY_CONTRACT_VERSION = "nets-ai-v2.1-eligibility-v1"
ELIGIBILITY_STATES = ("include", "exclude", "unresolved")
REASON_CODES = (
    "classification_unavailable",
    "target_construct_not_supported",
    "explicitly_inactive",
    "candidate_refuted",
    "identity_inconsistent",
    "target_construct_unresolved",
    "target_construct_conflicting",
    "business_status_unknown",
    "candidate_plausibility_unresolved",
    "candidate_plausibility_conflicting",
    "identity_ambiguous",
    "identity_not_assessable",
    "all_required_dimensions_supported",
)
_PLACE_ID = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
_CANDIDATE_ID = re.compile(r"^candidate_[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _contract_payload() -> dict[str, Any]:
    return {
        "contract_version": ELIGIBILITY_CONTRACT_VERSION,
        "dimensions": (
            "target_construct_status",
            "business_status",
            "candidate_plausibility",
            "identity_consistency",
        ),
        "explicit_negative_rules": (
            ("target_construct_status", "NotSupported", "target_construct_not_supported"),
            ("business_status", "Inactive", "explicitly_inactive"),
            ("candidate_plausibility", "Refuted", "candidate_refuted"),
            ("identity_consistency", "Inconsistent", "identity_inconsistent"),
        ),
        "unresolved_rules": (
            ("target_construct_status", "Unresolved", "target_construct_unresolved"),
            ("target_construct_status", "Conflicting", "target_construct_conflicting"),
            ("business_status", "Unknown", "business_status_unknown"),
            ("candidate_plausibility", "Unresolved", "candidate_plausibility_unresolved"),
            ("candidate_plausibility", "Conflicting", "candidate_plausibility_conflicting"),
            ("identity_consistency", "Ambiguous", "identity_ambiguous"),
            ("identity_consistency", "NotAssessable", "identity_not_assessable"),
        ),
        "explicit_negative_precedence": True,
        "conflicting_alone_excludes": False,
        "classification_decision_hash_domain": (
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
        ),
        "non_inputs": (
            "calculated_naics",
            "historical_year_value",
            "historical_year_type",
            "claim_confidence",
            "board",
            "linkage",
            "neighborhood",
            "spatial",
        ),
        "reason_order": REASON_CODES,
    }


ELIGIBILITY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _contract_payload(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()


def _classification_decision_payload(decision: LLMDecisionV2) -> dict[str, Any]:
    """Return the exact public decision domain authenticated by classifier metadata."""
    if not isinstance(decision, LLMDecisionV2):
        raise TypeError("classification hash requires a validated v2.1 decision")
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


def classification_decision_sha256(decision: LLMDecisionV2) -> str:
    """Hash a validated decision with the classifier's canonical JSON encoding."""
    return hashlib.sha256(
        json.dumps(
            _classification_decision_payload(decision),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


_calculate_classification_decision_sha256 = classification_decision_sha256


def _classified_outcome(
    target_construct_status: str,
    business_status: str,
    candidate_plausibility: str,
    identity_consistency: str,
) -> tuple[str, tuple[str, ...]]:
    """Apply the frozen four-dimension truth table and canonical reason order."""
    allowed = (
        (target_construct_status, {"Supported", "NotSupported", "Unresolved", "Conflicting"}),
        (business_status, {"Active", "Inactive", "Unknown"}),
        (candidate_plausibility, {"Supported", "Refuted", "Unresolved", "Conflicting"}),
        (identity_consistency, {"Consistent", "Ambiguous", "Inconsistent", "NotAssessable"}),
    )
    if any(not isinstance(value, str) or value not in values
           for value, values in allowed):
        raise ValueError("invalid classified eligibility judgment")

    applicable: set[str] = set()
    explicit_negative = {
        ("target_construct_status", "NotSupported"):
            "target_construct_not_supported",
        ("business_status", "Inactive"): "explicitly_inactive",
        ("candidate_plausibility", "Refuted"): "candidate_refuted",
        ("identity_consistency", "Inconsistent"): "identity_inconsistent",
    }
    unresolved = {
        ("target_construct_status", "Unresolved"): "target_construct_unresolved",
        ("target_construct_status", "Conflicting"):
            "target_construct_conflicting",
        ("business_status", "Unknown"): "business_status_unknown",
        ("candidate_plausibility", "Unresolved"):
            "candidate_plausibility_unresolved",
        ("candidate_plausibility", "Conflicting"):
            "candidate_plausibility_conflicting",
        ("identity_consistency", "Ambiguous"): "identity_ambiguous",
        ("identity_consistency", "NotAssessable"): "identity_not_assessable",
    }
    judgments = {
        "target_construct_status": target_construct_status,
        "business_status": business_status,
        "candidate_plausibility": candidate_plausibility,
        "identity_consistency": identity_consistency,
    }
    explicit = {
        reason for (field, value), reason in explicit_negative.items()
        if judgments[field] == value
    }
    pending = {
        reason for (field, value), reason in unresolved.items()
        if judgments[field] == value
    }
    if explicit:
        state = "exclude"
        applicable = explicit | pending
    elif pending:
        state = "unresolved"
        applicable = pending
    else:
        state = "include"
        applicable = {"all_required_dimensions_supported"}
    return state, tuple(item for item in REASON_CODES if item in applicable)


@dataclass(frozen=True, slots=True, repr=False)
class EligibilityDecisionV2:
    place_id: str
    candidate_id: str
    classification_decision_sha256: str | None
    contract_version: str
    contract_sha256: str
    target_construct_status: str | None
    business_status: str | None
    candidate_plausibility: str | None
    identity_consistency: str | None
    eligibility_state: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.place_id, str) or _PLACE_ID.fullmatch(self.place_id) is None:
            raise ValueError("invalid eligibility Place ID")
        expected_candidate_id = evidence_id(
            "candidate", {"source_place_id": self.place_id}
        )
        if (
            not isinstance(self.candidate_id, str)
            or _CANDIDATE_ID.fullmatch(self.candidate_id) is None
            or self.candidate_id != expected_candidate_id
        ):
            raise ValueError("eligibility candidate ID does not match Place ID")
        if self.classification_decision_sha256 is not None and (
            not isinstance(self.classification_decision_sha256, str)
            or _SHA256.fullmatch(self.classification_decision_sha256) is None
        ):
            raise ValueError("invalid classification decision hash")
        if self.contract_version != ELIGIBILITY_CONTRACT_VERSION:
            raise ValueError("invalid eligibility contract version")
        if self.contract_sha256 != ELIGIBILITY_CONTRACT_SHA256:
            raise ValueError("invalid eligibility contract hash")
        if self.eligibility_state not in ELIGIBILITY_STATES:
            raise ValueError("invalid eligibility state")
        if (
            not isinstance(self.reason_codes, tuple)
            or not self.reason_codes
            or len(self.reason_codes) != len(set(self.reason_codes))
            or any(item not in REASON_CODES for item in self.reason_codes)
        ):
            raise ValueError("invalid eligibility reason codes")
        ordered = tuple(item for item in REASON_CODES if item in self.reason_codes)
        if ordered != self.reason_codes:
            raise ValueError("eligibility reason codes are not in contract order")
        missing_classification = self.classification_decision_sha256 is None
        judgments = (
            self.target_construct_status,
            self.business_status,
            self.candidate_plausibility,
            self.identity_consistency,
        )
        if missing_classification:
            if any(item is not None for item in judgments):
                raise ValueError("missing classification cannot retain judgments")
            if (
                self.eligibility_state != "unresolved"
                or self.reason_codes != ("classification_unavailable",)
            ):
                raise ValueError("missing classification must remain unresolved")
        else:
            if any(item is None for item in judgments):
                raise ValueError("classified eligibility row lacks a judgment")
            expected_state, expected_reasons = _classified_outcome(
                self.target_construct_status,
                self.business_status,
                self.candidate_plausibility,
                self.identity_consistency,
            )
            if (self.eligibility_state != expected_state
                    or self.reason_codes != expected_reasons):
                raise ValueError(
                    "eligibility state or reasons contradict classified judgments"
                )

    @staticmethod
    def contract_payload() -> dict[str, Any]:
        """Return a fresh JSON-domain copy of the authenticated reducer contract."""
        return json.loads(json.dumps(_contract_payload()))

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "place_id": self.place_id,
            "candidate_id": self.candidate_id,
            "classification_decision_sha256": self.classification_decision_sha256,
            "contract_version": self.contract_version,
            "contract_sha256": self.contract_sha256,
            "target_construct_status": self.target_construct_status,
            "business_status": self.business_status,
            "candidate_plausibility": self.candidate_plausibility,
            "identity_consistency": self.identity_consistency,
            "eligibility_state": self.eligibility_state,
            "reason_codes": list(self.reason_codes),
        }


def reduce_eligibility(
    decision: LLMDecisionV2 | None,
    *,
    place_id: str,
    candidate_id: str,
    classification_decision_sha256: str | None,
) -> EligibilityDecisionV2:
    """Reduce one validated classifier decision without consulting downstream data."""
    if decision is None:
        if classification_decision_sha256 is not None:
            raise ValueError("missing classification cannot have a decision hash")
        return EligibilityDecisionV2(
            place_id,
            candidate_id,
            None,
            ELIGIBILITY_CONTRACT_VERSION,
            ELIGIBILITY_CONTRACT_SHA256,
            None,
            None,
            None,
            None,
            "unresolved",
            ("classification_unavailable",),
        )
    if not isinstance(decision, LLMDecisionV2):
        raise TypeError("eligibility reducer requires a validated v2.1 decision")
    if not isinstance(classification_decision_sha256, str):
        raise ValueError("classified eligibility requires a decision hash")
    expected_decision_hash = _calculate_classification_decision_sha256(decision)
    if classification_decision_sha256 != expected_decision_hash:
        raise ValueError("classification decision hash does not match decision content")
    state, ordered_reasons = _classified_outcome(
        decision.target_construct_status,
        decision.business_status,
        decision.candidate_plausibility,
        decision.identity_consistency,
    )
    return EligibilityDecisionV2(
        place_id,
        candidate_id,
        classification_decision_sha256,
        ELIGIBILITY_CONTRACT_VERSION,
        ELIGIBILITY_CONTRACT_SHA256,
        decision.target_construct_status,
        decision.business_status,
        decision.candidate_plausibility,
        decision.identity_consistency,
        state,
        ordered_reasons,
    )


__all__ = [
    "ELIGIBILITY_CONTRACT_SHA256",
    "ELIGIBILITY_CONTRACT_VERSION",
    "ELIGIBILITY_STATES",
    "REASON_CODES",
    "EligibilityDecisionV2",
    "classification_decision_sha256",
    "reduce_eligibility",
]
