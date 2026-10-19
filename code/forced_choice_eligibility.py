"""Board-blind deterministic reducer for forced-choice classifications."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from forced_choice_models import ForcedChoiceDecision, candidate_id_for_place_id


ELIGIBILITY_CONTRACT_VERSION = "nets-ai-v2.2-forced-choice-eligibility-v1"
REASON_CODES = (
    "classification_unavailable",
    "non_target",
    "inactive",
    "implausible",
    "identity_inconsistent",
    "all_four_core_judgments_positive",
)
_CONTRACT = {
    "contract_version": ELIGIBILITY_CONTRACT_VERSION,
    "positive_rule": {
        "target_construct_status": "target_pharmacy",
        "business_status": "active",
        "candidate_plausibility": "plausible",
        "identity_consistency": "consistent",
    },
    "any_substantive_negative_excludes": True,
    "missing_technical_classification_unresolved": True,
    "confidence_affects_eligibility": False,
    "board_blind": True,
}
ELIGIBILITY_CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(
        _CONTRACT, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
).hexdigest()
_PLACE_ID = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, repr=False)
class ForcedChoiceEligibilityDecision:
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
        if self.candidate_id != candidate_id_for_place_id(self.place_id):
            raise ValueError("eligibility candidate binding is invalid")
        if self.contract_version != ELIGIBILITY_CONTRACT_VERSION or (
            self.contract_sha256 != ELIGIBILITY_CONTRACT_SHA256
        ):
            raise ValueError("invalid forced-choice eligibility contract")
        if self.eligibility_state not in {"include", "exclude", "unresolved"}:
            raise ValueError("invalid eligibility state")
        if (
            not isinstance(self.reason_codes, tuple)
            or not self.reason_codes
            or len(self.reason_codes) != len(set(self.reason_codes))
            or any(item not in REASON_CODES for item in self.reason_codes)
            or self.reason_codes != tuple(
                item for item in REASON_CODES if item in self.reason_codes
            )
        ):
            raise ValueError("invalid eligibility reason codes")
        fields = (
            self.target_construct_status,
            self.business_status,
            self.candidate_plausibility,
            self.identity_consistency,
        )
        if self.classification_decision_sha256 is None:
            if any(value is not None for value in fields) or (
                self.eligibility_state != "unresolved"
                or self.reason_codes != ("classification_unavailable",)
            ):
                raise ValueError("technical unresolved eligibility is inconsistent")
        elif _SHA256.fullmatch(self.classification_decision_sha256) is None:
            raise ValueError("invalid classification decision hash")

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


def reduce_forced_choice_eligibility(
    decision: ForcedChoiceDecision | None,
    *,
    place_id: str,
    candidate_id: str,
) -> ForcedChoiceEligibilityDecision:
    if decision is None:
        return ForcedChoiceEligibilityDecision(
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
    if not isinstance(decision, ForcedChoiceDecision):
        raise TypeError("forced-choice reducer requires a validated decision")
    if decision.candidate_id != candidate_id:
        raise ValueError("decision and eligibility candidate IDs differ")
    negative = []
    if decision.target_construct_status == "non_target":
        negative.append("non_target")
    if decision.business_status == "inactive":
        negative.append("inactive")
    if decision.candidate_plausibility == "implausible":
        negative.append("implausible")
    if decision.identity_consistency == "inconsistent":
        negative.append("identity_inconsistent")
    state = "exclude" if negative else "include"
    reasons = tuple(negative) if negative else ("all_four_core_judgments_positive",)
    return ForcedChoiceEligibilityDecision(
        place_id,
        candidate_id,
        decision.decision_sha256,
        ELIGIBILITY_CONTRACT_VERSION,
        ELIGIBILITY_CONTRACT_SHA256,
        decision.target_construct_status,
        decision.business_status,
        decision.candidate_plausibility,
        decision.identity_consistency,
        state,
        reasons,
    )


__all__ = [
    "ELIGIBILITY_CONTRACT_SHA256",
    "ELIGIBILITY_CONTRACT_VERSION",
    "ForcedChoiceEligibilityDecision",
    "reduce_forced_choice_eligibility",
]
