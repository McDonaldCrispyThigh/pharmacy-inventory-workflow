"""Candidate-bound citation namespace correction; original decisions remain frozen."""
from dataclasses import dataclass, InitVar
from forced_choice_models import *
from forced_choice_models import _CANDIDATE, _SHA256, LABELS, YEAR_TYPES, CORE_CONFIDENCE, HISTORICAL_CONFIDENCE

@dataclass(frozen=True, slots=True, repr=False)
class ClosureDecision(ForcedChoiceDecision):
    allowed_ids: InitVar[frozenset[str]]
    def __post_init__(self, allowed_ids: frozenset[str]) -> None:
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
                not isinstance(item, str) or item not in allowed_ids
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

