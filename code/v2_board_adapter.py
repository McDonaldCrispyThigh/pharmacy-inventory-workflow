"""Deterministic Board-reference linkage downstream of v2.1 eligibility."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

import geo_match
from study_design import board_exclusion_reason
from v2_downstream_models import CandidateLocationV2
from v2_eligibility import EligibilityDecisionV2


LINKAGE_ADAPTER_VERSION = "nets-ai-v2.1-board-adapter-v1"
_PLACE_ID = re.compile(r"^[A-Za-z0-9_-]{10,256}$")


def _zip5(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    match = re.match(r"^(\d{5})", text)
    return match.group(1) if match else ""


def _license(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _percent(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 2) if denominator else None


@dataclass(frozen=True, slots=True)
class LinkageConfigV2:
    threshold: float = 75.0
    name_weight: float = 1.0
    address_weight: float = 0.8
    primary_objective: str = "max_weight"
    sensitivity_objective: str = "max_cardinality_then_weight"
    threshold_sensitivity: tuple[float, ...] = (70.0, 75.0, 80.0)

    def __post_init__(self) -> None:
        if type(self.threshold) not in {int, float} or not 0 <= float(self.threshold) <= 100:
            raise ValueError("invalid linkage threshold")
        if type(self.name_weight) not in {int, float} or float(self.name_weight) < 0:
            raise ValueError("invalid name weight")
        if type(self.address_weight) not in {int, float} or float(self.address_weight) < 0:
            raise ValueError("invalid address weight")
        if not (float(self.name_weight) > 0 or float(self.address_weight) > 0):
            raise ValueError("at least one linkage weight is required")
        allowed = {"max_weight", "max_cardinality_then_weight"}
        if self.primary_objective not in allowed or self.sensitivity_objective not in allowed:
            raise ValueError("invalid linkage objective")
        thresholds = tuple(float(value) for value in self.threshold_sensitivity)
        if not thresholds or any(not 0 <= value <= 100 for value in thresholds):
            raise ValueError("invalid threshold sensitivity")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "name_weight", float(self.name_weight))
        object.__setattr__(self, "address_weight", float(self.address_weight))
        object.__setattr__(self, "threshold_sensitivity", thresholds)


@dataclass(frozen=True, slots=True, repr=False)
class BoardFramesV2:
    full_active_target: pd.DataFrame
    screened_target: pd.DataFrame
    full_active_extended: pd.DataFrame
    screened_extended: pd.DataFrame
    audit: dict[str, Any]
    target_zips: tuple[str, ...] = ()
    extended_zips: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, repr=False)
class BoardLinkageResultV2:
    links: pd.DataFrame
    unmatched_google: pd.DataFrame
    unmatched_board: pd.DataFrame
    summary: dict[str, Any]


def prepare_v2_google_frame(
    locations: Sequence[CandidateLocationV2],
    eligibility: Sequence[EligibilityDecisionV2],
    classification_lineage: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    """Join immutable in-memory location packets to Board-blind reducer outputs."""
    if (
        not isinstance(locations, (list, tuple))
        or not isinstance(eligibility, (list, tuple))
        or not isinstance(classification_lineage, (list, tuple))
    ):
        raise TypeError("Google downstream inputs must be ordered sequences")
    if any(not isinstance(item, CandidateLocationV2) for item in locations):
        raise TypeError("invalid candidate location envelope")
    if any(not isinstance(item, EligibilityDecisionV2) for item in eligibility):
        raise TypeError("invalid eligibility record")
    if any(not isinstance(item, dict) for item in classification_lineage):
        raise TypeError("invalid classification lineage record")
    location_ids = [item.place_id for item in locations]
    decision_ids = [item.place_id for item in eligibility]
    if len(location_ids) != len(set(location_ids)):
        raise ValueError("downstream Place IDs must be unique")
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("eligibility Place IDs must be unique")
    lineage_ids = [item.get("place_id") for item in classification_lineage]
    if len(lineage_ids) != len(set(lineage_ids)):
        raise ValueError("classification lineage Place IDs must be unique")
    if set(location_ids) != set(decision_ids) or set(location_ids) != set(lineage_ids):
        raise ValueError("location, eligibility, and classification Place-ID sets differ")
    decisions = {item.place_id: item for item in eligibility}
    lineage = {item["place_id"]: item for item in classification_lineage}
    rows: list[dict[str, Any]] = []
    for location in locations:
        decision = decisions[location.place_id]
        classified = lineage[location.place_id]
        if location.candidate_id != decision.candidate_id:
            raise ValueError("location and eligibility candidate IDs differ")
        required_lineage = {
            "place_id", "candidate_id", "candidate_snapshot_sha256",
            "classification_decision_sha256",
        }
        if not required_lineage <= set(classified):
            raise ValueError("classification lineage is incomplete")
        if (
            classified["candidate_id"] != location.candidate_id
            or classified["candidate_snapshot_sha256"]
            != location.candidate_snapshot_sha256
            or classified["classification_decision_sha256"]
            != decision.classification_decision_sha256
            or not isinstance(classified["candidate_snapshot_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", classified["candidate_snapshot_sha256"])
            is None
        ):
            raise ValueError("downstream snapshot lineage does not match classification")
        rows.append({
            "place_id": location.place_id,
            "candidate_id": location.candidate_id,
            "Company": location.name,
            "Street_Address": location.formatted_address,
            # `Zip_Code` is present solely for the existing score_matrix interface.
            # It is provider-returned or blank; query ZIP is retained separately.
            "Zip_Code": location.returned_postal_code or "",
            "zip5": location.returned_postal_code or "",
            "query_zip": location.query_provenance.source_zip,
            "query_definition_id": location.query_provenance.query_definition_id,
            "Latitude": location.latitude,
            "Longitude": location.longitude,
            "retrieved_at": location.retrieved_at.isoformat().replace("+00:00", "Z"),
            "eligibility_state": decision.eligibility_state,
            "eligibility_reason_codes": decision.reason_codes,
            "classification_decision_sha256": decision.classification_decision_sha256,
            "candidate_snapshot_sha256": location.candidate_snapshot_sha256,
            "eligibility_contract_version": decision.contract_version,
            "eligibility_contract_sha256": decision.contract_sha256,
        })
    columns = (
        "place_id", "candidate_id", "Company", "Street_Address", "Zip_Code",
        "zip5", "query_zip", "query_definition_id", "Latitude", "Longitude",
        "retrieved_at", "eligibility_state", "eligibility_reason_codes",
        "classification_decision_sha256", "candidate_snapshot_sha256",
        "eligibility_contract_version", "eligibility_contract_sha256",
    )
    result = pd.DataFrame(rows, columns=columns).sort_values("place_id").reset_index(drop=True)
    if not result.empty and result["place_id"].duplicated().any():
        raise ValueError("prepared Google frame has duplicate Place IDs")
    return result


def merge_v2_board_sources(
    board_raw: pd.DataFrame,
    geocode_cache: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild the active-Minnesota Board source before any ZIP-frame filter."""
    if not isinstance(board_raw, pd.DataFrame) or not isinstance(
        geocode_cache, pd.DataFrame
    ):
        raise TypeError("Board raw source and geocode cache must be DataFrames")
    raw_required = {
        "facility_name", "license_nbr", "description", "address_line1",
        "city", "state", "zip",
    }
    cache_required = {
        "facility_name", "license_nbr", "address_line1", "city", "zip5",
        "_lat", "_lon", "match", "mtype",
    }
    if missing := raw_required - set(board_raw.columns):
        raise ValueError(f"raw Board source lacks columns: {sorted(missing)}")
    if missing := cache_required - set(geocode_cache.columns):
        raise ValueError(f"Board geocode cache lacks columns: {sorted(missing)}")
    raw = board_raw.copy(deep=True)
    cache = geocode_cache.copy(deep=True)
    raw["license_nbr"] = raw["license_nbr"].map(_license)
    raw["zip5"] = raw["zip"].map(_zip5)
    raw = raw.loc[
        raw["description"].astype(str).str.startswith("Active", na=False)
        & raw["state"].astype(str).str.strip().str.upper().eq("MN")
    ].copy()
    cache["license_nbr"] = cache["license_nbr"].map(_license)
    cache["zip5"] = cache["zip5"].map(_zip5)
    for label, frame in (("active Board source", raw), ("Board geocode cache", cache)):
        if frame["license_nbr"].eq("").any() or frame["license_nbr"].duplicated().any():
            raise ValueError(f"{label} license numbers must be nonempty and unique")
    raw_ids = set(raw["license_nbr"])
    cache_ids = set(cache["license_nbr"])
    if extra := sorted(cache_ids - raw_ids):
        raise ValueError(f"Board geocode cache contains non-active/non-MN licenses: {extra[:10]}")
    raw_lookup = raw.set_index("license_nbr")
    cache_lookup = cache.set_index("license_nbr")
    for column in ("facility_name", "address_line1", "city", "zip5"):
        left = raw_lookup.loc[cache_lookup.index, column].fillna("").astype(str).str.strip()
        right = cache_lookup[column].fillna("").astype(str).str.strip()
        if not left.eq(right).all():
            bad = left.index[~left.eq(right)].tolist()[:10]
            raise ValueError(f"Board cache differs from raw {column} for licenses {bad}")
    raw_allowed = [
        "facility_name", "license_nbr", "description", "address_line1",
        "city", "state", "zip5",
    ]
    merged = raw[raw_allowed].merge(
        cache[["license_nbr", "_lat", "_lon", "match", "mtype"]],
        on="license_nbr", how="left", validate="one_to_one",
    )
    return merged.sort_values("license_nbr").reset_index(drop=True)


def prepare_v2_board_frames(
    board: pd.DataFrame,
    target_zips: Iterable[str],
    observed_google_zips: Iterable[str] = (),
) -> BoardFramesV2:
    """Create pre-linkage screened/all-active Board frames from one source table."""
    if not isinstance(board, pd.DataFrame):
        raise TypeError("Board source must be a DataFrame")
    required = {
        "facility_name", "license_nbr", "description", "address_line1",
        "city", "state", "_lat", "_lon", "match", "mtype",
    }
    if missing := required - set(board.columns):
        raise ValueError(f"Board source lacks required columns: {sorted(missing)}")
    if "zip5" not in board and "zip" not in board:
        raise ValueError("Board source lacks a postal-code column")
    target = {_zip5(value) for value in target_zips}
    observed = {_zip5(value) for value in observed_google_zips}
    if "" in target or not target:
        raise ValueError("target ZIP frame is invalid")
    observed.discard("")
    extended = target | observed

    source_zip_column = "zip5" if "zip5" in board else "zip"
    # Explicit allowlist prevents source email and unrelated registration fields from
    # drifting into downstream/private artifacts.
    frame = pd.DataFrame({
        "facility_name": board["facility_name"],
        "license_nbr": board["license_nbr"],
        "description": board["description"],
        "address_line1": board["address_line1"],
        "city": board["city"],
        "state": board["state"],
        "zip5": board[source_zip_column],
        "_lat": board["_lat"],
        "_lon": board["_lon"],
        "match": board["match"],
        "mtype": board["mtype"],
    }).copy(deep=True)
    frame["license_nbr"] = frame["license_nbr"].map(_license)
    if frame["license_nbr"].eq("").any() or frame["license_nbr"].duplicated().any():
        raise ValueError("Board license numbers must be nonempty and unique")
    frame["zip5"] = frame["zip5"].map(_zip5)
    active_mask = (
        frame["description"].astype(str).str.startswith("Active", na=False)
        & frame["state"].astype(str).str.strip().str.upper().eq("MN")
    )
    frame = frame.loc[active_mask].copy()
    frame["eligibility_exclusion_reason"] = frame["facility_name"].map(
        board_exclusion_reason
    )
    frame["access_eligible"] = frame["eligibility_exclusion_reason"].eq("")
    frame["_lat"] = pd.to_numeric(frame["_lat"], errors="coerce")
    frame["_lon"] = pd.to_numeric(frame["_lon"], errors="coerce")
    frame["coordinate_complete"] = frame[["_lat", "_lon"]].notna().all(axis=1)
    frame["coordinate_valid"] = (
        frame["coordinate_complete"]
        &
        frame["_lat"].between(-90.0, 90.0)
        & frame["_lon"].between(-180.0, 180.0)
    )
    frame["has_geocode"] = frame["coordinate_valid"]
    frame = frame.sort_values("license_nbr").reset_index(drop=True)
    target_frame = frame.loc[frame["zip5"].isin(target)].copy().reset_index(drop=True)
    extended_frame = frame.loc[frame["zip5"].isin(extended)].copy().reset_index(drop=True)
    screened_target = target_frame.loc[target_frame["access_eligible"]].copy().reset_index(drop=True)
    screened_extended = extended_frame.loc[extended_frame["access_eligible"]].copy().reset_index(drop=True)
    audit = {
        "adapter_version": LINKAGE_ADAPTER_VERSION,
        "target_zip_count": len(target),
        "extended_zip_count": len(extended),
        "source_rows": len(board),
        "active_mn_rows": len(frame),
        "full_active_target_rows": len(target_frame),
        "screened_target_rows": len(screened_target),
        "full_active_extended_rows": len(extended_frame),
        "screened_extended_rows": len(screened_extended),
        "screen_exclusion_counts": {
            str(key): int(value)
            for key, value in target_frame.loc[
                ~target_frame["access_eligible"], "eligibility_exclusion_reason"
            ].value_counts().items()
        },
    }
    return BoardFramesV2(
        target_frame, screened_target, extended_frame, screened_extended, audit,
        tuple(sorted(target)), tuple(sorted(extended)),
    )


_STRATA = (
    ("include_screened_primary", ("include",), "screened_target"),
    ("include_all_active_sensitivity", ("include",), "full_active_target"),
    ("unresolved_screened_diagnostic", ("unresolved",), "screened_target"),
    ("exclude_screened_diagnostic", ("exclude",), "screened_target"),
    (
        "include_unresolved_screened_conditional",
        ("include", "unresolved"),
        "screened_target",
    ),
    (
        "all_candidates_screened_retrieval",
        ("include", "unresolved", "exclude"),
        "screened_target",
    ),
)


def _pairs(
    google: pd.DataFrame,
    board: pd.DataFrame,
    config: LinkageConfigV2,
    *,
    threshold: float | None = None,
    objective: str | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    if google.empty or board.empty:
        return np.empty((len(google), len(board))), []
    matrix = geo_match.score_matrix(
        google,
        board,
        {"w_name": config.name_weight, "w_addr": config.address_weight},
    )
    result = geo_match.assign_hungarian(
        matrix,
        config.threshold if threshold is None else threshold,
        objective=config.primary_objective if objective is None else objective,
    )
    return matrix, result


def run_v2_board_linkage(
    google: pd.DataFrame,
    boards: BoardFramesV2,
    config: LinkageConfigV2,
) -> BoardLinkageResultV2:
    """Run every declared eligibility stratum as an independent assignment."""
    if not isinstance(google, pd.DataFrame):
        raise TypeError("Google linkage input must be a DataFrame")
    if not isinstance(boards, BoardFramesV2) or not isinstance(config, LinkageConfigV2):
        raise TypeError("invalid Board linkage contract")
    required_google = {
        "place_id", "candidate_id", "Company", "Street_Address", "Zip_Code",
        "Latitude", "Longitude", "eligibility_state", "eligibility_reason_codes",
    }
    if missing := required_google - set(google.columns):
        raise ValueError(f"Google linkage frame lacks columns: {sorted(missing)}")
    if google["place_id"].duplicated().any() or any(
        not isinstance(item, str) or _PLACE_ID.fullmatch(item) is None
        for item in google["place_id"]
    ):
        raise ValueError("Google linkage Place IDs must be valid and unique")
    if not set(google["eligibility_state"]) <= {"include", "exclude", "unresolved"}:
        raise ValueError("Google linkage frame has invalid eligibility state")
    if not boards.target_zips:
        raise ValueError("Board linkage frames lack the fixed target ZIP contract")
    target_zips = set(boards.target_zips)
    returned_zip = google["Zip_Code"].fillna("").astype(str)
    in_target = returned_zip.isin(target_zips)
    missing_returned_zip = returned_zip.eq("")
    fixed_google = google.loc[in_target].copy()

    link_rows: list[dict[str, Any]] = []
    unmatched_google_rows: list[dict[str, Any]] = []
    unmatched_board_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []

    for stratum, states, board_attribute in _STRATA:
        left = fixed_google.loc[fixed_google["eligibility_state"].isin(states)].copy()
        left = left.sort_values("place_id").reset_index(drop=True)
        right = getattr(boards, board_attribute).copy()
        right = right.sort_values("license_nbr").reset_index(drop=True)
        matrix, pairs = _pairs(left, right, config)
        matched_left = {left_index for left_index, _, _ in pairs}
        matched_right = {right_index for _, right_index, _ in pairs}
        for left_index, right_index, score in pairs:
            google_row = left.iloc[left_index]
            board_row = right.iloc[right_index]
            link_rows.append({
                "stratum": stratum,
                "place_id": google_row["place_id"],
                "candidate_id": google_row["candidate_id"],
                "eligibility_state": google_row["eligibility_state"],
                "license_nbr": board_row["license_nbr"],
                "composite_score": round(float(score), 4),
                "threshold": config.threshold,
                "objective": config.primary_objective,
            })
        for index, row in left.iterrows():
            if index not in matched_left:
                unmatched_google_rows.append({
                    "stratum": stratum,
                    "place_id": row["place_id"],
                    "candidate_id": row["candidate_id"],
                    "eligibility_state": row["eligibility_state"],
                    "reason_codes": list(row["eligibility_reason_codes"]),
                })
        for index, row in right.iterrows():
            if index not in matched_right:
                unmatched_board_rows.append({
                    "stratum": stratum,
                    "license_nbr": row["license_nbr"],
                })
        linked = len(pairs)
        summaries.append({
            "stratum": stratum,
            "google_states": list(states),
            "board_frame": board_attribute,
            "google_rows": len(left),
            "board_rows": len(right),
            "linked_pairs": linked,
            "unmatched_google_rows": len(left) - linked,
            "unmatched_board_rows": len(right) - linked,
            "board_recovery_pct": _percent(linked, len(right)),
            "google_link_agreement_pct": _percent(linked, len(left)),
        })
        if stratum == "include_screened_primary":
            for threshold in config.threshold_sensitivity:
                _, sensitivity_pairs = _pairs(
                    left, right, config, threshold=threshold
                )
                threshold_rows.append({
                    "threshold": threshold,
                    "linked_pairs": len(sensitivity_pairs),
                    "board_recovery_pct": _percent(len(sensitivity_pairs), len(right)),
                    "google_link_agreement_pct": _percent(len(sensitivity_pairs), len(left)),
                })
            for objective in (config.primary_objective, config.sensitivity_objective):
                _, sensitivity_pairs = _pairs(
                    left, right, config, objective=objective
                )
                objective_rows.append({
                    "objective": objective,
                    "linked_pairs": len(sensitivity_pairs),
                    "total_similarity": round(
                        sum(score for _, _, score in sensitivity_pairs), 4
                    ),
                })

    links = pd.DataFrame(link_rows, columns=(
        "stratum", "place_id", "candidate_id", "eligibility_state",
        "license_nbr", "composite_score", "threshold", "objective",
    ))
    unmatched_google = pd.DataFrame(unmatched_google_rows, columns=(
        "stratum", "place_id", "candidate_id", "eligibility_state", "reason_codes",
    ))
    unmatched_board = pd.DataFrame(
        unmatched_board_rows, columns=("stratum", "license_nbr")
    )
    summary = {
        "adapter_version": LINKAGE_ADAPTER_VERSION,
        "configuration": {
            "score": "name_address_weighted_mean_0_100_cross_zip_times_0_85",
            "name_weight": config.name_weight,
            "address_weight": config.address_weight,
            "threshold": config.threshold,
            "primary_objective": config.primary_objective,
            "sensitivity_objective": config.sensitivity_objective,
        },
        "board_frames": dict(boards.audit),
        "google_state_counts": {
            str(key): int(value)
            for key, value in fixed_google["eligibility_state"].value_counts().items()
        },
        "google_frame_qa": {
            "retrieved_rows": len(google),
            "in_target_rows": int(in_target.sum()),
            "outside_target_rows": int((~in_target & ~missing_returned_zip).sum()),
            "missing_returned_zip_rows": int(missing_returned_zip.sum()),
            "membership_basis": "provider_returned_postal_code",
        },
        "strata": summaries,
        "primary_threshold_sensitivity": threshold_rows,
        "primary_objective_sensitivity": objective_rows,
        "independence_note": (
            "Each stratum is assigned independently; matched counts must not be summed "
            "to reconstruct another stratum."
        ),
    }
    return BoardLinkageResultV2(links, unmatched_google, unmatched_board, summary)


__all__ = [
    "LINKAGE_ADAPTER_VERSION",
    "BoardFramesV2",
    "BoardLinkageResultV2",
    "LinkageConfigV2",
    "merge_v2_board_sources",
    "prepare_v2_board_frames",
    "prepare_v2_google_frame",
    "run_v2_board_linkage",
]
