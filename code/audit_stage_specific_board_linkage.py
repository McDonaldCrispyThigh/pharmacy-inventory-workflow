"""Local stage-specific Board-linkage diagnostic for frozen run 20260902T012636Z.

This script reads authenticated local artifacts only.  It does not call a provider,
change a classification, replace an existing result, or run a spatial analysis.
Local audit outputs retain only identifiers already present in the safe delivery plus
derived scores, hashes, classifications, and aggregate counts; private names and
addresses are used transiently for the declared linkage calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import geo_match  # noqa: E402
from study_design import (  # noqa: E402
    TARGET_ZIPS,
    board_exclusion_reason,
    normalized_location_key,
)


RUN_ID = "20260902T012636Z"
RUN_ROOT = ROOT / "data" / "v2" / RUN_ID
DELIVERY = RUN_ROOT / "delivery"
DEFAULT_OUTPUT = (
    ROOT
    / "analysis"
    / "v2"
    / f"board_linkage_stage_diagnostic_{RUN_ID}_v5"
)
PRIMARY_THRESHOLD = 75.0
NAME_WEIGHT = 1.0
ADDRESS_WEIGHT = 0.8
PRIMARY_OBJECTIVE = "max_weight"
PRIMARY_STRATUM = "include_screened_primary"
RAW_STRATUM = "all_candidates_screened_retrieval"

EXPECTED_HASHES = {
    "run_manifest": "5f0463e7fe99c942f9a0854bf20a6bd0a741b62d75542df1a41ed347350da9f6",
    "delivery_manifest": "e7138170e740647993afcdae42a02c5171c0f8292db04c0ddae73debeafa1d78",
    "classification": "552283d3f9c100844a674e21dab528e2c93b37411cf57482e24f389c66f36c3d",
    "eligibility": "3a5eaff83b32e5bff49510c2972f450fee770de0d7472056d047c73071d32513",
    "candidate_inventory": "3002774457794ab19908b3da0f1b940473f59c082d361008f395a0732cf6d824",
    "board_raw": "97c94e1c82f6982b174864512ecd0ee3dac4da46208fee890bfb8f2aa388100b",
    "board_cache": "aaea04df2f5131e84a4f08f4179ad1bcd2e0319122e2d0dce8d38a14e161cd6e",
    "board_links": "0fcd836ff3f5d66d380891ad395e4115a7534fe062c80cd301dec9e9e7bd53ef",
    "unmatched_board": "99a04b470e642e5e954c5d1df696e1c3c4c2d2d8b132571ddf999b9f1f7f09ac",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_hash(value: object, *, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}:{value}".encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(numerator: int, denominator: int) -> float | None:
    return round(100.0 * numerator / denominator, 2) if denominator else None


def numeric_summary(values: Sequence[float] | pd.Series) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {
            key: None
            for key in ("n", "minimum", "q1", "median", "mean", "q3", "maximum")
        }
    return {
        "n": int(len(series)),
        "minimum": round(float(series.min()), 6),
        "q1": round(float(series.quantile(0.25)), 6),
        "median": round(float(series.median()), 6),
        "mean": round(float(series.mean()), 6),
        "q3": round(float(series.quantile(0.75)), 6),
        "maximum": round(float(series.max()), 6),
    }


def distribution(values: Iterable[object]) -> dict[str, dict[str, Any]]:
    counter = Counter(str(value) for value in values)
    denominator = sum(counter.values())
    return {
        key: {"count": int(value), "pct": pct(int(value), denominator)}
        for key, value in sorted(counter.items())
    }


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


def assert_expected_hashes() -> dict[str, str]:
    paths = {
        "run_manifest": RUN_ROOT / "run_manifest.json",
        "delivery_manifest": DELIVERY / "run_manifest.json",
        "classification": RUN_ROOT / "classification_forced_choice.jsonl",
        "eligibility": DELIVERY / "eligibility.csv",
        "candidate_inventory": DELIVERY / "candidate_inventory.csv",
        "board_raw": ROOT / "data" / "Board" / "F.1  Pharmacy (PH).csv",
        "board_cache": ROOT / "data" / "board_geocoded.csv",
        "board_links": DELIVERY / "board_links.csv",
        "unmatched_board": DELIVERY / "unmatched_board.csv",
    }
    actual: dict[str, str] = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is missing: {path}")
        actual[key] = sha256_file(path)
        if actual[key] != EXPECTED_HASHES[key]:
            raise ValueError(
                f"frozen input hash changed for {key}: "
                f"{actual[key]} != {EXPECTED_HASHES[key]}"
            )
    return actual


def authenticate_input_manifest() -> dict[str, str]:
    path = ROOT / "analysis" / "input_manifest.json"
    manifest = read_json(path)
    entries = manifest.get("inputs")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("invalid frozen input manifest")
    wanted = {
        "data/Board/F.1  Pharmacy (PH).csv",
        "data/board_geocoded.csv",
    }
    by_path = {str(item.get("path")): item for item in entries if isinstance(item, dict)}
    output = {"analysis/input_manifest.json": sha256_file(path)}
    for relative in sorted(wanted):
        entry = by_path.get(relative)
        source = ROOT / relative
        if not isinstance(entry, dict) or not source.is_file():
            raise ValueError(f"frozen Board input is undeclared or missing: {relative}")
        actual = sha256_file(source)
        if entry.get("sha256") != actual:
            raise ValueError(f"frozen Board input failed authentication: {relative}")
        output[relative] = actual
    return output


def load_candidates() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = read_json(RUN_ROOT / "run_manifest.json")
    declared = manifest.get("completed_unit_hashes")
    completed_root = RUN_ROOT / "private" / "completed_candidates"
    if not isinstance(declared, dict) or len(declared) != 1226:
        raise ValueError("completed-unit manifest does not contain 1,226 hashes")
    files = sorted(completed_root.glob("*.json"))
    if len(files) != 1226 or {path.name for path in files} != set(declared):
        raise ValueError("private completed-unit file set differs from the manifest")

    required_confidence = {
        "target_construct_status",
        "business_status",
        "candidate_plausibility",
        "identity_consistency",
    }
    rows: list[dict[str, Any]] = []
    missing = Counter()
    unit_hashes: dict[str, str] = {}
    for path in files:
        actual_hash = sha256_file(path)
        unit_hashes[path.name] = actual_hash
        if actual_hash != declared[path.name]:
            raise ValueError(f"completed unit failed hash authentication: {path.name}")
        unit = read_json(path)
        location = unit.get("location")
        decision = unit.get("decision")
        if not isinstance(location, dict) or not isinstance(decision, dict):
            raise ValueError(f"completed unit lacks location or decision: {path.name}")
        confidence = decision.get("claim_confidence")
        if not isinstance(confidence, dict) or not required_confidence <= set(confidence):
            raise ValueError(f"completed unit lacks field confidence: {path.name}")
        bindings = {
            str(unit.get("place_id")),
            str(location.get("place_id")),
            str(decision.get("place_id", unit.get("place_id"))),
        }
        candidate_bindings = {
            str(unit.get("candidate_id")),
            str(location.get("candidate_id")),
            str(decision.get("candidate_id")),
        }
        if len(bindings) != 1 or len(candidate_bindings) != 1:
            raise ValueError(f"completed-unit identifier binding failed: {path.name}")
        if (
            unit.get("model") != manifest.get("model")
            or unit.get("returned_model") not in manifest.get("promoted_returned_models", [])
            or unit.get("fixed_frame_sha256") != manifest.get("fixed_frame_sha256")
            or unit.get("prompt_template_sha256") != manifest.get("prompt_template_sha256")
            or unit.get("scientific_implementation_sha256")
            != manifest.get("scientific_implementation_sha256")
        ):
            raise ValueError(f"completed-unit frozen contract mismatch: {path.name}")
        values = {
            "place_id": unit.get("place_id"),
            "candidate_id": unit.get("candidate_id"),
            "public_name": location.get("name"),
            "returned_address": location.get("formatted_address"),
            "returned_zip": location.get("returned_postal_code"),
            "target_construct_status": decision.get("target_construct_status"),
            "business_status": decision.get("business_status"),
            "candidate_plausibility": decision.get("candidate_plausibility"),
            "identity_consistency": decision.get("identity_consistency"),
        }
        for field, value in values.items():
            if value is None or not str(value).strip():
                missing[field] += 1
        returned_zip = str(location.get("returned_postal_code") or "").strip()
        if re.fullmatch(r"\d{5}", returned_zip) is None:
            missing["returned_zip_not_exactly_five_digits"] += 1
        rows.append(
            {
                "place_id": str(unit["place_id"]),
                "candidate_id": str(unit["candidate_id"]),
                "Company": str(location.get("name") or ""),
                "Street_Address": str(location.get("formatted_address") or ""),
                "Zip_Code": returned_zip,
                "Latitude": pd.to_numeric(location.get("latitude"), errors="coerce"),
                "Longitude": pd.to_numeric(location.get("longitude"), errors="coerce"),
                "target_construct_status": str(decision.get("target_construct_status") or ""),
                "business_status": str(decision.get("business_status") or ""),
                "candidate_plausibility": str(decision.get("candidate_plausibility") or ""),
                "identity_consistency": str(decision.get("identity_consistency") or ""),
                "target_construct_status_confidence": str(
                    confidence["target_construct_status"]
                ),
                "business_status_confidence": str(confidence["business_status"]),
                "candidate_plausibility_confidence": str(
                    confidence["candidate_plausibility"]
                ),
                "identity_consistency_confidence": str(
                    confidence["identity_consistency"]
                ),
                "classification_decision_sha256": str(
                    unit.get("classification_decision_sha256") or ""
                ),
            }
        )
    if missing:
        raise ValueError(f"required local linkage fields are missing: {dict(missing)}")

    frame = pd.DataFrame(rows).sort_values("place_id").reset_index(drop=True)
    if (
        len(frame) != 1226
        or frame["place_id"].duplicated().any()
        or frame["candidate_id"].duplicated().any()
    ):
        raise ValueError("authenticated candidate identifiers are incomplete or duplicated")

    fixed_rows = read_jsonl(RUN_ROOT / "place_ids.jsonl")
    fixed_ids = [str(row.get("place_id") or "") for row in fixed_rows]
    if len(fixed_ids) != 1226 or len(set(fixed_ids)) != 1226:
        raise ValueError("frozen Place-ID frame is incomplete or duplicated")
    if set(fixed_ids) != set(frame["place_id"]):
        raise ValueError("completed units differ from the frozen Place-ID frame")

    classifications = read_jsonl(RUN_ROOT / "classification_forced_choice.jsonl")
    if len(classifications) != 1226:
        raise ValueError("frozen classification artifact does not contain 1,226 rows")
    classified = {str(row.get("place_id")): row for row in classifications}
    if len(classified) != 1226 or set(classified) != set(frame["place_id"]):
        raise ValueError("frozen classification identifiers differ from candidates")
    core_fields = (
        "target_construct_status",
        "business_status",
        "candidate_plausibility",
        "identity_consistency",
    )
    for row in frame.itertuples(index=False):
        frozen = classified[row.place_id]
        if (
            str(frozen.get("candidate_id")) != row.candidate_id
            or str(frozen.get("classification_decision_sha256"))
            != row.classification_decision_sha256
            or any(str(frozen.get(field)) != str(getattr(row, field)) for field in core_fields)
        ):
            raise ValueError(f"classification binding failed for {row.place_id}")

    eligibility = pd.read_csv(DELIVERY / "eligibility.csv", dtype=str).fillna("")
    if (
        len(eligibility) != 1226
        or eligibility["place_id"].duplicated().any()
        or set(eligibility["place_id"]) != set(frame["place_id"])
    ):
        raise ValueError("frozen eligibility artifact is incomplete")
    eligibility = eligibility.set_index("place_id")
    frame = frame.join(
        eligibility[["eligibility_state", "reason_codes"]],
        on="place_id",
        validate="one_to_one",
    )
    if not set(frame["eligibility_state"]) <= {"include", "exclude"}:
        raise ValueError("the final frozen reducer contains a non-binary eligibility state")
    for row in frame.itertuples(index=False):
        frozen = classified[row.place_id]
        if str(frozen.get("classification_decision_sha256")) != str(
            eligibility.loc[row.place_id, "classification_decision_sha256"]
        ):
            raise ValueError(f"eligibility/classification binding failed: {row.place_id}")

    safe_inventory = pd.read_csv(
        DELIVERY / "candidate_inventory.csv", dtype=str
    ).fillna("")
    if len(safe_inventory) != 1226 or safe_inventory["place_id"].duplicated().any():
        raise ValueError("safe candidate inventory is incomplete")
    safe_inventory = safe_inventory.set_index("place_id")
    for row in frame.itertuples(index=False):
        safe = safe_inventory.loc[row.place_id]
        if (
            str(safe["returned_postal_code"]) != row.Zip_Code
            or str(safe["eligibility_state"]) != row.eligibility_state
            or str(safe["candidate_id"]) != row.candidate_id
        ):
            raise ValueError(f"safe inventory binding failed: {row.place_id}")

    qa = {
        "candidate_rows": len(frame),
        "unique_place_ids": int(frame["place_id"].nunique()),
        "unique_candidate_ids": int(frame["candidate_id"].nunique()),
        "missing_required_field_counts": {
            "place_id": 0,
            "public_name": 0,
            "returned_address": 0,
            "five_digit_returned_zip": 0,
            "frozen_eligibility_decision": 0,
        },
        "all_completed_unit_hashes_match": len(unit_hashes) == 1226,
        "fixed_frame_identifier_set_matches": True,
        "classification_identifier_and_decision_hashes_match": True,
        "eligibility_identifier_and_decision_hashes_match": True,
        "safe_inventory_zip_and_state_match": True,
        "model": manifest.get("model"),
        "returned_models": manifest.get("promoted_returned_models"),
        "fixed_frame_sha256": manifest.get("fixed_frame_sha256"),
        "prompt_template_version": manifest.get("prompt_template_version"),
        "prompt_template_sha256": manifest.get("prompt_template_sha256"),
        "scientific_implementation_sha256": manifest.get(
            "scientific_implementation_sha256"
        ),
        "completed_units_sha256": manifest.get("completed_units_sha256"),
    }
    return frame, qa


def load_screened_board() -> tuple[pd.DataFrame, dict[str, Any]]:
    authenticate_input_manifest()
    raw = pd.read_csv(
        ROOT / "data" / "Board" / "F.1  Pharmacy (PH).csv",
        encoding="utf-8-sig",
        skiprows=3,
        low_memory=False,
    )
    cache = pd.read_csv(ROOT / "data" / "board_geocoded.csv", low_memory=False)
    required_raw = {
        "facility_name",
        "license_nbr",
        "description",
        "address_line1",
        "city",
        "state",
        "zip",
    }
    required_cache = {
        "facility_name",
        "license_nbr",
        "address_line1",
        "city",
        "zip5",
        "_lat",
        "_lon",
        "match",
        "mtype",
    }
    if required_raw - set(raw) or required_cache - set(cache):
        raise ValueError("Board source or geocode cache schema changed")
    raw = raw.copy()
    cache = cache.copy()
    raw["license_nbr"] = raw["license_nbr"].map(_license)
    raw["zip5"] = raw["zip"].map(_zip5)
    raw = raw.loc[
        raw["description"].astype(str).str.startswith("Active", na=False)
        & raw["state"].astype(str).str.strip().str.upper().eq("MN")
    ].copy()
    cache["license_nbr"] = cache["license_nbr"].map(_license)
    cache["zip5"] = cache["zip5"].map(_zip5)
    for label, table in (("active Board source", raw), ("Board geocode cache", cache)):
        if table["license_nbr"].eq("").any() or table["license_nbr"].duplicated().any():
            raise ValueError(f"{label} license identifiers are empty or duplicated")
    if set(cache["license_nbr"]) - set(raw["license_nbr"]):
        raise ValueError("Board cache contains a non-active or non-Minnesota license")
    raw_lookup = raw.set_index("license_nbr")
    cache_lookup = cache.set_index("license_nbr")
    for column in ("facility_name", "address_line1", "city", "zip5"):
        left = raw_lookup.loc[cache_lookup.index, column].fillna("").astype(str).str.strip()
        right = cache_lookup[column].fillna("").astype(str).str.strip()
        if not left.eq(right).all():
            raise ValueError(f"Board cache differs from source field {column}")
    merged = raw[
        [
            "facility_name",
            "license_nbr",
            "description",
            "address_line1",
            "city",
            "state",
            "zip5",
        ]
    ].merge(
        cache[["license_nbr", "_lat", "_lon", "match", "mtype"]],
        on="license_nbr",
        how="left",
        validate="one_to_one",
    )
    merged["eligibility_exclusion_reason"] = merged["facility_name"].map(
        board_exclusion_reason
    )
    target = merged.loc[merged["zip5"].isin(TARGET_ZIPS)].copy()
    target = target.sort_values("license_nbr").reset_index(drop=True)
    screened = target.loc[target["eligibility_exclusion_reason"].eq("")].copy()
    screened = screened.sort_values("license_nbr").reset_index(drop=True)
    exclusions = target.loc[~target["eligibility_exclusion_reason"].eq("")]
    audit = {
        "active_minnesota_rows": len(merged),
        "full_active_60_zip_rows": len(target),
        "screened_60_zip_rows": len(screened),
        "exclusion_counts": {
            str(key): int(value)
            for key, value in exclusions["eligibility_exclusion_reason"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    if audit != {
        "active_minnesota_rows": 1110,
        "full_active_60_zip_rows": 245,
        "screened_60_zip_rows": 178,
        "exclusion_counts": {
            "institutional_or_hospital": 64,
            "noncommunity_service": 3,
        },
    }:
        raise ValueError(f"screened Board frame changed: {audit}")
    return screened, audit


def score_with_components(
    candidates: pd.DataFrame,
    board: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = candidates.reset_index(drop=True)
    board = board.reset_index(drop=True)
    geo_match.set_parser("regex")
    matrix = geo_match.score_matrix(
        candidates,
        board,
        {"w_name": NAME_WEIGHT, "w_addr": ADDRESS_WEIGHT},
    )
    name_scores = np.zeros_like(matrix)
    address_scores = np.zeros_like(matrix)
    zip_multipliers = np.ones_like(matrix)
    board_names = board["facility_name"].astype(str).tolist()
    board_addresses = board["address_line1"].astype(str).tolist()
    board_zips = board["zip5"].astype(str).tolist()
    for left_index, candidate in candidates.iterrows():
        candidate_names = candidate.get("candidate_names", candidate["Company"])
        candidate_address = str(candidate["Street_Address"])
        candidate_zip = str(candidate["Zip_Code"])[:5]
        for right_index in range(len(board)):
            name_score = geo_match.candidate_name_sim(
                candidate_names, board_names[right_index]
            )
            address_score = geo_match.address_sim(
                candidate_address, board_addresses[right_index]
            )
            multiplier = (
                0.85
                if candidate_zip
                and board_zips[right_index]
                and candidate_zip != board_zips[right_index]
                else 1.0
            )
            name_scores[left_index, right_index] = name_score
            address_scores[left_index, right_index] = address_score
            zip_multipliers[left_index, right_index] = multiplier
    independently_reconstructed = (
        (NAME_WEIGHT * name_scores + ADDRESS_WEIGHT * address_scores)
        / (NAME_WEIGHT + ADDRESS_WEIGHT)
        * zip_multipliers
    )
    independently_reconstructed = np.clip(independently_reconstructed, 0.0, 100.0)
    if not np.allclose(matrix, independently_reconstructed, rtol=0.0, atol=1e-12):
        maximum_difference = float(np.max(np.abs(matrix - independently_reconstructed)))
        raise ValueError(
            "shared score_matrix differs from independently reconstructed declared score: "
            f"max difference {maximum_difference}"
        )
    return matrix, name_scores, address_scores, zip_multipliers


def assignment_records(
    candidates: pd.DataFrame,
    board: pd.DataFrame,
    matrix: np.ndarray,
) -> tuple[list[tuple[int, int, float]], list[dict[str, Any]]]:
    pairs = geo_match.assign_hungarian(
        matrix, PRIMARY_THRESHOLD, objective=PRIMARY_OBJECTIVE
    )
    records = []
    for left_index, right_index, score in pairs:
        candidate = candidates.iloc[left_index]
        records.append(
            {
                "left_identifier": str(
                    candidate.get(
                        "place_id", candidate.get("location_group_sha256", "")
                    )
                ),
                "license_nbr": str(board.iloc[right_index]["license_nbr"]),
                "composite_score": round(float(score), 6),
            }
        )
    return pairs, records


def assignment_summary(
    unit_label: str,
    candidates: pd.DataFrame,
    board: pd.DataFrame,
    pairs: Sequence[tuple[int, int, float]],
) -> dict[str, Any]:
    return {
        "candidate_unit": unit_label,
        "candidate_rows": len(candidates),
        "board_rows": len(board),
        "linked_pairs": len(pairs),
        "unmatched_candidate_rows": len(candidates) - len(pairs),
        "unmatched_board_rows": len(board) - len(pairs),
        "candidate_linked_share_pct": pct(len(pairs), len(candidates)),
        "board_reference_coverage_pct": pct(len(pairs), len(board)),
        "score_summary": numeric_summary([score for _, _, score in pairs]),
    }


def verify_existing_assignments(
    board: pd.DataFrame,
    raw_records: Sequence[Mapping[str, Any]],
    included_records: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    existing = pd.read_csv(DELIVERY / "board_links.csv", dtype=str).fillna("")
    expected = {
        "raw": existing.loc[existing["stratum"].eq(RAW_STRATUM)].copy(),
        "included": existing.loc[existing["stratum"].eq(PRIMARY_STRATUM)].copy(),
    }
    observed = {"raw": raw_records, "included": included_records}
    checks: dict[str, bool] = {}
    for label in ("raw", "included"):
        expected_pairs = {
            (str(row.place_id), str(row.license_nbr)): round(
                float(row.composite_score), 4
            )
            for row in expected[label].itertuples(index=False)
        }
        observed_pairs = {
            (str(row["left_identifier"]), str(row["license_nbr"])): round(
                float(row["composite_score"]), 4
            )
            for row in observed[label]
        }
        checks[f"{label}_assignment_pair_set_matches_delivery"] = (
            set(expected_pairs) == set(observed_pairs)
        )
        checks[f"{label}_assignment_rounded_scores_match_delivery"] = (
            expected_pairs == observed_pairs
        )
    expected_board = set(board["license_nbr"].astype(str))
    delivered_primary_board = set(expected["included"]["license_nbr"].astype(str))
    unmatched = pd.read_csv(DELIVERY / "unmatched_board.csv", dtype=str).fillna("")
    delivered_unmatched_board = set(
        unmatched.loc[
            unmatched["stratum"].eq(PRIMARY_STRATUM), "license_nbr"
        ].astype(str)
    )
    checks["delivered_primary_board_identifiers_within_screened_frame"] = (
        delivered_primary_board <= expected_board
    )
    checks["delivered_primary_linked_and_unmatched_partition_screened_board"] = (
        delivered_primary_board.isdisjoint(delivered_unmatched_board)
        and delivered_primary_board | delivered_unmatched_board == expected_board
        and len(delivered_primary_board) == 146
        and len(delivered_unmatched_board) == 32
    )
    if not all(checks.values()):
        raise ValueError(f"recomputed assignments differ from frozen delivery: {checks}")
    return checks


STAGE_LABELS = {
    "A": "no qualified raw candidate edge",
    "B": "qualified raw edge, but no qualified included edge",
    "C": "qualified included edge, but not linked in included assignment",
    "D": "covered in included Hungarian assignment",
}
STAGE_INTERPRETATIONS = {
    "A": "candidate-retrieval and/or linkage-representation gap under the declared rule",
    "B": "LLM-associated attrition conditional on the candidate frame and linkage rule",
    "C": "assignment competition",
    "D": "Board-reference coverage in the final assignment",
}


def board_stage_decomposition(
    board: pd.DataFrame,
    raw_matrix: np.ndarray,
    included_matrix: np.ndarray,
    raw_candidates: pd.DataFrame,
    included_candidates: pd.DataFrame,
    raw_pairs: Sequence[tuple[int, int, float]],
    included_pairs: Sequence[tuple[int, int, float]],
    *,
    unit_term: str = "candidate",
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    raw_qualified = raw_matrix >= PRIMARY_THRESHOLD
    included_qualified = included_matrix >= PRIMARY_THRESHOLD
    raw_assigned_board = {right for _, right, _ in raw_pairs}
    included_assigned_board = {right for _, right, _ in included_pairs}
    identifier_column = (
        "place_id" if "place_id" in raw_candidates else "location_group_sha256"
    )
    raw_identifiers = raw_candidates[identifier_column].astype(str).tolist()
    included_identifiers = included_candidates[identifier_column].astype(str).tolist()
    stage_labels = {
        "A": f"no qualified raw {unit_term} edge",
        "B": f"qualified raw edge, but no qualified included {unit_term} edge",
        "C": f"qualified included {unit_term} edge, but not linked in included assignment",
        "D": "covered in included Hungarian assignment",
    }
    detail_rows: list[dict[str, Any]] = []
    for right_index, board_row in board.iterrows():
        raw_indices = np.flatnonzero(raw_qualified[:, right_index])
        included_indices = np.flatnonzero(included_qualified[:, right_index])
        raw_endpoint_ids = {raw_identifiers[int(index)] for index in raw_indices}
        included_endpoint_ids = {
            included_identifiers[int(index)] for index in included_indices
        }
        removed_endpoint_ids = raw_endpoint_ids - included_endpoint_ids
        if len(raw_indices) == 0:
            stage = "A"
        elif len(included_indices) == 0:
            stage = "B"
        elif right_index not in included_assigned_board:
            stage = "C"
        else:
            stage = "D"
        license_nbr = str(board_row["license_nbr"])
        detail_rows.append(
            {
                "license_nbr": license_nbr,
                "board_record_sha256": safe_hash(
                    license_nbr, namespace="board-license"
                ),
                "stage": stage,
                "stage_label": stage_labels[stage],
                "raw_qualified_edge_count": len(raw_indices),
                "included_qualified_edge_count": len(included_indices),
                "excluded_qualified_endpoint_count": len(removed_endpoint_ids),
                "has_excluded_qualified_endpoint": bool(removed_endpoint_ids),
                "raw_assignment_covered": right_index in raw_assigned_board,
                "included_assignment_covered": right_index in included_assigned_board,
                "maximum_raw_score": (
                    round(float(raw_matrix[raw_indices, right_index].max()), 6)
                    if len(raw_indices)
                    else ""
                ),
                "maximum_included_score": (
                    round(float(included_matrix[included_indices, right_index].max()), 6)
                    if len(included_indices)
                    else ""
                ),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values("license_nbr").reset_index(drop=True)
    summary = []
    for stage in ("A", "B", "C", "D"):
        count = int(detail["stage"].eq(stage).sum())
        summary.append(
            {
                "stage": stage,
                "definition": stage_labels[stage],
                "count": count,
                "pct_of_178_board_records": pct(count, len(board)),
                "interpretation": STAGE_INTERPRETATIONS[stage],
            }
        )
    raw_edge_board = int(raw_qualified.any(axis=0).sum())
    included_edge_board = int(included_qualified.any(axis=0).sum())
    stage_b = int(detail["stage"].eq("B").sum())
    core = {
        "raw_candidate_edge_coverage": {
            "numerator": raw_edge_board,
            "denominator": len(board),
            "pct": pct(raw_edge_board, len(board)),
        },
        "included_candidate_edge_coverage": {
            "numerator": included_edge_board,
            "denominator": len(board),
            "pct": pct(included_edge_board, len(board)),
        },
        "llm_retention_among_raw_retrievable_board_records": {
            "numerator": included_edge_board,
            "denominator": raw_edge_board,
            "pct": pct(included_edge_board, raw_edge_board),
        },
        "llm_associated_attrition_among_raw_retrievable_board_records": {
            "numerator": stage_b,
            "denominator": raw_edge_board,
            "pct": pct(stage_b, raw_edge_board),
        },
        "included_assignment_coverage": {
            "numerator": len(included_pairs),
            "denominator": len(board),
            "pct": pct(len(included_pairs), len(board)),
        },
        "qualified_edge_coverage_is_not_assignment_coverage": True,
    }
    if sum(row["count"] for row in summary) != len(board):
        raise ValueError("Board stages do not partition the screened reference")
    if included_edge_board > raw_edge_board:
        raise ValueError("included-edge Board coverage exceeds raw-edge coverage")
    return summary, detail, core


def board_assignment_identity_comparison(
    board: pd.DataFrame,
    raw_pairs: Sequence[tuple[int, int, float]],
    included_pairs: Sequence[tuple[int, int, float]],
) -> tuple[dict[str, int], pd.DataFrame]:
    raw = {right for _, right, _ in raw_pairs}
    included = {right for _, right, _ in included_pairs}
    detail = []
    labels = Counter()
    for index, row in board.iterrows():
        in_raw = index in raw
        in_included = index in included
        if in_raw and in_included:
            label = "linked in both assignments"
        elif in_raw:
            label = "linked only before LLM/reducer filtering"
        elif in_included:
            label = "linked only after filtering because assignment changed"
        else:
            label = "linked in neither assignment"
        labels[label] += 1
        license_nbr = str(row["license_nbr"])
        detail.append(
            {
                "license_nbr": license_nbr,
                "board_record_sha256": safe_hash(
                    license_nbr, namespace="board-license"
                ),
                "assignment_identity_category": label,
            }
        )
    ordered_labels = (
        "linked in both assignments",
        "linked only before LLM/reducer filtering",
        "linked only after filtering because assignment changed",
        "linked in neither assignment",
    )
    summary = {label: int(labels[label]) for label in ordered_labels}
    if sum(summary.values()) != len(board):
        raise ValueError("Board assignment identities do not partition the reference")
    return summary, pd.DataFrame(detail).sort_values("license_nbr").reset_index(drop=True)


def linked_board_endpoint_stability(
    board: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    included_candidates: pd.DataFrame,
    raw_pairs: Sequence[tuple[int, int, float]],
    included_pairs: Sequence[tuple[int, int, float]],
    *,
    identifier_column: str,
) -> tuple[dict[str, int], pd.DataFrame]:
    raw_by_board = {
        right: str(raw_candidates.iloc[left][identifier_column])
        for left, right, _ in raw_pairs
    }
    included_by_board = {
        right: str(included_candidates.iloc[left][identifier_column])
        for left, right, _ in included_pairs
    }
    shared = sorted(set(raw_by_board) & set(included_by_board))
    changed = [
        right
        for right in shared
        if raw_by_board[right] != included_by_board[right]
    ]
    detail = pd.DataFrame(
        [
            {
                "license_nbr": str(board.iloc[right]["license_nbr"]),
                "board_record_sha256": safe_hash(
                    board.iloc[right]["license_nbr"], namespace="board-license"
                ),
                "raw_assignment_endpoint": raw_by_board[right],
                "included_assignment_endpoint": included_by_board[right],
            }
            for right in changed
        ],
        columns=[
            "license_nbr",
            "board_record_sha256",
            "raw_assignment_endpoint",
            "included_assignment_endpoint",
        ],
    )
    return {
        "board_records_linked_in_both_assignments": len(shared),
        "same_assignment_endpoint": len(shared) - len(changed),
        "changed_assignment_endpoint_after_filtering": len(changed),
    }, detail


def qualified_edge_table(
    candidates: pd.DataFrame,
    board: pd.DataFrame,
    matrix: np.ndarray,
    name_scores: np.ndarray,
    address_scores: np.ndarray,
    zip_multipliers: np.ndarray,
    *,
    identifier_column: str,
    scope: str,
) -> pd.DataFrame:
    rows = []
    for left_index, right_index in np.argwhere(matrix >= PRIMARY_THRESHOLD):
        candidate = candidates.iloc[int(left_index)]
        board_row = board.iloc[int(right_index)]
        row = {
            "candidate_scope": scope,
            identifier_column: str(candidate[identifier_column]),
            "license_nbr": str(board_row["license_nbr"]),
            "name_score": round(float(name_scores[left_index, right_index]), 6),
            "address_score": round(float(address_scores[left_index, right_index]), 6),
            "zip_mismatch_multiplier": round(
                float(zip_multipliers[left_index, right_index]), 2
            ),
            "composite_score": round(float(matrix[left_index, right_index]), 6),
            "qualified_at_75": True,
        }
        if identifier_column == "place_id":
            row["candidate_id"] = str(candidate["candidate_id"])
            row["eligibility_state"] = str(candidate["eligibility_state"])
        else:
            row["constituent_place_ids"] = "|".join(
                candidate["constituent_place_ids"]
            )
            row["place_id_count"] = int(candidate["place_id_count"])
            row["public_name_variant_count"] = int(
                candidate["public_name_variant_count"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        [identifier_column, "license_nbr"]
    ).reset_index(drop=True)


def candidate_cross_tab_and_excluded_detail(
    raw_candidates: pd.DataFrame,
    raw_matrix: np.ndarray,
    board: pd.DataFrame,
    included_pairs: Sequence[tuple[int, int, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    any_edge = (raw_matrix >= PRIMARY_THRESHOLD).any(axis=1)
    labels = []
    for has_edge, state in zip(any_edge, raw_candidates["eligibility_state"]):
        prefix = "any qualified Board edge" if bool(has_edge) else "no qualified Board edge"
        labels.append(f"{prefix} + LLM/reducer {state}")
    order = (
        "any qualified Board edge + LLM/reducer include",
        "any qualified Board edge + LLM/reducer exclude",
        "no qualified Board edge + LLM/reducer include",
        "no qualified Board edge + LLM/reducer exclude",
    )
    counter = Counter(labels)
    cross_tab = pd.DataFrame(
        [
            {
                "candidate_category": label,
                "count": int(counter[label]),
                "pct_of_raw_in_frame_candidates": pct(
                    int(counter[label]), len(raw_candidates)
                ),
            }
            for label in order
        ]
    )
    if int(cross_tab["count"].sum()) != len(raw_candidates):
        raise ValueError("candidate cross-tab does not partition raw candidates")

    included_board_indices = {right for _, right, _ in included_pairs}
    included_location_keys = {
        normalized_location_key(row.Street_Address, row.Zip_Code)
        for row in raw_candidates.loc[
            raw_candidates["eligibility_state"].eq("include")
        ].itertuples(index=False)
    }
    detail_rows = []
    excluded_indices = np.flatnonzero(
        any_edge & raw_candidates["eligibility_state"].eq("exclude").to_numpy()
    )
    for left_index in excluded_indices:
        row = raw_candidates.iloc[int(left_index)]
        qualified_board = np.flatnonzero(
            raw_matrix[int(left_index), :] >= PRIMARY_THRESHOLD
        )
        scores = raw_matrix[int(left_index), qualified_board]
        max_position = int(np.argmax(scores))
        max_board_index = int(qualified_board[max_position])
        covered_flags = [int(index) in included_board_indices for index in qualified_board]
        location_key = normalized_location_key(row["Street_Address"], row["Zip_Code"])
        detail_rows.append(
            {
                "place_id": str(row["place_id"]),
                "candidate_id": str(row["candidate_id"]),
                "target_construct_status": str(row["target_construct_status"]),
                "target_construct_status_confidence": str(
                    row["target_construct_status_confidence"]
                ),
                "business_status": str(row["business_status"]),
                "business_status_confidence": str(
                    row["business_status_confidence"]
                ),
                "candidate_plausibility": str(row["candidate_plausibility"]),
                "candidate_plausibility_confidence": str(
                    row["candidate_plausibility_confidence"]
                ),
                "identity_consistency": str(row["identity_consistency"]),
                "identity_consistency_confidence": str(
                    row["identity_consistency_confidence"]
                ),
                "maximum_board_linkage_score": round(
                    float(scores[max_position]), 6
                ),
                "maximum_score_board_license_nbr": str(
                    board.iloc[max_board_index]["license_nbr"]
                ),
                "qualified_board_record_count": len(qualified_board),
                "qualified_board_license_nbrs": "|".join(
                    str(board.iloc[int(index)]["license_nbr"])
                    for index in qualified_board
                ),
                "shares_normalized_address_zip_with_included_candidate": (
                    location_key in included_location_keys
                ),
                "maximum_score_board_covered_by_included_assignment": (
                    max_board_index in included_board_indices
                ),
                "any_qualified_board_covered_by_included_assignment": any(
                    covered_flags
                ),
                "all_qualified_boards_covered_by_included_assignment": all(
                    covered_flags
                ),
            }
        )
    detail = pd.DataFrame(detail_rows).sort_values("place_id").reset_index(drop=True)
    confidence_fields = (
        "target_construct_status_confidence",
        "business_status_confidence",
        "candidate_plausibility_confidence",
        "identity_consistency_confidence",
    )
    diagnostic = {
        "label": "Board-linked raw candidates excluded by the LLM/reducer",
        "count": len(detail),
        "target_pharmacy_judgment": distribution(detail["target_construct_status"]),
        "operating_status_judgment": distribution(detail["business_status"]),
        "candidate_plausibility_judgment": distribution(
            detail["candidate_plausibility"]
        ),
        "identity_consistency_judgment": distribution(
            detail["identity_consistency"]
        ),
        "field_specific_confidence": {
            field: distribution(detail[field]) for field in confidence_fields
        },
        "maximum_board_linkage_score": numeric_summary(
            detail["maximum_board_linkage_score"]
        ),
        "shares_normalized_address_zip_with_included_candidate": {
            "count": int(
                detail["shares_normalized_address_zip_with_included_candidate"].sum()
            ),
            "denominator": len(detail),
            "pct": pct(
                int(
                    detail[
                        "shares_normalized_address_zip_with_included_candidate"
                    ].sum()
                ),
                len(detail),
            ),
        },
        "maximum_score_board_covered_by_included_assignment": {
            "count": int(
                detail[
                    "maximum_score_board_covered_by_included_assignment"
                ].sum()
            ),
            "denominator": len(detail),
            "pct": pct(
                int(
                    detail[
                        "maximum_score_board_covered_by_included_assignment"
                    ].sum()
                ),
                len(detail),
            ),
        },
        "any_qualified_board_covered_by_included_assignment": {
            "count": int(
                detail["any_qualified_board_covered_by_included_assignment"].sum()
            ),
            "denominator": len(detail),
            "pct": pct(
                int(
                    detail[
                        "any_qualified_board_covered_by_included_assignment"
                    ].sum()
                ),
                len(detail),
            ),
        },
    }
    return cross_tab, detail, diagnostic


def collapse_locations(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = candidates.copy()
    primary["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(primary["Street_Address"], primary["Zip_Code"])
    ]
    if primary["location_key"].eq("|").any():
        raise ValueError("blank location key cannot define the location-unit diagnostic")
    rows = []
    safe_rows = []
    for location_key, members in primary.groupby("location_key", sort=True):
        ordered = members.sort_values("place_id").reset_index(drop=True)
        parsed_addresses = {
            geo_match.parse_address(str(value)) for value in ordered["Street_Address"]
        }
        zip_values = {str(value)[:5] for value in ordered["Zip_Code"]}
        if len(parsed_addresses) != 1 or len(zip_values) != 1:
            raise ValueError("collapsed group does not retain one linkage address and ZIP")
        names = tuple(
            sorted({str(value) for value in ordered["Company"] if str(value).strip()})
        )
        if not names:
            raise ValueError("collapsed location has no public-name representation")
        representative = ordered.iloc[0]
        location_hash = safe_hash(location_key, namespace="normalized-location")
        constituent_ids = tuple(sorted(ordered["place_id"].astype(str)))
        name_hashes = tuple(
            safe_hash(name, namespace="public-name-variant") for name in names
        )
        rows.append(
            {
                "_location_key": location_key,
                "location_group_sha256": location_hash,
                "Company": names[0],
                "candidate_names": names,
                "Street_Address": representative["Street_Address"],
                "Zip_Code": str(representative["Zip_Code"])[:5],
                "Latitude": representative["Latitude"],
                "Longitude": representative["Longitude"],
                "constituent_place_ids": constituent_ids,
                "place_id_count": len(ordered),
                "public_name_variant_count": len(names),
            }
        )
        safe_rows.append(
            {
                "location_group_sha256": location_hash,
                "constituent_place_ids": "|".join(constituent_ids),
                "place_id_count": len(ordered),
                "public_name_variant_count": len(names),
                "public_name_variant_sha256s": "|".join(name_hashes),
            }
        )
    # Preserve the established sensitivity's stable sort on the unhashed location
    # key for assignment.  The private key remains in memory and is never exported.
    collapsed = pd.DataFrame(rows).sort_values("_location_key").reset_index(drop=True)
    safe = pd.DataFrame(safe_rows).sort_values("location_group_sha256").reset_index(
        drop=True
    )
    return collapsed, safe


def assignment_identity_rows(
    summary: Mapping[str, int], *, unit: str
) -> list[dict[str, Any]]:
    return [
        {"candidate_unit": unit, "board_identity_category": key, "count": value}
        for key, value in summary.items()
    ]


def flat_assignment_rows(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in summaries:
        score = item["score_summary"]
        rows.append(
            {
                key: value
                for key, value in item.items()
                if key != "score_summary"
            }
            | {
                "assigned_score_n": score["n"],
                "assigned_score_minimum": score["minimum"],
                "assigned_score_q1": score["q1"],
                "assigned_score_median": score["median"],
                "assigned_score_mean": score["mean"],
                "assigned_score_q3": score["q3"],
                "assigned_score_maximum": score["maximum"],
            }
        )
    return rows


def build_report() -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, str]]:
    starting_hashes = assert_expected_hashes()
    frozen_board_input_hashes = authenticate_input_manifest()
    candidates, candidate_qa = load_candidates()
    board, board_audit = load_screened_board()
    target_zips = set(TARGET_ZIPS)
    raw = candidates.loc[candidates["Zip_Code"].isin(target_zips)].copy()
    raw = raw.sort_values("place_id").reset_index(drop=True)
    outside = candidates.loc[~candidates["Zip_Code"].isin(target_zips)].copy()
    included = raw.loc[raw["eligibility_state"].eq("include")].copy()
    included = included.sort_values("place_id").reset_index(drop=True)
    if (len(raw), len(outside), len(included), len(board)) != (984, 242, 339, 178):
        raise ValueError("fixed primary frame counts changed")

    # The task requires all 1,226 frozen candidates to be scored.  Primary
    # summaries are then formed only from the 984 rows whose returned ZIP is in
    # the fixed 60-ZIP frame.
    (
        all_candidate_matrix,
        all_candidate_name,
        all_candidate_address,
        all_candidate_zip_multiplier,
    ) = score_with_components(candidates, board)
    all_index = {
        place_id: index for index, place_id in enumerate(candidates["place_id"])
    }
    raw_positions = [all_index[value] for value in raw["place_id"]]
    included_positions = [all_index[value] for value in included["place_id"]]
    raw_matrix = all_candidate_matrix[raw_positions, :]
    raw_name = all_candidate_name[raw_positions, :]
    raw_address = all_candidate_address[raw_positions, :]
    raw_zip_multiplier = all_candidate_zip_multiplier[raw_positions, :]
    included_matrix = all_candidate_matrix[included_positions, :]
    included_name = all_candidate_name[included_positions, :]
    included_address = all_candidate_address[included_positions, :]
    included_zip_multiplier = all_candidate_zip_multiplier[included_positions, :]
    raw_index = {place_id: index for index, place_id in enumerate(raw["place_id"])}
    included_from_raw = raw_matrix[
        [raw_index[value] for value in included["place_id"]], :
    ]
    if not np.array_equal(included_matrix, included_from_raw):
        raise ValueError("included scores are not an exact subset of raw scores")

    raw_pairs, raw_assignment_records = assignment_records(raw, board, raw_matrix)
    included_pairs, included_assignment_records = assignment_records(
        included, board, included_matrix
    )
    existing_checks = verify_existing_assignments(
        board, raw_assignment_records, included_assignment_records
    )
    stages, board_stage_detail, core = board_stage_decomposition(
        board,
        raw_matrix,
        included_matrix,
        raw,
        included,
        raw_pairs,
        included_pairs,
        unit_term="candidate",
    )
    assignment_identity, assignment_identity_detail = (
        board_assignment_identity_comparison(board, raw_pairs, included_pairs)
    )
    endpoint_stability, endpoint_change_detail = linked_board_endpoint_stability(
        board,
        raw,
        included,
        raw_pairs,
        included_pairs,
        identifier_column="place_id",
    )
    cross_tab, excluded_detail, excluded_diagnostic = (
        candidate_cross_tab_and_excluded_detail(
            raw, raw_matrix, board, included_pairs
        )
    )
    raw_edges = qualified_edge_table(
        raw,
        board,
        raw_matrix,
        raw_name,
        raw_address,
        raw_zip_multiplier,
        identifier_column="place_id",
        scope="all raw in-frame candidates",
    )

    board_with_excluded = board_stage_detail.loc[
        board_stage_detail["has_excluded_qualified_endpoint"]
    ]
    excluded_endpoint_board_summary = {
        "board_records_with_any_excluded_qualified_candidate": len(board_with_excluded),
        "excluded_candidate_present_and_board_covered_by_an_included_assignment": int(
            board_with_excluded["stage"].eq("D").sum()
        ),
        "excluded_candidate_present_with_included_edge_but_not_assigned": int(
            board_with_excluded["stage"].eq("C").sum()
        ),
        "all_qualified_candidates_excluded_llm_associated_attrition": int(
            board_with_excluded["stage"].eq("B").sum()
        ),
    }

    raw_locations, raw_location_safe = collapse_locations(raw)
    included_locations, included_location_safe = collapse_locations(included)
    (
        raw_location_matrix,
        raw_location_name,
        raw_location_address,
        raw_location_zip_multiplier,
    ) = score_with_components(raw_locations, board)
    (
        included_location_matrix,
        included_location_name,
        included_location_address,
        included_location_zip_multiplier,
    ) = score_with_components(included_locations, board)
    raw_location_pairs, raw_location_assignment_records = assignment_records(
        raw_locations, board, raw_location_matrix
    )
    included_location_pairs, included_location_assignment_records = assignment_records(
        included_locations, board, included_location_matrix
    )
    (
        location_stages,
        location_board_stage_detail,
        location_core,
    ) = board_stage_decomposition(
        board,
        raw_location_matrix,
        included_location_matrix,
        raw_locations,
        included_locations,
        raw_location_pairs,
        included_location_pairs,
        unit_term="outlet-location",
    )
    location_identity, location_identity_detail = board_assignment_identity_comparison(
        board, raw_location_pairs, included_location_pairs
    )
    location_endpoint_stability, location_endpoint_change_detail = (
        linked_board_endpoint_stability(
            board,
            raw_locations,
            included_locations,
            raw_location_pairs,
            included_location_pairs,
            identifier_column="location_group_sha256",
        )
    )
    raw_location_edges = qualified_edge_table(
        raw_locations,
        board,
        raw_location_matrix,
        raw_location_name,
        raw_location_address,
        raw_location_zip_multiplier,
        identifier_column="location_group_sha256",
        scope="all raw collapsed outlet locations",
    )
    included_location_edges = qualified_edge_table(
        included_locations,
        board,
        included_location_matrix,
        included_location_name,
        included_location_address,
        included_location_zip_multiplier,
        identifier_column="location_group_sha256",
        scope="LLM/reducer-included collapsed outlet locations",
    )
    location_edges = pd.concat(
        [raw_location_edges, included_location_edges], ignore_index=True
    ).sort_values(
        ["candidate_scope", "location_group_sha256", "license_nbr"]
    ).reset_index(drop=True)

    included_primary_pair_set = {
        (
            safe_hash(
                normalized_location_key(
                    included.iloc[left]["Street_Address"],
                    included.iloc[left]["Zip_Code"],
                ),
                namespace="normalized-location",
            ),
            str(board.iloc[right]["license_nbr"]),
        )
        for left, right, _ in included_pairs
    }
    included_location_pair_set = {
        (
            str(included_locations.iloc[left]["location_group_sha256"]),
            str(board.iloc[right]["license_nbr"]),
        )
        for left, right, _ in included_location_pairs
    }
    existing_closure = read_json(
        ROOT
        / "analysis"
        / "v2"
        / f"results_closure_audit_{RUN_ID}"
        / "closure_audit.json"
    )
    closure_location = existing_closure["collapsed_location_sensitivity"][
        "collapsed_outlet_locations"
    ]
    location_cross_checks = {
        "included_location_count_matches_existing_closure": (
            len(included_locations) == int(closure_location["workflow_locations"])
        ),
        "included_location_link_count_matches_existing_closure": (
            len(included_location_pairs) == int(closure_location["linked_pairs"])
        ),
        "included_location_pair_set_matches_mapped_primary_place_id_pairs": (
            included_location_pair_set == included_primary_pair_set
        ),
    }
    if not all(location_cross_checks.values()):
        raise ValueError(
            f"included location-unit result differs from existing closure: {location_cross_checks}"
        )

    place_assignment_summaries = [
        assignment_summary("raw Place-ID rows", raw, board, raw_pairs),
        assignment_summary(
            "LLM/reducer-included Place-ID rows", included, board, included_pairs
        ),
    ]
    location_assignment_summaries = [
        assignment_summary(
            "raw exact-address-plus-ZIP outlet locations",
            raw_locations,
            board,
            raw_location_pairs,
        ),
        assignment_summary(
            "LLM/reducer-included exact-address-plus-ZIP outlet locations",
            included_locations,
            board,
            included_location_pairs,
        ),
    ]

    llm_attrition = board_stage_detail.loc[
        board_stage_detail["stage"].eq("B"),
        [
            "license_nbr",
            "board_record_sha256",
            "raw_qualified_edge_count",
            "excluded_qualified_endpoint_count",
            "maximum_raw_score",
        ],
    ].copy()
    llm_attrition.insert(2, "classification", "LLM-associated attrition")

    location_attrition = location_board_stage_detail.loc[
        location_board_stage_detail["stage"].eq("B"),
        [
            "license_nbr",
            "board_record_sha256",
            "raw_qualified_edge_count",
            "excluded_qualified_endpoint_count",
            "maximum_raw_score",
        ],
    ].copy()
    location_attrition.insert(
        2, "classification", "location-unit LLM-associated attrition"
    )

    verification = {
        **existing_checks,
        **location_cross_checks,
        "required_candidate_fields_complete": True,
        "all_1226_candidates_scored_against_178_board_records": (
            all_candidate_matrix.shape == (1226, 178)
        ),
        "raw_candidate_count_is_984": len(raw) == 984,
        "outside_primary_frame_candidate_count_is_242": len(outside) == 242,
        "included_candidate_count_is_339": len(included) == 339,
        "screened_board_count_is_178": len(board) == 178,
        "included_assignment_count_is_existing_146": len(included_pairs) == 146,
        "board_stage_sum_is_178": sum(item["count"] for item in stages) == 178,
        "candidate_cross_tab_sum_is_984": int(cross_tab["count"].sum()) == 984,
        "included_edges_are_exact_subset_of_raw_edges": bool(
            np.array_equal(included_matrix, included_from_raw)
        ),
        "included_edge_coverage_not_above_raw_edge_coverage": (
            core["included_candidate_edge_coverage"]["numerator"]
            <= core["raw_candidate_edge_coverage"]["numerator"]
        ),
        "regex_address_parser_explicitly_selected": (
            geo_match._PARSER is geo_match.parse_address_regex
        ),
        "all_composite_scores_reconstructed_from_declared_components": True,
    }
    if not all(verification.values()):
        raise ValueError(f"diagnostic verification failed: {verification}")

    recommendation = {
        "enter_thesis": True,
        "reason": (
            "The diagnostic directly decomposes preclassification candidate-frame "
            "correspondence, LLM/reducer retention, and assignment competition under "
            "the existing RQ2 frame without changing the primary result."
        ),
        "maximum_recommended_integration": {
            "results_table": 1,
            "methods_paragraph": 1,
            "results_paragraphs": "2-3",
            "discussion_paragraph": 1,
            "new_main_text_figure": 0,
        },
        "placement": (
            "If approved later, place one compact diagnostic table and two or three "
            "paragraphs in Results Section 4.2 after the current primary and "
            "location-unit summary; add one Methods paragraph after the primary "
            "score/threshold/assignment paragraph."
        ),
        "interpretive_limit": (
            "Describe only the incremental effect of the frozen LLM/reducer decision "
            "conditional on the fixed candidate frame, screened Board reference, "
            "60-ZIP geography, and declared linkage rule; do not present it as a "
            "general estimate of model performance."
        ),
    }

    report = {
        "schema_version": "nets-ai-stage-specific-board-linkage-diagnostic-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": RUN_ID,
        "scope": (
            "local read-only stage-specific Board-linkage diagnostic; no provider, "
            "model, classification, Board-screen, spatial, or thesis changes"
        ),
        "provenance": {
            "input_hashes": starting_hashes,
            "frozen_board_input_hashes": frozen_board_input_hashes,
            "delivery_semantic_manifest_sha256": read_json(
                DELIVERY / "run_manifest.json"
            ).get("semantic_manifest_sha256"),
            "model": candidate_qa["model"],
            "returned_models": candidate_qa["returned_models"],
            "fixed_frame_sha256": candidate_qa["fixed_frame_sha256"],
            "prompt_template_version": candidate_qa["prompt_template_version"],
            "prompt_template_sha256": candidate_qa["prompt_template_sha256"],
            "scientific_implementation_sha256": candidate_qa[
                "scientific_implementation_sha256"
            ],
            "candidate_frame_source_run": read_json(
                ROOT / "data" / "v2" / "canonical_thesis_empirical_run.json"
            ).get("canonical_run_id"),
            "canonical_source_run_pointer_sha256": sha256_file(
                ROOT / "data" / "v2" / "canonical_thesis_empirical_run.json"
            ),
            "implementation_file_hashes": {
                "code/geo_match.py": sha256_file(ROOT / "code" / "geo_match.py"),
                "code/study_design.py": sha256_file(
                    ROOT / "code" / "study_design.py"
                ),
                "code/v2_board_adapter.py": sha256_file(
                    ROOT / "code" / "v2_board_adapter.py"
                ),
                "code/audit_stage_specific_board_linkage.py": sha256_file(
                    Path(__file__).resolve()
                ),
            },
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scipy": importlib.metadata.version("scipy"),
                "rapidfuzz": importlib.metadata.version("rapidfuzz"),
            },
            "provider_calls": False,
            "model_calls": False,
            "classification_changes": 0,
            "spatial_analysis_run": False,
            "thesis_files_modified": False,
        },
        "input_availability": candidate_qa,
        "comparison_frame": {
            "target_zip_count": len(TARGET_ZIPS),
            "candidate_membership_basis": "provider-returned five-digit address ZIP",
            "raw_candidates_scored_before_geographic_reporting_filter": 1226,
            "raw_candidates_in_primary_60_zip_frame": len(raw),
            "raw_candidates_outside_primary_60_zip_frame": len(outside),
            "raw_candidates_with_missing_or_non_five_digit_zip": 0,
            "included_candidates_in_primary_frame": len(included),
            "excluded_candidates_in_primary_frame": int(
                raw["eligibility_state"].eq("exclude").sum()
            ),
            "screened_board_records_in_primary_frame": len(board),
            "outside_frame_candidates_role": "diagnostic count only; excluded from primary denominators",
        },
        "board_screen": board_audit,
        "linkage_contract": {
            "name_normalization": "study_design.normalize_match_name",
            "name_similarity": "RapidFuzz token_sort_ratio; strongest alias at a collapsed location",
            "address_normalization": "geo_match.parse_address_regex",
            "address_similarity": (
                "0.6 exact house-number agreement plus 0.4 street-name similarity "
                "when both numbers exist; otherwise 0.4 street-name similarity"
            ),
            "name_weight": NAME_WEIGHT,
            "address_weight": ADDRESS_WEIGHT,
            "zip_mismatch_multiplier": 0.85,
            "threshold": PRIMARY_THRESHOLD,
            "threshold_applied_before_assignment": True,
            "assignment": "Hungarian one-to-one with dummy rows and columns",
            "objective": PRIMARY_OBJECTIVE,
        },
        "place_id_analysis": {
            "board_four_stage_decomposition": stages,
            "core_stage_specific_measures": core,
            "raw_vs_included_hungarian": place_assignment_summaries,
            "board_assignment_identity_comparison": assignment_identity,
            "linked_in_both_endpoint_stability": endpoint_stability,
            "candidate_cross_tab": cross_tab.to_dict(orient="records"),
            "excluded_candidate_diagnostic": excluded_diagnostic,
            "board_records_with_excluded_qualified_endpoints": (
                excluded_endpoint_board_summary
            ),
            "llm_associated_attrition_board_records": llm_attrition.to_dict(
                orient="records"
            ),
        },
        "outlet_location_unit_analysis": {
            "role": "post hoc unit-of-analysis robustness analysis; Place ID remains primary",
            "collapse_key": "exact normalized full returned address plus five-digit returned ZIP",
            "name_handling": (
                "all constituent Place IDs and distinct public-name variants retained "
                "during scoring; strongest compatible alias used for each Board edge"
            ),
            "raw_place_ids": len(raw),
            "raw_collapsed_locations": len(raw_locations),
            "included_place_ids": len(included),
            "included_collapsed_locations": len(included_locations),
            "board_four_stage_decomposition": location_stages,
            "core_stage_specific_measures": location_core,
            "raw_vs_included_hungarian": location_assignment_summaries,
            "board_assignment_identity_comparison": location_identity,
            "linked_in_both_endpoint_stability": location_endpoint_stability,
            "llm_associated_attrition_board_records": location_attrition.to_dict(
                orient="records"
            ),
        },
        "interpretation_boundary": {
            "conditional_on": [
                "fixed 1,226-candidate platform frame",
                "screened 178-record Board reference",
                "fixed 60-ZIP primary geography",
                "declared normalization, score, threshold, and assignment rule",
            ],
            "supports": [
                "preclassification candidate-frame correspondence",
                "raw-candidate Board coverage",
                "LLM retention",
                "LLM-associated attrition",
                "assignment competition",
                "Board-reference coverage",
            ],
            "does_not_estimate": (
                "general LLM performance beyond this fixed conditional diagnostic"
            ),
            "board_role": (
                "dated screened operational reference; no claim of definitive "
                "real-world truth"
            ),
            "prohibited_metric_labels_used": False,
        },
        "recommendation": recommendation,
        "verification_checks": verification,
    }

    tables: dict[str, pd.DataFrame] = {
        "board_four_stage_decomposition.csv": pd.DataFrame(stages),
        "board_stage_safe_detail.csv": board_stage_detail,
        "candidate_cross_tab.csv": cross_tab,
        "board_linked_raw_candidates_excluded_safe_detail.csv": excluded_detail,
        "qualified_edges_place_id.csv": raw_edges,
        "raw_vs_included_place_id.csv": pd.DataFrame(
            flat_assignment_rows(place_assignment_summaries)
        ),
        "board_assignment_identity_comparison.csv": assignment_identity_detail,
        "board_assignment_identity_summary.csv": pd.DataFrame(
            assignment_identity_rows(assignment_identity, unit="Place ID")
        ),
        "board_assignment_endpoint_changes.csv": endpoint_change_detail,
        "llm_associated_board_attrition.csv": llm_attrition,
        "outlet_location_four_stage_decomposition.csv": pd.DataFrame(location_stages),
        "outlet_location_board_stage_safe_detail.csv": location_board_stage_detail,
        "raw_vs_included_outlet_location.csv": pd.DataFrame(
            flat_assignment_rows(location_assignment_summaries)
        ),
        "outlet_location_assignment_identity_comparison.csv": (
            location_identity_detail
        ),
        "outlet_location_assignment_identity_summary.csv": pd.DataFrame(
            assignment_identity_rows(location_identity, unit="outlet location")
        ),
        "outlet_location_assignment_endpoint_changes.csv": (
            location_endpoint_change_detail
        ),
        "outlet_location_safe_groups_raw.csv": raw_location_safe,
        "outlet_location_safe_groups_included.csv": included_location_safe,
        "qualified_edges_outlet_location.csv": location_edges,
        "outlet_location_llm_associated_board_attrition.csv": location_attrition,
    }
    ending_hashes = assert_expected_hashes()
    if ending_hashes != starting_hashes:
        raise ValueError("a frozen input changed during the diagnostic")
    thesis_paths = [
        ROOT / "docs" / "thesis" / "chapters" / "03_methodology.tex",
        ROOT / "docs" / "thesis" / "chapters" / "04_results.tex",
    ]
    thesis_hashes = {path.as_posix(): sha256_file(path) for path in thesis_paths}
    return report, tables, thesis_hashes


def render_markdown(report: Mapping[str, Any]) -> str:
    frame = report["comparison_frame"]
    place = report["place_id_analysis"]
    location = report["outlet_location_unit_analysis"]
    lines = [
        "# Stage-specific Board-linkage diagnostic",
        "",
        f"Frozen run: `{report['run_id']}`. Status: **PASS**.",
        "",
        "This local diagnostic did not call Google, OpenAI, Wayback, or any other "
        "provider; it did not change a classification, Board screen, linkage rule, "
        "spatial output, thesis file, or existing empirical result.",
        "",
        "## Input and frame verification",
        "",
        "All 1,226 authenticated candidates contain a nonblank public name, returned "
        "address, Place ID, exact five-digit returned ZIP, and frozen include/exclude "
        "decision. No provider reacquisition was needed.",
        "",
        "| Frame item | Count |",
        "|---|---:|",
        f"| All frozen candidates scored | {frame['raw_candidates_scored_before_geographic_reporting_filter']:,} |",
        f"| Raw candidates in primary 60-ZIP frame | {frame['raw_candidates_in_primary_60_zip_frame']:,} |",
        f"| Included candidates in primary frame | {frame['included_candidates_in_primary_frame']:,} |",
        f"| Excluded candidates in primary frame | {frame['excluded_candidates_in_primary_frame']:,} |",
        f"| Candidates outside primary frame (diagnostic only) | {frame['raw_candidates_outside_primary_60_zip_frame']:,} |",
        f"| Screened Board records | {frame['screened_board_records_in_primary_frame']:,} |",
        "",
        "Primary statistics use provider-returned candidate ZIPs and Board mailing ZIPs "
        "inside the original 60 target ZIPs. The 242 outside-frame candidates do not "
        "change a primary numerator or denominator.",
        "",
        "## Place-ID Board-stage decomposition",
        "",
        "| Stage | Definition | Count | Percent of 178 |",
        "|---|---|---:|---:|",
    ]
    for item in place["board_four_stage_decomposition"]:
        lines.append(
            f"| {item['stage']} | {item['definition']} | {item['count']} | "
            f"{item['pct_of_178_board_records']:.2f}% |"
        )
    core = place["core_stage_specific_measures"]
    lines.extend(
        [
            "",
            "The four stages sum to 178. Qualified-edge coverage and one-to-one "
            "assignment coverage are different quantities.",
            "",
            "| Measure | Numerator / denominator | Percent |",
            "|---|---:|---:|",
        ]
    )
    core_labels = (
        ("raw_candidate_edge_coverage", "Raw-candidate edge coverage"),
        ("included_candidate_edge_coverage", "Included-candidate edge coverage"),
        (
            "llm_retention_among_raw_retrievable_board_records",
            "LLM/reducer retention among raw-retrievable Board records",
        ),
        (
            "llm_associated_attrition_among_raw_retrievable_board_records",
            "LLM-associated attrition among raw-retrievable Board records",
        ),
        ("included_assignment_coverage", "Final included-assignment coverage"),
    )
    for key, label in core_labels:
        item = core[key]
        lines.append(
            f"| {label} | {item['numerator']} / {item['denominator']} | "
            f"{item['pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Raw versus included Hungarian assignment",
            "",
            "| Candidate set | Candidate rows | Linked pairs | Unmatched candidates | Unmatched Board | Candidate linked share | Board coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in place["raw_vs_included_hungarian"]:
        lines.append(
            f"| {item['candidate_unit']} | {item['candidate_rows']} | "
            f"{item['linked_pairs']} | {item['unmatched_candidate_rows']} | "
            f"{item['unmatched_board_rows']} | {item['candidate_linked_share_pct']:.2f}% | "
            f"{item['board_reference_coverage_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Board identities across the two independent assignments:",
            "",
            "| Identity comparison | Count |",
            "|---|---:|",
        ]
    )
    for key, value in place["board_assignment_identity_comparison"].items():
        lines.append(f"| {key} | {value} |")
    endpoint = place["linked_in_both_endpoint_stability"]
    lines.extend(
        [
            "",
            f"Among the {endpoint['board_records_linked_in_both_assignments']} Board "
            "records linked in both assignments, "
            f"{endpoint['same_assignment_endpoint']} retained the same Place-ID endpoint "
            f"and {endpoint['changed_assignment_endpoint_after_filtering']} switched "
            "Place-ID endpoint without changing Board coverage.",
        ]
    )

    lines.extend(
        [
            "",
            "## Candidate-level cross-tab",
            "",
            "| Candidate category | Count | Percent of 984 |",
            "|---|---:|---:|",
        ]
    )
    for item in place["candidate_cross_tab"]:
        lines.append(
            f"| {item['candidate_category']} | {item['count']} | "
            f"{item['pct_of_raw_in_frame_candidates']:.2f}% |"
        )
    excluded = place["excluded_candidate_diagnostic"]
    lines.extend(
        [
            "",
            f"There were **{excluded['count']} Board-linked raw candidates excluded by "
            "the LLM/reducer**. This label describes correspondence conditional on the "
            "candidate frame and linkage rule; it does not independently establish a "
            "classification error.",
            "",
            "Core judgment distributions for those candidates:",
            "",
            "| Field | Distribution |",
            "|---|---|",
        ]
    )
    judgment_keys = (
        ("target_pharmacy_judgment", "Target-construct status"),
        ("operating_status_judgment", "Operating status"),
        ("candidate_plausibility_judgment", "Candidate plausibility"),
        ("identity_consistency_judgment", "Identity consistency"),
    )
    for key, label in judgment_keys:
        formatted = "; ".join(
            f"{level}: {value['count']} ({value['pct']:.2f}%)"
            for level, value in excluded[key].items()
        )
        lines.append(f"| {label} | {formatted} |")
    score = excluded["maximum_board_linkage_score"]
    lines.extend(
        [
            "",
            "Field-specific confidence distributions and safe row-level details are in "
            "the JSON and CSV files. Maximum Board linkage scores among these candidates "
            f"ranged from {score['minimum']:.2f} to {score['maximum']:.2f} "
            f"(median {score['median']:.2f}).",
            "",
            "At the Board-record level:",
            "",
            "| Excluded-qualified-endpoint outcome | Count |",
            "|---|---:|",
        ]
    )
    board_endpoint_labels = {
        "board_records_with_any_excluded_qualified_candidate": (
            "Any excluded qualified candidate endpoint"
        ),
        "excluded_candidate_present_and_board_covered_by_an_included_assignment": (
            "Excluded endpoint present, but Board record covered by included assignment"
        ),
        "excluded_candidate_present_with_included_edge_but_not_assigned": (
            "Excluded endpoint and included edge present, but Board record unassigned"
        ),
        "all_qualified_candidates_excluded_llm_associated_attrition": (
            "All qualified candidates excluded (LLM-associated attrition)"
        ),
    }
    for key, label in board_endpoint_labels.items():
        lines.append(
            f"| {label} | {place['board_records_with_excluded_qualified_endpoints'][key]} |"
        )

    lines.extend(
        [
            "",
            "## Exact-address-plus-ZIP outlet-location unit analysis",
            "",
            f"The {location['raw_place_ids']} raw Place-ID rows collapse to "
            f"{location['raw_collapsed_locations']} outlet-location units; the "
            f"{location['included_place_ids']} included rows collapse to "
            f"{location['included_collapsed_locations']} units. All constituent Place "
            "IDs and distinct public-name variants were retained during scoring, and the "
            "strongest alias contributed to each Board edge.",
            "",
            "| Stage | Definition | Count | Percent of 178 |",
            "|---|---|---:|---:|",
        ]
    )
    for item in location["board_four_stage_decomposition"]:
        lines.append(
            f"| {item['stage']} | {item['definition']} | {item['count']} | "
            f"{item['pct_of_178_board_records']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "| Candidate set | Units | Linked pairs | Unmatched units | Unmatched Board | Unit linked share | Board coverage |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in location["raw_vs_included_hungarian"]:
        lines.append(
            f"| {item['candidate_unit']} | {item['candidate_rows']} | "
            f"{item['linked_pairs']} | {item['unmatched_candidate_rows']} | "
            f"{item['unmatched_board_rows']} | {item['candidate_linked_share_pct']:.2f}% | "
            f"{item['board_reference_coverage_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Board identities across the two outlet-location assignments:",
            "",
            "| Identity comparison | Count |",
            "|---|---:|",
        ]
    )
    for key, value in location["board_assignment_identity_comparison"].items():
        lines.append(f"| {key} | {value} |")
    location_endpoint = location["linked_in_both_endpoint_stability"]
    lines.extend(
        [
            "",
            f"At the outlet-location unit, "
            f"{location_endpoint['same_assignment_endpoint']} of "
            f"{location_endpoint['board_records_linked_in_both_assignments']} Board "
            "records linked in both assignments retained the same location endpoint; "
            f"{location_endpoint['changed_assignment_endpoint_after_filtering']} changed.",
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation boundary and thesis recommendation",
            "",
            "This diagnostic estimates the incremental effect of the frozen LLM/reducer "
            "decision conditional on the fixed candidate frame, screened Board reference, "
            "60-ZIP geography, and declared linkage rule. The Board file is a dated "
            "screened operational reference rather than a definitive account of every "
            "outlet. The analysis does not estimate general LLM performance beyond this "
            "fixed conditional diagnostic.",
            "",
            "**Recommendation:** include this later, if approved, as one compact Results "
            "table, one Methods paragraph, two or three Results paragraphs, and one "
            "Discussion paragraph. Do not add a new main-text figure. The safe row-level "
            "attrition list should remain in the audit package rather than the thesis.",
            "",
            "## Verification",
            "",
            "Every recorded verification check passed, including exact reproduction of "
            "the existing raw and included Place-ID assignments, exact reproduction of "
            "the included location-unit result, the 178-record stage partition, and the "
            "required included-edge subset relationship. See "
            "`board_linkage_stage_diagnostic.json` for all checks and provenance hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output: Path,
    report: dict[str, Any],
    tables: Mapping[str, pd.DataFrame],
    thesis_hashes_before: Mapping[str, str],
) -> None:
    if output.exists():
        raise FileExistsError(f"audit output directory already exists: {output}")
    output.mkdir(parents=True)
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False, lineterminator="\n")
    (output / "board_linkage_stage_diagnostic.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    (output / "board_linkage_stage_diagnostic.md").write_text(
        render_markdown(report), encoding="utf-8", newline=""
    )
    thesis_hashes_after = {
        path: sha256_file(Path(path)) for path in thesis_hashes_before
    }
    if thesis_hashes_after != dict(thesis_hashes_before):
        raise ValueError("a thesis source changed while the audit package was written")
    artifact_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    manifest = {
        "schema_version": "nets-ai-stage-specific-board-linkage-artifact-manifest-v1",
        "run_id": RUN_ID,
        "artifact_count": len(artifact_hashes),
        "artifacts": artifact_hashes,
        "thesis_source_hashes_before_and_after": thesis_hashes_after,
        "thesis_sources_unchanged": True,
        "frozen_inputs_unchanged": assert_expected_hashes() == EXPECTED_HASHES,
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"this diagnostic is frozen to run {RUN_ID}")
    report, tables, thesis_hashes = build_report()
    write_outputs(args.output_dir.resolve(), report, tables, thesis_hashes)
    print(json.dumps({
        "status": "pass",
        "output_dir": str(args.output_dir.resolve()),
        "verification_checks": len(report["verification_checks"]),
        "all_verification_checks_pass": all(report["verification_checks"].values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
