"""Finalize the frozen forced-choice Results package without provider calls.

This script authenticates the completed candidate units and delivery artifacts for
run ``20260902T012636Z``, recomputes the declared descriptive analyses, audits the
193 unmatched primary-frame workflow locations, and creates aggregate tables and
publication figures.  It never calls Google, Wayback, or OpenAI and never alters a
classification or any completed run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

import geo_match
from audit_forced_choice_results import build_audit
from study_design import TARGET_ZIPS, normalized_location_key
from v2_board_adapter import merge_v2_board_sources, prepare_v2_board_frames
from v2_spatial_adapter import (
    SpatialContractV2,
    prepare_fixed_tract_origins,
    run_v2_spatial_analysis,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260902T012636Z"
PRIMARY_STRATUM = "include_screened_primary"
MILES_PER_METER = 1.0 / 1609.344
PRIMARY_THRESHOLD = 75.0

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
    "dark": "#222222",
    "white": "#FFFFFF",
}

# Ordered, predeclared lexical indicators.  They characterize public-name text;
# they do not establish service model, ownership, or Board eligibility.
NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "hospital",
        re.compile(r"\b(hospital|medical center|childrens hospital)\b"),
    ),
    (
        "institutional",
        re.compile(
            r"\b(university|college|campus|correctional|corrections|veterans|"
            r"veteran affairs|county jail|federal medical)\b"
        ),
    ),
    (
        "clinic",
        re.compile(r"\b(clinic|urgent care|health center|healthcare center)\b"),
    ),
    (
        "medical_service",
        re.compile(
            r"\b(infusion|oncology|dialysis|hospice|home health|long term care|"
            r"specialty|surgery|medical supply|health system)\b"
        ),
    ),
    (
        "big_box",
        re.compile(r"\b(walmart|wal mart|target|costco|sams club)\b"),
    ),
    (
        "grocery",
        re.compile(
            r"\b(grocery|supermarket|foods?|market|cub|hy vee|lunds|byerlys|"
            r"kowalski|fresh thyme|whole foods)\b"
        ),
    ),
)
PHARMACY_TOKEN = re.compile(r"\b(pharmacy|drug|drugstore|apothecary|rx)\b")
KNOWN_REPEATED_BRAND = re.compile(
    r"\b(cvs|walgreens|thrifty white|goodrich|fairview|healthpartners|allina|"
    r"m health|genoa|capsule)\b"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(numerator: int | float, denominator: int | float) -> float | None:
    return round(100.0 * float(numerator) / float(denominator), 2) if denominator else None


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )


def normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def distribution(values: Iterable[object], denominator: int) -> dict[str, dict[str, Any]]:
    counts = Counter(str(value) for value in values)
    return {
        key: {"count": count, "pct": pct(count, denominator)}
        for key, count in sorted(counts.items())
    }


def numeric_summary(values: Sequence[float] | pd.Series) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {key: None for key in ("n", "mean", "median", "q1", "q3", "p90", "min", "max")}
    return {
        "n": int(len(series)),
        "mean": round(float(series.mean()), 6),
        "median": round(float(series.median()), 6),
        "q1": round(float(series.quantile(0.25)), 6),
        "q3": round(float(series.quantile(0.75)), 6),
        "p90": round(float(series.quantile(0.90)), 6),
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
    }


def load_input_manifest() -> dict[str, dict[str, Any]]:
    manifest_path = ROOT / "analysis" / "input_manifest.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("inputs")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("invalid frozen input manifest")
    by_path = {entry["path"]: entry for entry in entries}
    for relative, entry in by_path.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"frozen input failed authentication: {relative}")
    return by_path


def load_board() -> tuple[pd.DataFrame, pd.DataFrame]:
    load_input_manifest()
    raw = pd.read_csv(
        ROOT / "data" / "Board" / "F.1  Pharmacy (PH).csv",
        encoding="utf-8-sig",
        skiprows=3,
        low_memory=False,
    )
    cache = pd.read_csv(ROOT / "data" / "board_geocoded.csv", low_memory=False)
    return merge_v2_board_sources(raw, cache), cache


def load_completed_units(run_root: Path) -> pd.DataFrame:
    manifest = read_json(run_root / "run_manifest.json")
    declared = manifest.get("completed_unit_hashes")
    completed_root = run_root / "private" / "completed_candidates"
    if not isinstance(declared, dict) or len(declared) != 1226:
        raise ValueError("completed-unit manifest is incomplete")
    files = sorted(completed_root.glob("*.json"))
    if len(files) != 1226 or {path.name for path in files} != set(declared):
        raise ValueError("completed-unit file set differs from manifest")
    rows: list[dict[str, Any]] = []
    for path in files:
        if sha256_file(path) != declared[path.name]:
            raise ValueError(f"completed unit failed hash authentication: {path.name}")
        unit = read_json(path)
        location = unit["location"]
        decision = unit["decision"]
        if (
            unit.get("model") != manifest["model"]
            or unit.get("fixed_frame_sha256") != manifest["fixed_frame_sha256"]
            or unit.get("prompt_template_sha256") != manifest["prompt_template_sha256"]
            or unit.get("scientific_implementation_sha256")
            != manifest["scientific_implementation_sha256"]
            or unit.get("classification_decision_sha256")
            != decision.get("classification_decision_sha256", unit.get("classification_decision_sha256"))
        ):
            raise ValueError(f"completed-unit contract mismatch: {path.name}")
        rows.append(
            {
                "place_id": unit["place_id"],
                "candidate_id": unit["candidate_id"],
                "name": location["name"],
                "formatted_address": location["formatted_address"],
                "returned_postal_code": str(location.get("returned_postal_code") or "")[:5],
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
                "target_construct_status": decision["target_construct_status"],
                "business_status": decision["business_status"],
                "candidate_plausibility": decision["candidate_plausibility"],
                "identity_consistency": decision["identity_consistency"],
                "target_construct_status_confidence": decision["claim_confidence"]["target_construct_status"],
                "business_status_confidence": decision["claim_confidence"]["business_status"],
                "candidate_plausibility_confidence": decision["claim_confidence"]["candidate_plausibility"],
                "identity_consistency_confidence": decision["claim_confidence"]["identity_consistency"],
            }
        )
    frame = pd.DataFrame(rows).sort_values("place_id").reset_index(drop=True)
    if frame["place_id"].duplicated().any() or frame["candidate_id"].duplicated().any():
        raise ValueError("completed candidate identifiers are not unique")
    return frame


def build_google_frame(units: pd.DataFrame, delivery: Path) -> pd.DataFrame:
    eligibility = pd.read_csv(delivery / "eligibility.csv", dtype=str).fillna("")
    if len(eligibility) != 1226 or eligibility["place_id"].duplicated().any():
        raise ValueError("eligibility artifact is incomplete")
    frame = units.merge(
        eligibility[["place_id", "eligibility_state", "reason_codes"]],
        on="place_id",
        how="left",
        validate="one_to_one",
    )
    if frame["eligibility_state"].eq("").any():
        raise ValueError("candidate-to-eligibility join is incomplete")
    return pd.DataFrame(
        {
            "place_id": frame["place_id"],
            "candidate_id": frame["candidate_id"],
            "Company": frame["name"],
            "Street_Address": frame["formatted_address"],
            "Zip_Code": frame["returned_postal_code"],
            "zip5": frame["returned_postal_code"],
            "Latitude": pd.to_numeric(frame["latitude"], errors="coerce"),
            "Longitude": pd.to_numeric(frame["longitude"], errors="coerce"),
            "eligibility_state": frame["eligibility_state"],
            "eligibility_reason_codes": frame["reason_codes"],
        }
    )


def linkage_result(
    google: pd.DataFrame,
    board: pd.DataFrame,
    *,
    weights: Mapping[str, float],
    assignment: str,
    threshold: float = PRIMARY_THRESHOLD,
    objective: str = "max_weight",
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    matrix = geo_match.score_matrix(google, board, dict(weights))
    if assignment == "greedy":
        pairs = geo_match.assign_greedy(matrix, threshold)
    elif assignment == "hungarian":
        pairs = geo_match.assign_hungarian(matrix, threshold, objective=objective)
    else:
        raise ValueError(f"unknown assignment: {assignment}")
    return matrix, pairs


def linkage_summary_row(
    label: str,
    assignment: str,
    left_n: int,
    right_n: int,
    pairs: Sequence[tuple[int, int, float]],
    *,
    threshold: float,
    objective: str,
) -> dict[str, Any]:
    linked = len(pairs)
    return {
        "specification": label,
        "assignment": assignment,
        "objective": objective,
        "threshold": threshold,
        "workflow_locations": left_n,
        "board_locations": right_n,
        "linked_pairs": linked,
        "unmatched_workflow": left_n - linked,
        "unmatched_board": right_n - linked,
        "workflow_linked_share_pct": pct(linked, left_n),
        "board_reference_coverage_pct": pct(linked, right_n),
        "total_similarity": round(float(sum(score for _, _, score in pairs)), 4),
    }


def run_linkage_sensitivities(
    google: pd.DataFrame, board: pd.DataFrame
) -> list[dict[str, Any]]:
    specs = (
        ("name_only_greedy", {"w_name": 1.0}, "greedy", "greedy"),
        ("name_only_hungarian", {"w_name": 1.0}, "hungarian", "max_weight"),
        ("address_only_hungarian", {"w_addr": 1.0}, "hungarian", "max_weight"),
        (
            "name_address_hungarian",
            {"w_name": 1.0, "w_addr": 0.8},
            "hungarian",
            "max_weight",
        ),
        (
            "name_address_geo_hungarian",
            {"w_name": 1.0, "w_addr": 0.8, "w_geo": 0.5, "geo_scale": 300.0},
            "hungarian",
            "max_weight",
        ),
        (
            "name_address_max_cardinality_then_weight",
            {"w_name": 1.0, "w_addr": 0.8},
            "hungarian",
            "max_cardinality_then_weight",
        ),
    )
    rows = []
    for label, weights, assignment, objective in specs:
        _, pairs = linkage_result(
            google,
            board,
            weights=weights,
            assignment=assignment,
            objective=objective,
        )
        rows.append(
            linkage_summary_row(
                label,
                assignment,
                len(google),
                len(board),
                pairs,
                threshold=PRIMARY_THRESHOLD,
                objective=objective,
            )
        )
    for threshold in (70.0, 80.0):
        _, pairs = linkage_result(
            google,
            board,
            weights={"w_name": 1.0, "w_addr": 0.8},
            assignment="hungarian",
            threshold=threshold,
        )
        rows.append(
            linkage_summary_row(
                f"name_address_hungarian_t{int(threshold)}",
                "hungarian",
                len(google),
                len(board),
                pairs,
                threshold=threshold,
                objective="max_weight",
            )
        )
    return rows


def geography_sensitivity(
    google_all: pd.DataFrame, board_all: pd.DataFrame, v1: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = {
        str(value).strip()[:5]
        for value in v1["Zip_Code"].dropna()
        if re.fullmatch(r"\d{5}(?:\.0)?", str(value).strip())
    }
    target = set(TARGET_ZIPS)
    union = target | observed
    overlap = target & observed
    spillover = observed - target
    target_only = target - observed
    if (len(target), len(observed), len(overlap), len(spillover), len(target_only), len(union)) != (
        60,
        101,
        55,
        46,
        5,
        106,
    ):
        raise ValueError("frozen v1 geography no longer matches the completed audit")
    definitions = (
        ("primary_60_zip", "Primary", target),
        ("v1_observed_101_zip", "Diagnostic", observed),
        ("historical_v1_union_106_zip", "Historical geography sensitivity", union),
    )
    rows = []
    for key, role, zips in definitions:
        boards = prepare_v2_board_frames(board_all, sorted(zips))
        left = (
            google_all.loc[
                google_all["eligibility_state"].eq("include")
                & google_all["Zip_Code"].isin(zips)
            ]
            .sort_values("place_id")
            .reset_index(drop=True)
        )
        right = boards.screened_target.sort_values("license_nbr").reset_index(drop=True)
        _, pairs = linkage_result(
            left,
            right,
            weights={"w_name": 1.0, "w_addr": 0.8},
            assignment="hungarian",
        )
        rows.append(
            {
                "frame": key,
                "role": role,
                "zip_count": len(zips),
                "workflow_locations": len(left),
                "active_board_locations": len(boards.full_active_target),
                "screened_board_locations": len(right),
                "linked_pairs": len(pairs),
                "unmatched_workflow": len(left) - len(pairs),
                "unmatched_board": len(right) - len(pairs),
                "workflow_linked_share_pct": pct(len(pairs), len(left)),
                "board_reference_coverage_pct": pct(len(pairs), len(right)),
            }
        )
    set_audit = {
        "target_zip_count": len(target),
        "v1_observed_zip_count": len(observed),
        "shared_zip_count": len(overlap),
        "v1_only_spillover_zip_count": len(spillover),
        "target_only_zip_count": len(target_only),
        "historical_union_zip_count": len(union),
        "target_only_zips": sorted(target_only),
        "historical_v1_board_rule": "TARGET_ZIPS union frozen-v1-observed ZIPs",
        "primary_rule": "pre-specified 60 target ZIPs",
        "diagnostic_rule": "101 ZIPs observed in the frozen v1 inventory",
        "new_v2_derived_primary_geography": False,
    }
    return set_audit, rows


def name_pattern_flags(name: str, recurrence_group: str) -> dict[str, bool]:
    text = normalized_text(name)
    flags = {label: bool(pattern.search(text)) for label, pattern in NAME_PATTERNS}
    flags["pharmacy_token"] = bool(PHARMACY_TOKEN.search(text))
    flags["known_repeated_brand_token"] = bool(KNOWN_REPEATED_BRAND.search(text))
    flags["independent_style"] = bool(
        flags["pharmacy_token"]
        and recurrence_group == "single-occurrence name component"
        and not flags["known_repeated_brand_token"]
        and not any(flags[label] for label, _ in NAME_PATTERNS)
    )
    return flags


def primary_name_pattern(flags: Mapping[str, bool]) -> str:
    for label, _ in NAME_PATTERNS:
        if flags[label]:
            return label
    if flags["independent_style"]:
        return "independent_style"
    if flags["pharmacy_token"]:
        return "other_pharmacy_style"
    return "other"


def unmatched_workflow_audit(
    units: pd.DataFrame,
    google_primary: pd.DataFrame,
    board_primary: pd.DataFrame,
    delivery: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    links = pd.read_csv(delivery / "board_links.csv", dtype=str).fillna("")
    matched_ids = set(
        links.loc[links["stratum"].eq(PRIMARY_STRATUM), "place_id"].astype(str)
    )
    unmatched = google_primary.loc[~google_primary["place_id"].isin(matched_ids)].copy()
    if len(unmatched) != 193:
        raise ValueError("primary unmatched workflow denominator changed")

    recurrence = pd.read_csv(delivery / "name_recurrence_detail.csv", dtype=str).fillna("")
    recurrence = recurrence.loc[
        recurrence["source"].eq("Workflow inventory locations")
        & recurrence["linkage_strategy"].eq("name_address_hungarian_t75")
        & recurrence["recurrence_threshold"].eq("2")
    ][["source_record_id", "name_component_group"]]
    if len(recurrence) != 339 or recurrence["source_record_id"].duplicated().any():
        raise ValueError("primary name-recurrence detail changed")
    recurrence["source_record_id"] = pd.to_numeric(
        recurrence["source_record_id"], errors="raise"
    ).astype(int)
    source_index = google_primary[["place_id"]].reset_index(
        names="source_record_id"
    )
    recurrence = recurrence.merge(
        source_index,
        on="source_record_id",
        how="left",
        validate="one_to_one",
    )
    if recurrence["place_id"].isna().any():
        raise ValueError("name-recurrence row indices do not bind to the primary frame")
    unmatched = unmatched.merge(
        recurrence[["place_id", "name_component_group"]],
        on="place_id",
        how="left",
        validate="one_to_one",
    )
    unmatched = unmatched.merge(
        units.drop(columns=["name", "formatted_address", "returned_postal_code", "latitude", "longitude"]),
        on=["place_id", "candidate_id"],
        how="left",
        validate="one_to_one",
    )

    primary_locations = google_primary.copy()
    primary_locations["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(
            primary_locations["Street_Address"], primary_locations["Zip_Code"]
        )
    ]
    location_counts = primary_locations["location_key"].value_counts()
    matched_location_keys = set(
        primary_locations.loc[
            primary_locations["place_id"].isin(matched_ids), "location_key"
        ]
    )
    unmatched["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(unmatched["Street_Address"], unmatched["Zip_Code"])
    ]

    name_matrix = geo_match.score_matrix(unmatched, board_primary, {"w_name": 1.0})
    address_matrix = geo_match.score_matrix(unmatched, board_primary, {"w_addr": 1.0})
    composite_matrix = geo_match.score_matrix(
        unmatched, board_primary, {"w_name": 1.0, "w_addr": 0.8}
    )
    rows = []
    for position, (_, row) in enumerate(unmatched.iterrows()):
        best_index = int(np.argmax(composite_matrix[position]))
        composite = float(composite_matrix[position, best_index])
        name_score = float(name_matrix[position, best_index])
        address_score = float(address_matrix[position, best_index])
        qualified = composite >= PRIMARY_THRESHOLD
        name_pass = name_score >= PRIMARY_THRESHOLD
        address_pass = address_score >= PRIMARY_THRESHOLD
        cross_zip = str(row["Zip_Code"])[:5] != str(board_primary.iloc[best_index]["zip5"])[:5]
        if qualified:
            failure_mode = "assignment_competition"
        elif name_pass and not address_pass:
            failure_mode = "address_disagreement"
        elif address_pass and not name_pass:
            failure_mode = "name_disagreement"
        elif name_pass and address_pass and cross_zip:
            failure_mode = "cross_zip_attenuation"
        elif composite >= 50.0:
            failure_mode = "name_and_address_disagreement"
        else:
            failure_mode = "no_plausible_board_candidate"
        flags = name_pattern_flags(str(row["Company"]), str(row["name_component_group"]))
        output_row = {
            "candidate_id": row["candidate_id"],
            "target_construct_status": row["target_construct_status"],
            "business_status": row["business_status"],
            "candidate_plausibility": row["candidate_plausibility"],
            "identity_consistency": row["identity_consistency"],
            "target_construct_status_confidence": row["target_construct_status_confidence"],
            "business_status_confidence": row["business_status_confidence"],
            "candidate_plausibility_confidence": row["candidate_plausibility_confidence"],
            "identity_consistency_confidence": row["identity_consistency_confidence"],
            "name_component_group": row["name_component_group"],
            "primary_name_pattern": primary_name_pattern(flags),
            "closest_board_failure_mode": failure_mode,
            "best_composite_score": round(composite, 4),
            "best_name_score": round(name_score, 4),
            "best_address_score": round(address_score, 4),
            "qualified_edge_exists": bool(qualified),
            "cross_zip_best_candidate": bool(cross_zip),
            "exact_address_workflow_location_count": int(
                location_counts.get(row["location_key"], 0)
            ),
            "shares_exact_address_with_linked_workflow": bool(
                row["location_key"] in matched_location_keys
            ),
        }
        for label in (*[key for key, _ in NAME_PATTERNS], "pharmacy_token", "independent_style"):
            output_row[f"name_flag_{label}"] = bool(flags[label])
        rows.append(output_row)
    detail = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    core_fields = (
        "target_construct_status",
        "business_status",
        "candidate_plausibility",
        "identity_consistency",
    )
    confidence_fields = tuple(f"{field}_confidence" for field in core_fields)
    flag_columns = [column for column in detail if column.startswith("name_flag_")]
    failure_distribution = distribution(
        detail["closest_board_failure_mode"], len(detail)
    )
    for label in (
        "name_disagreement",
        "address_disagreement",
        "name_and_address_disagreement",
        "no_plausible_board_candidate",
        "assignment_competition",
        "cross_zip_attenuation",
    ):
        failure_distribution.setdefault(label, {"count": 0, "pct": 0.0})
    failure_distribution = dict(sorted(failure_distribution.items()))
    failure_by_recurrence = (
        detail.groupby(
            ["closest_board_failure_mode", "name_component_group"], dropna=False
        )
        .size()
        .rename("count")
        .reset_index()
    )
    failure_by_address_relation = (
        detail.groupby(
            [
                "closest_board_failure_mode",
                "shares_exact_address_with_linked_workflow",
            ],
            dropna=False,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    audit = {
        "denominator": len(detail),
        "terminology": "unmatched workflow locations; not false positives",
        "core_judgment_distributions": {
            field: distribution(detail[field], len(detail)) for field in core_fields
        },
        "confidence_distributions": {
            field.removesuffix("_confidence"): distribution(detail[field], len(detail))
            for field in confidence_fields
        },
        "public_name_pattern": {
            "scope": "deterministic lexical indicators applied to locally retained public names",
            "interpretation": "name patterns only; not service-model or ownership determinations",
            "mutually_exclusive_primary": distribution(detail["primary_name_pattern"], len(detail)),
            "nonexclusive_indicators": {
                column.removeprefix("name_flag_"): {
                    "count": int(detail[column].sum()),
                    "pct": pct(int(detail[column].sum()), len(detail)),
                }
                for column in flag_columns
            },
        },
        "provider_place_types": {
            "status": "not_available_in_retained_artifacts",
            "detail": "The privacy-preserving completed-unit and delivery schemas did not retain provider place types.",
        },
        "name_recurrence": distribution(detail["name_component_group"], len(detail)),
        "exact_address_multiplicity": {
            "multiple_workflow_records_same_normalized_address": {
                "count": int(
                    detail["exact_address_workflow_location_count"].gt(1).sum()
                ),
                "pct": pct(
                    int(detail["exact_address_workflow_location_count"].gt(1).sum()),
                    len(detail),
                ),
            },
            "shares_normalized_address_with_linked_workflow_record": {
                "count": int(detail["shares_exact_address_with_linked_workflow"].sum()),
                "pct": pct(
                    int(detail["shares_exact_address_with_linked_workflow"].sum()),
                    len(detail),
                ),
            },
            "definition": "exact normalized street address plus five-digit returned postal code",
        },
        "closest_board_assessment": {
            "rule": (
                "Highest primary composite score among the 178 screened Board locations. "
                "Component agreement uses the declared 75-point threshold; composite scores "
                "from 50 to below 75 with neither component passing are classified as joint "
                "name-and-address disagreement, and scores below 50 as no plausible candidate. "
                "Qualified but unassigned edges and cross-ZIP attenuation remain separate."
            ),
            "failure_mode_distribution": failure_distribution,
            "failure_mode_by_recurrence": failure_by_recurrence.to_dict("records"),
            "failure_mode_by_exact_address_relation": failure_by_address_relation.to_dict("records"),
            "assignment_competition_address_summary": {
                "assignment_competition_count": int(
                    detail["closest_board_failure_mode"].eq(
                        "assignment_competition"
                    ).sum()
                ),
                "shares_normalized_address_with_linked_workflow_record": int(
                    (
                        detail["closest_board_failure_mode"].eq(
                            "assignment_competition"
                        )
                        & detail["shares_exact_address_with_linked_workflow"]
                    ).sum()
                ),
            },
            "best_composite_score": numeric_summary(detail["best_composite_score"]),
            "best_name_score": numeric_summary(detail["best_name_score"]),
            "best_address_score": numeric_summary(detail["best_address_score"]),
        },
    }
    return audit, detail


def spatial_enrichment(delivery: Path, base_spatial: Mapping[str, Any]) -> dict[str, Any]:
    tract = pd.read_csv(delivery / "tract_spatial.csv", dtype={"GEOID": str, "target_zcta": str})
    google_miles = tract["google_include_nearest_m"] * MILES_PER_METER
    board_miles = tract["board_screened_nearest_m"] * MILES_PER_METER
    signed = google_miles - board_miles
    quartiles: dict[str, dict[str, Any]] = {}
    for column, label in (("income_quartile", "income"), ("nonwhite_quartile", "nonwhite")):
        quartiles[label] = {}
        for quartile, group in tract.groupby(column, sort=True):
            g = group["google_include_nearest_m"] * MILES_PER_METER
            b = group["board_screened_nearest_m"] * MILES_PER_METER
            item = {
                "tract_count": len(group),
                "workflow_nearest_miles": numeric_summary(g),
                "board_nearest_miles": numeric_summary(b),
                "signed_workflow_minus_board_miles": numeric_summary(g - b),
                "thresholds": {},
            }
            for threshold, suffix in ((0.5, "0_5"), (1.0, "1_0"), (2.0, "2_0")):
                changed = group[f"discordance_type_{suffix}mi"].isin(
                    ["board_only_beyond", "google_only_beyond"]
                )
                item["thresholds"][str(threshold)] = {
                    "changed_count": int(changed.sum()),
                    "changed_pct": pct(int(changed.sum()), len(group)),
                }
            quartiles[label][str(int(quartile))] = item
    return {
        "origin_count": len(tract),
        "workflow_nearest_miles": numeric_summary(google_miles),
        "board_nearest_miles": numeric_summary(board_miles),
        "signed_workflow_minus_board_miles": numeric_summary(signed),
        "quartiles": quartiles,
        "prespecified_sensitivities": base_spatial["thresholds"],
    }


def paired_threshold_summary(
    board_beyond: pd.Series, workflow_beyond: pd.Series
) -> dict[str, Any]:
    """Summarize the four paired states without changing their scientific meaning."""
    known = board_beyond.notna() & workflow_beyond.notna()
    board = board_beyond.loc[known].astype(bool)
    workflow = workflow_beyond.loc[known].astype(bool)
    both_beyond = int((board & workflow).sum())
    workflow_only_within = int((board & ~workflow).sum())
    board_only_within = int((~board & workflow).sum())
    both_within = int((~board & ~workflow).sum())
    changed = workflow_only_within + board_only_within
    return {
        "paired_origins": int(known.sum()),
        "both_within": both_within,
        "both_beyond": both_beyond,
        "workflow_only_within": workflow_only_within,
        "board_only_within": board_only_within,
        "classification_changed_count": changed,
        "classification_changed_pct": pct(changed, int(known.sum())),
    }


def spatial_sensitivity_extension(
    google_all: pd.DataFrame,
    board_all: pd.DataFrame,
    delivery: Path,
) -> dict[str, Any]:
    """Recompute the remaining Methods-prespecified spatial sensitivities locally.

    This is the same deterministic geometry logic used by the frozen-v1 downstream
    implementation: geometric-centroid origins, collection-frame interiors, and
    conditional ranges for frame-edge and missing-Board-coordinate uncertainty.
    """
    contract = SpatialContractV2()
    tracts = gpd.read_file(
        f"zip://{(ROOT / 'data' / 'tl_2023_27_tract.zip').as_posix()}"
    )
    zctas = gpd.read_file(
        f"zip://{(ROOT / 'data' / 'tl_2023_us_zcta520.zip').as_posix()}"
    )
    acs = pd.read_csv(ROOT / "data" / "acs_tracts_2023.csv", dtype={"GEOID": str})
    vehicles = pd.read_csv(
        ROOT / "data" / "acs_tracts_vehicles_2023.csv", dtype={"GEOID": str}
    )
    origins = prepare_fixed_tract_origins(
        tracts, acs, vehicles, zctas, TARGET_ZIPS, contract
    )
    returned_zips = {
        str(value).strip()[:5]
        for value in google_all["Zip_Code"].dropna()
        if re.fullmatch(r"\d{5}(?:\.0)?", str(value).strip())
    }
    boards = prepare_v2_board_frames(
        board_all, TARGET_ZIPS, observed_google_zips=returned_zips
    )
    primary = run_v2_spatial_analysis(
        google_all, boards, origins, contract, analysis_scope="full"
    )
    delivered = pd.read_csv(
        delivery / "tract_spatial.csv", dtype={"GEOID": str, "target_zcta": str}
    ).sort_values("GEOID").reset_index(drop=True)
    recomputed = primary.tracts.sort_values("GEOID").reset_index(drop=True)
    if delivered["GEOID"].tolist() != recomputed["GEOID"].tolist():
        raise ValueError("recomputed fixed-origin frame differs from delivery")
    for column in ("google_include_nearest_m", "board_screened_nearest_m"):
        if not np.allclose(
            delivered[column].to_numpy(dtype=float),
            recomputed[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(f"recomputed primary spatial surface differs: {column}")

    tracts_work = tracts.copy()
    tracts_work["GEOID"] = tracts_work["GEOID"].astype(str)
    tract_polygons = (
        tracts_work.loc[tracts_work["GEOID"].isin(origins["GEOID"]), ["GEOID", "geometry"]]
        .to_crs(contract.distance_crs)
        .sort_values("GEOID")
        .reset_index(drop=True)
    )
    origins_sorted = origins.sort_values("GEOID").reset_index(drop=True)
    if tract_polygons["GEOID"].tolist() != origins_sorted["GEOID"].tolist():
        raise ValueError("tract polygons do not reconcile with fixed origins")
    centroids = tract_polygons.geometry.centroid
    centroid_outside = int(
        sum(
            not polygon.covers(centroid)
            for polygon, centroid in zip(tract_polygons.geometry, centroids)
        )
    )
    centroid_origins = gpd.GeoDataFrame(
        origins_sorted.drop(columns="geometry").copy(),
        geometry=centroids,
        crs=contract.distance_crs,
    )
    centroid = run_v2_spatial_analysis(
        google_all, boards, centroid_origins, contract, analysis_scope="full"
    ).tracts.sort_values("GEOID").reset_index(drop=True)

    zctas_work = zctas.copy()
    zctas_work["ZCTA5CE20"] = zctas_work["ZCTA5CE20"].astype(str).str.zfill(5)
    frame_zips = set(TARGET_ZIPS) | returned_zips
    frame_polygons = zctas_work.loc[zctas_work["ZCTA5CE20"].isin(frame_zips)].to_crs(
        contract.distance_crs
    )
    missing_frame_zctas = sorted(frame_zips - set(frame_polygons["ZCTA5CE20"]))
    if frame_polygons.empty:
        raise ValueError("collection-frame ZCTA geometry is empty")
    frame_union = frame_polygons.geometry.union_all()
    frame_edge_m = origins_sorted.geometry.distance(frame_union.boundary)

    missing_board = boards.screened_extended.loc[
        ~boards.screened_extended["has_geocode"]
    ].copy()
    missing_board_zips = set(missing_board["zip5"].dropna().astype(str))
    missing_zcta_polygons = zctas_work.loc[
        zctas_work["ZCTA5CE20"].isin(missing_board_zips)
    ].to_crs(contract.distance_crs)
    missing_board_zctas_without_geometry = sorted(
        missing_board_zips - set(missing_zcta_polygons["ZCTA5CE20"])
    )
    if missing_zcta_polygons.empty:
        missing_board_zcta_min_m = pd.Series(
            np.inf, index=origins_sorted.index, dtype=float
        )
    else:
        missing_union = missing_zcta_polygons.geometry.union_all()
        missing_board_zcta_min_m = origins_sorted.geometry.distance(missing_union)
    if missing_board_zctas_without_geometry:
        missing_board_zcta_min_m = pd.Series(
            0.0, index=origins_sorted.index, dtype=float
        )

    threshold_rows: dict[str, Any] = {}
    for threshold in contract.thresholds_miles:
        threshold_m = threshold * contract.meters_per_mile
        suffix = str(threshold).replace(".", "_") + "mi"
        google_beyond = recomputed[f"google_include_beyond_{suffix}"].astype("boolean")
        board_beyond = recomputed[f"board_screened_beyond_{suffix}"].astype("boolean")
        centroid_google_beyond = centroid[
            f"google_include_beyond_{suffix}"
        ].astype("boolean")
        centroid_board_beyond = centroid[
            f"board_screened_beyond_{suffix}"
        ].astype("boolean")
        interior = frame_edge_m.gt(threshold_m)

        google_definite_within = recomputed["google_include_nearest_m"].le(threshold_m)
        google_definite_beyond = (
            recomputed["google_include_nearest_m"].gt(threshold_m) & interior
        )
        google_uncertain = ~(google_definite_within | google_definite_beyond)
        board_definite_within = recomputed["board_screened_nearest_m"].le(threshold_m)
        board_definite_beyond = (
            recomputed["board_screened_nearest_m"].gt(threshold_m)
            & interior
            & missing_board_zcta_min_m.gt(threshold_m)
        )
        board_uncertain = ~(board_definite_within | board_definite_beyond)
        both_determinate = ~(google_uncertain | board_uncertain)

        threshold_rows[str(threshold)] = {
            "threshold_miles": threshold,
            "geometric_centroid_origin": paired_threshold_summary(
                centroid_board_beyond, centroid_google_beyond
            ),
            "collection_frame_interior": {
                "interior_origin_count": int(interior.sum()),
                "comparison": paired_threshold_summary(
                    board_beyond.loc[interior], google_beyond.loc[interior]
                ),
            },
            "conditional_ranges": {
                "status": "conditional scenarios, not confidence intervals or unconditional logical bounds",
                "workflow_beyond_rate_lower_pct": pct(
                    int(google_definite_beyond.sum()), len(recomputed)
                ),
                "workflow_beyond_rate_upper_pct": pct(
                    int((google_definite_beyond | google_uncertain).sum()),
                    len(recomputed),
                ),
                "workflow_conditionally_unresolved_count": int(google_uncertain.sum()),
                "board_beyond_rate_lower_pct": pct(
                    int(board_definite_beyond.sum()), len(recomputed)
                ),
                "board_beyond_rate_upper_pct": pct(
                    int((board_definite_beyond | board_uncertain).sum()),
                    len(recomputed),
                ),
                "board_conditionally_unresolved_count": int(board_uncertain.sum()),
                "both_sources_conditionally_determinate_count": int(
                    both_determinate.sum()
                ),
                "conditionally_determinate_comparison": paired_threshold_summary(
                    board_beyond.loc[both_determinate],
                    google_beyond.loc[both_determinate],
                ),
            },
        }

    return {
        "algorithm_source": "existing deterministic logic in code/board_google_comparison.py",
        "primary_delivery_recomputed_exactly": True,
        "distance_crs": contract.distance_crs,
        "origin_count": len(origins_sorted),
        "geometric_centroids_outside_own_tract": centroid_outside,
        "collection_frame": {
            "rule": "60 target ZIPs union ZIPs observed among all frozen forced-choice candidates",
            "zip_count": len(frame_zips),
            "missing_zcta_geometry": missing_frame_zctas,
        },
        "missing_board_coordinates": {
            "screened_extended_rows": len(boards.screened_extended),
            "missing_coordinate_rows": len(missing_board),
            "mailing_zip_count": len(missing_board_zips),
            "mailing_zctas_without_geometry": missing_board_zctas_without_geometry,
        },
        "conditional_assumptions": {
            "workflow": "Frozen retrieved candidates exhaust observed supply inside the declared collection frame; unobserved supply may occur outside it.",
            "board": "Geocoded screened Board records exhaust located supply inside the collection frame; a missing-coordinate license is constrained to the ZCTA sharing its mailing ZIP.",
        },
        "thresholds": threshold_rows,
    }


def table_frames(
    base: Mapping[str, Any],
    unmatched: Mapping[str, Any],
    linkage_sensitivities: Sequence[Mapping[str, Any]],
    geography_rows: Sequence[Mapping[str, Any]],
    spatial: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    rq1 = base["rq1_candidate_flow"]
    table1_rows = []
    for state in ("include", "exclude", "unresolved"):
        item = rq1["eligibility"][state]
        table1_rows.append(
            {"section": "eligibility", "measure": state, "count": item["count"], "percent": item["pct"], "denominator": 1226}
        )
    for field, values in rq1["core_judgment_distributions"].items():
        for label, item in values.items():
            table1_rows.append(
                {"section": field, "measure": label, "count": item["count"], "percent": item["pct"], "denominator": 1226}
            )
    table1 = pd.DataFrame(table1_rows)

    confidence_rows = []
    for field, values in rq1["core_confidence_distributions"].items():
        for label, item in values.items():
            confidence_rows.append(
                {"field": field, "confidence": label, "count": item["count"], "percent": item["pct"], "denominator": 1226}
            )
    exclusion_rows = []
    for label, item in rq1["reason_distributions"]["exclude"]["reason_code_combinations"].items():
        exclusion_rows.append(
            {"negative_judgment_combination": label, "count": item["count"], "percent_of_excludes": item["pct_of_state"], "denominator": 728}
        )

    table2 = pd.DataFrame(linkage_sensitivities)
    table2_geo = pd.DataFrame(geography_rows)
    recurrence = pd.DataFrame(base["rq2_name_recurrence"]["primary_rows"])

    threshold_rows = []
    for threshold in ("0.5", "1.0", "2.0"):
        scenario = base["rq3_spatial"]["thresholds"][threshold]["scenarios"][
            "primary_google_include_vs_board_screened"
        ]
        states = scenario["states"]
        threshold_rows.append(
            {
                "threshold_miles": float(threshold),
                "both_within": states["neither_beyond"]["count"],
                "both_beyond": states["both_beyond"]["count"],
                "workflow_only_within": states["board_only_beyond"]["count"],
                "board_only_within": states["google_only_beyond"]["count"],
                "changed_tracts": scenario["classification_changed"]["count"],
                "changed_percent": scenario["classification_changed"]["pct"],
                "denominator": scenario["paired_origins"],
            }
        )

    neighborhood_rows = []
    for dimension in ("income", "nonwhite"):
        for quartile in ("1", "2", "3", "4"):
            item = spatial["quartiles"][dimension][quartile]
            neighborhood_rows.append(
                {
                    "dimension": dimension,
                    "quartile": int(quartile),
                    "tracts": item["tract_count"],
                    "workflow_mean_miles": item["workflow_nearest_miles"]["mean"],
                    "workflow_median_miles": item["workflow_nearest_miles"]["median"],
                    "board_mean_miles": item["board_nearest_miles"]["mean"],
                    "board_median_miles": item["board_nearest_miles"]["median"],
                    "signed_mean_miles": item["signed_workflow_minus_board_miles"]["mean"],
                    "signed_median_miles": item["signed_workflow_minus_board_miles"]["median"],
                    "changed_0_5_count": item["thresholds"]["0.5"]["changed_count"],
                    "changed_0_5_pct": item["thresholds"]["0.5"]["changed_pct"],
                    "changed_1_count": item["thresholds"]["1.0"]["changed_count"],
                    "changed_1_pct": item["thresholds"]["1.0"]["changed_pct"],
                    "changed_2_count": item["thresholds"]["2.0"]["changed_count"],
                    "changed_2_pct": item["thresholds"]["2.0"]["changed_pct"],
                }
            )
    spatial_sensitivity_rows: list[dict[str, Any]] = []
    extension = spatial["remaining_prespecified_sensitivities"]
    for threshold in ("0.5", "1.0", "2.0"):
        base_threshold = base["rq3_spatial"]["thresholds"][threshold]
        for scenario_name, scenario in base_threshold["scenarios"].items():
            states = scenario["states"]
            spatial_sensitivity_rows.append(
                {
                    "threshold_miles": float(threshold),
                    "sensitivity_dimension": "delivered_inventory_scenario",
                    "scenario": scenario_name,
                    "paired_origins": scenario["paired_origins"],
                    "both_within": states["neither_beyond"]["count"],
                    "both_beyond": states["both_beyond"]["count"],
                    "workflow_only_within": states["board_only_beyond"]["count"],
                    "board_only_within": states["google_only_beyond"]["count"],
                    "changed_count": scenario["classification_changed"]["count"],
                    "changed_pct": scenario["classification_changed"]["pct"],
                }
            )
        for tolerance_name, tolerance in base_threshold["threshold_tolerance"].items():
            states = tolerance["states"]
            spatial_sensitivity_rows.append(
                {
                    "threshold_miles": float(threshold),
                    "sensitivity_dimension": "four_meter_threshold_tolerance",
                    "scenario": tolerance_name,
                    "paired_origins": 327,
                    "both_within": states.get("neither_beyond", 0),
                    "both_beyond": states.get("both_beyond", 0),
                    "workflow_only_within": states.get("board_only_beyond", 0),
                    "board_only_within": states.get("google_only_beyond", 0),
                    "changed_count": tolerance["paired_disagreements"],
                    "changed_pct": tolerance["paired_disagreement_pct"],
                }
            )
        extra = extension["thresholds"][threshold]
        for dimension, item in (
            ("geometric_centroid_origin", extra["geometric_centroid_origin"]),
            (
                "collection_frame_interior",
                extra["collection_frame_interior"]["comparison"],
            ),
            (
                "conditionally_determinate_origins",
                extra["conditional_ranges"][
                    "conditionally_determinate_comparison"
                ],
            ),
        ):
            spatial_sensitivity_rows.append(
                {
                    "threshold_miles": float(threshold),
                    "sensitivity_dimension": dimension,
                    "scenario": dimension,
                    "paired_origins": item["paired_origins"],
                    "both_within": item["both_within"],
                    "both_beyond": item["both_beyond"],
                    "workflow_only_within": item["workflow_only_within"],
                    "board_only_within": item["board_only_within"],
                    "changed_count": item["classification_changed_count"],
                    "changed_pct": item["classification_changed_pct"],
                }
            )
    unmatched_name = pd.DataFrame(
        [
            {"category": key, **value, "denominator": unmatched["denominator"]}
            for key, value in unmatched["public_name_pattern"]["mutually_exclusive_primary"].items()
        ]
    )
    unmatched_failure = pd.DataFrame(
        [
            {"category": key, **value, "denominator": unmatched["denominator"]}
            for key, value in unmatched["closest_board_assessment"]["failure_mode_distribution"].items()
        ]
    )
    unmatched_recurrence = pd.DataFrame(
        [
            {"group": key, **value, "denominator": unmatched["denominator"]}
            for key, value in unmatched["name_recurrence"].items()
        ]
    )
    unmatched_confidence = pd.DataFrame(
        [
            {
                "field": field,
                "confidence": confidence,
                **value,
                "denominator": unmatched["denominator"],
            }
            for field, levels in unmatched["confidence_distributions"].items()
            for confidence, value in levels.items()
        ]
    )
    unmatched_failure_recurrence = pd.DataFrame(
        unmatched["closest_board_assessment"]["failure_mode_by_recurrence"]
    )
    unmatched_failure_address = pd.DataFrame(
        unmatched["closest_board_assessment"][
            "failure_mode_by_exact_address_relation"
        ]
    )
    return {
        "table1_candidate_flow.csv": table1,
        "table1_confidence.csv": pd.DataFrame(confidence_rows),
        "table1_exclusion_combinations.csv": pd.DataFrame(exclusion_rows),
        "table2_linkage_sensitivities.csv": table2,
        "table2_geography_sensitivity.csv": table2_geo,
        "table3_name_recurrence.csv": recurrence,
        "table4_paired_spatial_classification.csv": pd.DataFrame(threshold_rows),
        "table5_neighborhood_descriptives.csv": pd.DataFrame(neighborhood_rows),
        "supplement_spatial_sensitivities.csv": pd.DataFrame(
            spatial_sensitivity_rows
        ),
        "unmatched_name_patterns.csv": unmatched_name,
        "unmatched_linkage_failure_modes.csv": unmatched_failure,
        "unmatched_name_recurrence.csv": unmatched_recurrence,
        "unmatched_confidence.csv": unmatched_confidence,
        "unmatched_failure_by_recurrence.csv": unmatched_failure_recurrence,
        "unmatched_failure_by_address_relation.csv": unmatched_failure_address,
    }


def style_context() -> mpl.rc_context:
    return mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "figure.titlesize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["dark"],
            "axes.linewidth": 0.8,
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["gray"],
            "ytick.color": COLORS["gray"],
            "hatch.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def export_figure(fig: mpl.figure.Figure, stem: Path) -> dict[str, Any]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    width, height = fig.get_size_inches()
    metadata = {
        "Title": stem.name,
        "Author": "NETS-AI thesis workflow",
        "Subject": "Aggregate Results figure; no provider-returned raw content",
        "CreationDate": None,
        "ModDate": None,
    }
    # Preserve the declared physical canvas.  A tight bounding box can silently
    # change PDF page dimensions and makes final-size typography harder to audit.
    fig.savefig(
        pdf,
        dpi=300,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        metadata=metadata,
    )
    fig.savefig(
        png,
        dpi=400,
        bbox_inches=None,
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    # Matplotlib writes PNG as RGBA even when the alpha channel is fully opaque.
    # Remove that redundant channel so the raster fallback is explicitly RGB.
    with Image.open(png) as rendered:
        if rendered.mode == "RGBA":
            if rendered.getchannel("A").getextrema() != (255, 255):
                raise ValueError(f"unexpected transparency in publication figure: {png}")
            rendered.convert("RGB").save(png, dpi=(400, 400), optimize=True)
        elif rendered.mode != "RGB":
            raise ValueError(f"unexpected PNG mode {rendered.mode}: {png}")
    plt.close(fig)
    return {
        "pdf": {"path": pdf.relative_to(ROOT).as_posix(), "sha256": sha256_file(pdf)},
        "png": {"path": png.relative_to(ROOT).as_posix(), "sha256": sha256_file(png)},
        "canvas_inches": {"width": float(width), "height": float(height)},
        "png_dpi": 400,
        "png_expected_pixels": {
            "width": int(round(float(width) * 400)),
            "height": int(round(float(height) * 400)),
        },
        "pdf_vector_primary": True,
        "fixed_canvas": True,
        "opaque_background": True,
    }


def figure_record_flow(figures: Path) -> dict[str, Any]:
    with style_context():
        fig = plt.figure(figsize=(6.5, 5.4))
        gs = fig.add_gridspec(2, 1, height_ratios=(1.25, 1), hspace=0.48)
        ax_workflow = fig.add_subplot(gs[0, 0])
        ax_board = fig.add_subplot(gs[1, 0])

        def segment_label(ax, left, width, y, text, dark_fill=False, small_threshold=52):
            if width < small_threshold:
                y_offset = -0.34 if left == 0 else 0.34
                ax.text(
                    left + width / 2,
                    y + y_offset,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="dimgrey",
                    weight="semibold",
                )
            else:
                ax.text(
                    left + width / 2,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if dark_fill else COLORS["dark"],
                    weight="semibold",
                )

        workflow_rows = [
            ("Frozen candidate frame", [(1226, COLORS["light_gray"], None, "1,226 candidates", False)]),
            ("Eligibility", [(498, COLORS["blue"], None, "498 included", True), (728, "#F2C14E", "///", "728 excluded", False)]),
            ("Included geography", [(339, COLORS["blue"], None, "339 in 60 ZIPs", True), (159, COLORS["light_gray"], "///", "159 outside", False)]),
            ("Primary linkage", [(146, COLORS["green"], None, "146 linked", True), (193, "#F2C14E", "///", "193 not linked", False)]),
        ]
        yy = np.arange(len(workflow_rows))
        for y_pos, (_, segments) in zip(yy, workflow_rows):
            left = 0
            for width, color, hatch, label, dark_fill in segments:
                ax_workflow.barh(
                    y_pos,
                    width,
                    left=left,
                    height=0.58,
                    color=color,
                    edgecolor="white" if not hatch else COLORS["dark"],
                    linewidth=0.5,
                    hatch=hatch,
                )
                segment_label(
                    ax_workflow,
                    left,
                    width,
                    y_pos,
                    label,
                    dark_fill,
                    small_threshold=220,
                )
                left += width
        ax_workflow.set(
            yticks=yy,
            yticklabels=[row[0] for row in workflow_rows],
            xlim=(0, 1275),
            xlabel="Workflow records (count)",
            title="A. Workflow candidates",
        )
        ax_workflow.set_ylim(len(workflow_rows) - 0.25, -0.55)

        board_rows = [
            ("Active Board in 60 ZIPs", [(245, COLORS["light_gray"], None, "245 records", False)]),
            ("Automatic screen", [(178, COLORS["purple"], None, "178 screened", True), (67, COLORS["light_gray"], "///", "67 excluded", False)]),
            ("Reference coverage", [(146, COLORS["green"], None, "146 covered", True), (32, "#F2C14E", "///", "32 not covered", False)]),
        ]
        yy = np.arange(len(board_rows))
        for y_pos, (_, segments) in zip(yy, board_rows):
            left = 0
            for width, color, hatch, label, dark_fill in segments:
                ax_board.barh(
                    y_pos,
                    width,
                    left=left,
                    height=0.58,
                    color=color,
                    edgecolor="white" if not hatch else COLORS["dark"],
                    linewidth=0.5,
                    hatch=hatch,
                )
                segment_label(
                    ax_board,
                    left,
                    width,
                    y_pos,
                    label,
                    dark_fill,
                    small_threshold=50,
                )
                left += width
        ax_board.set(
            yticks=yy,
            yticklabels=[row[0] for row in board_rows],
            xlim=(0, 270),
            xlabel="Board records (count)",
            title="B. Screened Board reference",
        )
        ax_board.set_ylim(len(board_rows) - 0.25, -0.55)

        for ax in (ax_workflow, ax_board):
            ax.grid(axis="x", color="#E6E6E6", linewidth=0.55, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0, labelcolor="dimgrey")
            ax.tick_params(axis="x", length=0, labelcolor="dimgrey")
        fig.subplots_adjust(left=0.24, right=0.97, top=0.94, bottom=0.10)
    return export_figure(fig, figures / "figure01_record_flow")


def figure_core_decisions(base: Mapping[str, Any], figures: Path) -> dict[str, Any]:
    rq1 = base["rq1_candidate_flow"]
    fields = [
        ("target_construct_status", "Target construct", "target_pharmacy", "non_target"),
        ("business_status", "Operating status", "active", "inactive"),
        ("candidate_plausibility", "Candidate plausibility", "plausible", "implausible"),
        ("identity_consistency", "Identity consistency", "consistent", "inconsistent"),
    ]
    with style_context():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.3), sharex=True)
        ax = axes[0]
        y = np.arange(5)
        labels = [item[1] for item in fields] + ["Final eligibility"]
        positive = []
        negative = []
        positive_counts = []
        negative_counts = []
        for field, _, pos, neg in fields:
            positive.append(rq1["core_judgment_distributions"][field][pos]["pct"])
            negative.append(rq1["core_judgment_distributions"][field][neg]["pct"])
            positive_counts.append(rq1["core_judgment_distributions"][field][pos]["count"])
            negative_counts.append(rq1["core_judgment_distributions"][field][neg]["count"])
        positive.append(rq1["eligibility"]["include"]["pct"])
        negative.append(rq1["eligibility"]["exclude"]["pct"])
        positive_counts.append(rq1["eligibility"]["include"]["count"])
        negative_counts.append(rq1["eligibility"]["exclude"]["count"])
        ax.hlines(y, 0, positive, color=COLORS["light_gray"], linewidth=2.2, zorder=1)
        ax.scatter(
            positive,
            y,
            color=COLORS["blue"],
            marker="o",
            s=46,
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        for y_pos, value, count, neg_value, neg_count in zip(
            y, positive, positive_counts, negative, negative_counts
        ):
            alignment = "right" if value > 91 else "left"
            x_pos = value - 2.0 if alignment == "right" else value + 2.0
            ax.text(
                x_pos,
                y_pos - 0.12,
                f"{value:.1f}% ({int(count):,})",
                ha=alignment,
                va="bottom",
                fontsize=7.4,
                color="dimgrey",
                weight="semibold",
            )
            ax.text(
                100,
                y_pos + 0.16,
                f"negative: {neg_value:.1f}% ({int(neg_count):,})",
                ha="right",
                va="top",
                fontsize=6.7,
                color="dimgrey",
            )
        ax.set(
            yticks=y,
            yticklabels=labels,
            xlim=(0, 100),
            xlabel="Affirmative / included candidates (%)",
            title="A. Substantive decisions",
        )
        ax.set_ylim(len(labels) - 0.30, -0.45)
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.55)

        ax = axes[1]
        y = np.arange(4)
        high = []
        moderate = []
        low = []
        high_counts = []
        moderate_counts = []
        low_counts = []
        for field, _, _, _ in fields:
            dist = rq1["core_confidence_distributions"][field]
            high.append(dist.get("High", {"pct": 0})["pct"])
            moderate.append(dist.get("Moderate", {"pct": 0})["pct"])
            low.append(dist.get("Low", {"pct": 0})["pct"])
            high_counts.append(dist.get("High", {"count": 0})["count"])
            moderate_counts.append(dist.get("Moderate", {"count": 0})["count"])
            low_counts.append(dist.get("Low", {"count": 0})["count"])
        ax.barh(y, high, color="#24476B", label="High")
        ax.barh(
            y,
            moderate,
            left=high,
            color="#4C78A8",
            hatch="///",
            edgecolor=COLORS["dark"],
            linewidth=0.45,
            label="Moderate",
        )
        left = np.array(high) + np.array(moderate)
        ax.barh(
            y,
            low,
            left=left,
            color=COLORS["light_gray"],
            hatch="xxx",
            edgecolor=COLORS["dark"],
            linewidth=0.45,
            label="Low",
        )
        ax.set(
            yticks=y,
            yticklabels=["Target construct", "Operating status", "Plausibility", "Identity consistency"],
            xlim=(0, 100),
            xlabel="Candidates (%)",
            title="B. Stored confidence",
        )
        ax.set_ylim(len(fields) - 0.35, -0.45)
        for y_pos, h_count, m_count, l_count in zip(
            y, high_counts, moderate_counts, low_counts
        ):
            ax.text(
                100,
                y_pos - 0.42,
                f"H/M/L: {int(h_count):,}/{int(m_count):,}/{int(l_count):,}",
                ha="right",
                va="bottom",
                fontsize=6.7,
                color="dimgrey",
            )
        ax.grid(False)
        confidence_handles, confidence_labels = ax.get_legend_handles_labels()
        for panel in axes:
            panel.set_xticks([0, 25, 50, 75, 100])
            panel.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
            panel.spines["left"].set_visible(False)
            panel.set_axisbelow(True)
        fig.suptitle("Forced-choice decisions and confidence (n = 1,226)", y=0.98)
        fig.legend(
            confidence_handles,
            confidence_labels,
            loc="lower center",
            bbox_to_anchor=(0.75, 0.015),
            ncol=3,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            edgecolor="lightgrey",
            labelcolor="dimgrey",
            handlelength=1.8,
        )
        fig.subplots_adjust(left=0.20, right=0.985, top=0.86, bottom=0.22, wspace=0.58)
    return export_figure(fig, figures / "figure02_core_decisions_confidence")


def figure_geography(rows: Sequence[Mapping[str, Any]], figures: Path) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    labels = ["60 ZIP\nPrimary", "101 ZIP\nDiagnostic", "106 ZIP\nHistorical sensitivity"]
    y = np.arange(len(frame))
    with style_context():
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.8), sharex=True, sharey=True)
        specifications = [
            (
                axes[0],
                "workflow_linked_share_pct",
                "workflow_locations",
                "A. Workflow linked share",
                COLORS["blue"],
                "o",
            ),
            (
                axes[1],
                "board_reference_coverage_pct",
                "screened_board_locations",
                "B. Board-reference coverage",
                COLORS["orange"],
                "s",
            ),
        ]
        for ax, metric, denominator_field, title, color, marker in specifications:
            values = frame[metric].astype(float).to_numpy()
            ax.hlines(y, 0, values, color=COLORS["light_gray"], linewidth=2.0, zorder=1)
            ax.scatter(
                values,
                y,
                color=color,
                marker=marker,
                s=54,
                edgecolor=COLORS["dark"],
                linewidth=0.6,
                zorder=2,
            )
            for index, row in frame.iterrows():
                value = float(row[metric])
                numerator = int(row["linked_pairs"])
                denominator = int(row[denominator_field])
                x_pos = value + 2.0 if value < 84 else value - 2.0
                ha = "left" if value < 84 else "right"
                ax.text(
                    x_pos,
                    index,
                    f"{numerator}/{denominator}\n{value:.1f}%",
                    ha=ha,
                    va="center",
                    fontsize=7.3,
                    color="dimgrey",
                    weight="semibold",
                )
            ax.set(xlim=(0, 100), xlabel="Locations (%)", title=title)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", color="#E6E6E6", linewidth=0.55)
            ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
            ax.spines["left"].set_visible(False)
            ax.set_axisbelow(True)
        axes[0].set(yticks=y, yticklabels=labels)
        axes[0].invert_yaxis()
        fig.suptitle("Correspondence by pre-defined ZIP frame", y=0.98)
        fig.subplots_adjust(left=0.20, right=0.985, top=0.82, bottom=0.18, wspace=0.18)
    return export_figure(fig, figures / "figure03_geography_correspondence")


def figure_recurrence(base: Mapping[str, Any], figures: Path) -> dict[str, Any]:
    rows = pd.DataFrame(base["rq2_name_recurrence"]["primary_rows"])
    rows = rows.loc[rows["recurrence_threshold"].eq(2) & ~rows["group"].eq("unresolved")].copy()
    panels = [
        ("Workflow inventory locations", "Workflow linked share", "A. Workflow locations"),
        ("Board keyword-screen reference", "Board-reference coverage", "B. Board locations"),
    ]
    order = ["repeated-name component", "single-occurrence name component"]
    with style_context():
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
        marker_by_group = {
            "repeated-name component": "o",
            "single-occurrence name component": "s",
        }
        color_by_group = {
            "repeated-name component": COLORS["blue"],
            "single-occurrence name component": COLORS["gray"],
        }
        for ax, (source, xlabel, title) in zip(axes, panels):
            subset = rows.loc[rows["source"].eq(source)].set_index("group").loc[order]
            values = subset["linked_pct"].astype(float).to_numpy()
            low = values - subset["cluster_bootstrap_95_low_pct"].astype(float).to_numpy()
            high = subset["cluster_bootstrap_95_high_pct"].astype(float).to_numpy() - values
            yy = np.arange(2)
            for y_pos, group, value, low_error, high_error in zip(
                yy, order, values, low, high
            ):
                ax.errorbar(
                    value,
                    y_pos,
                    xerr=np.array([[low_error], [high_error]]),
                    fmt=marker_by_group[group],
                    markersize=6.7,
                    markerfacecolor=color_by_group[group],
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    color=color_by_group[group],
                    ecolor=COLORS["dark"],
                    capsize=3,
                    linewidth=1.1,
                    zorder=3,
                )
            ax.set(yticks=yy, yticklabels=["Repeated-name", "Single-occurrence"], xlim=(0, 100), xlabel=f"{xlabel} (%)", title=title)
            ax.invert_yaxis()
            interval_high = subset["cluster_bootstrap_95_high_pct"].astype(float).to_numpy()
            interval_low = subset["cluster_bootstrap_95_low_pct"].astype(float).to_numpy()
            for y_pos, value, linked, records, low_end, high_end in zip(
                yy,
                values,
                subset["linked_records"],
                subset["records"],
                interval_low,
                interval_high,
            ):
                if high_end > 86:
                    x_pos, alignment = low_end - 2.5, "right"
                else:
                    x_pos, alignment = high_end + 2.5, "left"
                ax.text(
                    x_pos,
                    y_pos,
                    f"{int(linked)}/{int(records)} = {value:.1f}%",
                    ha=alignment,
                    va="center",
                    fontsize=7.8,
                    color="dimgrey",
                    weight="semibold",
                )
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.grid(axis="x", color="#E6E6E6", linewidth=0.55)
            ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
            ax.spines["left"].set_visible(False)
            ax.set_axisbelow(True)
        fig.suptitle(
            "Name recurrence and primary Board correspondence\n"
            "95% component-bootstrap intervals; 5,000 resamples",
            y=0.985,
        )
        fig.text(
            0.985,
            0.015,
            "Workflow grouping excludes one unresolved recurrence record; Board grouping has none.",
            ha="right",
            va="bottom",
            fontsize=7,
            color="dimgrey",
            style="italic",
        )
        fig.subplots_adjust(left=0.22, right=0.975, top=0.82, bottom=0.15, hspace=0.55)
    return export_figure(fig, figures / "figure04_name_recurrence")


def figure_spatial_maps(delivery: Path, figures: Path) -> dict[str, Any]:
    tract = pd.read_csv(delivery / "tract_spatial.csv", dtype={"GEOID": str})
    shapes = gpd.read_file(f"zip://{(ROOT / 'data' / 'tl_2023_27_tract.zip').as_posix()}")
    if shapes.crs is None:
        raise ValueError("tract geometry lacks CRS metadata")
    shapes["GEOID"] = shapes["GEOID"].astype(str)
    mapped = shapes[["GEOID", "geometry"]].merge(tract, on="GEOID", how="inner", validate="one_to_one")
    mapped = gpd.GeoDataFrame(mapped, geometry="geometry", crs=shapes.crs).to_crs(26915)
    if (
        len(mapped) != 327
        or mapped.geometry.isna().any()
        or mapped.geometry.is_empty.any()
        or (~mapped.geometry.is_valid).any()
    ):
        raise ValueError("eligible tract geometry frame is incomplete")
    if mapped.crs is None or mapped.crs.to_epsg() != 26915:
        raise ValueError("map must use the declared EPSG:26915 projected CRS")
    axis_units = {axis.unit_name.lower() for axis in mapped.crs.axis_info}
    if axis_units != {"metre"}:
        raise ValueError(f"unexpected map CRS units: {axis_units}")
    color_map = {
        "neither_beyond": "#E6E6E6",
        "both_beyond": "#595959",
        "board_only_beyond": COLORS["blue"],
        "google_only_beyond": COLORS["vermillion"],
    }
    label_map = {
        "neither_beyond": "Both within",
        "both_beyond": "Both beyond",
        "board_only_beyond": "Workflow only within",
        "google_only_beyond": "Board only within",
    }
    hatch_map = {
        "neither_beyond": None,
        "both_beyond": None,
        "board_only_beyond": "///",
        "google_only_beyond": "\\\\\\",
    }
    expected_states = set(color_map)
    for suffix in ("0_5", "1_0", "2_0"):
        column = f"discordance_type_{suffix}mi"
        observed = set(mapped[column].dropna().astype(str))
        if mapped[column].isna().any() or not observed.issubset(expected_states):
            raise ValueError(f"map classification contains missing or unknown states: {column}")

    min_x, min_y, max_x, max_y = mapped.total_bounds
    width = max_x - min_x
    height = max_y - min_y
    x_pad = width * 0.035
    y_pad = height * 0.035
    extent = (min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad)
    scale_length_m = 10_000 if width >= 40_000 else 5_000

    def add_scale_and_grid_north(ax):
        x0 = extent[0] + (extent[1] - extent[0]) * 0.055
        y0 = extent[2] + (extent[3] - extent[2]) * 0.055
        line = ax.plot(
            [x0, x0 + scale_length_m],
            [y0, y0],
            color=COLORS["dark"],
            linewidth=2.2,
            solid_capstyle="butt",
            zorder=10,
        )[0]
        line.set_path_effects([patheffects.Stroke(linewidth=3.8, foreground="white"), patheffects.Normal()])
        for x_tick in (x0, x0 + scale_length_m):
            tick = ax.plot(
                [x_tick, x_tick],
                [y0 - height * 0.012, y0 + height * 0.012],
                color=COLORS["dark"],
                linewidth=1.0,
                zorder=11,
            )[0]
            tick.set_path_effects([patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()])
        label = ax.text(
            x0 + scale_length_m / 2,
            y0 + height * 0.022,
            f"{scale_length_m / 1000:g} km",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=COLORS["dark"],
            zorder=12,
        )
        label.set_path_effects([patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()])
        ax.annotate(
            "",
            xy=(0.925, 0.91),
            xytext=(0.925, 0.79),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": COLORS["dark"],
                "linewidth": 0.9,
                "mutation_scale": 8,
            },
            zorder=12,
        )
        ax.text(
            0.925,
            0.945,
            "N",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            weight="bold",
            color=COLORS["dark"],
            zorder=12,
        )

    with style_context():
        fig = plt.figure(figsize=(6.5, 7.1))
        gs = fig.add_gridspec(2, 2, left=0.035, right=0.985, top=0.86, bottom=0.055, wspace=0.07, hspace=0.14)
        map_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
        legend_ax = fig.add_subplot(gs[1, 1])
        for panel_label, ax, (threshold, suffix) in zip(
            ("A", "B", "C"),
            map_axes,
            ((0.5, "0_5"), (1.0, "1_0"), (2.0, "2_0")),
        ):
            column = f"discordance_type_{suffix}mi"
            for state in ("neither_beyond", "both_beyond", "board_only_beyond", "google_only_beyond"):
                subset = mapped.loc[mapped[column].eq(state)]
                if not subset.empty:
                    subset.plot(
                        ax=ax,
                        facecolor=color_map[state],
                        edgecolor="white",
                        linewidth=0.28,
                        zorder=2,
                    )
                    if hatch_map[state]:
                        subset.plot(
                            ax=ax,
                            facecolor="none",
                            edgecolor=COLORS["dark"],
                            linewidth=0.18,
                            hatch=hatch_map[state],
                            zorder=3,
                        )
            mapped[["geometry"]].dissolve().boundary.plot(
                ax=ax,
                color=COLORS["dark"],
                linewidth=0.55,
                zorder=4,
            )
            changed = int(mapped[column].isin(["board_only_beyond", "google_only_beyond"]).sum())
            changed_pct = 100.0 * changed / len(mapped)
            ax.set_title(
                f"{panel_label}. {threshold:g}-mile threshold\n"
                f"{changed}/327 changed ({changed_pct:.1f}%)",
                loc="left",
                pad=4,
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")
            add_scale_and_grid_north(ax)
            ax.set_axis_off()
        handles = [
            Patch(
                facecolor=color_map[key],
                edgecolor=COLORS["dark"],
                hatch=hatch_map[key],
                linewidth=0.6,
                label=label_map[key],
            )
            for key in ("neither_beyond", "both_beyond", "board_only_beyond", "google_only_beyond")
        ]
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            edgecolor="lightgrey",
            labelcolor="dimgrey",
            fontsize=8.2,
            title="Paired classification",
            title_fontsize=8.5,
        )
        legend_ax.text(
            0.02,
            0.50,
            "Map specification\n"
            "Projection: NAD83 / UTM zone 15N\n"
            "EPSG:26915; units: metres\n"
            "North arrows indicate grid north\n"
            "Tract geometry: TIGER/Line 2023\n"
            "Common extent and scale across panels\n"
            "No missing class among 327 tracts\n"
            "No basemap or remote tiles",
            transform=legend_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="dimgrey",
            linespacing=1.35,
        )
        fig.suptitle("Paired tract classification by inventory (n = 327)", y=0.975)
    output = export_figure(fig, figures / "figure05_spatial_disagreement_maps")
    output["map_specification"] = {
        "geometry_source": "data/tl_2023_27_tract.zip",
        "geometry_vintage": 2023,
        "source_crs": str(shapes.crs),
        "display_crs": "EPSG:26915",
        "display_crs_name": mapped.crs.name,
        "linear_units": "metre",
        "common_extent_m": [float(value) for value in extent],
        "scale_bar_km": scale_length_m / 1000,
        "north_arrow": "grid north",
        "classification": "four fixed paired states; categorical",
        "missing_classification_count": 0,
        "basemap": None,
        "remote_tiles": False,
    }
    return output


def figure_distance_difference(delivery: Path, figures: Path) -> dict[str, Any]:
    tract = pd.read_csv(delivery / "tract_spatial.csv")
    values = (
        tract["google_include_nearest_m"] - tract["board_screened_nearest_m"]
    ) * MILES_PER_METER
    # Fixed 0.1-mile bins make the transformation explicit and reproducible.
    lower = math.floor(float(values.min()) * 10) / 10
    upper = math.ceil(float(values.max()) * 10) / 10 + 0.1
    bins = np.arange(lower, upper + 1e-9, 0.1)
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75]).tolist()
    with style_context():
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.axvspan(
            q1,
            q3,
            facecolor="#F6DFA6",
            edgecolor=COLORS["orange"],
            hatch="///",
            linewidth=0.6,
            alpha=0.55,
            label=f"IQR = {q1:.3f} to {q3:.3f} mi",
            zorder=0,
        )
        ax.hist(
            values,
            bins=bins,
            color="#4C78A8",
            edgecolor="white",
            linewidth=0.55,
            label="Tract count",
            zorder=2,
        )
        ax.axvline(
            0,
            color=COLORS["dark"],
            linewidth=1.2,
            linestyle="--",
            label="Equal distance",
            zorder=4,
        )
        ax.axvline(
            median,
            color=COLORS["vermillion"],
            linewidth=1.3,
            linestyle="-",
            label=f"Median = {median:.3f} mi",
            zorder=5,
        )
        ax.set(
            xlim=(float(bins.min()), float(bins.max())),
            xlabel="Workflow nearest distance − Board nearest distance (miles)",
            ylabel="Tract origins (count)",
            title="Signed nearest-distance differences (n = 327)",
        )
        ax.legend(
            handles=[
                Line2D([0], [0], color=COLORS["dark"], linewidth=1.2, linestyle="--", label="Equal distance"),
                Line2D([0], [0], color=COLORS["vermillion"], linewidth=1.3, linestyle="-", label=f"Median = {median:.3f} mi"),
                Patch(facecolor="#F6DFA6", edgecolor=COLORS["orange"], hatch="///", label=f"IQR = {q1:.3f} to {q3:.3f} mi"),
            ],
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.93,
            edgecolor="lightgrey",
            labelcolor="dimgrey",
        )
        ax.text(
            0.01,
            0.98,
            "Negative: workflow location is nearer\n"
            "Positive: Board location is nearer\n"
            "Fixed bin width: 0.1 mile",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="dimgrey",
        )
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.55)
        ax.tick_params(axis="both", which="both", length=0, labelcolor="dimgrey")
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.18)
    output = export_figure(fig, figures / "figure06_nearest_distance_difference")
    output["bin_width_miles"] = 0.1
    return output


def render_markdown(audit: Mapping[str, Any]) -> str:
    rq1 = audit["base_results_audit"]["rq1_candidate_flow"]
    unmatched = audit["unmatched_workflow_locations"]
    primary = audit["rq2"]["geography_sensitivity"][0]
    spatial = audit["rq3"]
    lines = [
        f"# Final forced-choice Results audit: `{audit['run_id']}`",
        "",
        "All quantities are recomputed from authenticated local artifacts. This report separates observed results from interpretation and contains no provider-returned review or archive text.",
        "",
        "## RQ1 - workflow output",
        "",
        "| Eligibility | Count | Percent |",
        "|---|---:|---:|",
    ]
    for state in ("include", "exclude", "unresolved"):
        item = rq1["eligibility"][state]
        lines.append(f"| {state} | {item['count']:,} | {item['pct']:.2f}% |")
    lines.extend(["", "### Exclusion judgment combinations", "", "| Combination | Count | Percent of excludes |", "|---|---:|---:|"])
    for label, item in rq1["reason_distributions"]["exclude"]["reason_code_combinations"].items():
        lines.append(f"| {label} | {item['count']:,} | {item['pct_of_state']:.2f}% |")
    lines.extend(
        [
            "",
            "The forced-choice pilot produced 24/24 structurally valid outputs with zero technical failures and no change in any core substantive field across A/B/C. Its purposive eight-candidate sample tests execution and evidence order; it does not estimate population performance.",
            "",
            "## RQ2 - Board correspondence",
            "",
            "| Geography | Role | Workflow | Board | Linked | Workflow linked share | Board coverage |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in audit["rq2"]["geography_sensitivity"]:
        lines.append(
            f"| {row['zip_count']} ZIP | {row['role']} | {row['workflow_locations']} | {row['screened_board_locations']} | {row['linked_pairs']} | {row['workflow_linked_share_pct']:.2f}% | {row['board_reference_coverage_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"The primary 60-ZIP frame contains {primary['workflow_locations']} workflow locations and {primary['screened_board_locations']} screened Board locations; {primary['linked_pairs']} pairs link, leaving {primary['unmatched_workflow']} workflow locations and {primary['unmatched_board']} Board locations unmatched.",
            "",
            "### The 193 unmatched workflow locations",
            "",
            "| Highest-score assessment | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for label, item in unmatched["closest_board_assessment"]["failure_mode_distribution"].items():
        lines.append(f"| {label} | {item['count']} | {item['pct']:.2f}% |")
    lines.extend(["", "| Primary public-name pattern | Count | Percent |", "|---|---:|---:|"])
    for label, item in unmatched["public_name_pattern"]["mutually_exclusive_primary"].items():
        lines.append(f"| {label} | {item['count']} | {item['pct']:.2f}% |")
    lines.extend(
        [
            "",
            "These are deterministic name-text and linkage-score descriptions, not manual corrections or error labels. Provider place types were not retained and therefore are not summarized.",
            "",
            "### Name recurrence by predefined cutoff",
            "",
            "| Cutoff | Source | Group | Records | Linked | Percent | 95% component-bootstrap interval |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in audit["rq2"]["name_recurrence"]["primary_rows"]:
        interval = (
            f"{row['cluster_bootstrap_95_low_pct']:.2f}-{row['cluster_bootstrap_95_high_pct']:.2f}%"
            if row["cluster_bootstrap_95_low_pct"] is not None
            else "-"
        )
        linked_pct = (
            f"{row['linked_pct']:.2f}%" if row["linked_pct"] is not None else "-"
        )
        lines.append(
            f"| {row['recurrence_threshold']} | {row['source']} | {row['group']} | {row['records']} | {row['linked_records']} | {linked_pct} | {interval} |"
        )
    lines.extend(
        [
            "",
            "## RQ3 - paired potential proximity",
            "",
            "| Surface | Mean | Median | Q1 | Q3 | P90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (("Workflow", "workflow_nearest_miles"), ("Board", "board_nearest_miles"), ("Workflow minus Board", "signed_workflow_minus_board_miles")):
        item = spatial[key]
        lines.append(f"| {label} | {item['mean']:.3f} | {item['median']:.3f} | {item['q1']:.3f} | {item['q3']:.3f} | {item['p90']:.3f} |")
    lines.extend(
        [
            "",
            "| Threshold | Both within | Both beyond | Workflow only within | Board only within | Changed |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    base_spatial = audit["base_results_audit"]["rq3_spatial"]
    for threshold in ("0.5", "1.0", "2.0"):
        scenario = base_spatial["thresholds"][threshold]["scenarios"]["primary_google_include_vs_board_screened"]
        states = scenario["states"]
        changed = scenario["classification_changed"]
        lines.append(f"| {threshold} mi | {states['neither_beyond']['count']} | {states['both_beyond']['count']} | {states['board_only_beyond']['count']} | {states['google_only_beyond']['count']} | {changed['count']} ({changed['pct']:.2f}%) |")
    lines.extend(
        [
            "",
            "### Predefined spatial sensitivities",
            "",
            "| Threshold | Delivered scenario range | Geometric centroid | Frame interior | Conditionally determinate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    extension = spatial["remaining_prespecified_sensitivities"]
    for threshold in ("0.5", "1.0", "2.0"):
        delivered_range = base_spatial["thresholds"][threshold][
            "delivered_scenario_disagreement_range"
        ]
        extra = extension["thresholds"][threshold]
        centroid = extra["geometric_centroid_origin"]
        interior = extra["collection_frame_interior"]["comparison"]
        determinate = extra["conditional_ranges"][
            "conditionally_determinate_comparison"
        ]
        lines.append(
            f"| {threshold} mi | {delivered_range['min_count']}-{delivered_range['max_count']} ({delivered_range['min_pct']:.2f}-{delivered_range['max_pct']:.2f}%) | "
            f"{centroid['classification_changed_count']}/{centroid['paired_origins']} ({centroid['classification_changed_pct']:.2f}%) | "
            f"{interior['classification_changed_count']}/{interior['paired_origins']} ({interior['classification_changed_pct']:.2f}%) | "
            f"{determinate['classification_changed_count']}/{determinate['paired_origins']} ({determinate['classification_changed_pct']:.2f}%) |"
        )
    lines.extend(
        [
            "",
            "The delivered and conditional ranges are scenario-dependent and are not confidence intervals or unconditional bounds. The complete predefined grid is in `tables/supplement_spatial_sensitivities.csv`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Verification",
            "",
            "| Check | Pass |",
            "|---|---:|",
        ]
    )
    for key, value in audit["verification_checks"].items():
        lines.append(f"| `{key}` | `{str(bool(value)).lower()}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis" / "v2" / f"final_results_{RUN_ID}",
    )
    args = parser.parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"this finalizer is frozen to {RUN_ID}")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("final Results output directory already exists")
    output.mkdir(parents=True)
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir()
    figures_dir.mkdir()

    run_root = ROOT / "data" / "v2" / RUN_ID
    delivery = run_root / "delivery"
    base = build_audit(RUN_ID)
    units = load_completed_units(run_root)
    google_all = build_google_frame(units, delivery)
    board_all, board_cache = load_board()
    boards = prepare_v2_board_frames(board_all, TARGET_ZIPS)
    google_primary = (
        google_all.loc[
            google_all["eligibility_state"].eq("include")
            & google_all["Zip_Code"].isin(TARGET_ZIPS)
        ]
        .sort_values("place_id")
        .reset_index(drop=True)
    )
    board_primary = boards.screened_target.sort_values("license_nbr").reset_index(drop=True)
    linkage_sensitivities = run_linkage_sensitivities(google_primary, board_primary)
    input_manifest = load_input_manifest()
    v1 = pd.read_csv(ROOT / "data" / "Minneapolis-StPaul_pharmacy_20260413_201606.csv", low_memory=False)
    geography_set_audit, geography_rows = geography_sensitivity(google_all, board_all, v1)
    unmatched, unmatched_detail = unmatched_workflow_audit(
        units, google_primary, board_primary, delivery
    )
    spatial = spatial_enrichment(delivery, base["rq3_spatial"])
    spatial["remaining_prespecified_sensitivities"] = spatial_sensitivity_extension(
        google_all, board_all, delivery
    )

    primary_linkage = next(
        row for row in linkage_sensitivities if row["specification"] == "name_address_hungarian"
    )
    primary_geo = geography_rows[0]
    checks = {
        "base_audit_checks": all(base["verification_checks"].values()),
        "completed_units_authenticated": len(units) == 1226,
        "primary_workflow_denominator": len(google_primary) == 339,
        "primary_board_denominator": len(board_primary) == 178,
        "primary_link_count": primary_linkage["linked_pairs"] == 146,
        "primary_geography_reconciles": primary_geo["linked_pairs"] == 146,
        "primary_unmatched_workflow": unmatched["denominator"] == 193,
        "technical_unresolved_zero": base["rq1_candidate_flow"]["technical_unresolved"]["count"] == 0,
        "geography_set_counts": geography_set_audit["historical_union_zip_count"] == 106,
        "spatial_origin_count": spatial["origin_count"] == 327,
        "spatial_changed_counts": [
            base["rq3_spatial"]["thresholds"][threshold]["scenarios"]["primary_google_include_vs_board_screened"]["classification_changed"]["count"]
            for threshold in ("0.5", "1.0", "2.0")
        ] == [46, 42, 4],
        "remaining_spatial_sensitivities_recomputed": spatial[
            "remaining_prespecified_sensitivities"
        ]["primary_delivery_recomputed_exactly"],
        "provider_types_not_invented": unmatched["provider_place_types"]["status"] == "not_available_in_retained_artifacts",
    }
    if not all(checks.values()):
        raise ValueError(f"final Results verification failed: {checks}")

    tables = table_frames(base, unmatched, linkage_sensitivities, geography_rows, spatial)
    for filename, frame in tables.items():
        frame.to_csv(tables_dir / filename, index=False, lineterminator="\n")
    unmatched_detail.to_csv(
        output / "unmatched_workflow_derived_detail.csv", index=False, lineterminator="\n"
    )

    figure_manifest = {
        "figure01_record_flow": figure_record_flow(figures_dir),
        "figure02_core_decisions_confidence": figure_core_decisions(base, figures_dir),
        "figure03_geography_correspondence": figure_geography(geography_rows, figures_dir),
        "figure04_name_recurrence": figure_recurrence(base, figures_dir),
        "figure05_spatial_disagreement_maps": figure_spatial_maps(delivery, figures_dir),
        "figure06_nearest_distance_difference": figure_distance_difference(delivery, figures_dir),
    }

    audit = {
        "schema_version": "nets-ai-v2.2-final-results-package-v1",
        "run_id": RUN_ID,
        "model": "gpt-5-mini",
        "returned_model": "gpt-5-mini-2025-08-07",
        "scope": "Final downstream descriptive analysis and Results production; no provider or model calls",
        "provenance": {
            "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
            "delivery_manifest_sha256": sha256_file(delivery / "run_manifest.json"),
            "delivery_semantic_manifest_sha256": base["delivery_semantic_manifest_sha256"],
            "scientific_implementation_sha256": base["scientific_implementation_sha256"],
            "fixed_frame_sha256": base["fixed_frame_sha256"],
            "forced_choice_classification_sha256": sha256_file(run_root / "classification_forced_choice.jsonl"),
            "frozen_v1_inventory_sha256": input_manifest["data/Minneapolis-StPaul_pharmacy_20260413_201606.csv"]["sha256"],
            "board_cache_sha256": input_manifest["data/board_geocoded.csv"]["sha256"],
            "board_raw_sha256": input_manifest["data/Board/F.1  Pharmacy (PH).csv"]["sha256"],
            "completed_units_sha256": canonical_hash(read_json(run_root / "run_manifest.json")["completed_unit_hashes"]),
            "text_search_rerun": False,
            "provider_calls": False,
            "classification_changes": 0,
            "resume_attempt": 1,
        },
        "base_results_audit": base,
        "rq2": {
            "linkage_method_sensitivities": linkage_sensitivities,
            "geography_set_audit": geography_set_audit,
            "geography_sensitivity": geography_rows,
            "name_recurrence": base["rq2_name_recurrence"],
        },
        "unmatched_workflow_locations": unmatched,
        "rq3": spatial,
        "table_files": {
            filename: sha256_file(tables_dir / filename) for filename in tables
        },
        "figure_files": figure_manifest,
        "verification_checks": checks,
    }
    write_json(output / "final_results_audit.json", audit)
    (output / "final_results_audit.md").write_text(
        render_markdown(audit), encoding="utf-8", newline=""
    )
    write_json(
        output / "figure_manifest.json",
        {
            "schema_version": "nets-ai-thesis-results-figures-v2",
            "run_id": RUN_ID,
            "source_delivery_semantic_manifest_sha256": base["delivery_semantic_manifest_sha256"],
            "figures": figure_manifest,
            "privacy": "aggregate results and tract polygons only; no raw provider text",
            "visualization_contract": {
                "primary_format": "PDF vector",
                "raster_fallback": "RGB PNG at 400 dpi",
                "fixed_canvas_width_inches": 6.5,
                "font_family": "DejaVu Sans",
                "background": "opaque white",
                "color_redundancy": "direct labels, marker shapes, line styles, and hatching as applicable",
                "map_projection": "EPSG:26915",
                "map_basemap": None,
                "source_data_changed": False,
                "scientific_estimand_changed": False,
            },
        },
    )
    output_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "package_manifest.json"
    }
    write_json(
        output / "package_manifest.json",
        {
            "schema_version": "nets-ai-v2.2-final-results-manifest-v1",
            "run_id": RUN_ID,
            "artifact_count": len(output_hashes),
            "artifacts": output_hashes,
            "semantic_manifest_sha256": canonical_hash(output_hashes),
            "all_verification_checks_pass": all(checks.values()),
        },
    )
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "output_dir": output.as_posix(),
                "all_checks_pass": all(checks.values()),
                "figure_count": len(figure_manifest),
                "table_count": len(tables),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
