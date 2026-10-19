"""Reproducible Board-reference and Google straight-line proximity comparison.

The analysis has four declared stages:

1. Apply the same deterministic keyword eligibility screen to every Board record
   before any linkage.  The full licensed inventory is retained as a
   sensitivity definition.
2. Link the frozen Google-retrieved candidates to the Board operational reference using
   deterministic name/address assignment, with name-only and geospatial strategies
   reported as sensitivities.
3. Define repeated-name and single-occurrence groups from exact recurrence of a
   normalized public name across distinct source-local addresses.  No chain list,
   ownership registry, LLM, or manual label is used.
4. Compare projected Euclidean nearest-pharmacy proximity from identical target-ZCTA tract
   internal points in EPSG:26915.  Missing Board geocodes, non-exact coordinate quality,
   and collection-frame edges are represented explicitly instead of silently treated as
   absence or exact location.

All inputs are local.  The script performs no network calls and never overwrites raw
source files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import pyogrio
import shapely

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import geo_match as linkage
from study_design import (
    BOARD_ACCESS_EXCLUSION_RULES,
    TARGET_ZIPS,
    board_exclusion_reason,
    normalize_legal_signature,
    normalize_name_signature,
    normalized_location_key,
    sha256_file,
)


DATA = ROOT / "data"

AI_PATH = DATA / "Minneapolis-StPaul_pharmacy_20260413_201606.csv"
BOARD_PATH = DATA / "board_geocoded.csv"
BOARD_RAW_PATH = DATA / "Board" / "F.1  Pharmacy (PH).csv"
TRACT_PATH = DATA / "tl_2023_27_tract.zip"
ZCTA_PATH = DATA / "tl_2023_us_zcta520.zip"
ACS_PATH = DATA / "acs_tracts_2023.csv"
VEHICLES_PATH = DATA / "acs_tracts_vehicles_2023.csv"
LEGACY_INSTITUTIONAL_PATH = DATA / "audit3_fn_institutional_20260414_172017.csv"
LEGACY_FALSE_NEGATIVE_PATH = DATA / "board_fn_20260414_170601.csv"

DEFAULT_OUTPUT_ROOT = ROOT / "analysis" / "delivery"
DEFAULT_INPUT_MANIFEST = ROOT / "analysis" / "input_manifest.json"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
PRIVATE_OUTPUT = OUTPUT_ROOT / "private"
RESULTS_OUTPUT = OUTPUT_ROOT / "results"
FIGURE_OUTPUT = OUTPUT_ROOT / "figures"

BOARD_CRITERION_OUT = PRIVATE_OUTPUT / "board_keyword_screen_fixed.csv"
BOARD_EXTENDED_OUT = PRIVATE_OUTPUT / "board_keyword_screen_extended.csv"
MATCH_OUT = PRIVATE_OUTPUT / "board_google_record_links.csv"
SOURCE_STATUS_OUT = PRIVATE_OUTPUT / "board_google_source_status.csv"
SIGNATURE_OUT = PRIVATE_OUTPUT / "board_google_name_signature_groups.csv"
TRACT_OUT = RESULTS_OUTPUT / "board_google_tract_proximity.csv"
LINKAGE_TABLE_OUT = RESULTS_OUTPUT / "linkage_strategy_comparison.csv"
PROXIMITY_TABLE_OUT = RESULTS_OUTPUT / "proximity_threshold_comparison.csv"
SUMMARY_OUT = RESULTS_OUTPUT / "analysis_summary.json"
SIGNATURE_SUMMARY_OUT = RESULTS_OUTPUT / "name_signature_summary.json"
FIGURE_OUT = FIGURE_OUTPUT / "board_google_proximity_comparison.png"
FIGURE_PDF_OUT = FIGURE_OUTPUT / "board_google_proximity_comparison.pdf"
SIGNATURE_FIGURE_OUT = FIGURE_OUTPUT / "name_signature_coverage.png"
SIGNATURE_FIGURE_PDF_OUT = FIGURE_OUTPUT / "name_signature_coverage.pdf"

DIST_CRS = "EPSG:26915"
SUPPLY_INPUT_CRS_ASSUMPTION = "EPSG:4326"
SPATIAL_COUNTY_FIPS = ("053", "123")
SPATIAL_COUNTY_NAMES = ("Hennepin", "Ramsey")
M_PER_MILE = 1609.344
ACCESS_THRESHOLDS_MI = (0.5, 1.0, 2.0)
SIGNATURE_THRESHOLDS = (2, 3, 5)
PRIMARY_LINKAGE = "name_address_hungarian"
PRIMARY_ASSIGNMENT_OBJECTIVE = "max_weight"
LINKAGE_THRESHOLD = 75.0
RANDOM_SEED = 20260822
FIGURE_METADATA_TIME = dt.datetime(2026, 8, 22, tzinfo=dt.timezone.utc)


def configure_output_root(output_root: Path) -> None:
    """Route every generated artifact beneath one isolated output directory."""
    global OUTPUT_ROOT, PRIVATE_OUTPUT, RESULTS_OUTPUT, FIGURE_OUTPUT
    global BOARD_CRITERION_OUT, BOARD_EXTENDED_OUT, MATCH_OUT, SOURCE_STATUS_OUT
    global SIGNATURE_OUT, TRACT_OUT, LINKAGE_TABLE_OUT, PROXIMITY_TABLE_OUT
    global SUMMARY_OUT, SIGNATURE_SUMMARY_OUT
    global FIGURE_OUT, FIGURE_PDF_OUT, SIGNATURE_FIGURE_OUT, SIGNATURE_FIGURE_PDF_OUT

    resolved = output_root.resolve()
    analysis_root = (ROOT / "analysis").resolve()
    if resolved == analysis_root or analysis_root not in resolved.parents:
        raise ValueError("output_root must be a dedicated directory beneath analysis/")
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(
            "output_root must be new or empty; refusing to mix canonical outputs with "
            "undeclared or stale files"
        )
    OUTPUT_ROOT = resolved
    PRIVATE_OUTPUT = OUTPUT_ROOT / "private"
    RESULTS_OUTPUT = OUTPUT_ROOT / "results"
    FIGURE_OUTPUT = OUTPUT_ROOT / "figures"
    BOARD_CRITERION_OUT = PRIVATE_OUTPUT / "board_keyword_screen_fixed.csv"
    BOARD_EXTENDED_OUT = PRIVATE_OUTPUT / "board_keyword_screen_extended.csv"
    MATCH_OUT = PRIVATE_OUTPUT / "board_google_record_links.csv"
    SOURCE_STATUS_OUT = PRIVATE_OUTPUT / "board_google_source_status.csv"
    SIGNATURE_OUT = PRIVATE_OUTPUT / "board_google_name_signature_groups.csv"
    TRACT_OUT = RESULTS_OUTPUT / "board_google_tract_proximity.csv"
    LINKAGE_TABLE_OUT = RESULTS_OUTPUT / "linkage_strategy_comparison.csv"
    PROXIMITY_TABLE_OUT = RESULTS_OUTPUT / "proximity_threshold_comparison.csv"
    SUMMARY_OUT = RESULTS_OUTPUT / "analysis_summary.json"
    SIGNATURE_SUMMARY_OUT = RESULTS_OUTPUT / "name_signature_summary.json"
    FIGURE_OUT = FIGURE_OUTPUT / "board_google_proximity_comparison.png"
    FIGURE_PDF_OUT = FIGURE_OUTPUT / "board_google_proximity_comparison.pdf"
    SIGNATURE_FIGURE_OUT = FIGURE_OUTPUT / "name_signature_coverage.png"
    SIGNATURE_FIGURE_PDF_OUT = FIGURE_OUTPUT / "name_signature_coverage.pdf"


def validate_input_manifest(manifest_path: Path) -> dict[str, Any]:
    """Fail closed unless the provenance manifest authenticates every input."""
    resolved = manifest_path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ValueError("input_manifest must be inside the repository")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    entries = manifest.get("inputs")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError("Unsupported or malformed input manifest")
    expected_paths = {
        path.relative_to(ROOT).as_posix()
        for path in (
            AI_PATH,
            BOARD_PATH,
            BOARD_RAW_PATH,
            TRACT_PATH,
            ZCTA_PATH,
            ACS_PATH,
            VEHICLES_PATH,
            LEGACY_INSTITUTIONAL_PATH,
            LEGACY_FALSE_NEGATIVE_PATH,
        )
    }
    declared_paths = {str(entry.get("path", "")) for entry in entries}
    if declared_paths != expected_paths or manifest.get("entry_count") != len(expected_paths):
        raise ValueError(
            "Input manifest path set differs from the canonical analysis contract: "
            f"missing={sorted(expected_paths - declared_paths)}, "
            f"extra={sorted(declared_paths - expected_paths)}"
        )
    for entry in entries:
        relative = Path(str(entry["path"]))
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Manifest input is missing: {relative.as_posix()}")
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            raise ValueError(
                f"Manifest SHA-256 mismatch for {relative.as_posix()}: "
                f"expected {entry.get('sha256')}, actual {actual}"
            )
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(resolved),
        "schema_version": manifest["schema_version"],
        "entry_count": len(entries),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _license_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def _zip5(value: object) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    return digits[:5].zfill(5) if digits else ""


def _ai_city(address: object) -> str:
    parts = [piece.strip() for piece in str(address).split(",")]
    return parts[1] if len(parts) >= 3 else ""


def _recount_signatures(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute recurrence inside the declared evaluation frame."""
    result = frame.copy().reset_index(drop=True)
    frequency = result.groupby("name_component_id")["location_key"].nunique()
    result["signature_location_count"] = (
        result["name_component_id"].map(frequency).fillna(0).astype(int)
    )
    return result


def _assign_exact_name_components(
    frame: pd.DataFrame,
    *,
    operating_column: str,
    legal_column: str | None = None,
) -> pd.DataFrame:
    """Join records sharing an exact operating or exact legal signature.

    Legal and operating namespaces remain distinct.  This deterministic union-find
    step links alternative DBAs under a repeated legal entity without any fuzzy
    threshold or chain dictionary.  The resulting label remains a name component,
    not an ownership determination.
    """
    result = frame.copy().reset_index(drop=True)
    parent = list(range(len(result)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    signature_columns = [("operating", operating_column)]
    if legal_column is not None:
        signature_columns.append(("legal", legal_column))
    for namespace, column in signature_columns:
        first_seen: dict[str, int] = {}
        for index, signature in enumerate(result[column].astype(str)):
            if signature == "unresolved" or not signature.strip():
                continue
            key = f"{namespace}:{signature}"
            if key in first_seen:
                union(first_seen[key], index)
            else:
                first_seen[key] = index

    components: dict[int, list[int]] = {}
    for index in range(len(result)):
        components.setdefault(find(index), []).append(index)

    labels: dict[int, str] = {}
    component_ids: dict[int, str] = {}
    for member_indices in components.values():
        operating_values = [
            str(result.at[index, operating_column])
            for index in member_indices
            if str(result.at[index, operating_column]) != "unresolved"
        ]
        legal_values = (
            [
                str(result.at[index, legal_column])
                for index in member_indices
                if str(result.at[index, legal_column]) != "unresolved"
            ]
            if legal_column is not None
            else []
        )
        candidates = operating_values or legal_values or ["unresolved"]
        counts = Counter(candidates)
        label = sorted(counts, key=lambda value: (-counts[value], value))[0]
        key_material = "|".join(
            sorted(
                {f"o:{value}" for value in operating_values}
                | {f"l:{value}" for value in legal_values}
            )
        ) or "unresolved"
        component_id = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
        for index in member_indices:
            labels[index] = label
            component_ids[index] = component_id

    result["name_signature"] = pd.Series(labels).sort_index().to_numpy()
    result["name_component_id"] = pd.Series(component_ids).sort_index().to_numpy()
    return _recount_signatures(result)


def cluster_bootstrap_interval(
    group: pd.DataFrame,
    *,
    seed: int,
    draws: int = 5000,
) -> tuple[float, float]:
    """Percentile interval for record-level linkage, resampling name components.

    Repeated locations with the same signature are dependent observations.  Each
    bootstrap draw therefore samples entire name-component clusters and recomputes
    the record-weighted linked proportion.  A group with fewer than two signatures
    has no estimable cluster-resampling interval.
    """
    cluster = group.groupby("name_component_id")["linked"].agg(["sum", "count"])
    if len(cluster) < 2:
        return float("nan"), float("nan")
    successes = cluster["sum"].to_numpy(dtype=float)
    totals = cluster["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(cluster), size=(draws, len(cluster)))
    sampled_successes = successes[indices].sum(axis=1)
    sampled_totals = totals[indices].sum(axis=1)
    estimates = sampled_successes / sampled_totals
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def legacy_outcome_conditioned_criterion(board_all: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the historical 405-record denominator for sensitivity only.

    Its 54 exclusions were identified among name-only false negatives, so this
    definition is outcome-conditioned and cannot serve as the primary criterion.
    """
    institutional = pd.read_csv(LEGACY_INSTITUTIONAL_PATH, low_memory=False)
    false_negatives = pd.read_csv(LEGACY_FALSE_NEGATIVE_PATH, low_memory=False)
    institutional["license_nbr"] = institutional["license_nbr"].map(_license_text)
    false_negatives["license_nbr"] = false_negatives["license_nbr"].map(_license_text)
    institutional_ids = set(
        institutional.loc[
            institutional["is_institutional"].fillna(False).astype(bool), "license_nbr"
        ]
    )
    specialty_ids = set(false_negatives["license_nbr"]) - set(institutional["license_nbr"])
    excluded_ids = institutional_ids | specialty_ids
    # The historical denominator was constructed from the already prefiltered
    # 459-record geocode cache, not from all 477 active records in the rebuilt
    # source frame.  Preserve that exact dependency only as a named sensitivity.
    legacy_pool = board_all.loc[board_all["in_geocode_cache"]].copy()
    legacy = legacy_pool.loc[~legacy_pool["license_nbr"].isin(excluded_ids)].copy()
    if len(legacy) != 405 or len(excluded_ids) != 54:
        raise ValueError(
            f"Legacy criterion invariant failed: {len(legacy)=}, {len(excluded_ids)=}"
        )
    return legacy.reset_index(drop=True)


def prepare_inventories() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ai = pd.read_csv(AI_PATH, low_memory=False)
    board_cache = pd.read_csv(BOARD_PATH, low_memory=False)
    board_raw = pd.read_csv(
        BOARD_RAW_PATH,
        encoding="utf-8-sig",
        skiprows=3,
        low_memory=False,
    )

    required_ai = {
        "Company", "Street_Address", "Zip_Code", "Latitude", "Longitude",
        "Calculated_NAICS", "Is_Target_Match", "Business_Website",
    }
    required_board_cache = {
        "facility_name", "license_nbr", "address_line1", "city", "zip5",
        "_lat", "_lon", "match", "mtype",
    }
    required_board_raw = {
        "facility_name", "license_nbr", "description", "address_line1",
        "city", "state", "zip",
    }
    if missing := required_ai - set(ai.columns):
        raise ValueError(f"AI input is missing required columns: {sorted(missing)}")
    if missing := required_board_cache - set(board_cache.columns):
        raise ValueError(f"Board geocode cache is missing required columns: {sorted(missing)}")
    if missing := required_board_raw - set(board_raw.columns):
        raise ValueError(f"Board raw input is missing required columns: {sorted(missing)}")

    ai = ai.reset_index(drop=True).copy()
    ai["ai_row_id"] = np.arange(len(ai), dtype=int)
    ai["zip5"] = ai["Zip_Code"].map(_zip5)
    ai["signature_city"] = ai["Street_Address"].map(_ai_city)
    ai["operating_signature"] = [
        normalize_name_signature(name, city)
        for name, city in zip(ai["Company"], ai["signature_city"])
    ]
    ai["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(ai["Street_Address"], ai["zip5"])
    ]
    ai = _assign_exact_name_components(
        ai,
        operating_column="operating_signature",
    )

    # Rebuild the declared Board frame from the dated raw Board report.  The
    # historical geocode file is a frozen coordinate cache, not the source frame.
    board_raw = board_raw.reset_index(names="board_raw_source_row_id").copy()
    board_raw["license_nbr"] = board_raw["license_nbr"].map(_license_text)
    board_raw["zip5"] = board_raw["zip"].map(_zip5)
    frame_zips = set(TARGET_ZIPS) | set(ai["zip5"].dropna().astype(str))
    board_raw = board_raw.loc[
        board_raw["description"].astype(str).str.startswith("Active", na=False)
        & board_raw["state"].astype(str).eq("MN")
        & board_raw["zip5"].isin(frame_zips)
    ].copy()
    if "e_mail" in board_raw:
        board_raw = board_raw.drop(columns="e_mail")
    board_raw = board_raw.sort_values("license_nbr").reset_index(drop=True)

    board_cache = board_cache.reset_index(drop=True).copy()
    board_cache["license_nbr"] = board_cache["license_nbr"].map(_license_text)
    board_cache["zip5"] = board_cache["zip5"].map(_zip5)
    for label, frame in (("raw Board frame", board_raw), ("Board geocode cache", board_cache)):
        if frame["license_nbr"].eq("").any() or frame["license_nbr"].duplicated().any():
            raise ValueError(f"{label} license numbers must be nonempty and unique")

    raw_ids = set(board_raw["license_nbr"])
    cache_ids = set(board_cache["license_nbr"])
    if extra_cache_ids := sorted(cache_ids - raw_ids):
        raise ValueError(
            "Board geocode cache contains licenses outside the rebuilt frame: "
            f"{extra_cache_ids[:10]}"
        )
    raw_lookup = board_raw.set_index("license_nbr")
    cache_lookup = board_cache.set_index("license_nbr")
    for raw_column, cache_column in (
        ("facility_name", "facility_name"),
        ("address_line1", "address_line1"),
        ("city", "city"),
        ("zip5", "zip5"),
    ):
        left = raw_lookup.loc[cache_lookup.index, raw_column].fillna("").astype(str).str.strip()
        right = cache_lookup[cache_column].fillna("").astype(str).str.strip()
        if not left.eq(right).all():
            bad = left.index[~left.eq(right)].tolist()[:10]
            raise ValueError(f"Board cache differs from raw {raw_column} for licenses {bad}")

    board = board_raw.merge(
        board_cache[["license_nbr", "_lat", "_lon", "match", "mtype"]],
        on="license_nbr",
        how="left",
        validate="one_to_one",
        indicator="_geocode_cache_merge",
    )
    board["in_geocode_cache"] = board["_geocode_cache_merge"].eq("both")
    board = board.drop(columns="_geocode_cache_merge")
    board["board_row_id"] = np.arange(len(board), dtype=int)
    board["eligibility_exclusion_reason"] = board["facility_name"].map(board_exclusion_reason)
    board["access_eligible"] = board["eligibility_exclusion_reason"].eq("")
    board["operating_signature"] = [
        normalize_name_signature(name, city)
        for name, city in zip(board["facility_name"], board["city"])
    ]
    board["legal_signature"] = board["facility_name"].map(normalize_legal_signature)
    board["location_key"] = [
        normalized_location_key(address, zip_code)
        for address, zip_code in zip(board["address_line1"], board["zip5"])
    ]
    board = _assign_exact_name_components(
        board,
        operating_column="operating_signature",
        legal_column="legal_signature",
    )
    board["has_geocode"] = board["_lat"].notna() & board["_lon"].notna()

    if len(ai) != 399:
        raise ValueError(f"Expected 399 AI records, found {len(ai)}")
    if len(board) != 477:
        raise ValueError(f"Expected 477 active Board records in the observed frame, found {len(board)}")
    if len(board_cache) != 459:
        raise ValueError(f"Expected 459 records in the frozen geocode cache, found {len(board_cache)}")
    if not ai[["Latitude", "Longitude"]].notna().all().all():
        raise ValueError("AI inventory contains missing coordinates")

    cache_missing = board.loc[~board["in_geocode_cache"]].copy()
    if cache_missing["access_eligible"].any():
        missing_eligible = cache_missing.loc[
            cache_missing["access_eligible"], "license_nbr"
        ].tolist()
        raise ValueError(
            "The geocode cache omits records that pass the uniform Board screen: "
            f"{missing_eligible}"
        )

    criterion = board.loc[board["access_eligible"]].copy().reset_index(drop=True)
    rule_counts = board.loc[~board["access_eligible"], "eligibility_exclusion_reason"].value_counts()
    audit = {
        "ai_rows": len(ai),
        "ai_model_target_rows": int(ai["Is_Target_Match"].fillna(False).astype(bool).sum()),
        "ai_unique_location_keys": int(ai["location_key"].nunique()),
        "board_raw_active_mn_observed_frame_rows": len(board),
        "board_geocode_cache_rows": len(board_cache),
        "board_geocode_cache_omitted_rows": len(cache_missing),
        "board_geocode_cache_omitted_eligible_rows": int(cache_missing["access_eligible"].sum()),
        "board_geocode_cache_omitted_exclusion_reason_counts": {
            str(key): int(value)
            for key, value in cache_missing["eligibility_exclusion_reason"].value_counts().items()
        },
        "board_access_eligible_rows": len(criterion),
        "board_excluded_rows": int((~board["access_eligible"]).sum()),
        "board_exclusion_reason_counts": {str(key): int(value) for key, value in rule_counts.items()},
        "board_all_active_geocoded_rows": int(board["has_geocode"].sum()),
        "board_access_eligible_geocoded_rows": int(criterion["has_geocode"].sum()),
        "board_access_eligible_missing_geocodes": int((~criterion["has_geocode"]).sum()),
        "board_geocode_match_counts": {
            str(key): int(value) for key, value in board["match"].fillna("missing").value_counts().items()
        },
        "board_geocode_type_counts": {
            str(key): int(value) for key, value in board["mtype"].fillna("missing").value_counts().items()
        },
        "board_unresolved_signatures": int(
            criterion["name_signature"].eq("unresolved").sum()
        ),
        "ai_unresolved_signatures": int(ai["name_signature"].eq("unresolved").sum()),
    }
    return ai, board, audit


def collapse_google_exact_locations(ai: pd.DataFrame) -> pd.DataFrame:
    """Collapse source listings sharing an exact normalized address and ZIP.

    The frozen Google snapshot has no stable Place ID.  Exact address+ZIP is
    therefore the strongest fully automatic deduplication key available locally.
    Every alias and source row ID is retained for audit, while median coordinates
    prevent source-row order from choosing among near-identical map pins.
    """
    if ai["location_key"].isna().any() or ai["location_key"].eq("").any():
        raise ValueError("Google location keys must be nonempty and determinate")

    rows: list[dict[str, Any]] = []
    for location_key, group in ai.groupby("location_key", sort=True):
        names = tuple(sorted(set(group["Company"].fillna("").astype(str))))
        if not names:
            raise ValueError(f"Google location {location_key!r} has no source name")
        preferred_name = sorted(
            names,
            key=lambda value: (
                -("pharmacy" in value.casefold()),
                -len(value.strip()),
                value.casefold(),
            ),
        )[0]
        representative = group.sort_values("ai_row_id").iloc[0].to_dict()
        coordinates = group[["Latitude", "Longitude"]].astype(float)
        maximum_spread_m = 0.0
        coordinate_rows = coordinates.drop_duplicates().to_numpy(dtype=float)
        for left in range(len(coordinate_rows)):
            for right in range(left + 1, len(coordinate_rows)):
                maximum_spread_m = max(
                    maximum_spread_m,
                    linkage.haversine_m(
                        coordinate_rows[left, 0],
                        coordinate_rows[left, 1],
                        coordinate_rows[right, 0],
                        coordinate_rows[right, 1],
                    ),
                )
        representative.update(
            {
                "Company": preferred_name,
                "candidate_names": names,
                "source_ai_row_ids": ";".join(
                    str(value) for value in sorted(group["ai_row_id"].astype(int))
                ),
                "source_listing_count": int(len(group)),
                "Latitude": float(coordinates["Latitude"].median()),
                "Longitude": float(coordinates["Longitude"].median()),
                "max_coordinate_spread_m": round(float(maximum_spread_m), 3),
                "google_location_id": "gl_"
                + hashlib.sha256(str(location_key).encode("utf-8")).hexdigest()[:12],
            }
        )
        rows.append(representative)

    result = pd.DataFrame(rows).sort_values("google_location_id").reset_index(drop=True)
    result["ai_row_id"] = np.arange(len(result), dtype=int)
    result["signature_city"] = result["Street_Address"].map(_ai_city)
    result["operating_signature"] = [
        normalize_name_signature(name, city)
        for name, city in zip(result["Company"], result["signature_city"])
    ]
    result = _assign_exact_name_components(result, operating_column="operating_signature")
    if result["google_location_id"].duplicated().any() or result["location_key"].duplicated().any():
        raise ValueError("Collapsed Google locations must be unique")
    return result


def run_linkage(
    ai: pd.DataFrame,
    criterion: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, list[tuple[int, int, float]]]]:
    strategies: tuple[tuple[str, dict[str, float], str], ...] = (
        ("name_only_greedy", {"w_name": 1.0}, "greedy"),
        ("name_only_hungarian", {"w_name": 1.0}, "hungarian"),
        ("address_only_hungarian", {"w_name": 0.0, "w_addr": 1.0}, "hungarian"),
        ("name_address_hungarian", {"w_name": 1.0, "w_addr": 0.8}, "hungarian"),
        (
            "name_address_geo_hungarian",
            {"w_name": 1.0, "w_addr": 0.8, "w_geo": 0.5, "geo_scale": 300.0},
            "hungarian",
        ),
    )

    pairs_by_strategy: dict[str, list[tuple[int, int, float]]] = {}
    strategy_rows: list[dict[str, Any]] = []
    matched_licenses: dict[str, set[str]] = {}
    for name, configuration, assignment in strategies:
        matrix = linkage.score_matrix(ai, criterion, configuration)
        pairs = (
            linkage.assign_greedy(matrix, LINKAGE_THRESHOLD)
            if assignment == "greedy"
            else linkage.assign_hungarian(
                matrix,
                LINKAGE_THRESHOLD,
                objective=PRIMARY_ASSIGNMENT_OBJECTIVE,
            )
        )
        pairs_by_strategy[name] = pairs
        matched_licenses[name] = {
            str(criterion.iloc[board_index]["license_nbr"])
            for _, board_index, _ in pairs
        }
        linked = len(pairs)
        candidate_linked = linked / len(ai)
        criterion_covered = linked / len(criterion)
        strategy_rows.append(
            {
                "strategy": name,
                "assignment": (
                    assignment
                    if assignment == "greedy"
                    else f"threshold_aware_optional_{PRIMARY_ASSIGNMENT_OBJECTIVE}"
                ),
                "score_weights": configuration,
                "score_scale": "0-100 weighted mean; cross-ZIP score multiplied by 0.85",
                "threshold": LINKAGE_THRESHOLD,
                "linked_pairs": linked,
                "unlinked_ai": len(ai) - linked,
                "unlinked_board": len(criterion) - linked,
                "candidate_linked_pct": round(100.0 * candidate_linked, 2),
                "criterion_covered_pct": round(100.0 * criterion_covered, 2),
            }
        )

    threshold_rows: list[dict[str, Any]] = []
    primary_configuration = {"w_name": 1.0, "w_addr": 0.8}
    primary_matrix = linkage.score_matrix(ai, criterion, primary_configuration)
    for threshold in (70.0, 75.0, 80.0):
        pairs = linkage.assign_hungarian(
            primary_matrix,
            threshold,
            objective=PRIMARY_ASSIGNMENT_OBJECTIVE,
        )
        linked = len(pairs)
        candidate_linked = linked / len(ai)
        criterion_covered = linked / len(criterion)
        threshold_rows.append(
            {
                "threshold": threshold,
                "linked_pairs": linked,
                "candidate_linked_pct": round(100.0 * candidate_linked, 2),
                "criterion_covered_pct": round(100.0 * criterion_covered, 2),
            }
        )

    objective_rows: list[dict[str, Any]] = []
    for objective in ("max_weight", "max_cardinality_then_weight"):
        objective_pairs = linkage.assign_hungarian(
            primary_matrix,
            LINKAGE_THRESHOLD,
            objective=objective,
        )
        objective_rows.append(
            {
                "objective": objective,
                "linked_pairs": len(objective_pairs),
                "total_similarity": round(sum(pair[2] for pair in objective_pairs), 4),
            }
        )

    overlap_rows: list[dict[str, Any]] = []
    primary_set = matched_licenses[PRIMARY_LINKAGE]
    for name, _, _ in strategies:
        current = matched_licenses[name]
        overlap_rows.append(
            {
                "strategy": name,
                "matched_board_licenses": len(current),
                "overlap_with_primary": len(current & primary_set),
                "only_this_strategy": len(current - primary_set),
                "only_primary": len(primary_set - current),
            }
        )

    return {
        "criterion_rows": len(criterion),
        "ai_rows": len(ai),
        "primary_strategy": PRIMARY_LINKAGE,
        "primary_assignment_objective": PRIMARY_ASSIGNMENT_OBJECTIVE,
        "strategies": strategy_rows,
        "primary_threshold_sensitivity": threshold_rows,
        "primary_objective_sensitivity": objective_rows,
        "matched_license_overlap": overlap_rows,
    }, pairs_by_strategy


def build_pair_table(
    ai: pd.DataFrame,
    criterion: pd.DataFrame,
    pairs: Iterable[tuple[int, int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ai_index, board_index, score in pairs:
        ai_row = ai.iloc[ai_index]
        board_row = criterion.iloc[board_index]
        distance_m = float("nan")
        if pd.notna(board_row["_lat"]) and pd.notna(board_row["_lon"]):
            distance_m = linkage.haversine_m(
                float(ai_row["Latitude"]),
                float(ai_row["Longitude"]),
                float(board_row["_lat"]),
                float(board_row["_lon"]),
            )
        rows.append(
            {
                "ai_row_id": int(ai_row["ai_row_id"]),
                "google_location_id": ai_row.get("google_location_id", ""),
                "google_source_ai_row_ids": ai_row.get(
                    "source_ai_row_ids", str(int(ai_row["ai_row_id"]))
                ),
                "google_source_listing_count": int(ai_row.get("source_listing_count", 1)),
                "board_row_id": int(board_row["board_row_id"]),
                "board_license_nbr": str(board_row["license_nbr"]),
                "ai_name": ai_row["Company"],
                "board_name": board_row["facility_name"],
                "ai_address": ai_row["Street_Address"],
                "board_address": board_row["address_line1"],
                "ai_zip5": ai_row["zip5"],
                "board_zip5": board_row["zip5"],
                "composite_score": round(float(score), 4),
                "name_similarity": round(
                    linkage.candidate_name_sim(
                        ai_row.get("candidate_names", ai_row["Company"]),
                        board_row["facility_name"],
                    ),
                    4,
                ),
                "address_similarity": round(
                    linkage.address_sim(ai_row["Street_Address"], board_row["address_line1"]), 4
                ),
                "coordinate_distance_m": round(distance_m, 3) if np.isfinite(distance_m) else np.nan,
                "ai_name_signature": ai_row["name_signature"],
                "board_name_signature": board_row["name_signature"],
                "ai_name_component_id": ai_row["name_component_id"],
                "board_name_component_id": board_row["name_component_id"],
                "ai_signature_location_count": int(ai_row["signature_location_count"]),
                "board_signature_location_count": int(board_row["signature_location_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["board_license_nbr", "ai_row_id"]).reset_index(drop=True)


def build_source_status(
    ai: pd.DataFrame,
    criterion: pd.DataFrame,
    pair_table: pd.DataFrame,
) -> pd.DataFrame:
    """Return one auditable status row for every record in both primary sources."""
    ai_links = pair_table.set_index("ai_row_id")["board_license_nbr"].astype(str)
    board_links = pair_table.set_index("board_license_nbr")["ai_row_id"].astype(int)

    ai_status = pd.DataFrame(
        {
            "source": "Google exact-address clusters",
            "source_record_id": ai["ai_row_id"].astype(str),
            "stable_location_id": ai.get("google_location_id", ai["ai_row_id"].astype(str)),
            "source_listing_count": ai.get(
                "source_listing_count", pd.Series(1, index=ai.index, dtype=int)
            ),
            "name": ai["Company"],
            "address": ai["Street_Address"],
            "zip5": ai["zip5"],
            "name_signature": ai["name_signature"],
            "name_component_id": ai["name_component_id"],
            "signature_location_count": ai["signature_location_count"],
            "linked": ai["ai_row_id"].isin(ai_links.index),
            "linked_source_record_id": ai["ai_row_id"].map(ai_links),
        }
    )
    board_status = pd.DataFrame(
        {
            "source": "Board keyword-screen reference",
            "source_record_id": criterion["license_nbr"].astype(str),
            "name": criterion["facility_name"],
            "address": criterion["address_line1"],
            "zip5": criterion["zip5"],
            "name_signature": criterion["name_signature"],
            "name_component_id": criterion["name_component_id"],
            "signature_location_count": criterion["signature_location_count"],
            "linked": criterion["license_nbr"].astype(str).isin(board_links.index),
            "linked_source_record_id": criterion["license_nbr"].astype(str).map(board_links),
        }
    )
    status = pd.concat([ai_status, board_status], ignore_index=True)
    status["linked_source_record_id"] = status["linked_source_record_id"].astype("string")
    return status.sort_values(["source", "source_record_id"]).reset_index(drop=True)


def signature_analysis(
    ai: pd.DataFrame,
    criterion: pd.DataFrame,
    pairs_by_strategy: dict[str, list[tuple[int, int, float]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    bootstrap_seed_offset = 0
    for strategy, pairs in pairs_by_strategy.items():
        matched_ai = {int(ai.iloc[ai_index]["ai_row_id"]) for ai_index, _, _ in pairs}
        matched_board = {
            str(criterion.iloc[board_index]["license_nbr"])
            for _, board_index, _ in pairs
        }
        for recurrence_threshold in SIGNATURE_THRESHOLDS:
            below_label = (
                "single-occurrence name component"
                if recurrence_threshold == 2
                else "below-threshold name component"
            )
            for source, frame, id_column, matched_ids, linked_label in (
                (
                    "Board keyword-screen reference",
                    criterion,
                    "license_nbr",
                    matched_board,
                    "Google-linked",
                ),
                (
                    "Google exact-address clusters",
                    ai,
                    "ai_row_id",
                    matched_ai,
                    "Board-linked",
                ),
            ):
                work = frame[
                    [
                        id_column,
                        "name_signature",
                        "name_component_id",
                        "signature_location_count",
                        "location_key",
                    ]
                ].copy()
                work["source"] = source
                work["linkage_strategy"] = strategy
                work["recurrence_threshold"] = recurrence_threshold
                work["name_component_group"] = np.where(
                    work["name_signature"].eq("unresolved"),
                    "unresolved",
                    np.where(
                        work["signature_location_count"] >= recurrence_threshold,
                        "repeated-name component",
                        below_label,
                    ),
                )
                work["linked"] = work[id_column].astype(str).isin(
                    {str(value) for value in matched_ids}
                )
                work["source_record_id"] = work[id_column].astype(str)
                work["linked_label"] = linked_label
                detail_frames.append(
                    work[
                        [
                            "source",
                            "source_record_id",
                            "name_signature",
                            "name_component_id",
                            "location_key",
                            "signature_location_count",
                            "linkage_strategy",
                            "recurrence_threshold",
                            "name_component_group",
                            "linked",
                            "linked_label",
                        ]
                    ]
                )

                for group_code, group_name in (
                    ("repeated", "repeated-name component"),
                    ("below_threshold", below_label),
                    ("unresolved", "unresolved"),
                ):
                    group = work.loc[work["name_component_group"].eq(group_name)]
                    denominator = len(group)
                    numerator = int(group["linked"].sum())
                    lower, upper = cluster_bootstrap_interval(
                        group,
                        seed=RANDOM_SEED + bootstrap_seed_offset,
                    )
                    bootstrap_seed_offset += 1
                    summary_rows.append(
                        {
                            "source": source,
                            "linkage_strategy": strategy,
                            "recurrence_threshold": recurrence_threshold,
                            "group_code": group_code,
                            "group": group_name,
                            "records": denominator,
                            "distinct_name_components": int(
                                group["name_component_id"].nunique()
                            ),
                            "linked_records": numerator,
                            "linked_pct": round(100.0 * numerator / denominator, 2)
                            if denominator
                            else None,
                            "cluster_bootstrap_95_low_pct": round(100.0 * lower, 2)
                            if np.isfinite(lower)
                            else None,
                            "cluster_bootstrap_95_high_pct": round(100.0 * upper, 2)
                            if np.isfinite(upper)
                            else None,
                        }
                    )

    summary_frame = pd.DataFrame(summary_rows)
    gaps: list[dict[str, Any]] = []
    for strategy in pairs_by_strategy:
        for source in (
            "Board keyword-screen reference",
            "Google exact-address clusters",
        ):
            for recurrence_threshold in SIGNATURE_THRESHOLDS:
                subset = summary_frame.loc[
                    summary_frame["source"].eq(source)
                    & summary_frame["linkage_strategy"].eq(strategy)
                    & summary_frame["recurrence_threshold"].eq(recurrence_threshold)
                ].set_index("group_code")
                repeated = subset.loc["repeated", "linked_pct"]
                below = subset.loc["below_threshold", "linked_pct"]
                gaps.append(
                    {
                        "source": source,
                        "linkage_strategy": strategy,
                        "recurrence_threshold": recurrence_threshold,
                        "repeated_minus_below_threshold_percentage_points": round(
                            float(repeated - below), 2
                        ),
                    }
                )

    return pd.concat(detail_frames, ignore_index=True), {
        "definition": (
            "Exact legal-name or DBA/public-name components followed by recurrence "
            "across distinct source-local addresses; no fuzzy clustering, chain "
            "dictionary, ownership inference, LLM label, or manual label"
        ),
        "group_labels": {
            "repeated-name component": "component occurs at least k distinct locations",
            "below-threshold name component": (
                "component occurs at fewer than k distinct locations; this equals a "
                "single-occurrence group only when k=2"
            ),
            "single-occurrence name component": "component occurs at one distinct location",
            "unresolved": "normalization yielded no distinctive name token",
        },
        "uncertainty": (
            "Exploratory 95% percentile bootstrap intervals resample complete name-component "
            "clusters (5,000 draws; deterministic seeds), preserving within-component dependence"
        ),
        "thresholds": list(SIGNATURE_THRESHOLDS),
        "rows": summary_rows,
        "repeated_minus_below_threshold_gaps": gaps,
    }


def _read_zctas(
    zip_codes: set[str],
) -> tuple[gpd.GeoDataFrame, list[str], dict[str, Any]]:
    # The bounding box is expressed in the source NAD83 geographic CRS and only
    # limits local file I/O; the exact ZCTA set is selected by its stable code.
    zctas = gpd.read_file(
        f"zip://{ZCTA_PATH.as_posix()}",
        bbox=(-94.6, 44.1, -91.9, 46.1),
    )
    if zctas.crs is None:
        raise ValueError("ZCTA source has no CRS metadata")
    key = "ZCTA5CE20"
    if key not in zctas.columns:
        raise ValueError(f"ZCTA source is missing {key}")
    zctas[key] = zctas[key].astype(str).str.zfill(5)
    geometry_audit = {
        "bbox_rows_read": len(zctas),
        "duplicate_zcta_ids": int(zctas[key].duplicated().sum()),
        "null_geometries": int(zctas.geometry.isna().sum()),
        "empty_geometries": int(zctas.geometry.is_empty.sum()),
        "invalid_geometries": int((~zctas.geometry.is_valid).sum()),
        "source_crs": str(zctas.crs),
    }
    if any(
        geometry_audit[field]
        for field in (
            "duplicate_zcta_ids",
            "null_geometries",
            "empty_geometries",
            "invalid_geometries",
        )
    ):
        raise ValueError(f"ZCTA geometry audit failed: {geometry_audit}")
    selected = zctas.loc[zctas[key].isin(zip_codes), [key, "geometry"]].copy()
    missing = sorted(zip_codes - set(selected[key]))
    geometry_audit["selected_rows"] = len(selected)
    return selected, missing, geometry_audit


def _transformer_audit(source_crs: object, target_crs: object) -> dict[str, Any]:
    group = pyproj.transformer.TransformerGroup(
        source_crs,
        target_crs,
        always_xy=True,
    )
    if not group.transformers:
        raise ValueError(f"No coordinate transformation available: {source_crs} -> {target_crs}")
    transformer = group.transformers[0]
    return {
        "source_crs": str(pyproj.CRS.from_user_input(source_crs)),
        "target_crs": str(pyproj.CRS.from_user_input(target_crs)),
        "best_available": bool(group.best_available),
        "selected_operation": transformer.description,
        "declared_accuracy_m": (
            None if transformer.accuracy is None or transformer.accuracy < 0 else transformer.accuracy
        ),
        "unavailable_operation_count": len(group.unavailable_operations),
    }


def _nearest_details(
    origins: gpd.GeoDataFrame,
    supply: pd.DataFrame,
    latitude: str,
    longitude: str,
    source_id: str,
) -> pd.DataFrame:
    valid = supply[latitude].notna() & supply[longitude].notna()
    if source_id not in supply.columns:
        raise ValueError(f"Supply inventory is missing nearest-source ID column {source_id}")
    points = gpd.GeoDataFrame(
        supply.loc[valid].copy(),
        geometry=gpd.points_from_xy(
            supply.loc[valid, longitude], supply.loc[valid, latitude]
        ),
        crs=SUPPLY_INPUT_CRS_ASSUMPTION,
    ).to_crs(DIST_CRS)
    if points.empty:
        raise ValueError("Supply inventory contains no valid coordinates")
    joined = gpd.sjoin_nearest(
        origins[["GEOID", "geometry"]],
        points[[source_id, "geometry"]],
        how="left",
        distance_col="nearest_m",
    )
    joined[source_id] = joined[source_id].astype(str)
    minimum = joined.groupby("GEOID")["nearest_m"].min()
    joined = joined.loc[
        np.isclose(
            joined["nearest_m"].to_numpy(dtype=float),
            joined["GEOID"].map(minimum).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-7,
        )
    ].copy()
    tie_count = joined.groupby("GEOID")[source_id].nunique()
    nearest_ids = joined.groupby("GEOID")[source_id].agg(
        lambda values: ";".join(sorted(set(values)))
    )
    result = pd.DataFrame(index=origins.index)
    result["nearest_m"] = origins["GEOID"].map(minimum)
    result["nearest_tie_count"] = origins["GEOID"].map(tie_count).astype(int)
    result["nearest_source_ids"] = origins["GEOID"].map(nearest_ids)
    if result["nearest_m"].isna().any() or result["nearest_tie_count"].lt(1).any():
        raise ValueError("Nearest-distance join left unresolved tract origins")
    return result


def _quartile(series: pd.Series) -> pd.Series:
    return pd.qcut(series, q=4, labels=[1, 2, 3, 4], duplicates="raise").astype(int)


def _rate_percent(binary: pd.Series) -> float:
    return round(100.0 * float(binary.mean()), 2) if len(binary) else float("nan")


def _weighted_rate_percent(binary: pd.Series, weights: pd.Series) -> float | None:
    valid = binary.notna() & weights.notna() & weights.gt(0)
    if not valid.any() or float(weights.loc[valid].sum()) <= 0:
        return None
    return round(
        100.0
        * float(
            np.average(
                binary.loc[valid].astype(float),
                weights=weights.loc[valid].astype(float),
            )
        ),
        2,
    )


def _discordance_type(board_beyond: pd.Series, google_beyond: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                board_beyond & google_beyond,
                ~board_beyond & ~google_beyond,
                ~board_beyond & google_beyond,
                board_beyond & ~google_beyond,
            ],
            [
                "both_beyond_threshold",
                "both_within_threshold",
                "google_only_beyond_threshold",
                "board_only_beyond_threshold",
            ],
            default="unresolved",
        ),
        index=board_beyond.index,
        dtype="string",
    )


def _paired_threshold_summary(board_beyond: pd.Series, google_beyond: pd.Series) -> dict[str, Any]:
    """Describe paired threshold classifications without implying model accuracy."""
    board_beyond = board_beyond.astype(bool)
    google_beyond = google_beyond.astype(bool)
    both_beyond = int((board_beyond & google_beyond).sum())
    both_within = int((~board_beyond & ~google_beyond).sum())
    google_only_beyond = int((~board_beyond & google_beyond).sum())
    board_only_beyond = int((board_beyond & ~google_beyond).sum())
    total = len(board_beyond)
    board_positive = int(board_beyond.sum())
    board_negative = total - board_positive
    return {
        "n": total,
        "both_beyond_threshold": both_beyond,
        "both_within_threshold": both_within,
        "google_only_beyond_threshold": google_only_beyond,
        "board_only_beyond_threshold": board_only_beyond,
        "agreement_pct": round(100.0 * (both_beyond + both_within) / total, 2)
        if total
        else None,
        "google_beyond_given_board_beyond_pct": round(
            100.0 * both_beyond / board_positive, 2
        )
        if board_positive
        else None,
        "google_within_given_board_within_pct": round(
            100.0 * both_within / board_negative, 2
        )
        if board_negative
        else None,
    }


def spatial_analysis(
    ai: pd.DataFrame,
    criterion: pd.DataFrame,
    legacy_criterion: pd.DataFrame,
    board_all: pd.DataFrame,
    *,
    ai_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ai_raw = ai if ai_raw is None else ai_raw
    board_exact_geocode = criterion.loc[
        criterion["has_geocode"] & criterion["mtype"].eq("Exact")
    ].copy()
    tracts = gpd.read_file(f"zip://{TRACT_PATH.as_posix()}")
    if tracts.crs is None:
        raise ValueError("Tract source has no CRS metadata")
    tracts = tracts.loc[tracts["COUNTYFP"].isin(SPATIAL_COUNTY_FIPS)].copy()
    # TIGER and the ACS cache both contain a column named NAME.  Preserve the
    # TIGER label explicitly so the ACS human-readable NAME remains unambiguous.
    tracts = tracts.rename(columns={"NAME": "tiger_tract_name"})
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    if tracts["GEOID"].duplicated().any():
        raise ValueError("Tract GEOIDs are not unique")

    acs = pd.read_csv(ACS_PATH, dtype={"GEOID": str})
    vehicles = pd.read_csv(VEHICLES_PATH, dtype={"GEOID": str})
    if acs["GEOID"].duplicated().any() or vehicles["GEOID"].duplicated().any():
        raise ValueError("ACS GEOIDs must be unique before merging")
    tracts = tracts.merge(acs, on="GEOID", how="left", validate="one_to_one")
    tracts = tracts.merge(vehicles, on="GEOID", how="left", validate="one_to_one")
    valid_income = pd.to_numeric(tracts["med_hh_income"], errors="coerce").gt(0)
    invalid_or_missing_income_rows = int((~valid_income).sum())
    missing_household_rows = int(tracts["total_hh"].isna().sum())
    nonpositive_household_rows = int(tracts["total_hh"].fillna(0).le(0).sum())
    eligible = tracts.loc[
        valid_income
        & tracts["total_hh"].notna()
        & tracts["total_hh"].gt(0)
    ].copy()

    frame_zips = (
        set(TARGET_ZIPS)
        | set(ai["zip5"].dropna().astype(str))
        | set(ai_raw["zip5"].dropna().astype(str))
    )
    zctas, missing_frame_zctas, zcta_geometry_audit = _read_zctas(frame_zips)
    target_zctas = zctas.loc[zctas["ZCTA5CE20"].isin(TARGET_ZIPS)].copy()
    if missing_target := sorted(set(TARGET_ZIPS) - set(target_zctas["ZCTA5CE20"])):
        raise ValueError(f"Target ZCTA geometries are missing: {missing_target}")

    eligible_projected = eligible.to_crs(DIST_CRS).copy()
    axis_units = [axis.unit_name.lower() for axis in eligible_projected.crs.axis_info]
    if not axis_units or any(unit not in {"metre", "meter"} for unit in axis_units):
        raise ValueError(f"Unexpected EPSG:26915 axis units: {axis_units}")
    tract_geometry = eligible_projected.geometry.copy()
    centroid_geometry = eligible_projected.geometry.centroid
    eligible_projected["centroid_within_own_tract"] = [
        tract.covers(centroid)
        for tract, centroid in zip(tract_geometry, centroid_geometry)
    ]
    internal_points = gpd.GeoSeries(
        gpd.points_from_xy(
            pd.to_numeric(eligible["INTPTLON"], errors="coerce"),
            pd.to_numeric(eligible["INTPTLAT"], errors="coerce"),
        ),
        index=eligible.index,
        crs=tracts.crs,
    ).to_crs(DIST_CRS)
    eligible_projected["centroid_geometry"] = centroid_geometry
    eligible_projected["internal_point_within_own_tract"] = [
        tract.covers(point)
        for tract, point in zip(tract_geometry, internal_points)
    ]
    if not eligible_projected["internal_point_within_own_tract"].all():
        raise ValueError("TIGER internal points must fall within their source tract")
    # Census-provided internal points are the primary tract origins because they
    # are stable and guaranteed to lie inside their own tract.  Geometric
    # centroids are retained as a declared origin-definition sensitivity.
    eligible_projected["geometry"] = internal_points

    target_union = target_zctas.to_crs(DIST_CRS).geometry.union_all()
    frame_union = zctas.to_crs(DIST_CRS).geometry.union_all()
    in_target = eligible_projected.geometry.within(target_union) | eligible_projected.geometry.touches(
        target_union
    )
    origins = eligible_projected.loc[in_target].copy().reset_index(drop=True)
    if origins["GEOID"].duplicated().any():
        raise ValueError("Fixed tract origins are not unique")
    origins["frame_edge_m"] = origins.geometry.distance(frame_union.boundary)
    target_join = gpd.sjoin(
        origins[["GEOID", "geometry"]],
        target_zctas.to_crs(DIST_CRS)[["ZCTA5CE20", "geometry"]],
        how="left",
        predicate="intersects",
    )
    if target_join["GEOID"].duplicated().any() or target_join["ZCTA5CE20"].isna().any():
        raise ValueError("Each fixed tract origin must map to exactly one target ZCTA")
    origins["target_zcta"] = origins["GEOID"].map(
        target_join.set_index("GEOID")["ZCTA5CE20"]
    )
    origins["income_quartile"] = _quartile(origins["med_hh_income"])
    origins["nonwhite_quartile"] = _quartile(origins["pct_nonwhite"])

    nearest_definitions = (
        ("google", ai, "Latitude", "Longitude", "google_location_id"),
        ("google_raw_candidate", ai_raw, "Latitude", "Longitude", "google_location_id"),
        ("board", criterion, "_lat", "_lon", "license_nbr"),
        (
            "board_exact_geocode_only",
            board_exact_geocode,
            "_lat",
            "_lon",
            "license_nbr",
        ),
        ("board_legacy_405", legacy_criterion, "_lat", "_lon", "license_nbr"),
        ("board_all_licensed", board_all, "_lat", "_lon", "license_nbr"),
    )
    for prefix, supply, latitude, longitude, source_id in nearest_definitions:
        details = _nearest_details(origins, supply, latitude, longitude, source_id)
        origins[f"{prefix}_nearest_m"] = details["nearest_m"]
        origins[f"{prefix}_nearest_tie_count"] = details["nearest_tie_count"]
        origins[f"{prefix}_nearest_source_ids"] = details["nearest_source_ids"]
    origins["google_minus_board_m"] = (
        origins["google_nearest_m"] - origins["board_nearest_m"]
    )
    origins["abs_google_minus_board_m"] = origins["google_minus_board_m"].abs()

    centroid_origins = gpd.GeoDataFrame(
        origins.drop(columns="geometry").copy(),
        geometry=origins["centroid_geometry"],
        crs=DIST_CRS,
    )
    origins["centroid_google_nearest_m"] = _nearest_details(
        centroid_origins,
        ai,
        "Latitude",
        "Longitude",
        "google_location_id",
    )["nearest_m"]
    origins["centroid_board_nearest_m"] = _nearest_details(
        centroid_origins,
        criterion,
        "_lat",
        "_lon",
        "license_nbr",
    )["nearest_m"]

    missing_board = criterion.loc[~criterion["has_geocode"]].copy()
    missing_board_mailing_zips = set(missing_board["zip5"].dropna().astype(str))
    missing_mailing_zip_zcta_polygons = zctas.loc[
        zctas["ZCTA5CE20"].isin(missing_board_mailing_zips)
    ].to_crs(DIST_CRS)
    missing_board_mailing_zip_zctas_without_geometry = sorted(
        missing_board_mailing_zips
        - set(missing_mailing_zip_zcta_polygons["ZCTA5CE20"])
    )
    if len(missing_mailing_zip_zcta_polygons):
        missing_union = missing_mailing_zip_zcta_polygons.geometry.union_all()
        origins["missing_board_mailing_zip_zcta_min_m"] = origins.geometry.distance(
            missing_union
        )
    else:
        origins["missing_board_mailing_zip_zcta_min_m"] = np.inf
    if missing_board_mailing_zip_zctas_without_geometry:
        # Under the mailing-ZIP/ZCTA condition, an unresolved ZIP geometry leaves every
        # beyond-threshold classification potentially alterable.
        origins["missing_board_mailing_zip_zcta_min_m"] = 0.0

    threshold_summaries: list[dict[str, Any]] = []
    for threshold_mi in ACCESS_THRESHOLDS_MI:
        threshold_m = threshold_mi * M_PER_MILE
        suffix = str(threshold_mi).replace(".", "_") + "mi"
        ai_beyond_column = f"google_beyond_{suffix}"
        raw_ai_beyond_column = f"google_raw_candidate_beyond_{suffix}"
        board_beyond_column = f"board_beyond_{suffix}"
        board_exact_column = f"board_exact_geocode_only_beyond_{suffix}"
        board_legacy_column = f"board_legacy_405_beyond_{suffix}"
        board_all_column = f"board_all_licensed_beyond_{suffix}"
        ai_determinate_column = f"google_conditionally_determinate_{suffix}"
        board_determinate_column = f"board_conditionally_determinate_{suffix}"

        origins[ai_beyond_column] = origins["google_nearest_m"].gt(threshold_m)
        origins[raw_ai_beyond_column] = origins["google_raw_candidate_nearest_m"].gt(
            threshold_m
        )
        origins[board_beyond_column] = origins["board_nearest_m"].gt(threshold_m)
        origins[board_exact_column] = origins["board_exact_geocode_only_nearest_m"].gt(
            threshold_m
        )
        origins[board_legacy_column] = origins["board_legacy_405_nearest_m"].gt(
            threshold_m
        )
        origins[board_all_column] = origins["board_all_licensed_nearest_m"].gt(threshold_m)
        origins[f"discordance_type_{suffix}"] = _discordance_type(
            origins[board_beyond_column], origins[ai_beyond_column]
        )
        origins[ai_determinate_column] = (
            origins["google_nearest_m"].le(threshold_m)
            | origins["frame_edge_m"].gt(threshold_m)
        )
        origins[board_determinate_column] = (
            origins["board_nearest_m"].le(threshold_m)
            | (
                origins["frame_edge_m"].gt(threshold_m)
                & origins["missing_board_mailing_zip_zcta_min_m"].gt(threshold_m)
            )
        )
        both_determinate = origins[ai_determinate_column] & origins[board_determinate_column]
        determinate = origins.loc[both_determinate]

        board_definite_covered = origins["board_nearest_m"].le(threshold_m)
        board_definite_low = (
            origins["board_nearest_m"].gt(threshold_m)
            & origins["frame_edge_m"].gt(threshold_m)
            & origins["missing_board_mailing_zip_zcta_min_m"].gt(threshold_m)
        )
        board_uncertain = ~(board_definite_covered | board_definite_low)
        ai_definite_covered = origins["google_nearest_m"].le(threshold_m)
        ai_definite_low = origins["google_nearest_m"].gt(threshold_m) & origins[
            "frame_edge_m"
        ].gt(threshold_m)
        ai_uncertain = ~(ai_definite_covered | ai_definite_low)

        edge_interior = origins.loc[origins["frame_edge_m"].gt(threshold_m)]
        centroid_google_beyond = origins["centroid_google_nearest_m"].gt(threshold_m)
        centroid_board_beyond = origins["centroid_board_nearest_m"].gt(threshold_m)

        threshold_summaries.append(
            {
                "threshold_miles": threshold_mi,
                "threshold_meters": round(threshold_m, 3),
                "all_fixed_origins": len(origins),
                "google_beyond_threshold_count": int(origins[ai_beyond_column].sum()),
                "google_beyond_threshold_pct": _rate_percent(origins[ai_beyond_column]),
                "google_raw_candidate_beyond_threshold_count": int(
                    origins[raw_ai_beyond_column].sum()
                ),
                "google_raw_candidate_beyond_threshold_pct": _rate_percent(
                    origins[raw_ai_beyond_column]
                ),
                "board_beyond_threshold_count": int(origins[board_beyond_column].sum()),
                "board_beyond_threshold_pct": _rate_percent(origins[board_beyond_column]),
                "board_exact_geocode_only_beyond_threshold_count": int(
                    origins[board_exact_column].sum()
                ),
                "board_exact_geocode_only_beyond_threshold_pct": _rate_percent(
                    origins[board_exact_column]
                ),
                "board_exact_geocode_only_minus_primary_count": int(
                    origins[board_exact_column].sum() - origins[board_beyond_column].sum()
                ),
                "legacy_405_beyond_threshold_count": int(origins[board_legacy_column].sum()),
                "legacy_405_beyond_threshold_pct": _rate_percent(origins[board_legacy_column]),
                "board_all_licensed_beyond_threshold_count": int(
                    origins[board_all_column].sum()
                ),
                "board_all_licensed_beyond_threshold_pct": _rate_percent(
                    origins[board_all_column]
                ),
                "google_minus_board_percentage_points": round(
                    _rate_percent(origins[ai_beyond_column])
                    - _rate_percent(origins[board_beyond_column]),
                    2,
                ),
                "board_beyond_rate_conditional_lower_pct": round(
                    100.0 * board_definite_low.mean(), 2
                ),
                "board_beyond_rate_conditional_upper_pct": round(
                    100.0 * (board_definite_low | board_uncertain).mean(), 2
                ),
                "board_conditionally_unresolved_count": int(board_uncertain.sum()),
                "google_beyond_rate_conditional_lower_pct": round(
                    100.0 * ai_definite_low.mean(), 2
                ),
                "google_beyond_rate_conditional_upper_pct": round(
                    100.0 * (ai_definite_low | ai_uncertain).mean(), 2
                ),
                "google_conditionally_unresolved_count": int(ai_uncertain.sum()),
                "observed_supply_comparison": _paired_threshold_summary(
                    origins[board_beyond_column], origins[ai_beyond_column]
                ),
                "google_vs_board_exact_geocode_only_comparison": _paired_threshold_summary(
                    origins[board_exact_column], origins[ai_beyond_column]
                ),
                "both_sources_conditionally_determinate_count": int(
                    both_determinate.sum()
                ),
                "conditionally_determinate_comparison": _paired_threshold_summary(
                    determinate[board_beyond_column], determinate[ai_beyond_column]
                ),
                "population_weighted": {
                    "google_beyond_pct": _weighted_rate_percent(
                        origins[ai_beyond_column], origins["total_pop"]
                    ),
                    "board_beyond_pct": _weighted_rate_percent(
                        origins[board_beyond_column], origins["total_pop"]
                    ),
                },
                "household_weighted": {
                    "google_beyond_pct": _weighted_rate_percent(
                        origins[ai_beyond_column], origins["total_hh"]
                    ),
                    "board_beyond_pct": _weighted_rate_percent(
                        origins[board_beyond_column], origins["total_hh"]
                    ),
                },
                "collection_frame_interior_sensitivity": {
                    "tracts": len(edge_interior),
                    "google_beyond_count": int(edge_interior[ai_beyond_column].sum()),
                    "google_beyond_pct": _rate_percent(edge_interior[ai_beyond_column]),
                    "board_beyond_count": int(edge_interior[board_beyond_column].sum()),
                    "board_beyond_pct": _rate_percent(edge_interior[board_beyond_column]),
                    "comparison": _paired_threshold_summary(
                        edge_interior[board_beyond_column], edge_interior[ai_beyond_column]
                    ),
                },
                "geometric_centroid_origin_sensitivity": {
                    "tracts": len(origins),
                    "google_beyond_count": int(centroid_google_beyond.sum()),
                    "google_beyond_pct": _rate_percent(centroid_google_beyond),
                    "board_beyond_count": int(centroid_board_beyond.sum()),
                    "board_beyond_pct": _rate_percent(centroid_board_beyond),
                    "comparison": _paired_threshold_summary(
                        centroid_board_beyond, centroid_google_beyond
                    ),
                },
                "four_meter_transform_tolerance": {
                    "google_origins_within_4m_of_threshold": int(
                        origins["google_nearest_m"].sub(threshold_m).abs().le(4.0).sum()
                    ),
                    "google_beyond_count_at_threshold_minus_4m": int(
                        origins["google_nearest_m"].gt(threshold_m - 4.0).sum()
                    ),
                    "google_beyond_count_at_threshold_plus_4m": int(
                        origins["google_nearest_m"].gt(threshold_m + 4.0).sum()
                    ),
                    "board_origins_within_4m_of_threshold": int(
                        origins["board_nearest_m"].sub(threshold_m).abs().le(4.0).sum()
                    ),
                    "board_beyond_count_at_threshold_minus_4m": int(
                        origins["board_nearest_m"].gt(threshold_m - 4.0).sum()
                    ),
                    "board_beyond_count_at_threshold_plus_4m": int(
                        origins["board_nearest_m"].gt(threshold_m + 4.0).sum()
                    ),
                },
            }
        )

    distance_difference = origins["google_minus_board_m"]
    distance_summary = {
        "n": len(origins),
        "google_nearest_median_m": round(float(origins["google_nearest_m"].median()), 2),
        "board_nearest_median_m": round(float(origins["board_nearest_m"].median()), 2),
        "google_minus_board_mean_m": round(float(distance_difference.mean()), 2),
        "google_minus_board_median_m": round(float(distance_difference.median()), 2),
        "google_minus_board_q25_m": round(float(distance_difference.quantile(0.25)), 2),
        "google_minus_board_q75_m": round(float(distance_difference.quantile(0.75)), 2),
        "google_farther_count": int(distance_difference.gt(0).sum()),
        "equal_distance_count": int(np.isclose(distance_difference, 0.0).sum()),
        "google_closer_count": int(distance_difference.lt(0).sum()),
    }

    strata_rows: list[dict[str, Any]] = []
    for stratifier in ("income_quartile", "nonwhite_quartile"):
        for quartile in (1, 2, 3, 4):
            subset = origins.loc[origins[stratifier].eq(quartile)]
            row: dict[str, Any] = {
                "stratifier": stratifier,
                "quartile": quartile,
                "tracts": len(subset),
                "google_minus_board_median_m": round(
                    float(subset["google_minus_board_m"].median()), 2
                ),
                "google_minus_board_mean_m": round(
                    float(subset["google_minus_board_m"].mean()), 2
                ),
            }
            for threshold_mi in ACCESS_THRESHOLDS_MI:
                suffix = str(threshold_mi).replace(".", "_") + "mi"
                row[f"google_beyond_pct_{suffix}"] = _rate_percent(
                    subset[f"google_beyond_{suffix}"]
                )
                row[f"board_beyond_pct_{suffix}"] = _rate_percent(
                    subset[f"board_beyond_{suffix}"]
                )
                row[f"google_minus_board_pp_{suffix}"] = round(
                    row[f"google_beyond_pct_{suffix}"]
                    - row[f"board_beyond_pct_{suffix}"],
                    2,
                )
            strata_rows.append(row)

    target_zctas_without_origin = sorted(set(TARGET_ZIPS) - set(origins["target_zcta"]))
    spatial_audit = {
        "distance_crs": DIST_CRS,
        "distance_crs_axis_units": axis_units,
        "origin_definition": (
            "2023 TIGER/Line internal point for each Hennepin or Ramsey County tract "
            "whose internal point intersects a target ZCTA and whose joined 2023 ACS "
            "record has positive median household income and a positive household count"
        ),
        "spatial_origin_frame": {
            "county_fips": list(SPATIAL_COUNTY_FIPS),
            "county_names": list(SPATIAL_COUNTY_NAMES),
            "acs_eligibility": (
                "med_hh_income greater than zero, nonmissing total_hh, and total_hh "
                "greater than zero; negative ACS sentinel values are ineligible"
            ),
            "target_membership": (
                "TIGER internal point intersects one of the 60 requested target ZCTAs"
            ),
        },
        "supply_coordinate_crs_assumption": {
            "assumed_crs": SUPPLY_INPUT_CRS_ASSUMPTION,
            "scope": "Google Latitude/Longitude and Board _lat/_lon input columns",
            "basis": (
                "column semantics and coordinate ranges; neither frozen tabular input "
                "contains machine-readable CRS metadata"
            ),
        },
        "distance_interpretation": (
            "Projected Euclidean nearest-facility distance; a potential-proximity proxy, "
            "not a road, transit, walking, cycling, or travel-time measure"
        ),
        "conditional_range_assumptions": {
            "status": "conditional sensitivity range, not an unconditional logical bound",
            "google": (
                "The frozen retrieved candidates exhaust observed supply inside the declared "
                "collection frame; unobserved supply may occur outside that frame"
            ),
            "board": (
                "The geocoded operational reference exhausts located supply inside the "
                "declared collection frame, and each missing-coordinate license is constrained "
                "to the Census ZCTA sharing its mailing ZIP code; mailing ZIP and physical "
                "location are not guaranteed to coincide"
            ),
        },
        "coordinate_transformations": {
            "tiger_nad83_to_distance_crs": _transformer_audit(tracts.crs, DIST_CRS),
            "supply_wgs84_to_distance_crs": _transformer_audit(
                SUPPLY_INPUT_CRS_ASSUMPTION, DIST_CRS
            ),
        },
        "collection_frame_definition": (
            "Observed supply frame: fixed target ZIPs union ZIPs observed among Google "
            "candidates; fixed-origin and frame-interior results are both reported"
        ),
        "tract_source_rows_hennepin_ramsey": len(tracts),
        "tract_rows_invalid_or_missing_median_household_income": (
            invalid_or_missing_income_rows
        ),
        "tract_rows_missing_household_count": missing_household_rows,
        "tract_rows_nonpositive_or_missing_household_count": nonpositive_household_rows,
        "tract_geometry_null": int(tracts.geometry.isna().sum()),
        "tract_geometry_empty": int(tracts.geometry.is_empty.sum()),
        "tract_geometry_invalid": int((~tracts.geometry.is_valid).sum()),
        "eligible_tracts_after_acs": len(eligible),
        "fixed_target_zcta_origins": len(origins),
        "target_zctas_with_at_least_one_origin": int(origins["target_zcta"].nunique()),
        "internal_point_origins_outside_own_tract": int(
            (~origins["internal_point_within_own_tract"]).sum()
        ),
        "geometric_centroids_outside_own_tract": int(
            (~origins["centroid_within_own_tract"]).sum()
        ),
        "target_zctas_requested": len(TARGET_ZIPS),
        "target_zctas_loaded": len(target_zctas),
        "target_zctas_without_eligible_internal_point_origin": target_zctas_without_origin,
        "collection_frame_zip_codes": len(frame_zips),
        "collection_frame_zctas_loaded": len(zctas),
        "collection_frame_zctas_missing": missing_frame_zctas,
        "zcta_geometry_audit": zcta_geometry_audit,
        "board_keyword_screen_supply_rows": len(criterion),
        "board_keyword_screen_supply_geocoded": int(criterion["has_geocode"].sum()),
        "board_keyword_screen_supply_exact_geocodes": len(board_exact_geocode),
        "board_keyword_screen_supply_non_exact_geocodes": int(
            (criterion["has_geocode"] & criterion["mtype"].eq("Non_Exact")).sum()
        ),
        "board_primary_coordinate_policy": (
            "All cached coordinates labelled Exact or Non_Exact are used in the primary "
            "surface; an Exact-only surface is reported as a coordinate-quality sensitivity"
        ),
        "board_keyword_screen_supply_missing_geocodes": len(missing_board),
        "board_legacy_405_supply_rows": len(legacy_criterion),
        "board_legacy_405_supply_geocoded": int(
            legacy_criterion["has_geocode"].sum()
        ),
        "board_all_licensed_supply_rows": len(board_all),
        "board_all_licensed_supply_geocoded": int(board_all["has_geocode"].sum()),
        "missing_board_mailing_zip_codes": sorted(missing_board_mailing_zips),
        "missing_board_mailing_zip_zctas_without_geometry": (
            missing_board_mailing_zip_zctas_without_geometry
        ),
        "google_model_target_exact_location_supply_rows": len(ai),
        "google_raw_candidate_exact_location_supply_rows": len(ai_raw),
        "nearest_ties": {
            "google_origins_with_ties": int(origins["google_nearest_tie_count"].gt(1).sum()),
            "google_max_tie_count": int(origins["google_nearest_tie_count"].max()),
            "board_origins_with_ties": int(origins["board_nearest_tie_count"].gt(1).sum()),
            "board_max_tie_count": int(origins["board_nearest_tie_count"].max()),
            "board_exact_geocode_only_origins_with_ties": int(
                origins["board_exact_geocode_only_nearest_tie_count"].gt(1).sum()
            ),
            "board_exact_geocode_only_max_tie_count": int(
                origins["board_exact_geocode_only_nearest_tie_count"].max()
            ),
        },
        "distance_difference": distance_summary,
        "thresholds": threshold_summaries,
        "neighborhood_strata": strata_rows,
    }

    output_columns = [
        "GEOID", "target_zcta", "COUNTYFP", "NAME", "med_hh_income", "total_pop",
        "pct_nonwhite", "total_hh", "carless_rate", "income_quartile",
        "nonwhite_quartile", "internal_point_within_own_tract",
        "centroid_within_own_tract", "frame_edge_m",
        "missing_board_mailing_zip_zcta_min_m",
        "google_nearest_m", "google_nearest_tie_count",
        "google_raw_candidate_nearest_m", "google_raw_candidate_nearest_tie_count",
        "board_nearest_m", "board_nearest_tie_count",
        "board_exact_geocode_only_nearest_m",
        "board_exact_geocode_only_nearest_tie_count",
        "board_legacy_405_nearest_m", "board_all_licensed_nearest_m",
        "google_minus_board_m", "abs_google_minus_board_m",
        "centroid_google_nearest_m", "centroid_board_nearest_m",
    ]
    for threshold_mi in ACCESS_THRESHOLDS_MI:
        suffix = str(threshold_mi).replace(".", "_") + "mi"
        output_columns.extend(
            [
                f"google_beyond_{suffix}", f"google_raw_candidate_beyond_{suffix}",
                f"board_beyond_{suffix}",
                f"board_exact_geocode_only_beyond_{suffix}",
                f"board_legacy_405_beyond_{suffix}",
                f"board_all_licensed_beyond_{suffix}",
                f"discordance_type_{suffix}",
                f"google_conditionally_determinate_{suffix}",
                f"board_conditionally_determinate_{suffix}",
            ]
        )
    return pd.DataFrame(origins.drop(columns="geometry"))[output_columns], spatial_audit


def create_figure(tracts: pd.DataFrame, spatial_summary: dict[str, Any]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "board": "#0072B2",
        "board_exact": "#56B4E9",
        "google": "#D55E00",
        "both_beyond": "#0072B2",
        "google_only": "#D55E00",
        "board_only": "#009E73",
        "both_within": "#666666",
    }
    fig = plt.figure(figsize=(6.5, 7.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1.0))
    axes = (fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))

    # Panel A: paired Euclidean nearest distances; shape and color both encode status.
    suffix = "0_5mi"
    board_beyond = tracts[f"board_beyond_{suffix}"].astype(bool)
    google_beyond = tracts[f"google_beyond_{suffix}"].astype(bool)
    categories = {
        "Both within 0.5 mi": (~board_beyond & ~google_beyond, "o", colors["both_within"]),
        "Both beyond 0.5 mi": (board_beyond & google_beyond, "s", colors["both_beyond"]),
        "Google only beyond": (~board_beyond & google_beyond, "^", colors["google_only"]),
        "Board only beyond": (board_beyond & ~google_beyond, "D", colors["board_only"]),
    }
    x_miles = tracts["board_nearest_m"] / M_PER_MILE
    y_miles = tracts["google_nearest_m"] / M_PER_MILE
    upper = float(max(x_miles.max(), y_miles.max()) * 1.04)
    axes[0].plot([0, upper], [0, upper], color="black", linewidth=0.8, linestyle="--")
    for label, (mask, marker, color) in categories.items():
        axes[0].scatter(
            x_miles.loc[mask], y_miles.loc[mask], s=17, marker=marker,
            facecolors="none" if label == "Both within 0.5 mi" else color,
            edgecolors=color, linewidths=0.7, alpha=0.78, label=label,
        )
    axes[0].set(
        xlim=(0, upper), ylim=(0, upper),
        xlabel="Board Euclidean nearest distance (miles)",
        ylabel="Google Euclidean nearest distance (miles)",
        title=f"A. Hennepin/Ramsey residential origins (n = {len(tracts)})",
    )
    axes[0].legend(frameon=False, loc="upper left", ncol=2)

    # Panel B: observed-coordinate classifications, a coordinate-quality
    # sensitivity, and conditional ranges.  The brackets are not confidence intervals.
    thresholds = spatial_summary["thresholds"]
    x_values = np.arange(len(thresholds))
    board_points = np.array([row["board_beyond_threshold_pct"] for row in thresholds])
    board_exact_points = np.array(
        [row["board_exact_geocode_only_beyond_threshold_pct"] for row in thresholds]
    )
    board_lower = np.array(
        [row["board_beyond_rate_conditional_lower_pct"] for row in thresholds]
    )
    board_upper = np.array(
        [row["board_beyond_rate_conditional_upper_pct"] for row in thresholds]
    )
    google_points = np.array([row["google_beyond_threshold_pct"] for row in thresholds])
    google_lower = np.array(
        [row["google_beyond_rate_conditional_lower_pct"] for row in thresholds]
    )
    google_upper = np.array(
        [row["google_beyond_rate_conditional_upper_pct"] for row in thresholds]
    )
    for offset, points, lower, upper_range, color, marker, label, linestyle in (
        (-0.08, board_points, board_lower, board_upper, colors["board"], "o", "Board: all geocodes", "-"),
        (0.08, google_points, google_lower, google_upper, colors["google"], "s", "Google model-target", "--"),
    ):
        positions = x_values + offset
        axes[1].vlines(positions, lower, upper_range, color=color, linewidth=2.0, alpha=0.45)
        axes[1].plot(positions, points, color=color, marker=marker, linestyle=linestyle, label=label)
    axes[1].plot(
        x_values,
        board_exact_points,
        color=colors["board_exact"],
        marker="^",
        linestyle=":",
        label="Board: Exact only",
    )
    axes[1].set_xticks(x_values, [f"{row['threshold_miles']:.1f}" for row in thresholds])
    axes[1].set(
        xlabel="Euclidean threshold (miles)",
        ylabel="Tract origins beyond threshold (%)",
        ylim=(0, 100),
        title="B. Threshold comparison",
    )
    axes[1].legend(frameon=False, fontsize=6.8)
    axes[1].text(
        0.025,
        0.025,
        "Ranges (not CIs):\nBoard ZIP-constrained;\nGoogle frame-conditional",
        transform=axes[1].transAxes, va="bottom", fontsize=6.2, color="#444444",
    )

    # Panel C: signed paired difference by income quartile with raw points.
    groups = [
        tracts.loc[tracts["income_quartile"].eq(quartile), "google_minus_board_m"]
        / M_PER_MILE
        for quartile in (1, 2, 3, 4)
    ]
    box = axes[2].boxplot(
        groups, positions=[1, 2, 3, 4], widths=0.55, patch_artist=True,
        showfliers=False, medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch in box["boxes"]:
        patch.set(facecolor="#D9EAF3", edgecolor=colors["board"], linewidth=0.9)
    rng = np.random.default_rng(RANDOM_SEED)
    for index, values in enumerate(groups, start=1):
        jitter = rng.uniform(-0.17, 0.17, size=len(values))
        axes[2].scatter(
            index + jitter, values, s=7, color="#555555", alpha=0.34, linewidths=0,
        )
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set(
        xticks=[1, 2, 3, 4],
        xticklabels=["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"],
        xlabel="Tract household-income quartile",
        ylabel="Google minus Board Euclidean distance (miles)",
        title="C. Difference by income quartile",
    )
    for index, values in enumerate(groups, start=1):
        axes[2].text(index, axes[2].get_ylim()[1], f"n={len(values)}", ha="center", va="top", fontsize=7.2)

    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55, alpha=0.75)
    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_OUT,
        dpi=300,
        facecolor="white",
        transparent=False,
        metadata={"Software": "NETS-AI canonical analysis"},
    )
    fig.savefig(
        FIGURE_PDF_OUT,
        facecolor="white",
        transparent=False,
        metadata={
            "Creator": "NETS-AI canonical analysis",
            "CreationDate": FIGURE_METADATA_TIME,
            "ModDate": FIGURE_METADATA_TIME,
        },
    )
    plt.close(fig)


def create_signature_figure(signature_summary: dict[str, Any]) -> None:
    """Visualize automatic name-component linkage without ownership language."""
    rows = pd.DataFrame(signature_summary["rows"])
    rows = rows.loc[rows["source"].eq("Board keyword-screen reference")].copy()
    strategy_labels = {
        "name_only_hungarian": "Name only",
        "address_only_hungarian": "Address only",
        "name_address_hungarian": "Name + address",
    }
    colors = {"repeated": "#0072B2", "below_threshold": "#E69F00"}
    hatches = {"repeated": "///", "below_threshold": "..."}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.9), layout="constrained")

    k2 = rows.loc[
        rows["recurrence_threshold"].eq(2)
        & rows["linkage_strategy"].isin(strategy_labels)
        & rows["group_code"].isin(["repeated", "below_threshold"])
    ].copy()
    x = np.arange(len(strategy_labels))
    width = 0.34
    for offset, group_code, label in (
        (-width / 2, "repeated", "Repeated name"),
        (width / 2, "below_threshold", "Single occurrence"),
    ):
        ordered = (
            k2.loc[k2["group_code"].eq(group_code)]
            .set_index("linkage_strategy")
            .reindex(strategy_labels)
        )
        values = ordered["linked_pct"].to_numpy(dtype=float)
        axes[0].bar(
            x + offset,
            values,
            width,
            color=colors[group_code],
            edgecolor="#333333",
            hatch=hatches[group_code],
            label=label,
            linewidth=0.45,
        )
        for position, value, records, components in zip(
            x + offset,
            values,
            ordered["records"].astype(int),
            ordered["distinct_name_components"].astype(int),
        ):
            axes[0].text(
                position,
                value + 2.0,
                f"{value:.1f}\nn={records}/{components}",
                ha="center",
                fontsize=6.6,
            )
    axes[0].set_xticks(x, [strategy_labels[key] for key in strategy_labels])
    axes[0].set(
        ylim=(0, 118),
        ylabel="Board records linked (%)",
        title="A. Descriptive automatic name groups (k = 2)",
    )
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        title="Bar labels: linked %; n = records / name components",
        title_fontsize=6.6,
    )

    gaps = pd.DataFrame(signature_summary["repeated_minus_below_threshold_gaps"])
    gaps = gaps.loc[
        gaps["source"].eq("Board keyword-screen reference")
        & gaps["linkage_strategy"].isin(strategy_labels)
    ]
    for strategy, label, marker, color, linestyle in (
        ("name_only_hungarian", "Name only", "o", "#CC79A7", "-"),
        ("address_only_hungarian", "Address only", "^", "#0072B2", ":"),
        ("name_address_hungarian", "Name + address", "s", "#009E73", "--"),
    ):
        series = gaps.loc[gaps["linkage_strategy"].eq(strategy)].sort_values(
            "recurrence_threshold"
        )
        axes[1].plot(
            series["recurrence_threshold"],
            series["repeated_minus_below_threshold_percentage_points"],
            marker=marker,
            linewidth=1.4,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[1].set_xticks(list(SIGNATURE_THRESHOLDS))
    axes[1].set(
        xlabel="Recurrence threshold, k\n(minimum distinct locations)",
        ylabel="Linkage-rate gap (percentage points)",
        title="B. Recurrence-threshold sensitivity",
    )
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.55, alpha=0.75)

    SIGNATURE_FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        SIGNATURE_FIGURE_OUT,
        dpi=300,
        facecolor="white",
        transparent=False,
        metadata={"Software": "NETS-AI canonical analysis"},
    )
    fig.savefig(
        SIGNATURE_FIGURE_PDF_OUT,
        facecolor="white",
        transparent=False,
        metadata={
            "Creator": "NETS-AI canonical analysis",
            "CreationDate": FIGURE_METADATA_TIME,
            "ModDate": FIGURE_METADATA_TIME,
        },
    )
    plt.close(fig)


def main(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    input_manifest: Path = DEFAULT_INPUT_MANIFEST,
) -> None:
    configure_output_root(output_root)
    PRIVATE_OUTPUT.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    linkage.set_parser("regex")
    manifest_audit = validate_input_manifest(input_manifest)

    ai_snapshot, board_all, inventory_audit = prepare_inventories()
    target_mask = ai_snapshot["Is_Target_Match"].fillna(False).astype(bool)
    ai_target_listings_extended = _recount_signatures(ai_snapshot.loc[target_mask].copy())
    ai_raw_listings_primary = _recount_signatures(
        ai_snapshot.loc[ai_snapshot["zip5"].isin(TARGET_ZIPS)].copy()
    )
    ai_target_listings_primary = _recount_signatures(
        ai_target_listings_extended.loc[
            ai_target_listings_extended["zip5"].isin(TARGET_ZIPS)
        ].copy()
    )
    ai_raw_locations_extended = collapse_google_exact_locations(ai_snapshot)
    ai_target_locations_extended = collapse_google_exact_locations(ai_target_listings_extended)
    ai_raw_locations_primary = collapse_google_exact_locations(ai_raw_listings_primary)
    ai_primary = collapse_google_exact_locations(ai_target_listings_primary)

    criterion_extended = _recount_signatures(
        board_all.loc[board_all["access_eligible"]].copy()
    )
    legacy_extended = _recount_signatures(legacy_outcome_conditioned_criterion(board_all))

    criterion_primary = _recount_signatures(
        criterion_extended.loc[criterion_extended["zip5"].isin(TARGET_ZIPS)].copy()
    )
    legacy_primary = _recount_signatures(
        legacy_extended.loc[legacy_extended["zip5"].isin(TARGET_ZIPS)].copy()
    )
    all_licensed_primary = _recount_signatures(
        board_all.loc[board_all["zip5"].isin(TARGET_ZIPS)].copy()
    )

    inventory_audit.update(
        {
            "fixed_target_zip_count": len(TARGET_ZIPS),
            "google_snapshot_rows": len(ai_snapshot),
            "google_model_target_listing_rows": len(ai_target_listings_extended),
            "google_raw_exact_address_locations": len(ai_raw_locations_extended),
            "google_model_target_exact_address_locations": len(ai_target_locations_extended),
            "google_fixed_target_raw_listing_rows": len(ai_raw_listings_primary),
            "google_fixed_target_model_target_listing_rows": len(ai_target_listings_primary),
            "google_fixed_target_raw_exact_address_locations": len(ai_raw_locations_primary),
            "google_fixed_target_model_target_exact_address_locations": len(ai_primary),
            "google_duplicate_listing_rows_removed_primary": len(ai_target_listings_primary)
            - len(ai_primary),
            "google_max_within_location_coordinate_spread_m": round(
                float(ai_target_locations_extended["max_coordinate_spread_m"].max()), 3
            ),
            "board_uniform_criterion_fixed_target_rows": len(criterion_primary),
            "board_uniform_criterion_fixed_target_geocoded_rows": int(
                criterion_primary["has_geocode"].sum()
            ),
            "board_uniform_criterion_extended_rows": len(criterion_extended),
            "board_uniform_criterion_extended_geocoded_rows": int(
                criterion_extended["has_geocode"].sum()
            ),
            "board_legacy_405_fixed_target_rows": len(legacy_primary),
            "board_all_active_477_fixed_target_rows": len(all_licensed_primary),
            "board_uniform_geocode_type_counts": {
                str(key): int(value)
                for key, value in criterion_extended["mtype"]
                .fillna("missing")
                .value_counts()
                .items()
            },
        }
    )

    linkage_primary, pairs_by_strategy = run_linkage(ai_primary, criterion_primary)
    linkage_sensitivities: dict[str, Any] = {}
    for label, ai_frame, board_frame in (
        ("historical_prefiltered_405_fixed_target", ai_primary, legacy_primary),
        ("all_477_active_board_fixed_target", ai_primary, all_licensed_primary),
        (
            "uniform_criterion_extended_observed_frame",
            ai_target_locations_extended,
            criterion_extended,
        ),
        ("model_target_listing_unit_fixed_target", ai_target_listings_primary, criterion_primary),
        ("raw_candidate_exact_location_fixed_target", ai_raw_locations_primary, criterion_primary),
        ("raw_candidate_listing_unit_fixed_target", ai_raw_listings_primary, criterion_primary),
    ):
        linkage_sensitivities[label] = run_linkage(ai_frame, board_frame)[0]

    pair_table = build_pair_table(
        ai_primary,
        criterion_primary,
        pairs_by_strategy[PRIMARY_LINKAGE],
    )
    source_status = build_source_status(ai_primary, criterion_primary, pair_table)
    signature_detail, signature_summary = signature_analysis(
        ai_primary,
        criterion_primary,
        pairs_by_strategy,
    )
    tract_results, spatial_summary = spatial_analysis(
        ai_target_locations_extended,
        criterion_extended,
        legacy_extended,
        board_all,
        ai_raw=ai_raw_locations_extended,
    )

    criterion_primary.to_csv(BOARD_CRITERION_OUT, index=False)
    criterion_extended.to_csv(BOARD_EXTENDED_OUT, index=False)
    pair_table.to_csv(MATCH_OUT, index=False)
    source_status.to_csv(SOURCE_STATUS_OUT, index=False)
    signature_detail.to_csv(SIGNATURE_OUT, index=False)
    tract_results.to_csv(TRACT_OUT, index=False)
    pd.json_normalize(linkage_primary["strategies"], sep=".").to_csv(
        LINKAGE_TABLE_OUT,
        index=False,
    )
    pd.json_normalize(spatial_summary["thresholds"], sep=".").to_csv(
        PROXIMITY_TABLE_OUT,
        index=False,
    )
    _write_json(SIGNATURE_SUMMARY_OUT, signature_summary)
    create_figure(tract_results, spatial_summary)
    create_signature_figure(signature_summary)

    output_paths = [
        BOARD_CRITERION_OUT,
        BOARD_EXTENDED_OUT,
        MATCH_OUT,
        SOURCE_STATUS_OUT,
        SIGNATURE_OUT,
        TRACT_OUT,
        LINKAGE_TABLE_OUT,
        PROXIMITY_TABLE_OUT,
        SIGNATURE_SUMMARY_OUT,
        FIGURE_OUT,
        FIGURE_PDF_OUT,
        SIGNATURE_FIGURE_OUT,
        SIGNATURE_FIGURE_PDF_OUT,
    ]
    summary = {
        "analysis_status": (
            "descriptive source audit, deterministic record linkage, and projected "
            "straight-line proximity comparison"
        ),
        "criterion_definition": (
            "Active Minnesota Board license records in the fixed 60-target-ZIP evaluation "
            "frame passing the declared pre-linkage automatic keyword screen; this is an "
            "operational reference, not a manually adjudicated census of public access"
        ),
        "google_primary_unit": (
            "Model-target listings collapsed by exact normalized address plus ZIP because "
            "the frozen snapshot contains no stable Google Place ID"
        ),
        "spatial_supply_definition": (
            "Exact-address Google model-target locations and uniformly keyword-screened "
            "Board records in the extended observed frame. The primary Board surface uses "
            "all available cached coordinates and reports an Exact-only coordinate-quality "
            "sensitivity; results represent projected straight-line potential proximity, "
            "not road-network accessibility"
        ),
        "spatial_origin_definition": (
            "TIGER internal points for 2023 ACS-eligible Hennepin and Ramsey County "
            "tracts whose internal point intersects one of the 60 requested target ZCTAs; "
            f"{spatial_summary['fixed_target_zcta_origins']} origins cover "
            f"{spatial_summary['target_zctas_with_at_least_one_origin']} target ZCTAs "
            "in the frozen inputs"
        ),
        "criterion_sensitivity_definitions": {
            "historical_prefiltered_405": (
                "Historical outcome-conditioned denominator from the 459-record geocode "
                "cache, retained for sensitivity only"
            ),
            "all_477_active": (
                "All active Minnesota Board records in the declared observed ZIP frame; "
                "459 appear in the frozen cache and 416 have usable cached coordinates"
            ),
        },
        "manual_labels_used": False,
        "ownership_inference_made": False,
        "naics_accuracy_claim_made": False,
        "inventory": inventory_audit,
        "input_manifest": manifest_audit,
        "eligibility_rules": [
            {"reason": reason, "pattern": pattern}
            for reason, pattern in BOARD_ACCESS_EXCLUSION_RULES
        ],
        "linkage": {
            "primary_fixed_target_frame": linkage_primary,
            "sensitivities": linkage_sensitivities,
            "interpretation": (
                "Linked-pair proportions describe deterministic pipeline agreement; "
                "no manual pair labels were used, so they are not precision, recall, or accuracy"
            ),
            "unit_warning": (
                "Listing-unit and exact-address-location-unit analyses are separate "
                "sensitivities and are never combined into one F-score"
            ),
        },
        "name_signatures": signature_summary,
        "spatial": spatial_summary,
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "pandas": pd.__version__,
            "geopandas": gpd.__version__,
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
            "shapely": shapely.__version__,
            "pyproj": pyproj.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "rapidfuzz": importlib.metadata.version("rapidfuzz"),
            "pyogrio": pyogrio.__version__,
            "gdal": pyogrio.__gdal_version_string__,
            "geos": shapely.geos_version_string,
            "proj": pyproj.proj_version_str,
            "geopandas_io_engine": gpd.options.io_engine or "auto",
            "address_parser": "regex",
        },
        "code_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (
                HERE / "study_design.py",
                HERE / "geo_match.py",
                HERE / "board_google_comparison.py",
                HERE / "verify_board_google_outputs.py",
            )
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in (
                AI_PATH,
                BOARD_PATH,
                BOARD_RAW_PATH,
                TRACT_PATH,
                ZCTA_PATH,
                ACS_PATH,
                VEHICLES_PATH,
                LEGACY_INSTITUTIONAL_PATH,
                LEGACY_FALSE_NEGATIVE_PATH,
            )
        },
        "output_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in output_paths
        },
    }
    _write_json(SUMMARY_OUT, summary)

    primary = next(
        row for row in linkage_primary["strategies"] if row["strategy"] == PRIMARY_LINKAGE
    )
    half_mile = next(
        row for row in spatial_summary["thresholds"] if row["threshold_miles"] == 0.5
    )
    print(
        json.dumps(
            {
                "board_keyword_screen_reference_fixed_target": inventory_audit[
                    "board_uniform_criterion_fixed_target_rows"
                ],
                "board_keyword_screen_geocoded_fixed_target": inventory_audit[
                    "board_uniform_criterion_fixed_target_geocoded_rows"
                ],
                "primary_linkage": primary,
                "fixed_tract_origins": spatial_summary["fixed_target_zcta_origins"],
                "half_mile_comparison": half_mile,
                "summary": str(SUMMARY_OUT.relative_to(ROOT)),
                "figure": str(FIGURE_OUT.relative_to(ROOT)),
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Dedicated repository-local directory for all generated artifacts",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help="Repository-local manifest authenticating all canonical input bytes",
    )
    arguments = parser.parse_args()
    main(arguments.output_root, arguments.input_manifest)
