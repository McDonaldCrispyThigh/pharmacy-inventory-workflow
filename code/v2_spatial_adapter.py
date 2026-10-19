"""Fixed-rule projected straight-line potential-proximity analysis for v2.1."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from v2_board_adapter import BoardFramesV2


SPATIAL_ADAPTER_VERSION = "nets-ai-v2.1-spatial-adapter-v2"
M_PER_MILE = 1609.344


@dataclass(frozen=True, slots=True)
class SpatialContractV2:
    distance_crs: str = "EPSG:26915"
    supply_crs: str = "EPSG:4326"
    origin_source_crs: str = "EPSG:4269"
    thresholds_miles: tuple[float, ...] = (0.5, 1.0, 2.0)
    origin_id_column: str = "GEOID"
    county_fips: tuple[str, ...] = ("053", "123")
    meters_per_mile: float = M_PER_MILE
    tie_tolerance_m: float = 1e-7
    origin_rule: str = "TIGER_2023_internal_point"
    threshold_rule: str = "strict_distance_greater_than_threshold"

    def __post_init__(self) -> None:
        distance = CRS.from_user_input(self.distance_crs)
        supply = CRS.from_user_input(self.supply_crs)
        origin_source = CRS.from_user_input(self.origin_source_crs)
        if not distance.is_projected or not supply.is_geographic:
            raise ValueError("spatial contract requires projected distance and geographic supply CRS")
        if not origin_source.is_geographic:
            raise ValueError("TIGER internal-point CRS must be geographic")
        units = {axis.unit_name.casefold() for axis in distance.axis_info}
        if not units or not units <= {"metre", "meter"}:
            raise ValueError("distance CRS axes must use meters")
        values = tuple(float(value) for value in self.thresholds_miles)
        if not values or any(value <= 0 for value in values) or tuple(sorted(set(values))) != values:
            raise ValueError("spatial thresholds must be unique, positive, and increasing")
        if not isinstance(self.origin_id_column, str) or not self.origin_id_column:
            raise ValueError("origin ID column is required")
        counties = tuple(str(value).zfill(3) for value in self.county_fips)
        if (
            not counties
            or len(set(counties)) != len(counties)
            or any(len(value) != 3 or not value.isdigit() for value in counties)
        ):
            raise ValueError("county FIPS contract is invalid")
        if (
            type(self.meters_per_mile) not in {int, float}
            or not np.isfinite(float(self.meters_per_mile))
            or float(self.meters_per_mile) <= 0
        ):
            raise ValueError("meters-per-mile contract is invalid")
        if (
            type(self.tie_tolerance_m) not in {int, float}
            or not np.isfinite(float(self.tie_tolerance_m))
            or float(self.tie_tolerance_m) <= 0
        ):
            raise ValueError("tie tolerance must be finite and positive")
        if self.origin_rule != "TIGER_2023_internal_point":
            raise ValueError("unsupported origin rule")
        if self.threshold_rule != "strict_distance_greater_than_threshold":
            raise ValueError("unsupported threshold rule")
        object.__setattr__(self, "distance_crs", distance.to_string())
        object.__setattr__(self, "supply_crs", supply.to_string())
        object.__setattr__(self, "origin_source_crs", origin_source.to_string())
        object.__setattr__(self, "thresholds_miles", values)
        object.__setattr__(self, "county_fips", counties)
        object.__setattr__(self, "meters_per_mile", float(self.meters_per_mile))
        object.__setattr__(self, "tie_tolerance_m", float(self.tie_tolerance_m))

    @property
    def contract_sha256(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True, slots=True, repr=False)
class SpatialResultV2:
    tracts: pd.DataFrame
    coordinate_qa: pd.DataFrame
    coordinate_ledger: pd.DataFrame
    summary: dict[str, Any]


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _origin_sha256(origins: gpd.GeoDataFrame, id_column: str) -> str:
    payload = {
        "crs": origins.crs.to_string(),
        "ordered_origins": [
            {"id": str(identifier), "geometry_wkb_hex": geometry.wkb_hex}
            for identifier, geometry in zip(origins[id_column], origins.geometry)
        ],
    }
    return _sha256_json(payload)


def _frame_input_sha256(
    frame: pd.DataFrame, id_column: str, columns: tuple[str, ...]
) -> str:
    present = [column for column in columns if column in frame]
    if id_column not in present:
        present.insert(0, id_column)
    work = frame[present].copy().sort_values(id_column).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for row in work.to_dict(orient="records"):
        cleaned: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
                cleaned[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                scalar = value.item()
                if isinstance(scalar, float) and not np.isfinite(scalar):
                    cleaned[key] = "+Infinity" if scalar > 0 else "-Infinity"
                else:
                    cleaned[key] = scalar
            elif isinstance(value, float) and not np.isfinite(value):
                cleaned[key] = "+Infinity" if value > 0 else "-Infinity"
            else:
                cleaned[key] = value
        records.append(cleaned)
    return _sha256_json({"columns": present, "records": records})


def _require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame:
        raise ValueError(f"{label} lacks {column}")
    values = frame[column]
    if values.isna().any() or values.astype(str).str.strip().eq("").any() or values.duplicated().any():
        raise ValueError(f"{label} {column} values must be nonempty and unique")


def _quartile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    valid = numeric.notna()
    if valid.sum() < 4:
        return result
    ranked = numeric.loc[valid].rank(method="first")
    result.loc[valid] = pd.qcut(ranked, 4, labels=(1, 2, 3, 4)).astype(int)
    return result


def prepare_fixed_tract_origins(
    tracts: gpd.GeoDataFrame,
    acs: pd.DataFrame,
    vehicles: pd.DataFrame,
    zctas: gpd.GeoDataFrame,
    target_zips: Iterable[str],
    contract: SpatialContractV2 | None = None,
) -> gpd.GeoDataFrame:
    """Build the declared 2023 TIGER/ACS internal-point origin frame locally."""
    contract = SpatialContractV2() if contract is None else contract
    if not isinstance(contract, SpatialContractV2):
        raise TypeError("invalid spatial contract")
    if not isinstance(tracts, gpd.GeoDataFrame) or tracts.crs is None:
        raise ValueError("tract source requires authoritative CRS metadata")
    if not isinstance(zctas, gpd.GeoDataFrame) or zctas.crs is None:
        raise ValueError("ZCTA source requires authoritative CRS metadata")
    required_tract = {"GEOID", "COUNTYFP", "NAME", "INTPTLON", "INTPTLAT", tracts.geometry.name}
    required_acs = {"GEOID", "med_hh_income", "total_pop", "pct_nonwhite"}
    required_vehicles = {"GEOID", "total_hh", "carless_rate"}
    if missing := required_tract - set(tracts.columns):
        raise ValueError(f"tract source lacks columns: {sorted(missing)}")
    if missing := required_acs - set(acs.columns):
        raise ValueError(f"ACS source lacks columns: {sorted(missing)}")
    if missing := required_vehicles - set(vehicles.columns):
        raise ValueError(f"vehicle source lacks columns: {sorted(missing)}")
    if "ZCTA5CE20" not in zctas:
        raise ValueError("ZCTA source lacks ZCTA5CE20")
    target = {str(value).zfill(5) for value in target_zips}
    if not target or any(len(value) != 5 or not value.isdigit() for value in target):
        raise ValueError("invalid target ZIP frame")

    source = tracts.loc[
        tracts["COUNTYFP"].astype(str).str.zfill(3).isin(contract.county_fips)
    ].copy()
    source["GEOID"] = source["GEOID"].astype(str)
    _require_unique(source, "GEOID", "tract source")
    if (
        source.geometry.isna().any()
        or source.geometry.is_empty.any()
        or (~source.geometry.is_valid).any()
        or not source.geometry.geom_type.isin(("Polygon", "MultiPolygon")).all()
    ):
        raise ValueError("tract source contains unusable polygon geometry")
    acs_work = acs.copy()
    vehicle_work = vehicles.copy()
    acs_work["GEOID"] = acs_work["GEOID"].astype(str)
    vehicle_work["GEOID"] = vehicle_work["GEOID"].astype(str)
    _require_unique(acs_work, "GEOID", "ACS source")
    _require_unique(vehicle_work, "GEOID", "vehicle source")
    joined = source.merge(
        acs_work[["GEOID", "med_hh_income", "total_pop", "pct_nonwhite"]],
        on="GEOID", how="left", validate="one_to_one",
    ).merge(
        vehicle_work[["GEOID", "total_hh", "carless_rate"]],
        on="GEOID", how="left", validate="one_to_one",
    )
    income = pd.to_numeric(joined["med_hh_income"], errors="coerce")
    households = pd.to_numeric(joined["total_hh"], errors="coerce")
    joined = joined.loc[income.gt(0) & households.gt(0)].copy()
    longitude = pd.to_numeric(joined["INTPTLON"], errors="coerce")
    latitude = pd.to_numeric(joined["INTPTLAT"], errors="coerce")
    if (
        longitude.isna().any()
        or latitude.isna().any()
        or not np.isfinite(longitude).all()
        or not np.isfinite(latitude).all()
        or not longitude.between(-180.0, 180.0).all()
        or not latitude.between(-90.0, 90.0).all()
    ):
        raise ValueError("eligible tracts have invalid TIGER internal point coordinates")
    geographic_points = gpd.GeoDataFrame(
        joined.drop(columns=tracts.geometry.name),
        geometry=gpd.points_from_xy(longitude, latitude),
        crs=contract.origin_source_crs,
    )
    coverage_points = geographic_points.to_crs(source.crs)
    source_polygons = gpd.GeoSeries(joined[tracts.geometry.name], crs=source.crs)
    covered = np.fromiter(
        (
            polygon.covers(point)
            for polygon, point in zip(source_polygons, coverage_points.geometry)
        ),
        dtype=bool,
        count=len(joined),
    )
    if not covered.all():
        raise ValueError("each TIGER internal point must be covered by its source tract")
    target_zctas = zctas.loc[
        zctas["ZCTA5CE20"].astype(str).str.zfill(5).isin(target),
        ["ZCTA5CE20", zctas.geometry.name],
    ].copy()
    target_zctas["target_zcta"] = target_zctas["ZCTA5CE20"].astype(str).str.zfill(5)
    if set(target_zctas["target_zcta"]) != target:
        raise ValueError("one or more target ZCTA geometries are missing")
    _require_unique(target_zctas, "target_zcta", "target ZCTA source")
    if (
        target_zctas.geometry.isna().any()
        or target_zctas.geometry.is_empty.any()
        or (~target_zctas.geometry.is_valid).any()
        or not target_zctas.geometry.geom_type.isin(("Polygon", "MultiPolygon")).all()
    ):
        raise ValueError("target ZCTA source contains unusable polygon geometry")
    points_for_join = geographic_points.to_crs(target_zctas.crs)
    matches = gpd.sjoin(
        points_for_join[["GEOID", points_for_join.geometry.name]],
        target_zctas[["target_zcta", target_zctas.geometry.name]],
        how="inner", predicate="intersects",
    )[["GEOID", "target_zcta"]]
    if matches["GEOID"].duplicated().any():
        raise ValueError("one tract internal point intersects multiple target ZCTAs")
    safe_columns = [
        "GEOID", "COUNTYFP", "med_hh_income", "total_pop", "pct_nonwhite",
        "total_hh", "carless_rate",
    ]
    result = geographic_points[safe_columns + [geographic_points.geometry.name]].merge(
        matches, on="GEOID", how="inner", validate="one_to_one"
    )
    result = gpd.GeoDataFrame(
        result, geometry=geographic_points.geometry.name, crs=geographic_points.crs
    ).to_crs(contract.distance_crs)
    _require_unique(result, "GEOID", "fixed origin frame")
    result["income_quartile"] = _quartile(result["med_hh_income"])
    result["nonwhite_quartile"] = _quartile(result["pct_nonwhite"])
    return result.sort_values("GEOID").reset_index(drop=True)


def _coordinate_state(
    frame: pd.DataFrame, latitude_column: str, longitude_column: str
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if latitude_column not in frame or longitude_column not in frame:
        raise ValueError("spatial surface lacks coordinate columns")
    raw_lat = frame[latitude_column]
    raw_lon = frame[longitude_column]
    latitude = pd.to_numeric(raw_lat, errors="coerce")
    longitude = pd.to_numeric(raw_lon, errors="coerce")

    def missing_value(value: object) -> bool:
        if value is None or pd.isna(value):
            return True
        return isinstance(value, str) and not value.strip()

    latitude_missing = raw_lat.map(missing_value)
    longitude_missing = raw_lon.map(missing_value)
    both_missing = latitude_missing & longitude_missing
    partial_missing = latitude_missing ^ longitude_missing
    nonnumeric = (
        ~(latitude_missing | longitude_missing)
        & (latitude.isna() | longitude.isna())
    )
    finite = pd.Series(
        np.isfinite(latitude.to_numpy(dtype=float, na_value=np.nan))
        & np.isfinite(longitude.to_numpy(dtype=float, na_value=np.nan)),
        index=frame.index,
    )
    nonfinite = (
        ~(latitude_missing | longitude_missing | nonnumeric) & ~finite
    )
    out_of_range = (
        ~(latitude_missing | longitude_missing | nonnumeric | nonfinite)
        & (
            ~latitude.between(-90.0, 90.0)
            | ~longitude.between(-180.0, 180.0)
        )
    )
    state = pd.Series("valid", index=frame.index, dtype="string")
    state.loc[both_missing] = "missing"
    state.loc[partial_missing] = "partial_missing"
    state.loc[nonnumeric] = "nonnumeric"
    state.loc[nonfinite] = "nonfinite"
    state.loc[out_of_range] = "out_of_range"
    return latitude, longitude, state


def _surface(
    origins_projected: gpd.GeoDataFrame,
    frame: pd.DataFrame,
    *,
    surface: str,
    id_column: str,
    latitude_column: str,
    longitude_column: str,
    contract: SpatialContractV2,
) -> tuple[np.ndarray, pd.Series, dict[str, Any], pd.DataFrame]:
    _require_unique(frame, id_column, f"{surface} facilities")
    latitude, longitude, coordinate_state = _coordinate_state(
        frame, latitude_column, longitude_column
    )
    valid = coordinate_state.eq("valid")
    usable = frame.loc[valid].copy()
    if usable.empty:
        distance = np.full(len(origins_projected), np.nan)
        tie_count = pd.Series(
            pd.array([pd.NA] * len(origins_projected), dtype="Int64")
        )
    else:
        usable_lat = latitude.loc[valid]
        usable_lon = longitude.loc[valid]
        facilities = gpd.GeoDataFrame(
            usable[[id_column]].reset_index(drop=True),
            geometry=gpd.points_from_xy(
                usable_lon.reset_index(drop=True), usable_lat.reset_index(drop=True)
            ),
            crs=contract.supply_crs,
        ).to_crs(contract.distance_crs)
        origin_xy = np.column_stack((origins_projected.geometry.x, origins_projected.geometry.y))
        facility_xy = np.column_stack((facilities.geometry.x, facilities.geometry.y))
        pairwise = np.sqrt(
            np.square(origin_xy[:, None, 0] - facility_xy[None, :, 0])
            + np.square(origin_xy[:, None, 1] - facility_xy[None, :, 1])
        )
        distance = pairwise.min(axis=1)
        tie_count = pd.Series(
            np.isclose(
                pairwise, distance[:, None], rtol=0.0,
                atol=contract.tie_tolerance_m,
            ).sum(axis=1).astype(int),
            dtype="Int64",
        )
        if (distance < 0).any() or tie_count.lt(1).any():
            raise ValueError("nearest-distance result violates spatial invariants")
    state_counts = coordinate_state.value_counts().to_dict()
    qa = {
        "surface": surface,
        "surface_status": "ok" if len(usable) else "empty_surface",
        "input_rows": len(frame),
        "valid_coordinate_rows": int(valid.sum()),
        "missing_coordinate_rows": int(
            state_counts.get("missing", 0) + state_counts.get("partial_missing", 0)
        ),
        "both_missing_coordinate_rows": int(state_counts.get("missing", 0)),
        "partial_missing_coordinate_rows": int(state_counts.get("partial_missing", 0)),
        "nonnumeric_coordinate_rows": int(state_counts.get("nonnumeric", 0)),
        "nonfinite_coordinate_rows": int(state_counts.get("nonfinite", 0)),
        "out_of_range_coordinate_rows": int(state_counts.get("out_of_range", 0)),
        "invalid_coordinate_rows": int((~valid).sum()),
    }
    ledger = pd.DataFrame({
        "surface": surface,
        "record_id": frame[id_column].astype(str).to_numpy(),
        "coordinate_state": coordinate_state.astype(str).to_numpy(),
        "enters_surface": valid.astype(bool).to_numpy(),
    })
    return distance, tie_count, qa, ledger


def _beyond(distance: pd.Series, threshold_m: float) -> pd.Series:
    result = distance.gt(threshold_m).astype("boolean")
    result.loc[distance.isna()] = pd.NA
    return result


def _rate(values: pd.Series) -> float | None:
    known = values.dropna()
    return round(100.0 * float(known.astype(bool).mean()), 2) if len(known) else None


def _paired(board: pd.Series, google: pd.Series) -> dict[str, Any]:
    known = board.notna() & google.notna()
    left = board.loc[known].astype(bool)
    right = google.loc[known].astype(bool)
    return {
        "paired_origins": int(known.sum()),
        "both_beyond": int((left & right).sum()),
        "board_only_beyond": int((left & ~right).sum()),
        "google_only_beyond": int((~left & right).sum()),
        "neither_beyond": int((~left & ~right).sum()),
        "paired_disagreements": int((left != right).sum()),
    }


def _discordance(board: pd.Series, google: pd.Series) -> pd.Series:
    values = pd.Series("unknown", index=board.index, dtype="string")
    known = board.notna() & google.notna()
    left = board.loc[known].astype(bool)
    right = google.loc[known].astype(bool)
    values.loc[known & board.astype("boolean").fillna(False) & google.astype("boolean").fillna(False)] = "both_beyond"
    values.loc[known & board.astype("boolean").fillna(False) & ~google.astype("boolean").fillna(False)] = "board_only_beyond"
    values.loc[known & ~board.astype("boolean").fillna(False) & google.astype("boolean").fillna(False)] = "google_only_beyond"
    values.loc[known & ~board.astype("boolean").fillna(False) & ~google.astype("boolean").fillna(False)] = "neither_beyond"
    return values


def _assert_monotone(smaller: pd.Series, larger: pd.Series, label: str) -> None:
    comparable = smaller.notna() & larger.notna()
    if (smaller.loc[comparable] > larger.loc[comparable] + 1e-8).any():
        raise ValueError(f"surface set-inclusion invariant failed: {label}")


def run_v2_spatial_analysis(
    google: pd.DataFrame,
    boards: BoardFramesV2,
    origins: gpd.GeoDataFrame,
    contract: SpatialContractV2,
    *,
    analysis_scope: str,
) -> SpatialResultV2:
    """Compute all declared surfaces from the same fixed tract-origin frame."""
    if analysis_scope != "full":
        raise ValueError("substantive tract spatial analysis requires analysis_scope='full'")
    if not isinstance(google, pd.DataFrame) or not isinstance(boards, BoardFramesV2):
        raise TypeError("invalid v2 spatial input")
    if not isinstance(origins, gpd.GeoDataFrame) or origins.crs is None:
        raise ValueError("fixed origins require CRS metadata")
    if not isinstance(contract, SpatialContractV2):
        raise TypeError("invalid spatial contract")
    required_google = {"place_id", "eligibility_state", "Latitude", "Longitude"}
    if missing := required_google - set(google.columns):
        raise ValueError(f"Google spatial frame lacks columns: {sorted(missing)}")
    _require_unique(google, "place_id", "Google spatial frame")
    if not set(google["eligibility_state"]) <= {"include", "exclude", "unresolved"}:
        raise ValueError("Google spatial frame has invalid eligibility state")
    _require_unique(origins, contract.origin_id_column, "fixed origin frame")
    if origins.geometry.isna().any() or origins.geometry.is_empty.any() or (~origins.geometry.is_valid).any():
        raise ValueError("fixed origins contain unusable geometry")
    if not origins.geometry.geom_type.eq("Point").all():
        raise ValueError("fixed origins must be Point geometries")

    projected = origins.to_crs(contract.distance_crs).copy()
    if projected.crs is None or projected.crs.is_geographic:
        raise ValueError("origin projection failed")
    units = {axis.unit_name.casefold() for axis in projected.crs.axis_info}
    if not units <= {"metre", "meter"}:
        raise ValueError("projected origin CRS does not use meters")
    if not np.isfinite(projected.geometry.x).all() or not np.isfinite(projected.geometry.y).all():
        raise ValueError("projected origins contain nonfinite coordinates")
    origin_sha256 = _origin_sha256(projected, contract.origin_id_column)

    surfaces = {
        "google_include": (
            google.loc[google["eligibility_state"].eq("include")].copy(),
            "place_id", "Latitude", "Longitude",
        ),
        "google_include_unresolved": (
            google.loc[google["eligibility_state"].isin(("include", "unresolved"))].copy(),
            "place_id", "Latitude", "Longitude",
        ),
        "board_screened": (
            boards.screened_extended.copy(), "license_nbr", "_lat", "_lon",
        ),
        "board_screened_exact": (
            boards.screened_extended.loc[
                boards.screened_extended["mtype"].astype(str).eq("Exact")
            ].copy(),
            "license_nbr", "_lat", "_lon",
        ),
        "board_all_active": (
            boards.full_active_extended.copy(), "license_nbr", "_lat", "_lon",
        ),
    }

    safe_origin_columns = (
        contract.origin_id_column, "target_zcta", "COUNTYFP", "med_hh_income",
        "total_pop", "pct_nonwhite", "total_hh", "carless_rate",
        "income_quartile", "nonwhite_quartile",
    )
    output = pd.DataFrame(
        origins[[column for column in safe_origin_columns if column in origins]].copy()
    ).reset_index(drop=True)
    qa_rows: list[dict[str, Any]] = []
    coordinate_ledgers: list[pd.DataFrame] = []
    for name, (facilities, identifier, lat, lon) in surfaces.items():
        distances, ties, qa, ledger = _surface(
            projected, facilities, surface=name, id_column=identifier,
            latitude_column=lat, longitude_column=lon, contract=contract,
        )
        output[f"{name}_nearest_m"] = distances
        output[f"{name}_nearest_tie_count"] = ties.array
        qa_rows.append(qa)
        coordinate_ledgers.append(ledger)

    _assert_monotone(
        output["google_include_unresolved_nearest_m"],
        output["google_include_nearest_m"],
        "include plus unresolved versus include",
    )
    _assert_monotone(
        output["board_all_active_nearest_m"],
        output["board_screened_nearest_m"],
        "all active versus screened Board",
    )
    _assert_monotone(
        output["board_screened_nearest_m"],
        output["board_screened_exact_nearest_m"],
        "screened Board versus exact-only Board",
    )

    threshold_summaries: list[dict[str, Any]] = []
    surface_names = tuple(surfaces)
    for threshold_miles in contract.thresholds_miles:
        threshold_m = threshold_miles * contract.meters_per_mile
        suffix = str(threshold_miles).replace(".", "_") + "mi"
        for name in surface_names:
            output[f"{name}_beyond_{suffix}"] = _beyond(
                output[f"{name}_nearest_m"], threshold_m
            )
        board_primary = output[f"board_screened_beyond_{suffix}"]
        google_primary = output[f"google_include_beyond_{suffix}"]
        google_conditional = output[f"google_include_unresolved_beyond_{suffix}"]
        output[f"discordance_type_{suffix}"] = _discordance(
            board_primary, google_primary
        )
        output[f"conditional_discordance_type_{suffix}"] = _discordance(
            board_primary, google_conditional
        )
        threshold_summaries.append({
            "threshold_miles": threshold_miles,
            "threshold_meters": round(threshold_m, 3),
            "origin_count": len(output),
            "surfaces": {
                name: {
                    "known_origins": int(output[f"{name}_beyond_{suffix}"].notna().sum()),
                    "beyond_count": int(output[f"{name}_beyond_{suffix}"].sum(skipna=True)),
                    "beyond_pct": _rate(output[f"{name}_beyond_{suffix}"]),
                }
                for name in surface_names
            },
            "primary_google_vs_board": _paired(board_primary, google_primary),
            "conditional_google_vs_board": _paired(board_primary, google_conditional),
        })

    if "income_quartile" not in output and "med_hh_income" in output:
        output["income_quartile"] = _quartile(output["med_hh_income"])
    if "nonwhite_quartile" not in output and "pct_nonwhite" in output:
        output["nonwhite_quartile"] = _quartile(output["pct_nonwhite"])
    summary = {
        "adapter_version": SPATIAL_ADAPTER_VERSION,
        "analysis_scope": analysis_scope,
        "estimand": "projected straight-line potential proximity",
        "distance_crs": CRS.from_user_input(contract.distance_crs).to_string(),
        "distance_units": "meter",
        "supply_input_crs": CRS.from_user_input(contract.supply_crs).to_string(),
        "origin_count": len(output),
        "origin_id_column": contract.origin_id_column,
        "origin_sha256": origin_sha256,
        "spatial_contract_sha256": contract.contract_sha256,
        "spatial_contract": asdict(contract),
        "origin_status": "ok" if len(output) else "empty_origin_frame",
        "input_sha256": {
            "google": _frame_input_sha256(
                google, "place_id",
                ("place_id", "eligibility_state", "Latitude", "Longitude"),
            ),
            "board_screened_extended": _frame_input_sha256(
                boards.screened_extended, "license_nbr",
                ("license_nbr", "_lat", "_lon", "mtype"),
            ),
            "board_full_active_extended": _frame_input_sha256(
                boards.full_active_extended, "license_nbr",
                ("license_nbr", "_lat", "_lon", "mtype"),
            ),
        },
        "surface_definitions": {
            "google_include": "Google candidates reduced to include",
            "google_include_unresolved": "conditional sensitivity: Google include plus unresolved",
            "board_screened": "automatically screened active Board operational reference",
            "board_screened_exact": "screened Board records with Exact cached coordinates only",
            "board_all_active": "all active Board records in the extended frame",
        },
        "thresholds": threshold_summaries,
        "coordinate_qa": qa_rows,
    }
    qa_frame = pd.DataFrame(qa_rows).sort_values("surface").reset_index(drop=True)
    coordinate_ledger = pd.concat(
        coordinate_ledgers, ignore_index=True
    ) if coordinate_ledgers else pd.DataFrame(columns=(
        "surface", "record_id", "coordinate_state", "enters_surface",
    ))
    coordinate_ledger = coordinate_ledger.sort_values(
        ["surface", "record_id"]
    ).reset_index(drop=True)
    return SpatialResultV2(output, qa_frame, coordinate_ledger, summary)


__all__ = [
    "M_PER_MILE",
    "SPATIAL_ADAPTER_VERSION",
    "SpatialContractV2",
    "SpatialResultV2",
    "prepare_fixed_tract_origins",
    "run_v2_spatial_analysis",
]
