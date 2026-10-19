"""Read-only scientific/cartographic preflight for the v2 figure rebuild."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "analysis/v2/final_results_20260902T012636Z_figure_rebuild_v1"
FIGURE_DATA = ROOT / "analysis/v2/final_results_20260902T012636Z_visual_redesign_v3/figure_data"
CONTEXT = ROOT / "data/cartographic_context/tigerline_2023"
TARGET_CRS = "EPSG:26915"
COUNTIES = {"053": "Hennepin", "123": "Ramsey"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layer_record(path: Path, county_boundary) -> dict:
    frame = gpd.read_file(path)
    metric = frame.to_crs(TARGET_CRS)
    intersects = metric.geometry.intersects(county_boundary)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "features": int(len(frame)),
        "geometry_types": {str(k): int(v) for k, v in frame.geom_type.value_counts().items()},
        "crs": str(frame.crs),
        "bounds_source_crs": [round(float(value), 6) for value in frame.total_bounds],
        "null_geometry": int(frame.geometry.isna().sum()),
        "empty_geometry": int(frame.geometry.is_empty.sum()),
        "invalid_geometry": int((~frame.geometry.is_valid & frame.geometry.notna()).sum()),
        "features_intersecting_two_counties": int(intersects.sum()),
    }


def main() -> None:
    paths = {
        "tracts": ROOT / "data/tl_2023_27_tract.zip",
        "places": CONTEXT / "tl_2023_27_place.zip",
        "hennepin_water": CONTEXT / "tl_2023_27053_areawater.zip",
        "ramsey_water": CONTEXT / "tl_2023_27123_areawater.zip",
        "roads": CONTEXT / "tl_2023_27_prisecroads.zip",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    tracts = gpd.read_file(paths["tracts"])
    county_tracts = tracts.loc[
        (tracts["STATEFP"] == "27") & tracts["COUNTYFP"].isin(COUNTIES)
    ].copy()
    county_metric = county_tracts.to_crs(TARGET_CRS)
    county_boundary = county_metric.geometry.union_all()

    states = pd.read_csv(FIGURE_DATA / "figure05_tract_states.csv", dtype={"GEOID": str})
    state_geoids = set(states["GEOID"])
    county_geoids = set(county_tracts["GEOID"])

    places = gpd.read_file(paths["places"])
    selected_places = places.loc[places["NAME"].isin(["Minneapolis", "St. Paul"])]

    water = pd.concat(
        [gpd.read_file(paths["hennepin_water"]), gpd.read_file(paths["ramsey_water"])],
        ignore_index=True,
    )
    water = gpd.GeoDataFrame(water, geometry="geometry", crs="EPSG:4269").to_crs(TARGET_CRS)
    water["area_km2"] = water.geometry.area / 1_000_000
    names = water.get("FULLNAME", pd.Series(index=water.index, dtype="object")).fillna("")
    major_water = water.loc[(water["area_km2"] >= 0.20) | names.str.contains("Mississippi", case=False)]

    roads = gpd.read_file(paths["roads"]).to_crs(TARGET_CRS)
    roads_in_counties = roads.loc[roads.geometry.intersects(county_boundary)].copy()
    road_types = {
        str(key): int(value)
        for key, value in roads_in_counties["MTFCC"].value_counts(dropna=False).items()
    }

    records = {
        key: layer_record(path, county_boundary) for key, path in paths.items()
    }
    audit = {
        "scope": "local cartographic preflight; no provider/model workflow",
        "target_crs": TARGET_CRS,
        "county_filter": COUNTIES,
        "layers": records,
        "derived_checks": {
            "two_county_tracts": int(len(county_tracts)),
            "frozen_analyzed_tracts": int(len(states)),
            "frozen_geoids_unique": bool(states["GEOID"].is_unique),
            "frozen_geoids_missing_from_two_county_tracts": sorted(state_geoids - county_geoids),
            "selected_place_features": int(len(selected_places)),
            "selected_place_names": selected_places["NAME"].tolist(),
            "major_water_selection_rule": "area_km2 >= 0.20 OR FULLNAME contains Mississippi",
            "major_water_features": int(len(major_water)),
            "major_water_named_examples": (
                major_water.loc[major_water["FULLNAME"].notna()]
                .sort_values("area_km2", ascending=False)[["FULLNAME", "area_km2"]]
                .head(15)
                .round({"area_km2": 3})
                .to_dict("records")
            ),
            "roads_intersecting_two_counties": int(len(roads_in_counties)),
            "road_mtfcc_counts": road_types,
        },
        "usefulness": {
            "tracts": "Full Hennepin/Ramsey tract context, county dissolves, and 327 analyzed tract joins.",
            "places": "Minneapolis and St. Paul outlines/label anchors only.",
            "water": "Mississippi River and major lake context under a declared area/name rule.",
            "roads": "Version B only; primary/secondary Census roads clipped to the two counties.",
        },
    }
    PACKAGE.mkdir(parents=True, exist_ok=True)
    (PACKAGE / "asset_inventory.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
