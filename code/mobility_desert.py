"""Mobility-weighted pharmacy desert analysis.

Formulation
-----------
For each census tract t:
  - carless(t) = ACS B08201_002 / B08201_001 (share of households with 0 vehicles)
  - d(t) = distance from tract centroid to nearest AI-collected pharmacy
  - T_walk  = 0.5 mi = 804.7 m  (Qato pedestrian threshold)
  - T_drive = 2.0 mi = 3218.7 m (rough 4-5 minute urban drive)

Mobility-weighted desert score per tract:
  MWDR(t) = carless(t) * I[d(t) > T_walk]
          + (1 - carless(t)) * I[d(t) > T_drive]

Bounded in [0, 1]. It is the share of tract HOUSEHOLDS whose typical mode (walk
if carless, drive otherwise) does not reach a pharmacy within their effective
threshold.

Units: carless(t) is a household proportion, so every quantity derived from it
stays in households. It must not be multiplied by tract population -- ACS
publishes vehicle availability per household, and carless households are
systematically smaller than car-owning ones, so a "carless population" built
that way estimates nothing ACS measures.

Status: EXPLORATORY / SENSITIVITY ONLY. MWDR is a composite index constructed
for this study; it has no external validation and no published precedent. The
literature-faithful Qato-style classification is the primary desert measure.
Report MWDR as a robustness check, never as a validated accessibility metric.

Outputs
-------
- data/figures_board/_mobility_desert_tract.csv (per-tract)
- data/figures_board/_mobility_desert_quartile.json (Q1-Q4 summary)
- data/figures_board/_threshold_sensitivity.json (0.5 / 1.0 / 2.0 mi by quartile)
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

DATA = Path("data")
OUT = DATA / "figures_board"
OUT.mkdir(parents=True, exist_ok=True)

T_WALK_M  = 804.7   # 0.5 mi
T_DRIVE_M = 3218.7  # 2.0 mi
T_MILE_M  = 1609.34 # 1.0 mi

# ── Qato-style conditional desert definition ────────────────────────────────
#
# Qato et al. 2014 do NOT define a pharmacy desert as "any tract further than
# 0.5 mi from a pharmacy". The published definition is conditional on BOTH
# community income and vehicle access: a low-income community qualifies at a
# SHORTER distance when few households have a car, and at a longer distance
# otherwise. Applying a flat 0.5 mi rule everywhere is a different, unvalidated
# metric that happens to share the number.
#
# !! VERIFY BEFORE PUBLICATION !!
# The exact operationalisation below is reconstructed from secondary summaries.
# Qato et al. 2014 (doi:10.1377/hlthaff.2013.1397) is paywalled and is on the
# CU-Libraries request list (References/_missing/dois_need_library.txt). Open the
# primary article and confirm: the income cut, the vehicle-access cut, and both
# distance thresholds. Adjust the constants here; do not adjust the numbers in
# the manuscript to match this code.
QATO_LOWINCOME_QUANTILE = 0.50   # "low-income community": below-median tract income
QATO_LOWVEHICLE_RATE    = 0.20   # ">=20% of households with no vehicle"
QATO_DIST_LOWVEH_M      = T_WALK_M   # low income + low vehicle access
QATO_DIST_DEFAULT_M     = T_MILE_M   # low income, adequate vehicle access

# !! CIRCULARITY WARNING -- read before using qato_desert in the equity analysis.
#
# The Qato definition has income built into it: a tract cannot be classified as a
# desert unless it is low-income. Cross-tabulating qato_desert against income
# quartile therefore CANNOT test whether pharmacy access is associated with
# income -- the upper quartiles are 0% by construction, not by observation.
#
# Use the two measures for different jobs:
#   qato_desert          -> descriptive prevalence, comparable to published work
#   flat_desert_0_5mi    -> the income-gradient test in RQ4, since it is
#                           income-blind and can therefore be correlated with income
#
# The flat rule is the defensible choice for RQ4, but it must be described as a
# distance-threshold accessibility measure, NOT as "Qato's pharmacy desert
# definition". Conflating the two is what produced the current draft's
# 68.7% "pharmacy desert rate" figure attributed to Qato et al.

# ── Load tract polygons (TIGER 2023, MN counties 053 + 123) ──
tract = gpd.read_file("zip://data/tl_2023_27_tract.zip").to_crs(4326)
tract = tract[tract["COUNTYFP"].isin(["053", "123"])].copy()
print(f"tracts (Hennepin+Ramsey): {len(tract)}")

# ── Merge ACS + vehicles ──
acs = pd.read_csv("data/acs_tracts_2023.csv", dtype={"GEOID": str})
veh = pd.read_csv("data/acs_tracts_vehicles_2023.csv", dtype={"GEOID": str})
tract["GEOID"] = tract["GEOID"].astype(str)
tract = tract.merge(acs, on="GEOID", how="left").merge(veh, on="GEOID", how="left")

# Drop tracts with no income or no households (no analytic basis)
tract = tract[tract["med_hh_income"].notna() & (tract["total_hh"] > 0)].copy()
print(f"tracts after merge: {len(tract)}")

# ── Load AI pharmacy points ──
tp = pd.read_csv("data/audit2_tp_analysis_20260414_172017.csv")
fp = pd.read_csv("data/audit2_fp_analysis_20260414_172017.csv")
ai = pd.concat([tp, fp], ignore_index=True).dropna(subset=["Latitude", "Longitude"])
ai_gdf = gpd.GeoDataFrame(
    ai, geometry=gpd.points_from_xy(ai["Longitude"], ai["Latitude"]),
    crs="EPSG:4326",
)

# Project to UTM 15N (EPSG:26915) for accurate metric distances in MN
proj = "EPSG:26915"
tract_p = tract.to_crs(proj)
ai_p    = ai_gdf.to_crs(proj)

# ── Tract centroids and nearest-pharmacy distance ──
tract_p["centroid"] = tract_p.geometry.centroid
centroids = gpd.GeoDataFrame(
    tract_p[["GEOID", "med_hh_income", "total_pop", "carless_rate", "total_hh"]],
    geometry=tract_p["centroid"],
    crs=proj,
)
nearest = gpd.sjoin_nearest(centroids, ai_p[["geometry"]], how="left",
                             distance_col="dist_m")
# Aggregate (sjoin_nearest may return >1 if ties at exactly equal distance)
nearest = nearest.groupby("GEOID", as_index=False).agg({
    "med_hh_income": "first",
    "total_pop": "first",
    "carless_rate": "first",
    "total_hh": "first",
    "dist_m": "min",
})

# ── Classify ──
# flat_desert_0_5mi: the pure-distance rule. Kept for comparability with studies
# that use it, and as the 0.5 mi row of the sensitivity analysis. It is NOT the
# Qato definition and must not be labelled as such.
nearest["flat_desert_0_5mi"] = (nearest["dist_m"] > T_WALK_M).astype(int)

# qato_desert: the literature-faithful conditional definition (see constants).
_inc_cut = nearest["med_hh_income"].quantile(QATO_LOWINCOME_QUANTILE)
nearest["is_low_income"]   = (nearest["med_hh_income"] <= _inc_cut).astype(int)
nearest["is_low_vehicle"]  = (nearest["carless_rate"] >= QATO_LOWVEHICLE_RATE).astype(int)
nearest["qato_threshold_m"] = np.where(
    nearest["is_low_vehicle"] == 1, QATO_DIST_LOWVEH_M, QATO_DIST_DEFAULT_M
)
nearest["qato_desert"] = (
    (nearest["is_low_income"] == 1) & (nearest["dist_m"] > nearest["qato_threshold_m"])
).astype(int)

nearest["walk_inaccessible"] = (nearest["dist_m"] > T_WALK_M).astype(int)
nearest["drive_inaccessible"] = (nearest["dist_m"] > T_DRIVE_M).astype(int)

# Mobility-weighted desert score, household-weighted within tract.
#
# carless_rate is B08201_002 / B08201_001 -- a proportion of HOUSEHOLDS. It was
# previously multiplied by total_pop to produce a "carless population", which
# mixes measurement units: ACS publishes vehicle availability per household, not
# per person, and carless households are systematically smaller than car-owning
# ones, so the product is not an estimate of anything ACS measures. Everything
# below therefore stays in household units end to end.
nearest["carless_hh"] = nearest["total_hh"] * nearest["carless_rate"]
nearest["car_hh"]     = nearest["total_hh"] * (1 - nearest["carless_rate"])
nearest["mwdr_hh"]    = (
    nearest["carless_hh"] * nearest["walk_inaccessible"]
    + nearest["car_hh"]   * nearest["drive_inaccessible"]
)
nearest["mwdr_score"]  = np.where(
    nearest["total_hh"] > 0,
    nearest["mwdr_hh"] / nearest["total_hh"],
    0.0,
)

# ── Income quartile assignment ──
qcuts = nearest["med_hh_income"].quantile([0.25, 0.5, 0.75]).values
def quartile(x):
    if x <= qcuts[0]: return 1
    if x <= qcuts[1]: return 2
    if x <= qcuts[2]: return 3
    return 4
nearest["Qinc"] = nearest["med_hh_income"].apply(quartile)

# ── Per-quartile aggregation ──
def agg_q(df, q):
    sub = df[df["Qinc"] == q]
    n = len(sub)
    qato_rate = sub["qato_desert"].mean() * 100 if n else 0
    flat_rate = sub["flat_desert_0_5mi"].mean() * 100 if n else 0
    # Household-weighted MWDR for the quartile
    hh_total = sub["total_hh"].sum()
    hh_inaccessible = sub["mwdr_hh"].sum()
    mwdr = (hh_inaccessible / hh_total * 100) if hh_total else 0
    median_carless = sub["carless_rate"].median() * 100
    median_dist_mi = sub["dist_m"].median() / 1609.34
    return {
        "Qinc": q, "tracts": n,
        "median_carless_pct": round(median_carless, 1),
        "median_dist_mi": round(median_dist_mi, 2),
        "qato_desert_rate_pct": round(qato_rate, 1),        # conditional, primary
        "flat_0_5mi_desert_pct": round(flat_rate, 1),       # pure distance, sensitivity
        "mwdr_pct": round(mwdr, 1),                          # exploratory only
    }

qsummary = [agg_q(nearest, q) for q in (1, 2, 3, 4)]
print("--- quartile summary ---")
for r in qsummary: print(r)
(OUT / "_mobility_desert_quartile.json").write_text(json.dumps(qsummary, indent=2))

# Overall MSA
overall_hh = nearest["total_hh"].sum()
overall_inaccess = nearest["mwdr_hh"].sum()
overall_qato = nearest["qato_desert"].mean() * 100
overall_flat = nearest["flat_desert_0_5mi"].mean() * 100
overall_mwdr = overall_inaccess / overall_hh * 100
print(f"--- overall ---")
print(f"Qato conditional desert rate (PRIMARY) : {overall_qato:.1f}%")
print(f"Flat 0.5-mi distance rate (sensitivity): {overall_flat:.1f}%")
print(f"MWDR households (exploratory only)     : {overall_mwdr:.1f}%")
print(f"  low-income tracts : {int(nearest['is_low_income'].sum())} / {len(nearest)}")
print(f"  low-vehicle tracts: {int(nearest['is_low_vehicle'].sum())} / {len(nearest)}")

# ── Threshold sensitivity (0.5 / 1.0 / 2.0 mi) ──
thresh_rows = []
for q in (1, 2, 3, 4):
    sub = nearest[nearest["Qinc"] == q]
    row = {"Qinc": q, "tracts": len(sub)}
    for label, t_mi in (("0.5 mi", 0.5), ("1.0 mi", 1.0), ("2.0 mi", 2.0)):
        t_m = t_mi * 1609.34
        row[f"desert_pct_{label}"] = round((sub["dist_m"] > t_m).mean() * 100, 1)
    thresh_rows.append(row)
print("--- threshold sensitivity ---")
for r in thresh_rows: print(r)
(OUT / "_threshold_sensitivity.json").write_text(json.dumps(thresh_rows, indent=2))

# Save per-tract for figure use
nearest.to_csv(OUT / "_mobility_desert_tract.csv", index=False)
print(f"wrote {OUT/'_mobility_desert_tract.csv'}")
