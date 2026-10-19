"""Fix the FN race breakdown analysis: directly use TIGER tract spatial join
on FN address coordinates we don't have, so use ZIP -> ALL intersecting tracts
weighted average of pct_nonwhite.

Approach: build a ZIP -> tract crosswalk from TIGER tract polygons
intersecting USPS ZCTA boundaries (TIGER 2023 ZCTA shapefile), then for each
FN ZIP, average pct_nonwhite across the intersecting tracts.
"""
import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA = Path("data")
OUT = DATA / "figures_board"

# Load FN retail
fn = pd.read_csv(DATA / "audit3_fn_institutional_20260414_172017.csv",
                 dtype={"zip5": str})
fn_retail = fn[fn["is_institutional"] == False].copy()

# Need ZCTA shapefile to map ZIP -> tract overlap
zcta_url = "https://www2.census.gov/geo/tiger/TIGER2023/ZCTA520/tl_2023_us_zcta520.zip"
zcta_path = DATA / "tl_2023_us_zcta520.zip"
if not zcta_path.exists():
    print(f"Downloading ZCTA shapefile (~500 MB) ...")
    urllib.request.urlretrieve(zcta_url, zcta_path)

# This is a huge file. Filter to MN ZIPs (550xx-565xx) by reading geometry
# and using the ZCTA5 code prefix.
print("Reading ZCTA file (filtered to MN)...")
zcta = gpd.read_file(f"zip://{zcta_path}",
                     where="ZCTA5CE20 LIKE '55%' OR ZCTA5CE20 LIKE '56%'").to_crs(4326)
zcta = zcta.rename(columns={"ZCTA5CE20": "zip5"})

# Tracts with ACS
tract = gpd.read_file("zip://data/tl_2023_27_tract.zip").to_crs(4326)
tract = tract[tract["COUNTYFP"].isin(["053", "123"])].copy()
acs = pd.read_csv(DATA / "acs_tracts_2023.csv", dtype={"GEOID": str})
tract = tract.merge(acs, on="GEOID", how="left")

# Intersect ZCTA with tracts; for each ZIP, list tracts that intersect.
zcta_to_tract = gpd.sjoin(zcta[["zip5", "geometry"]], tract[["GEOID", "pct_nonwhite", "med_hh_income", "geometry"]],
                            how="inner", predicate="intersects")

# Average pct_nonwhite per ZIP
zip_stats = zcta_to_tract.groupby("zip5").agg(
    pct_nonwhite=("pct_nonwhite", "mean"),
    med_hh_income=("med_hh_income", "mean"),
    n_tracts=("GEOID", "nunique"),
).reset_index()
print(f"Built ZIP-tract crosswalk for {len(zip_stats)} MN ZIPs")

# Merge with FN retail
fn_with_pct = fn_retail.merge(zip_stats, on="zip5", how="left")
n_unmapped = int(fn_with_pct["pct_nonwhite"].isna().sum())
print(f"FN unmapped: {n_unmapped} of {len(fn_with_pct)}")

# Compute pct_nonwhite quartile bins on FULL tract universe
nw_quartiles = tract["pct_nonwhite"].quantile([0.25, 0.5, 0.75]).values
def nw_q(x):
    if pd.isna(x): return None
    if x <= nw_quartiles[0]: return 1
    if x <= nw_quartiles[1]: return 2
    if x <= nw_quartiles[2]: return 3
    return 4
fn_with_pct["nw_q"] = fn_with_pct["pct_nonwhite"].apply(nw_q)

race_breakdown = fn_with_pct["nw_q"].value_counts().sort_index().to_dict()
out = {
    "nw_quartile_thresholds_pct": [round(float(x), 1) for x in nw_quartiles],
    "n_total": int(len(fn_retail)),
    "n_unmapped": n_unmapped,
    "fn_count_by_nw_quartile": {f"NW_Q{int(k)}": int(v) for k, v in race_breakdown.items()},
    "fn_count_by_nw_quartile_pct": {
        f"NW_Q{int(k)}": round(int(v)/len(fn_with_pct.dropna(subset=['nw_q']))*100, 1)
        for k, v in race_breakdown.items()
    },
}
print("--- 4 (fixed). Retail FN by pct_nonwhite quartile ---")
print(json.dumps(out, indent=2))
(OUT / "_supp_fn_race_breakdown.json").write_text(json.dumps(out, indent=2))

# Also compute carless and income concentration
fn_carless_breakdown = fn_with_pct[["zip5", "facility_name", "city",
                                     "pct_nonwhite", "med_hh_income"]].copy()
fn_carless_breakdown["nw_q"] = fn_carless_breakdown["pct_nonwhite"].apply(nw_q)
fn_carless_breakdown.to_csv(OUT / "_fn_with_demographics.csv", index=False)
print(f"wrote {OUT/'_fn_with_demographics.csv'}")
