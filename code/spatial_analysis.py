"""
spatial_analysis.py  --  Spatial analysis pipeline for NETS-AI thesis.

Joins AI-collected pharmacy data and NPPES false-negatives to Census tract
socioeconomic variables for equity analysis.

Steps
-----
  1. Re-fetch NPPES to extract FN records not matched by AI
  2. Geocode FN records via Census Geocoder API
  3. Fetch ACS 5-year estimates for Hennepin + Ramsey census tracts
  4. Download TIGER/Line tract shapefile for Minnesota
  5. Spatial join AI and FN points to tracts
  6. Write output CSVs
  7. North Minneapolis case study (55411, 55412)
  8. Summary statistics

Usage
-----
    python code/spatial_analysis.py

All intermediate results are cached in data/ to avoid re-fetching on reruns.
"""

import os
import sys
import time
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from urllib.parse import urlencode

# Make validate_nppes importable from code/
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_CODE_DIR, ".."))
for _p in (_CODE_DIR, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validate_nppes import fetch_nppes_pharmacies, nppes_to_df

DATA_DIR = os.path.join(_ROOT, "data")

# ---- Census API endpoints ----
CENSUS_ACS_URL = "https://api.census.gov/data/2023/acs/acs5"
TIGER_URL      = ("https://www2.census.gov/geo/tiger/TIGER2023/TRACT/"
                  "tl_2023_27_tract.zip")
GEOCODER_URL   = ("https://geocoding.geo.census.gov/geocoder/locations/"
                  "onelineaddress")

ACS_VARS = {
    "B19013_001E": "med_hh_income",
    "B03002_001E": "total_pop",
    "B03002_003E": "pop_nonhisp_white",
    "B25003_001E": "total_housing_units",
}

# Chain keywords for AI dataset chain/independent classification
_CHAIN_KW = frozenset({
    "cvs", "walgreens", "rite aid", "walmart", "target",
    "costco", "kroger", "hy-vee", "hyvee", "cub",
})

NORTH_MPLS_ZIPS = {"55411", "55412"}


# ---- Classification helpers ----

def is_chain(name: str) -> bool:
    n = str(name).lower()
    return any(kw in n for kw in _CHAIN_KW)


def classify_nppes_fn(name: str) -> str:
    """
    Classify a NPPES FN record into four categories for the thesis taxonomy:
      closed_chain         -- acquired or shuttered retail chain
      specialty_nonretail  -- non-consumer-facing (institutional, infusion, mail)
      corporate_legal_name -- parent entity name with no consumer-facing signals
      possible_missed_retail -- likely real retail pharmacy AI failed to find
    """
    n = str(name).lower()

    # Closed / acquired chains
    if any(x in n for x in [
        "snyder", "bioscrip", "bio scrip", "kerr drug", "revco", "eckerd",
    ]):
        return "closed_chain"

    # Specialty / non-consumer-facing
    if any(x in n for x in [
        "wellcare", "infusion", "institutional", "long-term", "ltc",
        "hospice", "oncology", "home health", "nuclear",
    ]):
        return "specialty_nonretail"
    if "mail" in n:
        return "specialty_nonretail"
    if "specialty" in n and not any(x in n for x in ["pharmacy", "drug"]):
        return "specialty_nonretail"

    # Corporate parent / legal entity name
    if any(x in n for x in [
        "supervalu", "holdings", "enterprises", "solutions",
        "management", "midwest medical",
    ]):
        return "corporate_legal_name"
    # Generic corporate suffix without a retail keyword
    has_suffix  = any(x in n for x in [" inc", " llc", " corp", " corporation"])
    has_retail  = any(x in n for x in ["pharmacy", "drug", "rx", "apothecary"])
    if has_suffix and not has_retail:
        return "corporate_legal_name"

    return "possible_missed_retail"


# ---- ACS data ----

def fetch_acs_tracts(state: str = "27",
                     counties: list | None = None) -> pd.DataFrame:
    """Fetch ACS 5-year tract estimates for Hennepin (053) and Ramsey (123)."""
    if counties is None:
        counties = ["053", "123"]

    rows = []
    for county in counties:
        params = {
            "get": ",".join(["NAME"] + list(ACS_VARS.keys())),
            "for": "tract:*",
            "in":  f"state:{state} county:{county}",
        }
        url = CENSUS_ACS_URL + "?" + urlencode(params)
        print(f"    ACS county {county} ...", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            header = data[0]
            for rec in data[1:]:
                rows.append(dict(zip(header, rec)))
            print(f"{len(data) - 1} tracts")
        except Exception as e:
            print(f"FAILED: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    for api_col, new_col in ACS_VARS.items():
        if api_col in df.columns:
            df[new_col] = pd.to_numeric(df[api_col], errors="coerce")
    df["pct_nonwhite"] = np.where(
        df["total_pop"] > 0,
        (df["total_pop"] - df["pop_nonhisp_white"]) / df["total_pop"] * 100,
        np.nan,
    )
    keep = ["GEOID", "NAME", "med_hh_income", "total_pop",
            "pop_nonhisp_white", "total_housing_units", "pct_nonwhite"]
    return df[[c for c in keep if c in df.columns]]


# ---- TIGER/Line shapefile ----

def load_tract_shapefile(cache_dir: str | None = None) -> gpd.GeoDataFrame:
    """Download MN tract shapefile once; cache as zip; filter to target counties."""
    if cache_dir is None:
        cache_dir = DATA_DIR
    cache_path = os.path.join(cache_dir, "tl_2023_27_tract.zip")

    if not os.path.exists(cache_path):
        print("    Downloading MN TIGER/Line tract shapefile (~28 MB) ...")
        resp = requests.get(TIGER_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"    Saved to {cache_path}")
    else:
        print(f"    Using cached shapefile: {cache_path}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(cache_path, "r") as z:
            z.extractall(tmp)
        shp_files = list(Path(tmp).glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError("No .shp in TIGER zip")
        gdf = gpd.read_file(str(shp_files[0]))

    gdf = gdf[gdf["COUNTYFP"].isin(["053", "123"])].copy()
    gdf = gdf.to_crs(epsg=4326)
    print(f"    {len(gdf)} tracts loaded (Hennepin + Ramsey).")
    return gdf


# ---- Census Geocoder ----

def _geocode_one(address: str, city: str,
                 state: str, zipcode: str) -> tuple[float | None, float | None]:
    """One address -> (lat, lon) via Census Geocoder. Returns (None, None) on failure."""
    full = ", ".join(filter(None, [address, city, state])).strip()
    if zipcode:
        full = f"{full} {zipcode}"
    params = {
        "address":   full,
        "benchmark": "Public_AR_Current",
        "format":    "json",
    }
    try:
        resp    = requests.get(GEOCODER_URL, params=params, timeout=12)
        matches = resp.json().get("result", {}).get("addressMatches", [])
        if matches:
            c = matches[0]["coordinates"]
            return float(c["y"]), float(c["x"])
    except Exception:
        pass
    return None, None


def geocode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add Latitude / Longitude columns to a DataFrame of NPPES records."""
    lats, lons = [], []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        if i == 0 or (i + 1) % 25 == 0:
            print(f"    Geocoding {i + 1}/{n} ...", flush=True)
        lat, lon = _geocode_one(
            str(row.get("Address", "")),
            str(row.get("City",    "")),
            str(row.get("State",   "MN")),
            str(row.get("ZIP",     ""))[:5],
        )
        lats.append(lat)
        lons.append(lon)
        time.sleep(0.5)

    out = df.copy()
    out["Latitude"]  = lats
    out["Longitude"] = lons
    return out


# ---- NPPES FN extraction ----

def load_nppes_fn(ai_df: pd.DataFrame, val_df: pd.DataFrame,
                  state: str = "MN") -> pd.DataFrame:
    """
    Re-fetch NPPES for all AI ZIPs and return records NOT matched by AI
    (false negatives). Adds FN_Category column.
    """
    # NPIs are stored as float64 (e.g. 1740711753.0); convert via int to get
    # clean string "1740711753" matching NPPES API output.
    matched_npis = set(
        val_df.loc[val_df["Is_NPPES_Match"] == True, "NPPES_Match_NPI"]
              .dropna()
              .astype(int)
              .astype(str)
    )
    zip_codes = sorted(
        ai_df["Zip_Code"].dropna().astype(str).str[:5].unique().tolist()
    )
    print(f"    Querying NPPES for {len(zip_codes)} ZIP codes ...")
    records  = fetch_nppes_pharmacies(state=state, zip_codes=zip_codes)
    nppes_df = nppes_to_df(records)
    print(f"    {len(nppes_df)} active NPPES records total.")

    fn_df = nppes_df[~nppes_df["NPI"].astype(str).isin(matched_npis)].copy()
    fn_df["FN_Category"] = fn_df["Name"].apply(classify_nppes_fn)
    print(f"    {len(fn_df)} false-negative records identified.")
    return fn_df


# ---- Spatial join ----

def spatial_join_to_tracts(points_df: pd.DataFrame,
                            tracts_gdf: gpd.GeoDataFrame,
                            acs_df:     pd.DataFrame,
                            lat_col:    str = "Latitude",
                            lon_col:    str = "Longitude") -> pd.DataFrame:
    """
    Spatial join points -> census tracts -> ACS variables.
    Records with missing coordinates are kept with NULL GEOID.
    """
    has_coords = points_df[[lat_col, lon_col]].notna().all(axis=1)
    pts_valid  = gpd.GeoDataFrame(
        points_df[has_coords].copy(),
        geometry=gpd.points_from_xy(
            points_df.loc[has_coords, lon_col],
            points_df.loc[has_coords, lat_col],
        ),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        pts_valid,
        tracts_gdf[["GEOID", "geometry"]],
        how="left", predicate="within",
    )
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    # Attach ACS variables
    acs_df["GEOID"] = acs_df["GEOID"].astype(str)
    joined = joined.merge(acs_df, on="GEOID", how="left")

    # Re-attach records with no coordinates
    no_coord = points_df[~has_coords].copy()
    if not no_coord.empty:
        for col in list(acs_df.columns) + ["GEOID"]:
            if col not in no_coord.columns:
                no_coord[col] = np.nan
        joined = pd.concat([joined, no_coord], ignore_index=True)

    return joined.reset_index(drop=True)


# ---- North Minneapolis analysis ----

def north_mpls_analysis(ai_df: pd.DataFrame, fn_df: pd.DataFrame,
                         output_path: str) -> None:
    """Write North Minneapolis case study report to output_path."""
    ai_north = ai_df[
        ai_df["Zip_Code"].astype(str).str[:5].isin(NORTH_MPLS_ZIPS)
    ].copy()
    fn_north = fn_df[
        fn_df["ZIP"].astype(str).str[:5].isin(NORTH_MPLS_ZIPS)
    ].copy()

    cat_labels = {
        "closed_chain":          "Closed / Acquired Chain",
        "corporate_legal_name":  "Corporate Legal Name (non-retail)",
        "specialty_nonretail":   "Specialty / Non-Retail",
        "possible_missed_retail":"Possible Missed Retail Pharmacy",
    }
    cat_order = list(cat_labels.keys())

    lines = [
        "North Minneapolis Pharmacy Analysis",
        "ZIP codes 55411 and 55412",
        "=" * 60,
        "",
        f"AI-Collected Pharmacies  (n={len(ai_north)})",
        "-" * 40,
    ]

    for _, r in ai_north.iterrows():
        wb = r.get("Wayback_Snapshot_Count", "?")
        try:
            wb = int(wb)
            wb_label = ("chain" if wb == -1
                        else "no web" if wb == 0
                        else f"{wb} snapshot-yrs")
        except (TypeError, ValueError):
            wb_label = str(wb)
        lines.append(
            f"  {r['Company']}\n"
            f"    ZIP {r.get('Zip_Code', '')} | "
            f"Lat {float(r.get('Latitude', 0)):.4f} "
            f"Lon {float(r.get('Longitude', 0)):.4f} | "
            f"Wayback: {wb_label}"
        )

    lines += [
        "",
        f"NPPES False Negatives  (n={len(fn_north)})",
        "-" * 40,
    ]
    for cat in cat_order:
        subset = fn_north[fn_north["FN_Category"] == cat]
        lines.append(f"\n  {cat_labels[cat]}  (n={len(subset)})")
        for _, r in subset.iterrows():
            lines.append(
                f"    {r['Name']} | {r.get('Address', '')},"
                f" {r.get('City', '')} {r.get('ZIP', '')}"
            )

    lines += ["", "Category Summary", "-" * 40]
    for cat in cat_order:
        cnt = (fn_north["FN_Category"] == cat).sum()
        lines.append(f"  {cat_labels[cat]}: {cnt}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Written: {output_path}")


# ---- Summary statistics ----

def generate_summary(val_df:    pd.DataFrame,
                     fn_df:     pd.DataFrame,
                     ai_joined: pd.DataFrame,
                     output_path: str) -> None:
    """Write aggregate statistics to analysis_summary.txt."""
    from datetime import date

    tp = int(val_df["Is_NPPES_Match"].sum())
    fp = len(val_df) - tp
    fn = len(fn_df)

    cat_labels = {
        "closed_chain":          "Closed / Acquired Chain",
        "corporate_legal_name":  "Corporate Legal Name",
        "specialty_nonretail":   "Specialty / Non-Retail",
        "possible_missed_retail":"Possible Missed Retail",
    }

    lines = [
        "NETS-AI Pharmacy Dataset -- Analysis Summary",
        f"Generated: {date.today()}",
        "=" * 60,
        "",
        "1. Baseline Counts",
        f"   AI records collected    : {len(val_df)}",
        f"   NPPES records (same ZIPs): {tp + fn}",
        f"   True Positives (TP)     : {tp}",
        f"   False Positives (FP)    : {fp}",
        f"   False Negatives (FN)    : {fn}",
    ]
    prec = tp / len(val_df)     if len(val_df) > 0 else 0.0
    rec  = tp / (tp + fn)       if (tp + fn)    > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    lines += [
        f"   Precision               : {prec:.1%}",
        f"   Recall                  : {rec:.1%}",
        f"   F1                      : {f1:.1%}",
        "",
        "2. Chain vs Independent (AI data, name-keyword classification)",
        f"   Chain pharmacies        : {val_df['Company'].apply(is_chain).sum()}",
        f"   Independent pharmacies  : {(~val_df['Company'].apply(is_chain)).sum()}",
        "",
        "3. NPPES FN Classification",
    ]
    fn_counts = fn_df["FN_Category"].value_counts()
    for cat, label in cat_labels.items():
        lines.append(f"   {label:<37}: {fn_counts.get(cat, 0)}")

    lines.append("")
    lines.append("4. Coverage by Tract Income Quartile")
    if ("med_hh_income" in ai_joined.columns
            and ai_joined["med_hh_income"].notna().any()):
        tract_df = (
            ai_joined.dropna(subset=["GEOID"])
            .groupby("GEOID")
            .agg(
                n_pharm=("Company",       "count"),
                income =("med_hh_income", "first"),
                pop    =("total_pop",     "first"),
            )
            .query("income > 0 and pop > 0")
        )
        tract_df["density"] = tract_df["n_pharm"] / tract_df["pop"] * 1000
        try:
            q_labels = ["Q1 (lowest income)", "Q2", "Q3", "Q4 (highest income)"]
            tract_df["q"] = pd.qcut(tract_df["income"], q=4,
                                     labels=q_labels, duplicates="drop")
            for q in q_labels:
                sub = tract_df[tract_df["q"] == q]
                lines.append(
                    f"   {q:<22}: {sub['density'].mean():.2f} pharm/1k pop "
                    f"(n={len(sub)} tracts)"
                )
        except Exception as e:
            lines.append(f"   [Could not compute quartiles: {e}]")
    else:
        lines.append("   [ACS income data not available]")

    lines += [
        "",
        "5. North Minneapolis (55411, 55412)",
    ]
    for z in sorted(NORTH_MPLS_ZIPS):
        ai_z = (val_df["Zip_Code"].astype(str).str[:5] == z).sum()
        fn_z = (fn_df["ZIP"].astype(str).str[:5] == z).sum()
        lines.append(f"   ZIP {z}: {ai_z} AI records, {fn_z} NPPES FN records")

    lines += [
        "",
        "6. Wayback Machine Coverage (AI data)",
    ]
    wb_labels = {-1: "Known chain (sentinel -1)",
                  0: "No web presence (0)"}
    for val, cnt in val_df["Wayback_Snapshot_Count"].value_counts().sort_index().items():
        lbl = wb_labels.get(int(val), f"{int(val)} snapshot-years")
        lines.append(f"   {lbl:<37}: {cnt}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Written: {output_path}")


# ---- Pharmacy desert statistics ----

# Urban desert threshold: 0.5 miles in meters (Qato et al. 2014)
_DESERT_M = 804

# Metric CRS for every distance / centroid / nearest-neighbour operation.
# UTM Zone 15N is true-to-scale over Minnesota. Web Mercator (EPSG:3857) must
# never be used here: at ~45degN it inflates ground distance by ~1.41x, which
# makes the 804 m threshold behave like ~0.35 mi and overstates the desert rate
# by roughly 15 percentage points.
_DIST_CRS = "EPSG:26915"
_MPLS_BBOX = {"lon_min": -93.8, "lon_max": -92.8, "lat_min": 44.7, "lat_max": 45.3}


def generate_desert_stats(val_df: pd.DataFrame,
                          tracts_gdf: gpd.GeoDataFrame,
                          acs_df: pd.DataFrame,
                          output_path: str) -> None:
    """
    Compute pharmacy desert metrics and append a new section to analysis_summary.txt.
    Desert = tract centroid >= 804 m (0.5 mi) from nearest AI pharmacy.
    """
    # Filter AI pharmacies to MSA bbox
    pharm = val_df[
        val_df["Latitude"].notna() & val_df["Longitude"].notna() &
        val_df["Latitude"].between(_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]) &
        val_df["Longitude"].between(_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"])
    ].copy()

    # Build tract GeoDataFrame in the metric measurement CRS with ACS income
    acs_df = acs_df.copy()
    acs_df["GEOID"] = acs_df["GEOID"].astype(str)
    tracts = tracts_gdf[["GEOID", "geometry"]].copy()
    tracts["GEOID"] = tracts["GEOID"].astype(str)
    tracts = tracts.merge(acs_df[["GEOID", "med_hh_income", "total_pop"]], on="GEOID", how="left")
    tracts_utm = tracts.to_crs(_DIST_CRS).copy()
    tracts_utm["geometry"] = tracts_utm.geometry.centroid

    if pharm.empty:
        print("    [desert stats] No pharmacy coordinates available, skipping.")
        return

    pharm_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pharm["Longitude"], pharm["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(_DIST_CRS)

    joined = gpd.sjoin_nearest(
        tracts_utm[["GEOID", "med_hh_income", "total_pop", "geometry"]],
        pharm_pts[["geometry"]],
        how="left",
        distance_col="dist_m",
    ).drop_duplicates(subset=["GEOID"])

    joined["is_desert"] = joined["dist_m"] >= _DESERT_M

    n_total   = len(joined)
    n_desert  = int(joined["is_desert"].sum())
    n_covered = n_total - n_desert

    # Per income-quartile desert rate
    has_income = joined["med_hh_income"].notna() & (joined["med_hh_income"] > 0)
    q_df = joined[has_income].copy()
    q_lines = []
    if len(q_df) >= 4:
        try:
            q_labels = ["Q1 (lowest income)", "Q2", "Q3", "Q4 (highest income)"]
            q_df["q"] = pd.qcut(q_df["med_hh_income"], q=4,
                                 labels=q_labels, duplicates="drop")
            for q in q_labels:
                sub = q_df[q_df["q"] == q]
                rate = sub["is_desert"].mean() * 100 if len(sub) > 0 else float("nan")
                q_lines.append(
                    f"   {q:<22}: {rate:.1f}% desert  (n={len(sub)} tracts)"
                )
        except Exception as e:
            q_lines.append(f"   [Could not compute quartile desert rates: {e}]")
    else:
        q_lines.append("   [Insufficient income data for quartile breakdown]")

    # North Minneapolis tract status
    geojson_path = os.path.join(DATA_DIR, "north_mpls_zcta.geojson")
    nm_lines = []
    if os.path.exists(geojson_path):
        try:
            zcta = gpd.read_file(geojson_path).to_crs("EPSG:4326")
            # Centroids in the measurement CRS so they agree with the distance join
            cent4326 = tracts[["GEOID", "geometry"]].copy().to_crs(_DIST_CRS)
            cent4326["geometry"] = cent4326.geometry.centroid
            cent4326 = cent4326.to_crs("EPSG:4326")
            in_north = gpd.sjoin(cent4326, zcta[["geometry"]], how="inner",
                                 predicate="within")["GEOID"].astype(str).tolist()
            nm_sub = joined[joined["GEOID"].astype(str).isin(in_north)]
            nm_n_desert  = int(nm_sub["is_desert"].sum())
            nm_n_covered = len(nm_sub) - nm_n_desert
            nm_lines.append(
                f"   55411/55412 tracts: {len(nm_sub)} total, "
                f"{nm_n_desert} desert, {nm_n_covered} covered"
            )
        except Exception as e:
            nm_lines.append(f"   [North Mpls spatial join failed: {e}]")
    else:
        nm_lines.append("   [north_mpls_zcta.geojson not found; run visualize.py first]")

    new_section = (
        "\n7. Low-Access Analysis (flat 0.5-mi distance threshold, EPSG:26915)\n"
        "   NOT the Qato et al. 2014 pharmacy desert definition, which is\n"
        "   conditional on community income and vehicle access. This is an\n"
        "   income-blind distance measure; the 0.5 mi cut follows Qato only in\n"
        "   its choice of distance. Being income-blind is what makes it usable\n"
        "   for the income-gradient test below. See code/mobility_desert.py for\n"
        "   the literature-faithful conditional classification.\n"
        f"   Total tracts analyzed   : {n_total}\n"
        f"   Covered (< 804 m)       : {n_covered}\n"
        f"   Low-access (>= 804 m)   : {n_desert}\n"
        f"   Overall low-access rate : {n_desert / n_total * 100:.1f}%\n"
        "\n   Desert rate by income quartile:\n"
        + "\n".join(q_lines) + "\n"
        "\n   North Minneapolis (55411 / 55412):\n"
        + "\n".join(nm_lines)
    )

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(new_section + "\n")
    print(f"    Appended desert stats to: {output_path}")


# ---- Main ----

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    ai_path  = os.path.join(DATA_DIR, "Minneapolis-StPaul_pharmacy_20260413_201606.csv")
    val_path = os.path.join(DATA_DIR, "validation_nppes_20260413.csv")
    ai_df    = pd.read_csv(ai_path)
    val_df   = pd.read_csv(val_path)
    print(f">>> Loaded {len(ai_df)} AI records, {len(val_df)} validation records.")

    # ---- Step 1: NPPES FN ----
    fn_cache = os.path.join(DATA_DIR, "nppes_fn_raw.csv")
    if os.path.exists(fn_cache):
        fn_df = pd.read_csv(fn_cache)
        print(f">>> NPPES FN: loaded {len(fn_df)} records from cache.")
    else:
        print(">>> Step 1: Fetching NPPES false negatives ...")
        fn_df = load_nppes_fn(ai_df, val_df)
        fn_df.to_csv(fn_cache, index=False)
        print(f"    Cached to {fn_cache}")

    # ---- Step 2: Geocode FN ----
    fn_geo_cache = os.path.join(DATA_DIR, "nppes_fn_geocoded.csv")
    if os.path.exists(fn_geo_cache):
        fn_df = pd.read_csv(fn_geo_cache)
        n_geo = fn_df["Latitude"].notna().sum()
        print(f">>> Geocoded FN: loaded from cache ({n_geo}/{len(fn_df)} with coords).")
    else:
        print(">>> Step 2: Geocoding NPPES FN records ...")
        fn_df = geocode_df(fn_df)
        fn_df.to_csv(fn_geo_cache, index=False)
        print(f"    Cached to {fn_geo_cache}")

    # ---- Step 3: ACS data ----
    acs_cache = os.path.join(DATA_DIR, "acs_tracts_2023.csv")
    if os.path.exists(acs_cache):
        acs_df = pd.read_csv(acs_cache, dtype={"GEOID": str})
        print(f">>> ACS: loaded {len(acs_df)} tracts from cache.")
    else:
        print(">>> Step 3: Fetching ACS socioeconomic data ...")
        acs_df = fetch_acs_tracts()
        acs_df.to_csv(acs_cache, index=False)
        print(f"    Cached to {acs_cache}")

    # ---- Step 4: Tract shapefile ----
    print(">>> Step 4: Loading TIGER/Line tract shapefile ...")
    tracts_gdf = load_tract_shapefile()
    tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str)
    acs_df["GEOID"]     = acs_df["GEOID"].astype(str)

    # ---- Step 5: Spatial join ----
    print(">>> Step 5: Spatial join ...")
    # Use val_df (has match columns) as the AI side
    ai_joined = spatial_join_to_tracts(val_df, tracts_gdf, acs_df)
    fn_joined = spatial_join_to_tracts(fn_df,  tracts_gdf, acs_df)
    print(f"    AI: {ai_joined['GEOID'].notna().sum()}/{len(ai_joined)} assigned to tracts")
    print(f"    FN: {fn_joined['GEOID'].notna().sum()}/{len(fn_joined)} assigned to tracts")

    # ---- Step 6: Save joined CSVs ----
    ai_out = os.path.join(DATA_DIR, "spatial_ai_tracts.csv")
    fn_out = os.path.join(DATA_DIR, "spatial_nppes_fn_tracts.csv")
    ai_joined.to_csv(ai_out, index=False)
    fn_joined.to_csv(fn_out, index=False)
    print(f">>> Saved: {ai_out}")
    print(f">>> Saved: {fn_out}")

    # ---- Step 7: North Minneapolis ----
    print(">>> Step 7: North Minneapolis analysis ...")
    nm_out = os.path.join(DATA_DIR, "north_mpls_analysis.txt")
    north_mpls_analysis(val_df, fn_df, nm_out)

    # ---- Step 8: Summary ----
    print(">>> Step 8: Summary statistics ...")
    summary_out = os.path.join(DATA_DIR, "analysis_summary.txt")
    generate_summary(val_df, fn_df, ai_joined, summary_out)

    # ---- Step 9: Pharmacy desert analysis ----
    print(">>> Step 9: Pharmacy desert analysis ...")
    generate_desert_stats(val_df, tracts_gdf, acs_df, summary_out)

    print("\n>>> spatial_analysis.py complete.")


if __name__ == "__main__":
    main()
