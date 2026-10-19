"""
visualize.py  --  Generate spatial/statistical figures for NETS-AI thesis.

Reads outputs from spatial_analysis.py. Run AFTER spatial_analysis.py.

Usage
-----
    python code/visualize.py

Outputs (saved to data/figures/, gitignored)
    figure1_coverage_map.png          Coverage map with income choropleth
    figure2a_desert_map.png           Pharmacy desert binary map (0.5-mile threshold)
    figure2b_distance_scatter.png     Income vs nearest-pharmacy distance
    figure3_wayback_distribution.png  Wayback snapshot distribution (exploratory)
"""

import os
import sys
import zipfile
import tempfile
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe for all platforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_CODE_DIR, ".."))

DATA_DIR    = os.path.join(_ROOT, "data")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

# Figures 1/2a/2b/3 are written twice: a 1200-DPI archival copy under
# data/figures, and a 200-DPI copy under docs/thesis/figures for the manuscript.
#
# Previously only the archival copy was produced and the manuscript copies were
# moved across by hand, so the two drifted: after the projection fix the
# archival figures were regenerated in UTM while the thesis kept including
# months-old Web Mercator renders of the same figure numbers. Figures 4-11 are
# already emitted straight to the thesis directory by their own scripts; this
# closes the same loop for 1-3 so one run regenerates everything.
THESIS_FIGURES_DIR = os.path.join(_ROOT, "docs", "thesis", "figures")
_ARCHIVE_DPI = 1200
_THESIS_DPI = 200


def save_figure(fig, filename: str) -> str:
    """Write the archival and manuscript copies of a figure. Returns the archival path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    archival = os.path.join(FIGURES_DIR, filename)
    fig.savefig(archival, dpi=_ARCHIVE_DPI, bbox_inches="tight")
    os.makedirs(THESIS_FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(THESIS_FIGURES_DIR, filename),
                dpi=_THESIS_DPI, bbox_inches="tight")
    return archival

_CHAIN_KW = frozenset({
    "cvs", "walgreens", "rite aid", "walmart", "target",
    "costco", "kroger", "hy-vee", "hyvee", "cub",
})

# Minneapolis MSA bounding box for outlier filtering
_MPLS_BBOX = {"lon_min": -93.8, "lon_max": -92.8, "lat_min": 44.7, "lat_max": 45.3}

# North Minneapolis ZIP codes to highlight
_NORTH_ZIPS = frozenset({"55411", "55412"})

# Urban pharmacy desert threshold: 0.5 miles in meters (Qato et al. 2014)
_DESERT_M = 804

# CRS split — these are NOT interchangeable.
#
# _DIST_CRS  metric distance computation. UTM Zone 15N covers Minnesota and is
#            true-to-scale locally. Every distance, centroid and nearest-neighbour
#            join must use it.
# _MAP_CRS   display only, because contextily basemap tiles are published in Web
#            Mercator. Web Mercator is NOT valid for distance: at ~45degN it
#            inflates ground distance by ~1/cos(45deg) ~ 1.41x, so an 804 m
#            threshold behaves like ~0.35 mi and the desert rate is overstated.
_DIST_CRS = "EPSG:26915"   # UTM 15N  -- all measurement
_MAP_CRS  = "EPSG:3857"    # Web Mercator -- all rendering


def is_chain(name: str) -> bool:
    n = str(name).lower()
    return any(kw in n for kw in _CHAIN_KW)


def _load_zip_boundaries(data_dir: str):
    """
    Fetch 55411 / 55412 boundaries from Census TIGERweb REST API (just two polygons,
    no large download). Caches result as a GeoJSON file.
    """
    cache_path = os.path.join(data_dir, "north_mpls_zcta.geojson")
    if not os.path.exists(cache_path):
        params = urllib.parse.urlencode({
            "where": "ZCTA5 IN ('55411','55412')",
            "outFields": "ZCTA5",
            "f": "geojson",
        })
        url = (
            "https://tigerweb.geo.census.gov/arcgis/rest/services/"
            "TIGERweb/PUMA_TAD_TAZ_UGA_ZCTA/MapServer/1/query?" + params
        )
        print("    Fetching ZIP boundaries from Census TIGERweb REST API ...")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            with open(cache_path, "w") as fh:
                json.dump(data, fh)
        except Exception as exc:
            print(f"    Warning: TIGERweb fetch failed ({exc}); ZIP outlines skipped.")
            return None
    try:
        gdf = gpd.read_file(cache_path)
        return gdf.to_crs(epsg=4326) if not gdf.empty else None
    except Exception as exc:
        print(f"    Warning: ZCTA GeoJSON load failed ({exc}); ZIP outlines skipped.")
        return None


def _load_tract_gdf(acs_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Load TIGER/Line shapefile and merge ACS data."""
    cache_path = os.path.join(DATA_DIR, "tl_2023_27_tract.zip")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(cache_path, "r") as z:
            z.extractall(tmp)
        shp = next(Path(tmp).glob("*.shp"))
        gdf = gpd.read_file(str(shp))
    gdf = gdf[gdf["COUNTYFP"].isin(["053", "123"])].copy()
    gdf = gdf.to_crs(epsg=4326)
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    acs_df["GEOID"] = acs_df["GEOID"].astype(str)
    gdf = gdf.merge(acs_df, on="GEOID", how="left")
    return gdf


def load_data():
    """Load all required files for visualization."""
    ai_df  = pd.read_csv(os.path.join(DATA_DIR, "spatial_ai_tracts.csv"))
    fn_df  = pd.read_csv(os.path.join(DATA_DIR, "spatial_nppes_fn_tracts.csv"))
    acs_df = pd.read_csv(os.path.join(DATA_DIR, "acs_tracts_2023.csv"),
                         dtype={"GEOID": str})
    gdf    = _load_tract_gdf(acs_df)
    return ai_df, fn_df, gdf


# ---- Figure 1: Coverage map ----

def figure1_coverage_map(ai_df: pd.DataFrame, fn_df: pd.DataFrame,
                          gdf: gpd.GeoDataFrame) -> None:
    """
    OSM basemap + income quintile choropleth (alpha=0.4) in EPSG:3857.
    Blue dots = AI pharmacies. Red triangles = NPPES possible-missed-retail.
    Dashed outlines = ZIP 55411 / 55412 (North Minneapolis).
    Falls back to no basemap if contextily or network unavailable.
    """
    try:
        import contextily as ctx
        _has_ctx = True
    except ImportError:
        _has_ctx = False
        print("    Warning: contextily not installed; OSM basemap skipped.")

    _CRS = "EPSG:3857"

    # Reproject census tracts to Web Mercator
    gdf_3857 = gdf.to_crs(_CRS)
    has_income = gdf_3857["med_hh_income"].notna() & (gdf_3857["med_hh_income"] > 0)
    gdf_data = gdf_3857[has_income].copy()
    gdf_na   = gdf_3857[~has_income]

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    gdf_na.plot(ax=ax, color="#d9d9d9", linewidth=0.2, edgecolor="white", alpha=0.35)

    if not gdf_data.empty:
        gdf_data["q"] = pd.qcut(gdf_data["med_hh_income"], q=5,
                                  labels=False, duplicates="drop")
        # YlOrRd_r: q=0 (lowest income) -> red, q=4 (highest income) -> yellow
        cmap = matplotlib.colormaps.get_cmap("YlOrRd_r").resampled(5)
        for q in range(5):
            sub = gdf_data[gdf_data["q"] == q]
            if not sub.empty:
                sub.plot(ax=ax, color=cmap(q / 4), linewidth=0.2,
                         edgecolor="white", alpha=0.35)

    # AI pharmacy points: convert lat/lon -> EPSG:3857 via GeoDataFrame
    ai_valid = ai_df[
        ai_df["Latitude"].notna() & ai_df["Longitude"].notna() &
        ai_df["Latitude"].between(_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]) &
        ai_df["Longitude"].between(_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"])
    ].copy()
    if len(ai_valid) > 0:
        ai_pts = gpd.GeoDataFrame(
            ai_valid,
            geometry=gpd.points_from_xy(ai_valid["Longitude"], ai_valid["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(_CRS)
        ax.scatter(ai_pts.geometry.x, ai_pts.geometry.y,
                   c="#1f78b4", s=1, zorder=5, alpha=0.85)

    # NPPES possible-missed-retail: same projection approach
    fn_missed = fn_df[
        (fn_df.get("FN_Category", "") == "possible_missed_retail")
        & fn_df["Latitude"].notna()
        & fn_df["Longitude"].notna()
    ]
    fn_missed = fn_missed[
        fn_missed["Latitude"].between(_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]) &
        fn_missed["Longitude"].between(_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"])
    ].copy()
    if not fn_missed.empty:
        fn_pts = gpd.GeoDataFrame(
            fn_missed,
            geometry=gpd.points_from_xy(fn_missed["Longitude"], fn_missed["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(_CRS)
        ax.scatter(fn_pts.geometry.x, fn_pts.geometry.y,
                   c="#e31a1c", s=2, zorder=5, alpha=0.85, marker="^")

    # North Minneapolis ZIP outlines in EPSG:3857
    zip_gdf = _load_zip_boundaries(DATA_DIR)
    if zip_gdf is not None:
        zip_3857 = zip_gdf.to_crs(_CRS)
        zip_3857.boundary.plot(ax=ax, color="#222222", linewidth=1.5,
                               linestyle="--", zorder=4)
        for _, row in zip_3857.iterrows():
            centroid = row.geometry.centroid
            _, _, _, ymax = row.geometry.bounds  # northernmost edge in meters
            ax.annotate(
                row["ZCTA5"],
                xy=(centroid.x, ymax + 600),    # 600 m above polygon top
                fontsize=9, color="#111111", fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    # Clip view to MSA bbox (convert corners to EPSG:3857)
    bbox_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"]],
            [_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]],
        ),
        crs="EPSG:4326",
    ).to_crs(_CRS)
    x_min, y_min = bbox_pts.geometry.x[0], bbox_pts.geometry.y[0]
    x_max, y_max = bbox_pts.geometry.x[1], bbox_pts.geometry.y[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # OSM basemap (zorder=0 by default, sits beneath all layers)
    if _has_ctx:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                            zoom=13, crs=_CRS)
        except Exception as exc:
            print(f"    Warning: basemap failed ({exc}); continuing without.")

    # Convert meter tick positions back to lat/lon labels
    import pyproj
    _t = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xticks = [v for v in ax.get_xticks() if x_min <= v <= x_max]
    yticks = [v for v in ax.get_yticks() if y_min <= v <= y_max]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{_t.transform(x, y_min)[0]:.2f}\u00b0" for x in xticks],
                       fontsize=8)
    ax.set_yticklabels([f"{_t.transform(x_min, y)[1]:.2f}\u00b0" for y in yticks],
                       fontsize=8)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)

    # Legend
    if not gdf_data.empty:
        cmap = matplotlib.colormaps.get_cmap("YlOrRd_r").resampled(5)
        income_patches = [
            mpatches.Patch(color=cmap(i / 4), alpha=0.6,
                           label=f"Income Quintile Q{i+1}")
            for i in range(5)
        ]
    else:
        income_patches = []

    point_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f78b4",
               markersize=5, label=f"AI-Collected Pharmacy (n={len(ai_valid)})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e31a1c",
               markersize=5,
               label=f"NPPES-Only Possible Retail (n={len(fn_missed)})"),
        mpatches.Patch(color="#d9d9d9", label="No income data"),
        Line2D([0], [0], color="#222222", linewidth=1.5, linestyle="--",
               label="ZIP 55411 / 55412 boundary"),
    ]
    ax.legend(handles=income_patches + point_handles,
              loc="lower right", fontsize=8, framealpha=0.9, title="Legend",
              markerscale=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "AI-Collected vs NPPES-Only Pharmacies\n"
        "Minneapolis MSA by Median Household Income Quintile",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Interim ground truth disclaimer -- above OSM attribution
    ax.text(
        0.01, 0.04,
        "Note: NPPES NPI Registry used as interim ground truth; "
        "MN Board of Pharmacy licensure data requested and pending.",
        transform=ax.transAxes, fontsize=8, style="italic", color="#666666",
        ha="left", va="bottom",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3),
        zorder=10,
    )

    out = save_figure(fig, "figure1_coverage_map.png")
    plt.close(fig)
    print(f"    Saved: {out}")


# ---- Shared spatial helper for Figure 2a / 2b ----

def compute_nearest_pharmacy_dist(ai_df: pd.DataFrame,
                                   gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    For every census tract centroid, compute distance (meters) to nearest AI pharmacy.
    Distances are computed in UTM Zone 15N (_DIST_CRS), never in Web Mercator.
    Returns DataFrame with GEOID, dist_m, med_hh_income, total_pop, pct_nonwhite.
    """
    pharm_valid = ai_df[
        ai_df["Latitude"].notna() & ai_df["Longitude"].notna() &
        ai_df["Latitude"].between(_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]) &
        ai_df["Longitude"].between(_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"])
    ].copy()

    acs_cols = [c for c in ["GEOID", "med_hh_income", "total_pop", "pct_nonwhite"]
                if c in gdf.columns]
    tracts_utm = gdf[acs_cols + ["geometry"]].copy().to_crs(_DIST_CRS)
    tracts_utm["geometry"] = tracts_utm.geometry.centroid

    if pharm_valid.empty:
        tracts_utm["dist_m"] = float("inf")
        return tracts_utm[acs_cols + ["dist_m"]].reset_index(drop=True)

    pharm_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pharm_valid["Longitude"], pharm_valid["Latitude"]),
        crs="EPSG:4326",
    ).to_crs(_DIST_CRS)

    joined = gpd.sjoin_nearest(
        tracts_utm[acs_cols + ["geometry"]],
        pharm_pts[["geometry"]],
        how="left",
        distance_col="dist_m",
    ).drop_duplicates(subset=["GEOID"])

    return joined[acs_cols + ["dist_m"]].reset_index(drop=True)


def _north_mpls_tract_geoids(gdf: gpd.GeoDataFrame) -> set:
    """Return GEOIDs of tracts whose centroids fall within ZIP 55411 or 55412."""
    geojson_path = os.path.join(DATA_DIR, "north_mpls_zcta.geojson")
    if not os.path.exists(geojson_path):
        return set()
    try:
        zcta = gpd.read_file(geojson_path).to_crs("EPSG:4326")
        # Centroids in the measurement CRS so they agree with the distance join
        centroids = gdf[["GEOID", "geometry"]].copy().to_crs(_DIST_CRS)
        centroids["geometry"] = centroids.geometry.centroid
        centroids = centroids.to_crs("EPSG:4326")
        joined = gpd.sjoin(centroids, zcta[["geometry"]], how="inner", predicate="within")
        return set(joined["GEOID"].astype(str).tolist())
    except Exception:
        return set()


# ---- Figure 2a: Pharmacy desert binary map ----

def figure2a_desert_map(ai_df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> None:
    """
    Binary pharmacy desert choropleth.
    Desert = tract centroid >= 804 m (0.5 mi, Qato et al. 2014) from nearest AI pharmacy.
    Overlay: AI pharmacy points, CartoDB Positron basemap, ZIP 55411/55412 outlines.
    """
    try:
        import contextily as ctx
        _has_ctx = True
    except ImportError:
        _has_ctx = False

    _CRS = "EPSG:3857"

    dist_df = compute_nearest_pharmacy_dist(ai_df, gdf)
    dist_df["is_desert"] = dist_df["dist_m"] >= _DESERT_M

    # Merge classification back to geometry for plotting
    gdf_3857 = gdf[["GEOID", "geometry"]].copy().to_crs(_CRS)
    gdf_3857 = gdf_3857.merge(dist_df[["GEOID", "is_desert"]], on="GEOID", how="left")
    gdf_3857["is_desert"] = gdf_3857["is_desert"].fillna(True)

    covered = gdf_3857[~gdf_3857["is_desert"]]
    desert  = gdf_3857[gdf_3857["is_desert"]]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    covered.plot(ax=ax, color="#c8e6c9", linewidth=0.2, edgecolor="white", alpha=0.6)
    desert.plot(ax=ax, color="#8b0000",  linewidth=0.2, edgecolor="white", alpha=0.6)

    # AI pharmacy points in EPSG:3857
    ai_valid = ai_df[
        ai_df["Latitude"].notna() & ai_df["Longitude"].notna() &
        ai_df["Latitude"].between(_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]) &
        ai_df["Longitude"].between(_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"])
    ].copy()
    if len(ai_valid) > 0:
        ai_pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(ai_valid["Longitude"], ai_valid["Latitude"]),
            crs="EPSG:4326",
        ).to_crs(_CRS)
        ax.scatter(ai_pts.geometry.x, ai_pts.geometry.y,
                   c="#1f78b4", s=2, zorder=5, alpha=0.9)

    # ZIP 55411/55412 outlines
    zip_gdf = _load_zip_boundaries(DATA_DIR)
    if zip_gdf is not None:
        zip_3857 = zip_gdf.to_crs(_CRS)
        zip_3857.boundary.plot(ax=ax, color="#222222", linewidth=1.5,
                               linestyle="--", zorder=4)
        for _, row in zip_3857.iterrows():
            centroid = row.geometry.centroid
            _, _, _, ymax = row.geometry.bounds
            ax.annotate(
                row["ZCTA5"],
                xy=(centroid.x, ymax + 600),
                fontsize=9, color="#111111", fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

    # Clip view to MSA bbox
    bbox_pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [_MPLS_BBOX["lon_min"], _MPLS_BBOX["lon_max"]],
            [_MPLS_BBOX["lat_min"], _MPLS_BBOX["lat_max"]],
        ),
        crs="EPSG:4326",
    ).to_crs(_CRS)
    x_min, y_min = bbox_pts.geometry.x[0], bbox_pts.geometry.y[0]
    x_max, y_max = bbox_pts.geometry.x[1], bbox_pts.geometry.y[1]
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if _has_ctx:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                            zoom=12, crs=_CRS)
        except Exception as exc:
            print(f"    Warning: basemap failed ({exc})")

    # Lat/lon tick labels
    import pyproj
    _t = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    xticks = [v for v in ax.get_xticks() if x_min <= v <= x_max]
    yticks = [v for v in ax.get_yticks() if y_min <= v <= y_max]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{_t.transform(x, y_min)[0]:.2f}\u00b0" for x in xticks], fontsize=8)
    ax.set_yticklabels([f"{_t.transform(x_min, y)[1]:.2f}\u00b0" for y in yticks], fontsize=8)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)

    n_desert  = int(gdf_3857["is_desert"].sum())
    n_covered = len(gdf_3857) - n_desert
    legend_handles = [
        mpatches.Patch(color="#c8e6c9", alpha=0.8,
                       label=f"Covered (< 804 m)  n = {n_covered}"),
        mpatches.Patch(color="#8b0000", alpha=0.8,
                       label=f"Pharmacy Desert (>= 804 m)  n = {n_desert}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f78b4",
               markersize=5, label=f"AI Pharmacy (n = {len(ai_valid)})"),
        Line2D([0], [0], color="#222222", linewidth=1.5, linestyle="--",
               label="ZIP 55411 / 55412 boundary"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.9, title="Legend")

    ax.set_title(
        "Pharmacy Desert Classification by 0.5-Mile Threshold (AI-Collected Data)\n"
        "Threshold: 804 m from nearest pharmacy (Qato et al. 2014)",
        fontsize=12, fontweight="bold", pad=12,
    )

    out = save_figure(fig, "figure2a_desert_map.png")
    plt.close(fig)
    print(f"    Saved: {out}")


# ---- Figure 2b: Income vs nearest-pharmacy distance ----

def figure2b_distance_scatter(ai_df: pd.DataFrame, gdf: gpd.GeoDataFrame) -> None:
    """
    Scatter: tract median income (x) vs distance to nearest AI pharmacy in meters (y).
    All tracts included. OLS + 95% CI. North Mpls tracts (55411/55412) annotated.
    """
    dist_df = compute_nearest_pharmacy_dist(ai_df, gdf)

    plot_df = dist_df[
        dist_df["med_hh_income"].notna() & (dist_df["med_hh_income"] > 0) &
        dist_df["total_pop"].notna()     & (dist_df["total_pop"] > 0) &
        dist_df["dist_m"].notna()        & np.isfinite(dist_df["dist_m"])
    ].copy().reset_index(drop=True)

    if len(plot_df) < 3:
        print("    Figure 2b: fewer than 3 tracts, skipping.")
        return

    plot_df["pct_nonwhite"] = plot_df["pct_nonwhite"].fillna(0)

    # Identify North Mpls tracts
    north_geoids = _north_mpls_tract_geoids(gdf)
    plot_df["is_north"] = plot_df["GEOID"].astype(str).isin(north_geoids)

    fig, ax = plt.subplots(figsize=(10, 8))

    sizes = ((plot_df["total_pop"] / plot_df["total_pop"].max()) * 40 + 4).clip(4, 44)
    norm  = mcolors.Normalize(vmin=plot_df["pct_nonwhite"].min(),
                               vmax=plot_df["pct_nonwhite"].max())

    # All non-North-Mpls tracts
    mask = ~plot_df["is_north"]
    sc = ax.scatter(
        plot_df.loc[mask, "med_hh_income"], plot_df.loc[mask, "dist_m"],
        s=sizes[mask], c=plot_df.loc[mask, "pct_nonwhite"],
        cmap="PuRd", norm=norm,
        alpha=0.72, edgecolors="grey", linewidths=0.4, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="% Non-White Population")

    # North Mpls tracts -- larger diamond marker, bold edge, no text labels
    nm = plot_df[plot_df["is_north"]]
    if not nm.empty:
        ax.scatter(
            nm["med_hh_income"], nm["dist_m"],
            s=sizes[plot_df["is_north"]] * 2.5,
            c=nm["pct_nonwhite"], cmap="PuRd", norm=norm,
            alpha=0.95, edgecolors="#111111", linewidths=1.8,
            marker="D", zorder=5,
        )

    # OLS + 95% CI
    x = plot_df["med_hh_income"].values
    y = plot_df["dist_m"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_pred  = np.linspace(x.min(), x.max(), 200)
    y_pred  = slope * x_pred + intercept
    n       = len(x)
    x_bar   = x.mean()
    SSxx    = np.sum((x - x_bar) ** 2)
    s_e     = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
    se_pred = s_e * np.sqrt(1 / n + (x_pred - x_bar) ** 2 / SSxx)
    t_crit  = stats.t.ppf(0.975, df=n - 2)

    ax.plot(x_pred, y_pred, color="#333333", linewidth=1.5, linestyle="--",
            label=f"OLS fit  r = {r:.2f}, p = {p:.3f}")
    ax.fill_between(x_pred,
                    y_pred - t_crit * se_pred,
                    y_pred + t_crit * se_pred,
                    color="#333333", alpha=0.12, label="95% CI")

    # Horizontal desert threshold line
    ax.axhline(_DESERT_M, color="#8b0000", linewidth=1.2, linestyle=":",
               label=f"0.5-mi desert threshold ({_DESERT_M} m)")

    # Always add North Mpls legend entry (diamonds present or not)
    ax.scatter([], [], marker="D", c="#888888", s=20,
               edgecolors="#111111", linewidths=1.0,
               label="North Mpls tracts (55411/55412)")

    ax.set_xlabel("Median Household Income ($)", fontsize=11)
    ax.set_ylabel("Distance to Nearest AI Pharmacy (m)", fontsize=11)
    ax.set_title(
        "Distance to Nearest Pharmacy by Neighborhood Income\n"
        "(bubble size = tract population; color = % non-white; "
        "diamonds = North Mpls 55411/55412)",
        fontsize=12, fontweight="bold",
    )
    ax.text(0.01, 0.01,
            "Higher values indicate greater pharmacy access barriers",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Y-axis cap at 12 km
    ax.set_ylim(bottom=0, top=12000)
    n_outliers = int((plot_df["dist_m"] > 12000).sum())
    if n_outliers > 0:
        ax.text(0.98, 0.97, f"{n_outliers} tracts > 12 km (not shown)",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="gray", style="italic")

    out = save_figure(fig, "figure2b_distance_scatter.png")
    plt.close(fig)
    print(f"    Saved: {out}")


# ---- Figure 3: Wayback distribution ----

def figure3_wayback_distribution(ai_df: pd.DataFrame) -> None:
    # Exploratory figure only -- not part of primary Comparison A/B/C framework
    # Wayback Machine enrichment is an optional pipeline feature, not a core methodology
    """
    Grouped bar chart: Wayback snapshot bucket vs pharmacy count.
    Two bars per bucket: chain vs independent (by name keyword).
    """
    print("    Note: Figure 3 is exploratory. "
          "For thesis main text, use Figures 1, 2a, 2b.")
    if "Wayback_Snapshot_Count" not in ai_df.columns:
        print("    Figure 3: Wayback_Snapshot_Count not found, skipping.")
        return

    df = ai_df.copy()
    df["Is_Chain"] = df["Company"].apply(is_chain)

    def _bucket(v) -> str:
        try:
            v = int(v)
        except (TypeError, ValueError):
            return "Unknown"
        if v < 0:
            return "Chain\n(sentinel -1)"
        if v == 0:
            return "No Web\n(0 yrs)"
        if v <= 5:
            return "Small\n(1-5 yrs)"
        if v <= 10:
            return "Moderate\n(6-10 yrs)"
        return "Established\n(10+ yrs)"

    bucket_order = [
        "Chain\n(sentinel -1)", "No Web\n(0 yrs)", "Small\n(1-5 yrs)",
        "Moderate\n(6-10 yrs)", "Established\n(10+ yrs)",
    ]
    df["bucket"] = df["Wayback_Snapshot_Count"].apply(_bucket)

    chains = df[df["Is_Chain"]]["bucket"].value_counts()
    indeps = df[~df["Is_Chain"]]["bucket"].value_counts()

    x     = np.arange(len(bucket_order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, [chains.get(b, 0) for b in bucket_order],
           width, label="Chain Pharmacy",       color="#1f78b4", alpha=0.85)
    ax.bar(x + width / 2, [indeps.get(b, 0) for b in bucket_order],
           width, label="Independent Pharmacy", color="#ff7f00", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_order, fontsize=10)
    ax.set_xlabel("Wayback Machine Snapshot Coverage", fontsize=11)
    ax.set_ylabel("Number of Pharmacies", fontsize=11)
    ax.set_title(
        "Wayback Machine Coverage by Pharmacy Type\n"
        "(AI-collected Minneapolis pharmacies)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    out = save_figure(fig, "figure3_wayback_distribution.png")
    plt.close(fig)
    print(f"    Saved: {out}")


# ---- Main ----

def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    required = [
        os.path.join(DATA_DIR, "spatial_ai_tracts.csv"),
        os.path.join(DATA_DIR, "spatial_nppes_fn_tracts.csv"),
        os.path.join(DATA_DIR, "tl_2023_27_tract.zip"),
        os.path.join(DATA_DIR, "acs_tracts_2023.csv"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("ERROR: Run spatial_analysis.py first. Missing:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print(">>> Loading data ...")
    ai_df, fn_df, gdf = load_data()
    print(f"    AI joined: {len(ai_df)} rows | FN joined: {len(fn_df)} rows")

    print(">>> Figure 1: Coverage map ...")
    figure1_coverage_map(ai_df, fn_df, gdf)

    print(">>> Figure 2a: Pharmacy desert map ...")
    figure2a_desert_map(ai_df, gdf)

    print(">>> Figure 2b: Income vs nearest-pharmacy distance scatter ...")
    figure2b_distance_scatter(ai_df, gdf)

    print(">>> Figure 3: Wayback distribution ...")
    figure3_wayback_distribution(ai_df)

    print(f"\n>>> visualize.py complete. Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
