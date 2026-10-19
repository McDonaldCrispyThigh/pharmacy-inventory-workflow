"""
generate_figures_board.py  --  Produce Figures B1–B5 (v2) for Board of Pharmacy validation.

Outputs written to data/figures_board/  (never modifies existing files).
1200 DPI PNG.

Usage:
    python code/generate_figures_board.py
"""

import sys, io, warnings, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA   = Path("data")
OUT    = DATA / "figures_board"
OUT.mkdir(exist_ok=True)

BOARD_FN   = DATA / "board_fn_20260414_170601.csv"
AUDIT3     = DATA / "audit3_fn_institutional_20260414_172017.csv"
AUDIT5     = DATA / "audit5_sensitivity_20260414_172017.csv"
QSUMMARY   = DATA / "spatial_quartile_summary_20260414.csv"
BFNQ       = DATA / "board_fn_quartile_proxy_20260414.csv"
TIGER_ZIP  = DATA / "tl_2023_27_tract.zip"
ACS        = DATA / "acs_tracts_2023.csv"
AI_TRACTS  = DATA / "spatial_ai_tracts.csv"

DPI = 1200

# ── Style ──────────────────────────────────────────────────────────────────────
BLUE_DARK  = "#1B4F8A"
BLUE_LIGHT = "#7FB3D3"
GRAY_DARK  = "#555555"
GRAY_LIGHT = "#BBBBBB"
ORANGE     = "#E87722"
RED        = "#C0392B"
GREEN      = "#27AE60"

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B1 v2 — Validation metrics: raw vs adjusted
# ══════════════════════════════════════════════════════════════════════════════

def figure_b1():
    print("  Generating Figure B1 v2 ...")

    metrics = {
        "Precision": (80.5, 80.5),   # raw, adj (precision unchanged)
        "Recall":    (69.9, 79.3),
        "F1":        (74.8, 79.9),
    }

    labels  = list(metrics.keys())
    raw     = [metrics[k][0] for k in labels]
    adj     = [metrics[k][1] for k in labels]

    x    = np.arange(len(labels))
    w    = 0.32
    gap  = 0.06

    fig, ax = plt.subplots(figsize=(12, 7))

    bars_raw = ax.bar(x - w/2 - gap/2, raw, w, color=BLUE_DARK,
                      label="Raw  (n = 459 Board records)", zorder=3)
    bars_adj = ax.bar(x + w/2 + gap/2, adj, w, color=BLUE_LIGHT,
                      label="Adjusted retail-only  (n = 405)", zorder=3,
                      edgecolor=BLUE_DARK, linewidth=0.8)

    # Value labels
    for bar, val in zip(bars_raw, raw):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=BLUE_DARK)
    for bar, val in zip(bars_adj, adj):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=BLUE_DARK)

    # 80 % reference line
    ax.axhline(80, color=RED, linestyle="--", linewidth=1.4, zorder=2)
    ax.text(2.62, 80.8, "80% reference", color=RED, fontsize=9.5, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=11, loc="lower right", framealpha=0.9)

    # Annotation box
    annot = (
        "Adjustment excludes 54 non-retail records\n"
        "from Board reference denominator:\n"
        "  • 50 institutional  (hospital outpatient,\n"
        "       FQHC, mental health)\n"
        "  •   4 specialty non-retail  (oncology)"
    )
    ax.text(0.015, 0.97, annot, transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor=GRAY_DARK, alpha=0.95))

    fig.suptitle(
        "AI Pipeline Performance Against Minnesota Board of Pharmacy Licensure Data",
        fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(
        "Minneapolis–St. Paul MSA · April 2026 · n=399 AI records · "
        "n=459 Board records (raw) / n=405 (retail-only)",
        fontsize=10, color=GRAY_DARK, pad=6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "figure_b1_validation_metrics_v2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B2 v2 — Board FN spatial distribution map
# ══════════════════════════════════════════════════════════════════════════════

def figure_b2():
    print("  Generating Figure B2 v2 ...")
    import geopandas as gpd
    try:
        import contextily as ctx
        HAS_CTX = True
    except Exception:
        HAS_CTX = False
        print("    contextily not available — skipping basemap")

    # Load TIGER tracts
    tiger = gpd.read_file(f"zip://{TIGER_ZIP.resolve()}").to_crs(epsg=4326)
    acs   = pd.read_csv(ACS)
    tiger["GEOID"] = tiger["GEOID"].astype(str)
    acs["GEOID"]   = acs["GEOID"].astype(str)
    tracts = tiger.merge(acs, on="GEOID", how="left")

    # Restrict to MSA bounding box
    TARGET_ZIPS = {
        "55401","55402","55403","55404","55405","55406","55407","55408","55409","55410",
        "55411","55412","55413","55414","55415","55454","55455",
        "55101","55102","55103","55104","55105","55106","55107","55108","55116","55117",
        "55118","55119","55130","55113","55126",
        "55421","55422","55423","55424","55425","55426","55427","55428","55429","55430",
        "55431","55432","55433","55434","55435","55436","55437","55438","55439",
        "55441","55442","55443","55444","55445","55446","55447","55448","55369",
    }
    ai_tracts_df = pd.read_csv(AI_TRACTS)
    ai_zips = set(ai_tracts_df["Zip_Code"].dropna().astype(str).str[:5].unique())
    msa_zips = TARGET_ZIPS | ai_zips

    # Get ZIP centroids from AI pharmacy locations
    ai_geo = ai_tracts_df[["Zip_Code","Latitude","Longitude"]].dropna()
    ai_geo["zip5"] = ai_geo["Zip_Code"].astype(str).str[:5]
    zip_centroids = ai_geo.groupby("zip5").agg(
        lat=("Latitude","median"), lon=("Longitude","median")).reset_index()

    # Board FN data
    fn_all  = pd.read_csv(BOARD_FN)
    fn_pmr  = fn_all[fn_all["FN_Category"] == "possible_missed_retail"].copy()
    fn_pmr["zip5"] = fn_pmr["zip5"].astype(str).str[:5]

    a3 = pd.read_csv(AUDIT3)
    a3["zip5"] = a3["zip5"].astype(str).str[:5]
    fn_inst   = a3[a3["is_institutional"] == True].copy()
    fn_genuine= a3[a3["is_institutional"] == False].copy()

    # Aggregate FN counts by ZIP
    fn_zip_all = fn_pmr.groupby("zip5").size().reset_index(name="fn_count")
    fn_zip_all = fn_zip_all.merge(zip_centroids, on="zip5", how="inner")
    fn_zip_gen = fn_genuine.groupby("zip5").size().reset_index(name="fn_count")
    fn_zip_gen = fn_zip_gen.merge(zip_centroids, on="zip5", how="inner")
    fn_zip_inst= fn_inst.groupby("zip5").size().reset_index(name="fn_count")
    fn_zip_inst= fn_zip_inst.merge(zip_centroids, on="zip5", how="inner")

    # Determine MSA extent
    msa_tracts = tracts[tracts["GEOID"].isin(acs["GEOID"])]
    minx, miny, maxx, maxy = msa_tracts.total_bounds
    pad = 0.1
    bbox = [minx-pad, miny-pad, maxx+pad, maxy+pad]

    tracts_msa = tracts.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    tracts_web = tracts_msa.to_crs(epsg=3857)

    # North Minneapolis ZIPs
    n_mpls_zips = {"55411","55412"}

    fig, ax = plt.subplots(figsize=(14, 12))

    # Base: tract income quartile choropleth
    tracts_web_valid = tracts_web[tracts_web["med_hh_income"].notna()].copy()
    tracts_web_valid.plot(
        column="med_hh_income", ax=ax, cmap="YlOrRd_r",
        alpha=0.55, legend=False, linewidth=0.2, edgecolor="white")

    # Add colorbar manually
    sm = plt.cm.ScalarMappable(
        cmap="YlOrRd_r",
        norm=plt.Normalize(
            vmin=tracts_web_valid["med_hh_income"].quantile(0.05),
            vmax=tracts_web_valid["med_hh_income"].quantile(0.95)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.6)
    cbar.set_label("Median Household Income ($)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v,_: f"${v/1000:.0f}K"))

    # Highlight North Minneapolis ZIPs
    # Use ZIP centroids to find North Mpls tracts (approximate by bbox)
    n_mpls_center = zip_centroids[zip_centroids["zip5"].isin(n_mpls_zips)]
    if len(n_mpls_center):
        import pyproj
        transformer = pyproj.Transformer.from_crs("epsg:4326","epsg:3857",always_xy=True)
        n_cx = float(n_mpls_center["lon"].mean())
        n_cy = float(n_mpls_center["lat"].mean())
        nx3857, ny3857 = transformer.transform(n_cx, n_cy)
        buf = 4000  # ~4 km buffer
        from shapely.geometry import box
        n_box = box(nx3857-buf, ny3857-buf, nx3857+buf, ny3857+buf)
        import geopandas as gpd2
        n_box_gdf = gpd2.GeoDataFrame(geometry=[n_box], crs="epsg:3857")
        n_tracts = tracts_web[tracts_web.intersects(n_box)]
        n_tracts.boundary.plot(ax=ax, color="black", linewidth=1.8, zorder=6)

    # Contextily basemap
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron,
                            attribution_size=7)
        except Exception as e:
            print(f"    basemap failed: {e}")

    # Plot FN bubbles — genuine misses
    import pyproj
    transformer = pyproj.Transformer.from_crs("epsg:4326","epsg:3857",always_xy=True)

    def plot_bubbles(df, color, marker, size_scale, label, zorder=8, alpha=0.8):
        if df.empty: return
        xs, ys = transformer.transform(df["lon"].values, df["lat"].values)
        sizes  = np.sqrt(df["fn_count"].values) * size_scale
        ax.scatter(xs, ys, s=sizes, c=color, marker=marker,
                   alpha=alpha, zorder=zorder, edgecolors="white",
                   linewidths=0.6, label=label)
        for x, y, cnt in zip(xs, ys, df["fn_count"].values):
            if cnt >= 3:
                ax.annotate(str(cnt), (x, y), fontsize=7, ha="center", va="center",
                            color="white", fontweight="bold", zorder=zorder+1)

    plot_bubbles(fn_zip_gen,  color=BLUE_DARK,  marker="o", size_scale=120,
                 label="Genuinely missed retail FN (84 records)")
    plot_bubbles(fn_zip_inst, color=ORANGE, marker="^", size_scale=100,
                 label="Institutional FN — excluded from adj. denominator (50 records)")

    # Mark NorthPoint Health Center
    northpoint_lon, northpoint_lat = -93.3085, 44.9823  # 2220 Plymouth Ave N
    np_x, np_y = transformer.transform(northpoint_lon, northpoint_lat)
    ax.scatter([np_x], [np_y], s=200, c=RED, marker="*", zorder=10,
               edgecolors="white", linewidths=0.8)
    ax.annotate("NorthPoint\nHealth Center\nPharmacy (55411)",
                (np_x, np_y), xytext=(np_x+2200, np_y+2800),
                fontsize=8.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.92, markerscale=1.2)
    ax.set_axis_off()

    fig.suptitle("Geographic Distribution of AI Pipeline Coverage Gaps",
                 fontsize=14, fontweight="bold", y=0.99)
    ax.set_title(
        "Minnesota Board of Pharmacy false negatives · possible_missed_retail · "
        "outer suburban ZIPs predominate\n"
        "Background: median household income by census tract  |  "
        "Black border: North Minneapolis (55411–55412)",
        fontsize=9.5, color=GRAY_DARK, pad=8)

    fig.tight_layout()
    out = OUT / "figure_b2_board_fn_map_v2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B3 v2 — FN rate by income quartile (two panels)
# ══════════════════════════════════════════════════════════════════════════════

def figure_b3():
    print("  Generating Figure B3 v2 ...")

    # Load data
    bfq = pd.read_csv(BFNQ)  # 77 of 134 FNs with ZIP→income match
    a3  = pd.read_csv(AUDIT3)
    a3["zip5"] = a3["zip5"].astype(str).str[:5]

    # Build ZIP→income proxy from AI tracts
    ai_t = pd.read_csv(AI_TRACTS)
    zip_inc = ai_t[["Zip_Code","med_hh_income"]].dropna()
    zip_inc["zip5"] = zip_inc["Zip_Code"].astype(str).str[:5]
    zip_med = zip_inc.groupby("zip5")["med_hh_income"].median()

    LABELS = ["Q1\n(<$70K)", "Q2\n($70K–$90K)", "Q3\n($90K–$117K)", "Q4\n(>$117K)"]
    N_TRACTS = 118  # each quartile

    # Panel A data — all 134 possible_missed_retail (using 77-record proxy)
    pa_counts = bfq["board_fn_count"].values  # 22, 17, 19, 19
    pa_per_tract = pa_counts / N_TRACTS

    # Panel B data — 84 genuine retail misses only
    fn_genuine = a3[a3["is_institutional"] == False].copy()
    fn_genuine["zip_income"] = fn_genuine["zip5"].map(zip_med)
    fn_gen_valid = fn_genuine[fn_genuine["zip_income"].notna()].copy()

    q_edges = [0, 70366, 90100, 117110, 999999]
    q_labels_raw = ["Q1 (<$70K)", "Q2 ($70K-$90K)", "Q3 ($90K-$117K)", "Q4 (>$117K)"]
    fn_gen_valid["Qinc"] = pd.cut(
        fn_gen_valid["zip_income"], bins=q_edges, labels=q_labels_raw)
    pb_q = fn_gen_valid.groupby("Qinc", observed=True).size().reindex(q_labels_raw, fill_value=0)
    pb_counts   = pb_q.values
    pb_per_tract= pb_counts / N_TRACTS

    # Equal-distribution reference
    pa_equal = pa_counts.sum() / (4 * N_TRACTS)
    pb_equal = pb_counts.sum() / (4 * N_TRACTS)

    colors = [BLUE_DARK, BLUE_DARK, BLUE_DARK, BLUE_DARK]
    alphas = [1.0, 0.80, 0.65, 0.50]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    x = np.arange(4)
    w = 0.5

    for ax, counts, per_tract, equal, n_total, title, ratio_q1, ratio_q4 in [
        (axes[0], pa_counts, pa_per_tract, pa_equal, 134,
         "All Possible Missed Retail FNs\n(n = 134, ZIP-proxy subset n = 77)",
         pa_counts[0], pa_counts[3]),
        (axes[1], pb_counts, pb_per_tract, pb_equal, 84,
         "Genuine Retail Misses Only\n(n = 84, excl. institutional)",
         pb_counts[0], pb_counts[3]),
    ]:
        bar_colors = [plt.cm.Blues(0.4 + 0.5*(i/3)) for i in range(4)]
        bars = ax.bar(x, per_tract, w, color=bar_colors, zorder=3, edgecolor="white")
        ax.axhline(equal, color=GRAY_DARK, linestyle="--", linewidth=1.3,
                   label=f"Equal distribution ({equal:.3f}/tract)", zorder=2)
        # Labels
        for bar, val, cnt in zip(bars, per_tract, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + equal*0.06,
                    f"{val:.3f}\n(n={cnt})",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=BLUE_DARK)

        # Q1:Q4 ratio annotation
        ratio = ratio_q1 / ratio_q4 if ratio_q4 > 0 else float("inf")
        ax.text(0.97, 0.94,
                f"Q1:Q4 = {ratio:.2f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=11, fontweight="bold", color=RED,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose",
                          edgecolor=RED, alpha=0.9))

        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=10)
        ax.set_ylabel("FN count per tract", fontsize=11)
        ax.set_ylim(0, max(per_tract)*1.45)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "AI Coverage Gaps Show No Income Gradient Before or After Institutional Removal",
        fontsize=13, fontweight="bold", y=1.01)
    axes[0].text(0.5, -0.16,
        "Board of Pharmacy false negatives per census tract by neighborhood income quartile\n"
        "(ZIP-level income proxy; 118 tracts per quartile)",
        ha="center", transform=axes[0].transAxes, fontsize=9, color=GRAY_DARK)

    fig.tight_layout()
    out = OUT / "figure_b3_fn_income_quartile_v2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B4 — RETIRED
# ══════════════════════════════════════════════════════════════════════════════
#
# This slot held "AI Pipeline vs. NETS Commercial Database", a placeholder chart
# whose NETS series was hard-coded to zero and annotated "data pending —
# April 27, 2026".
#
# NETS proved unobtainable and the comparison was never run, so the figure is
# retired rather than regenerated: a chart that draws an empty comparison series
# implies a comparison the project cannot make. The figure was never referenced
# by any thesis chapter.
#
# RQ3 is answered against the two administrative sources in hand (MN Board of
# Pharmacy licensure registry and NPPES); see figure_b1 for the validation
# metrics this slot was meant to sit beside, and docs/thesis/nets_reframing_memo.md
# for the framing consequences.
#
# If a comparative leg is restored later (OpenStreetMap via Overpass, Census ZIP
# Business Patterns, or Data Axle), rebuild here against the same n=405
# adjusted retail-only reference set so the bars stay commensurable.


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B5 — Threshold sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def figure_b5():
    print("  Generating Figure B5 ...")

    sens = pd.read_csv(AUDIT5)
    thresh   = sens["threshold"].values
    prec_v   = sens["precision"].values * 100
    rec_v    = sens["recall"].values   * 100
    f1_v     = sens["f1"].values       * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(thresh, prec_v, color=BLUE_DARK,  linestyle="-",  linewidth=2.2,
            marker="o", markersize=8, label="Precision",  zorder=4)
    ax.plot(thresh, rec_v,  color=ORANGE,     linestyle="--", linewidth=2.2,
            marker="s", markersize=8, label="Recall",     zorder=4)
    ax.plot(thresh, f1_v,   color=GREEN,      linestyle=":",  linewidth=2.2,
            marker="^", markersize=9, label="F1 Score",   zorder=4)

    # Selected threshold marker
    ax.axvline(75, color=GRAY_DARK, linestyle="--", linewidth=1.4, zorder=2,
               label="Selected threshold (0.75)")
    ax.text(75.2, min(f1_v)-1.5, "Selected\nthreshold", fontsize=9,
            color=GRAY_DARK, va="top")

    # F1 swing annotation
    f1_swing = max(f1_v) - min(f1_v)
    ax.annotate("",
                xy=(thresh[-1], f1_v[-1]), xytext=(thresh[0], f1_v[0]),
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=1.5))
    mid_x = (thresh[0] + thresh[-1]) / 2
    mid_y = (f1_v[0] + f1_v[-1]) / 2
    ax.text(mid_x, mid_y + 0.5,
            f"F1 swing: {f1_swing:.1f} pp", ha="center", fontsize=10,
            color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=GREEN, alpha=0.9))

    # Value labels
    for t, p, r, f in zip(thresh, prec_v, rec_v, f1_v):
        ax.text(t, p+0.3, f"{p:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=BLUE_DARK)
        ax.text(t, r-1.2, f"{r:.1f}%", ha="center", va="top",
                fontsize=8.5, color=ORANGE)
        ax.text(t, f+0.3, f"{f:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=GREEN)

    ax.set_xlabel("Matching Threshold (RapidFuzz token_sort_ratio × 100)", fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlim(68, 82)
    ax.set_ylim(65, 87)
    ax.set_xticks([70, 75, 80])
    ax.set_xticklabels(["0.70", "0.75", "0.80"], fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)

    fig.suptitle("Validation Metrics Are Robust to Matching Threshold Choice",
                 fontsize=13, fontweight="bold")
    ax.set_title(
        "RapidFuzz token_sort_ratio · AI pipeline vs. Minnesota Board of Pharmacy",
        fontsize=10, color=GRAY_DARK)

    fig.tight_layout()
    out = OUT / "figure_b5_threshold_sensitivity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Manifest
# ══════════════════════════════════════════════════════════════════════════════

def update_manifest():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entries = [
        f"figure_b1_validation_metrics_v2.png | {ts} | Raw vs adjusted precision/recall/F1 grouped bar chart",
        f"figure_b2_board_fn_map_v2.png       | {ts} | Board FN spatial distribution choropleth map",
        f"figure_b3_fn_income_quartile_v2.png | {ts} | FN rate by income quartile, two panels (all vs genuine)",
        f"figure_b5_threshold_sensitivity.png | {ts} | Threshold sensitivity line chart (0.70/0.75/0.80)",
        f"# figure_b4 retired: AI-vs-NETS placeholder; NETS unobtainable, comparison never run",
    ]
    manifest = OUT / "figure_manifest.txt"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    with open(manifest, "a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"\n# Added {ts} (v2 Board validation figures)\n")
        for e in entries:
            f.write(e + "\n")
    print(f"    Manifest updated: {manifest}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 66)
    print("  Generating Board of Pharmacy validation figures (v2)")
    print(f"  Output: {OUT.resolve()}")
    print("=" * 66)

    figure_b1()
    figure_b2()
    figure_b3()
    # figure_b4() retired — see the FIGURE B4 block above
    figure_b5()
    update_manifest()

    print("\n" + "=" * 66)
    print("  ✓ ALL FIGURES COMPLETE")
    # List outputs with file sizes
    for f in sorted(OUT.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<50}  {size_kb:>8.1f} KB")
    print("=" * 66)


if __name__ == "__main__":
    main()
