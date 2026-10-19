"""HOLC 1937 redlining overlay vs April 2026 AI pharmacy locations.

Source: Mapping Inequality (DSL Richmond) full-US HOLC GeoJSON, filtered to
Minneapolis + St. Paul (n=108 zones, 24 D-grade redlined).

Output: docs/thesis/figures/figure8_holc_overlay.png
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

FIG_DIR = Path("docs/thesis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load HOLC zones (Twin Cities subset)
holc = gpd.read_file("data/holc_twin_cities.geojson").to_crs(4326)

# Load AI pharmacy points
tp = pd.read_csv("data/audit2_tp_analysis_20260414_172017.csv")
fp = pd.read_csv("data/audit2_fp_analysis_20260414_172017.csv")
all_ai = pd.concat([tp, fp], ignore_index=True)
all_ai = all_ai.dropna(subset=["Latitude", "Longitude"])
ai_gdf = gpd.GeoDataFrame(
    all_ai,
    geometry=gpd.points_from_xy(all_ai["Longitude"], all_ai["Latitude"]),
    crs="EPSG:4326",
)

# HOLC grade colors (canonical Mapping Inequality palette)
grade_colors = {
    "A": "#4CAF50",  # green - "Best"
    "B": "#2196F3",  # blue  - "Still Desirable"
    "C": "#FFC107",  # yellow - "Definitely Declining"
    "D": "#E53935",  # red   - "Hazardous" / redlined
}

fig, ax = plt.subplots(figsize=(9, 8.5))

# Plot HOLC zones by grade
for grade in ["A", "B", "C", "D"]:
    sub = holc[holc["grade"] == grade]
    if len(sub):
        sub.plot(ax=ax,
                 facecolor=grade_colors[grade], edgecolor="black",
                 linewidth=0.4, alpha=0.55,
                 label=f"Grade {grade} (n={len(sub)})")

# Plot AI pharmacies on top
ai_gdf.plot(ax=ax, color="black", markersize=8, alpha=0.75,
            marker="o", label=f"AI-collected pharmacy (n={len(ai_gdf)})")

# Set bounds to Twin Cities core
minx, miny, maxx, maxy = holc.total_bounds
ax.set_xlim(minx - 0.02, maxx + 0.02)
ax.set_ylim(miny - 0.02, maxy + 0.02)

# Custom legend
legend_handles = [
    mpatches.Patch(color=grade_colors["A"], alpha=0.55, label='Grade A "Best" (n=20)'),
    mpatches.Patch(color=grade_colors["B"], alpha=0.55, label='Grade B "Still Desirable" (n=34)'),
    mpatches.Patch(color=grade_colors["C"], alpha=0.55, label='Grade C "Definitely Declining" (n=28)'),
    mpatches.Patch(color=grade_colors["D"], alpha=0.55, label='Grade D "Hazardous" / redlined (n=24)'),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markersize=8, label="AI-collected pharmacy"),
]
ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=9)

ax.set_title(
    "HOLC 1937 redlining overlay with current AI-collected pharmacy locations\n"
    "Twin Cities urban core (Minneapolis + St. Paul, n=108 HOLC zones, n=399 pharmacies)",
    fontsize=11, pad=10,
)
ax.set_axis_off()

plt.tight_layout()
plt.savefig(FIG_DIR / "figure8_holc_overlay.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"wrote {FIG_DIR / 'figure8_holc_overlay.png'}")

# ── Compute pharmacy density by HOLC grade ──
holc_proj = holc.to_crs(3857)
ai_proj = ai_gdf.to_crs(3857)
joined = gpd.sjoin(ai_proj, holc_proj, how="inner", predicate="within")
density_rows = []
for grade in ["A", "B", "C", "D"]:
    sub = holc_proj[holc_proj["grade"] == grade]
    n_zones = len(sub)
    area_km2 = sub.geometry.area.sum() / 1e6
    n_phar = (joined["grade"] == grade).sum()
    density = n_phar / area_km2 if area_km2 else 0
    density_rows.append({
        "grade": grade, "zones": n_zones,
        "area_km2": round(area_km2, 2),
        "pharmacies_in_zone": int(n_phar),
        "phar_per_km2": round(density, 3),
    })
print()
print("Pharmacy density by HOLC grade:")
for r in density_rows:
    print(f"  {r}")

import json
Path("data/figures_board/_holc_density.json").write_text(json.dumps(density_rows, indent=2))
