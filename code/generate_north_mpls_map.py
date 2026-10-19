"""Focused close-up map of North Minneapolis (ZIPs 55411 + 55412) showing:
 - the two ZCTA boundaries
 - the active retail pharmacy (Cub at 701 W Broadway)
 - the closed Walgreens (627 W Broadway, Feb 2023)
 - NorthPoint Health Center Pharmacy (FQHC, institutional)

Output: docs/thesis/figures/figure7_north_mpls_focus.png
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd

FIG_DIR = Path("docs/thesis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

zcta = gpd.read_file("data/north_mpls_zcta.geojson").to_crs(4326)

# Hand-coded landmark coordinates (west of the river, North Mpls).
# Sources: Google Maps geocoded street addresses, April 2026.
landmarks = pd.DataFrame([
    {"name": "Cub Pharmacy\n701 W Broadway",   "lat": 44.99989, "lon": -93.28912, "kind": "active_retail"},
    {"name": "Walgreens (closed Feb 2023)\n627 W Broadway", "lat": 44.99908, "lon": -93.28663, "kind": "closed"},
    {"name": "NorthPoint Health Center\nPharmacy (FQHC)",   "lat": 44.99965, "lon": -93.30013, "kind": "fqhc"},
])
landmarks = gpd.GeoDataFrame(
    landmarks,
    geometry=gpd.points_from_xy(landmarks.lon, landmarks.lat),
    crs="EPSG:4326",
)

fig, ax = plt.subplots(figsize=(7.5, 7.0))
zcta.plot(ax=ax, facecolor="#F0E6E6", edgecolor="black", linewidth=1.0)

# ZIP labels
for _, row in zcta.iterrows():
    c = row.geometry.centroid
    ax.annotate(row["ZCTA5"], (c.x, c.y),
                ha="center", va="center", fontsize=14, weight="bold",
                color="#7A4848", alpha=0.6)

# Markers
colors = {"active_retail": "#2E7D32", "closed": "#C0504D", "fqhc": "#1F4E8C"}
markers = {"active_retail": "o", "closed": "X", "fqhc": "s"}
for _, row in landmarks.iterrows():
    ax.plot(row.geometry.x, row.geometry.y,
            marker=markers[row["kind"]], markersize=14,
            markerfacecolor=colors[row["kind"]],
            markeredgecolor="white", markeredgewidth=1.5,
            linestyle="none")
    # Offset label
    dx, dy = 0.005, 0.003
    if row["kind"] == "fqhc":
        dx, dy = -0.005, -0.008
        ha = "right"
    elif row["kind"] == "closed":
        dx, dy = 0.005, -0.008
        ha = "left"
    else:
        dx, dy = 0.005, 0.003
        ha = "left"
    ax.annotate(row["name"], (row.geometry.x + dx, row.geometry.y + dy),
                fontsize=8.5, ha=ha, va="center",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray",
                          alpha=0.9, linewidth=0.5))

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["active_retail"],
           markersize=10, label="Active retail pharmacy (Board confirmed)"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor=colors["closed"],
           markersize=10, label="Closed location (former chain)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=colors["fqhc"],
           markersize=10, label="FQHC dispensary (institutional)"),
]
ax.legend(handles=legend_elements, loc="lower left", frameon=True, fontsize=9)

ax.set_title(
    "North Minneapolis pharmacy landscape, April 2026\n"
    "ZIP codes 55411 and 55412 — 18 census tracts, 16 classified as pharmacy deserts",
    fontsize=11, pad=10,
)
ax.set_axis_off()

# Bound a tighter view
minx, miny, maxx, maxy = zcta.total_bounds
buf = 0.005
ax.set_xlim(minx - buf, maxx + buf)
ax.set_ylim(miny - buf, maxy + buf)

plt.tight_layout()
plt.savefig(FIG_DIR / "figure7_north_mpls_focus.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"wrote {FIG_DIR / 'figure7_north_mpls_focus.png'}")
