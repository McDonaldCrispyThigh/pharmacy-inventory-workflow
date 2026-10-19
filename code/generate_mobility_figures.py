"""Generate mobility-weighted analysis figures.

- figure9_mobility_desert.png  : Qato desert rate vs MWDR by quartile
- figure10_threshold_sensitivity.png : 0.5 / 1.0 / 2.0 mi by quartile
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG = Path("docs/thesis/figures")
DATA = Path("data/figures_board")

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# ════════════════════════════════════════════════════════════════════════════
# Figure 9: Qato vs MWDR by income quartile
# ════════════════════════════════════════════════════════════════════════════
mwdr = json.loads((DATA / "_mobility_desert_quartile.json").read_text())

fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = np.arange(len(mwdr))
width = 0.35
qato = [r["qato_desert_rate_pct"] for r in mwdr]
mwdr_v = [r["mwdr_pct"] for r in mwdr]
labels = [f"Q{r['Qinc']}\n({r['median_carless_pct']}% carless)" for r in mwdr]

b1 = ax.bar(x - width/2, qato, width, label="Qato 0.5-mi binary (UTM)",
            color="#C0504D", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + width/2, mwdr_v, width, label="Mobility-weighted (MWDR)",
            color="#2E5C8A", edgecolor="black", linewidth=0.5)

for bar, val in list(zip(b1, qato)) + list(zip(b2, mwdr_v)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.set_ylabel("Desert rate (%)")
ax.set_xlabel("Income quartile (median carless household share)")
ax.set_title(
    "Mobility-weighted vs Qato desert rate by income quartile\n"
    "(MWDR weights walking inaccessibility for carless households,\n"
    "driving inaccessibility >2 mi for car-owning households)",
    fontsize=11, pad=10,
)
ax.legend(loc="upper left", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(FIG / "figure9_mobility_desert.png", bbox_inches="tight")
plt.close()
print(f"wrote {FIG/'figure9_mobility_desert.png'}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 10: threshold sensitivity (0.5 / 1.0 / 2.0 mi)
# ════════════════════════════════════════════════════════════════════════════
ts = json.loads((DATA / "_threshold_sensitivity.json").read_text())

fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = np.arange(len(ts))
width = 0.27
v05 = [r["desert_pct_0.5 mi"] for r in ts]
v10 = [r["desert_pct_1.0 mi"] for r in ts]
v20 = [r["desert_pct_2.0 mi"] for r in ts]

c1 = "#C0504D"; c2 = "#E5A04C"; c3 = "#4F8C51"
ax.bar(x - width, v05, width, label="0.5 mi (Qato)",  color=c1, edgecolor="black", linewidth=0.4)
ax.bar(x,         v10, width, label="1.0 mi",         color=c2, edgecolor="black", linewidth=0.4)
ax.bar(x + width, v20, width, label="2.0 mi (drive)", color=c3, edgecolor="black", linewidth=0.4)

for xi, (a, b, c) in enumerate(zip(v05, v10, v20)):
    ax.text(xi - width, a + 1.5, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(xi,         b + 1.5, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(xi + width, c + 1.5, f"{c:.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x); ax.set_xticklabels([f"Q{r['Qinc']}" for r in ts])
ax.set_ylim(0, 100)
ax.set_ylabel("Tracts classified as desert (%)")
ax.set_xlabel("Income quartile")
ax.set_title("Threshold sensitivity: desert rate by distance threshold and quartile",
             fontsize=11)
ax.legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(FIG / "figure10_threshold_sensitivity.png", bbox_inches="tight")
plt.close()
print(f"wrote {FIG/'figure10_threshold_sensitivity.png'}")
