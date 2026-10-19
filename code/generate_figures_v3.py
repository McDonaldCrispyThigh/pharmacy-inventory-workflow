"""Generate three new figures for thesis Chapters 4-5:

- figure4_wayback_match_rate.png
- figure5_chain_indep.png
- figure6_fn_hierarchy.png

Outputs go to docs/thesis/figures/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG_DIR = Path("docs/thesis/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT = Path("data/figures_board")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# ════════════════════════════════════════════════════════════════════════════
# Figure 4: Wayback × Board match rate
# ════════════════════════════════════════════════════════════════════════════
wb = json.loads((DATA_OUT / "_wayback_segments.json").read_text())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
labels = [r["group"].replace(">=", "≥") for r in wb]
rates = [r["match_rate_pct"] for r in wb]
ns = [r["n"] for r in wb]
colors = ["#2E5C8A", "#C0504D", "#9BBB59", "#8064A2"]

bars = ax.bar(labels, rates, color=colors, edgecolor="black", linewidth=0.5)
for bar, rate, n in zip(bars, rates, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, rate + 1.5,
            f"{rate:.1f}%\n(n={n})",
            ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, 110)
ax.set_ylabel("Board of Pharmacy match rate (%)")
ax.set_xlabel("Wayback Machine CDX longevity group")
ax.set_title("Match rate by web-presence longevity (n = 399 AI-collected records)",
             fontsize=11)
ax.axhline(y=80.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(3.4, 81.5, "Overall precision: 80.5%", fontsize=8, color="gray", ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "figure4_wayback_match_rate.png", bbox_inches="tight")
plt.close()
print(f"wrote {FIG_DIR / 'figure4_wayback_match_rate.png'}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 5: Chain vs Independent precision and recall
# ════════════════════════════════════════════════════════════════════════════
ci = json.loads((DATA_OUT / "_chain_indep.json").read_text())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
segments = [r["segment"] for r in ci]
precision = [r["precision_pct"] for r in ci]
recall = [r["recall_pct"] for r in ci]

x = np.arange(len(segments))
width = 0.35
b1 = ax.bar(x - width/2, precision, width, label="Precision", color="#2E5C8A", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + width/2, recall,    width, label="Recall",    color="#C0504D", edgecolor="black", linewidth=0.5)

for bar, val in list(zip(b1, precision)) + list(zip(b2, recall)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10)

ax.set_xticks(x); ax.set_xticklabels(segments)
ax.set_ylim(0, 110)
ax.set_ylabel("Performance (%)")
ax.set_title("Precision and recall by segment\n"
             "(Chain n=247, Independent n=152 AI records)", fontsize=11)
ax.legend(loc="upper right", frameon=False)

# Annotation for the gap
gap_p = precision[0] - precision[1]
gap_r = recall[0]    - recall[1]
ax.text(0.5, 35,
        f"Precision gap: {gap_p:.1f}pp\nRecall gap: {gap_r:.1f}pp",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "figure5_chain_indep.png", bbox_inches="tight")
plt.close()
print(f"wrote {FIG_DIR / 'figure5_chain_indep.png'}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 6: FN hierarchical breakdown (Sankey-like horizontal flow)
# ════════════════════════════════════════════════════════════════════════════
fn = json.loads((DATA_OUT / "_fn_breakdown.json").read_text())

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.axis("off")

# Three columns: total → top-level (institutional vs retail) → sub-categories
def box(x, y, w, h, label, color, fontsize=10, weight="normal"):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02",
                                    linewidth=0.8, edgecolor="black",
                                    facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fontsize, weight=weight)

# Column 1: total 134
box(0.02, 0.40, 0.18, 0.20, f"All apparent FNs\nn = {fn['total_fn']}", "#D9D9D9", 11, "bold")

# Column 2: institutional 50, retail 84
box(0.32, 0.65, 0.20, 0.20, f"Institutional\n(automated)\nn = {fn['institutional']}",
    "#A6CEE3", 10)
box(0.32, 0.15, 0.20, 0.20, f"Retail-classified\nn = {fn['retail']}",
    "#FDBF6F", 10)

# Column 3: 3 sub-categories of retail
box(0.66, 0.71, 0.30, 0.13,
    f"DBA chain match failure\nn = {fn['dba_chain_failure']} ({fn['dba_chain_failure']/fn['retail']*100:.1f}%)",
    "#FFD9A6", 9)
box(0.66, 0.43, 0.30, 0.13,
    f"Institutional slipthrough\n(retained, flagged)\nn = {fn['institutional_slipthrough']} ({fn['institutional_slipthrough']/fn['retail']*100:.1f}%)",
    "#FFD9A6", 9)
box(0.66, 0.15, 0.30, 0.13,
    f"Genuine independent miss\nn = {fn['genuine_independent_miss']} ({fn['genuine_independent_miss']/fn['retail']*100:.1f}%)",
    "#FFD9A6", 9)

# Arrows
ax.annotate("", xy=(0.32, 0.75), xytext=(0.20, 0.55),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
ax.annotate("", xy=(0.32, 0.25), xytext=(0.20, 0.45),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
ax.annotate("", xy=(0.66, 0.77), xytext=(0.52, 0.30),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="gray", alpha=0.6))
ax.annotate("", xy=(0.66, 0.49), xytext=(0.52, 0.27),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="gray", alpha=0.6))
ax.annotate("", xy=(0.66, 0.21), xytext=(0.52, 0.22),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="gray", alpha=0.6))

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Multi-stage validity audit decomposition of 134 Board of Pharmacy false negatives",
             fontsize=11, pad=15)
plt.tight_layout()
plt.savefig(FIG_DIR / "figure6_fn_hierarchy.png", bbox_inches="tight")
plt.close()
print(f"wrote {FIG_DIR / 'figure6_fn_hierarchy.png'}")
