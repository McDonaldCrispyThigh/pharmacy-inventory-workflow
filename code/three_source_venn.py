"""Three-source coverage Venn for AI vs Board vs NPPES.

Build sets of canonical names for each source restricted to MSA ZIPs:
- AI:    321 TPs (Board-confirmed retail) — see audit2_tp / cross_analysis
- Board: 459 records (raw) — but for Venn, use 405 retail-only
- NPPES: 717 FN + 155 TP = 872 raw; for retail Venn use 252 "possible missed retail"
         + 155 TP records that match a retail Board record

For the Venn we want to know:
  - How many retail pharmacies are visible to ALL three sources (most digitally salient)
  - How many ONLY to NPPES (administrative-only / closed / corporate)
  - How many ONLY to Board (regulatory ground truth, no consumer/admin trail)
  - How many to Board+NPPES but NOT AI -- these are the "digitally invisible" ones
    that the visibility-bias argument predicts

We approximate using the existing audit results.
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUT = DATA / "figures_board"

# ── Load Board retail-only universe ──
fn = pd.read_csv(DATA / "audit3_fn_institutional_20260414_172017.csv")
fn_retail = fn[fn["is_institutional"] == False].copy()  # 84 retail FNs

tp = pd.read_csv(DATA / "audit2_tp_analysis_20260414_172017.csv")  # 321 TPs

# Board retail = AI-matched (321 TPs) + retail FN (84) = 405
board_retail_count = 321 + len(fn_retail)
print(f"Board retail denom: {board_retail_count}")

# ── NPPES coverage data ──
# We have 252 "possible missed retail" NPPES records.
# These represent NPPES coverage of retail-likely records that AI missed.
nppes_fn = pd.read_csv(DATA / "spatial_nppes_fn_tracts.csv")
nppes_possible_retail = nppes_fn[nppes_fn["FN_Category"] == "possible_missed_retail"]
n_nppes_possible_retail = len(nppes_possible_retail)
print(f"NPPES possible missed retail: {n_nppes_possible_retail}")

# AI ∩ Board: 321 records (TPs against retail Board). Measured.
n_ai_board = 321

# ── Board ∩ NPPES, among the 84 AI-missed retail records ────────────────────
#
# This region IS computable and is now computed, not assumed. The previous
# version of this file asserted the split from a comment that read "assume that
# ~half of NPPES possible-retail also appear in Board", which is not a
# measurement. The manuscript meanwhile describes this quantity as "measured
# directly by fuzzy-matching", so it must actually be measured here.
#
# Method mirrors the project's own matcher: RapidFuzz token_sort_ratio at the
# same 0.75 threshold used in validate_board.py, blocked on 5-digit ZIP, over
# the full NPPES false-negative set.
from rapidfuzz.fuzz import token_sort_ratio

_STRIP = re.compile(r'\b(inc|llc|l\.l\.c|corp|co|company|the|of|dba)\b')


def _norm_name(s: str) -> str:
    s = _STRIP.sub('', str(s).lower())
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', s).split())


_npp_all = pd.read_csv(DATA / "nppes_fn_raw.csv")
_npp_all["_k"] = _npp_all["Name"].map(_norm_name)
_npp_all["_z"] = _npp_all["ZIP"].astype(str).str[:5]
fn_retail["_k"] = fn_retail["facility_name"].map(_norm_name)
fn_retail["_z"] = fn_retail["zip"].astype(str).str[:5]

_MATCH_T = 75
_in_nppes = 0
for _, _r in fn_retail.iterrows():
    _cand = _npp_all[_npp_all["_z"] == _r["_z"]]
    if _cand.empty:
        _cand = _npp_all
    if any(token_sort_ratio(_r["_k"], _k) >= _MATCH_T for _k in _cand["_k"]):
        _in_nppes += 1
n_board_nppes = _in_nppes
n_board_only = len(fn_retail) - _in_nppes
print(f"MEASURED  Board retail FN = {len(fn_retail)}: "
      f"{n_board_nppes} also in NPPES, {n_board_only} in neither "
      f"(token_sort_ratio >= {_MATCH_T}, ZIP-blocked)")

# ── Regions that remain unmeasured ──────────────────────────────────────────
#
# The all-three and AI-plus-Board split requires a record-level join of AI
# against NPPES, which has not been run. It is left as None rather than filled
# with a plausible-looking number: a figure that shows an invented count is
# indistinguishable, to a reader, from one that shows a measured count.
venn = {
    "board_nppes_only": n_board_nppes,   # MEASURED
    "board_only":       n_board_only,    # MEASURED
    "ai_board_total":   n_ai_board,      # MEASURED (AI TPs against retail Board)
    "ai_only":          78,              # MEASURED (AI FPs)
    "nppes_only":       465,             # MEASURED (331 corp + 95 closed + 39 specialty)
    "all_three":        None,            # NOT MEASURED - needs AI x NPPES join
    "ai_board_not_nppes": None,          # NOT MEASURED - derived from all_three
    "_provenance": {
        "measured_with": f"rapidfuzz token_sort_ratio >= {_MATCH_T}, ZIP-blocked",
        "board_retail_fn_n": int(len(fn_retail)),
        "nppes_fn_pool_n": int(len(_npp_all)),
        "note": ("all_three and ai_board_not_nppes require an AI-vs-NPPES "
                 "record-level join that has not been performed. Do not plot a "
                 "triple intersection until it has."),
    },
}

(OUT / "_three_source_venn.json").write_text(json.dumps(venn, indent=2))
print(json.dumps(venn, indent=2))

# Generate figure (matplotlib_venn)
import matplotlib.pyplot as plt
FIG = Path("docs/thesis/figures")

# ── Figure ──────────────────────────────────────────────────────────────────
#
# A Venn diagram is deliberately NOT drawn. Two of its seven regions
# (all_three, ai_board_not_nppes) require an AI-vs-NPPES record-level join that
# has not been run, and a Venn with invented regions reads to a viewer exactly
# like one with measured regions. The previous version filled them from a
# comment that began "assume that ~half", and the manuscript then cited the
# result as measured.
#
# What is plotted instead is only what was measured: the fate of the 84 Board
# retail false negatives, which is the quantity the discussion actually argues
# from.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.2))
cats = ["Also in NPPES\n(administratively visible,\nGoogle-invisible)",
        "In neither NPPES nor AI\n(visible only to the\nlicensure registry)"]
vals = [venn["board_nppes_only"], venn["board_only"]]
bars = ax.barh(cats, vals, color=["#C0504D", "#7F4F1F"],
               edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{v}  ({v / sum(vals) * 100:.0f}%)", va="center", fontsize=10)
ax.set_xlabel("Board-licensed retail pharmacies missed by the AI pipeline")
ax.set_xlim(0, max(vals) * 1.25)
ax.set_title(
    f"Where the {sum(vals)} missed Board retail pharmacies show up\n"
    f"Measured by RapidFuzz token_sort_ratio >= {_MATCH_T}, ZIP-blocked, "
    f"against {len(_npp_all)} NPPES records",
    fontsize=10, pad=10)
plt.tight_layout()
out = FIG / "figure11_board_fn_disposition.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
plt.close()
print(f"wrote {out}")

_old = FIG / "figure11_three_source_venn.png"
if _old.exists():
    _old.unlink()
    print(f"removed superseded {_old} (contained unmeasured regions)")
