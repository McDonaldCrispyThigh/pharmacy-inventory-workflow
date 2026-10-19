"""Extract data needed for thesis tables/figures from audit CSVs.

Outputs:
- data/figures_board/_19_misses.csv
- data/figures_board/_wayback_segments.json
- data/figures_board/_chain_indep.json
- data/figures_board/_fn_breakdown.json
- data/figures_board/_quartile_summary.json
"""
import json
import re
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUT = DATA / "figures_board"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load
tp = pd.read_csv(DATA / "audit2_tp_analysis_20260414_172017.csv", dtype={"zip5_ai": str})
fp = pd.read_csv(DATA / "audit2_fp_analysis_20260414_172017.csv", dtype={"zip5_ai": str})
fn = pd.read_csv(DATA / "audit3_fn_institutional_20260414_172017.csv", dtype={"zip5": str})

tp["is_tp"] = True
fp["is_tp"] = False
all_ai = pd.concat([tp, fp], ignore_index=True)
fn_retail = fn[fn["is_institutional"] == False].copy()

# ── Wayback × match rate
def wb_group(x):
    if x == -1: return "Chain sentinel"
    if x == 0:  return "Zero web presence"
    if 1 <= x <= 7: return "Low (1-7 yrs)"
    return "Established (>=20 yrs)"

all_ai["wb_grp"] = all_ai["Wayback_Snapshot_Count"].apply(wb_group)
order = ["Chain sentinel", "Zero web presence", "Low (1-7 yrs)", "Established (>=20 yrs)"]
wb_rows = []
for g in order:
    grp = all_ai[all_ai["wb_grp"] == g]
    n = len(grp); tps = int(grp["is_tp"].sum()); fps = n - tps
    rate = (tps / n * 100) if n else 0.0
    wb_rows.append({"group": g, "n": n, "tps": tps, "fps": fps, "match_rate_pct": round(rate, 1)})
(OUT / "_wayback_segments.json").write_text(json.dumps(wb_rows, indent=2))

# ── Chain vs independent
chain_kw = ["walgreens", "cvs", "walmart", "target", "costco", "hy-vee", "hyvee", "cub pharmacy"]
pat = "|".join(chain_kw)
all_ai["is_chain"] = all_ai["Company"].str.lower().str.contains(pat)
fn_retail["is_chain"] = fn_retail["facility_name"].str.lower().str.contains(pat)

tp_c = tp[tp["Company"].str.lower().str.contains(pat)]; tp_i = tp[~tp["Company"].str.lower().str.contains(pat)]
fp_c = fp[fp["Company"].str.lower().str.contains(pat)]; fp_i = fp[~fp["Company"].str.lower().str.contains(pat)]
fn_c = fn_retail[fn_retail["is_chain"]]; fn_i = fn_retail[~fn_retail["is_chain"]]
bd_c = len(tp_c) + len(fn_c); bd_i = len(tp_i) + len(fn_i)

ci_rows = [
    {"segment": "Chain",
     "ai": len(tp_c)+len(fp_c), "tps": len(tp_c), "fps": len(fp_c),
     "board_denom": bd_c, "fns": len(fn_c),
     "precision_pct": round(len(tp_c)/(len(tp_c)+len(fp_c))*100, 1),
     "recall_pct": round(len(tp_c)/bd_c*100, 1)},
    {"segment": "Independent",
     "ai": len(tp_i)+len(fp_i), "tps": len(tp_i), "fps": len(fp_i),
     "board_denom": bd_i, "fns": len(fn_i),
     "precision_pct": round(len(tp_i)/(len(tp_i)+len(fp_i))*100, 1),
     "recall_pct": round(len(tp_i)/bd_i*100, 1)},
]
(OUT / "_chain_indep.json").write_text(json.dumps(ci_rows, indent=2))

# ── FN sub-categorization (mirrors cross_analysis.py)
inst_slipthrough_kw = [
    "treatment center", "omnicare", "cardinal health", "ebenezer",
    "evexia", "purescripts", "genoa healthcare", "hcmc", "jubilant",
    "mobe", "nura", "open cities health", "option care", "our lady of peace",
    "pediatric home", "petnet", "pharmerica", "pillpack", "amazon pharmacy",
    "post acute", "prairiecare", "roundtablerx", "medication repository",
    "aliveness project", "thrive rx", "tria pharmacy", "united family practice",
    "riverland community", "west side community health"
]

def classify_fn_sub(name):
    n = name.lower()
    chain_legal = ["grand st. paul cvs", "walgreen co", "walmart inc.", "wal-mart",
                   "sam's west", "sam west", "supervalu", "hy-vee", "costco wholesale"]
    if any(p in n for p in chain_legal):
        return "DBA chain match failure"
    if any(p in n for p in inst_slipthrough_kw):
        return "Institutional slipthrough"
    return "Genuine independent miss"

fn_retail["sub_cat"] = fn_retail["facility_name"].apply(classify_fn_sub)
counts = fn_retail["sub_cat"].value_counts().to_dict()
breakdown = {
    "total_fn": int(len(fn)),
    "institutional": int((fn["is_institutional"] == True).sum()),
    "retail": int(len(fn_retail)),
    "dba_chain_failure": int(counts.get("DBA chain match failure", 0)),
    "institutional_slipthrough": int(counts.get("Institutional slipthrough", 0)),
    "genuine_independent_miss": int(counts.get("Genuine independent miss", 0)),
}
(OUT / "_fn_breakdown.json").write_text(json.dumps(breakdown, indent=2))

# ── 19 misses for Appendix
misses = fn_retail[fn_retail["sub_cat"] == "Genuine independent miss"].copy()
cols = [c for c in ["facility_name", "address", "city", "zip5"] if c in misses.columns]
misses_out = misses[cols].sort_values(["city", "facility_name"])
misses_out.to_csv(OUT / "_19_misses.csv", index=False)

# ── Quartile summary (already in spatial output)
qsum_path = DATA / "spatial_quartile_summary_20260414.csv"
if qsum_path.exists():
    q = pd.read_csv(qsum_path)
    q.to_csv(OUT / "_quartile_summary.csv", index=False)

print("--- Wayback ---")
for r in wb_rows: print(r)
print("--- Chain/Indep ---")
for r in ci_rows: print(r)
print("--- FN breakdown ---")
print(breakdown)
print(f"--- 19 misses written to {OUT/'_19_misses.csv'} ---")
print(misses_out.to_string(index=False))
