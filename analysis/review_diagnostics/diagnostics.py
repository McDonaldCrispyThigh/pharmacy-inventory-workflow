"""Read-only checks behind the 2026-10-19 committee-review response (no analysis rerun, no provider calls).

1. The 11 screened Board pharmacies that no raw Google candidate linked at or above 60 under the primary setting
   (1.2, 90, 0.90, 60): best candidate overall, most similar candidate in the same ZIP, and candidates sharing the
   house number, from the frozen pair components (analysis/rq2_oat/20261012_v3_factorial/core.py).
2. Historical-year outputs in the 836 supplemental final classifications (first attempts plus retries)."""
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
R = H.parents[2]
sys.path.insert(0, str(R / "analysis/rq2_oat/20261012_v3_factorial"))
from core import load, score  # noqa: E402

wf, bd, comp, inc = load()
S = score(comp, 1.2, 90.0, 0.90)
stage_a = np.flatnonzero(S.max(axis=0) < 60)
misses = []
for j in stage_a:
    same_zip = ~comp["zipdiff"][:, j]
    N = np.where(same_zip, comp["N"][:, j], -1)
    i = int(N.argmax())
    house = "match" if comp["house"][i, j] else ("missing" if comp["house_missing"][i, j] else "differ")
    misses.append({"license": str(bd.license_nbr.iloc[j]), "board_name": bd.facility_name.iloc[j],
                   "board_address": f"{bd.address_line1.iloc[j]}, {bd.zip5.iloc[j]}",
                   "best_score_any_candidate": round(float(S[:, j].max()), 2),
                   "most_similar_same_zip": wf.Company.iloc[i], "its_address": str(wf.Street_Address.iloc[i]),
                   "name_similarity": round(float(N[i]), 1), "house_number": house, "its_score": round(float(S[i, j]), 2),
                   "candidates_sharing_house_number": int(comp["house"][:, j].sum())})
# Manual reading of the table above (names and addresses): retrieved but failed at linkage vs. no candidate for the site.
RETRIEVED = {"Cub Pharmacy 246 vs 250 57th Ave NE (different house number)",
             "CVS #16714 at 900 Nicollet Mall (house number lost behind 'Target Plaza South')",
             "Evexia RX (dba) Purescripts (Board address line holds the company name; Google lists 5206 W 84th St)"}

sup = R / "data/supplemental_836/20260907_v1"


def find(o, key, depth=0):
    if depth > 6:
        return []
    out = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                out.append(v)
            out += find(v, key, depth + 1)
    elif isinstance(o, list):
        for v in o:
            out += find(v, key, depth + 1)
    return out


first = collections.Counter()
retried = []
for f in sorted(glob.glob(str(sup / "classification/private/classifications/*.json"))):
    o = json.load(open(f, encoding="utf-8"))
    v = (o.get("decision") or {}).get("historical_year_type")
    if v is None:
        retried.append(o["place_id"])
    else:
        first[v] += 1
retry = collections.Counter()
for f in sorted(glob.glob(str(sup / "classification_retry/**/*.json"), recursive=True)):
    try:
        o = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    d = o.get("decision") if isinstance(o, dict) else None
    if isinstance(d, dict) and "historical_year_type" in d:
        retry[(o.get("place_id"), d["historical_year_type"])] += 1
retry_types = collections.Counter(t for (pid, t) in retry if pid in set(retried))
report = {"stage_a_misses": misses, "retrieved_but_unlinked_reading": sorted(RETRIEVED),
          "supplemental_historical_year": {"first_attempt": dict(first), "retried_place_ids": len(retried),
                                           "retried_decisions_by_type": dict(retry_types)}}
(H / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"stage_a": len(misses), "supplemental": report["supplemental_historical_year"]}, indent=1))
