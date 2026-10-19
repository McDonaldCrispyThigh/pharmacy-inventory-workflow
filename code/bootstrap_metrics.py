"""Bootstrap 95% CI for precision / recall / F1 against the Board of Pharmacy
adjusted retail-only denominator.

Method
------
Sample with replacement from:
  - 321 TPs (AI matched to Board)
  - 78 FPs (AI not matched)
  - 84 retail FNs (Board not matched by AI)
Compute precision, recall, F1 on each resampled population.
Repeat 5,000 times. Report point estimate + percentile CI.
"""
import json
import random
from pathlib import Path

OUT = Path("data/figures_board")

TP, FP, FN = 321, 78, 84
N_BOOT = 5000
random.seed(20260501)

# Universe of items: TP labels, FP labels, FN labels
items = (["tp"] * TP) + (["fp"] * FP) + (["fn"] * FN)
n = len(items)

precs, recs, f1s = [], [], []
for _ in range(N_BOOT):
    sample = [items[random.randrange(n)] for _ in range(n)]
    tp_s = sample.count("tp")
    fp_s = sample.count("fp")
    fn_s = sample.count("fn")
    p = tp_s / (tp_s + fp_s) if (tp_s + fp_s) else 0
    r = tp_s / (tp_s + fn_s) if (tp_s + fn_s) else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    precs.append(p); recs.append(r); f1s.append(f1)

def ci(arr, q1=0.025, q2=0.975):
    a = sorted(arr)
    return a[int(q1 * len(a))], a[int(q2 * len(a))]

# Point estimates
p_pt = TP / (TP + FP) * 100
r_pt = TP / (TP + FN) * 100
f1_pt = 2 * (p_pt/100) * (r_pt/100) / ((p_pt/100) + (r_pt/100)) * 100

p_lo, p_hi = ci(precs);   p_lo *= 100; p_hi *= 100
r_lo, r_hi = ci(recs);    r_lo *= 100; r_hi *= 100
f1_lo, f1_hi = ci(f1s);   f1_lo *= 100; f1_hi *= 100

result = {
    "n_boot": N_BOOT,
    "point": {"precision_pct": round(p_pt, 2),
              "recall_pct":    round(r_pt, 2),
              "f1_pct":        round(f1_pt, 2)},
    "ci_95": {
        "precision_pct": [round(p_lo, 2), round(p_hi, 2)],
        "recall_pct":    [round(r_lo, 2), round(r_hi, 2)],
        "f1_pct":        [round(f1_lo, 2), round(f1_hi, 2)],
    },
}
print(json.dumps(result, indent=2))
(OUT / "_bootstrap_ci.json").write_text(json.dumps(result, indent=2))
