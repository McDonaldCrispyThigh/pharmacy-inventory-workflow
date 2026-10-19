"""Extra appendix quantities: excluded-only links, calibration rows, factorial summaries and variance shares."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]  # repository root (analysis/rq2_oat/20261012_v3_factorial/..)
from core import load, score, assign_hungarian
sys.path.insert(0, str(REPO / 'analysis/expanded_final/20260908_v1/implementation/code'))
from study_design import normalized_location_key
wf, bd, comp, inc = load()
inc_idx = np.flatnonzero(inc); exc_idx = np.flatnonzero(~inc)
key = np.array([normalized_location_key(a, z) for a, z in zip(wf.Street_Address, wf.Zip_Code)])
S = score(comp, 1.2, 90.0, 0.90)
ex = assign_hungarian(S[exc_idx], 60.0)
inc_keys = set(key[inc_idx])
names = [wf.Company.iloc[int(exc_idx[i])] for i, j, s in ex]
print('excluded-only links', len(ex), 'at an included record address', sum(key[int(exc_idx[i])] in inc_keys for i, j, s in ex))
print(pd.Series(names).value_counts().head(12))
c = pd.read_csv('calibration_runs.csv')
def row(w, h, z, t):
    r = c[(c.w_A == w) & (c.h == h) & (c.z == z) & (c.tau == t)].iloc[0]
    return {k: (round(float(r[k]), 2) if k == 'margin' else int(r[k])) for k in ['links', 'wrong_main', 'missed_main', 'err_main', 'margin', 'err_as_same', 'err_as_diff']}
cal = {'selected': row(1.2, 90, 0.90, 60), 'previous': row(0.8, 60, 0.85, 75), 'as_same_best': row(1.2, 70, 0.85, 60), 'unamended': row(1.5, 60, 0.85, 60)}
print(json.dumps(cal, indent=0))
f = pd.read_csv('factorial_runs.csv')
f['gain'] = f.precision - f.pre_precision; f['rloss'] = f.pre_recall - f.recall
f['wgap'] = f.wf_rep_share - f.wf_single_share; f['bgap'] = f.bd_rep_cov - f.bd_single_cov
def rng(s, d=2): return f'{s.min():.{d}f}–{s.max():.{d}f}'
by = []
for t, g in f.groupby('tau'):
    by.append(dict(tau=int(t), n=len(g), links=f'{g.post_links.min()}–{g.post_links.max()}', precision=rng(g.precision), recall=rng(g.recall), pre_precision=rng(g.pre_precision)))
g = f; by.append(dict(tau='all', n=len(g), links=f'{g.post_links.min()}–{g.post_links.max()}', precision=rng(g.precision), recall=rng(g.recall), pre_precision=rng(g.pre_precision)))
print(pd.DataFrame(by).to_string())
print('gain', rng(f.gain), 'rloss', rng(f.rloss), 'wgap', rng(f.wgap), 'bgap', rng(f.bgap), 'min gaps>0', (f.wgap > 0).all(), (f.bgap > 0).all())
# variance decomposition of post-filter links (balanced full factorial)
import itertools
y = f.post_links.astype(float); mu = y.mean(); tot = ((y - mu) ** 2).sum()
fac = ['w_A', 'h', 'z', 'tau']; ss = {}
def eff(cols):
    # inclusion-exclusion on group means
    m = f.groupby(list(cols)).post_links.transform('mean')
    val = m.copy()
    for k in range(1, len(cols)):
        for sub in itertools.combinations(cols, k):
            sgn = (-1) ** (len(cols) - k)
            val = val + sgn * f.groupby(list(sub)).post_links.transform('mean')
    val = val + ((-1) ** len(cols)) * mu
    return float((val ** 2).sum())
for k in range(1, 5):
    for cols in itertools.combinations(fac, k):
        ss['×'.join(cols)] = 100 * eff(cols) / tot
s = pd.Series(ss).sort_values(ascending=False)
print(s.round(2).head(12)); print('sum', s.sum().round(3))
grid = f[(f.h == 90) & (f.z == 0.9)].pivot(index='w_A', columns='tau', values='post_links')
json.dump(dict(calibration=cal, by_threshold=by, ranges=dict(gain=rng(f.gain), rloss=rng(f.rloss), wgap=rng(f.wgap), bgap=rng(f.bgap)),
               variance_shares=s.round(2).to_dict(), grid_h90_z090=grid.to_dict()), open('appendix_numbers2.json', 'w'), indent=1, ensure_ascii=False, default=str)
