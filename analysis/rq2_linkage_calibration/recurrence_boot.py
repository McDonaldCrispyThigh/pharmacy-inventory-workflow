"""Name-recurrence shares with the frozen component bootstrap (5,000 draws, seed 20260822 + row offset)."""
import numpy as np, pandas as pd, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]  # repository root (analysis/rq2_oat/20261012_v3_factorial/..)
from core import *
FZ = str(REPO / 'analysis/expanded_final/20260908_v1')
RANDOM_SEED = 20260822
def interval(group, seed, draws=5000):
    cluster = group.groupby('name_component_id_sha256')['linked'].agg(['sum', 'count'])
    if len(cluster) < 2: return float('nan'), float('nan')
    s = cluster['sum'].to_numpy(float); t = cluster['count'].to_numpy(float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(cluster), size=(draws, len(cluster)))
    est = s[idx].sum(1) / t[idx].sum(1); lo, hi = np.quantile(est, [0.025, 0.975]); return float(lo), float(hi)
detail = pd.read_csv(f'{FZ}/rq2/name_recurrence_detail.csv', dtype=str, keep_default_na=False)
detail = detail[detail.linkage_strategy.eq('name_address_hungarian_t75')].copy()
def summarize(det):
    rows = []; off = 0
    for thr in ['2', '3', '5']:
        for source in ['Board keyword-screen reference', 'Workflow inventory locations']:
            sub = det[(det.recurrence_threshold == thr) & (det.source == source)].copy()
            sub['linked'] = sub['linked'].astype(str).eq('True').astype(int)
            below = 'single-occurrence name component' if thr == '2' else 'below-threshold name component'
            for code, gname in [('repeated', 'repeated-name component'), ('below_threshold', below), ('unresolved', 'unresolved')]:
                g = sub[sub.name_component_group == gname]
                lo, hi = interval(g, RANDOM_SEED + off); off += 1
                rows.append(dict(source=source, recurrence_threshold=int(thr), group_code=code, group=gname, records=len(g),
                                 distinct_name_components=g.name_component_id_sha256.nunique(), linked_records=int(g.linked.sum()),
                                 linked_pct=round(100*g.linked.sum()/len(g), 2) if len(g) else None,
                                 cluster_bootstrap_95_low_pct=round(100*lo, 2) if np.isfinite(lo) else None,
                                 cluster_bootstrap_95_high_pct=round(100*hi, 2) if np.isfinite(hi) else None))
    return pd.DataFrame(rows)
old = summarize(detail)
ref = pd.read_csv(f'{FZ}/approved_figures/name_recurrence.csv')
m = old.merge(ref, on=['source','recurrence_threshold','group_code'], suffixes=('', '_ref'))
cols = ['linked_pct','cluster_bootstrap_95_low_pct','cluster_bootstrap_95_high_pct','records','linked_records']
ok = all(((m[c].isna() & m[c+'_ref'].isna()) | (m[c] == m[c+'_ref'])).all() for c in cols)
print('reproduces frozen recurrence table exactly:', ok)
if not ok: print(m[['source','recurrence_threshold','group_code'] + [x for c in cols for x in (c, c+'_ref')]].to_string())
# new linked flags
wf, bd, comp, inc = load()
_, _, post = run(comp, inc, 1.2, 90.0, 0.90, 60.0)
inc_index = wf.included_index.to_numpy()
new_wf = {str(int(inc_index[i])) for i, j, s in post}; new_bd = {str(bd.board_id[j]) for i, j, s in post}
nd = detail.copy()
isw = nd.source.eq('Workflow inventory locations')
nd['linked'] = np.where(isw, nd.source_record_id.isin(new_wf), nd.source_record_id.isin(new_bd)).astype(str)
assert nd[isw & nd.recurrence_threshold.eq('2')].linked.eq('True').sum() == 161
new = summarize(nd)
new.to_csv('name_recurrence_new.csv', index=False)
print(new[new.recurrence_threshold == 2][['source','group_code','records','linked_records','linked_pct','cluster_bootstrap_95_low_pct','cluster_bootstrap_95_high_pct']].to_string(index=False))
gaps = {}
for thr in [2, 3, 5]:
    for s in ['Board keyword-screen reference', 'Workflow inventory locations']:
        x = new[(new.recurrence_threshold == thr) & (new.source == s)].set_index('group_code')
        gaps[f'{s}|{thr}'] = round(x.loc['repeated','linked_pct'] - x.loc['below_threshold','linked_pct'], 2)
print(gaps)
