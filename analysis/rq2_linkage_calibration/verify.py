import time, numpy as np, pandas as pd
from core import *
wf, bd, comp, inc = load()
print(len(wf), inc.sum(), len(bd))
S0 = score(comp, 0.8, 60.0, 0.85)
print('baseline max abs diff vs frozen score:', np.abs(S0 - comp['score_baseline']).max())
t = time.time(); S, pre, post = run(comp, inc, 0.8, 60.0, 0.85, 75.0); print('one setting secs', round(time.time()-t, 2), len(pre), len(post))
# compare every one of the 75 frozen settings: counts and pair identities
us = pd.read_csv(SRC/'oat_out/unique_settings.csv', float_precision='round_trip')
ap = pd.read_csv(SRC/'oat_out/assignment_pairs.csv', dtype={'workflow_id':str,'board_id':str})
wi = {k:i for i,k in enumerate(wf.workflow_id)}; bi = {k:j for j,k in enumerate(bd.board_id)}
bad = 0; t = time.time()
for _, r in us.iterrows():
    S, pre, post = run(comp, inc, r.w_A, r.h, r.z, r.tau)
    sub = ap[ap.setting_id == r.setting_id]
    fz = {st: {(wi[a], bi[b]) for a, b in zip(g.workflow_id, g.board_id)} for st, g in sub.groupby('stage')}
    mine = {'pre': {(i, j) for i, j, s in pre}, 'post': {(i, j) for i, j, s in post}}
    ok = mine['pre'] == fz.get('pre', set()) and mine['post'] == fz.get('post', set()) and len(pre) == r.pre_links and len(post) == r.post_links
    bad += (not ok)
print('settings', len(us), 'mismatches', bad, 'secs', round(time.time()-t, 1))
