"""One-parameter profiles on fine grids (others at primary) and the interval where the rules hold."""
import numpy as np, pandas as pd
from core import *
wf, bd, comp, inc = load()
P = dict(w=0.8, h=60.0, z=0.85, T=75.0)
_, _, post0 = run(comp, inc, 0.8, 60.0, 0.85, 75.0); base = {(i, j) for i, j, s in post0}
grids = {'w': np.round(np.arange(0.40, 1.5001, 0.01), 2), 'h': np.arange(30.0, 90.001, 1.0),
         'z': np.round(np.arange(0.70, 1.0001, 0.01), 2), 'T': np.round(np.arange(60, 90.001, 0.1), 1)}
rows = []
for k, vals in grids.items():
    for v in vals:
        p = dict(P); p[k] = float(v)
        _, _, post = run(comp, inc, p['w'], p['h'], p['z'], p['T']); ps = {(i, j) for i, j, s in post}
        lock = p['T'] > (100 + p['w']*(100-p['h']))/(1+p['w']) + 1e-12
        zipok = 100*p['z'] >= p['T'] - 1e-12
        rows.append(dict(param=k, value=float(v), links=len(ps), added=len(ps-base), lost=len(base-ps), lock=lock, zip_ok=zipok,
                         name_floor=p['T']*(1+p['w'])-100*p['w']))
d = pd.DataFrame(rows); d.to_csv('profiles_fine.csv', index=False)
for k in grids:
    s = d[d.param == k]; ok = s[s.lock & s.zip_ok]
    same = s[(s.added == 0) & (s.lost == 0)]
    print(f"{k}: rules hold for {ok.value.min()}..{ok.value.max()} -> links {ok.links.min()}-{ok.links.max()}, max added {ok.added.max()}, max lost {ok.lost.max()}; identical pairs for {same.value.min()}..{same.value.max()} ({len(same)} grid values)")
    if k == 'w':
        sub = ok[ok.value < 1.0]; print('   w in rule interval and < 1:', sub.value.min(), sub.value.max(), 'links', sub.links.min(), sub.links.max(), 'added', sub.added.max(), 'lost', sub.lost.max())
    if k == 'h':
        sub = ok[ok.value <= 90]; print('   h links by value:', dict(zip(sub.value.astype(int), sub.links)))
