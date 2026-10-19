import numpy as np, pandas as pd
from core import *
wf, bd, comp, inc = load()
_, _, post0 = run(comp, inc, 0.8, 60.0, 0.85, 75.0); base = {(i, j) for i, j, s in post0}
rows = []
for T in np.round(np.arange(60, 90.001, 0.1), 1):
    _, pre, post = run(comp, inc, 0.8, 60.0, 0.85, float(T)); ps = {(i, j) for i, j, s in post}
    rows.append(dict(tau=T, links=len(post), added=len(ps-base), lost=len(base-ps), house_nm=sum(not comp['house'][i, j] for i, j in ps)))
p = pd.DataFrame(rows); p.to_csv('profile_tau.csv', index=False)
same = p[(p.added == 0) & (p.lost == 0)]
print('tau values reproducing the primary link set exactly:', same.tau.min(), '-', same.tau.max())
for lo, hi in [(73.4, 80), (70, 80), (65, 85)]:
    s = p[(p.tau >= lo) & (p.tau <= hi)]; print(lo, hi, s.links.min(), s.links.max(), 'max added', s.added.max(), 'max lost', s.lost.max())
print(p[(p.tau>=73)&(p.tau<=76)].to_string(index=False))
