import itertools, time, json, numpy as np, pandas as pd
from core import *
wf, bd, comp, inc = load()
G = {'w_A':[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5], 'h':[30.0,40.0,50.0,60.0,70.0,80.0,90.0],
     'z':[0.7,0.75,0.8,0.85,0.9,0.95,1.0], 'tau':[60.0,65.0,70.0,75.0,80.0,85.0,90.0]}
cat = {'repeated-name component':'rep','single-occurrence name component':'single'}
wrec = wf.recurrence.map(cat).to_numpy(); brec = bd.recurrence.map(cat).to_numpy()
inc_idx = np.flatnonzero(inc)
_, pre0, post0 = run(comp, inc, 0.8, 60.0, 0.85, 75.0)
base = {(i, j) for i, j, s in post0}
rows = []; pairs_long = []
t = time.time()
for k, (w, h, z, tau) in enumerate(itertools.product(G['w_A'], G['h'], G['z'], G['tau'])):
    S, pre, post = run(comp, inc, w, h, z, tau)
    ps = {(i, j) for i, j, s in post}
    hd = sum(not comp['house'][i, j] for i, j in ps)
    zd = sum(bool(comp['zipdiff'][i, j]) for i, j in ps)
    lw = {i for i, j in ps}; lb = {j for i, j in ps}
    r = dict(w_A=w, h=h, z=z, tau=tau, pre_links=len(pre), post_links=len(post),
             precision=100*len(post)/len(inc_idx), recall=100*len(post)/len(bd), pre_precision=100*len(pre)/len(wf),
             pre_recall=100*len(pre)/len(bd), house_not_match_links=hd, zip_mismatch_links=zd,
             base_kept=len(ps & base), added=len(ps - base), lost=len(base - ps))
    for lab in ('rep', 'single'):
        wi_ = [i for i in inc_idx if wrec[i] == lab]; bj_ = [j for j in range(len(bd)) if brec[j] == lab]
        r[f'wf_{lab}_share'] = 100*sum(i in lw for i in wi_)/len(wi_)
        r[f'bd_{lab}_cov'] = 100*sum(j in lb for j in bj_)/len(bj_)
    rows.append(r)
    for i, j in ps ^ base:
        pairs_long.append(dict(w_A=w, h=h, z=z, tau=tau, i=i, j=j, kind='added' if (i, j) in ps else 'lost'))
print('settings', len(rows), 'secs', round(time.time()-t, 1))
df = pd.DataFrame(rows)
# derived design quantities
w, h, z, T = df.w_A, df.h, df.z, df.tau
df['max_nonmatch_house'] = (100 + w*(100-h))/(1+w)          # best score a pair with differing/missing house number can reach (same ZIP)
df['lock_margin'] = T - df.max_nonmatch_house                 # >0 : house-number lock holds
df['name_floor'] = T*(1+w) - 100*w                            # name similarity needed at an identical address, same ZIP
df['zip_ok'] = 100*z >= T - 1e-9                              # identical record with different ZIP can still link
df['name_floor_zip'] = T*(1+w)/z - 100*w                      # same, when ZIPs differ
df['lock'] = df.lock_margin > 0
df.to_csv('factorial_runs.csv', index=False)
pd.DataFrame(pairs_long).to_csv('factorial_pair_changes.csv', index=False)
