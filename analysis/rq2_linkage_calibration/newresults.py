"""RQ2 quantities quoted in the thesis text, for the primary and the calibrated setting."""
import numpy as np, pandas as pd, json
from core import *
wf, bd, comp, inc = load()
inc_idx = np.flatnonzero(inc)
print('included', len(inc_idx), 'distinct location keys among included:', wf.location_key_sha256[inc].nunique())
res = {}
for name, (w, h, z, T) in {'primary': (0.8, 60.0, 0.85, 75.0), 'new': (1.2, 90.0, 0.90, 60.0)}.items():
    S, pre, post = run(comp, inc, w, h, z, T)
    pre_b = {j for i, j, s in pre}; post_b = {j for i, j, s in post}; post_w = {i for i, j, s in post}
    r = dict(pre_links=len(pre), post_links=len(post), pre_prec=100*len(pre)/len(wf), post_prec=100*len(post)/len(inc_idx),
             pre_rec=100*len(pre)/len(bd), post_rec=100*len(post)/len(bd),
             retained=len(pre_b & post_b), retention=100*len(pre_b & post_b)/len(pre_b),
             removed=int((~inc).sum()), removed_pct=100*(~inc).sum()/len(wf))
    # unmatched Board decomposition
    E = S >= T
    un = [j for j in range(len(bd)) if j not in post_b]
    no_raw = [j for j in un if not E[:, j].any()]
    only_excl = [j for j in un if E[:, j].any() and not E[inc_idx, j].any()]
    inc_unassigned = [j for j in un if E[inc_idx, j].any()]
    r.update(unmatched_board=len(un), no_raw_edge=len(no_raw), only_excluded_edges=len(only_excl), included_edge_unassigned=len(inc_unassigned))
    # co-located unmatched workflow records
    key = wf.location_key_sha256.to_numpy()
    matched_keys = {key[i] for i in post_w}
    un_w = [i for i in inc_idx if i not in post_w]
    r['unmatched_wf'] = len(un_w); r['unmatched_wf_colocated'] = sum(key[i] in matched_keys for i in un_w)
    # grouped units: unit score = max over members
    units = pd.Series(inc_idx).groupby(key[inc_idx]).apply(list)
    U = np.vstack([S[m].max(axis=0) for m in units])
    up = assign_hungarian(U, T)
    r.update(units=len(units), unit_links=len(up), unit_prec=100*len(up)/len(units), unit_rec=100*len(up)/len(bd))
    # recurrence
    cat = {'repeated-name component':'rep','single-occurrence name component':'single'}
    wr = wf.recurrence.map(cat).to_numpy(); br = bd.recurrence.map(cat).to_numpy()
    for lab in ['rep', 'single']:
        wi_ = [i for i in inc_idx if wr[i] == lab]; bj_ = [j for j in range(len(bd)) if br[j] == lab]
        r[f'wf_{lab}'] = f"{sum(i in post_w for i in wi_)}/{len(wi_)}"; r[f'bd_{lab}'] = f"{sum(j in post_b for j in bj_)}/{len(bj_)}"
    r['unmatched_board_by_rec'] = {lab: sum(br[j] == lab for j in un) for lab in ['rep', 'single']}
    r['unmatched_board_names'] = [bd.facility_name[j] + ' | ' + bd.address_line1[j] for j in un] if name == 'new' else None
    r['zip_mismatch_links'] = int(sum(comp['zipdiff'][i, j] for i, j, s in post))
    res[name] = r
for k, v in res.items():
    print('\n==', k)
    for a, b in v.items():
        if a != 'unmatched_board_names': print(f'  {a}: {b}')
print('\nunmatched Board records under new setting:')
for s in res['new']['unmatched_board_names']: print('  ', s)
json.dump(res, open('new_rq2_numbers.json', 'w'), indent=1, ensure_ascii=False, default=str)
