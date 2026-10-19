"""Appendix C.2 quantities under the calibrated setting (1.2, 90, 0.90, 60)."""
import json, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]  # repository root (analysis/rq2_oat/20261012_v3_factorial/..)
from core import load, score, assign_hungarian, run
from scipy.optimize import linear_sum_assignment
U = str(REPO / 'analysis')
W, H, Z, T = 1.2, 90.0, 0.90, 60.0
wf, bd, comp, inc = load()
inc_idx = np.flatnonzero(inc); exc_idx = np.flatnonzero(~inc)
nW, nB = len(inc_idx), len(bd)
out = {}
def pr(links, nw, nb=nB): return dict(links=links, un_w=nw-links, un_b=nb-links, precision=round(100*links/nw, 2), recall=round(100*links/nb, 2))

S = score(comp, W, H, Z)
def zmul(M): return np.where(comp['zipdiff'], M*Z, M)
A = H*comp['house'] + ((100-H)/100)*comp['street']; A = np.where(comp['street_empty'], 0.0, A)
# geography
def hav(lat1, lon1, lat2, lon2):
    R = 6371000.0; p1, p2 = np.radians(lat1), np.radians(lat2); dp, dl = p2-p1, np.radians(lon2-lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2; return 2*R*np.arcsin(np.sqrt(a))
wl = pd.to_numeric(wf.Latitude, errors='coerce').to_numpy()[:, None]; wo = pd.to_numeric(wf.Longitude, errors='coerce').to_numpy()[:, None]
bl = pd.to_numeric(bd._lat, errors='coerce').to_numpy()[None, :]; bo = pd.to_numeric(bd._lon, errors='coerce').to_numpy()[None, :]
d = hav(wl, wo, bl, bo); G = np.where(np.isfinite(d), 100*np.exp(-d/300.0), 0.0)
specs = {
 'name_only': np.clip(zmul(comp['N']), 0, 100),
 'address_only': np.clip(zmul(A), 0, 100),
 'name_address': S,
 'name_address_geo': np.clip(zmul((comp['N'] + W*A + 0.5*G)/(1+W+0.5)), 0, 100),
}
def assign_card(Sm, thr):
    sc = np.asarray(Sm, float); n, m = sc.shape; el = sc >= thr
    ru = sc.copy(); mx = max(float(np.abs(sc[el]).max()), 1.0); ru[el] += (min(n, m)+1)*mx
    size = n+m; mu = max(float(np.abs(ru[el]).max()), 1.0); util = np.zeros((size, size)); util[:n, :m] = np.where(el, ru, -(size+1)*mu)
    r, c = linear_sum_assignment(-util); return sorted((int(i), int(j), float(sc[i, j])) for i, j in zip(r, c) if i < n and j < m and el[i, j])
c11 = {}
main_pairs = assign_hungarian(S[inc_idx], T)
for k, M in specs.items():
    p = assign_hungarian(M[inc_idx], T); c11[k] = pr(len(p), nW)
    c11[k]['same_as_main'] = len(set((i, j) for i, j, s in p) & set((i, j) for i, j, s in main_pairs))
p = assign_card(S[inc_idx], T); c11['max_cardinality'] = pr(len(p), nW); c11['max_cardinality']['same_as_main'] = len(set((i, j) for i, j, s in p) & set((i, j) for i, j, s in main_pairs))
_, _, po = run(comp, inc, 0.8, 60.0, 0.85, 75.0); c11['previous_setting'] = pr(len(po), nW)
c11['previous_setting']['same_as_main'] = len(set((inc_idx.tolist().index(i), j) for i, j, s in po) & set((i, j) for i, j, s in main_pairs))
import sys; sys.path.insert(0, str(REPO / 'analysis/expanded_final/20260908_v1/implementation/code'))
from study_design import normalized_location_key
key = np.array([normalized_location_key(a, z_) for a, z_ in zip(wf.Street_Address, wf.Zip_Code)])
ks = wf.location_key_sha256.to_numpy()
assert pd.Series(key[inc_idx]).groupby(ks[inc_idx]).nunique().max() == 1 and pd.Series(ks[inc_idx]).groupby(key[inc_idx]).nunique().max() == 1
units = pd.Series(inc_idx).groupby(key[inc_idx]).apply(list)
Uu = np.vstack([S[m_].max(axis=0) for m_ in units]); up = assign_hungarian(Uu, T); c11['location_groups'] = pr(len(up), len(units))
out['C11'] = c11
# C12 strata (all-active from allactive_results.json)
pre = assign_hungarian(S, T); ex = assign_hungarian(S[exc_idx], T)
aa = json.load(open('allactive_results.json'))['full_active_target|calibrated']
out['C12'] = {'included_screened': pr(len(main_pairs), nW), 'included_allactive': dict(pr(aa['links'], nW, 245)),
              'excluded_screened': pr(len(ex), len(exc_idx)), 'all_screened': pr(len(pre), len(wf))}
# C13 counting units
allunits = pd.Series(np.arange(len(wf))).groupby(key).apply(list)
Ua = np.vstack([S[m_].max(axis=0) for m_ in allunits]); ua = assign_hungarian(Ua, T)
out['C13'] = {'raw_place': pr(len(pre), len(wf)), 'inc_place': pr(len(main_pairs), nW), 'raw_groups': pr(len(ua), len(allunits)), 'inc_groups': pr(len(up), len(units))}
pre_b = {j for i, j, s in pre}; post_b = {j for i, j, s in main_pairs}
out['C13']['retained'] = len(pre_b & post_b); out['C13']['pre_board'] = len(pre_b)
# endpoints changed between raw and included assignments
pre_map = {j: i for i, j, s in pre}; post_map = {j: int(inc_idx[i]) for i, j, s in main_pairs}
out['C13']['same_endpoint'] = sum(pre_map[j] == post_map[j] for j in pre_b & post_b)
# C14
E = S >= T
inc_edge = E[inc_idx].any(1); exc_edge = E[exc_idx].any(1)
out['C14A'] = {'inc_edge': int(inc_edge.sum()), 'exc_edge': int(exc_edge.sum()), 'inc_noedge': int((~inc_edge).sum()), 'exc_noedge': int((~exc_edge).sum())}
un = [j for j in range(nB) if j not in post_b]
out['C14B'] = {'no_raw_edge': sum(not E[:, j].any() for j in un), 'only_excluded': sum(E[:, j].any() and not E[inc_idx, j].any() for j in un),
               'inc_edge_unassigned': sum(E[inc_idx, j].any() for j in un), 'matched': len(post_b)}
# C15
post_w = {int(inc_idx[i]) for i, j, s in main_pairs}
un_w = [i for i in inc_idx if i not in post_w]
kc = pd.Series(key[inc_idx]).value_counts()
sizes = pd.Series(key[inc_idx]).groupby(key[inc_idx]).size().value_counts().sort_index()
matched_keys = {key[i] for i in post_w}
out['C15'] = {'group_sizes': {int(k): int(v) for k, v in sizes.items()}, 'n_groups': len(units), 'unmatched': len(un_w),
              'unmatched_shared': sum(kc[key[i]] > 1 for i in un_w), 'unmatched_colocated_matched': sum(key[i] in matched_keys for i in un_w),
              'unmatched_with_edge': sum(E[i].any() for i in un_w), 'unmatched_no_edge': sum(not E[i].any() for i in un_w)}
# C16 excluded with edges
el = pd.read_csv(f'{U}/expanded_final/20260908_v1/rq2/eligibility.csv', dtype=str, keep_default_na=False).set_index('place_id')
exe = [i for i in exc_idx if E[i].any()]
lab = el.loc[wf.place_id.iloc[exe]]
inc_keys = set(key[inc_idx])
top = np.array([S[i].max() for i in exe])
topj = [int(S[i].argmax()) for i in exe]
out['C16'] = {'n': len(exe), 'target': lab.target_construct_status.value_counts().to_dict(), 'business': lab.business_status.value_counts().to_dict(),
              'plaus': lab.candidate_plausibility.value_counts().to_dict(), 'identity': lab.identity_consistency.value_counts().to_dict(),
              'top_board_covered': sum(j in post_b for j in topj), 'any_board_covered': sum(any(E[i, j] and j in post_b for j in range(nB)) for i in exe),
              'shares_key_with_included': sum(key[i] in inc_keys for i in exe),
              'top_mean': round(float(top.mean()), 2), 'top_median': round(float(np.median(top)), 2), 'top_min': round(float(top.min()), 2), 'top_max': round(float(top.max()), 2),
              'names': [f"{wf.Company.iloc[i]} | {wf.Street_Address.iloc[i]} -> {bd.facility_name.iloc[topj[k]]} ({S[i, topj[k]]:.2f})" for k, i in enumerate(exe)]}
# C18 unmatched Board
stage = {}
for j in un:
    stage[j] = 'no_raw_edge' if not E[:, j].any() else ('only_excluded' if not E[inc_idx, j].any() else 'inc_edge_unassigned')
out['C18'] = [dict(name=bd.facility_name[j], address=bd.address_line1[j], city=bd.city[j], zip=bd.zip5[j], stage=stage[j], recurrence=bd.recurrence[j]) for j in un]
json.dump(out, open('appendix_numbers.json', 'w'), indent=1, ensure_ascii=False, default=int)
for k, v in out.items():
    if k != 'C18': print(k, json.dumps(v, ensure_ascii=False, default=int)[:1500])
for r in out['C18']: print(r)
# location-level identity overlap with the main assignment
main_loc = {(key[int(inc_idx[i])], j) for i, j, s in main_pairs}
ov = {}
for k, M in specs.items():
    p = assign_hungarian(M[inc_idx], T)
    ov[k] = dict(same_board=len({j for i, j, s in p} & post_b), same_loc_pair=len({(key[int(inc_idx[i])], j) for i, j, s in p} & main_loc))
p = assign_card(S[inc_idx], T); ov['max_cardinality'] = dict(same_board=len({j for i, j, s in p} & post_b), same_loc_pair=len({(key[int(inc_idx[i])], j) for i, j, s in p} & main_loc))
ov['previous_setting'] = dict(same_board=len({j for i, j, s in po} & post_b), same_loc_pair=len({(key[i], j) for i, j, s in po} & main_loc))
print('overlap', ov)
out['C11_overlap'] = ov
json.dump(out, open('appendix_numbers.json', 'w'), indent=1, ensure_ascii=False, default=int)
