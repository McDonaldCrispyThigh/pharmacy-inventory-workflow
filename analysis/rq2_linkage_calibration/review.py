"""Boundary pairs: every pair that enters or leaves relative to the primary setting, with a preliminary same-pharmacy reading."""
import pandas as pd, numpy as np
from core import *
wf, bd, comp, inc = load()
df = pd.read_csv('factorial_runs.csv'); ch = pd.read_csv('factorial_pair_changes.csv')
key = ['w_A','h','z','tau']
ch = ch.merge(df[key+['lock','zip_ok','name_floor']], on=key)
S0, pre0, post0 = run(comp, inc, 0.8, 60.0, 0.85, 75.0)
lw = {i: j for i, j, s in post0}; lb = {j: i for i, j, s in post0}
def hs(i, j): return 'match' if comp['house'][i, j] else ('differ' if comp['house_differ'][i, j] else 'missing')
groups = {
 'A_lock_fails_house_mismatch': ch[(ch.kind=='added') & ~ch.lock],
 'B_added_rules_hold': ch[(ch.kind=='added') & ch.lock & ch.zip_ok],
 'C_lost_rules_hold': ch[(ch.kind=='lost') & ch.lock & ch.zip_ok],
}
rows = []
for g, d in groups.items():
    if g.startswith('A'):
        d = d[[hs(int(i), int(j)) != 'match' for i, j in zip(d.i, d.j)]]
    agg = d.groupby(['i','j']).agg(settings=('kind','size'), floor_min=('name_floor','min'), floor_max=('name_floor','max')).reset_index()
    for _, r in agg.iterrows():
        i, j = int(r.i), int(r.j)
        wa = wf.Street_Address[i]; ba = bd.address_line1[j]; wn = wf.Company[i]; bn = bd.facility_name[j]
        # preliminary reading (to be confirmed by the author)
        if g.startswith('A'):
            lab = '不同门店'
            if '57th Ave NE' in wa and '246 57th' in ba: lab = '可能同一家（门牌号250与246）'
            if 'Target Plaza South' in wa: lab = '不确定（市中心Target店内CVS，一侧缺门牌号）'
        elif g.startswith('B'):
            lab = '同一家'
            if wn.startswith('Lakes Pharmacy'): lab = '可能同一家（同址，名称不同）'
            if wn.startswith("Sam's Club") and 'Cub' in bn: lab = '不同门店（门牌号相同，街道不同）'
            if wn.startswith('Thrive'): lab = '不确定（同址诊所与药房）'
            if wn == 'Target': lab = '不确定（Target店内CVS）'
        else:
            lab = '同一家'
            if wn.startswith('Haas Peter'): lab = '不确定（药剂师个人条目）'
        rows.append(dict(group=g, workflow_name=wn, workflow_address=wa.split(',')[0], board_name=bn, board_address=ba,
            place_id=wf.workflow_id[i], license_nbr=bd.board_id[j], house_status=hs(i, j), name_sim=round(comp['N'][i, j], 1),
            street_sim=round(comp['street'][i, j], 1), zip_differs=int(comp['zipdiff'][i, j]), primary_score=round(S0[i, j], 2),
            in_primary=int(lb.get(j) == i), board_linked_in_primary=int(j in lb), workflow_linked_in_primary=int(i in lw),
            settings=int(r.settings), name_floor_range=f'{r.floor_min:.1f}-{r.floor_max:.1f}', preliminary_reading=lab, author_reading=''))
out = pd.DataFrame(rows).sort_values(['group', 'settings'], ascending=[True, False])
out.to_csv('boundary_pairs_review.csv', index=False, encoding='utf-8-sig')
print(out.groupby(['group', 'preliminary_reading']).size())
# near-threshold pairs among included records
inc_idx = np.flatnonzero(inc)
near = []
for a, j in zip(*np.nonzero((S0[inc_idx] > 73.0) & (S0[inc_idx] < 75.0))):
    i = inc_idx[a]
    near.append(dict(primary_score=round(S0[i, j], 2), house_status=hs(i, j), workflow_name=wf.Company[i], workflow_address=wf.Street_Address[i].split(',')[0],
                     board_name=bd.facility_name[j], board_address=bd.address_line1[j], board_linked_in_primary=int(j in lb), workflow_linked_in_primary=int(i in lw)))
near = pd.DataFrame(near).sort_values('primary_score', ascending=False)
near.to_csv('near_threshold_pairs.csv', index=False, encoding='utf-8-sig')
print(near[['primary_score','house_status','board_linked_in_primary','workflow_linked_in_primary']].value_counts())
