"""Score every setting against the pair readings, following selection_rule.md."""
import numpy as np, pandas as pd
from core import *
wf, bd, comp, inc = load()
wi = {k:i for i,k in enumerate(wf.workflow_id)}; bi = {k:j for j,k in enumerate(bd.board_id)}
R = pd.read_csv('pair_readings_270.csv', dtype={'workflow_id':str,'board_id':str})
lab = {(r.workflow_id, r.board_id): r.reading for r in R.itertuples()}
runs = pd.read_csv('factorial_runs.csv')
ch = pd.read_csv('factorial_pair_changes.csv', dtype={'workflow_id':str,'board_id':str})
key = ['w_A','h','z','tau']
_, _, post0 = run(comp, inc, 0.8, 60.0, 0.85, 75.0)
base = {(wf.workflow_id[i], bd.board_id[j]) for i,j,s in post0}
groups = {k: g for k, g in ch.groupby(key)}
# readings per Board record
def errors(links, mode):
    def L(p):
        r = lab[p]
        if r == '不确定' and mode == 'as_same': return '同一家'
        if r == '不确定' and mode == 'as_diff': return '不同'
        return r
    true_sets = {}
    for p in lab:
        if L(p) == '同一家': true_sets.setdefault(p[1], set()).add(p[0])
    linked_b = {}
    for (a, b) in links: linked_b[b] = a
    wrong = sum(1 for (a, b) in links if L((a, b)) == '不同')
    missed = sum(1 for b, ts in true_sets.items() if b not in linked_b)
    return wrong, missed
# components for margin
pairs_dec = [(wi[a], bi[b]) for (a, b), r in lab.items() if r in ('同一家', '不同')]
ii = np.array([p[0] for p in pairs_dec]); jj = np.array([p[1] for p in pairs_dec])
sub = {k: comp[k][ii, jj] for k in ['N','house','street','street_empty','zipdiff']}
out = []
for r in runs.itertuples():
    k = (r.w_A, r.h, r.z, r.tau)
    links = set(base)
    if k in groups:
        g = groups[k]
        links -= {(a, b) for a, b, kd in zip(g.workflow_id, g.board_id, g.kind) if kd == 'lost'}
        links |= {(a, b) for a, b, kd in zip(g.workflow_id, g.board_id, g.kind) if kd == 'added'}
    assert len(links) == r.post_links
    row = dict(w_A=r.w_A, h=r.h, z=r.z, tau=r.tau, links=len(links))
    for mode in ['main', 'as_same', 'as_diff']:
        w_, m_ = errors(links, mode); row[f'wrong_{mode}'] = w_; row[f'missed_{mode}'] = m_; row[f'err_{mode}'] = w_ + m_
    A = r.h*sub['house'] + ((100-r.h)/100)*sub['street']; A = np.where(sub['street_empty'], 0, A)
    S = (sub['N'] + r.w_A*A)/(1+r.w_A); S = np.where(sub['zipdiff'], S*r.z, S); S = np.clip(S, 0, 100)
    row['margin'] = float(np.min(np.abs(S - r.tau)))
    row['n_same_as_primary'] = int(r.w_A==0.8) + int(r.h==60) + int(r.z==0.85) + int(r.tau==75)
    G = {'w_A':[0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5],'h':[30,40,50,60,70,80,90],'z':[0.7,0.75,0.8,0.85,0.9,0.95,1.0],'tau':[60,65,70,75,80,85,90]}
    P = {'w_A':0.8,'h':60,'z':0.85,'tau':75}
    row['grid_dist'] = sum(abs(G[k].index(getattr(r, k)) - G[k].index(P[k])) for k in G)
    row['lock'] = r.lock; row['zip_ok'] = r.zip_ok; row['name_floor'] = r.name_floor
    row['precision'] = r.precision; row['recall'] = r.recall
    out.append(row)
D = pd.DataFrame(out); D.to_csv('calibration_runs.csv', index=False)
prim = D[(D.w_A==0.8)&(D.h==60)&(D.z==0.85)&(D.tau==75)].iloc[0]
print('primary:', prim[['links','wrong_main','missed_main','err_main','err_as_same','err_as_diff','margin']].to_dict())
best = D.err_main.min(); B = D[D.err_main == best]
print('min errors (main):', best, 'settings achieving it:', len(B))
B = B.sort_values(['margin','n_same_as_primary','grid_dist'], ascending=[False, False, True])
print(B.head(12)[['w_A','h','z','tau','links','wrong_main','missed_main','margin','n_same_as_primary','grid_dist','lock','zip_ok','name_floor','err_as_same','err_as_diff']].to_string(index=False))
print('\nerror distribution:', D.err_main.value_counts().sort_index().head(10).to_dict())
for mode in ['as_same','as_diff']:
    m = D[f'err_{mode}'].min(); print(mode, 'min', m, '| chosen in its min set:', bool(B.iloc[0][f'err_{mode}'] == m), '| primary', prim[f'err_{mode}'])
