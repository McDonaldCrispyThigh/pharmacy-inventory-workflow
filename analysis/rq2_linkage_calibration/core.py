"""Shared loader, scorer and assignment, replicating the frozen v2 OAT runner."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
# Frozen v2 inputs sit next to this folder in the repository.
SRC = HERE.parent / '20261007_v2'
if not (SRC / 'pairs.csv').exists():
    SRC = Path('/mnt/user-data/uploads/NETS-AI/analysis/rq2_oat/20261007_v2')
CACHE = HERE / 'comp.npz'   # local cache of the dense component matrices; safe to delete

def load():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        comp = {k: z[k] for k in ['N','house','street','zipdiff','street_empty','score_baseline','house_differ','house_missing']}
        wf = pd.read_csv(SRC/'workflow.csv', dtype={'workflow_id':str}, keep_default_na=False)
        bd = pd.read_csv(SRC/'board.csv', dtype={'board_id':str}, keep_default_na=False)
        return wf, bd, comp, wf.included.astype(int).astype(bool).to_numpy()
    wf = pd.read_csv(SRC/'workflow.csv', dtype={'workflow_id':str}, keep_default_na=False, float_precision='round_trip')
    bd = pd.read_csv(SRC/'board.csv', dtype={'board_id':str}, keep_default_na=False, float_precision='round_trip')
    pr = pd.read_csv(SRC/'pairs.csv', dtype={'workflow_id':str,'board_id':str}, keep_default_na=False, float_precision='round_trip',
                     usecols=['workflow_id','board_id','name_sim','house_status','street_sim','zip_differs','score_baseline','street_empty'])
    assert wf.workflow_id.tolist() == sorted(wf.workflow_id) and bd.board_id.tolist() == sorted(bd.board_id)
    wi = {k:i for i,k in enumerate(wf.workflow_id)}; bi = {k:j for j,k in enumerate(bd.board_id)}
    rr = pr.workflow_id.map(wi).to_numpy(); cc = pr.board_id.map(bi).to_numpy(); shape = (len(wf), len(bd))
    def dense(v, dt=float):
        a = np.zeros(shape, dtype=dt); a[rr, cc] = v; return a
    hs = pr.house_status.astype(str).str.strip().str.lower()
    comp = {'N': dense(pr.name_sim.astype(float).to_numpy()),
            'house': dense(hs.eq('match').to_numpy(), bool),
            'street': dense(pr.street_sim.astype(float).to_numpy()),
            'zipdiff': dense(pd.to_numeric(pr.zip_differs).astype(int).to_numpy().astype(bool), bool),
            'street_empty': dense(pd.to_numeric(pr.street_empty).astype(int).to_numpy().astype(bool), bool),
            'score_baseline': dense(pr.score_baseline.astype(float).to_numpy()),
            'house_differ': dense(hs.eq('differ').to_numpy(), bool),
            'house_missing': dense(hs.eq('missing').to_numpy(), bool)}
    np.savez(CACHE, **comp)
    return wf, bd, comp, wf.included.astype(int).astype(bool).to_numpy()

def score(comp, w_A, h, z):
    A = h*comp['house'] + ((100.0-h)/100.0)*comp['street']
    A = np.where(comp['street_empty'], 0.0, A)
    S = (comp['N'] + w_A*A)/(1.0+w_A)
    S = np.where(comp['zipdiff'], S*z, S)
    return np.clip(S, 0.0, 100.0)

def assign_hungarian(S, thr):
    """Verbatim logic of frozen geo_match.assign_hungarian(objective='max_weight')."""
    scores = np.asarray(S, dtype=float); n, m = scores.shape
    eligible = scores >= float(thr)
    if not eligible.any(): return []
    real_utility = scores.copy(); size = n + m
    max_utility = max(float(np.max(np.abs(real_utility[eligible]))), 1.0)
    forbidden = -(size + 1) * max_utility
    utility = np.zeros((size, size), dtype=float)
    utility[:n, :m] = np.where(eligible, real_utility, forbidden)
    rows, cols = linear_sum_assignment(-utility)
    pairs = [(int(i), int(j), float(scores[i, j])) for i, j in zip(rows, cols) if i < n and j < m and eligible[i, j]]
    return sorted(pairs, key=lambda p: (p[0], p[1]))

def run(comp, included, w_A, h, z, tau):
    S = score(comp, w_A, h, z)
    inc_idx = np.flatnonzero(included)
    pre = assign_hungarian(S, tau)
    local = assign_hungarian(S[inc_idx, :], tau)
    post = [(int(inc_idx[i]), j, s) for i, j, s in local]
    return S, pre, post
