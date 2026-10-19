"""All-active Board sensitivity (245 records): reproduce the frozen 179 links / 73.06%, then rerun with the calibrated setting."""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]  # repository root (analysis/rq2_oat/20261012_v3_factorial/..)
FROZEN_CODE = str(REPO / 'analysis/expanded_final/20260908_v1/implementation/code')
sys.path.insert(0, FROZEN_CODE)
import geo_match
from v2_board_adapter import merge_v2_board_sources, prepare_v2_board_frames
from core import assign_hungarian
U = str(REPO)
raw = pd.read_csv(f'{U}/data/Board/F.1  Pharmacy (PH).csv', encoding='utf-8-sig', skiprows=3, low_memory=False)
cache = pd.read_csv(f'{U}/data/board_geocoded.csv', low_memory=False)
target = sorted(json.load(open(f'{U}/analysis/expanded_final/20260908_v1/frames.json'))['primary_60'])
frames = prepare_v2_board_frames(merge_v2_board_sources(raw, cache), target)
print('frame rows:', len(frames.screened_target), len(frames.full_active_target))
wf = pd.read_csv(f'{U}/analysis/rq2_oat/20261007_v2/workflow.csv', dtype={'Zip_Code': str}, keep_default_na=False)
ai = wf[wf.included.astype(int) == 1].reset_index(drop=True)
geo_match.set_parser('regex')
def components(ai, board):
    n, m = len(ai), len(board)
    N = np.empty((n, m)); H = np.zeros((n, m), bool); St = np.empty((n, m)); E = np.zeros((n, m), bool); Z = np.zeros((n, m), bool)
    bnames = board.facility_name.astype(str).tolist(); baddr = board.address_line1.astype(str).tolist(); bz = board.zip5.astype(str).tolist()
    bparsed = [geo_match.parse_address_regex(a) for a in baddr]
    for i, row in ai.iterrows():
        an, ast_ = geo_match.parse_address_regex(str(row.Street_Address)); az = str(row.Zip_Code)[:5]
        for j in range(m):
            bn, bst = bparsed[j]
            N[i, j] = geo_match.name_sim(row.Company, bnames[j])
            H[i, j] = bool(an and bn and an == bn)
            St[i, j] = geo_match.token_sort_ratio(ast_, bst)
            E[i, j] = not ast_ or not bst
            Z[i, j] = bool(az and bz[j] and az != bz[j])
    return dict(N=N, house=H, street=St, street_empty=E, zipdiff=Z)
def score(c, w, h, z):
    A = h*c['house'] + ((100-h)/100)*c['street']; A = np.where(c['street_empty'], 0.0, A)
    S = (c['N'] + w*A)/(1+w); S = np.where(c['zipdiff'], S*z, S); return np.clip(S, 0, 100)
out = {}
for fname in ['screened_target', 'full_active_target']:
    board = getattr(frames, fname).sort_values('license_nbr').reset_index(drop=True)
    c = components(ai, board)
    # check against the frozen scorer for the original setting
    base = geo_match.score_matrix(ai, board, {'w_name': 1.0, 'w_addr': 0.8})
    print(fname, 'max |recomposed - frozen|:', float(np.abs(score(c, 0.8, 60.0, 0.85) - base).max()))
    for lab, p in [('original', (0.8, 60.0, 0.85, 75.0)), ('calibrated', (1.2, 90.0, 0.90, 60.0))]:
        pairs = assign_hungarian(score(c, *p[:3]), p[3])
        out[(fname, lab)] = dict(board_rows=len(board), links=len(pairs), recall=round(100*len(pairs)/len(board), 2), precision=round(100*len(pairs)/len(ai), 2))
        print(fname, lab, out[(fname, lab)])
json.dump({f'{a}|{b}': v for (a, b), v in out.items()}, open('allactive_results.json', 'w'), indent=1)
