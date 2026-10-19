import sys; sys.path.insert(0, "code")
import pandas as pd, numpy as np
import geo_match as G

ai = G.load_ai()
ai_zips = set(ai["Zip_Code"].dropna().astype(str).str[:5].unique())
board = pd.read_csv("data/board_geocoded.csv", low_memory=False)   # has _lat/_lon
board["zip5"] = board["zip5"].astype(str).str.zfill(5)

# reference sets
a3 = pd.read_csv("data/audit3_fn_institutional_20260414_172017.csv")
inst_lic = set(a3[a3.is_institutional.astype(bool)].license_nbr)
fn138 = pd.read_csv("data/board_fn_20260414_170601.csv")
spec_lic = set(fn138.license_nbr) - set(a3.license_nbr)
nonretail_lic = inst_lic | spec_lic
retail_lic = set(board.license_nbr) - nonretail_lic
CHAIN=['walgreen','cvs','walmart','wal-mart',"sam's west",'sam west',"sam's club",'target','costco','hy-vee','hyvee','cub','supervalu','coborn','thrifty white']
isch=lambda s: any(p in str(s).lower() for p in CHAIN)
retail_fn=a3[~a3.is_institutional.astype(bool)]
dba_lic=set(retail_fn[retail_fn.facility_name.apply(isch)].license_nbr)

def metrics(pairs, name):
    matched={board.iloc[j].license_nbr for _,j,_ in pairs}
    tp=len(pairs); rtp=len(matched & retail_lic); itp=len(matched & nonretail_lic)
    prec_ret=rtp/len(ai); rec=rtp/len(retail_lic); f1=2*prec_ret*rec/(prec_ret+rec) if prec_ret+rec else 0
    return dict(strategy=name, retail_TP=rtp, inst_match=itp,
                precision=round(prec_ret*100,1), adj_recall=round(rec*100,1),
                adj_F1=round(f1*100,1), matched_lic=matched)

geo_ok = int(board["_lat"].notna().sum())
strategies=[
 ("A. name-only (greedy, CURRENT)", dict(w_name=1.0), "greedy"),
 ("C. name + address-string",       dict(w_name=1.0, w_addr=0.8), "hungarian"),
 ("D. name + geocoded distance",    dict(w_name=1.0, w_geo=0.8, geo_scale=300.0), "hungarian"),
 ("E. name + address + distance",   dict(w_name=1.0, w_addr=0.8, w_geo=0.5, geo_scale=300.0), "hungarian"),
]
res={}
for label,cfg,method in strategies:
    S=G.score_matrix(ai, board, cfg)
    pairs=G.assign_greedy(S,75.0) if method=="greedy" else G.assign_hungarian(S,75.0)
    res[label]=metrics(pairs,label)

base=res[strategies[0][0]]["matched_lic"]
tbl=pd.DataFrame([{k:v for k,v in r.items() if k!="matched_lic"} for r in res.values()])
print("="*94); print(f"MATCHER HEAD-TO-HEAD  (Board geocoded: {geo_ok}/459; 43 No_Match, 101 Non_Exact)"); print("="*94)
print(tbl.to_string(index=False))
print(f"\nretail reference = {len(retail_lic)}; institutional/specialty = {len(nonretail_lic)}; AI records = {len(ai)}")

# DBA recovery + geocoder-specific failures
for label in [s[0] for s in strategies[1:]]:
    rec = res[label]["matched_lic"] - base
    dba = len(rec & dba_lic)
    print(f"\n{label}: net-new retail+inst matches vs baseline = {len(rec & (retail_lic|nonretail_lic))}, of which chain-DBA recoveries = {dba}")

# Where geocoder (D) fails that address-string (C) succeeds
recC=res["C. name + address-string"]["matched_lic"]-base
recD=res["D. name + geocoded distance"]["matched_lic"]-base
onlyC=recC-recD
print(f"\nRecovered by address-string but MISSED by geocoder: {len(onlyC & (retail_lic|nonretail_lic))}")
for lic in list(onlyC)[:12]:
    row=board[board.license_nbr==lic].iloc[0]
    geo = "NO_GEOCODE" if pd.isna(row["_lat"]) else row.get("mtype","?")
    print(f"   {str(row.facility_name)[:46]:46s} | {str(row.address_line1)[:22]:22s} | geocode={geo}")
