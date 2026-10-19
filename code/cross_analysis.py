"""
Cross-analysis: Wayback x Board match rate, Chain vs Independent recall.
Output: cross_analysis_results.txt
"""
import pandas as pd, numpy as np, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA = 'c:/Users/16067/NETS-AI/data/'

tp = pd.read_csv(DATA + 'audit2_tp_analysis_20260414_172017.csv')
fp = pd.read_csv(DATA + 'audit2_fp_analysis_20260414_172017.csv')
fn = pd.read_csv(DATA + 'audit3_fn_institutional_20260414_172017.csv')

tp['is_tp'] = True
fp['is_tp'] = False
all_ai = pd.concat([tp, fp], ignore_index=True)

fn_retail = fn[fn['is_institutional'] == False].copy()

chain_kw = ['walgreens', 'cvs', 'walmart', 'target', 'costco', 'hy-vee', 'hyvee', 'cub pharmacy']
all_ai['is_chain'] = all_ai['Company'].str.lower().str.contains('|'.join(chain_kw))
fn_retail['is_chain'] = fn_retail['facility_name'].str.lower().str.contains('|'.join(chain_kw))

# ── TABLE 1: Wayback groups ──────────────────────────────────────────
def wb_group(x):
    if x == -1: return 'Chain sentinel (-1)'
    if x == 0:  return 'Zero web presence'
    if 1 <= x <= 7: return 'Low (1-7 yrs)'
    return 'Established (20+ yrs)'

all_ai['wb_grp'] = all_ai['Wayback_Snapshot_Count'].apply(wb_group)

order = ['Chain sentinel (-1)', 'Zero web presence', 'Low (1-7 yrs)', 'Established (20+ yrs)']
rows = []
for g in order:
    grp = all_ai[all_ai['wb_grp'] == g]
    n = len(grp); tp_n = int(grp['is_tp'].sum()); fp_n = n - tp_n
    rows.append({'Wayback Group': g, 'n': n, 'TPs': tp_n,
                 'FPs': fp_n, 'Board Match Rate': f'{tp_n/n*100:.1f}%'})
tbl1 = pd.DataFrame(rows)
print('TABLE 1: Board match rate by Wayback snapshot group (all 399 AI records)')
print(tbl1.to_string(index=False))
print()

# ── TABLE 2: Chain vs independent ───────────────────────────────────
tp_c = tp[tp['Company'].str.lower().str.contains('|'.join(chain_kw))]
tp_i = tp[~tp['Company'].str.lower().str.contains('|'.join(chain_kw))]
fp_c = fp[fp['Company'].str.lower().str.contains('|'.join(chain_kw))]
fp_i = fp[~fp['Company'].str.lower().str.contains('|'.join(chain_kw))]
fn_c = fn_retail[fn_retail['is_chain']]
fn_i = fn_retail[~fn_retail['is_chain']]

bd_c = len(tp_c) + len(fn_c)
bd_i = len(tp_i) + len(fn_i)

rows2 = [
    {'Segment': 'Chain', 'AI records': len(tp_c)+len(fp_c), 'TPs': len(tp_c),
     'FPs': len(fp_c), 'Board denom': bd_c, 'Retail FNs': len(fn_c),
     'Precision': f'{len(tp_c)/(len(tp_c)+len(fp_c))*100:.1f}%',
     'Recall': f'{len(tp_c)/bd_c*100:.1f}%'},
    {'Segment': 'Independent', 'AI records': len(tp_i)+len(fp_i), 'TPs': len(tp_i),
     'FPs': len(fp_i), 'Board denom': bd_i, 'Retail FNs': len(fn_i),
     'Precision': f'{len(tp_i)/(len(tp_i)+len(fp_i))*100:.1f}%',
     'Recall': f'{len(tp_i)/bd_i*100:.1f}%'},
]
tbl2 = pd.DataFrame(rows2)
print('TABLE 2: Precision and recall by chain vs independent segment')
print(tbl2.to_string(index=False))
print()

# ── Manual FN audit summary ──────────────────────────────────────────
inst_slipthrough_kw = [
    'treatment center', 'omnicare', 'cardinal health', 'ebenezer',
    'evexia', 'purescripts', 'genoa healthcare', 'hcmc', 'jubilant',
    'mobe', 'nura', 'open cities health', 'option care', 'our lady of peace',
    'pediatric home', 'petnet', 'pharmerica', 'pillpack', 'amazon pharmacy',
    'post acute', 'prairiecare', 'roundtablerx', 'medication repository',
    'aliveness project', 'thrive rx', 'tria pharmacy', 'united family practice',
    'riverland community', 'west side community health'
]

def classify_fn_sub(name):
    n = name.lower()
    chain_legal = ['grand st. paul cvs', 'walgreen co', 'walmart inc.', 'wal-mart',
                   "sam's west", 'sam west', 'supervalu', 'hy-vee', 'costco wholesale']
    if any(p in n for p in chain_legal):
        return 'DBA chain match failure'
    if any(p in n for p in inst_slipthrough_kw):
        return 'Institutional slipthrough'
    return 'Genuine independent miss'

fn_retail['sub_cat'] = fn_retail['facility_name'].apply(classify_fn_sub)

print('TABLE 3: Manual audit of 84 apparent retail FNs')
print(fn_retail['sub_cat'].value_counts().to_string())
print()
print('Genuine independent misses:')
for _, r in fn_retail[fn_retail['sub_cat'] == 'Genuine independent miss'].iterrows():
    print(f'  {r["facility_name"][:55]:55s}  {r["city"]:15s}  {r["zip5"]}')

print()
print('=== KEY NUMBERS FOR THESIS ===')
print(f'Zero-web Board match rate:         {all_ai[all_ai["wb_grp"]=="Zero web presence"]["is_tp"].mean()*100:.1f}%')
print(f'Established Board match rate:      {all_ai[all_ai["wb_grp"]=="Established (20+ yrs)"]["is_tp"].mean()*100:.1f}%')
print(f'Chain sentinel Board match rate:   {all_ai[all_ai["wb_grp"]=="Chain sentinel (-1)"]["is_tp"].mean()*100:.1f}%')
print(f'Chain precision:  {len(tp_c)/(len(tp_c)+len(fp_c))*100:.1f}%')
print(f'Chain recall:     {len(tp_c)/bd_c*100:.1f}%')
print(f'Indep precision:  {len(tp_i)/(len(tp_i)+len(fp_i))*100:.1f}%')
print(f'Indep recall:     {len(tp_i)/bd_i*100:.1f}%')
print(f'Match rate gap (chain-indep): {(len(tp_c)/(len(tp_c)+len(fp_c)) - len(tp_i)/(len(tp_i)+len(fp_i)))*100:.1f}pp (precision)')
print(f'Recall gap (chain-indep):     {(len(tp_c)/bd_c - len(tp_i)/bd_i)*100:.1f}pp')
