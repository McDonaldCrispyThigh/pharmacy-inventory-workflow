"""Fetch and screen only the 836 novel candidates; no LLM/archive capability."""
from pathlib import Path
import argparse,csv,json,os,re,sys,time,uuid,unicodedata
from collections import Counter
from urllib.request import Request,build_opener
from urllib.error import HTTPError
from urllib.parse import unquote
from email.utils import parsedate_to_datetime
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from diagnose_saturated_zip_search import sha,save,now,digest,NoRedirect
from study_design import TARGET_ZIPS
from wayback_snapshot_fetcher import website_key
WORK=ROOT/'analysis/maintenance/supplemental_details_836_20260907_v1'
GRID=ROOT/'analysis/maintenance/saturated_zip_grid_20260907_v1'
REFINE=ROOT/'analysis/maintenance/saturated_zip_grid_55369_refinement_20260907_v2'
INVENTORY=ROOT/'analysis/closed_phase3/20260907T192546Z_v1/rq2/candidate_inventory.csv'
MASK='id,displayName,formattedAddress,addressComponents,location,types,primaryType,businessStatus,websiteUri,movedPlaceId'
PHARMACY=re.compile(r'\b(?:pharmacy|pharmacies|drug[\s_-]*stores?|apothecary|apothecaries|compounding|prescriptions?|rx)\b',re.I)
BRAND=re.compile(r'\b(?:cvs|walgreens)\b',re.I)
GROUPS=['primary_60_zip','supply_99_only','outside_99_zip','missing_or_invalid_returned_zip','non_us_address','details_failure','pending']

def fold(value):return unicodedata.normalize('NFKD',value).encode('ascii','ignore').decode().casefold()

def returned_zip(details):
    found=set();invalid=False;countries=set()
    for c in details.get('addressComponents',[]):
        types=c.get('types',[])
        if 'country' in types:countries.add(c.get('shortText','').upper())
        if 'postal_code' in types:
            value=c.get('longText') or c.get('shortText') or ''
            m=re.fullmatch(r'(\d{5})(?:-\d{4})?',value.strip())
            if m:found.add(m[1])
            else:invalid=True
    if len(found)==1 and not invalid:return next(iter(found)),'valid',sorted(countries)
    return '',('conflicting' if len(found)>1 else 'invalid' if invalid else 'missing'),sorted(countries)

def derive(details,frames):
    postal,zstate,countries=returned_zip(details)
    if countries and countries!=['US']:group='non_us_address'
    elif not postal:group='missing_or_invalid_returned_zip'
    elif postal in frames['primary_60']:group='primary_60_zip'
    elif postal in frames['supply_99']:group='supply_99_only'
    else:group='outside_99_zip'
    name=details.get('displayName',{}).get('text','')
    site=details.get('websiteUri','')
    types=sorted(set(details.get('types',[])))
    primary=details.get('primaryType','')
    alltypes=set(types)|({primary} if primary else set())
    if not site:website='missing'
    else:
        try:website_key(site);website='usable'
        except (ValueError,TypeError):website='invalid'
    name_matches=sorted({m.group().lower() for m in PHARMACY.finditer(fold(name))})
    url_matches=sorted({m.group().lower() for m in PHARMACY.finditer(fold(unquote(site)))})
    brand=bool(BRAND.search(fold(name+' '+unquote(site))))
    if 'pharmacy' in alltypes:signal='pharmacy_type'
    elif alltypes&{'drugstore','drug_store'}:signal='drugstore_type_only'
    elif name_matches or url_matches:signal='explicit_name_or_url_only'
    elif brand:signal='recognized_brand_only'
    else:signal='no_prespecified_signal'
    explicit=signal in {'pharmacy_type','drugstore_type_only','explicit_name_or_url_only'}
    return {'place_id':details['id'],'details_status':'success','name':name,'formatted_address':details.get('formattedAddress',''),
        'returned_zip':postal,'zip_parse_status':zstate,'returned_countries':countries,'frame_group':group,
        'types':types,'primary_type':primary,'business_status':details.get('businessStatus') or 'MISSING',
        'website':site,'website_state':website,'website_returned':bool(site),'pharmacy_type_signal':'pharmacy' in alltypes,
        'drugstore_type_signal':bool(alltypes&{'drugstore','drug_store'}),'name_signal_matches':name_matches,'url_signal_matches':url_matches,
        'recognized_brand_hint':brand,'signal_group':signal,'explicit_pharmacy_related_signal':explicit,
        'any_signal_including_brand':explicit or brand,'moved_place_id':details.get('movedPlaceId','')}

def validate(details,pid):
    if not isinstance(details,dict) or details.get('id')!=pid:raise ValueError('identity_mismatch')
    if not set(details)<=set(MASK.split(',')):raise ValueError('unexpected_field')
    for name in ['formattedAddress','primaryType','businessStatus','websiteUri','movedPlaceId']:
        if name in details and not isinstance(details[name],str):raise ValueError('invalid_'+name)
    if not isinstance(details.get('types',[]),list) or not all(isinstance(x,str) for x in details.get('types',[])):raise ValueError('invalid_types')
    if not isinstance(details.get('displayName',{}),dict) or not isinstance(details.get('displayName',{}).get('text',''),str):raise ValueError('invalid_name')
    if not isinstance(details.get('addressComponents',[]),list):raise ValueError('invalid_address_components')
    for c in details.get('addressComponents',[]):
        if not isinstance(c,dict) or not isinstance(c.get('types',[]),list):raise ValueError('invalid_component')
        if not all(isinstance(t,str) for t in c.get('types',[])):raise ValueError('invalid_component_type')
        if not all(isinstance(c.get(k,''),str) for k in ['shortText','longText']):raise ValueError('invalid_component_text')
    return details

def prepare(work):
    assert not (work/'protocol.json').exists(),'protocol already frozen'
    base=set(json.loads((GRID/'baseline_place_ids.json').read_text()))
    union=set(json.loads((REFINE/'cumulative_candidate_union.json').read_text()))
    ids=sorted(union-base);assert len(ids)==836 and len(base)==1226
    for directory in [GRID,REFINE]:
        manifest=json.loads((directory/'final_manifest.json').read_text())
        for p,h in manifest['files'].items():assert sha(directory/p)==h
    # Read only the two required columns; classifier/Board outcomes are not used.
    import pandas as pd
    inv=pd.read_csv(INVENTORY,usecols=['place_id','returned_postal_code'],dtype=str).fillna('')
    assert set(inv.place_id)==base and len(inv)==1226
    primary=set(TARGET_ZIPS)
    supply=primary|{z for z in inv.returned_postal_code if re.fullmatch(r'\d{5}',z)}
    assert len(primary)==60 and len(supply)==99
    work.mkdir(parents=True,exist_ok=True)
    save(work/'input_place_ids.json',ids)
    save(work/'frames.json',{'primary_60':sorted(primary),'supply_99':sorted(supply),'supply_only_39':sorted(supply-primary)})
    provenance=json.loads((GRID/'novel_id_provenance.json').read_text())
    for pid,jobs in json.loads((REFINE/'new_id_provenance.json').read_text()).items():
        provenance[pid]=[{'query_job':job,'source_zip':'55369','level':3,'state':'exhausted'} for job in jobs]
    assert set(provenance)==set(ids);save(work/'search_provenance.json',provenance)
    hashes=json.loads((GRID/'protected_hashes.json').read_text())
    hashes[INVENTORY.relative_to(ROOT).as_posix()]=sha(INVENTORY)
    for d in [GRID,REFINE]:
        hashes[(d/'final_manifest.json').relative_to(ROOT).as_posix()]=sha(d/'final_manifest.json')
    save(work/'protected_hashes.json',hashes)
    save(work/'protocol.json',{'frozen_at':now(),'authorization':'Fetch Details for the 836 new IDs, stratify by returned ZIP using fixed 60/99 frames, summarize provider types/status/websites/pharmacy signals, then stop for review before Wayback or LLM.',
        'candidate_count':836,'endpoint':'https://places.googleapis.com/v1/places/{place_id}','field_mask':MASK,
        'field_scope':'Screening fields only; no review or generative-summary request. Future classification evidence would require separate authorization/acquisition.',
        'zip_rule':'Use returned postal_code address component; accept exactly five digits or ZIP+4 and retain first five. Missing, invalid, conflicting and non-US addresses are explicitly separate. Never substitute source query ZIP or enlarge frozen 99-ZIP frame.',
        'signal_rule':{'types':['pharmacy','drugstore','drug_store'],'name_or_url_regex':PHARMACY.pattern,'brand_hint_regex':BRAND.pattern,'normalization':'ASCII NFKD casefold; percent-decode URL; word boundaries','explicit_signal':'type or explicit name/URL expression; brand-only is separate','excluded_automatic_signals':['health alone','store alone','supermarket alone','Walmart/Target/Costco brand alone'],'meaning':'Broad provider signal for triage, not verified dispensing/public access/eligibility.'},
        'mutually_exclusive_signal_precedence':['pharmacy_type','drugstore_type_only','explicit_name_or_url_only','recognized_brand_only','no_prespecified_signal'],
        'transport':'Sequential GET, 30s timeout, >=0.15s spacing, maximum three attempts per candidate per invocation, global Retry-After/exponential backoff; stop on 401/403 or three consecutive candidate failures; preserve every received response and resume successes.',
        'inputs':{p:sha(work/p) for p in ['input_place_ids.json','frames.json','search_provenance.json']},
        'implementation_sha256':sha(Path(__file__)),'website_validator_sha256':sha(ROOT/'code/wayback_snapshot_fetcher.py'),
        'scope_stop':'No Wayback, OpenAI, Board joins, classification, downstream recomputation, or thesis editing.'})
    print('Frozen: 836 new IDs; 60 primary ZIPs; unchanged 99 supply ZIPs; signal rules declared.')

class Transport:
    def __init__(self,key):self.key=key;self.opener=build_opener(NoRedirect())
    def get(self,pid):
        req=Request('https://places.googleapis.com/v1/places/'+pid,headers={'X-Goog-Api-Key':self.key,'X-Goog-FieldMask':MASK,'Accept':'application/json'})
        try:
            with self.opener.open(req,timeout=30) as r:return r.status,r.read(2_000_001),r.headers.get('Retry-After')
        except HTTPError as e:return e.code,e.read(2_000_001),e.headers.get('Retry-After')

def delay_seconds(value,attempt):
    delay=2**(attempt+1)
    if value:
        try:delay=max(delay,float(value))
        except ValueError:
            try:delay=max(delay,(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds())
            except (ValueError,TypeError):pass
    return delay

def acquire(pid,work,transport,sleep=time.sleep):
    assert re.fullmatch(r'[A-Za-z0-9_-]{10,256}',pid)
    folder=work/'private/details'/pid;folder.mkdir(parents=True,exist_ok=True)
    completed=folder/'result.json'
    if completed.exists():
        r=json.loads(completed.read_text())
        if r['state']=='success':
            assert sha(folder/r['raw_file'])==r['response_sha256']
            validate(json.loads((folder/r['raw_file']).read_bytes()),pid)
            return r
    # Recover a saved valid response if interrupted before publishing its result.
    for p in sorted(folder.glob('*_attempt.json')):
        r=json.loads(p.read_text())
        if r['state']=='success':
            assert sha(folder/r['raw_file'])==r['response_sha256']
            validate(json.loads((folder/r['raw_file']).read_bytes()),pid);save(completed,r);return r
    for attempt in range(3):
        sleep(0.15);stamp=uuid.uuid4().hex;raw=b'';status=None;retry=None;reason=None;requested=now()
        try:
            status,raw,retry=transport.get(pid)
            if len(raw)>2_000_000:raise ValueError('response_size')
            if status!=200:reason=f'http_{status}'
            else:validate(json.loads(raw),pid)
        except Exception as e:reason='transport_or_schema_'+type(e).__name__
        rawfile=stamp+'_response.bin';(folder/rawfile).write_bytes(raw)
        record={'place_id':pid,'requested_at':requested,'retrieved_at':now(),'http_status':status,'state':'success' if reason is None else 'collection_failure','reason':reason,'raw_file':rawfile,'response_sha256':digest(raw),'field_mask':MASK,'retry_after':retry}
        save(folder/(stamp+'_attempt.json'),record)
        if reason is None:save(completed,record);return record
        if status in {401,403}:raise RuntimeError(f'provider_http_{status}')
        if status in {400,404}:break
        if attempt<2:
            delay=delay_seconds(retry,attempt)
            if delay>60:raise RuntimeError('provider_retry_after_over_60s_pause')
            sleep(delay)
    save(completed,record);return record

def check(work):
    p=json.loads((work/'protocol.json').read_text())
    assert sha(Path(__file__))==p['implementation_sha256']
    assert sha(ROOT/'code/wayback_snapshot_fetcher.py')==p['website_validator_sha256']
    for f,h in p['inputs'].items():assert sha(work/f)==h

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def report(work):
    frames=json.loads((work/'frames.json').read_text());ids=json.loads((work/'input_place_ids.json').read_text());rows=[];statuses=Counter()
    for pid in ids:
        folder=work/'private/details'/pid;p=folder/'result.json'
        if not p.exists():rows.append({'place_id':pid,'details_status':'pending','frame_group':'pending'});continue
        result=json.loads(p.read_text());statuses[str(result['http_status'])]+=1
        if result['state']!='success':rows.append({'place_id':pid,'details_status':'collection_failure','frame_group':'details_failure','reason':result['reason']});continue
        assert sha(folder/result['raw_file'])==result['response_sha256']
        details=validate(json.loads((folder/result['raw_file']).read_bytes()),pid)
        rows.append({**derive(details,frames),'retrieved_at':result['retrieved_at'],'response_sha256':result['response_sha256']})
    success=[r for r in rows if r['details_status']=='success']
    groups=[]
    for group in GROUPS:
        rs=[r for r in rows if r['frame_group']==group];valid=[r for r in rs if r['details_status']=='success']
        groups.append({'frame_group':group,'candidates':len(rs),'explicit_signal':sum(r['explicit_pharmacy_related_signal'] for r in valid),
            'brand_only':sum(r['signal_group']=='recognized_brand_only' for r in valid),'any_signal_including_brand':sum(r['any_signal_including_brand'] for r in valid),
            'operational':sum(r['business_status']=='OPERATIONAL' for r in valid),'usable_website':sum(r['website_state']=='usable' for r in valid),
            'explicit_operational':sum(r['explicit_pharmacy_related_signal'] and r['business_status']=='OPERATIONAL' for r in valid),
            'explicit_operational_usable_website':sum(r['explicit_pharmacy_related_signal'] and r['business_status']=='OPERATIONAL' and r['website_state']=='usable' for r in valid),
            'business_status_counts':dict(Counter(r['business_status'] for r in valid)),'website_counts':dict(Counter(r['website_state'] for r in valid)),
            'signal_counts':dict(Counter(r['signal_group'] for r in valid))})
    types=Counter(t for r in success for t in r['types'])
    summary={'at':now(),'total_candidates':len(ids),'details_success':len(success),'details_failures':sum(r['details_status']=='collection_failure' for r in rows),'pending':sum(r['details_status']=='pending' for r in rows),
        'last_http_statuses':dict(statuses),'complete':len(success)==len(ids),'frames':groups,'signal_counts':dict(Counter(r['signal_group'] for r in success)),
        'explicit_pharmacy_related_signal':sum(r['explicit_pharmacy_related_signal'] for r in success),'any_signal_including_brand':sum(r['any_signal_including_brand'] for r in success),
        'business_status_counts':dict(Counter(r['business_status'] for r in success)),'website_counts':dict(Counter(r['website_state'] for r in success)),
        'place_type_counts':dict(types.most_common()),'primary_type_counts':dict(Counter(r['primary_type'] or 'MISSING' for r in success).most_common()),
        'moved_place_id_count':sum(bool(r['moved_place_id']) for r in success),
        'stop':'Provider metadata screening only. No archive or model calls; no automatic inclusion/exclusion or promotion.'}
    save(work/'summary.json',summary);save(work/'candidate_screening.json',rows)
    columns=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(work/'candidate_screening.csv',[{k:r.get(k,'') for k in columns} for r in rows]);write_csv(work/'frame_summary.csv',groups)
    type_rows=[{'frame_group':g,'provider_type':t,'candidates':n} for g in GROUPS for t,n in Counter(t for r in success if r['frame_group']==g for t in r['types']).most_common()]
    if type_rows:write_csv(work/'types_by_frame.csv',type_rows)
    cross=[{'frame_group':g,'signal_group':s,'business_status':b,'website_state':w,'candidates':n} for (g,s,b,w),n in sorted(Counter((r['frame_group'],r['signal_group'],r['business_status'],r['website_state']) for r in success).items())]
    if cross:write_csv(work/'signal_status_website_by_frame.csv',cross)
    return summary

def run(work,limit=None):
    check(work)
    from dotenv import dotenv_values
    key=os.environ.get('GOOGLE_MAPS_API_KEY') or dotenv_values(ROOT/'.env').get('GOOGLE_MAPS_API_KEY')
    if not key:raise RuntimeError('GOOGLE_MAPS_API_KEY missing')
    transport=Transport(key);failures=0
    try:
        for i,pid in enumerate(json.loads((work/'input_place_ids.json').read_text()),1):
            r=acquire(pid,work,transport)
            failures=failures+1 if r['state']!='success' else 0
            if i%20==0 or r['state']!='success':print(json.dumps({'processed':i,'last_state':r['state']}),flush=True)
            if failures>=3:raise RuntimeError('three_consecutive_details_failures')
            if limit and i>=limit:break
    finally:report(work)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=WORK);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
    if a.prepare:prepare(a.work)
    if a.run:run(a.work,a.limit)
    if a.report:print(json.dumps(report(a.work),indent=2))
