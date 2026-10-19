"""Isolated, resumable ID-only search coverage diagnostic; never classifies.

Selection depends only on frozen search saturation, never Board/model outcomes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = ROOT/'analysis/maintenance/saturated_zip_grid_20260907_v1'
BASE = ROOT/'data/v2/20260902T012636Z'
ENDPOINT = 'https://places.googleapis.com/v1/places:searchText'
MASK = 'places.id,nextPageToken'
TERMS = {'pharmacy_places_new_v2':'pharmacy', 'drug_store_places_new_v2':'drug store', 'drugstore_places_new_v2':'drugstore'}

def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''): h.update(part)
    return h.hexdigest()
def digest(b): return hashlib.sha256(b).hexdigest()
def encoded(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(p)

def saturated(queries):
    return sorted([q for q in queries if sum(p['returned_place_id_count'] for p in q['pages'])>=60],key=lambda q:(q['source_zip'],q['query_definition_id']))

def split(bounds):
    west,south,east,north=bounds
    mx,my=(west+east)/2,(south+north)/2
    return [[west,south,mx,my],[mx,south,east,my],[west,my,mx,north],[mx,my,east,north]]

def request_body(job,token=None):
    w,s,e,n=job['bounds']
    body={'textQuery':job['text_query'],'pageSize':20,'locationRestriction':{'rectangle':{'low':{'latitude':s,'longitude':w},'high':{'latitude':n,'longitude':e}}}}
    if token is not None: body['pageToken']=token
    return body

def parse_page(raw):
    d=json.loads(raw)
    if not isinstance(d,dict) or not set(d)<= {'places','nextPageToken'}:raise ValueError('invalid_payload')
    places=d.get('places',[])
    if not isinstance(places,list) or len(places)>20:raise ValueError('invalid_places')
    ids=[]
    for x in places:
        if not isinstance(x,dict) or set(x)!={'id'} or not isinstance(x['id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{10,256}',x['id']):raise ValueError('invalid_place_id')
        ids.append(x['id'])
    token=d.get('nextPageToken')
    if token is not None and (not isinstance(token,str) or not token or len(token)>4096 or any(ord(c)<32 for c in token)):raise ValueError('invalid_token')
    return ids,token

def prepare(work):
    if (work/'protocol.json').exists(): raise RuntimeError('protocol already frozen')
    import geopandas as gpd
    import shapely
    from shapely.geometry import box
    import pyproj
    pyproj.network.set_network_enabled(False)
    ledger=json.loads((BASE/'search_provenance.json').read_text())
    chosen=saturated(ledger['queries'])
    ids=[json.loads(x)['place_id'] for x in (BASE/'place_ids.jsonl').read_text().splitlines() if x.strip()]
    zips=sorted({q['source_zip'] for q in chosen})
    assert len(ids)==len(set(ids))==1226 and len(chosen)==112 and len(zips)==59
    work.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/tl_2023_us_zcta520.zip'
    source_hash=sha(source)
    # Extract only expected shapefile members into a checked local directory.
    local=(work/'geometry_source').resolve();local.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for entry in archive.infolist():
            if Path(entry.filename).suffix.lower() not in {'.shp','.shx','.dbf','.prj','.cpg'}:continue
            target=(local/entry.filename).resolve()
            if target.parent!=local or not target.name.startswith('tl_2023_us_zcta520.'):raise ValueError('unexpected_archive_member')
            with archive.open(entry) as a,target.open('wb') as b:shutil.copyfileobj(a,b)
    shp=local/'tl_2023_us_zcta520.shp'
    where='ZCTA5CE20 IN ('+','.join("'"+z+"'" for z in zips)+')'
    frame=gpd.read_file(shp,engine='pyogrio',columns=['ZCTA5CE20'],where=where)
    assert frame.crs is not None and len(frame)==59 and not frame.ZCTA5CE20.duplicated().any()
    assert frame.geometry.notna().all() and (~frame.geometry.is_empty).all() and frame.geometry.is_valid.all()
    source_crs=str(frame.crs)
    frame=frame.to_crs('EPSG:4326')
    assert frame.geometry.is_valid.all()
    polygons=dict(zip(frame.ZCTA5CE20.astype(str),frame.geometry))
    jobs=[]
    for q in chosen:
        z,qid=q['source_zip'],q['query_definition_id']
        polygon=polygons[z]
        for i,bounds in enumerate(split(polygon.bounds)):
            if not polygon.intersects(box(*bounds)):continue
            job={'id':f'{z}_{qid}_c{i}','source_zip':z,'query_definition_id':qid,'text_query':f'{TERMS[qid]} in Minneapolis-St. Paul {z}','level':1,'bounds':bounds,'parent':None,'children':[]}
            for j,child in enumerate(split(bounds)):
                if polygon.intersects(box(*child)):
                    job['children'].append({'id':job['id']+f'_c{j}','source_zip':z,'query_definition_id':qid,'text_query':job['text_query'],'level':2,'bounds':child,'parent':job['id'],'children':[]})
            jobs.append(job)
    protected=[BASE/'search_provenance.json',BASE/'place_ids.jsonl']
    protected += [p for p in (ROOT/'docs/thesis').rglob('*') if p.is_file() and p.suffix in {'.tex','.bib','.docx','.pdf','.png'}]
    save(work/'protected_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in protected})
    save(work/'baseline_place_ids.json',sorted(ids))
    save(work/'jobs.json',jobs)
    save(work/'selected_queries.json',chosen)
    frame.to_file(work/'selected_zctas.gpkg',driver='GPKG',index=False)
    reopened=gpd.read_file(work/'selected_zctas.gpkg')
    assert len(reopened)==59 and set(reopened.ZCTA5CE20)==set(zips)
    protocol={'version':1,'frozen_at':now(),'authorization':'User: 那是不是得补做一下 只补做那些返回了60个id的zip把',
      'scope':'Supplemental candidate discovery only, limited to saturated ZIP/query combinations. No Details, Wayback, OpenAI, Board joins, thesis edits or automatic candidate-frame promotion.',
      'baseline_run':'20260902T012636Z','baseline_candidates':1226,'selected_queries':len(chosen),'selected_zips':zips,
      'selection':'Frozen query returned at least 60 IDs; no Board or classifier field is read.',
      'geometry':'2023 cached ZCTA polygon bounding boxes, EPSG:4326; split each into 2x2 equal-angular rectangles. Query only rectangles intersecting the ZCTA. Rectangles are not clipped polygons or equal-area parcels; border spillover is possible.',
      'refinement':'When a successful parent returns at least 60 IDs or reaches page_cap, split it into four precomputed children, once only (maximum 4x4 leaf resolution). Failure is never treated as an empty area.',
      'text_query':'Original keyword in Minneapolis-St. Paul ZIP, unchanged; rectangular locationRestriction added; no type/name/outcome filters.',
      'endpoint':ENDPOINT,'field_mask':MASK,'page_size':20,'max_pages':10,'max_attempts_per_page_per_invocation':3,'request_timeout_seconds':25,
      'global_policy':'Sequential requests, >=0.2s spacing; Retry-After or exponential backoff on 429/5xx; abort invocation on 401/403 or three consecutive failed queries.',
      'initial_cells':len(jobs),'maximum_child_cells':sum(len(x['children']) for x in jobs),
      'interpretation':'Post hoc targeted coverage diagnostic. Novel IDs are candidate records, not confirmed pharmacies or verified mailing-ZIP membership. Time/provider variability is confounded with the changed search geometry. No unsaturated-ZIP completeness claim.',
      'zcta_archive_sha256':source_hash,'source_crs':source_crs,'query_crs':'EPSG:4326','geometry_repair':'none',
      'versions':{'geopandas':gpd.__version__,'shapely':shapely.__version__,'pyproj':pyproj.__version__},
      'inputs':{x:sha(work/x) for x in ['baseline_place_ids.json','jobs.json','selected_queries.json','selected_zctas.gpkg']},
      'implementation_sha256':sha(Path(__file__))}
    save(work/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ['selected_queries','initial_cells','maximum_child_cells']},indent=2))

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class LiveTransport:
    def __init__(self,key):self.key=key;self.opener=build_opener(NoRedirect())
    def post(self,body):
        req=Request(ENDPOINT,data=encoded(body),headers={'X-Goog-Api-Key':self.key,'X-Goog-FieldMask':MASK,'Content-Type':'application/json'},method='POST')
        try:
            with self.opener.open(req,timeout=25) as response:return response.status,response.read(2_000_001),response.headers.get('Retry-After')
        except HTTPError as e:return e.code,e.read(2_000_001),e.headers.get('Retry-After')

def collect(job,work,transport,sleep=time.sleep):
    folder=work/'queries'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    token=None;seen=set();all_ids=[];pages=[]
    for number in range(1,11):
        body=request_body(job,token);bh=digest(encoded(body));cache=folder/f'page_{number:02d}.json'
        if cache.exists():
            page=json.loads(cache.read_text())
            assert page['request_sha256']==bh
            raw_path=folder/page['raw_file'];assert sha(raw_path)==page['raw_sha256']
            ids,next_token=parse_page(raw_path.read_bytes())
        else:
            reason='unknown';status=None
            for attempt in range(3):
                sleep(0.2)
                stamp=uuid.uuid4().hex
                save(folder/(stamp+'_request.json'),body)
                raw=b'';retry=None;status=None
                try:
                    status,raw,retry=transport.post(body)
                    if len(raw)>2_000_000:raise ValueError('response_too_large')
                    reason=f'http_{status}'
                    if status==200:ids,next_token=parse_page(raw);reason=None
                except Exception as e: reason='transport_or_parse_'+type(e).__name__
                raw_file=stamp+'_response.bin';(folder/raw_file).write_bytes(raw)
                meta={'retrieved_at':now(),'request_sha256':bh,'status':status,'reason':reason,'raw_file':raw_file,'raw_sha256':digest(raw),'retry_after':retry,'page_number':number}
                save(folder/(stamp+'_attempt.json'),meta)
                if reason is None:
                    page={**meta,'ids':ids,'next_token':next_token};save(cache,page);break
                if status in {401,403}:raise RuntimeError(f'provider_http_{status}')
                if attempt<2:
                    try:delay=max(2**(attempt+1),float(retry or 0))
                    except ValueError:delay=2**(attempt+1)
                    if delay>60:raise RuntimeError('provider_retry_after_exceeds_invocation_wait')
                    sleep(delay)
            else:
                result={'job':job,'state':'collection_failure','reason':reason,'pages':pages,'ids':sorted(set(all_ids)),'finished_at':now()}
                save(folder/'result.json',result);return result
        pages.append({'page_number':number,'count':len(ids),'request_sha256':bh,'response_sha256':page['raw_sha256'],'next_token_present':next_token is not None})
        all_ids.extend(ids)
        if next_token is None:state='exhausted';break
        if next_token in seen:state='collection_failure';reason='cyclic_token';break
        seen.add(next_token);token=next_token
    else:state='page_cap'
    result={'job':job,'state':state,'reason':reason if state=='collection_failure' else None,'pages':pages,'ids':sorted(set(all_ids)),'returned_count':len(all_ids),'saturated':len(all_ids)>=60 or state=='page_cap','finished_at':now()}
    save(folder/'result.json',result);return result

def needs_children(result):return result['state'] in {'exhausted','page_cap'} and result.get('saturated',False)

def report(work):
    jobs=json.loads((work/'jobs.json').read_text());baseline=set(json.loads((work/'baseline_place_ids.json').read_text()))
    expected=list(jobs);results=[];pending=[]
    for job in expected:
        p=work/'queries'/job['id']/'result.json'
        if not p.exists():pending.append(job['id']);continue
        r=json.loads(p.read_text());results.append(r)
        if job['level']==1 and needs_children(r):expected.extend(job['children'])
    found=set().union(*(set(r['ids']) for r in results)) if results else set()
    new=found-baseline
    by_zip=[]
    for z in sorted({j['source_zip'] for j in jobs}):
        rs=[r for r in results if r['job']['source_zip']==z]
        zi=set().union(*(set(r['ids']) for r in rs)) if rs else set()
        by_zip.append({'source_zip':z,'queries_finished':len(rs),'unique_ids':len(zi),'novel_vs_baseline':len(zi-baseline),'failures':sum(r['state']=='collection_failure' for r in rs)})
    summary={'at':now(),'selected_zip_count':59,'selected_original_query_count':112,'baseline_candidates':len(baseline),
        'expected_cells':len(expected),'finished_cells':len(results),'pending_cells':len(pending),'query_states':dict(Counter(r['state'] for r in results)),
        'saturated_parent_cells':sum(r['job']['level']==1 and needs_children(r) for r in results),
        'saturated_leaf_cells':sum(r['job']['level']==2 and needs_children(r) for r in results),
        'returned_occurrences':sum(sum(p['count'] for p in r['pages']) for r in results),'unique_returned_ids':len(found),
        'overlap_with_baseline':len(found&baseline),'novel_place_ids':len(new),'union_candidate_ids':len(baseline|found),
        'complete':not pending and all(r['state']!='collection_failure' for r in results),
        'limitations':['Candidate IDs only, not verified pharmacies or mailing-ZIP membership.','Targeted saturated-query subset; no exhaustive-coverage claim.','Later retrieval date and changed geographic restriction both contribute to differences.'],
        'by_zip':by_zip}
    save(work/'summary.json',summary);save(work/'novel_place_ids.json',sorted(new));save(work/'all_returned_place_ids.json',sorted(found))
    save(work/'pending_cells.json',pending)
    provenance={pid:[] for pid in sorted(new)}
    for r in results:
        for pid in set(r['ids'])&new:provenance[pid].append({'query_job':r['job']['id'],'source_zip':r['job']['source_zip'],'level':r['job']['level'],'state':r['state']})
    save(work/'novel_id_provenance.json',provenance)
    return summary

def run(work,limit=None):
    from dotenv import dotenv_values
    protocol=json.loads((work/'protocol.json').read_text())
    assert sha(Path(__file__))==protocol['implementation_sha256'],'implementation changed after freeze'
    for p,h in protocol['inputs'].items():assert sha(work/p)==h
    key=os.environ.get('GOOGLE_MAPS_API_KEY') or dotenv_values(ROOT/'.env').get('GOOGLE_MAPS_API_KEY')
    if not key:raise RuntimeError('GOOGLE_MAPS_API_KEY missing')
    transport=LiveTransport(key)
    jobs=json.loads((work/'jobs.json').read_text());done=0;consecutive=0
    try:
        for job in jobs:
            p=work/'queries'/job['id']/'result.json'
            old=json.loads(p.read_text()) if p.exists() else None
            if old and old['state'] in {'exhausted','page_cap'}:r=old
            else:
                r=collect(job,work,transport);done+=1
                consecutive=consecutive+1 if r['state']=='collection_failure' else 0
                print(json.dumps({'cells_processed':done,'zip':job['source_zip'],'level':job['level'],'state':r['state'],'ids':len(r['ids'])}),flush=True)
                if consecutive>=3:raise RuntimeError('three_consecutive_query_failures')
            if job['level']==1 and needs_children(r):jobs.extend(job['children'])
            if limit and done>=limit:break
    finally:report(work)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=DEFAULT_WORK);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.prepare:prepare(a.work)
    if a.run:run(a.work,a.limit)
    if a.report:print(json.dumps(report(a.work),indent=2))
