"""Same-capture retrieval comparison only. No full-run or classification entrypoint."""
from pathlib import Path
from collections import Counter
import argparse
import json
import os
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from corrected_run_store import read_sealed,seal_json,file_sha,sha,unit_key,utcnow,run_resumable_queue,atomic_json
from archive_exact_page import verify_official_snapshot,ExactSnapshotOutscraper,ArchivePageRejected,validate_retained_source
from archive_raw_retrieval import RawSnapshotTransport,ProxyNotConfigured,assess_chain
from wayback_state import WaybackStateStore,WaybackDeferred
from run_archive_transport_repair import verify_stage
from outscraper_wayback import load_api_key

BASELINE=ROOT/'data/v2/archive_transport/20260905T222825Z'
DEFAULT=ROOT/'data/v2/archive_retrieval/20260905T230718Z'
ROUTES=('direct','proxy','outscraper')


def prepare(root):
    if (root/'plan.json').exists():
        return read_sealed(root/'plan.json')
    baseline=verify_stage(BASELINE,'diagnostic')
    old_plan=read_sealed(BASELINE/'plan.json')
    records={read_sealed(p)['place_id']:read_sealed(p) for p in (BASELINE/'diagnostic/evidence').glob('*.json')}
    rows=[]
    for selected in old_plan['diagnostic_sample']:
        record=records[selected['place_id']]
        identities={sha(p['official_snapshot']):p['official_snapshot'] for p in record['pages']}
        if len(identities)>1:
            raise ValueError('baseline candidate has multiple selected snapshots')
        snapshot=next(iter(identities.values()),None)
        if snapshot and snapshot['method']!='availability':
            raise ValueError('expected Availability-confirmed snapshot')
        rows.append({**selected,'snapshot':snapshot,'baseline_outcome':record['outcome'],
                     'baseline_evidence_sha256':file_sha(BASELINE/'diagnostic/evidence'/f'{unit_key(record["place_id"])}.json'),
                     'availability_queries':record['discoveries']['availability']['queries']})
    if len(rows)!=40 or sum(bool(r['snapshot']) for r in rows)!=31:
        raise ValueError('baseline diagnostic frame changed')
    protected=read_sealed(BASELINE/'preserved_before.json')['files']
    # Recommit the actual approved current versions, including its completed tests.
    protected={p:file_sha(ROOT/p) for p in protected}
    for parent in (BASELINE,ROOT/'analysis/maintenance/archive_transport_repair_20260905T222825Z'):
        for path in parent.rglob('*'):
            if path.is_file():protected[path.relative_to(ROOT).as_posix()]=file_sha(path)
    seal_json(root/'preserved_before.json',{'files':protected,'created_at':utcnow()})
    for row in rows:
        for query in row['availability_queries']:
            digest=query['provenance_sha256']
            relative=Path('responses')/digest[:2]/(digest+'.json')
            source=BASELINE/'private/diagnostic_state'/relative
            target=root/'private/provider_state'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    plan={'schema':'nets-ai-fixed-snapshot-retrieval-plan-v1','created_at':utcnow(),
          'run_id':root.name,'baseline_directory':BASELINE.as_posix(),
          'baseline_manifest_sha256':file_sha(BASELINE/'diagnostic/manifest.json'),
          'sample_seed':old_plan['sample_seed'],'rows':rows,'sample_count':40,'confirmed_snapshot_count':31,
          'availability_schedule':old_plan['availability_schedule'],
          'discovery_policy':'Reuse frozen official Availability receipts and exact baseline selected identities. No new CDX, TimeMap, or candidate-specific snapshot search. Limited negatives remain unresolved.',
          'routes':list(ROUTES),'route_order':'Each candidate independently requests direct, proxy, then Outscraper; pending routes do not block other configured routes.',
          'proxy_contract':'Explicit HTTP(S) CONNECT proxy; end-to-end verified TLS, no AI, no rendering; WAYBACK_RAW_PROXY_URL and optional WAYBACK_RAW_PROXY_PROVIDER.',
          'bounds':{'timeout_seconds':20,'max_body_bytes':2000000,'max_redirects':5,'attempts_per_hop':1,'minimum_interval_seconds':1,'implicit_429_backoff_seconds':5,'retry_after_honored':True},
          'comparison_gate':'All three routes completed for all 31 confirmed snapshots; usable page content in at least 16/31, no accepted binding/provenance violation. Diagnostic review only, never automatic full acquisition.',
          'no_snapshot_rows':'Nine baseline collection_failure rows remain identified as carried-forward observations, not newly attempted page requests.',
          'historical_rules_unchanged':True,'snapshot_metadata_establishes_opening_or_active_by_year':False,
          'full_acquisition_authorized':False,'phase2_authorized':False,'openai_calls_allowed':False,'google_calls_allowed':False}
    seal_json(root/'plan.json',plan)
    source=Path('C:/Users/16067/.codex/attachments/ba5997e4-0f87-441c-b21d-3634df7eebd2/pasted-text.txt')
    shutil.copyfile(source,root/'user_instruction.txt')
    return plan


def config(name):
    # Only retrieve the explicitly named transport setting. Never echo values.
    if os.environ.get(name):return os.environ[name]
    from dotenv import dotenv_values
    return dotenv_values(ROOT/'.env',interpolate=False).get(name)


def validate_route(root,plan,row,route,value):
    scope=plan['run_id']+':exact_retrieval'
    store=WaybackStateStore(root/'private/provider_state')
    verify_official_snapshot(store,row['website'],row['snapshot'])
    if value.get('official_snapshot')!=row['snapshot']:
        raise ValueError('retained snapshot selection changed')
    if route in {'direct','proxy'}:
        paths=sorted((root/'raw'/route/unit_key(row['place_id'])).glob('*.json'),key=lambda p:int(p.stem))
        records=[read_sealed(p) for p in paths]
        regenerated=assess_chain(records,row['website'],row['snapshot'],route,value['provider'],scope,row['place_id'])
        if value!=regenerated:raise ValueError('raw page result no longer reproduces')
    elif value['status']=='success':
        validate_retained_source(store,row['website'],value,scope=scope)
    elif value.get('route')!='outscraper' or not value.get('reason'):
        raise ValueError('invalid Outscraper failure record')


def execute(root,routes):
    plan=read_sealed(root/'plan.json')
    if not set(routes)<=set(ROUTES):raise ValueError('unsupported route')
    if file_sha(BASELINE/'diagnostic/manifest.json')!=plan['baseline_manifest_sha256']:
        raise ValueError('baseline manifest changed')
    implementation=read_sealed(root/'implementation.json')
    if any(file_sha(ROOT/p)!=h for p,h in implementation['files'].items()):
        raise ValueError('committed implementation changed')
    scope=plan['run_id']+':exact_retrieval'
    store=WaybackStateStore(root/'private/provider_state')
    backends={}
    for route in routes:
        if route=='outscraper':
            backends[route]=ExactSnapshotOutscraper(store,load_api_key(ROOT/'.env'),scope)
        else:
            backends[route]=RawSnapshotTransport(root,store,scope,route,
                proxy_url=config('WAYBACK_RAW_PROXY_URL') if route=='proxy' else None,
                provider=config('WAYBACK_RAW_PROXY_PROVIDER') or 'http_connect')
    def collect(row):
        deadlines=[]
        for route in routes:
            path=root/'results'/route/(unit_key(row['place_id'])+'.json')
            if path.exists():
                validate_route(root,plan,row,route,read_sealed(path));continue
            try:
                if route=='outscraper':
                    value=backends[route].fetch(row['website'],row['snapshot'])
                    value.update(route=route,official_snapshot=row['snapshot'])
                else:value=backends[route].fetch(row['place_id'],row['website'],row['snapshot'])
            except WaybackDeferred as exc:
                deadlines.append(exc.retry_at or time.time()+2);continue
            except ArchivePageRejected as exc:
                if route!='outscraper':raise
                value={'route':route,'official_snapshot':row['snapshot'],'status':'failure','reason':exc.reason}
            validate_route(root,plan,row,route,value)
            seal_json(path,value)
        if deadlines:raise WaybackDeferred('exact_retrieval_pending',min(deadlines))
        return {'place_id':row['place_id'],'status':'complete'}
    last=[0]
    def event(row,value):
        if value['status']=='complete' or time.time()-last[0]>45:
            last[0]=time.time()
            print(json.dumps({'event':value,'routes':routes,'updated_at':utcnow()}),flush=True)
    run_resumable_queue([r for r in plan['rows'] if r['snapshot']],collect,max_active=8,on_event=event)
    return report(root)


def report(root):
    plan=read_sealed(root/'plan.json')
    counts={route:{'tested':0,'success':0,'failures':Counter()} for route in ROUTES}
    overlap=Counter();outcomes=Counter();confirmed=[]
    for row in plan['rows']:
        if not row['snapshot']:continue
        results={}
        for route in ROUTES:
            path=root/'results'/route/(unit_key(row['place_id'])+'.json')
            if path.exists():
                value=read_sealed(path);validate_route(root,plan,row,route,value)
                results[route]=value;counts[route]['tested']+=1
                if value['status']=='success':counts[route]['success']+=1
                else:counts[route]['failures'][value['reason']]+=1
        success={k for k,v in results.items() if v['status']=='success'}
        if len(results)==3:
            key='both' if {'proxy','outscraper'}<=success else 'proxy_only' if 'proxy' in success else 'outscraper_only' if 'outscraper' in success else 'neither'
            overlap[key]+=1
            outcome=next((state for route,state in [('direct','direct_raw'),('proxy','proxy_raw'),('outscraper','outscraper_extracted')] if route in success),'structural_only')
            outcomes[outcome]+=1
        confirmed.append({'place_id':row['place_id'],'tested_routes':sorted(results),'successful_routes':sorted(success)})
    complete=all(c['tested']==31 for c in counts.values())
    available=sum(bool(r['successful_routes']) for r in confirmed)
    result={'run_id':plan['run_id'],'sample_candidates':40,'confirmed_snapshots':31,'complete':complete,
            'routes':counts,'combined_substantive_observed':available,'combined_rate':available/31 if complete else None,
            'overlap':{k:overlap[k] for k in ('both','proxy_only','outscraper_only','neither')} if complete else None,
            'confirmed_snapshot_outcomes':dict(outcomes) if complete else None,
            'carried_forward_baseline_collection_failure':9,
            'diagnostic_ready_for_review':complete and available>=16,
            'full_acquisition_authorized':False,'phase2_authorized':False,'openai_calls':0,'google_calls':0,
            'candidates':confirmed}
    atomic_json(root/'progress_report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','run','report'))
    p.add_argument('--routes',nargs='+',choices=ROUTES,default=list(ROUTES))
    args=p.parse_args()
    if args.mode=='prepare':prepare(DEFAULT)
    elif args.mode=='run':execute(DEFAULT,args.routes)
    else:report(DEFAULT)
