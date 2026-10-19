"""Official discovery -> exact snapshot pages. No Google or OpenAI execution API."""
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import urlencode, urlsplit, urlunsplit, quote
import argparse
import json
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from archive_transport_plan import initialize_plan,diagnostic_gate,OUTCOMES
from archive_transport_http import OfficialHTTP,official_url
from archive_transport_binding import urls_bind,snapshot_identity
from archive_official_discovery import parse_cdx,parse_timemap,parse_availability
from archive_exact_page import validate_direct_page,verify_official_snapshot,ExactSnapshotOutscraper,ArchivePageRejected
from corrected_run_store import read_sealed,seal_json,atomic_json,file_sha,sha,unit_key,utcnow,run_resumable_queue
from wayback_state import WaybackStateStore,WaybackDeferred
from outscraper_wayback import load_api_key

CDX='https://web.archive.org/cdx/search/cdx'
AVAILABLE='https://archive.org/wayback/available'
TIMEMAP='https://web.archive.org/web/timemap/link/'


def timemap_transport_url(url):
    p=urlsplit(url)
    if p.scheme not in {'http','https'} or p.hostname!='web.archive.org' or p.port not in {None,80 if p.scheme=='http' else 443}:
        raise ValueError('invalid official TimeMap continuation')
    value=urlunsplit(('https','web.archive.org',p.path,p.query,p.fragment))
    official_url(value)
    return value


def _http_parse(response, parser, *args):
    if response['status']!=200:
        return {'status':'transport_failure','reason':f'http_{response["status"]}' if response['status'] else 'transport_error',
                'complete':False,'snapshots':[],'next_cursor':None}
    if response['final_url']!=response['request_url'] or response['metadata'].get('body_truncated'):
        return {'status':'transport_failure','reason':'redirect_or_truncated_response','complete':False,'snapshots':[],'next_cursor':None}
    return parser(response['body'],*args)


def discover(method, row, http, plan):
    website=row['website'];ident=row['place_id']
    queries=[];snapshots=[];complete=False;cdx_rows=0
    if method=='availability':
        for requested in plan['availability_schedule']:
            url=AVAILABLE+'?'+urlencode([('url',website),('timestamp',requested)])
            response=http.get(url,method=method,candidate_id=ident)
            result=_http_parse(response,parse_availability,website,requested)
            queries.append({'requested_timestamp':requested,'http_status':response['status'],
                            'provenance_sha256':response['provenance_sha256'],**result})
            snapshots.extend({**s,'method':method,'provenance_sha256':response['provenance_sha256'],
                              'requested_timestamp':requested}for s in result['snapshots'])
    else:
        cursor=None;queue=[TIMEMAP+website];visited=set()
        for number in range(plan['transport']['max_cdx_pages'] if method=='cdx' else plan['transport']['max_timemap_pages']):
            if method=='cdx':
                params=[('url',website),('output','json'),('matchType','exact'),('filter','statuscode:200'),
                        ('filter','mimetype:text/html'),('fl','timestamp,original,mimetype,statuscode,digest,length'),
                        ('limit','100'),('showResumeKey','true')]
                if cursor:params.append(('resumeKey',cursor))
                url=CDX+'?'+urlencode(params)
            else:
                if not queue:complete=True;break
                url=timemap_transport_url(queue.pop(0))
                official_url(url)
            if url in visited:
                queries.append({'status':'parser_failure','reason':'pagination_cycle','complete':False,'snapshots':[]})
                break
            visited.add(url)
            response=http.get(url,method=method,candidate_id=ident)
            result=_http_parse(response,parse_cdx if method=='cdx' else parse_timemap,website)
            if method=='timemap' and result.get('self_url') and timemap_transport_url(result['self_url'])!=response['request_url']:
                result={'status':'binding_failure','reason':'timemap_self_url_mismatch','complete':False,'snapshots':[],'next_cursor':None}
            queries.append({'http_status':response['status'],'provenance_sha256':response['provenance_sha256'],**result})
            remaining=plan['transport']['max_cdx_rows']-cdx_rows
            page_snapshots=[s for s in result['snapshots'] if s.get('row_number',1)<=remaining] if method=='cdx' else result['snapshots']
            snapshots.extend({**s,'method':method,'provenance_sha256':response['provenance_sha256']}for s in page_snapshots)
            if result['status'] not in ('success','empty'):break
            if method=='cdx':
                cursor=result.get('next_cursor')
                cdx_rows+=result.get('returned_rows',len(result['snapshots']))
                if cdx_rows>plan['transport']['max_cdx_rows']:break
                if result['complete'] and not cursor:complete=True;break
                if not cursor:break
                if cdx_rows>=plan['transport']['max_cdx_rows']:break
            else:
                cyclic=False
                for link in result.get('next_cursors',([result['next_cursor']] if result.get('next_cursor') else [])):
                    link=timemap_transport_url(link)
                    if link in visited:cyclic=True
                    elif link not in queue:queue.append(link)
                if cyclic:
                    queries.append({'status':'parser_failure','reason':'pagination_cycle','complete':False,'snapshots':[]})
                    break
                if not queue:complete=True;break
    # Preserve nearest responses even when outside the eligibility time boundary.
    cutoff=''.join(c for c in plan['created_at'].split('.')[0] if c.isdigit())[:14]
    eligible=[s for s in snapshots if s['timestamp']<=cutoff]
    if len(eligible)!=len(snapshots):complete=False
    return {'method':method,'queries':queries,'snapshots':eligible,'complete':complete,
            'status':'success' if eligible else 'empty' if complete else 'collection_failure',
            'schedule_completed':all(q['status'] in ('success','empty') for q in queries) if method=='availability' else None,
            'future_snapshots_rejected':len(snapshots)-len(eligible)}


def select_snapshots(discoveries, mode):
    unique={}
    for method in ('cdx','timemap','availability'):
        for row in discoveries.get(method,{}).get('snapshots',[]):
            key=(row['timestamp'],row['exact_url'])
            unique.setdefault(key,row)
    rows=sorted(unique.values(),key=lambda s:(s['timestamp'],s['exact_url']))
    if mode=='diagnostic':return rows[-1:]  # Fixed transport test, identical source for both routes.
    grouped=defaultdict(list)
    for row in rows:
        if len(grouped[row['timestamp'][:4]])<3:grouped[row['timestamp'][:4]].append(row)
    return [row for year in sorted(grouped) for row in grouped[year]]


def outcome_for(discoveries,pages):
    sources=[p for p in pages if p['status']=='success']
    if any(s['source_type']=='direct_raw' for s in sources):return 'direct_raw'
    if sources:return 'outscraper_extracted'
    if any(d['snapshots'] for d in discoveries.values()):return 'structural_only'
    if any(d['complete'] and d['status']=='empty' for d in discoveries.values()):return 'no_archive'
    return 'collection_failure'


def checkpoint_save(path,state):
    atomic_json(path,{'payload':state,'sha256':sha(state)})


def validate_record(record,row,store,plan,mode):
    if record['place_id']!=row['place_id'] or record['input_details_sha256']!=row['details_sha256'] or record['website']!=row['website']:
        raise ValueError('archive evidence candidate binding differs')
    if row['prior_outcome']=='no_website':
        if record['outcome']!='no_website' or record['discoveries'] or record['pages']:
            raise ValueError('missing URL evidence changed')
        return
    from archive_exact_page import validate_retained_source
    for method,discovery in record['discoveries'].items():
        receipts={}
        for query in discovery['queries']:
            digest=query.get('provenance_sha256')
            if not digest:continue
            raw=store.read_record(digest)
            if (raw['provider']!='direct_wayback' or raw['metadata'].get('scope')!=plan['run_id']+'/'+mode
                    or raw['metadata'].get('candidate_id')!=row['place_id'] or raw['metadata'].get('method')!=method):
                raise ValueError('official discovery raw scope differs')
            receipts[raw['request_url']]=OfficialHTTP._response(raw,digest)
        class Replay:
            def get(self,url,**kwargs):
                return receipts[quote(url,safe=":/?&=%;,@!$'()*+-._~[]")]
        if discover(method,row,Replay(),plan)!=discovery:
            raise ValueError('retained official discovery differs from raw responses')
    sources=[]
    for page in record['pages']:
        if page['status']=='success':
            validate_retained_source(store,row['website'],page,scope=plan['run_id']+'/'+mode)
            sources.append(page)
    if record['outcome']!=outcome_for(record['discoveries'],record['pages']) or record['sources']!=sources:
        raise ValueError('archive evidence outcome or sources differ')


class ArchiveRun:
    def __init__(self,root,mode,plan,*,http=None,outscraper=None):
        self.root,self.mode,self.plan=Path(root),mode,plan
        self.output=self.root/mode
        self.store=WaybackStateStore(self.root/'private'/f'{mode}_state')
        cfg=plan['transport']
        self.http=http or OfficialHTTP(self.store,plan['run_id']+'/'+mode,timeout=cfg['timeout_seconds'],
               max_body_bytes=cfg['max_body_bytes'],interval=cfg['min_interval_seconds'],backoff=cfg['implicit_429_cooldown_seconds'])
        self.outscraper=outscraper or ExactSnapshotOutscraper(self.store,load_api_key(ROOT/'.env'),scope=plan['run_id']+'/'+mode,
               max_job_age_seconds=cfg['max_outscraper_age_seconds'],max_jobs=cfg['max_outscraper_jobs'],
               max_total_response_bytes=10000000000,max_response_bytes=4000000)

    def collect(self,row):
        key=unit_key(row['place_id']);destination=self.output/'evidence'/(key+'.json')
        old=Path(self.plan['old_run_directory'])/row['details_relative_path']
        if file_sha(old)!=row['details_sha256']:raise ValueError('source Details changed')
        if destination.exists():
            result=read_sealed(destination)
            validate_record(result,row,self.store,self.plan,self.mode)
            return result
        state_path=self.output/'checkpoints'/(key+'.json')
        if state_path.exists():
            saved=json.loads(state_path.read_text(encoding='utf-8'))
            if saved['sha256']!=sha(saved['payload']):raise ValueError('archive checkpoint checksum differs')
            state=saved['payload']
        else:state={'discoveries':{},'pages':{}}
        if row['prior_outcome']=='no_website':
            result={'place_id':row['place_id'],'input_details_sha256':row['details_sha256'],'outcome':'no_website',
                    'website':row['website'],'discoveries':{},'pages':[],'sources':[],
                    'reason':'preserved_missing_or_invalid_website','completed_at':utcnow()}
            seal_json(destination,result);return result
        deferred=[]
        for method in ('cdx','timemap','availability'):
            if method in state['discoveries']:continue
            # Diagnostic independently exercises every discovery route. The full
            # attempt uses the declared fallback order and stops after successful discovery.
            if self.mode=='full' and any(d['snapshots'] or (d['complete'] and d['status']=='empty')for d in state['discoveries'].values()):break
            try:state['discoveries'][method]=discover(method,row,self.http,self.plan)
            except WaybackDeferred as exc:
                deferred.append(exc)
                if self.mode=='full':break
            checkpoint_save(state_path,state)
        if deferred:
            dates=[e.retry_at for e in deferred if e.retry_at is not None]
            raise WaybackDeferred('official_discovery_pending',min(dates) if dates else time.time()+1)
        selected=select_snapshots(state['discoveries'],self.mode)
        successful_years=set()
        for snapshot in selected:
            year=snapshot['timestamp'][:4]
            if self.mode=='full' and year in successful_years:continue
            verify_official_snapshot(self.store,row['website'],snapshot)
            snapkey=sha(snapshot)
            for route in ('direct','outscraper'):
                page_key=snapkey+'/'+route
                if page_key in state['pages']:
                    if state['pages'][page_key]['status']=='success':successful_years.add(year)
                    continue
                direct=state['pages'].get(snapkey+'/direct')
                if self.mode=='full' and route=='outscraper':
                    if direct and direct['status']=='success':continue
                    if direct and direct.get('reason') not in {'direct_replay_http_failure','archive_error_or_interstitial','direct_transport_failure'}:continue
                try:
                    if route=='direct':
                        response=self.http.get(snapshot['exact_url'],method='replay',candidate_id=row['place_id'])
                        try:page=validate_direct_page(response,row['website'],snapshot)
                        except ArchivePageRejected as exc:
                            page={'status':'failure','reason':exc.reason,'http_status':response['status'],
                                  'provenance_sha256':response['provenance_sha256']}
                    else:
                        try:page=self.outscraper.fetch(row['website'],snapshot)
                        except ArchivePageRejected as exc:page={'status':'failure','reason':exc.reason}
                except WaybackDeferred as exc:
                    deferred.append(exc)
                    if self.mode=='full':break
                    continue
                page.update(route=route,official_snapshot=snapshot)
                state['pages'][page_key]=page
                if page['status']=='success':successful_years.add(year)
                checkpoint_save(state_path,state)
            if deferred and self.mode=='full':break
        if deferred:
            dates=[e.retry_at for e in deferred if e.retry_at is not None]
            raise WaybackDeferred('exact_page_pending',min(dates) if dates else time.time()+2)
        pages=list(state['pages'].values());outcome=outcome_for(state['discoveries'],pages)
        result={'schema':'nets-ai-official-archive-evidence-v1','place_id':row['place_id'],'website':row['website'],
                'input_details_sha256':row['details_sha256'],'prior_outcome':row['prior_outcome'],
                'outcome':outcome,'discoveries':state['discoveries'],'pages':pages,
                'sources':[p for p in pages if p['status']=='success'],'completed_at':utcnow(),
                'reason':('no_official_snapshot_identified; limited Availability negatives are not exhaustive archive absence' if outcome=='collection_failure' else None),
                'accepted_source_binding_failures':0}
        validate_record(result,row,self.store,self.plan,self.mode)
        seal_json(destination,result)
        return result


def transport_report(records):
    matrix={}
    for method in ('cdx','timemap','availability'):
        observed=[r['discoveries'][method]for r in records if method in r['discoveries']]
        matrix[method]={'candidates_tested':len(observed),'discovery_success':sum(bool(d['snapshots'])for d in observed),
             'lookup_complete':sum(d['complete']for d in observed),
             'http_response_counts':dict(Counter(str(q.get('http_status'))for d in observed for q in d['queries'])),
             'parse_status_counts':dict(Counter(q['status']for d in observed for q in d['queries']))}
    for route in ('direct','outscraper'):
        pages=[(r,p)for r in records for p in r['pages']if p['route']==route]
        matrix[route+'_page']={'candidates_tested':len({r['place_id']for r,p in pages}),
             'successful_candidates':len({r['place_id']for r,p in pages if p['status']=='success'}),
             'successful_sources':sum(p['status']=='success'for _,p in pages),
             'failure_reasons':dict(Counter(p['reason']for _,p in pages if p['status']!='success'))}
    binding=sum(q['status']=='binding_failure'for r in records for d in r['discoveries'].values()for q in d['queries'])
    binding+=sum(('binding' in p.get('reason','') or 'capture_changed' in p.get('reason','') or 'target_mismatch' in p.get('reason',''))for r in records for p in r['pages'])
    identities=sum(any(d['snapshots']for d in r['discoveries'].values())for r in records)
    return {'candidate_count':len(records),'wayback_outcomes':{key:sum(r['outcome']==key for r in records)for key in OUTCOMES},
            'transport_matrix':matrix,'exact_snapshot_identity_candidates':identities,'snapshot_identity_confirmation_candidates':identities,
            'url_binding_rejection_count':binding,'collection_failure_rate':sum(r['outcome']=='collection_failure'for r in records)/len(records),
            'openai_calls':0,'google_details_calls':0,'classification_allowed':False}


def verify_stage(root,mode):
    """Authenticate the exact sample/frame and receipts before resume or promotion."""
    root=Path(root);plan=read_sealed(root/'plan.json')
    manifest=read_sealed(root/mode/'manifest.json')
    report=read_sealed(root/mode/'preflight.json')
    expected=plan['diagnostic_sample'] if mode=='diagnostic' else plan['rows']
    if (manifest['run_id']!=plan['run_id'] or manifest['mode']!=mode or
            manifest['plan_sha256']!=file_sha(root/'plan.json') or
            manifest['preflight_sha256']!=file_sha(root/mode/'preflight.json')):
        raise ValueError('archive stage manifest scope differs')
    paths={f'{mode}/evidence/{unit_key(r["place_id"])}.json'for r in expected}
    if set(manifest['evidence_files'])!=paths:raise ValueError('archive stage frame differs')
    records=[];by_id={r['place_id']:r for r in expected}
    store=WaybackStateStore(root/'private'/f'{mode}_state')
    for section in ('evidence_files','raw_response_files'):
        for relative,digest in manifest[section].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or file_sha(path)!=digest:raise ValueError('archive stage file hash differs')
            if section=='evidence_files':
                record=read_sealed(path)
                if record['place_id'] not in by_id or record['input_details_sha256']!=by_id[record['place_id']]['details_sha256']:
                    raise ValueError('archive stage Details binding differs')
                validate_record(record,by_id[record['place_id']],store,plan,mode)
                records.append(record)
    regenerated=transport_report(records)
    if any(report.get(k)!=v for k,v in regenerated.items()):raise ValueError('archive report no longer reconciles')
    if mode=='diagnostic' and report['full_archive_gate']!=diagnostic_gate(records):raise ValueError('diagnostic gate differs')
    if report.get('mode')!=mode or report.get('run_id')!=plan['run_id']:raise ValueError('archive report scope differs')
    return report


def execute(root,mode):
    plan=read_sealed(root/'plan.json')
    old_root=Path(plan['old_run_directory'])
    if file_sha(old_root/'evidence_manifest.json')!=plan['old_manifest_sha256']:
        raise ValueError('approved failed acquisition manifest changed')
    for row in plan['rows']:
        if file_sha(old_root/row['details_relative_path'])!=row['details_sha256']:
            raise ValueError('frozen September 5 Details changed before archive execution')
    if mode=='full':
        prior=verify_stage(root,'diagnostic')
        if not prior['full_archive_gate']['passed']:raise ValueError('40-candidate diagnostic does not authorize a full archive attempt')
    rows=plan['diagnostic_sample'] if mode=='diagnostic' else plan['rows']
    output=root/mode
    if (output/'manifest.json').exists():
        return verify_stage(root,mode)
    engine=ArchiveRun(root,mode,plan)
    last=[0]
    def event(item,value):
        if value.get('status')=='complete' or time.time()-last[0]>45:
            last[0]=time.time()
            record={'mode':mode,'event':value,'frozen':len(list((output/'evidence').glob('*.json'))),'total':len(rows),'updated_at':utcnow()}
            atomic_json(output/'progress.json',record);print(json.dumps(record),flush=True)
    results=run_resumable_queue(rows,engine.collect,max_active=12,on_event=event)
    report=transport_report(results)
    report.update(run_id=plan['run_id'],mode=mode,completed_at=utcnow(),full_archive_gate=diagnostic_gate(results) if mode=='diagnostic' else None)
    report['technical_quality_ready_for_review']=all(r['outcome'] not in {'collection_failure','structural_only'}for r in results if r['outcome']!='no_website')
    report['phase2_scientifically_ready']=mode=='full' and report['technical_quality_ready_for_review']
    report['phase2_authorized']=False
    if len(results)!=len(rows) or len({r['place_id']for r in results})!=len(rows):raise ValueError('incomplete archive frame')
    files={p.relative_to(root).as_posix():file_sha(p)for p in (output/'evidence').glob('*.json')}
    raw={p.relative_to(root).as_posix():file_sha(p)for p in (engine.store.root/'responses').glob('*/*.json')}
    for p in raw:engine.store.read_record(Path(p).stem)
    if (output/'preflight.json').exists():
        saved=read_sealed(output/'preflight.json')
        if any(saved.get(k)!=v for k,v in report.items() if k!='completed_at'):
            raise ValueError('interrupted archive freeze no longer reconciles')
        report=saved
    seal_json(output/'preflight.json',report)
    seal_json(output/'manifest.json',{'run_id':plan['run_id'],'mode':mode,'plan_sha256':file_sha(root/'plan.json'),
              'evidence_files':files,'raw_response_files':raw,'preflight_sha256':file_sha(output/'preflight.json')})
    return verify_stage(root,mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['diagnostic','full'])
    parser.add_argument('--run-id',default='20260905T222825Z')
    args=parser.parse_args()
    if not __import__('re').fullmatch(r'\d{8}T\d{6}Z',args.run_id):parser.error('invalid run ID')
    root=ROOT/'data/v2/archive_transport'/args.run_id
    print(json.dumps(execute(root,args.mode),indent=2))


if __name__=='__main__':main()
