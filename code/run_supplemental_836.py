"""Isolated supplemental candidate run; frozen scientific modules are reused unchanged."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse, json, os, time, sys, threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urllib.request import Request, build_opener
from urllib.error import HTTPError
from wayback_snapshot_fetcher import encoded, digest, read_json, write_json, write, now, website_key, Crawler, HTTPTransport
from clean_archive_discovery import SCHEDULE, Discovery, AvailabilityHTTP
from clean_archive_closure_rules import revalidate
from clean_phase2_evidence import DETAIL_KEYS
from corrected_run_store import sanitize_details
from diagnose_saturated_zip_search import NoRedirect

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'data/supplemental_836/20260907_v1'
SCREEN=REPO/'analysis/maintenance/supplemental_details_836_20260907_v1'
OLD=REPO/'data/clean_classification/20260907T192546Z'
CLOSED=REPO/'data/clean_classification_closure/20260907_closure_v1'
OLDARCH=REPO/'data/clean_archive/20260906T040605Z'
MASK='id,nationalPhoneNumber,internationalPhoneNumber,regularOpeningHours,rating,userRatingCount,reviews'
DETAIL_LOCK=threading.Lock()
DETAIL_NEXT=0.0

def save(path,obj):
    write_json(path,obj,immutable=True)

def inventory(folder):
    return {str(p.relative_to(REPO)).replace('\\','/'):{'sha256':digest(p.read_bytes()),'mtime_ns':p.stat().st_mtime_ns}
            for p in sorted(folder.rglob('*')) if p.is_file()}

def prepare():
    if (ROOT/'protocol.json').exists():return
    ids=read_json(SCREEN/'input_place_ids.json')
    index=read_json(CLOSED/'final_classification_index.json')['rows']
    assert len(ids)==len(set(ids))==836
    assert len(index)==1226 and not set(ids)&{r['place_id'] for r in index}
    for r in index:assert digest(Path(r['result_file']).read_bytes())==r['result_file_sha256']
    protected={}
    for folder in (OLD,CLOSED,REPO/'data/clean_archive_closure/20260907_closure_v1',SCREEN,REPO/'docs/thesis'):
        protected.update(inventory(folder))
    # Retain the original raw acquisition seals without inspecting quarantined bodies.
    for name in ('plan.json','acquisition_freeze.json','raw_postfetch_manifest.json'):
        p=OLDARCH/name;protected[p.relative_to(REPO).as_posix()]={'sha256':digest(p.read_bytes()),'mtime_ns':p.stat().st_mtime_ns}
    save(ROOT/'preservation_before.json',protected)
    save(ROOT/'input_place_ids.json',ids)
    save(ROOT/'frames.json',read_json(SCREEN/'frames.json'))
    save(ROOT/'old_final_index.json',index)
    pins={p.relative_to(REPO).as_posix():digest(p.read_bytes()) for p in (REPO/'code').glob('*.py')
          if p.name.startswith(('clean_archive','clean_phase2','forced_choice','wayback_snapshot','run_clean_phase2'))}
    save(ROOT/'contract_pins.json',pins)
    save(ROOT/'protocol.json',{'schema':'supplemental-836-v1','created_at':now(),'candidate_count':836,
        'candidate_selection':'All supplemental IDs; no signal, brand, ZIP, Board or classification selection.',
        'schedule':list(SCHEDULE),'snapshot_cutoff':read_json(OLDARCH/'plan.json')['snapshot_cutoff'],
        'details_reuse':'Preserve all screening responses; request only previously unrequested classifier fields, retaining separate times and hashes.',
        'augmentation_field_mask':MASK,'model':'gpt-5-mini-2025-08-07',
        'classification':'Original full-evidence prompt/schema first; frozen closure validator; exact-ID closure retry only on technical failure.',
        'discovery':{'timeout_seconds':20,'max_attempts':3,'minimum_interval_seconds':0.1},
        'fetch':{'concurrency':1,'timeout':20,'max_attempts':3,'backoff_base':30,'backoff_cap':300},
        'rq2_rq3_allowed':False,'thesis_edits_allowed':False})
    print('Prepared all 836 IDs; preserved files:',len(protected),flush=True)

def check_pins():
    pins=read_json(ROOT/'contract_pins.json')
    if (ROOT/'historical_contract_pins.json').exists():pins={**pins,**read_json(ROOT/'historical_contract_pins.json')}
    for p,h in pins.items():
        assert digest((REPO/p).read_bytes())==h,p

def details_one(pid,key):
    global DETAIL_NEXT
    folder=ROOT/'private/augmentation'/pid
    result_path=folder/'result.json'
    if result_path.exists():return read_json(result_path)
    opener=build_opener(NoRedirect())
    for i in range(1,7):
        receipt=folder/f'{i}.json';rawfile=folder/f'{i}.bin'
        if receipt.exists():
            record=read_json(receipt);body=rawfile.read_bytes()
            assert digest(body)==record['sha256']
        else:
            while True:
                with DETAIL_LOCK:
                    delay=DETAIL_NEXT-time.time()
                    if delay<=0:
                        DETAIL_NEXT=time.time()+0.3
                        break
                time.sleep(min(delay,5))
            req=Request('https://places.googleapis.com/v1/places/'+pid,headers={
                'X-Goog-Api-Key':key,'X-Goog-FieldMask':MASK,'Accept':'application/json'})
            started=now();status=None;body=b'';error=None;retry=None
            try:
                with opener.open(req,timeout=30) as r:status=r.status;body=r.read(2_000_001)
            except HTTPError as e:status=e.code;body=e.read(2_000_001);retry=e.headers.get('Retry-After')
            except OSError as e:error=type(e).__name__
            record={'place_id':pid,'requested_at':started,'received_at':now(),'http_status':status,
                    'sha256':digest(body),'field_mask':MASK,'error':error,'retry_after':retry}
            write(rawfile,body,immutable=True);save(receipt,record)
            if status==429:
                from wayback_snapshot_fetcher import retry_after
                with DETAIL_LOCK:DETAIL_NEXT=max(DETAIL_NEXT,time.time()+max(60,retry_after(retry,time.time()) or 0))
        if record['http_status']==200 and not record['error']:
            value=json.loads(body)
            assert len(body)<=2_000_000 and value['id']==pid
            assert not set(value)-set(MASK.split(','))
            assert len(value.get('reviews',[]))<=5
            save(result_path,{**record,'raw_file':rawfile.relative_to(ROOT).as_posix()})
            return read_json(result_path)
        if record['http_status'] in (400,401,403,404):raise RuntimeError('Details HTTP '+str(record['http_status']))
        time.sleep(2**i)
    raise RuntimeError('Details augmentation unresolved '+pid)

def details():
    from dotenv import dotenv_values
    check_pins()
    cfg=dotenv_values(REPO/'.env',interpolate=False)
    key=os.environ.get('GOOGLE_MAPS_API_KEY') or cfg.get('GOOGLE_MAPS_API_KEY') or cfg.get('GOOGLE_API_KEY')
    if not key:raise RuntimeError('Missing Google credential')
    ids=read_json(ROOT/'input_place_ids.json')
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(details_one,pid,key):pid for pid in ids}
        for n,f in enumerate(as_completed(futures),1):
            f.result()
            if n%25==0:print('Details augmentation',n,'/836',flush=True)
    rows=[]
    for pid in ids:
        basefolder=SCREEN/'private/details'/pid;base=read_json(basefolder/'result.json')
        bpath=basefolder/base['raw_file'];body=bpath.read_bytes();assert digest(body)==base['response_sha256']
        aug=read_json(ROOT/'private/augmentation'/pid/'result.json');apath=ROOT/aug['raw_file']
        assert digest(apath.read_bytes())==aug['sha256']
        original=json.loads(body);extra=json.loads(apath.read_bytes())
        assert original['id']==extra['id']==pid
        merged=sanitize_details({**{k:v for k,v in original.items() if k in DETAIL_KEYS},**extra})
        path=ROOT/f'private/details/{pid}.json'
        payload={'place_id':pid,'details':merged,'source_responses':[
            {'file':bpath.relative_to(REPO).as_posix(),'sha256':base['response_sha256'],'retrieved_at':base['retrieved_at'],'fields':base['field_mask']},
            {'file':apath.relative_to(REPO).as_posix(),'sha256':aug['sha256'],'retrieved_at':aug['received_at'],'fields':MASK}],
            'field_merge':'Original requested fields retained; augmentation adds only original-contract fields absent from screening field mask.'}
        save(path,{'schema':'nets-ai-corrected-sealed-json-v1','payload':payload,'sha256':digest(encoded(payload))})
        site=merged.get('websiteUri');usable=False;reason='missing_website'
        if site:
            try:website_key(site);usable=True;reason=None
            except ValueError:reason='invalid_website_under_frozen_contract'
        rows.append({'candidate_id':pid,'website_url':site,'usable_website':usable,'no_website_reason':reason,
            'details_file':path.relative_to(ROOT).as_posix(),'details_sha256':digest(path.read_bytes())})
    assert len(rows)==836 and sum(r['usable_website'] for r in rows)==762
    save(ROOT/'plan.json',{**read_json(ROOT/'protocol.json'),'candidates':rows})
    print('All 836 full Details merged, website count 762.',flush=True)

def discover():
    check_pins();plan=read_json(ROOT/'plan.json');engine=Discovery(ROOT/'discovery',plan)
    snapshots={};counts=Counter()
    for i,row in enumerate(plan['candidates'],1):
        if not row['usable_website']:continue
        queries=[]
        for date in plan['schedule']:
            q=engine.query(row['website_url'],date)
            # Preserve original query result; apply the accepted closure correction only offline.
            if q['state']!='no_snapshot':
                receipt=read_json(ROOT/'discovery'/q['final_receipt'])
                if receipt['http_status']==200 and not receipt['error']:
                    body=(ROOT/'discovery'/q['final_receipt']).with_suffix('.bin').read_bytes()
                    corrected=revalidate(body,row['website_url'],date,plan['snapshot_cutoff'])
                    q={**q,**corrected}
            queries.append(q)
        sids=[]
        for q in queries:
            if q['state']!='confirmed':continue
            sid='snapshot_'+digest(encoded([q['confirmed_snapshot_timestamp'],website_key(q['original_target_url'])]))
            if sid not in snapshots:
                snapshots[sid]={'candidate_id':sid,'original_website_url':q['original_target_url'],
                    'requested_archive_timestamp':q['requested_timestamp'],'confirmed_snapshot_timestamp':q['confirmed_snapshot_timestamp'],
                    'exact_wayback_snapshot_url':q['exact_wayback_snapshot_url'],'candidate_bindings':[]}
            if row['candidate_id'] not in snapshots[sid]['candidate_bindings']:snapshots[sid]['candidate_bindings'].append(row['candidate_id'])
            if sid not in sids:sids.append(sid)
        failed=any(q['state']=='failure' for q in queries)
        proof={'candidate_id':row['candidate_id'],'queries':queries,'snapshot_ids':sids,
               'discovery_failure':failed,'bounded_lookup_complete':not failed}
        save(ROOT/f'closure/discovery/candidates/{row["candidate_id"]}.json',proof)
        counts.update(q['state'] for q in queries)
        if i%10==0:print('Discovery',i,'/836',dict(counts),'snapshots',len(snapshots),flush=True)
    blocked=read_json(OLDARCH/'security/quarantine_receipt.json')
    save(ROOT/'confirmed_snapshots.json',{'records':list(snapshots.values())})
    save(ROOT/'excluded_snapshots.json',{'records':[r for r in snapshots.values() if r['candidate_id']==blocked['snapshot_id']],
         'source_quarantine_sha256':digest((OLDARCH/'security/quarantine_receipt.json').read_bytes())})
    save(ROOT/'snapshots.json',{'records':[r for r in snapshots.values() if r['candidate_id']!=blocked['snapshot_id']]})
    save(ROOT/'discovery_summary.json',{'query_states':dict(counts),'unique_snapshots':len(snapshots)})
    print('Discovery complete',dict(counts),len(snapshots),flush=True)

def discover_parallel():
    """Only transport scheduling changes; each query uses the frozen discovery engine."""
    from wayback_snapshot_fetcher import retry_after
    check_pins();plan=read_json(ROOT/'plan.json')
    save(ROOT/'discovery_transport_amendment.json',{'workers':4,'minimum_global_interval_seconds':1,
        'global_transient_backoff_seconds':[30,60,120,240,300],'retry_after_respected':True,
        'frozen_discovery_parser_and_closure_rules_unchanged':True,
        'reason':'Bound latency while retaining shared rate-limit backoff; no evidence-selection change.'})
    save(ROOT/'discovery_idle_connection_repair.json',{'maximum_idle_connection_seconds':10,
        'reason':'RemoteDisconnected responses followed prolonged global cooldowns on retained persistent connections. Rebuild an idle TLS connection before the next request.',
        'global_cooldown_preserved':True,'request_payload_and_frozen_validation_unchanged':True,
        'previous_attempts_and_completed_queries_preserved':True})
    lock=threading.Lock();local=threading.local();statepath=ROOT/'discovery_global_cooldown.json'
    state=read_json(statepath) if statepath.exists() else {'next_at':0,'streak':0}
    class SharedHTTP:
        def __init__(self):
            self.http=AvailabilityHTTP(timeout=plan['discovery']['timeout_seconds']);self.last_used=None
        def get(self,website,date):
            while True:
                with lock:
                    delay=state['next_at']-time.time()
                    if delay<=0:
                        state['next_at']=time.time()+1;write_json(statepath,state);break
                time.sleep(min(5,delay))
            if self.last_used is not None and time.time()-self.last_used>10:
                connection=getattr(self.http,'connection',None)
                if connection is not None:connection.close();self.http.connection=None
            response=self.http.get(website,date);self.last_used=time.time()
            with lock:
                transient=response['http_status'] in {429,500,502,503,504} or bool(response['error'])
                if transient:
                    state['streak']+=1
                    headers={k.lower():v for k,v in response['headers']}
                    pause=retry_after(headers.get('retry-after'),time.time())
                    if pause is None:pause=min(300,30*2**min(state['streak']-1,4))
                    state['next_at']=max(state['next_at'],time.time()+pause)
                else:state['streak']=0
                write_json(statepath,state)
            return response
    queries=list(dict.fromkeys((r['website_url'],date) for r in plan['candidates'] if r['usable_website'] for date in plan['schedule']))
    verifier=Discovery(ROOT/'discovery',plan)
    def one(pair):
        website,date=pair;key=digest(encoded([website,date]));central=ROOT/'discovery/requests'/key/'result.json'
        if central.exists():
            value=read_json(central);verifier.verify(value);return value['state']
        if not hasattr(local,'engine'):
            local.engine=Discovery(ROOT/'discovery_workers'/str(threading.get_ident()),plan,transport=SharedHTTP())
        # Resume any preserved incomplete worker transaction under its original attempt paths.
        previous=sorted((ROOT/'discovery_workers').glob('*/requests/'+key))
        origin=previous[0].parent.parent if previous else local.engine.root
        engine=Discovery(origin,plan,transport=local.engine.transport)
        value=engine.query(website,date)
        # Complete query receipts are copied verbatim, never edited, to their canonical commitment paths.
        for relative,h in value['files'].items():
            body=(engine.root/relative).read_bytes();assert digest(body)==h
            write(ROOT/'discovery'/relative,body,immutable=True)
        save(central,value);verifier.verify(value)
        return value['state']
    counts=Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,q) for q in queries]
        for n,f in enumerate(as_completed(futures),1):
            counts[f.result()]+=1
            if n%25==0:print('Official query receipts',n,'/',len(queries),dict(counts),flush=True)
    discover()

class IdleAwareSnapshotHTTP(HTTPTransport):
    """Keep the frozen exact-replay transport; replace only idle connections."""
    def get(self,url):
        last=getattr(self.local,'last_used',None)
        if last is not None and time.time()-last>10:
            connection=getattr(self.local,'connection',None)
            if connection is not None:connection.close();self.local.connection=None
        response=super().get(url)
        self.local.last_used=time.time()
        return response

def fetch():
    check_pins();plan=read_json(ROOT/'plan.json')
    settings=read_json(ROOT/'fetch_transport_amendment.json')['settings']
    rawroot=ROOT/'raw_retrieval'
    if os.name=='nt':rawroot=Path('\\\\?\\'+str(rawroot))
    save(ROOT/'raw_idle_connection_repair.json',{'maximum_idle_connection_seconds':10,
        'same_observed_persistent_connection_issue_as_discovery':True,
        'frozen_exact_replay_transport_and_crawler_unchanged':True,'global_admission_and_backoff_unchanged':True})
    transport=IdleAwareSnapshotHTTP(settings['timeout'],8_000_000)
    result=Crawler(rawroot,transport=transport,**settings).run(ROOT/'snapshots.json')
    save(ROOT/'raw_fetch_seal.json',{'execution':result['execution'],'manifest_sha256':digest(encoded(result)),
        'snapshot_manifest_sha256':digest((ROOT/'snapshots.json').read_bytes())})
    print('Fetch complete',result['state_counts'],flush=True)

def checked_proof(row,plan):
    if not row['usable_website']:return None
    proof=read_json(ROOT/f'closure/discovery/candidates/{row["candidate_id"]}.json')
    assert proof['candidate_id']==row['candidate_id']
    assert [q['requested_timestamp'] for q in proof['queries']]==plan['schedule']
    verifier=Discovery(ROOT/'discovery',plan);sids=[]
    for q in proof['queries']:
        key=digest(encoded([row['website_url'],q['requested_timestamp']]))
        original=read_json(ROOT/'discovery/requests'/key/'result.json')
        assert original['website_url']==row['website_url'] and original['request_key']==key
        verifier.verify(original)
        expected=dict(original)
        receipt_path=ROOT/'discovery'/original['final_receipt'];receipt=read_json(receipt_path)
        if original['state']!='no_snapshot' and receipt['http_status']==200 and not receipt['error']:
            expected.update(revalidate(receipt_path.with_suffix('.bin').read_bytes(),row['website_url'],q['requested_timestamp'],plan['snapshot_cutoff']))
        assert encoded(expected)==encoded(q),'Candidate query does not reproduce from official response'
        if q['state']=='confirmed':
            sid='snapshot_'+digest(encoded([q['confirmed_snapshot_timestamp'],website_key(q['original_target_url'])]))
            if sid not in sids:sids.append(sid)
    assert sids==proof['snapshot_ids']
    failed=any(q['state']=='failure' for q in proof['queries'])
    assert proof['discovery_failure']==failed and proof['bounded_lookup_complete']==(not failed)
    return proof

def context_check(record):
    """Local estimate with conservative schema and request-framing reserves."""
    request=record['request'];size=len(encoded(request))
    if size+128000+4096<=400000:
        return {'method':'conservative_utf8_byte_bound','input_upper_bound':size}
    assert set(request)=={'model','input','text','store'} and request['store'] is False
    sys.path.append(str(REPO/'tools/local_tokenizer'))
    import tiktoken,hashlib
    cache=REPO/'tools/local_tokenizer/cache'
    url='https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken'
    vocabulary=cache/hashlib.sha1(url.encode()).hexdigest()
    assert digest(vocabulary.read_bytes())=='446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d'
    os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
    encoder=tiktoken.encoding_for_model(request['model'])
    assert encoder.name=='o200k_base'
    tokens=len(encoder.encode(encoded(request).decode('utf-8'),disallowed_special=()))
    # Serialized JSON tokenization is not the server's exact framing count.
    # Additionally reserve the complete schema byte size plus 4,096 tokens.
    schema_reserve=len(encoded(request['text']))
    assert tokens+schema_reserve+128000+4096<=400000,'Full evidence exceeds conservative local token bound'
    return {'method':'local_tiktoken_with_schema_and_framing_reserves','request_bytes':size,
            'serialized_request_tokens':tokens,'schema_byte_reserve':schema_reserve,
            'framing_reserve':4096,'maximum_output_reserve':128000,'context_limit':400000,
            'encoding':'o200k_base','tiktoken_version':tiktoken.__version__,'exact_server_count':False,
            'classification_request_sha256':digest(encoded(request)),'network_calls':0}

def packets():
    from clean_archive_preflight import verify_raw, candidate_state, opening_summary, technical_raw_failure
    from clean_archive_closure import provenance_page, classifier_item, artifact
    from clean_phase2_evidence import build_packet, validate_record, STATES
    from run_clean_archive import unseal
    if (ROOT/'evidence_freeze.json').exists():
        from clean_archive_preflight import verify_files
        check_pins();verify_files(ROOT,read_json(ROOT/'evidence_freeze.json')['files'])
        print('Existing full evidence freeze verified.',flush=True)
        return
    check_pins();plan=read_json(ROOT/'plan.json')
    write(ROOT/'implementation/packet_runner_v2.py',Path(__file__).read_bytes(),immutable=True)
    records=read_json(ROOT/'snapshots.json')['records']
    raw,manifest=verify_raw(ROOT,records,{**plan,'fetch':read_json(ROOT/'fetch_transport_amendment.json')['settings']})
    excluded=read_json(ROOT/'excluded_snapshots.json')['records']
    for r in excluded:raw[r['candidate_id']]={'state':'quarantined','reason':'prior_security_quarantine','http_status':None}
    allrecords={r['candidate_id']:r for r in read_json(ROOT/'confirmed_snapshots.json')['records']}
    counts=Counter({state:0 for state in STATES});failures=[];files={};sizes=[];parse_reasons=Counter()
    for n,row in enumerate(plan['candidates'],1):
        cid=row['candidate_id'];detail_path=ROOT/row['details_file']
        assert digest(detail_path.read_bytes())==row['details_sha256']
        detail=unseal(detail_path)['details']
        assert detail['id']==cid and detail.get('websiteUri')==row['website_url']
        proof=checked_proof(row,plan)
        pages=[];outcomes=[]
        for sid in sorted(proof['snapshot_ids'] if proof else []):
            value=raw[sid]
            outcomes.append({'snapshot_id':sid,'timestamp':allrecords[sid]['confirmed_snapshot_timestamp'],
                'state':value['state'],'reason':value['reason'],'http_status':value.get('http_status',value.get('provenance',{}).get('http_status'))})
            if value['state']=='success':
                pages.append(provenance_page(ROOT,ROOT/'closure',row,sid,value,proof['queries'],detail,plan['snapshot_cutoff']))
        state=candidate_state(row,proof,pages,outcomes);counts[state]+=1
        parse_reasons.update(str(p['reason']) for p in pages if not p['substantive_parse'])
        active=[p for p in pages if p['active_by_eligible']]
        packet={'candidate_id':cid,'details':detail,'frozen_details':artifact(ROOT,ROOT/row['details_file'],'source'),
            'website_url':row['website_url'],'usable_website':row['usable_website'],'excluded_website_reason':row['no_website_reason'],
            'state':state,'bounded_discovery_failure':bool(proof and proof['discovery_failure']),
            'bounded_lookup_complete':bool(proof and proof['bounded_lookup_complete']),
            'snapshot_ids':sorted(proof['snapshot_ids'] if proof else []),'raw_snapshot_outcomes':outcomes,
            'archive_pages':pages,'classifier_archive_evidence':[classifier_item(p) for p in pages if p['substantive_parse']],
            'substantive_page_count':sum(p['substantive_parse'] for p in pages),
            'active_by_year':min((p['active_by_year'] for p in active),default=None),**opening_summary(pages)}
        source=ROOT/f'closure/private/packets/{cid}.json';save(source,packet)
        result=build_packet(packet,digest(source.read_bytes()),str(ROOT/'closure'))
        validate_record(result)
        size=len(encoded(result['request']))
        relative=f'classification/private/inputs/{cid}.json';save(ROOT/relative,result)
        files[relative]=digest((ROOT/relative).read_bytes());files[source.relative_to(ROOT).as_posix()]=digest(source.read_bytes())
        sizes.append(size)
        reasons=[]
        if proof and proof['discovery_failure']:reasons.append('discovery_validation_or_transport_failure')
        reasons.extend('raw:'+str(v['reason']) for v in outcomes if technical_raw_failure(v))
        reasons.extend('parser:'+str(p['reason']) for p in pages if p['reason'] in {'invalid_text_encoding','parser_exception'})
        if reasons:failures.append({'candidate_id':cid,'has_text':bool(packet['substantive_page_count']),'reasons':sorted(set(reasons))})
        if n%50==0:print('Frozen evidence packets',n,'/836',flush=True)
    save(ROOT/'candidate_packet_freeze.json',{'candidate_count':836,'files':files})
    context_checks={pid:context_check(read_json(ROOT/f'classification/private/inputs/{pid}.json'))
                    for pid in read_json(ROOT/'input_place_ids.json')}
    save(ROOT/'context_checks.json',context_checks)
    no_text_fail=sum(not f['has_text'] for f in failures)
    technical_snapshots=sum(technical_raw_failure(v) for v in raw.values())
    terminal429=sum(v.get('http_status')==429 for v in raw.values())
    ready=(no_text_fail/762<=.05 and technical_snapshots/max(1,len(raw))<=.05 and terminal429==0
           and counts['archive_content_available']>no_text_fail)
    report={'candidate_count':836,'usable_websites':762,'archive_states':dict(counts),
        'text_coverage_among_websites':counts['archive_content_available']/762,
        'text_coverage_all_candidates':counts['archive_content_available']/836,
        'technical_candidates_without_text':no_text_fail,'technical_candidates':failures,
        'technical_reason_candidate_counts':dict(Counter(reason for f in failures for reason in f['reasons'])),
        'all_parser_rejection_reasons':dict(parse_reasons),
        'raw_snapshot_failure_reasons':dict(Counter(str(v['reason']) for v in raw.values() if v['state']!='success')),
        'technical_snapshot_count':technical_snapshots,'terminal_429':terminal429,
        'new_model_calls_at_preflight':0,'maximum_request_bytes':max(sizes),'classification_gate_passed':ready}
    save(ROOT/'acquisition_preflight.json',report)
    # Freeze all new acquisition receipts and full packets, not only the derived summary.
    files.update({p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for folder in ('private','discovery','closure','token_counts')
                  for p in (ROOT/folder).rglob('*') if p.is_file()})
    for name in ('plan.json','snapshots.json','confirmed_snapshots.json','excluded_snapshots.json','raw_fetch_seal.json','acquisition_preflight.json',
                 'protocol.json','contract_pins.json','user_instruction.txt','frames.json','input_place_ids.json','details_audit.json',
                 'provenance_roots.json','discovery_transport_amendment.json','fetch_transport_amendment.json','offline_validation.json',
                 'implementation/packet_runner.py','implementation/packet_runner_v2.py','candidate_packet_freeze.json','context_checks.json','preservation_policy.json','preservation_during_acquisition.json',
                 'discovery_idle_connection_repair.json','transport_repair_validation.json','raw_idle_connection_repair.json',
                 'historical_contract_pins.json','scientific_contract_comparison.json','raw_transport_repair_validation.json'):
        files[name]=digest((ROOT/name).read_bytes())
    save(ROOT/'evidence_freeze.json',{'candidate_count':836,'files':files,'frozen_at':now(),'all_inputs_frozen_before_model_calls':True})
    print('Acquisition preflight',dict(counts),'gate',ready,flush=True)

def classify():
    from dotenv import dotenv_values
    from openai import OpenAI
    from clean_archive_preflight import verify_files
    from clean_phase2_closure_runtime import Runtime
    from clean_phase2_closure_contract import build_retry_record
    from clean_phase2_evidence import build_packet
    check_pins();freeze=read_json(ROOT/'evidence_freeze.json');verify_files(ROOT,freeze['files'])
    assert freeze['candidate_count']==836 and read_json(ROOT/'acquisition_preflight.json')['classification_gate_passed']
    key=os.environ.get('OPENAI_API_KEY') or dotenv_values(REPO/'.env',interpolate=False).get('OPENAI_API_KEY')
    if not key:raise RuntimeError('Missing OpenAI credential')
    client=OpenAI(api_key=key,base_url='https://api.openai.com/v1',max_retries=0,timeout=180)
    runtime=Runtime(ROOT/'classification',client)
    retryruntime=Runtime(ROOT/'classification_retry',client)
    records=[read_json(ROOT/f'classification/private/inputs/{pid}.json') for pid in read_json(ROOT/'input_place_ids.json')]
    for record in records:
        source=ROOT/f'closure/private/packets/{record["place_id"]}.json'
        assert encoded(build_packet(read_json(source),digest(source.read_bytes()),str(ROOT/'closure')))==encoded(record)
    source_code=Path(__file__).read_bytes()
    write(ROOT/'implementation/classification_runner.py',source_code,immutable=True)
    save(ROOT/'classification_implementation.json',{'runner_sha256':digest(source_code),
        'frozen_contract_modules':read_json(ROOT/'contract_pins.json'),'all_836_requests_reconstructed_before_calls':True,
        'sdk_max_retries':0,'workers':6,'base_url':'https://api.openai.com/v1','model_probe_calls':0})
    def one(record,engine,origin):
        result=engine.candidate(record)
        return {'place_id':record['place_id'],'origin':origin,'status':result['status'],
                'result_file':f'{origin}/private/classifications/{record["place_id"]}.json'}
    outcomes=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(one,r,runtime,'classification') for r in records]
        try:
            for n,f in enumerate(as_completed(futures),1):
                outcomes.append(f.result())
                write_json(ROOT/'classification_progress.json',{'completed':n,'total':836,'states':dict(Counter(r['status'] for r in outcomes)),'updated_at':now()})
                if n%10==0:print('Classification',n,'/836',dict(Counter(r['status'] for r in outcomes)),flush=True)
        except BaseException:
            runtime.stop.set();retryruntime.stop.set()
            for f in futures:f.cancel()
            raise
    # Match the accepted closure sequence: finish the full initial frame, freeze
    # its valid results, then call only the technical-unresolved subset again.
    initial=sorted(outcomes,key=lambda r:r['place_id'])
    save(ROOT/'initial_classification_index.json',initial)
    unresolved_ids={r['place_id'] for r in initial if r['status']=='technical_unresolved'}
    retry_records=[];retry_files={}
    for record in records:
        if record['place_id'] not in unresolved_ids:continue
        retry=build_retry_record(record)
        context_check(retry)
        relative=f'classification_retry/private/inputs/{record["place_id"]}.json'
        save(ROOT/relative,retry);retry_files[relative]=digest((ROOT/relative).read_bytes());retry_records.append(retry)
    valid_files={r['result_file']:digest((ROOT/r['result_file']).read_bytes()) for r in initial if r['status']=='completed'}
    save(ROOT/'retry_scope_freeze.json',{'place_ids':sorted(unresolved_ids),'retry_input_files':retry_files,
        'original_valid_result_files':valid_files,'retry_calls_only_after_initial_full_frame_closes':True})
    replacements={}
    if retry_records:
        print('Initial frame closed; exact-citation technical retries:',len(retry_records),flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(one,r,retryruntime,'classification_retry') for r in retry_records]
            try:
                for n,f in enumerate(as_completed(futures),1):
                    result=f.result();replacements[result['place_id']]=result
                    print('Technical closure',n,'/',len(retry_records),result['status'],flush=True)
            except BaseException:
                retryruntime.stop.set()
                for f in futures:f.cancel()
                raise
    verify_files(ROOT,valid_files)
    outcomes=[replacements.get(r['place_id'],r) for r in initial]
    save(ROOT/'supplemental_classification_index.json',sorted(outcomes,key=lambda r:r['place_id']))
    print('All 836 classifications processed',dict(Counter(r['status'] for r in outcomes)),flush=True)

def report():
    import csv,io
    from clean_phase2_closure_runtime import verify_completed
    from clean_archive_preflight import verify_files
    check_pins();verify_files(ROOT,read_json(ROOT/'evidence_freeze.json')['files'])
    index=read_json(ROOT/'supplemental_classification_index.json')
    screening={r['place_id']:r for r in read_json(SCREEN/'candidate_screening.json')}
    assert len(index)==836 and {r['place_id'] for r in index}==set(screening)
    rows=[];results=[]
    for r in index:
        cid=r['place_id'];origin=ROOT/r['origin']
        packet=read_json(origin/f'private/inputs/{cid}.json')
        result=verify_completed(origin,packet,read_json(ROOT/r['result_file']))
        results.append(result);s=screening[cid]
        rows.append({'place_id':cid,'eligibility':result['reducer']['eligibility_state'],
            'technical_status':result['status'],'frame_group':s['frame_group'],'returned_zip':s['returned_zip'],
            'signal_group':s['signal_group'],'explicit_pharmacy_related_signal':s['explicit_pharmacy_related_signal'],
            'any_signal_including_brand':s['any_signal_including_brand'],
            'archive_state':packet['archive']['state'],'result_file':r['result_file'],
            'result_sha256':digest((ROOT/r['result_file']).read_bytes())})
    old=Counter();old_frame_rows=[]
    from screen_supplemental_details import returned_zip
    frames=read_json(ROOT/'frames.json')
    for r in read_json(ROOT/'old_final_index.json'):
        path=Path(r['result_file']);assert digest(path.read_bytes())==r['result_file_sha256']
        eligibility=read_json(path)['reducer']['eligibility_state'];old[eligibility]+=1
        original=read_json(OLD/f'private/inputs/{r["place_id"]}.json')
        postal,zstatus,countries=returned_zip(original['google_metadata'])
        assert zstatus=='valid' and countries==['US']
        group='primary_60_zip' if postal in frames['primary_60'] else 'supply_99_only' if postal in frames['supply_99'] else 'outside_99_zip'
        old_frame_rows.append({'frame_group':group,'eligibility':eligibility})
    assert old=={'include':539,'exclude':687}
    new=Counter(r['eligibility'] for r in rows)
    def full_counts(counts):return {k:counts.get(k,0) for k in ('include','exclude','unresolved')}
    def tab(field):
        return {str(group):full_counts(Counter(r['eligibility'] for r in rows if r[field]==group)) for group in sorted({r[field] for r in rows})}
    requests=[]
    for segment in ('classification','classification_retry'):
        for path in sorted((ROOT/segment/'private/transactions').glob('*/*.request.json')):
            req=read_json(path);pid=req['binding']['place_id'];assert pid in screening
            assert req['requested_at']>read_json(ROOT/'evidence_freeze.json')['frozen_at']
            packet=read_json(ROOT/segment/f'private/inputs/{pid}.json')
            assert req['parameters']==packet['request'] and req['binding']['packet_sha256']==packet['packet_sha256']
            response=path.with_name(path.name.replace('.request.','.response.'))
            error=path.with_name(path.name.replace('.request.','.error.'))
            assert response.exists() or error.exists(),'Unresolved dispatched model request'
            requests.append({'request_file':path.relative_to(ROOT).as_posix(),'request_sha256':digest(path.read_bytes()),
                'place_id':pid,'response_received':response.exists(),'segment':segment})
    protected=read_json(ROOT/'preservation_before.json')
    volatile=set(read_json(ROOT/'preservation_policy.json')['volatile_office_owner_files'])
    assert all(p in protected and Path(p).name.startswith('~$') and Path(p).suffix=='.docx' for p in volatile)
    volatile_states={}
    for i,(p,expected) in enumerate(protected.items(),1):
        path=REPO/p
        if p in volatile:
            volatile_states[p]={'state':'present' if path.exists() else 'missing',
                'original_sha256':expected['sha256'],'current_sha256':digest(path.read_bytes()) if path.exists() else None}
            continue
        assert path.stat().st_mtime_ns==expected['mtime_ns'] and digest(path.read_bytes())==expected['sha256'],p
        if i%3000==0:print('Preservation verified',i,'/',len(protected),flush=True)
    unresolved=sum(r['status']!='completed' for r in results)
    summary={'supplemental_candidates':836,'supplemental_eligibility':full_counts(new),'technical_unresolved':unresolved,
        'old_candidates':1226,'old_eligibility':full_counts(old),'combined_candidates':2062,'combined_eligibility':full_counts(old+new),
        'supplemental_eligibility_by_returned_zip_frame':tab('frame_group'),'signal_group_by_eligibility':tab('signal_group'),
        'combined_eligibility_by_returned_zip_frame':{group:full_counts(Counter(r['eligibility'] for r in rows+old_frame_rows if r['frame_group']==group))
            for group in ('primary_60_zip','supply_99_only','outside_99_zip')},
        'explicit_signal_by_eligibility':tab('explicit_pharmacy_related_signal'),
        'signal_including_brand_by_eligibility':tab('any_signal_including_brand'),
        'archive_preflight':read_json(ROOT/'acquisition_preflight.json'),
        'exact_new_model_api_calls':len(requests),'model_calls_by_segment':dict(Counter(r['segment'] for r in requests)),
        'separate_non_generation_token_count_api_calls':len(list((ROOT/'token_counts').glob('*/*.request.json'))),
        'new_model_response_count':sum(r['response_received'] for r in requests),'protected_persistent_files_unchanged':len(protected)-len(volatile),
        'volatile_office_owner_file_observations':volatile_states,
        'all_old_classifications_and_evidence_unchanged':True,'rq2_rq3_executed':False,'thesis_modified':False,
        'technical_closure_complete':unresolved==0,'stop_for_user_review':True}
    save(ROOT/'model_call_ledger.json',requests);save(ROOT/'eligibility_rows.json',rows);save(ROOT/'FINAL_REPORT.json',summary)
    buffer=io.StringIO(newline='')
    writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(ROOT/'eligibility_rows.csv',buffer.getvalue().encode('utf-8-sig'),immutable=True)
    text=['# Supplemental 836-candidate closure','',f'Supplemental eligibility: {dict(new)}; technical unresolved: {unresolved}.',
          f'Combined 2,062 eligibility: {dict(old+new)}.',f'New model API calls, including retries: {len(requests)}.',
          '', 'All supplemental IDs were classified without provider-signal or ZIP prefiltering.',
          'Existing screening Details were retained; missing original-contract fields were acquired separately with distinct source timestamps.',
          'The seven-date lookup is bounded and does not establish exhaustive archive absence.',
          'Provider signals are diagnostic; eligibility is the frozen four-judgment reducer outcome.',
          f'All {len(protected)-len(volatile)} protected persistent files retained their bytes and modification times. The separately recorded Office owner/session lock file is transient. RQ2, RQ3 and thesis changes were not executed.',
          '', '## Returned-ZIP frame',json.dumps(tab('frame_group'),indent=2),
          '', '## Provider-signal overlap',json.dumps(tab('signal_group'),indent=2),
          '', '## Archive evidence',json.dumps(summary['archive_preflight'],indent=2)]
    write(ROOT/'REPORT.md','\n'.join(text).encode(),immutable=True)
    save(ROOT/'final_freeze.json',{'files':{p.relative_to(ROOT).as_posix():digest(p.read_bytes())
        for p in ROOT.rglob('*') if p.is_file() and p.name!='final_freeze.json'},'classification_count':836})
    print(json.dumps({k:v for k,v in summary.items() if k!='archive_preflight'},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','details','discover','discover_parallel','fetch','packets','classify','report'))
    stage=p.parse_args().stage
    if stage=='classify':
        import sqlite3
        owner=sqlite3.connect(ROOT/'classification_process_lock.sqlite3',timeout=0)
        try:
            owner.execute('BEGIN IMMEDIATE')
            classify()
        finally:owner.rollback();owner.close()
    else:globals()[stage]()
