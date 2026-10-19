"""Bounded official Availability discovery only; no page retrieval or providers."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import http.client
import json
import ssl
import time

from wayback_snapshot_fetcher import encoded,digest,write,write_json,read_json,now,parse_replay,website_key,retry_after

VERSION='clean-availability-discovery/1.0'
SCHEDULE=('19960101000000','20010101000000','20060101000000','20110101000000',
          '20160101000000','20210101000000','20260905000000')


def parse_available(body,website,requested_timestamp,cutoff):
    """A limited negative is never a claim of exhaustive archive absence."""
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    try:
        payload=json.loads(body,object_pairs_hook=unique)
        if not isinstance(payload,dict) or not isinstance(payload.get('archived_snapshots'),dict):
            raise ValueError('invalid Availability schema')
        if 'url' in payload and website_key(payload['url'])!=website_key(website):
            return {'state':'failure','reason':'response_website_binding_failed'}
        if 'timestamp' in payload and str(payload['timestamp'])!=requested_timestamp:
            raise ValueError('requested timestamp response mismatch')
        snapshots=payload['archived_snapshots']
        if not snapshots:
            return {'state':'no_snapshot','reason':'no_currently_accessible_snapshot_returned'}
        closest=snapshots.get('closest')
        if not isinstance(closest,dict) or type(closest.get('available')) is not bool:
            raise ValueError('invalid closest snapshot schema')
        metadata={'returned_timestamp':closest.get('timestamp'),'returned_url':closest.get('url'),
                  'availability_status':closest.get('status'),'available':closest['available']}
        if not closest['available']:
            return {'state':'no_snapshot','reason':'closest_reported_unavailable',**metadata}
        if str(closest.get('status'))!='200':
            return {'state':'failure','reason':'unexpected_available_snapshot_status',**metadata}
        identity=parse_replay(closest['url'])
        if identity['timestamp']!=str(closest.get('timestamp')):
            return {'state':'failure','reason':'snapshot_timestamp_mismatch',**metadata}
        if website_key(identity['original_url'])!=website_key(website):
            return {'state':'failure','reason':'snapshot_website_binding_failed',**metadata}
        if identity['timestamp']>cutoff:
            return {'state':'failure','reason':'future_snapshot_returned',**metadata}
        # Fixed raw replay form, preserving the official capture time and target.
        exact=f'https://web.archive.org/web/{identity["timestamp"]}id_/{identity["original_url"]}'
        return {'state':'confirmed','confirmed_snapshot_timestamp':identity['timestamp'],
                'original_target_url':identity['original_url'],'exact_wayback_snapshot_url':exact,**metadata}
    except (ValueError,TypeError,KeyError,UnicodeError) as exc:
        return {'state':'failure','reason':'invalid_availability_response','error_type':type(exc).__name__}


class AvailabilityHTTP:
    """Single persistent TLS session to exactly one official API path."""
    def __init__(self,timeout=20,max_bytes=1_000_000):
        self.timeout,self.max_bytes,self.connection=timeout,max_bytes,None
    def get(self,website,timestamp):
        path='/wayback/available?'+urlencode({'url':website,'timestamp':timestamp})
        if self.connection is None:
            self.connection=http.client.HTTPSConnection('archive.org',timeout=self.timeout,context=ssl.create_default_context())
        body=b'';status=None;headers=[];error=None
        try:
            self.connection.request('GET',path,headers={'User-Agent':VERSION+' (academic bounded archive lookup)',
                                                       'Accept':'application/json','Accept-Encoding':'identity'})
            response=self.connection.getresponse()
            status,headers=response.status,response.getheaders()
            body=response.read(self.max_bytes+1);response.close()
            if len(body)>self.max_bytes:error='body_limit_exceeded'
        except (OSError,http.client.HTTPException) as exc:error=type(exc).__name__
        if error:
            self.connection.close();self.connection=None
        return {'http_status':status,'headers':headers,'body':body,'error':error,
                'request_url':'https://archive.org'+path,'final_url':'https://archive.org'+path}


class Discovery:
    def __init__(self,root,plan,*,transport=None,clock=time.time,sleep=time.sleep):
        self.root,self.plan=Path(root),plan
        self.transport=transport or AvailabilityHTTP(timeout=plan['discovery']['timeout_seconds'])
        self.clock,self.sleep=clock,sleep
    def query(self,website,timestamp):
        if timestamp not in self.plan['schedule']:raise ValueError('timestamp outside frozen common schedule')
        key=digest(encoded([website,timestamp]))
        directory=self.root/'requests'/key
        result_path=directory/'result.json'
        if result_path.exists():
            result=read_json(result_path)
            self.verify(result)
            return result
        response=None;parsed=None;attempt_files=[]
        for attempt in range(1,self.plan['discovery']['max_attempts']+1):
            receipt_path=directory/f'{attempt}.json';body_path=directory/f'{attempt}.bin'
            if receipt_path.exists():
                receipt=read_json(receipt_path);body=body_path.read_bytes()
                if digest(body)!=receipt['body_sha256']:raise ValueError('discovery receipt body changed')
                response={**receipt,'body':body}
            else:
                cooldown=self.root/'cooldown.json'
                state=read_json(cooldown) if cooldown.exists() else {}
                while state.get('next_at',0)>self.clock():self.sleep(min(30,state['next_at']-self.clock()))
                started=now()
                response=self.transport.get(website,timestamp)
                response.update(requested_at=started,received_at=now(),body_sha256=digest(response['body']),
                                body_bytes=len(response['body']),website_url=website,requested_timestamp=timestamp)
                headers={k.lower():v for k,v in response['headers']}
                pause=self.plan['discovery']['minimum_interval_seconds']
                if response['http_status'] in {429,500,502,503,504} or response['error']:
                    delay=retry_after(headers.get('retry-after'),self.clock())
                    pause=max(pause,delay if delay is not None else min(300,30*2**(attempt-1)))
                write_json(cooldown,{'next_at':self.clock()+pause,'last_status':response['http_status']})
                write(body_path,response['body'],immutable=True)
                write_json(receipt_path,{k:v for k,v in response.items() if k!='body'},immutable=True)
            attempt_files.extend([body_path.relative_to(self.root).as_posix(),receipt_path.relative_to(self.root).as_posix()])
            if response['http_status']==200 and not response['error']:
                parsed=parse_available(response['body'],website,timestamp,self.plan['snapshot_cutoff'])
                break
            parsed={'state':'failure','reason':'availability_http_failure','http_status':response['http_status'],'error':response['error']}
            if response['http_status'] not in {429,500,502,503,504,None}:break
        result={'website_url':website,'requested_timestamp':timestamp,'request_key':key,**parsed,
                'final_receipt':receipt_path.relative_to(self.root).as_posix(),
                'raw_response_sha256':response['body_sha256'],
                'files':{p:digest((self.root/p).read_bytes()) for p in attempt_files}}
        self.verify(result)
        write_json(result_path,result,immutable=True)
        return result
    def verify(self,result):
        if result['requested_timestamp'] not in self.plan['schedule']:
            raise ValueError('discovery timestamp outside committed schedule')
        body_relative=str(Path(result['final_receipt']).with_suffix('.bin')).replace('\\','/')
        if not {result['final_receipt'],body_relative}<=set(result['files']):
            raise ValueError('discovery commitment omits terminal response')
        for relative,expected in result['files'].items():
            path=(self.root/relative).resolve()
            if not path.is_relative_to(self.root.resolve()) or digest(path.read_bytes())!=expected:
                raise ValueError('discovery response commitment changed')
        receipt=read_json(self.root/result['final_receipt'])
        website,timestamp=result['website_url'],result['requested_timestamp']
        expected_url='https://archive.org/wayback/available?'+urlencode({'url':website,'timestamp':timestamp})
        if (receipt['request_url']!=expected_url or receipt['final_url']!=expected_url
                or receipt['website_url']!=website or receipt['requested_timestamp']!=timestamp):
            raise ValueError('discovery request binding changed')
        body=(self.root/body_relative).read_bytes()
        if digest(body)!=result['raw_response_sha256'] or digest(body)!=receipt['body_sha256']:
            raise ValueError('discovery body commitment changed')
        if receipt['http_status']==200 and not receipt['error']:
            expected=parse_available(body,website,timestamp,self.plan['snapshot_cutoff'])
        else:
            expected={'state':'failure','reason':'availability_http_failure','http_status':receipt['http_status'],'error':receipt['error']}
        if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('discovery parsed result changed')


def run_discovery(root,plan):
    root=Path(root)
    engine=Discovery(root/'discovery',plan)
    candidates=[];snapshots={};confirmed_occurrences=0
    for number,row in enumerate(plan['candidates'],1):
        if not row['usable_website']:continue
        queries=[engine.query(row['website_url'],date) for date in plan['schedule']]
        identities=[]
        for q in queries:
            if q['state']!='confirmed':continue
            confirmed_occurrences+=1
            identity=digest(encoded([q['confirmed_snapshot_timestamp'],website_key(q['original_target_url'])]))
            if identity not in snapshots:
                snapshots[identity]={'candidate_id':'snapshot_'+identity,'original_website_url':row['website_url'],
                    'requested_archive_timestamp':q['requested_timestamp'],'confirmed_snapshot_timestamp':q['confirmed_snapshot_timestamp'],
                    'exact_wayback_snapshot_url':q['exact_wayback_snapshot_url'],'candidate_bindings':[],
                    'discovery_receipts':[]}
            selected=snapshots[identity]
            if row['candidate_id'] not in selected['candidate_bindings']:selected['candidate_bindings'].append(row['candidate_id'])
            selected['discovery_receipts'].append({'candidate_id':row['candidate_id'],'request_key':q['request_key'],
                'requested_timestamp':q['requested_timestamp'],'raw_response_sha256':q['raw_response_sha256'],
                'returned_exact_url':q['exact_wayback_snapshot_url']})
            if identity not in identities:identities.append(identity)
        failed=any(q['state']=='failure' for q in queries)
        result={'candidate_id':row['candidate_id'],'queries':queries,'snapshot_ids':identities,
                'discovery_failure':failed,'bounded_lookup_complete':not failed,
                'state':'discovery_failure' if failed else 'confirmed' if identities else 'no_snapshot_in_bounded_lookup'}
        write_json(root/'discovery/candidates'/(row['candidate_id']+'.json'),result,immutable=True)
        candidates.append(result)
        if len(candidates)%25==0:
            print(json.dumps({'discovered_websites':len(candidates),'total':1088,'unique_snapshots':len(snapshots),'updated_at':now()}),flush=True)
    if len(candidates)!=1088:raise ValueError('bounded discovery frame incomplete')
    counts={'usable_websites':1088,'websites_with_confirmed_snapshot':sum(bool(r['snapshot_ids']) for r in candidates),
            'websites_with_no_snapshot_under_schedule':sum(r['state']=='no_snapshot_in_bounded_lookup' for r in candidates),
            'discovery_failure_websites':sum(r['discovery_failure'] for r in candidates),
            'unique_confirmed_snapshots':len(snapshots),'confirmed_candidate_date_occurrences':confirmed_occurrences,
            'duplicate_identities_across_target_dates':confirmed_occurrences-sum(len(r['snapshot_ids']) for r in candidates),
            'candidate_snapshot_pairs':sum(len(r['snapshot_ids']) for r in candidates),
            'duplicate_identities_across_candidates':sum(len(r['snapshot_ids']) for r in candidates)-len(snapshots),
            'all_candidates_processed':True,'all_queries_successful':all(not r['discovery_failure'] for r in candidates)}
    files={p.relative_to(root).as_posix():digest(p.read_bytes()) for parent in (root/'discovery/candidates',root/'discovery/requests')
           for p in parent.rglob('*') if p.is_file()}
    discovery={'schema':VERSION,'plan_sha256':digest((root/'plan.json').read_bytes()),'counts':counts,'files':files}
    write_json(root/'discovery_manifest.json',discovery,immutable=True)
    frozen={'schema':'clean-exact-snapshot-manifest-v1','discovery_manifest_sha256':digest((root/'discovery_manifest.json').read_bytes()),
            'records':sorted(snapshots.values(),key=lambda r:r['candidate_id'])}
    write_json(root/'snapshots.json',frozen,immutable=True)
    write_json(root/'snapshot_seal.json',{'sha256':digest((root/'snapshots.json').read_bytes()),'snapshot_count':len(snapshots),
                                       'discovery_manifest_sha256':frozen['discovery_manifest_sha256']},immutable=True)
    print(json.dumps(counts,indent=2),flush=True)
    return counts
