"""Standalone exact-snapshot HTTP retrieval. Python standard library only.

No discovery, proxy, extraction, model, or repository workflow dependencies.
HTTP 200 is a transport observation, never a business-evidence judgment.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit
import hashlib
import http.client
import json
import math
import os
import platform
import re
import socket
import sqlite3
import ssl
import threading
import time
import uuid

VERSION = 'standalone-wayback-snapshot/1.0'
TRANSIENT = {429, 500, 502, 503, 504}
REDIRECTS = {301, 302, 303, 307, 308}
MODIFIERS = {'', 'id_', 'ir_', 'mp_', 'if_', 'im_', 'js_', 'cs_', 'oe_', 'fr_', 'tf_', 'bn_', 'sw_', 'wkrf_'}
UNRESERVED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(body):
    return hashlib.sha256(body).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def write(path, body, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != body:
            raise ValueError('refusing to overwrite committed artifact: ' + str(path))
        return
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value, *, immutable=False):
    write(path, encoded(value), immutable=immutable)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def percent_key(text):
    if re.search(r'%(?![0-9a-fA-F]{2})', text):
        raise ValueError('malformed percent encoding')
    text = quote(text, safe="/%:@!$&'()*+,;=-._~?")
    return re.sub(r'%([0-9a-fA-F]{2})', lambda m: chr(int(m[1],16)) if chr(int(m[1],16)) in UNRESERVED else '%'+m[1].upper(), text)


def website_key(url):
    if not isinstance(url,str) or re.search(r'[\x00-\x20\x7f\\]',url):
        raise ValueError('invalid website URL')
    p = urlsplit(url)
    if p.scheme.lower() not in {'http','https'} or not p.hostname or p.username is not None or p.password is not None or p.fragment:
        raise ValueError('invalid website authority')
    host = p.hostname.lower()
    if not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?',host) or '..' in host:
        raise ValueError('invalid ASCII hostname')
    if host.startswith('www.'):
        host = host[4:]
    port = p.port
    if port == (80 if p.scheme.lower()=='http' else 443):
        port = None
    path = percent_key(p.path or '/')
    # Dot-segment normalization without collapsing repeated slashes or path case.
    segments = []
    for item in path.split('/'):
        if item == '.':continue
        if item == '..':
            if len(segments)>1:segments.pop()
        else:segments.append(item)
    path = '/'.join(segments) or '/'
    if path != '/' and path.endswith('/') and not path.endswith('//'):
        path = path[:-1]
    query = []
    for item in p.query.split('&') if p.query else []:
        key, sep, value = item.partition('=')
        key, value = percent_key(key), percent_key(value)
        # Known case-insensitive marketing tags only; opaque IDs keep case.
        if key.lower().startswith('utm_'):
            key, value = key.lower(), value.lower()
        query.append((key,sep,value))
    return (host,port,path,tuple(sorted(query,key=lambda item:item[0])))


def parse_replay(url):
    if not isinstance(url,str) or re.search(r'[\x00-\x20\x7f\\]',url):
        raise ValueError('invalid replay URL')
    p = urlsplit(url)
    if (p.scheme.lower() not in {'http','https'} or p.hostname!='web.archive.org'
            or p.port not in {None,80 if p.scheme.lower()=='http' else 443}
            or p.username is not None or p.password is not None or p.fragment):
        raise ValueError('replay must use the official Wayback host')
    tail = p.path + ('?'+p.query if '?' in url else '')
    match = re.fullmatch(r'/web/(\d{14})([a-z]{1,6}_)?/(https?://.+)',tail)
    if not match or (match[2] or '') not in MODIFIERS:
        raise ValueError('exact timestamp and supported replay modifier required')
    stamp, modifier, original = match[1],match[2] or '',match[3]
    datetime.strptime(stamp,'%Y%m%d%H%M%S')
    website_key(original)
    return {'timestamp':stamp,'modifier':modifier,'original_url':original,'replay_url':url}


def bound(record,url):
    try:
        replay = parse_replay(url)
        return (replay['timestamp']==record['confirmed_snapshot_timestamp']
                and website_key(replay['original_url'])==website_key(record['original_website_url']))
    except (ValueError,TypeError):
        return False


def input_records(body):
    document=json.loads(body)
    records=document['records'] if isinstance(document,dict) else document
    if not isinstance(records,list) or not records:
        raise ValueError('nonempty records list required')
    keys=set()
    for r in records:
        required=('candidate_id','original_website_url','requested_archive_timestamp','confirmed_snapshot_timestamp','exact_wayback_snapshot_url')
        if not isinstance(r,dict) or any(not isinstance(r.get(k),str) for k in required):
            raise ValueError('snapshot input fields must be strings')
        ident=r['candidate_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',ident) or re.fullmatch(r'(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])',ident):
            raise ValueError('candidate ID must be a portable safe path component')
        datetime.strptime(r['requested_archive_timestamp'],'%Y%m%d%H%M%S')
        key=(ident,r['confirmed_snapshot_timestamp'])
        if key in keys:raise ValueError('duplicate candidate/timestamp input')
        keys.add(key)
        if not bound(r,r['exact_wayback_snapshot_url']):
            raise ValueError('supplied snapshot timestamp/website binding failed')
    return records


def retry_after(value, clock_time):
    if value is None:return None
    try:
        seconds=float(value)
        return max(0,seconds) if math.isfinite(seconds) and seconds>=0 else None
    except (ValueError,TypeError):
        try:
            date=parsedate_to_datetime(value)
            return max(0,date.timestamp()-clock_time) if date.tzinfo else None
        except (ValueError,TypeError,OverflowError):
            return None


class HTTPTransport:
    """One persistent standard-library connection per worker; no ambient proxies."""
    identifier='python-http.client-direct-verified-tls'
    def __init__(self,timeout,max_bytes):
        self.timeout,self.max_bytes=timeout,max_bytes
        self.local=threading.local()

    def get(self,url):
        replay=parse_replay(url)  # No network capability for discovery/live hosts.
        p=urlsplit(url)
        scheme=p.scheme.lower()
        connection=getattr(self.local,'connection',None)
        if connection is None or getattr(self.local,'scheme',None)!=scheme:
            if connection:connection.close()
            connection=(http.client.HTTPSConnection('web.archive.org',timeout=self.timeout,context=ssl.create_default_context())
                        if scheme=='https' else http.client.HTTPConnection('web.archive.org',timeout=self.timeout))
            self.local.connection,self.local.scheme=connection,scheme
        body=b'';headers=[];status=None;error=None
        try:
            target=quote(p.path+('?' + p.query if p.query else ''),safe="/%:@!$&'()*+,;=-._~?[]")
            connection.request('GET',target,headers={'User-Agent':VERSION+' (academic exact-snapshot transport diagnostic)',
                                                    'Accept-Encoding':'identity','Accept':'*/*'})
            response=connection.getresponse()
            status,headers=response.status,response.getheaders()
            chunks=[];size=0
            try:
                while size<=self.max_bytes:
                    try:chunk=response.read(min(65536,self.max_bytes+1-size))
                    except http.client.IncompleteRead as exc:
                        chunks.append(exc.partial)
                        raise
                    if not chunk:break
                    chunks.append(chunk);size+=len(chunk)
            finally:
                body=b''.join(chunks)
                response.close()
            if len(body)>self.max_bytes:error='body_limit_exceeded'
        except (socket.timeout,TimeoutError):error='timeout'
        except (ConnectionResetError,ConnectionAbortedError,BrokenPipeError,http.client.RemoteDisconnected):error='connection_reset'
        except (OSError,http.client.HTTPException) as exc:error=type(exc).__name__
        if error:
            connection.close();self.local.connection=None
        return {'status':status,'headers':headers,'body':body,'final_url':url,'error':error,
                'tls_verified':scheme=='https'}


class Crawler:
    def __init__(self,output,*,timeout=20,concurrency=1,max_attempts=3,backoff_base=30,backoff_cap=300,
                 max_bytes=8_000_000,max_redirects=5,transport=None,clock=time.time,sleep=time.sleep):
        if concurrency not in (1,2) or not 1<=max_attempts<=3 or not 0<timeout<=300:
            raise ValueError('concurrency 1/2, attempts 1..3, timeout (0,300] required')
        if not 0<backoff_base<=backoff_cap or not 0<max_bytes or not 0<=max_redirects<=10:
            raise ValueError('invalid bounded transport settings')
        self.root=Path(output).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.concurrency,self.max_attempts=concurrency,max_attempts
        self.base,self.cap,self.max_redirects=backoff_base,backoff_cap,max_redirects
        self.transport=transport or HTTPTransport(timeout,max_bytes)
        self.clock,self.sleep,self.lock=clock,sleep,threading.Lock()
        self.execution=uuid.uuid4().hex
        self.settings={'timeout':timeout,'concurrency':concurrency,'max_attempts':max_attempts,
                       'backoff_base':backoff_base,'backoff_cap':backoff_cap,'max_bytes':max_bytes,'max_redirects':max_redirects}

    def _admit(self):
        while True:
            with self.lock:
                state=read_json(self.root/'cooldown.json') if (self.root/'cooldown.json').exists() else {}
                delay=state.get('next_request_at',0)-self.clock()
                if delay<=0:
                    state['next_request_at']=self.clock()+1
                    write_json(self.root/'cooldown.json',state)
                    return
            self.sleep(min(delay,30))

    def _observe(self,response,attempt):
        with self.lock:
            state=read_json(self.root/'cooldown.json')
            status=response['status'];headers={k.lower():v for k,v in response['headers']}
            streak=state.get('consecutive_429',0)+1 if status==429 else 0
            state['consecutive_429']=streak
            if status in TRANSIENT or response['error'] in {'timeout','connection_reset'}:
                power=min(20,max(streak,attempt)-1)
                delay=retry_after(headers.get('retry-after'),self.clock())
                if delay is None:delay=min(self.cap,self.base*2**power)
                state['next_request_at']=max(state['next_request_at'],self.clock()+delay)
            state.update(last_status=status,updated_at=now())
            write_json(self.root/'cooldown.json',state)

    def _state_path(self,r):
        return self.root/'state'/r['candidate_id']/(r['confirmed_snapshot_timestamp']+'.json')

    def _verify_success(self,r,state):
        folder=self.root/'raw'/r['candidate_id']/r['confirmed_snapshot_timestamp']
        provenance=read_json(folder/'provenance.json')
        if provenance['input_record_sha256']!=digest(encoded(r)) or provenance['sha256']!=state['sha256']:
            raise ValueError('successful snapshot commitment changed')
        if digest((folder/'body.bin').read_bytes())!=state['sha256']:
            raise ValueError('successful body hash changed')
        if not {'body.bin','headers.json','request.json','response.json','provenance.json'}<=set(state['files']):
            raise ValueError('incomplete successful response commitment')
        for name,expected in state['files'].items():
            if Path(name).name!=name or digest((folder/name).read_bytes())!=expected:
                raise ValueError('successful response integrity failed; original retained')
        return state

    def _fetch(self,r):
        path=self._state_path(r)
        state=read_json(path) if path.exists() else {'state':'pending','history':[]}
        if state.get('input_record_sha256',digest(encoded(r)))!=digest(encoded(r)):
            raise ValueError('candidate state input changed')
        if state['state']=='success':return self._verify_success(r,state)
        if state['state']=='permanent_failure':return state
        state['input_record_sha256']=digest(encoded(r))
        folder=self.root/'raw'/r['candidate_id']/r['confirmed_snapshot_timestamp']
        if (folder/'provenance.json').exists():
            committed=read_json(folder/'provenance.json')
            state.update(state='success',sha256=committed['sha256'],reason=None,http_status=200,
                         files={**committed['files_sha256'],'provenance.json':digest((folder/'provenance.json').read_bytes())})
            self._verify_success(r,state)
            write_json(path,state)
            return state
        for number in range(1,self.max_attempts+1):
            directory=self.root/'attempts'/r['candidate_id']/r['confirmed_snapshot_timestamp']/self.execution/str(number)
            state.update(state='pending',current_attempt=directory.relative_to(self.root).as_posix())
            write_json(path,state)
            write_json(directory/'started.json',{'record':r,'execution':self.execution,'attempt':number,'started_at':now()},immutable=True)
            chain=[];url=r['exact_wayback_snapshot_url'];reason=None;response=None
            for hop in range(self.max_redirects+1):
                self._admit()
                requested_at=now()
                write_json(directory/str(hop)/'request.json',{'url':url,'method':'GET','requested_at':requested_at},immutable=True)
                response=self.transport.get(url)
                received_at=now()
                self._observe(response,number)  # Persist pause before admitting another snapshot.
                item={'requested_url':url,'final_url':response['final_url'],'http_status':response['status'],
                      'headers':response['headers'],'requested_at':requested_at,'response_at':received_at,
                      'body_length':len(response['body']),'sha256':digest(response['body']),
                      'error':response['error'],'tls_verified':response['tls_verified']}
                write(directory/str(hop)/'body.bin',response['body'],immutable=True)
                write_json(directory/str(hop)/'response.json',item,immutable=True)
                chain.append(item)
                if not bound(r,response['final_url']):reason='binding_failure';break
                if response['error']:reason=response['error'];break
                capture_date=next((v for k,v in response['headers'] if k.lower()=='memento-datetime'),None)
                if capture_date and response['status']==200:
                    try:
                        date=parsedate_to_datetime(capture_date)
                        if not date.tzinfo or date.astimezone(timezone.utc).strftime('%Y%m%d%H%M%S')!=r['confirmed_snapshot_timestamp']:
                            raise ValueError('different snapshot')
                    except (ValueError,TypeError,OverflowError):reason='snapshot_timestamp_header_mismatch';break
                if response['status'] not in REDIRECTS:break
                location=next((v for k,v in response['headers'] if k.lower()=='location'),None)
                target=urljoin(url,location) if location else ''
                item['redirect_target']=target
                if not bound(r,target):reason='redirect_binding_failure';break
                # Preserve allowed scheme/host/default-port/modifier changes explicitly.
                if target in [entry['requested_url'] for entry in chain]:reason='redirect_loop';break
                if hop==self.max_redirects:reason='redirect_limit';break
                url=target
            status=response['status']
            if reason is None and status!=200:reason='http_'+str(status)
            success=reason is None
            retryable=reason not in {'binding_failure','redirect_binding_failure','snapshot_timestamp_header_mismatch','redirect_loop','redirect_limit'} and (status in TRANSIENT or reason in {'timeout','connection_reset'})
            outcome='success' if success else 'retryable_failure' if retryable else 'permanent_failure'
            result={'state':outcome,'reason':reason,'http_status':status,'redirect_chain':chain,
                    'execution':self.execution,'attempt':number,'candidate_id':r['candidate_id'],
                    'confirmed_snapshot_timestamp':r['confirmed_snapshot_timestamp'],
                    'body_length':len(response['body']),'sha256':digest(response['body'])}
            write_json(directory/'result.json',result,immutable=True)
            state['history'].append((directory/'result.json').relative_to(self.root).as_posix())
            state.update(state=outcome,reason=reason,sha256=result['sha256'],http_status=status)
            if success:
                folder=self.root/'raw'/r['candidate_id']/r['confirmed_snapshot_timestamp']
                headers=response['headers'];content_type=next((v for k,v in headers if k.lower()=='content-type'),'')
                provenance={**r,'input_record_sha256':digest(encoded(r)),
                    'input_manifest_sha256':self.input_sha,'exact_requested_wayback_url':r['exact_wayback_snapshot_url'],
                    'final_url':response['final_url'],'final_replay':parse_replay(response['final_url']),
                    'request_time':chain[0]['requested_at'],'response_time':chain[-1]['response_at'],
                    'http_status':status,'redirect_chain':chain,'content_type':content_type,
                    'body_length':len(response['body']),'sha256':result['sha256'],
                    'crawler_version':VERSION,'python_version':platform.python_version(),
                    'transport_backend':self.transport.identifier,'semantic_parsing_performed':False,
                    'raw_means_received_entity_bytes_not_origin_server_equivalence':True}
                files={'body.bin':response['body'],'headers.json':encoded(headers),
                       'request.json':encoded({'input':r,'requested_url':chain[0]['requested_url'],'requested_at':chain[0]['requested_at']}),
                       'response.json':encoded(chain[-1])}
                if content_type.split(';')[0].lower() in {'text/html','application/xhtml+xml'}:
                    files['page.html']=response['body']
                provenance['files_sha256']={name:digest(body) for name,body in files.items()}
                files['provenance.json']=encoded(provenance)  # Commit last, recoverable after interrupted state write.
                for name,body in files.items():write(folder/name,body,immutable=True)
                state['files']={name:digest(body) for name,body in files.items()}
            write_json(path,state)
            print(json.dumps({'candidate_id':r['candidate_id'],'attempt':number,'status':status,'state':outcome,'reason':reason}),flush=True)
            if not retryable:break
        return state

    def run(self,input_path):
        body=Path(input_path).read_bytes();self.input_sha=digest(body)
        # A portable SQLite transaction prevents two processes owning this output.
        owner=sqlite3.connect(self.root/'execution_lock.sqlite3',timeout=0)
        try:
            owner.execute('BEGIN IMMEDIATE')
            try:records=input_records(body)
            except (ValueError,TypeError,KeyError) as exc:
                write_json(self.root/'executions'/self.execution/'input_rejection.json',
                           {'input_sha256':self.input_sha,'reason':str(exc),'recorded_at':now()},immutable=True)
                raise
            if (self.root/'latest_manifest.json').exists():
                latest=read_json(self.root/'latest_manifest.json')
                if not re.fullmatch(r'[a-f0-9]{32}',latest['execution']):raise ValueError('invalid execution pointer')
                frozen=(self.root/'executions'/latest['execution']/'manifest.json').read_bytes()
                if digest(frozen)!=latest['manifest_sha256']:raise ValueError('prior retrieval manifest hash changed')
                for relative,expected in json.loads(frozen)['files'].items():
                    location=(self.root/relative).resolve()
                    if not location.is_relative_to(self.root) or digest(location.read_bytes())!=expected:
                        raise ValueError('prior response commitment changed')
            write(self.root/'input_manifest.json',body,immutable=True)
            write_json(self.root/'executions'/self.execution/'settings.json',
                       {'crawler_version':VERSION,'input_sha256':self.input_sha,'started_at':now(),**self.settings},immutable=True)
            if self.concurrency==1:states=[self._fetch(r) for r in records]
            else:
                with ThreadPoolExecutor(max_workers=2) as pool:states=list(pool.map(self._fetch,records))
            summary=self.summarize(records,states)
            write_json(self.root/'executions'/self.execution/'manifest.json',summary,immutable=True)
            write_json(self.root/'latest_manifest.json',{'execution':self.execution,'manifest_sha256':digest(encoded(summary))})
            return summary
        finally:
            owner.rollback();owner.close()

    def summarize(self,records,states):
        results=[];files={}
        for r,state in zip(records,states):
            for relative in state['history']:
                result=read_json(self.root/relative)
                if result['execution']==self.execution:results.append(result)
            for parent in (self.root/'attempts'/r['candidate_id']/r['confirmed_snapshot_timestamp'],
                           self.root/'raw'/r['candidate_id']/r['confirmed_snapshot_timestamp']):
                if parent.exists():
                    for p in parent.rglob('*'):
                        if p.is_file():files[p.relative_to(self.root).as_posix()]=digest(p.read_bytes())
        hops=[h for r in results for h in r['redirect_chain']]
        statuses=Counter(str(h['http_status']) for h in hops)
        successes=[]
        for r,s in zip(records,states):
            if s['state']=='success':successes.append(read_json(self.root/'raw'/r['candidate_id']/r['confirmed_snapshot_timestamp']/'provenance.json'))
        html=sum(p['content_type'].split(';')[0].lower() in {'text/html','application/xhtml+xml'} for p in successes)
        return {'crawler_version':VERSION,'execution':self.execution,'completed_at':now(),
                'input_manifest_sha256':self.input_sha,'input_count':len(records),'settings':self.settings,
                'state_counts':dict(Counter(s['state'] for s in states)),
                'raw_content_successes':len(successes),'successful_html_bodies':html,'successful_non_html_bodies':len(successes)-html,
                'http_status_counts_this_execution':dict(statuses),'redirect_responses':sum(h['http_status'] in REDIRECTS for h in hops),
                'responses_429':statuses['429'],'other_4xx':sum(400<=h['http_status']<500 and h['http_status']!=429 for h in hops if h['http_status'] is not None),
                'responses_5xx':sum(500<=h['http_status']<600 for h in hops if h['http_status'] is not None),
                'timeouts':sum(h['error']=='timeout' for h in hops),
                'binding_failures':sum(r['reason'] in {'binding_failure','redirect_binding_failure','snapshot_timestamp_header_mismatch'} for r in results),
                'failure_reasons':dict(Counter(r['reason'] for r in results if r['reason'])),
                'attempt_results':results,'successful_responses':successes,'files':files,
                'scientific_interpretation_performed':False}
