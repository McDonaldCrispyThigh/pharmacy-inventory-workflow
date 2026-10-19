"""Bounded official-only HTTP with shared host cooldown and immutable receipts."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
from http.client import HTTPException
from urllib.parse import urlsplit, quote
import base64
import time
import math
import re

from corrected_run_store import sha
from wayback_state import WaybackDeferred


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def official_url(url):
    p=urlsplit(url)
    if p.scheme!='https' or p.username or p.password or p.fragment:
        raise ValueError('official HTTPS archive endpoint required')
    valid=(p.netloc=='archive.org' and p.path=='/wayback/available') or (
        p.netloc=='web.archive.org' and (p.path=='/cdx/search/cdx' or p.path.startswith('/web/timemap/link/')
            or bool(re.match(r'/web/\d{14}(?:id_)?/https?://',p.path))))
    if not valid:
        raise ValueError('non-official or unsupported archive endpoint')
    return p.hostname


class OfficialHTTP:
    def __init__(self, store, scope, *, timeout=15, max_body_bytes=2000000,
                 interval=1, backoff=5, clock=time.time, opener=None):
        self.store,self.scope=store,scope
        self.timeout,self.max_body_bytes=timeout,max_body_bytes
        self.interval,self.backoff=interval,backoff
        self.clock,self.opener=clock,opener or build_opener(NoRedirects())

    def get(self, url, *, method, candidate_id):
        url=quote(url,safe=":/?&=%;,@!$'()*+-._~[]")
        host=official_url(url)
        key=sha([self.scope,method,candidate_id,url])
        cached=self.store.get_state('official_requests',key)
        if cached:
            record=self.store.read_record(cached['provenance_sha256'])
            if record['request_url']!=url or record['metadata']['scope']!=self.scope:
                raise ValueError('official request checkpoint mismatch')
            return self._response(record,cached['provenance_sha256'])
        try:
            with self.store.locked('official_http_'+host):
                state=self.store.get_state('official_host_admission',host) or {}
                if state.get('next_at',0)>self.clock():
                    raise WaybackDeferred('official_host_cooldown',state['next_at'])
                request=Request(url,headers={'User-Agent':'NETS-AI-honors-thesis-archive-transport/2.0',
                    'Accept':'text/html,application/json,application/link-format,text/plain;q=0.9','Accept-Encoding':'identity'})
                error=None
                try:
                    try:
                        response=self.opener.open(request,timeout=self.timeout)
                    except HTTPError as exc:
                        response=exc
                    with response:
                        status=response.code;body=response.read(self.max_body_bytes+1)
                        headers=dict(response.headers.items());final_url=response.geturl()
                except (URLError,TimeoutError,OSError,HTTPException) as exc:
                    status=None;body=b'';headers={};final_url=url
                    error=type(exc).__name__
                metadata={'scope':self.scope,'method':method,'candidate_id':candidate_id,
                          'error_type':error,'body_truncated':len(body)>self.max_body_bytes}
                if 'Location' in headers or 'location' in headers:
                    metadata['location']=headers.get('Location',headers.get('location'))
                now=self.clock();next_at=now+self.interval
                if status in (429,503):
                    next_at=max(next_at,now+self.backoff)
                    retry=next((v for k,v in headers.items() if k.lower()=='retry-after'),None)
                    if retry is not None:
                        try:
                            delay=float(retry)
                            if math.isfinite(delay) and delay>=0:next_at=max(next_at,now+delay)
                        except (ValueError,TypeError):
                            try:
                                when=parsedate_to_datetime(retry)
                                if when.tzinfo:next_at=max(next_at,when.timestamp())
                            except (ValueError,TypeError,OverflowError):pass
                self.store.set_state('official_host_admission',host,{'next_at':next_at,'status':status,'provenance_sha256':None})
                digest=self.store.record_response('direct_wayback',url,status,headers,body,final_url,metadata=metadata)
                self.store.set_state('official_host_admission',host,{'next_at':next_at,'status':status,'provenance_sha256':digest})
                self.store.set_state('official_requests',key,{'provenance_sha256':digest})
                return self._response(self.store.read_record(digest),digest)
        except WaybackDeferred as exc:
            if exc.retry_at is None:
                raise WaybackDeferred('official_provider_busy',self.clock()+1) from None
            raise

    @staticmethod
    def _response(record,digest):
        return {'status':record['response_status'],'body':base64.b64decode(record['body_base64']),
                'headers':record.get('headers',{}),'final_url':record['final_url'],
                'request_url':record['request_url'],'provenance_sha256':digest,
                'body_truncated':record['metadata'].get('body_truncated',False),
                'retrieved_at':record['retrieved_at'],'metadata':record['metadata']}
