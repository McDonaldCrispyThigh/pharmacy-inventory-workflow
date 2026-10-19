"""Exact-capture raw HTTP delivery; explicit proxy, verified TLS, local parsing.

This experiment-local module leaves the approved transport implementation intact.
Raw entity bytes are retained before decoding; redirects are manually inspected.
Proxy credentials and request authentication headers are never persisted.
"""
import base64
import hashlib
import math
import re
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from urllib3.exceptions import HTTPError as StreamError

from archive_transport_binding import snapshot_identity, urls_bind, normalize_site_url
from archive_exact_page import verify_official_snapshot, validate_direct_page, ArchivePageRejected
from corrected_run_store import sha, seal_json, read_sealed, utcnow, unit_key
from wayback_state import WaybackDeferred, _redact_unsafe_url


class ProxyNotConfigured(ValueError):
    pass


def same_capture(expected, url):
    try:
        actual = snapshot_identity(url)
        return actual['timestamp'] == expected['timestamp'] and urls_bind(actual['original_url'], expected['original_url'])
    except (ValueError, TypeError):
        return False


def secure_replay(url):
    """Upgrade only the outer replay scheme/authority; retain replay spelling."""
    snapshot_identity(url)
    p = urlsplit(url)
    return urlunsplit(('https', 'web.archive.org', p.path, p.query, ''))


def proxy_settings(value, provider):
    if not value:
        raise ProxyNotConfigured('WAYBACK_RAW_PROXY_URL is not configured')
    try:
        p = urlsplit(value)
        if p.scheme not in {'http', 'https'} or not p.hostname or p.path not in {'', '/'} or p.query or p.fragment:
            raise ValueError()
        p.port
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,60}', provider or ''):
            raise ValueError()
    except ValueError:
        raise ProxyNotConfigured('invalid proxy URL or provider label') from None
    return value, provider


def response_body(receipt):
    body = base64.b64decode(receipt['body_base64'], validate=True)
    if len(body) != receipt['body_bytes'] or hashlib.sha256(body).hexdigest() != receipt['body_sha256']:
        raise ValueError('raw body commitment differs')
    return body


class RawSnapshotTransport:
    def __init__(self, root, store, scope, route, *, proxy_url=None, provider='http_connect',
                 session=None, clock=time.time, timeout=20, max_bytes=2_000_000, max_redirects=5):
        if route not in {'direct', 'proxy'}:
            raise ValueError('raw route must be direct or proxy')
        self.root, self.store, self.scope, self.route = root, store, scope, route
        self.clock, self.timeout, self.max_bytes, self.max_redirects = clock, timeout, max_bytes, max_redirects
        self.provider = 'internet_archive_local' if route == 'direct' else provider
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.session.proxies = {}
        if route == 'proxy':
            value, self.provider = proxy_settings(proxy_url, provider)
            self.session.proxies = {'https': value}

    def _receipt(self, url, candidate_id, hop):
        path = self.root/'raw'/self.route/unit_key(candidate_id)/f'{hop}.json'
        if path.exists():
            record = read_sealed(path)
            if (record['requested_url'] != url or record['scope'] != self.scope
                    or record['candidate_id'] != candidate_id or record['provider'] != self.provider):
                raise ValueError('raw request scope changed')
            response_body(record)
            return record
        admission_key = self.route + ':' + self.provider
        with self.store.locked('raw_http_'+admission_key):
            admission = self.store.get_state('raw_admission', admission_key) or {}
            if admission.get('next_at', 0) > self.clock():
                raise WaybackDeferred('raw_transport_cooldown', admission['next_at'])
            started = utcnow()
            status, body, headers, final_url, error = None, b'', {}, url, None
            try:
                with self.session.get(url, allow_redirects=False, stream=True, timeout=self.timeout,
                        verify=True, headers={'Accept-Encoding':'identity',
                        'User-Agent':'NETS-AI-honors-thesis-exact-retrieval/1.0'}) as response:
                    status, final_url = response.status_code, response.url
                    allowed = {'content-type','content-length','content-encoding','location','retry-after','memento-datetime','date','etag','last-modified'}
                    headers = {k.lower():v for k,v in response.headers.items() if k.lower() in allowed}
                    if 'location' in headers:
                        headers['location'] = _redact_unsafe_url(headers['location'])
                    body = response.raw.read(self.max_bytes+1, decode_content=False)
            except (requests.RequestException, StreamError, TimeoutError, OSError) as exc:
                error = type(exc).__name__
            deadline = self.clock()+1
            if status in (429,503):
                deadline = max(deadline, self.clock()+5)
                retry = headers.get('retry-after')
                if retry is not None:
                    try:
                        delay = float(retry)
                        if math.isfinite(delay) and delay >= 0:
                            deadline = max(deadline, self.clock()+delay)
                    except (ValueError,TypeError):
                        try:
                            date = parsedate_to_datetime(retry)
                            if date.tzinfo:
                                deadline = max(deadline,date.timestamp())
                        except (ValueError,TypeError,OverflowError):
                            pass
            self.store.set_state('raw_admission',admission_key,{'next_at':deadline})
            record = {'schema':'nets-ai-exact-raw-response-v1','scope':self.scope,
                      'route':self.route,'provider':self.provider,'candidate_id':candidate_id,
                      'requested_at':started,'received_at':utcnow(),'requested_url':url,
                      'final_url':_redact_unsafe_url(final_url),'http_status':status,'headers':headers,
                      'body_base64':base64.b64encode(body).decode('ascii'),
                      'body_sha256':hashlib.sha256(body).hexdigest(),'body_bytes':len(body),
                      'body_truncated':len(body)>self.max_bytes,'read_error':error,
                      'tls_verification':True,'automatic_redirects':False,'content_transformed':False}
            seal_json(path,record)
            return record

    def fetch(self, candidate_id, website, snapshot):
        verify_official_snapshot(self.store,website,snapshot)
        url = secure_replay(snapshot['exact_url'])
        chain = []
        for hop in range(self.max_redirects+1):
            if not same_capture(snapshot,url):
                raise ArchivePageRejected('raw_redirect_binding_failed')
            record = self._receipt(url,candidate_id,hop)
            chain.append(record)
            if not same_capture(snapshot,record['final_url']):
                break
            if record['http_status'] not in {301,302,303,307,308}:
                break
            location = record['headers'].get('location')
            if not location or not same_capture(snapshot,urljoin(url,location)):
                break
            url = secure_replay(urljoin(url,location))
            if url in [r['requested_url'] for r in chain]:
                break
        return assess_chain(chain,website,snapshot,self.route,self.provider,self.scope,candidate_id)


def assess_chain(chain, website, snapshot, route, provider, scope, candidate_id):
    if not chain:
        raise ValueError('missing raw response chain')
    expected_url = secure_replay(snapshot['exact_url'])
    result = {'route':route,'provider':provider,'official_snapshot':snapshot,
              'candidate_url_raw':website,'candidate_url_normalized':normalize_site_url(website),
              'original_url_raw':snapshot['original_url'],
              'original_url_normalized':normalize_site_url(snapshot['original_url']),
              'redirect_chain':[{'requested_url':r['requested_url'],'final_url':r['final_url'],
                  'http_status':r['http_status'],'location':r['headers'].get('location'),
                  'receipt_payload_sha256':sha(r)} for r in chain]}
    try:
        for index,r in enumerate(chain):
            response_body(r)
            if (r['route']!=route or r['provider']!=provider or r['scope']!=scope
                    or r['candidate_id']!=candidate_id or r['requested_url']!=expected_url
                    or not r['tls_verification'] or r['content_transformed'] or r['automatic_redirects']):
                raise ArchivePageRejected('raw_provenance_binding_failed')
            if not same_capture(snapshot,r['requested_url']) or not same_capture(snapshot,r['final_url']):
                raise ArchivePageRejected('raw_response_binding_failed')
            if r['read_error']:
                raise ArchivePageRejected('raw_response_read_failed')
            if r['body_truncated']:
                raise ArchivePageRejected('truncated_archived_response')
            if r['http_status'] in {301,302,303,307,308}:
                target = urljoin(r['requested_url'],r['headers'].get('location',''))
                if not r['headers'].get('location') or not same_capture(snapshot,target):
                    raise ArchivePageRejected('raw_redirect_binding_failed')
                expected_url = secure_replay(target)
                if index == len(chain)-1:
                    raise ArchivePageRejected('raw_redirect_incomplete')
            elif index != len(chain)-1:
                raise ArchivePageRejected('raw_redirect_chain_invalid')
        r=chain[-1]
        if r['headers'].get('content-encoding','identity').lower() not in {'identity',''}:
            raise ArchivePageRejected('raw_content_encoding_unsupported')
        parsed=validate_direct_page({'status':r['http_status'],'body':response_body(r),
            'headers':r['headers'],'final_url':r['final_url'],'provenance_sha256':sha(r),
            'retrieved_at':r['received_at'],'body_truncated':r['body_truncated']},website,snapshot)
        parsed.update(source_type='direct_raw' if route=='direct' else 'proxy_raw',
                      source_fidelity='original_http',lossless_archive_response=True)
        return {**result,**parsed,'http_status':r['http_status']}
    except ArchivePageRejected as exc:
        return {**result,'status':'failure','reason':exc.reason,'http_status':chain[-1]['http_status']}
