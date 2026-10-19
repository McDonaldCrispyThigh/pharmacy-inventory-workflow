"""Private durable Wayback state and content-addressed HTTP provenance.

SQLite locks serialize a provider's request admission across processes. No
credentials or request authorization headers belong in this store. Hashes detect
corruption; they are integrity commitments, not signatures from either provider.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit


class WaybackDeferred(RuntimeError):
    """Acquisition is pending and must not be converted into missing evidence."""

    def __init__(self, reason: str, retry_at: float | None = None):
        if not isinstance(reason, str) or not re.fullmatch(r'[a-z0-9_]{1,100}', reason):
            raise ValueError('invalid deferred reason')
        if retry_at is not None and (type(retry_at) not in (int, float) or not math.isfinite(retry_at)):
            raise ValueError('invalid retry deadline')
        self.reason, self.retry_at = reason, retry_at
        super().__init__('Wayback acquisition deferred: ' + reason)

    def to_dict(self):
        return {'status': 'pending', 'reason': self.reason, 'retry_at': self.retry_at}


class WaybackStateError(RuntimeError):
    """A private checkpoint or provenance commitment could not be verified."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _redact_unsafe_url(value):
    """Keep exact safe spelling, but do not retain credentials in error URLs."""
    if value is None:
        return None
    sensitive = re.compile(r'^(?:key|api[-_]?key|access[-_]?key|.*token.*|.*secret.*|.*password.*|authorization|auth|signature|credential.*)$', re.I)
    def unsafe(url, depth=0):
        if depth > 4:
            return True
        try:
            parsed = urlsplit(url)
            if parsed.username is not None or parsed.password is not None:
                return True
            decoded = unquote(url)
            if decoded != url and unsafe(decoded, depth + 1):
                return True
            for key, item in parse_qsl(parsed.query, keep_blank_values=True):
                if sensitive.fullmatch(unquote(key)):
                    return True
                if item.lower().startswith(('http://', 'https://')) and unsafe(item, depth + 1):
                    return True
            if parsed.netloc.lower() == 'web.archive.org' and '/http' in parsed.path.lower():
                nested = parsed.path[parsed.path.lower().index('/http') + 1:]
                if unsafe(nested + (('?' + parsed.query) if parsed.query else ''), depth + 1):
                    return True
            return False
        except (TypeError, ValueError):
            return True
    return '[redacted-unsafe-url]' if unsafe(value) else value


class WaybackStateStore:
    def __init__(self, root: Path, clock=None):
        self.root = Path(root)
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _now(self):
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError('state clock must be timezone aware')
        return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

    @contextmanager
    def _connection(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / 'state.sqlite3', timeout=30)
        try:
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('CREATE TABLE IF NOT EXISTS state (namespace TEXT, key TEXT, payload BLOB NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(namespace,key))')
            yield connection
        finally:
            connection.close()

    def get_state(self, namespace, key):
        with self._connection() as connection:
            row = connection.execute('SELECT payload,sha256 FROM state WHERE namespace=? AND key=?',
                                     (namespace, key)).fetchone()
        if row is None:
            return None
        try:
            if hashlib.sha256(row[0]).hexdigest() != row[1]:
                raise ValueError('checksum')
            value = json.loads(row[0])
            if not isinstance(value, dict):
                raise ValueError('shape')
            return value
        except (ValueError, TypeError) as exc:
            raise WaybackStateError('Wayback checkpoint integrity failed') from exc

    def set_state(self, namespace, key, value):
        if not isinstance(value, dict):
            raise TypeError('state must be an object')
        encoded = canonical_json(value)
        with self._connection() as connection, connection:
            connection.execute('INSERT OR REPLACE INTO state VALUES (?,?,?,?)',
                               (namespace, key, encoded, hashlib.sha256(encoded).hexdigest()))

    @contextmanager
    def locked(self, name):
        """OS-released SQLite lock, including after process death; no stale leases."""
        directory = self.root / 'locks'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (hashlib.sha256(name.encode()).hexdigest() + '.sqlite3')
        connection = sqlite3.connect(path, timeout=1)
        try:
            try:
                connection.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as exc:
                raise WaybackDeferred('provider_busy') from exc
            yield
        finally:
            connection.rollback()
            connection.close()

    def record_response(self, provider, request_url, response_status, headers, body,
                        final_url, metadata=None):
        if provider not in {'direct_wayback', 'outscraper_ai'} or not isinstance(body, bytes):
            raise ValueError('invalid raw response')
        # Response headers may contain cookies or provider-private information.
        # Retain only transport evidence needed to reproduce parsing/backoff.
        allowed = {'content-type', 'content-length', 'content-encoding', 'retry-after',
                   'date', 'etag', 'last-modified', 'memento-datetime', 'location'}
        retained_headers = {str(k).lower(): str(v) for k, v in headers.items()
                            if str(k).lower() in allowed}
        if 'location' in retained_headers:
            retained_headers['location'] = _redact_unsafe_url(retained_headers['location'])
        record = {'schema_version': 'wayback_raw_response_v1', 'provider': provider,
                  'request_url': _redact_unsafe_url(request_url), 'response_status': response_status,
                  'headers': retained_headers, 'body_base64': base64.b64encode(body).decode('ascii'),
                  'response_sha256': hashlib.sha256(body).hexdigest(), 'response_bytes': len(body),
                  'final_url': _redact_unsafe_url(final_url), 'retrieved_at': self._now(),
                  'response_received': response_status is not None, 'metadata': metadata or {}}
        encoded = canonical_json(record)
        digest = hashlib.sha256(encoded).hexdigest()
        directory = self.root / 'responses' / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (digest + '.json')
        if path.exists():
            self.read_record(digest)
            return digest
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, prefix='.response-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return digest

    def read_record(self, digest):
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise WaybackStateError('invalid raw response reference')
        try:
            raw = (self.root / 'responses' / digest[:2] / (digest + '.json')).read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError('envelope checksum')
            value = json.loads(raw)
            body = base64.b64decode(value['body_base64'], validate=True)
            if (value['schema_version'] != 'wayback_raw_response_v1'
                    or len(body) != value['response_bytes']
                    or hashlib.sha256(body).hexdigest() != value['response_sha256']):
                raise ValueError('body checksum')
            return value
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise WaybackStateError('raw Wayback provenance integrity failed') from exc
