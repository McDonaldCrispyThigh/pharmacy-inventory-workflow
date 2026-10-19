"""Independently check outsourced snapshot identity with Wayback Availability.

Availability verifies capture metadata only. Even a confirmed result never
verifies extracted page text, an outlet relationship, or an opening statement.
Official response contract: https://archive.org/help/wayback_api.php
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from v2_evidence_models import canonical_http_url

AVAILABILITY_API = "https://archive.org/wayback/available"
SCHEMA_VERSION = "availability_snapshot_verification_v1"
_CHECKS = "availability_verification"
_CONTROL = "availability_control"


@dataclass(frozen=True, repr=False)
class AvailabilityResponse:
    status: int
    body: bytes
    headers: dict[str, str]
    final_url: str


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AvailabilityTransport:
    """Read only the fixed official Availability endpoint without credentials."""

    def __init__(self, max_response_bytes=1_000_000):
        self.max_response_bytes = max_response_bytes
        self.opener = build_opener(_NoRedirects())

    def get(self, url, params, timeout, *, follow_redirects=False):
        if url != AVAILABILITY_API or follow_redirects:
            raise ValueError("Availability transport requires its fixed official endpoint")
        pairs = tuple(params)
        if len(pairs) != 2 or {key for key, _ in pairs} != {"url", "timestamp"}:
            raise ValueError("Availability transport requires exact URL and timestamp parameters")
        values = dict(pairs)
        canonical_http_url(values["url"])
        _timestamp(values["timestamp"])
        request = Request(url + "?" + urlencode(pairs), headers={
            "Accept": "application/json",
            "User-Agent": "NETS-AI-honors-thesis-archive-verification/1.0",
        })
        try:
            response = self.opener.open(request, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            return AvailabilityResponse(response.code,
                                        response.read(self.max_response_bytes + 1),
                                        dict(response.headers.items()), response.geturl())


def _timestamp(value):
    if type(value) not in (str, int) or re.fullmatch(r"\d{14}", str(value)) is None:
        raise ValueError("invalid archive timestamp")
    value = str(value)
    datetime.strptime(value, "%Y%m%d%H%M%S")
    return value


def _replay_identity(value):
    if not isinstance(value, str):
        raise ValueError("invalid snapshot URL")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or parsed.netloc != "web.archive.org"
            or parsed.fragment):
        raise ValueError("invalid snapshot host")
    tail = parsed.path + (("?" + parsed.query) if parsed.query else "")
    match = re.fullmatch(r"/web/(\d{14})(?:id_)?/(https?://.+)", tail)
    if match is None:
        raise ValueError("invalid snapshot path")
    return _timestamp(match[1]), canonical_http_url(match[2])


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _source_identity(source):
    original = canonical_http_url(_field(source, "original_url"))
    timestamp, replay_original = _replay_identity(_field(source, "replay_url"))
    captured = _field(source, "captured_at")
    if isinstance(captured, str):
        captured = datetime.fromisoformat(captured.replace("Z", "+00:00"))
    if (not isinstance(captured, datetime) or captured.tzinfo is None
            or captured.microsecond != 0
            or captured.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S") != timestamp
            or original != replay_original):
        raise ValueError("source snapshot identity is inconsistent")
    return original, timestamp


def assess_availability(body, original, timestamp):
    """Validate the official closest-capture response, never page contents."""
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict) or not isinstance(payload.get("archived_snapshots"), dict):
            raise ValueError("invalid availability envelope")
        if "url" in payload and canonical_http_url(payload["url"]) != original:
            return {"status": "mismatch", "reason": "response_query_url_mismatch"}
        if "timestamp" in payload and _timestamp(payload["timestamp"]) != timestamp:
            return {"status": "mismatch", "reason": "response_query_timestamp_mismatch"}
        snapshots = payload["archived_snapshots"]
        if not snapshots:
            return {"status": "unavailable", "reason": "no_currently_accessible_snapshot"}
        closest = snapshots.get("closest")
        if not isinstance(closest, dict) or type(closest.get("available")) is not bool:
            raise ValueError("invalid closest snapshot")
        if closest["available"] is False:
            return {"status": "unavailable", "reason": "reported_snapshot_not_accessible"}
        returned_timestamp = _timestamp(closest.get("timestamp"))
        returned_url = closest.get("url")
        try:
            replay_timestamp, replay_original = _replay_identity(returned_url)
        except (ValueError, TypeError, UnicodeError):
            return {"status": "mismatch", "reason": "returned_snapshot_identity_invalid"}
        details = {"returned_snapshot_url": returned_url,
                   "returned_timestamp": returned_timestamp,
                   "returned_original_url": replay_original}
        if returned_timestamp != timestamp or replay_timestamp != timestamp:
            return {"status": "mismatch", "reason": "closest_timestamp_mismatch", **details}
        if replay_original != original:
            return {"status": "mismatch", "reason": "closest_original_url_mismatch", **details}
        if type(closest.get("status")) not in (str, int) or str(closest["status"]) != "200":
            return {"status": "unavailable", "reason": "closest_capture_not_http_200", **details}
        return {"status": "confirmed", "reason": "exact_snapshot_identity_confirmed", **details}
    except (ValueError, TypeError, UnicodeError):
        return {"status": "transport_failure", "reason": "invalid_availability_response"}


class _Verifier:
    def __init__(self, store, transport, *, scope, clock, sleep, max_attempts,
                 timeout, min_interval, retry_delay, rate_limit_backoff,
                 max_response_bytes):
        if not isinstance(scope, str) or not scope.strip() or len(scope) > 256:
            raise ValueError("invalid Availability verification scope")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("Availability attempts must be in [1,3]")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 4_000_000:
            raise ValueError("invalid Availability response byte bound")
        for name, value, lower, upper in (
            ("timeout", timeout, 0, 60), ("min_interval", min_interval, 0, 10),
            ("retry_delay", retry_delay, 0, 30), ("rate_limit_backoff", rate_limit_backoff, 0, 3600),
        ):
            if type(value) not in (int, float) or not math.isfinite(value) or not lower < value <= upper:
                raise ValueError("invalid Availability " + name)
        self.store, self.transport, self.scope = store, transport, scope
        self.clock, self.sleep = clock or time.time, sleep
        self.max_attempts, self.timeout = max_attempts, timeout
        self.min_interval, self.retry_delay = min_interval, retry_delay
        self.rate_limit_backoff, self.max_response_bytes = rate_limit_backoff, max_response_bytes

    def now(self):
        value = self.clock()
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("Availability clock must be timezone aware")
            return value.timestamp()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid Availability clock")
        return value

    def _result(self, original, timestamp, request_url, state, status, reason, **extra):
        return {"schema_version": SCHEMA_VERSION, "status": status, "reason": reason,
                "original_url": original, "requested_timestamp": timestamp,
                "request_url": request_url, "verification_endpoint": AVAILABILITY_API,
                "verification_scope": "snapshot_identity_and_timestamp_only",
                "snapshot_identity_confirmed": status == "confirmed",
                "independently_verified_text": False, "opening_statement_verified": False,
                "checked_at": datetime.fromtimestamp(self.now(), timezone.utc).isoformat().replace("+00:00", "Z"),
                "attempts": state["attempts"], "raw_response_sha256": list(state["raw_response_sha256"]),
                "provenance_sha256": state["raw_response_sha256"][-1] if state["raw_response_sha256"] else None,
                "retryable": False, "retry_at": None, **extra}

    def _save(self, key, state, result):
        state["result"] = result
        self.store.set_state(_CHECKS, key, state)
        return result

    def _cached(self, state, request_url, original, timestamp):
        for sha in state["raw_response_sha256"]:
            record = self.store.read_record(sha)
            metadata = record["metadata"]
            if (record["provider"] != "direct_wayback" or record["request_url"] != request_url
                    or metadata.get("verification_kind") != SCHEMA_VERSION
                    or metadata.get("verification_run_scope") != self.scope
                    or metadata.get("original_url") != original
                    or metadata.get("requested_timestamp") != timestamp):
                raise ValueError("Availability cache provenance binding failed")
        result = state.get("result")
        if result and result["status"] in {"confirmed", "mismatch", "unavailable"}:
            if not state["raw_response_sha256"]:
                raise ValueError("missing Availability verification provenance")
            record = self.store.read_record(state["raw_response_sha256"][-1])
            assessed = assess_availability(base64.b64decode(record["body_base64"], validate=True), original, timestamp)
            if (record["response_status"] != 200 or record["final_url"] != request_url
                    or assessed["status"] != result["status"] or assessed["reason"] != result["reason"]):
                raise ValueError("Availability cached verification does not match its raw response")
        return result

    def _rate_limit(self, headers, control):
        now = self.now()
        deadline = now + self.rate_limit_backoff
        raw = next((value for key, value in headers.items() if str(key).lower() == "retry-after"), None)
        if raw is not None:
            try:
                delay = float(raw)
                if math.isfinite(delay) and delay >= 0:
                    deadline = max(deadline, now + delay)
            except (ValueError, TypeError):
                try:
                    date = parsedate_to_datetime(str(raw))
                    if date.tzinfo is not None:
                        deadline = max(deadline, date.timestamp())
                except (ValueError, TypeError, OverflowError):
                    pass
        control["cooldown_until"] = max(deadline, control.get("cooldown_until", 0))
        self.store.set_state(_CONTROL, "admission", control)
        return control["cooldown_until"]

    def verify(self, original, timestamp):
        params = (("url", original), ("timestamp", timestamp))
        request_url = AVAILABILITY_API + "?" + urlencode(params)
        key = hashlib.sha256(json.dumps([SCHEMA_VERSION, self.scope, original, timestamp],
                                       separators=(",", ":")).encode()).hexdigest()
        with self.store.locked("wayback_availability"):
            state = self.store.get_state(_CHECKS, key) or {"attempts": 0, "raw_response_sha256": []}
            prior = self._cached(state, request_url, original, timestamp)
            if prior and not prior["retryable"]:
                return prior
            while state["attempts"] < self.max_attempts:
                control = self.store.get_state(_CONTROL, "admission") or {}
                if self.now() < control.get("cooldown_until", 0):
                    return self._save(key, state, self._result(
                        original, timestamp, request_url, state, "transport_failure", "availability_cooldown",
                        retryable=True, retry_at=control["cooldown_until"],
                    ))
                deadline = max(control.get("next_request_at", 0), (prior or {}).get("retry_at") or 0)
                delay = deadline - self.now()
                if delay > 0:
                    if delay > 30:
                        return self._save(key, state, self._result(
                            original, timestamp, request_url, state, "transport_failure", "retry_wait_pending",
                            retryable=True, retry_at=deadline,
                        ))
                    self.sleep(delay)
                    if self.now() < deadline:
                        return self._save(key, state, self._result(
                            original, timestamp, request_url, state, "transport_failure", "retry_wait_pending",
                            retryable=True, retry_at=deadline,
                        ))
                # Reads are idempotent, but count interrupted attempts so a
                # repeatedly dying worker cannot create an unbounded retry loop.
                state["attempts"] += 1
                state.pop("result", None)
                self.store.set_state(_CHECKS, key, state)
                control["next_request_at"] = self.now() + self.min_interval
                self.store.set_state(_CONTROL, "admission", control)
                try:
                    response = self.transport.get(AVAILABILITY_API, params, self.timeout, follow_redirects=False)
                except (OSError, URLError, HTTPException) as exc:
                    prior = self._result(original, timestamp, request_url, state, "transport_failure",
                                         "availability_transport_error", error_type=type(exc).__name__,
                                         retryable=state["attempts"] < self.max_attempts,
                                         retry_at=self.now() + self.retry_delay if state["attempts"] < self.max_attempts else None)
                    self._save(key, state, prior)
                    continue
                headers = dict(response.headers)
                retry_at = self._rate_limit(headers, control) if response.status == 429 else None
                raw = response.body
                if not isinstance(raw, bytes):
                    raise TypeError("Availability transport must return response bytes")
                oversized = len(raw) > self.max_response_bytes
                final_url = response.final_url or request_url
                metadata = {"verification_kind": SCHEMA_VERSION, "verification_run_scope": self.scope,
                            "original_url": original, "requested_timestamp": timestamp,
                            "verification_scope": "snapshot_identity_and_timestamp_only",
                            "independently_verified_text": False, "opening_statement_verified": False,
                            "body_truncated": oversized, "attempt": state["attempts"]}
                sha = self.store.record_response("direct_wayback", request_url, response.status,
                                                 headers, raw[:self.max_response_bytes + 1], final_url,
                                                 metadata=metadata)
                state["raw_response_sha256"].append(sha)
                if response.status == 429:
                    return self._save(key, state, self._result(
                        original, timestamp, request_url, state, "transport_failure", "availability_http_429",
                        http_status=429, retryable=state["attempts"] < self.max_attempts,
                        retry_at=retry_at if state["attempts"] < self.max_attempts else None,
                    ))
                if final_url != request_url:
                    return self._save(key, state, self._result(
                        original, timestamp, request_url, state, "transport_failure", "unexpected_response_url",
                        http_status=response.status,
                    ))
                if oversized:
                    return self._save(key, state, self._result(
                        original, timestamp, request_url, state, "transport_failure", "oversized_availability_response",
                        http_status=response.status,
                    ))
                if response.status == 200:
                    assessed = assess_availability(raw, original, timestamp)
                    transient = assessed["status"] == "transport_failure"
                else:
                    assessed = {"status": "transport_failure", "reason": "availability_http_failure"}
                    transient = response.status >= 500 or response.status in (408, 425)
                retryable = transient and state["attempts"] < self.max_attempts
                prior = self._result(original, timestamp, request_url, state, **assessed,
                                     http_status=response.status, retryable=retryable,
                                     retry_at=self.now() + self.retry_delay if retryable else None)
                self._save(key, state, prior)
                if not retryable:
                    return prior
            if prior and not prior["retryable"]:
                return prior
            return self._save(key, state, self._result(
                original, timestamp, request_url, state, "transport_failure", "verification_attempts_exhausted",
            ))


def verify_collection(collection, state_store, transport=None, *, scope="standalone", clock=None,
                      sleep=time.sleep, max_attempts=2, timeout=15, min_interval=1.0,
                      retry_delay=2.0, rate_limit_backoff=30.0, max_response_bytes=1_000_000):
    """Return JSON-friendly checks for outsourced snapshots without mutating evidence.

    A received 429 stops retrying immediately and persists a shared cooldown.
    Its record has ``status='transport_failure'`` and a retry deadline when a
    later bounded attempt is available. Unavailable means the endpoint reported
    no accessible capture; it never proves the site has never been archived.
    Supply the corrected run ID as ``scope`` when reusing a shared private store.
    """
    evidence = _field(collection, "archive_evidence")
    sources = _field(evidence, "source_snapshots")
    if not isinstance(sources, (list, tuple)):
        raise ValueError("verification requires a Wayback collection with source snapshots")
    outsourced = [source for source in sources if _field(source, "retrieval_source") == "outscraper_ai"]
    if not outsourced:
        return []
    verifier = _Verifier(state_store, transport or AvailabilityTransport(max_response_bytes),
                         scope=scope, clock=clock, sleep=sleep, max_attempts=max_attempts,
                         timeout=timeout, min_interval=min_interval, retry_delay=retry_delay,
                         rate_limit_backoff=rate_limit_backoff, max_response_bytes=max_response_bytes)
    records = []
    for source in outsourced:
        original, timestamp = _source_identity(source)
        result = verifier.verify(original, timestamp)
        records.append({**result, "source_evidence_id": _field(source, "evidence_id"),
                        "source_provenance_sha256": _field(source, "provenance_sha256"),
                        "source_retrieval_source": "outscraper_ai",
                        "requested_replay_url": _field(source, "replay_url")})
    return records
