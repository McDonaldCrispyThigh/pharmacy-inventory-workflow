"""Durable, bounded Outscraper fallback for exact Wayback CDX and replay URLs.

Provider envelopes are retained privately by WaybackStateStore. AI-extracted
bytes have structural validation only: they are never labelled original Archive
response bytes or evidence of lossless retrieval. Pending jobs do not sleep here.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values

from v2_evidence_models import canonical_http_url, parse_cdx_response_v2
from wayback_state import WaybackDeferred

API_ROOT = "https://api.outscraper.cloud"
PROMPT_VERSION = "wayback-outscraper-production-v1"
ROOT = Path(__file__).resolve().parents[1]
_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_JOBS = "outscraper_jobs"
_GLOBAL = "outscraper_control"


def _credential(value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if not value.isascii() or any(ord(char) < 33 or ord(char) == 127 for char in value):
        raise ValueError("invalid Outscraper credential format")
    return value


def load_api_key(env_file=ROOT / ".env"):
    """Read only explicitly named provider credentials; never display values."""
    path = Path(env_file)
    values = dotenv_values(path, interpolate=False) if path.is_file() else {}
    for source in (os.environ, values):
        for name in ("OUTSCRAPER_API_KEY", "OUTSCRAPER_API_TOKEN"):
            if value := _credential(source.get(name)):
                return value
    return None


@dataclass(frozen=True, repr=False)
class FallbackResponse:
    status: int
    body: bytes
    headers: dict[str, str]
    final_url: str
    provenance_sha256: str | None
    retrieved_at: datetime


@dataclass(frozen=True, repr=False)
class ProviderResponse:
    status: int
    body: bytes
    headers: dict[str, str]
    final_url: str


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OutscraperTransport:
    """One fixed-host request; never follow provider-controlled result URLs."""

    def __init__(self, key, *, timeout=20, max_response_bytes=4_000_000):
        self._key = _credential(key)
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.opener = build_opener(_NoRedirects())

    def get(self, path, params=()):
        if path != "/ai-scraper" and not re.fullmatch(r"/requests/[A-Za-z0-9_-]{1,128}", path):
            raise ValueError("invalid Outscraper endpoint")
        request_url = API_ROOT + path + (("?" + urlencode(params)) if params else "")
        headers = {"Accept": "application/json", "User-Agent": "NETS-AI-Wayback/1.0"}
        if self._key:
            headers["X-API-KEY"] = self._key
        request = Request(request_url, headers=headers)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            response = exc
        with response:
            return ProviderResponse(response.code, response.read(self.max_response_bytes + 1),
                                    dict(response.headers.items()), response.geturl())


def _target(kind, url, params):
    if kind not in {"cdx", "replay"}:
        raise ValueError("unsupported Wayback fallback kind")
    query = urlencode(tuple(params))
    target = url + (("&" if "?" in url else "?") + query if query else "")
    parsed = urlsplit(target)
    if (parsed.scheme != "https" or parsed.netloc != "web.archive.org"
            or parsed.fragment or any(ord(char) < 32 for char in target)):
        raise ValueError("fallback requires an exact HTTPS Wayback URL")
    if kind == "cdx":
        values = parse_qs(parsed.query, keep_blank_values=True)
        if (parsed.path != "/cdx/search/cdx" or len(values.get("url", [])) != 1
                or values.get("output") != ["json"]
                or values.get("matchType", ["exact"]) != ["exact"]):
            raise ValueError("fallback requires an exact CDX JSON query")
        original = values["url"][0]
        canonical_http_url(original)
        return target, original, None
    match = re.fullmatch(r"https://web\.archive\.org/web/(\d{14})id_/(https?://.+)", target)
    if match is None:
        raise ValueError("fallback requires a timestamped id_ replay URL")
    datetime.strptime(match[1], "%Y%m%d%H%M%S")
    canonical_http_url(match[2])
    return target, match[2], match[1]


def extraction_params(kind, target):
    common = (
        "Read only the supplied URL. Never follow links, browse another URL, substitute a live "
        "website, infer, summarize, or reconstruct missing content. Return target_url exactly "
        "as supplied and final_url as actually retrieved. Return retrieval_error=null only if "
        "retrieval succeeded; otherwise describe the observed failure and return null content. "
    )
    if kind == "cdx":
        prompt = common + (
            "Put the entire verbatim CDX JSON response for this requested page in cdx_json as a "
            "string, including its header, every row, and any trailing empty row and resume-key "
            "row. Preserve all query parameters including resumeKey. Set page_complete=true "
            "only if the complete requested response was read without truncation. Set "
            "resume_key_present to a boolean reflecting the actual trailing resume record. "
            "A complete header-only response may be returned, but never infer no captures "
            "from inaccessible content. Return only target_url, final_url, cdx_json, "
            "page_complete, resume_key_present, retrieval_error."
        )
    else:
        prompt = common + (
            "Extract the exact visible archived page text as page_text, and its exact "
            "14-digit archive timestamp as archive_timestamp (string or integer). Do not "
            "turn text into invented HTML. Return only target_url, final_url, page_text, "
            "archive_timestamp, retrieval_error."
        )
    # The diagnostic established that the documented GET schema parameter is
    # rejected by the service mapping parser. Prompt-only mode avoids that path.
    return (("query", target), ("prompt", prompt), ("limit", "1"), ("async", "true"))


def _extract(kind, target, original, timestamp, payload):
    data = payload.get("data")
    for _ in range(2):
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
    if not isinstance(data, dict):
        raise ValueError("unexpected_extraction_shape")
    if "retrieval_error" not in data or data["retrieval_error"] is not None:
        raise ValueError("reported_retrieval_error")
    if data.get("target_url") != target or data.get("final_url") != target:
        raise ValueError("target_mismatch")
    if kind == "cdx":
        text = data.get("cdx_json")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("missing_cdx_response")
        body = text.encode("utf-8")
        rows, resume_digest = parse_cdx_response_v2(body)
        if any(row.original_url != original for row in rows):
            raise ValueError("cdx_original_target_mismatch")
        if (data.get("page_complete") is not True
                or type(data.get("resume_key_present")) is not bool
                or data["resume_key_present"] != (resume_digest is not None)):
            raise ValueError("cdx_page_completeness_unverified")
        validation = {"requested_page_complete_reported": True,
                      "resume_key_present": resume_digest is not None,
                      "capture_count": len(rows), "cdx_schema_valid": True}
        content_type = "application/json"
    else:
        text = data.get("page_text")
        reported_timestamp = data.get("archive_timestamp")
        if (type(reported_timestamp) not in (str, int)
                or re.fullmatch(r"\d{14}", str(reported_timestamp)) is None
                or str(reported_timestamp) != timestamp):
            raise ValueError("archive_timestamp_mismatch")
        datetime.strptime(str(reported_timestamp), "%Y%m%d%H%M%S")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("missing_replay_text")
        body = text.encode("utf-8")
        validation = {"archive_timestamp": timestamp,
                      "reported_timestamp_type": type(reported_timestamp).__name__}
        content_type = "text/plain; charset=utf-8"
    validation.update(target_matches=True, reported_retrieval_error=False)
    return body, content_type, validation


def _utc(value):
    return datetime.fromtimestamp(value, timezone.utc)


class OutscraperFallback:
    """Submit each paid job once, resume by ID, and pause on uncertain acceptance.

    A single cross-process jobs lock encloses state and requests. A process crash
    after durable ``submitting`` but before an ID is retained therefore requires
    operator reconciliation, never an automatic duplicate paid submission.
    """

    def __init__(self, store, key, *, transport=None, clock=None, poll_interval=20,
                 max_job_age_seconds=3 * 3600, timeout=20,
                 max_response_bytes=4_000_000, max_jobs=10_000,
                 max_total_response_bytes=1_000_000_000, scope="standalone"):
        for name, value, upper in (("poll_interval", poll_interval, 3600),
                                   ("max_job_age_seconds", max_job_age_seconds, 3 * 3600),
                                   ("timeout", timeout, 60)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError("invalid Outscraper " + name)
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= 32_000_000:
            raise ValueError("invalid Outscraper response size cap")
        if type(max_jobs) is not int or not 0 <= max_jobs <= 100_000:
            raise ValueError("invalid Outscraper job budget")
        if type(max_total_response_bytes) is not int or not 0 <= max_total_response_bytes <= 100_000_000_000:
            raise ValueError("invalid Outscraper total raw body budget")
        if not isinstance(scope, str) or not scope.strip() or len(scope) > 256:
            raise ValueError("invalid Outscraper job scope")
        self.store, self._key = store, _credential(key)
        self.clock = clock or time.time
        self.poll_interval, self.max_job_age_seconds = poll_interval, max_job_age_seconds
        self.max_response_bytes, self.max_jobs = max_response_bytes, max_jobs
        self.max_total_response_bytes = max_total_response_bytes
        self.scope = scope
        self.transport = transport or OutscraperTransport(self._key, timeout=timeout,
                                                         max_response_bytes=max_response_bytes)

    def _save(self, identity, job):
        self.store.set_state(_JOBS, identity, job)

    def _defer(self, identity, job, reason, retry_at=None):
        job.update(reason=reason, retry_at=retry_at)
        self._save(identity, job)
        raise WaybackDeferred(reason, retry_at)

    def _failure(self, identity, job, status, reason, sha=None):
        job.update(state="failed", status=status, reason=reason,
                   provenance_sha256=sha or job.get("provenance_sha256"),
                   retrieved_at=_utc(self.clock()).isoformat())
        self._save(identity, job)
        return self._cached(job)

    def _cached(self, job):
        sha = job.get("provenance_sha256")
        record = self.store.read_record(sha) if sha else None
        retrieved_at = datetime.fromisoformat(job["retrieved_at"].replace("Z", "+00:00"))
        if job["state"] == "failed":
            return FallbackResponse(job["status"], b"", {}, job["target_url"], sha, retrieved_at)
        if record is None or record["provider"] != "outscraper_ai":
            raise ValueError("missing verified Outscraper success envelope")
        raw = base64.b64decode(record["body_base64"], validate=True)
        payload = json.loads(raw)
        body, content_type, _ = _extract(job["kind"], job["target_url"], job["original_url"],
                                        job["archive_timestamp"], payload)
        metadata = record["metadata"]
        if (payload.get("status") != "Success" or metadata.get("prompt_version") != PROMPT_VERSION
                or metadata.get("job_scope") != self.scope
                or metadata.get("target_url") != job["target_url"]
                or metadata.get("extracted_sha256") != hashlib.sha256(body).hexdigest()
                or metadata.get("source_fidelity") != "structural_only"):
            raise ValueError("Outscraper success provenance binding failed")
        return FallbackResponse(200, body, {"content-type": content_type}, job["target_url"],
                                sha, retrieved_at)

    def _cooldown(self, headers):
        now = self.clock()
        raw = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
        delay = self.poll_interval
        if raw is not None:
            try:
                numeric = float(raw)
                if math.isfinite(numeric) and numeric >= 0:
                    delay = max(delay, numeric)
            except (ValueError, TypeError):
                try:
                    date = parsedate_to_datetime(str(raw))
                    if date.tzinfo is not None:
                        delay = max(delay, date.timestamp() - now)
                except (ValueError, TypeError, OverflowError):
                    pass
        control = self.store.get_state(_GLOBAL, "rate_limit") or {}
        deadline = max(now + delay, control.get("retry_at", 0))
        self.store.set_state(_GLOBAL, "rate_limit", {"retry_at": deadline})
        return deadline

    def fetch(self, kind, url, params=()):
        target, original, timestamp = _target(kind, url, params)
        # Reuse jobs only within one collector run. Account admission controls
        # remain shared by all scopes in this private state store.
        identity = hashlib.sha256(json.dumps(
            [PROMPT_VERSION, self.scope, kind, target],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        with self.store.locked("outscraper_jobs"):
            now = self.clock()
            job = self.store.get_state(_JOBS, identity)
            if job and job["state"] in {"complete", "failed"}:
                return self._cached(job)
            if job and job["state"] in {"submitting", "uncertain"}:
                self._defer(identity, job, "uncertain_submission")
            if job and job.get("request_id") and now - job["created_at"] >= self.max_job_age_seconds:
                job["state"] = "expired"
                self._defer(identity, job, "job_expired")
            if job and job["state"] == "expired":
                self._defer(identity, job, "job_expired")
            cooldown = self.store.get_state(_GLOBAL, "rate_limit") or {}
            if now < cooldown.get("retry_at", 0):
                raise WaybackDeferred("outscraper_cooldown", cooldown["retry_at"])
            if job and now < (job.get("retry_at") or 0):
                raise WaybackDeferred(job.get("reason", "pending"), job["retry_at"])
            if job is None:
                job = {"kind": kind, "target_url": target, "original_url": original,
                       "scope": self.scope,
                       "archive_timestamp": timestamp, "created_at": now,
                       "state": "ready", "raw_response_sha256": []}
            blocked = self.store.get_state(_GLOBAL, "account_failure")
            if blocked or not self._key:
                return self._failure(identity, job, blocked["status"] if blocked else 401,
                                     blocked["reason"] if blocked else "missing_credentials",
                                     blocked.get("provenance_sha256") if blocked else None)
            submitting = not job.get("request_id")
            raw_budget = self.store.get_state(_GLOBAL, "raw_budget") or {"reserved_bytes": 0}
            reservation = self.max_response_bytes + 1
            if raw_budget["reserved_bytes"] + reservation > self.max_total_response_bytes:
                raise WaybackDeferred("outscraper_raw_budget_exhausted", None)
            if submitting:
                budget = self.store.get_state(_GLOBAL, "budget") or {"submissions": 0}
                if budget["submissions"] >= self.max_jobs:
                    raise WaybackDeferred("outscraper_job_budget_exhausted", None)
                budget["submissions"] += 1
                self.store.set_state(_GLOBAL, "budget", budget)
                job.update(state="submitting", created_at=now, retry_at=None)
                self._save(identity, job)
                path, request_params = "/ai-scraper", extraction_params(kind, target)
            else:
                request_id = job["request_id"]
                if not isinstance(request_id, str) or _JOB_ID.fullmatch(request_id) is None:
                    raise ValueError("unsafe retained Outscraper job ID")
                path, request_params = "/requests/" + request_id, ()
            request_url = API_ROOT + path + (("?" + urlencode(request_params)) if request_params else "")
            # Reserve enough for this bounded response before asking the provider.
            # A process crash leaves the reservation in place conservatively;
            # Base64/envelope storage is additional to this raw-body byte budget.
            raw_budget["reserved_bytes"] += reservation
            self.store.set_state(_GLOBAL, "raw_budget", raw_budget)
            try:
                response = self.transport.get(path, request_params)
            except (OSError, URLError, HTTPException):
                if submitting:
                    job["state"] = "uncertain"
                    self._defer(identity, job, "uncertain_submission")
                self._defer(identity, job, "poll_transport_error", self.clock() + self.poll_interval)

            headers = dict(response.headers)
            rate_limit_deadline = None
            if response.status == 429:
                # Admission must remember a provider cooldown even if writing
                # its raw body later fails or the process exits during that I/O.
                rate_limit_deadline = self._cooldown(headers)
                job.update(state="pending" if job.get("request_id") else "ready",
                           reason="outscraper_cooldown", retry_at=rate_limit_deadline)
                self._save(identity, job)

            # Save the provider ID before raw-envelope writes or validation. No
            # user-controlled results_location is ever used as an endpoint.
            raw = response.body
            if not isinstance(raw, bytes):
                raise TypeError("Outscraper transport must return response bytes")
            oversized = len(raw) > self.max_response_bytes
            retained_raw = raw[:self.max_response_bytes + 1]
            raw_budget["reserved_bytes"] -= reservation - len(retained_raw)
            self.store.set_state(_GLOBAL, "raw_budget", raw_budget)
            payload = None
            if not oversized:
                try:
                    value = json.loads(raw)
                    payload = value if isinstance(value, dict) else None
                except (ValueError, UnicodeError):
                    pass
            request_id = payload.get("id") if payload else None
            safe_id = isinstance(request_id, str) and _JOB_ID.fullmatch(request_id) is not None
            id_mismatch = bool(job.get("request_id") and request_id is not None
                               and request_id != job["request_id"])
            if safe_id and not id_mismatch:
                job.update(request_id=request_id, state="pending")
                self._save(identity, job)

            final_url = response.final_url or request_url
            metadata = {"prompt_version": PROMPT_VERSION, "target_url": target,
                        "job_scope": self.scope,
                        "kind": kind, "source_fidelity": "structural_only",
                        "ai_extracted": True, "lossless_archive_response": False,
                        "operation": "submit" if submitting else "poll",
                        "request_id": job.get("request_id"), "oversized": oversized,
                        "body_truncated": oversized}
            extraction_error = None
            extracted = None
            if response.status in (200, 202) and payload and payload.get("status") == "Success":
                try:
                    extracted = _extract(kind, target, original, timestamp, payload)
                    metadata.update(validation=extracted[2],
                                    extracted_sha256=hashlib.sha256(extracted[0]).hexdigest())
                except (ValueError, TypeError, UnicodeError) as exc:
                    extraction_error = type(exc).__name__
                    metadata["validation"] = {"status": "rejected", "error_type": extraction_error}
            sha = self.store.record_response("outscraper_ai", request_url, response.status,
                                             headers, retained_raw, final_url, metadata=metadata)
            job["raw_response_sha256"].append(sha)
            job["provenance_sha256"] = sha
            self._save(identity, job)
            if final_url != request_url or id_mismatch:
                return self._failure(identity, job, 502, "provider_response_binding_failed", sha)
            if response.status == 429:
                job["state"] = "pending" if job.get("request_id") else "ready"
                self._defer(identity, job, "outscraper_cooldown", rate_limit_deadline)
            if response.status in (401, 402):
                reason = "authentication_failed" if response.status == 401 else "billing_blocked"
                self.store.set_state(_GLOBAL, "account_failure",
                                     {"status": response.status, "reason": reason, "provenance_sha256": sha})
                return self._failure(identity, job, response.status, reason, sha)
            if oversized:
                if submitting and not job.get("request_id"):
                    job["state"] = "uncertain"
                    self._defer(identity, job, "uncertain_submission")
                return self._failure(identity, job, 502, "oversized_response", sha)
            if response.status >= 500 or response.status in (408, 425):
                if submitting and not job.get("request_id"):
                    job["state"] = "uncertain"
                    self._defer(identity, job, "uncertain_submission")
                self._defer(identity, job, "provider_http_retry", self.clock() + self.poll_interval)
            if response.status not in (200, 202):
                return self._failure(identity, job, response.status, "provider_http_failure", sha)
            if payload is None:
                if submitting and not job.get("request_id"):
                    job["state"] = "uncertain"
                    self._defer(identity, job, "uncertain_submission")
                return self._failure(identity, job, 502, "invalid_provider_envelope", sha)
            status = payload.get("status")
            if status == "Success":
                if extraction_error or extracted is None:
                    return self._failure(identity, job, 502, "extraction_validation_failed", sha)
                job.update(state="complete", status=200, retrieved_at=_utc(self.clock()).isoformat())
                self._save(identity, job)
                return self._cached(job)
            if status in ("Failure", "Failed", "Error"):
                return self._failure(identity, job, 502, "provider_job_failed", sha)
            if job.get("request_id") and status in ("Pending", "InProgress", "Processing"):
                job["state"] = "pending"
                self._defer(identity, job, "pending", self.clock() + self.poll_interval)
            if submitting and not job.get("request_id"):
                job["state"] = "uncertain"
                self._defer(identity, job, "uncertain_submission")
            return self._failure(identity, job, 502, "unrecognized_provider_status", sha)
