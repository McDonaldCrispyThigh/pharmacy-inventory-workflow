"""Versioned Wayback evidence collection plus the frozen v1 compatibility helper.

The v2 collector accepts a transient candidate, queries only Internet Archive
endpoints through an injected transport, and returns typed Wayback-only evidence.
It never constructs a network transport itself.  The historical
``get_snapshot_info`` function remains unchanged below the v2 implementation.

Historical helper -- query the Internet Archive CDX API for snapshot history.

Used as optional enrichment for pharmacy records that have a known website.
Free, no API key required.

CDX API: https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server

Returns three fields per record:
    Wayback_Earliest_Year  -- first year a snapshot was captured (None if no history)
    Wayback_Latest_Year    -- most recent snapshot year (None if no history)
    Wayback_Snapshot_Count -- distinct years with at least one snapshot

Snapshot_Count semantics:
    -1  : known major pharmacy chain domain (CVS, Walgreens, etc.); CDX not queried
           to avoid timeout; web presence considered confirmed by chain status.
     0  : CDX returned no records, or query failed / timed out.
    >0  : number of distinct years with at least one archived snapshot.

Design rationale:
    Chain pharmacies operate under massive corporate domains (cvs.com, walgreens.com)
    whose CDX queries time out even with small limits. They are identified via a
    known-chain domain list and skipped. Independent pharmacies with their own small
    domains are queried with matchType=prefix on the http://www.{domain}/ prefix,
    which is fast and reliable for sites with modest web histories. The chain vs
    independent asymmetry in Snapshot_Count is analytically useful:
    independent pharmacies without web presence show Snapshot_Count=0.
"""

import base64
import hashlib
import inspect
import json
import math
import os
import random as random_module
import re
import socket
import sys
import tempfile
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse
from urllib.request import urlopen

_CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_CODE_DIRECTORY))

from v2_evidence_models import (
    ArchiveAnnualSelectionV2,
    ArchiveCDXCaptureRowV2,
    ArchiveCDXPageV2,
    ArchiveClaimAssessmentV2,
    ArchiveEvidenceV2,
    ArchiveFetchRecordV2,
    ArchiveMatchAssessmentV2,
    ArchiveOpeningClaimV2,
    ArchiveSourceSnapshotV2,
    CandidateFactsV2,
    EvidenceValidationError,
    OPENING,
    OPENING_EXTRACTOR_RULE,
    OPENING_EXTRACTOR_VERSION,
    WaybackCollectionV2,
    canonical_http_url,
    capture_request_id,
    cdx_response_chain_hash,
    evidence_id,
    opening_claim_has_assertive_context,
    parse_cdx_response_v2,
)
from wayback_state import WaybackDeferred, WaybackStateError, WaybackStateStore


_CACHE_SCHEMA = "wayback_http_cache_v3"
_CACHE_HEADERS = frozenset({"content-length", "content-type"})
_CDX_FINAL_QUERY_KEYS = frozenset({
    "url", "output", "fl", "filter", "matchType", "limit",
    "showResumeKey", "gzip", "resumeKey",
})
_MAX_RESUME_KEY_CHARS = 1_024
_HTML_MIME = re.compile(r"^text/html(?:\s*;|$)", re.I)
_CORPORATE_CLAIM = re.compile(
    r"\b(?:parent\s+company|(?:this\s+|the\s+)?company\s+"
    r"(?:(?:was|is)\s+)?(?:opened|established|founded)|corporate|"
    r"corporation|brand\s+was\s+founded|chain\s+was\s+founded|"
    r"organization\s+was\s+founded|headquarters)\b",
    re.I,
)
_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5",
    "h6", "header", "hr", "li", "main", "nav", "ol", "p", "section", "td",
    "th", "tr", "ul",
})
_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
})
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})


@dataclass(frozen=True, slots=True, repr=False)
class WaybackHTTPResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    final_url: str

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise ValueError("invalid HTTP status")
        if not isinstance(self.headers, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.headers.items()
        ):
            raise ValueError("invalid HTTP headers")
        if not isinstance(self.body, bytes) or not isinstance(self.final_url, str):
            raise ValueError("invalid HTTP response")
        object.__setattr__(self, "headers", dict(self.headers))


class WaybackTransport(Protocol):
    def get(
        self,
        url: str,
        params: tuple[tuple[str, str], ...],
        timeout: float,
        *,
        follow_redirects: bool,
    ) -> WaybackHTTPResponse: ...


WAYBACK_CONTROL_URL = "http://example.com/"
AVAILABILITY_API = "https://archive.org/wayback/available"


class WaybackServiceUnavailable(RuntimeError):
    """The public control could not establish CDX plus replay readiness."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("Wayback service preflight failed; no empirical run started")


def probe_wayback_endpoint(
    transport: WaybackTransport, url: str, params: tuple[tuple[str, str], ...],
    timeout: float,
) -> tuple[dict[str, Any], WaybackHTTPResponse | None]:
    """One bounded request, no retries/cache, and no raw body in the report."""
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid probe timeout")
    try:
        response = transport.get(url, params, timeout, follow_redirects=False)
    except (OSError, URLError) as exc:
        cause = exc.reason if isinstance(exc, URLError) else exc
        status = ("permission_denied" if isinstance(cause, PermissionError) else
                  "timeout" if isinstance(cause, (TimeoutError, socket.timeout)) else
                  "network_error")
        return {"status": status, "error_type": type(cause).__name__}, None
    headers = {key.lower(): value for key, value in response.headers.items()}
    status = ("success" if response.status == 200 else
              "http_429" if response.status == 429 else
              "http_5xx" if response.status >= 500 else
              "redirect_blocked" if 300 <= response.status < 400 else "http_error")
    expected_final = {url, url + "?" + urlencode(params)}
    if response.final_url not in expected_final or len(response.body) > 1_000_000:
        status = "invalid_response"
    declared = headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) != len(response.body):
                status = "invalid_response"
        except ValueError:
            status = "invalid_response"
    report = {
        "status": status, "http_status": response.status,
        "response_bytes": len(response.body),
        "response_sha256": hashlib.sha256(response.body).hexdigest(),
        "retry_after_present": "retry-after" in headers,
    }
    return report, response if status == "success" else None


def probe_wayback_service(
    transport: WaybackTransport, *, timeout: float = 30.0,
) -> dict[str, Any]:
    """Check a public control; this does not assess any study candidate's history.

    Availability's closest snapshot is deliberately not an evidence fallback.
    An unavailable control stops acquisition without interpreting candidate absence.
    """
    params = (("url", WAYBACK_CONTROL_URL), ("output", "json"),
              ("fl", "timestamp,original,mimetype,statuscode,digest,length"),
              ("matchType", "exact"), ("filter", "statuscode:200"),
              ("filter", "mimetype:text/html"), ("limit", "1"), ("gzip", "false"))
    cdx, response = probe_wayback_endpoint(transport, CDX_API, params, timeout)
    report = {
        "schema_version": "wayback_service_probe_v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "control_url": WAYBACK_CONTROL_URL, "status": "unavailable",
        "cdx": cdx, "replay": {"status": "not_attempted"},
    }
    if response is None:
        return report
    try:
        rows, _ = parse_cdx_response_v2(response.body)
    except (ValueError, UnicodeError):
        cdx["status"] = "invalid_response"
        return report
    if not rows:
        cdx["status"] = "no_capture"
        return report
    row = rows[0]
    # CDX may normalize scheme and www, but must not substitute another domain.
    if urlparse(row.original_url).hostname not in {"example.com", "www.example.com"}:
        cdx["status"] = "invalid_response"
        return report
    replay_url = (f"https://web.archive.org/web/{row.captured_at.strftime('%Y%m%d%H%M%S')}id_/"
                  f"{row.original_url}")
    replay, response = probe_wayback_endpoint(transport, replay_url, (), timeout)
    report["replay"] = replay
    if response is None:
        return report
    mime = {key.lower(): value for key, value in response.headers.items()}.get("content-type", "")
    try:
        if not _HTML_MIME.match(mime) or not _html_to_text(response.body, mime, 5_000)[0].strip():
            raise ValueError("control replay must contain decodable HTML text")
    except (ValueError, UnicodeError, LookupError):
        replay["status"] = "invalid_response"
        return report
    report["status"] = "ready"
    return report


def require_wayback_service(transport: WaybackTransport) -> dict[str, Any]:
    report = probe_wayback_service(transport)
    if report["status"] != "ready":
        raise WaybackServiceUnavailable(report)
    return report


@dataclass(frozen=True, slots=True)
class WaybackCollectorConfig:
    max_cdx_pages: int = 10
    max_cdx_rows: int = 500
    max_attempts: int = 3
    max_capture_attempts_per_year: int = 3
    max_body_bytes: int = 1_000_000
    max_text_chars: int = 5_000
    cdx_page_size: int = 100
    request_timeout: float = 30.0
    base_backoff_seconds: float = 2.0
    rate_limit_backoff_seconds: float = 30.0
    max_retry_wait_seconds: float = 60.0

    def __post_init__(self) -> None:
        integers = (
            self.max_cdx_pages,
            self.max_cdx_rows,
            self.max_attempts,
            self.max_capture_attempts_per_year,
            self.max_body_bytes,
            self.max_text_chars,
            self.cdx_page_size,
        )
        if any(type(value) is not int or value < 1 for value in integers):
            raise ValueError("collector limits must be positive integers")
        if self.max_text_chars > 5_000:
            raise ValueError("archive text exceeds the evidence-model bound")
        timings = (self.request_timeout, self.base_backoff_seconds,
                   self.rate_limit_backoff_seconds, self.max_retry_wait_seconds)
        if (any(type(value) not in {int, float} or not math.isfinite(value)
                for value in timings)
                or self.request_timeout <= 0 or self.base_backoff_seconds < 0
                or self.rate_limit_backoff_seconds <= 0
                or self.max_retry_wait_seconds <= 0):
            raise ValueError("invalid request timing configuration")


@dataclass(frozen=True, slots=True, repr=False)
class _FetchResult:
    status: str
    attempts: int
    retrieved_at: datetime
    http_status: int | None
    response_sha256: str | None
    body: bytes | None
    headers: Mapping[str, str]
    final_url: str | None
    cache_provenance: str
    retrieval_source: str = "direct_wayback"
    provenance_sha256: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class _CDXDecoded:
    rows: tuple[ArchiveCDXCaptureRowV2, ...]
    resume_key: str | None


class _ArchiveTextParser(HTMLParser):
    """Nonexecuting, forgiving visible-text extractor for archived HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[tuple[str, bool]] = []
        self.stack: list[tuple[str, bool]] = []

    def _boundary(self) -> None:
        if not self.parts or not self.parts[-1][1]:
            self.parts.append(("", True))

    def _is_suppressed(self) -> bool:
        return any(suppressed for _, suppressed in self.stack)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        lowered = tag.lower()
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        marker = f"{attr_map.get('id', '')} {attr_map.get('class', '')}".lower()
        suppressed = (
            self._is_suppressed()
            or lowered in _SKIP_TAGS
            or "wm-ipp" in marker
            or "wayback-toolbar" in marker
        )
        if lowered in _BLOCK_TAGS and not suppressed:
            self._boundary()
        if lowered not in _VOID_TAGS:
            self.stack.append((lowered, suppressed))

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        prior_depth = len(self.stack)
        self.handle_starttag(tag, attrs)
        if len(self.stack) > prior_depth:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        matching = next(
            (index for index in range(len(self.stack) - 1, -1, -1)
             if self.stack[index][0] == lowered),
            None,
        )
        suppressed = self._is_suppressed()
        if matching is not None:
            del self.stack[matching:]
        if lowered in _BLOCK_TAGS and not suppressed:
            self._boundary()

    def handle_data(self, data: str) -> None:
        if not self._is_suppressed():
            self.parts.append((data, False))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("invalid cache timestamp")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("invalid cache timestamp")
    return parsed.astimezone(timezone.utc)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _headers(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in values.items():
        lowered = key.strip().lower()
        cleaned = value.strip()
        if (not lowered or _contains_control(lowered)
                or _contains_control(cleaned)):
            raise ValueError("invalid HTTP response header")
        normalized[lowered] = cleaned
    return normalized


def _cache_headers(values: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in values.items()
            if key in _CACHE_HEADERS}


def _response_status(status: int) -> str:
    if status == 200:
        return "success"
    if status == 429:
        return "http_429"
    if 400 <= status < 500:
        return "http_4xx"
    if 500 <= status < 600:
        return "http_5xx"
    return "other_error"


def _contains_control(value: str) -> bool:
    return any(
        ord(char) == 127 or unicodedata.category(char) in {"Cc", "Cf", "Cs"}
        for char in value
    )


def _archive_endpoint(value: str, *, cdx: bool = False) -> bool:
    if not isinstance(value, str) or len(value) > 4_096 or _contains_control(value):
        return False
    try:
        if cdx:
            parsed = urlparse(value)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True):
                if (key not in _CDX_FINAL_QUERY_KEYS or not item.isascii()
                        or _contains_control(item)):
                    return False
                if key == "url":
                    canonical_http_url(item)
        else:
            parsed = urlparse(canonical_http_url(value))
        try:
            port = parsed.port
        except ValueError:
            return False
        return (
            parsed.scheme == "https"
            and parsed.hostname == "web.archive.org"
            and port is None
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and (not cdx or parsed.path == "/cdx/search/cdx")
        )
    except (TypeError, ValueError):
        return False


def _response_final_url(kind: str, requested_url: str, value: str) -> str | None:
    """Return the safe cache form after validating the transport's final URL."""
    if kind == "cdx":
        return CDX_API if _archive_endpoint(value, cdx=True) else None
    if value == requested_url and _archive_endpoint(value):
        return requested_url
    return None


def _safe_resume_key(value: str) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _MAX_RESUME_KEY_CHARS
        and value.isascii()
        and not any(char.isspace() for char in value)
        and not _contains_control(value)
    )


def _canonical_candidate_url(value: str) -> str:
    """Canonicalize a candidate URL after dropping its non-request fragment."""
    if len(value) > 4_096 or _contains_control(value):
        raise EvidenceValidationError("invalid URL")
    try:
        fragmentless = urlparse(value)._replace(fragment="").geturl()
    except ValueError as exc:
        raise EvidenceValidationError("invalid URL") from exc
    return canonical_http_url(fragmentless)


def _normalize_text_parts(parts: Sequence[tuple[str, bool]]) -> str:
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        normalized = unicodedata.normalize("NFKC", "".join(current))
        cleaned = "".join(
            " " if char.isspace() else char
            for char in normalized
            if ((ord(char) != 127
                 and unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
                or char.isspace())
        )
        block = re.sub(r"\s+", " ", cleaned).strip()
        if block:
            blocks.append(block)
        current.clear()

    for value, boundary in parts:
        if boundary:
            flush()
        else:
            current.append(value)
    flush()
    return "\n".join(blocks)


def _html_to_text(body: bytes, content_type: str, limit: int) -> tuple[str, bool]:
    charset_match = re.search(r"charset\s*=\s*['\"]?([^;\s'\"]+)", content_type, re.I)
    charset = charset_match.group(1) if charset_match else "utf-8"
    decoded = body.decode(charset, errors="replace")
    parser = _ArchiveTextParser()
    parser.feed(decoded)
    parser.close()
    text = _normalize_text_parts(parser.parts)
    if not text:
        raise ValueError("archive page has no visible text")
    truncated = len(text) > limit
    return text[:limit], truncated


def _matching_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    normalized = "".join(
        char if char.isalnum() else " "
        for char in value
        if not unicodedata.category(char).startswith("M")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _phrase_in_text(phrase: str, text: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def _candidate_matches(
    candidate: CandidateFactsV2, text: str
) -> tuple[bool, bool, bool]:
    normalized = _matching_text(text)
    address_parts = [part.strip() for part in candidate.formatted_address.split(",")]
    street = _matching_text(address_parts[0]) if address_parts else ""
    zip_match = re.search(r"\b\d{5}(?:-\d{4})?\b", candidate.formatted_address)
    zip_code = zip_match.group(0)[:5] if zip_match else ""
    address_match = (
        bool(street and zip_code)
        and _phrase_in_text(street, normalized)
        and re.search(rf"(?<!\d){re.escape(zip_code)}(?!\d)", text) is not None
    )

    phone_digits = re.sub(r"\D", "", candidate.phone or "")
    if len(phone_digits) == 11 and phone_digits.startswith("1"):
        phone_digits = phone_digits[1:]
    phone_match = False
    if len(phone_digits) == 10:
        area, exchange, subscriber = (
            re.escape(phone_digits[:3]),
            re.escape(phone_digits[3:6]),
            re.escape(phone_digits[6:]),
        )
        layouts = (
            f"{area}{exchange}{subscriber}",
            f"{area}-{exchange}-{subscriber}",
            rf"{area}\.{exchange}\.{subscriber}",
            f"{area} {exchange} {subscriber}",
            rf"\({area}\) ?{exchange}-{subscriber}",
        )
        pattern = (
            r"(?<![\d+])(?<!\d[ .\-/])"
            r"(?:\+1(?:[ .-])?)?"
            rf"(?:{'|'.join(layouts)})"
            + r"(?!\d)"
        )
        phone_match = re.search(pattern, text) is not None

    name = _matching_text(candidate.name)
    city = _matching_text(address_parts[1]) if len(address_parts) > 1 else ""
    name_city = _phrase_in_text(name, normalized) and _phrase_in_text(city, normalized)
    return address_match, phone_match, name_city


def _claim_sentences(text: str) -> tuple[tuple[int, str, str], ...]:
    """Return year, exact relational excerpt, and its bounded source context."""
    found: list[tuple[int, str, str]] = []
    for match in OPENING.finditer(text):
        year = int(next(item for item in match.groups() if item is not None))
        left_marks = tuple(text.rfind(mark, 0, match.start())
                           for mark in ".?!\n")
        start = max(left_marks) + 1
        right_marks = tuple(
            position for position in (text.find(mark, match.end())
                                      for mark in ".?!\n")
            if position >= 0
        )
        end = min(right_marks) + 1 if right_marks else len(text)
        sentence = text[start:end].strip()
        excerpt = match.group(0).strip()
        if not excerpt or len(excerpt) > 200:
            continue
        context = text[max(0, start - 180):end]
        found.append((year, excerpt, context))
    return tuple(found)


class WaybackEvidenceCollector:
    """Deterministic v2 Wayback-only collector with injected I/O and clock."""

    def __init__(
        self,
        transport: WaybackTransport,
        cache_dir: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] | None = None,
        random: Callable[[], float] | None = None,
        config: WaybackCollectorConfig | None = None,
        state_store: WaybackStateStore | None = None,
        fallback: Any = None,
    ) -> None:
        if transport is None or not callable(getattr(transport, "get", None)):
            raise ValueError("an injected Wayback transport is required")
        try:
            redirect_parameter = inspect.signature(transport.get).parameters.get(
                "follow_redirects"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Wayback transport must require follow_redirects"
            ) from exc
        if (
            redirect_parameter is None
            or redirect_parameter.kind
            not in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ):
            raise ValueError("Wayback transport must require follow_redirects")
        self.transport = transport
        self.cache_dir = Path(cache_dir)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or time.sleep
        self.random = random or random_module.random
        self.config = config or WaybackCollectorConfig()
        self.state_store = state_store or WaybackStateStore(
            self.cache_dir / "research_state", clock=self.clock,
        )
        self.fallback = fallback

    def _now(self) -> datetime:
        value = self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("collector clock must return an aware datetime")
        return value.astimezone(timezone.utc)

    def _request_identity(
        self,
        kind: str,
        url: str,
        params: Sequence[tuple[str, str]],
    ) -> tuple[str, Path]:
        payload = {
            "schema_version": "wayback_request_v3",
            "kind": kind,
            "endpoint": url,
            "ordered_parameters": [list(item) for item in params],
        }
        digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
        return evidence_id("wayback_cache_v2", payload), self.cache_dir / f"{digest}.json"

    def _read_cache(
        self, path: Path, provenance: str, kind: str, requested_url: str,
        params: tuple[tuple[str, str], ...] = (),
    ) -> _FetchResult | None:
        if not path.exists():
            return None
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
            expected_fields = {
                "schema_version", "cache_provenance", "kind", "retrieved_at",
                "status", "attempts", "http_status", "response_sha256",
                "response_length", "headers", "final_url", "body_base64",
                "envelope_sha256",
                "retrieval_source", "provenance_sha256",
            }
            if type(value) is not dict or set(value) != expected_fields:
                raise ValueError("invalid cache envelope")
            checksum = value.pop("envelope_sha256")
            if (
                not isinstance(checksum, str)
                or hashlib.sha256(_canonical_json(value)).hexdigest() != checksum
                or value["schema_version"] != _CACHE_SCHEMA
                or value["cache_provenance"] != provenance
                or value["kind"] != kind
            ):
                raise ValueError("cache envelope authentication failed")
            body = base64.b64decode(value["body_base64"], validate=True)
            if (
                type(value["response_length"]) is not int
                or len(body) != value["response_length"]
                or hashlib.sha256(body).hexdigest() != value["response_sha256"]
                or not isinstance(value["response_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["response_sha256"]) is None
                or type(value["attempts"]) is not int
                or value["attempts"] < 1
                or type(value["http_status"]) is not int
                or not 100 <= value["http_status"] <= 599
                or value["status"] != _response_status(value["http_status"])
                or not isinstance(value["final_url"], str)
                or type(value["headers"]) is not dict
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in value["headers"].items())
                or not set(value["headers"]).issubset(_CACHE_HEADERS)
                or any(_contains_control(k) or _contains_control(v)
                       for k, v in value["headers"].items())
                or _response_final_url(kind, requested_url,
                                       value["final_url"]) != value["final_url"]
            ):
                raise ValueError("cache body authentication failed")
            # Old cache envelopes may contain transient 429/5xx responses.
            # Authenticate them before ignoring them; tampered caches still fail
            # closed. Only successful evidence is reusable across requests.
            if value["http_status"] != 200:
                return None
            raw_record = self.state_store.read_record(value["provenance_sha256"])
            full_url = requested_url + (("?" + urlencode(params)) if params else "")
            if (value["retrieval_source"] not in {"direct_wayback", "outscraper_ai"}
                    or raw_record["provider"] != value["retrieval_source"]
                    or (raw_record["response_sha256"] if value["retrieval_source"] == "direct_wayback"
                        else raw_record["metadata"].get("extracted_sha256")) != value["response_sha256"]):
                raise WaybackStateError("cache does not bind its raw response")
            if raw_record["metadata"].get("kind") != kind:
                raise WaybackStateError("cache raw response kind mismatch")
            if value["retrieval_source"] == "direct_wayback":
                if (raw_record["request_url"] != full_url
                        or raw_record["response_status"] != value["http_status"]
                        or raw_record["retrieved_at"] != value["retrieved_at"]
                        or _cache_headers(_headers(raw_record["headers"])) != value["headers"]
                        or _response_final_url(kind, requested_url, raw_record["final_url"]) != value["final_url"]):
                    raise WaybackStateError("cache raw HTTP provenance mismatch")
            elif (raw_record["metadata"].get("target_url") != full_url
                  or raw_record["metadata"].get("source_fidelity") != "structural_only"):
                raise WaybackStateError("cached extraction target mismatch")
            return _FetchResult(
                value["status"], value["attempts"], _parse_iso(value["retrieved_at"]),
                value["http_status"], value["response_sha256"], body,
                dict(value["headers"]), value["final_url"], provenance,
                value["retrieval_source"], value["provenance_sha256"],
            )
        except (OSError, UnicodeError, ValueError, TypeError, KeyError) as exc:
            raise WaybackStateError("Wayback cache integrity failed") from exc

    def _write_cache(self, path: Path, kind: str, result: _FetchResult) -> None:
        if (result.status != "success" or result.http_status != 200
                or result.body is None or result.final_url is None):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        value: dict[str, Any] = {
            "schema_version": _CACHE_SCHEMA,
            "cache_provenance": result.cache_provenance,
            "kind": kind,
            "retrieved_at": _iso(result.retrieved_at),
            "status": result.status,
            "attempts": result.attempts,
            "http_status": result.http_status,
            "response_sha256": result.response_sha256,
            "response_length": len(result.body),
            "headers": dict(result.headers),
            "final_url": result.final_url,
            "body_base64": base64.b64encode(result.body).decode("ascii"),
            "retrieval_source": result.retrieval_source,
            "provenance_sha256": result.provenance_sha256,
        }
        value["envelope_sha256"] = hashlib.sha256(_canonical_json(value)).hexdigest()
        encoded = _canonical_json(value)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.cache_dir, prefix=".tmp-wayback-", suffix=".json",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass

    def _retry_delay(
        self, attempt: int, status: str, headers: Mapping[str, str],
    ) -> float | None:
        """Honor a server deadline; defer rather than retry before a long one."""
        jitter = self.random()
        if type(jitter) not in {int, float} or not 0 <= jitter <= 1:
            raise ValueError("random jitter must be between zero and one")
        delay = self.config.base_backoff_seconds * (2 ** min(attempt - 1, 30))
        delay *= 1 + jitter
        if status == "http_429":
            delay = max(delay, self.config.rate_limit_backoff_seconds)
        value = headers.get("retry-after", "")
        server_delay = 0.0
        if re.fullmatch(r"[0-9]+", value):
            digits = value.lstrip("0") or "0"
            if len(digits) > 12:
                return None
            server_delay = float(digits)
        elif value:
            try:
                deadline = parsedate_to_datetime(value)
                if deadline.tzinfo is not None:
                    server_delay = max(0.0, (deadline - self._now()).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
        delay = max(delay, server_delay)
        # Returning the failure lets the caller pause; never clamp Retry-After
        # downwards and send an early request. The preflight uses one attempt.
        return delay if delay <= self.config.max_retry_wait_seconds else None

    def _fetch(
        self,
        kind: str,
        url: str,
        params: tuple[tuple[str, str], ...],
    ) -> _FetchResult:
        provenance, path = self._request_identity(kind, url, params)
        cached = self._read_cache(path, provenance, kind, url, params)
        if cached is not None:
            return cached

        last_response: WaybackHTTPResponse | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                current, raw_provenance = self._direct_get(kind, url, params, attempt)
                if not isinstance(current, WaybackHTTPResponse):
                    raise TypeError("transport returned an invalid response")
                last_response = current
                status = _response_status(current.status)
                normalized_headers = _headers(current.headers)
                safe_final_url = _response_final_url(
                    kind, url, current.final_url
                )
                bounded = len(current.body) <= self.config.max_body_bytes
                contained = (safe_final_url is not None and bounded
                             and current.final_url in {url, url + "?" + urlencode(params)})
                if not contained and status not in {
                    "http_429", "http_4xx", "http_5xx"
                }:
                    status = "other_error"
                retryable = (
                    contained and status == "http_5xx"
                )
                if retryable and attempt < self.config.max_attempts:
                    delay = self._retry_delay(attempt, status, normalized_headers)
                    if delay is not None:
                        self.sleep(delay)
                        continue
                body_hash = hashlib.sha256(current.body).hexdigest()
                result = _FetchResult(
                    status, attempt, _parse_iso(self.state_store.read_record(raw_provenance)["retrieved_at"]), current.status, body_hash,
                    current.body, _cache_headers(normalized_headers),
                    safe_final_url if bounded else None,
                    provenance, "direct_wayback", raw_provenance,
                )
                return result
            except (TimeoutError, socket.timeout):
                if attempt < self.config.max_attempts:
                    delay = self._retry_delay(attempt, "timeout", {})
                    if delay is not None:
                        self.sleep(delay)
                        continue
                return _FetchResult(
                    "timeout", attempt, self._now(), None, None, None, {}, None,
                    provenance,
                )
            except URLError as exc:
                if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                    if attempt < self.config.max_attempts:
                        delay = self._retry_delay(attempt, "timeout", {})
                        if delay is not None:
                            self.sleep(delay)
                            continue
                    return _FetchResult(
                        "timeout", attempt, self._now(), None, None, None, {}, None,
                        provenance,
                    )
                return _FetchResult(
                    "other_error", attempt, self._now(), None, None, None, {}, None,
                    provenance,
                )
            except (OSError, TypeError, ValueError):
                return _FetchResult(
                    "other_error", attempt, self._now(), None, None, None, {}, None,
                    provenance,
                )
        assert last_response is not None
        raise AssertionError("unreachable retry state")

    def _direct_get(self, kind, url, params, attempt):
        """Check and update one durable deadline for both CDX and replay."""
        full_url = url + (("?" + urlencode(params)) if params else "")
        with self.state_store.locked("direct_wayback"):
            state = self.state_store.get_state("cooldown", "direct_wayback") or {}
            now = self._now().timestamp()
            if state.get("retry_at", 0) > now:
                raise WaybackDeferred("direct_rate_limited", state["retry_at"])
            try:
                current = self.transport.get(url, params, float(self.config.request_timeout),
                                             follow_redirects=False)
            except (OSError, URLError) as exc:
                self.state_store.record_response(
                    "direct_wayback", full_url, None, {}, b"", None,
                    {"kind": kind, "attempt": attempt, "error_type": type(exc).__name__},
                )
                raise
            if not isinstance(current, WaybackHTTPResponse):
                raise TypeError("invalid direct response")
            headers = _headers(current.headers)
            cooldown = None
            if current.status == 429:
                now = self._now().timestamp()
                strikes = min(int(state.get("strikes", 0)) + 1, 20)
                delay = self.config.rate_limit_backoff_seconds * 2 ** (strikes - 1)
                jitter = self.random()
                if type(jitter) not in (int, float) or not 0 <= jitter <= 1:
                    raise ValueError("invalid rate limit jitter")
                delay *= 1 + jitter
                server = headers.get("retry-after", "")
                if re.fullmatch(r"[0-9]+", server):
                    # A huge valid deadline remains deferred, never clamped to
                    # the short local retry limit. Avoid integer/float overflow.
                    delay = max(delay, float(server) if len(server) < 300 else 1e300)
                elif server:
                    try:
                        parsed = parsedate_to_datetime(server)
                        if parsed.tzinfo is not None:
                            delay = max(delay, parsed.timestamp() - now)
                    except (TypeError, ValueError, OverflowError):
                        pass
                cooldown = {
                    "retry_at": now + delay, "strikes": strikes,
                    "provenance_sha256": None,
                }
                # Commit admission control first. A failed body write must not
                # cause a restart to forget a server deadline already observed.
                self.state_store.set_state("cooldown", "direct_wayback", cooldown)
            raw_provenance = self.state_store.record_response(
                "direct_wayback", full_url, current.status, current.headers,
                current.body, current.final_url, {"kind": kind, "attempt": attempt,
                                                "body_truncated": len(current.body) > self.config.max_body_bytes},
            )
            request_key, _ = self._request_identity(kind, url, params)
            history = self.state_store.get_state("direct_requests", request_key) or {
                "kind": kind, "raw_response_sha256": [],
            }
            history["raw_response_sha256"].append(raw_provenance)
            self.state_store.set_state("direct_requests", request_key, history)
            if cooldown is not None:
                cooldown["provenance_sha256"] = raw_provenance
                self.state_store.set_state("cooldown", "direct_wayback", cooldown)
            elif current.status == 200:
                self.state_store.set_state("cooldown", "direct_wayback", {"retry_at": 0, "strikes": 0})
            return current, raw_provenance

    def _fallback_fetch(self, kind, url, params, original=None):
        """An extraction has its own bytes and provider provenance, never fake HTML."""
        if self.fallback is None:
            if original is not None and original.status == "http_429":
                state = self.state_store.get_state("cooldown", "direct_wayback") or {}
                raise WaybackDeferred("direct_rate_limited", state.get("retry_at"))
            return original
        result = self.fallback.fetch(kind, url, params)
        provenance, _ = self._request_identity(kind, url, params)
        if result.status != 200:
            # Terminal failure of the fallback. Retain direct failure in the
            # evidence record; raw fallback diagnostics are kept by its adapter.
            if original is not None:
                return original
            return _FetchResult("other_error", 1, self._now(), None, None, None, {}, None, provenance)
        raw = self.state_store.read_record(result.provenance_sha256)
        full_url = url + (("?" + urlencode(params)) if params else "")
        body_hash = hashlib.sha256(result.body).hexdigest()
        if (raw["provider"] != "outscraper_ai"
                or raw["metadata"].get("extracted_sha256") != body_hash
                or raw["metadata"].get("target_url") != full_url
                or raw["metadata"].get("kind") != kind
                or result.final_url != full_url):
            raise WaybackStateError("Outscraper extraction provenance mismatch")
        return _FetchResult(
            "success", 1, result.retrieved_at, 200, body_hash, result.body,
            _cache_headers(_headers(result.headers)),
            _response_final_url(kind, url, result.final_url), provenance,
            "outscraper_ai", result.provenance_sha256,
        )

    def _fetch_or_fallback(self, kind, url, params):
        try:
            fetched = self._fetch(kind, url, params)
        except WaybackDeferred:
            if self.fallback is None:
                raise
            return self._fallback_fetch(kind, url, params)
        if fetched.status != "success":
            return self._fallback_fetch(kind, url, params, fetched)
        return fetched

    def _cache_validated(
        self,
        kind: str,
        url: str,
        params: tuple[tuple[str, str], ...],
        result: _FetchResult,
    ) -> None:
        provenance, path = self._request_identity(kind, url, params)
        if result.cache_provenance != provenance:
            raise ValueError("cache provenance changed across validation")
        self._write_cache(path, kind, result)

    def _discard_cached_response(
        self,
        kind: str,
        url: str,
        params: tuple[tuple[str, str], ...],
    ) -> None:
        _, path = self._request_identity(kind, url, params)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _empty(self, status: str) -> WaybackCollectionV2:
        evidence = ArchiveEvidenceV2(
            status, "not_attempted", "not_applicable", None,
            "candidate_website", self._now(), self.config.max_cdx_pages,
            self.config.max_cdx_rows, None, (), (), (), (), (),
        )
        return WaybackCollectionV2(evidence, (), ())

    def _parse_cdx(self, body: bytes) -> _CDXDecoded:
        rows, resume_digest = parse_cdx_response_v2(body)
        raw = json.loads(body.decode("utf-8"))
        resume: str | None = None
        if resume_digest is not None:
            resume = raw[-1][0]
            if (not _safe_resume_key(resume)
                    or hashlib.sha256(resume.encode("utf-8")).hexdigest()
                    != resume_digest):
                raise ValueError("CDX resume digest mismatch")
        return _CDXDecoded(rows, resume)

    def _page(
        self,
        number: int,
        incoming: str | None,
        fetch: _FetchResult,
        seen_resume: set[str],
        seen_rows: set[str],
        prior_last_row: ArchiveCDXCaptureRowV2 | None,
        expected_original_url: str,
    ) -> tuple[ArchiveCDXPageV2, str | None, ArchiveCDXCaptureRowV2 | None]:
        status = fetch.status
        rows: tuple[ArchiveCDXCaptureRowV2, ...] = ()
        resume: str | None = None
        complete = False
        if status == "success":
            if (
                fetch.body is None
                or len(fetch.body) > self.config.max_body_bytes
                or fetch.final_url is None
                or not _archive_endpoint(fetch.final_url, cdx=True)
                or (fetch.headers.get("content-length") is not None
                    and fetch.headers["content-length"] != str(len(fetch.body)))
            ):
                status = "other_error"
            else:
                try:
                    decoded = self._parse_cdx(fetch.body)
                    resume = decoded.resume_key
                    candidate_rows = decoded.rows
                    target_mismatch = any(
                        row.original_url != expected_original_url
                        for row in candidate_rows
                    )
                    retained_rows = tuple(
                        row for row in candidate_rows
                        if row.evidence_id not in seen_rows
                    )
                    future_capture = any(
                        row.captured_at > fetch.retrieved_at
                        for row in candidate_rows
                    )
                    exceeds_global_cap = (
                        len(seen_rows) + len(retained_rows)
                        > self.config.max_cdx_rows
                    )
                    out_of_order = prior_last_row is not None and retained_rows and (
                        retained_rows[0].captured_at,
                        retained_rows[0].original_url,
                        retained_rows[0].digest,
                        retained_rows[0].length,
                        retained_rows[0].evidence_id,
                    ) < (
                        prior_last_row.captured_at,
                        prior_last_row.original_url,
                        prior_last_row.digest,
                        prior_last_row.length,
                        prior_last_row.evidence_id,
                    )
                    rows = retained_rows
                    complete = resume is None
                    if future_capture or target_mismatch:
                        status, rows, resume, complete = "decode_error", (), None, False
                    elif exceeds_global_cap or out_of_order or (
                        resume is not None and resume in seen_resume
                    ):
                        status, rows, resume, complete = "other_error", (), None, False
                except (
                    UnicodeError, ValueError, TypeError, json.JSONDecodeError,
                    EvidenceValidationError,
                ):
                    status, rows, resume, complete = "decode_error", (), None, False

        incoming_digest = hashlib.sha256(incoming.encode()).hexdigest() if incoming else None
        outgoing_digest = hashlib.sha256(resume.encode()).hexdigest() if resume else None
        response_body = fetch.body
        response_length = len(fetch.body) if fetch.body is not None else None
        response_hash = fetch.response_sha256
        page = ArchiveCDXPageV2(
            number, status, fetch.attempts, fetch.retrieved_at, fetch.http_status,
            response_hash, response_length, len(rows),
            tuple(sorted({row.captured_at.year for row in rows})),
            incoming_digest, outgoing_digest, complete, fetch.cache_provenance,
            rows, response_body,
            fetch.retrieval_source, fetch.provenance_sha256,
        )
        if status == "success":
            seen_rows.update(row.evidence_id for row in rows)
            if resume is not None:
                seen_resume.add(resume)
        last_row = rows[-1] if rows else prior_last_row
        return page, resume, last_row

    def _query_pages(
        self, canonical_url: str
    ) -> tuple[tuple[ArchiveCDXPageV2, ...], str]:
        fixed = (
            ("url", canonical_url),
            ("output", "json"),
            ("fl", "timestamp,original,mimetype,statuscode,digest,length"),
            ("filter", "statuscode:200"),
            ("filter", "mimetype:text/html"),
            ("matchType", "exact"),
            ("limit", str(self.config.cdx_page_size)),
            ("showResumeKey", "true"),
            ("gzip", "false"),
        )
        pages: list[ArchiveCDXPageV2] = []
        incoming: str | None = None
        seen_resume: set[str] = set()
        seen_rows: set[str] = set()
        last_row: ArchiveCDXCaptureRowV2 | None = None
        cap_status = "not_capped"
        while True:
            remaining = self.config.max_cdx_rows - len(seen_rows)
            page_parameters = tuple((key, str(min(self.config.cdx_page_size, remaining)))
                                    if key == "limit" else (key, value) for key, value in fixed)
            params = page_parameters + ((("resumeKey", incoming),) if incoming is not None else ())
            fetched = self._fetch_or_fallback("cdx", CDX_API, params)
            page, outgoing, last_row = self._page(
                len(pages) + 1, incoming, fetched, seen_resume, seen_rows,
                last_row, canonical_url,
            )
            if (page.status != "success" and fetched.status == "success"
                    and fetched.retrieval_source == "direct_wayback" and self.fallback is not None):
                fetched = self._fallback_fetch("cdx", CDX_API, params, fetched)
                page, outgoing, last_row = self._page(
                    len(pages) + 1, incoming, fetched, seen_resume, seen_rows,
                    last_row, canonical_url,
                )
            pages.append(page)
            if (
                page.status in {
                    "success", "http_429", "http_4xx", "http_5xx",
                }
                and fetched.final_url is not None
                and _archive_endpoint(fetched.final_url, cdx=True)
            ):
                self._cache_validated("cdx", CDX_API, params, fetched)
            elif fetched.body is not None:
                self._discard_cached_response("cdx", CDX_API, params)
            if page.status != "success" or page.pagination_complete:
                break
            total_rows = sum(item.returned_rows for item in pages)
            if total_rows >= self.config.max_cdx_rows:
                cap_status = "max_records"
                break
            if len(pages) >= self.config.max_cdx_pages:
                cap_status = "max_pages"
                break
            assert outgoing is not None
            incoming = outgoing
        return tuple(pages), cap_status

    @staticmethod
    def _query_status(pages: tuple[ArchiveCDXPageV2, ...], cap_status: str) -> str:
        successes = tuple(page for page in pages if page.status == "success")
        failures = tuple(page for page in pages if page.status != "success")
        if not successes:
            statuses = {page.status for page in pages}
            return next(iter(statuses)) if len(statuses) == 1 else "other_error"
        complete = (
            not failures and cap_status == "not_capped"
            and pages[-1].pagination_complete
        )
        if not complete:
            return "partial"
        return "success" if any(page.returned_rows for page in pages) else "success_no_capture"

    def _fetch_record(
        self, row: ArchiveCDXCaptureRowV2, *, replacement: _FetchResult | None = None
    ) -> tuple[ArchiveFetchRecordV2, ArchiveSourceSnapshotV2 | None]:
        request_id = capture_request_id(row)
        replay_url = (
            f"https://web.archive.org/web/{row.captured_at.strftime('%Y%m%d%H%M%S')}id_/"
            f"{row.raw_original_url or row.original_url}"
        )
        fetched = replacement or self._fetch_or_fallback("replay", replay_url, ())
        status = fetched.status
        source: ArchiveSourceSnapshotV2 | None = None
        if status == "success":
            content_type = fetched.headers.get("content-type", "")
            declared = fetched.headers.get("content-length")
            invalid_length = False
            if declared:
                try:
                    invalid_length = int(declared) != len(fetched.body or b"")
                except ValueError:
                    invalid_length = True
            if (
                fetched.body is None
                or fetched.final_url != replay_url
                or not _archive_endpoint(fetched.final_url or "")
                or (not _HTML_MIME.match(content_type)
                    if fetched.retrieval_source == "direct_wayback"
                    else not content_type.lower().startswith("text/plain"))
                or len(fetched.body) > self.config.max_body_bytes
                or invalid_length
            ):
                status = "other_error"
            else:
                try:
                    if fetched.retrieval_source == "outscraper_ai":
                        extracted = fetched.body.decode("utf-8").strip()
                        text = extracted[:self.config.max_text_chars]
                        text_truncated = len(extracted) > self.config.max_text_chars
                    else:
                        text, text_truncated = _html_to_text(
                            fetched.body, content_type, self.config.max_text_chars
                        )
                    if not text.strip():
                        raise ValueError("empty archived page text")
                    page_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    payload = {
                        "capture_request_id": request_id,
                        "captured_at": _iso(row.captured_at),
                        "replay_url": replay_url,
                        "original_url": row.original_url,
                        "page_text_sha256": page_hash,
                    }
                    source = ArchiveSourceSnapshotV2(
                        evidence_id("archive_source_v2", payload), request_id,
                        row.captured_at, replay_url, row.original_url, text, False,
                        text_truncated, fetched.retrieval_source, fetched.provenance_sha256,
                    )
                except (UnicodeError, LookupError, ValueError, EvidenceValidationError):
                    status = "decode_error"

        if (status != "success" and fetched.status == "success" and replacement is None
                and fetched.retrieval_source == "direct_wayback" and self.fallback is not None):
            fallback = self._fallback_fetch("replay", replay_url, (), fetched)
            if fallback is not fetched:
                return self._fetch_record(row, replacement=fallback)

        source_id = source.evidence_id if source is not None else None
        record = ArchiveFetchRecordV2(
            request_id, source_id, status, fetched.attempts, fetched.retrieved_at,
            fetched.http_status, fetched.response_sha256, fetched.cache_provenance,
            fetched.retrieval_source, fetched.provenance_sha256,
        )
        if (
            status in {"success", "http_429", "http_4xx", "http_5xx"}
            and fetched.final_url is not None
            and _archive_endpoint(fetched.final_url)
        ):
            self._cache_validated("replay", replay_url, (), fetched)
        elif fetched.body is not None:
            self._discard_cached_response("replay", replay_url, ())
        return record, source

    def _claims(
        self, candidate: CandidateFactsV2, source: ArchiveSourceSnapshotV2
    ) -> tuple[tuple[ArchiveOpeningClaimV2, ArchiveClaimAssessmentV2], ...]:
        values: list[tuple[ArchiveOpeningClaimV2, ArchiveClaimAssessmentV2]] = []
        seen: set[str] = set()
        for year, sentence, context in _claim_sentences(source.archived_page_text):
            if year > source.captured_at.year:
                continue
            try:
                payload = {
                    "source_evidence_id": source.evidence_id,
                    "year": year,
                    "excerpt": sentence,
                    "page_text_sha256": source.page_text_sha256,
                    "extractor_rule": OPENING_EXTRACTOR_RULE,
                    "extractor_version": OPENING_EXTRACTOR_VERSION,
                }
                claim = ArchiveOpeningClaimV2(
                    evidence_id("opening_claim_v2", payload), source.evidence_id,
                    year, sentence, source.page_text_sha256,
                    OPENING_EXTRACTOR_RULE, OPENING_EXTRACTOR_VERSION,
                )
                if not opening_claim_has_assertive_context(
                        source.archived_page_text, claim.excerpt):
                    continue
                if _CORPORATE_CLAIM.search(context):
                    matches = (False, False, False)
                else:
                    matches = _candidate_matches(candidate, context)
                assessment = ArchiveClaimAssessmentV2(claim.evidence_id, *matches)
                if claim.evidence_id not in seen:
                    values.append((claim, assessment))
                    seen.add(claim.evidence_id)
            except EvidenceValidationError:
                continue
        return tuple(values)

    def check_health(self) -> dict[str, Any]:
        """One public CDX page and one selected replay, with the same fallback path.

        The control checks acquisition readiness, not completeness or scientific
        validity of any study candidate. Pending paid jobs remain resumable.
        """
        params = (("url", WAYBACK_CONTROL_URL), ("output", "json"),
                  ("fl", "timestamp,original,mimetype,statuscode,digest,length"),
                  ("matchType", "exact"), ("filter", "statuscode:200"),
                  ("filter", "mimetype:text/html"), ("limit", "1"),
                  ("showResumeKey", "true"), ("gzip", "false"))
        fetched = self._fetch_or_fallback("cdx", CDX_API, params)
        page, _, _ = self._page(1, None, fetched, set(), set(), None, WAYBACK_CONTROL_URL)
        if (page.status != "success" and fetched.status == "success"
                and fetched.retrieval_source == "direct_wayback" and self.fallback is not None):
            fetched = self._fallback_fetch("cdx", CDX_API, params, fetched)
            page, _, _ = self._page(1, None, fetched, set(), set(), None, WAYBACK_CONTROL_URL)
        report = {"schema_version": "wayback_research_health_v1", "checked_at": _iso(self._now()),
                  "status": "unavailable", "control_url": WAYBACK_CONTROL_URL,
                  "cdx": {"status": page.status, "retrieval_source": page.retrieval_source,
                          "provenance_sha256": page.provenance_sha256},
                  "replay": {"status": "not_attempted"}}
        if page.status != "success" or not page.capture_rows:
            if page.status == "success":
                report["cdx"]["status"] = "no_capture"
            raise WaybackServiceUnavailable(report)
        self._cache_validated("cdx", CDX_API, params, fetched)
        record, source = self._fetch_record(page.capture_rows[0])
        report["replay"] = {"status": record.status, "retrieval_source": record.retrieval_source,
                            "provenance_sha256": record.provenance_sha256}
        if source is None:
            raise WaybackServiceUnavailable(report)
        report["status"] = "ready"
        report["source_fidelity"] = ("structural_only" if "outscraper_ai" in
                                     {page.retrieval_source, source.retrieval_source} else "original_http")
        return report

    def collect(self, candidate: CandidateFactsV2) -> WaybackCollectionV2:
        if not isinstance(candidate, CandidateFactsV2):
            raise TypeError("collector requires CandidateFactsV2")
        if candidate.website is None or not candidate.website.strip():
            return self._empty("missing_url")
        try:
            canonical_url = _canonical_candidate_url(candidate.website)
        except EvidenceValidationError:
            return self._empty("invalid_url")

        pages, cap_status = self._query_pages(canonical_url)
        all_rows = tuple(row for page in pages for row in page.capture_rows)
        rows_by_year = {
            year: tuple(row for row in all_rows if row.captured_at.year == year)
            for year in sorted({row.captured_at.year for row in all_rows})
        }
        selections: list[ArchiveAnnualSelectionV2] = []
        fetches: list[ArchiveFetchRecordV2] = []
        sources: list[ArchiveSourceSnapshotV2] = []
        match_assessments: list[ArchiveMatchAssessmentV2] = []
        claims: list[ArchiveOpeningClaimV2] = []
        claim_assessments: list[ArchiveClaimAssessmentV2] = []

        for year, rows in rows_by_year.items():
            attempted: list[str] = []
            selected: ArchiveSourceSnapshotV2 | None = None
            for row in rows[:self.config.max_capture_attempts_per_year]:
                record, source = self._fetch_record(row)
                fetches.append(record)
                attempted.append(record.capture_request_id)
                if source is not None:
                    selected = source
                    sources.append(source)
                    source_matches = _candidate_matches(candidate, source.archived_page_text)
                    match_assessments.append(
                        ArchiveMatchAssessmentV2(source.evidence_id, *source_matches)
                    )
                    for claim, assessment in self._claims(candidate, source):
                        claims.append(claim)
                        claim_assessments.append(assessment)
                    break
            if selected is not None:
                stop_reason = "first_usable"
            elif len(attempted) == len(rows):
                stop_reason = "exhausted"
            else:
                stop_reason = "max_attempts"
            selections.append(ArchiveAnnualSelectionV2(
                year, self.config.max_capture_attempts_per_year, tuple(attempted),
                selected.evidence_id if selected is not None else None, stop_reason,
            ))

        if not fetches:
            page_status = "not_attempted"
        else:
            successful = sum(item.status == "success" for item in fetches)
            page_status = (
                "success" if successful == len(fetches)
                else "partial_success" if successful
                else "all_failed"
            )
        response_bodies = tuple(
            page.response_body for page in pages if page.response_body is not None
        )
        query_payload = {
            "query_definition": "exact_html_cdx_v2",
            "canonical_url": canonical_url,
        }
        evidence = ArchiveEvidenceV2(
            self._query_status(pages, cap_status), page_status, cap_status,
            evidence_id("wayback_query_v2", query_payload), "candidate_website",
            self._now(), self.config.max_cdx_pages, self.config.max_cdx_rows,
            cdx_response_chain_hash(*response_bodies), pages, tuple(selections),
            tuple(fetches), tuple(sources), tuple(claims),
        )
        return WaybackCollectionV2(
            evidence, tuple(match_assessments), tuple(claim_assessments)
        )

CDX_API = "https://web.archive.org/cdx/search/cdx"

_EMPTY = {
    "Wayback_Earliest_Year":  None,
    "Wayback_Latest_Year":    None,
    "Wayback_Snapshot_Count": 0,
}

_CHAIN_SENTINEL = {
    "Wayback_Earliest_Year":  None,
    "Wayback_Latest_Year":    None,
    "Wayback_Snapshot_Count": -1,
}

# Major pharmacy chain domains that would cause CDX query timeouts.
# Presence on any of these implies confirmed web history.
_CHAIN_DOMAINS = frozenset({
    "walgreens.com", "cvs.com", "riteaid.com", "walmart.com",
    "target.com", "costco.com", "kroger.com", "albertsons.com",
    "safeway.com", "publix.com", "meijer.com", "hy-vee.com",
    "hyvee.com", "duanereade.com", "thriftywhite.com",
})


def _root_domain(netloc: str) -> str:
    """Return two-part root domain, e.g. 'www.cvs.com' -> 'cvs.com'."""
    parts = netloc.lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc.lower()


def _bare_domain(netloc: str) -> str:
    """Return domain without www. prefix."""
    d = netloc.lower()
    return d[4:] if d.startswith("www.") else d


def get_snapshot_info(url: str, timeout: int = 8) -> dict:
    """
    Query CDX API for snapshot history of a business website.

    Parameters
    ----------
    url     : Business website URL from Google Maps (Business_Website field).
    timeout : Request timeout in seconds. Default 8 is pipeline-safe; independent
              pharmacy sites respond well within this window.

    Returns
    -------
    dict with Wayback_Earliest_Year, Wayback_Latest_Year, Wayback_Snapshot_Count.
    Never raises -- all exceptions are silently caught.
    """
    if not url:
        return dict(_EMPTY)

    try:
        parsed = urlparse(url if "://" in url else "https://" + url)
        netloc = parsed.netloc or parsed.path.split("/")[0]
    except Exception:
        return dict(_EMPTY)

    if not netloc:
        return dict(_EMPTY)

    # Short-circuit known chain domains to avoid CDX timeouts
    if _root_domain(netloc) in _CHAIN_DOMAINS:
        return dict(_CHAIN_SENTINEL)

    # For independent pharmacies, query using http://www.{domain}/ as the prefix.
    # matchType=prefix captures the homepage and direct subpages; fast on small sites.
    domain = _bare_domain(netloc)
    prefix_url = f"http://www.{domain}/"
    params = {
        "url":       prefix_url,
        "output":    "json",
        "fl":        "timestamp",
        "collapse":  "timestamp:4",
        "limit":     "20",
        "matchType": "prefix",
    }

    try:
        with urlopen(CDX_API + "?" + urlencode(params), timeout=timeout) as resp:
            raw = json.loads(resp.read())
    except (URLError, HTTPError, json.JSONDecodeError, Exception):
        return dict(_EMPTY)

    if not raw or len(raw) < 2:
        return dict(_EMPTY)

    years = []
    for row in raw[1:]:
        if row:
            ts = str(row[0])   # CDX returns int timestamps
            if len(ts) >= 4:
                try:
                    years.append(int(ts[:4]))
                except ValueError:
                    pass

    if not years:
        return dict(_EMPTY)

    return {
        "Wayback_Earliest_Year":  min(years),
        "Wayback_Latest_Year":    max(years),
        "Wayback_Snapshot_Count": len(years),
    }
