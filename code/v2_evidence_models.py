"""Transient Google packets and persistable, authenticated Wayback preflight evidence.

Google Places facts, reviews, classifier inputs, assessments, and decisions are
deliberately nonserializable.  Only the Wayback record classes below expose
serializers, and their query key is an opaque digest rather than a source URL.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse, urlunparse


class EvidenceValidationError(ValueError):
    """Raised for an invalid evidence-domain value without exposing internals."""


SHA = re.compile(r"^[0-9a-f]{64}$")
CDX_SHA1 = re.compile(r"^[A-Z2-7]{32}$")
PLACE = re.compile(r"^[A-Za-z0-9_-]{10,256}$")
NAICS = re.compile(r"^\d{6}$")
EVIDENCE_REF = re.compile(
    r"^(?:candidate|review|archive_source_v2|opening_claim_v2)_[0-9a-f]{64}$"
)
OPAQUE = re.compile(r"^(?:wayback_query_v2|wayback_cache_v2|capture_request_v2)_[0-9a-f]{64}$")
SENSITIVE = re.compile(r"(?:key|token|secret|password|auth|signature|credential)", re.I)
OPENING = re.compile(r"\b(?:opened|established|founded)\s+(?:in\s+)?(\d{4})\b|\b(?:in\s+business|at\s+this\s+location|serving\s+(?:this\s+community|this\s+location))\s+since\s+(\d{4})\b", re.I)
ASSERTIVE_OPENING = re.compile(
    r"^\s*(?:(?:opened|established|founded)\s+(?:in\s+)?\d{4}|"
    r"(?:(?:this|the)\s+(?:outlet|location|store|pharmacy|business|company)|it)\s+"
    r"(?:(?:was|is)\s+)?(?:opened|established|founded)\s+(?:in\s+)?\d{4}|"
    r"(?:in\s+business|at\s+this\s+location|"
    r"serving\s+(?:this\s+community|this\s+location))\s+since\s+\d{4}|"
    r"(?:(?:this|the)\s+(?:outlet|location|store|pharmacy|business)|it)\s+"
    r"(?:has\s+been\s+)(?:in\s+business|at\s+this\s+location|"
    r"serving\s+(?:this\s+community|this\s+location))\s+since\s+\d{4})"
    r"\s*[.!]?\s*$",
    re.I,
)
DISALLOWED_OPENING_CONTEXT = re.compile(
    r"(?:n['’]t\b|\b(?:not|never|no|neither|nor|without|false|will|would|may|might|"
    r"could|if|whether|unless|assuming|supposing|unknown|probably|apparently|appears?|"
    r"seems?|plan(?:ned|ning)?|expected|scheduled|proposed|future|possibly|perhaps|"
    r"reportedly|allegedly|purportedly|supposedly|disputed|uncertain|unconfirmed|"
    r"rumou?r(?:ed|s)?|denied|debunked|refuted|rejected|believed|"
    r"claim(?:ed|s|ing)?|intend(?:ed|s|ing)?)\b|\?)",
    re.I,
)
DISALLOWED_OPENING_SOURCE_CONTEXT = re.compile(
    r"\b(?:review(?:er|s|ed)?|comment(?:s|ed)?|article|post(?:ed)?|copyright|"
    r"published|publication|opening\s+hours?)\b|©",
    re.I,
)
OPENING_EXTRACTOR_RULE = "opening_context_v2"
OPENING_EXTRACTOR_VERSION = "2.0.0"
CLAIMS = frozenset({"calculated_naics", "target_construct_status", "business_status", "candidate_plausibility", "identity_consistency", "historical_year"})
CONFIDENCE = frozenset({"High", "Moderate", "Low", "NotAssessable"})
BUSINESS = frozenset({"Active", "Inactive", "Unknown"})
TARGET_CONSTRUCT = frozenset({"Supported", "NotSupported", "Unresolved", "Conflicting"})
PLAUSIBILITY = frozenset({"Supported", "Refuted", "Unresolved", "Conflicting"})
IDENTITY = frozenset({"Consistent", "Ambiguous", "Inconsistent", "NotAssessable"})
YEAR_TYPES = frozenset({"explicit_archived_opening_claim", "active_by_upper_bound", "unknown"})
RESPONSE = frozenset({"success", "timeout", "http_429", "http_4xx", "http_5xx", "decode_error", "other_error"})
QUERY = frozenset({"success", "success_no_capture", "partial", "missing_url", "invalid_url", "timeout", "http_429", "http_4xx", "http_5xx", "decode_error", "other_error"})
PAGE = frozenset({"not_attempted", "success", "partial_success", "all_failed"})
CAP = frozenset({"not_capped", "max_pages", "max_records", "not_applicable"})
ARCHIVE_RETRIEVAL_SOURCES = frozenset({"direct_wayback", "outscraper_ai"})


def _req(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _int(value: Any, name: str, minimum: int | None = None) -> int:
    _req(type(value) is int, f"{name} must be an integer")
    if minimum is not None:
        _req(value >= minimum, f"invalid {name}")
    return value


def _utc(value: Any, name: str = "timestamp") -> datetime:
    _req(isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None, f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _hash(value: Any, name: str = "hash") -> str:
    _req(isinstance(value, str) and SHA.fullmatch(value) is not None, f"invalid {name}")
    return value


def _validate_archive_retrieval(source: Any, provenance: Any) -> None:
    _req(isinstance(source, str) and source in ARCHIVE_RETRIEVAL_SOURCES,
         "invalid archive retrieval source")
    if provenance is not None:
        _hash(provenance, "archive retrieval provenance hash")
    _req(source != "outscraper_ai" or provenance is not None,
         "fallback archive retrieval requires provenance")


def _archive_retrieval_metadata(source: str, provenance: str | None) -> dict[str, Any]:
    """Keep legacy direct serialization stable; explicitly label new provenance.

    A fallback commitment points to a provider/adaptor envelope. Its extracted
    bytes support structural validation only, never a claim to original HTTP
    response fidelity. Direct provenance points to the archived HTTP envelope.
    """
    if source == "direct_wayback" and provenance is None:
        return {}
    return {"retrieval_source": source, "provenance_sha256": provenance,
            "source_fidelity": ("structural_only" if source == "outscraper_ai"
                                else "original_http")}


def _has_visible_text(value: str) -> bool:
    """Require lexical content; marks, selectors, blank glyphs, and controls do not count."""
    return any(unicodedata.category(char).startswith(("L", "N"))
               and "FILLER" not in unicodedata.name(char, "")
               for char in value)


def opening_claim_has_assertive_context(source_text: str, excerpt: str) -> bool:
    """Check an excerpt in its authenticated archive text block.

    This prevents a bare allowlisted phrase from being clipped out of a denial such
    as ``It is false that it opened in 1998``. Newlines are authenticated HTML block
    boundaries emitted by the collector. A short preceding label ending in a colon
    remains part of the context, so ``Denied claim:\nopened in 1998`` is rejected.
    The claim constructor authenticates the phrase itself; this check authenticates
    its local meaning in the saved page text.
    """
    _req(isinstance(source_text, str) and isinstance(excerpt, str) and excerpt,
         "invalid opening source context")
    start = 0
    while True:
        index = source_text.find(excerpt, start)
        if index < 0:
            return False
        block_start = source_text.rfind("\n", 0, index) + 1
        block_end = source_text.find("\n", index + len(excerpt))
        if block_end < 0:
            block_end = len(source_text)
        context_start = block_start
        if block_start > 0:
            previous_end = block_start - 1
            previous_start = source_text.rfind("\n", 0, previous_end) + 1
            previous = source_text[previous_start:previous_end].strip()
            if previous.endswith(":") and len(previous) <= 200:
                context_start = previous_start
        context = source_text[context_start:block_end].strip()
        if (DISALLOWED_OPENING_CONTEXT.search(context) is None
                and DISALLOWED_OPENING_SOURCE_CONTEXT.search(context) is None):
            return True
        start = index + 1


def _strs(value: Any, name: str) -> tuple[str, ...]:
    _req(isinstance(value, (list, tuple)) and all(isinstance(item, str) and item for item in value), f"invalid {name}")
    return tuple(value)


def freeze_json(value: Any) -> Any:
    """Convert the JSON domain to deeply immutable tuples and mapping proxies."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        _req(math.isfinite(value), "nonfinite JSON float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            _req(isinstance(key, str), "JSON keys must be strings")
            frozen[key] = freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise EvidenceValidationError("unsupported JSON value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def evidence_id(prefix: str, canonical_payload: Mapping[str, Any]) -> str:
    _req(isinstance(prefix, str) and prefix, "empty ID prefix")
    try:
        raw = json.dumps(_thaw(freeze_json(canonical_payload)), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError("invalid canonical payload") from exc
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()}"


def _contains_control(value: str) -> bool:
    return any(
        ord(char) == 127 or unicodedata.category(char) in {"Cc", "Cf", "Cs"}
        for char in value
    )


def _decode_layers(value: str) -> tuple[str, ...]:
    """Percent-decode to a fixed point and reject controls at every layer."""
    _req(len(value) <= 4096, "URL too long")
    layers: list[str] = []
    current = value
    for _ in range(len(value) + 1):
        _req(not _contains_control(current), "invalid URL")
        layers.append(current)
        following = unquote(current)
        if following == current:
            return tuple(layers)
        current = following
    raise EvidenceValidationError("URL decoding did not reach a fixed point")


def _safe_url(value: Any, *, replay: bool = False) -> str:
    _req(isinstance(value, str) and value, "invalid URL")
    _validate_url_layers(value, set(), 0)
    if replay:
        _parse_replay_url(value)
    return value


def _validate_url_layers(value: str, seen: set[str], depth: int) -> None:
    """Validate every decode layer and every nested absolute query URL."""
    _req(depth <= 8, "unsafe nested URL")
    if value in seen:
        return
    seen.add(value)
    for layer in _decode_layers(value):
        parsed = urlparse(layer)
        _req(parsed.scheme.lower() in {"http", "https"} and parsed.hostname
             and parsed.username is None and parsed.password is None
             and not parsed.fragment, "unsafe URL")
        try:
            parsed.port
        except ValueError as exc:
            raise EvidenceValidationError("unsafe URL") from exc
        if parsed.query:
            for key, raw_value in parse_qsl(parsed.query, keep_blank_values=True):
                for item in (key, raw_value):
                    _req(SENSITIVE.search(item) is None, "sensitive URL query")
                nested = raw_value.strip()
                if re.match(r"(?i)^https?://", nested):
                    _validate_url_layers(nested, seen, depth + 1)
                elif nested.startswith("//"):
                    _validate_url_layers("https:" + nested, seen, depth + 1)


def canonical_http_url(value: Any) -> str:
    """Return the conservative canonical HTTP(S) form used for provenance binding."""
    _safe_url(value)
    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    assert parsed.hostname is not None
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise EvidenceValidationError("unsafe URL") from exc
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None and not ((scheme == "http" and port == 80)
                                 or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    return urlunparse((scheme, host, parsed.path or "/", parsed.params, parsed.query, ""))


def _parse_replay_url(value: str) -> tuple[str, str]:
    parsed = urlparse(value)
    _req(parsed.scheme == "https" and parsed.netloc == "web.archive.org"
         and parsed.username is None and parsed.password is None and not parsed.fragment,
         "invalid replay URL")
    tail = (parsed.path + ((";" + parsed.params) if parsed.params else "")
            + (("?" + parsed.query) if parsed.query else ""))
    match = re.fullmatch(r"/web/(\d{14})id_/(https?://.+)", tail)
    _req(match is not None, "invalid replay URL")
    raw_target = match.group(2)
    target = canonical_http_url(raw_target)
    return match.group(1), target


def cdx_response_chain_hash(*bodies: bytes) -> str:
    """Commit to the ordered exact CDX response bytes with length framing."""
    _req(all(isinstance(body, bytes) for body in bodies),
         "invalid CDX response bytes")
    digest = hashlib.sha256()
    digest.update(b"wayback-cdx-response-v2\0")
    for body in bodies:
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def _opaque(value: Any, prefix: str, name: str) -> str:
    _req(isinstance(value, str) and re.fullmatch(rf"{re.escape(prefix)}_[0-9a-f]{{64}}", value) is not None, f"invalid {name}")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class QueryProvenanceV2:
    query_definition_id: str
    source_zip: str
    retrieval_channel: str

    def __post_init__(self) -> None:
        _req(isinstance(self.query_definition_id, str) and self.query_definition_id in {"pharmacy_places_new_v2", "drug_store_places_new_v2", "drugstore_places_new_v2"} and isinstance(self.source_zip, str) and re.fullmatch(r"\d{5}", self.source_zip) is not None and self.retrieval_channel == "places_search", "invalid approved query definition")


@dataclass(frozen=True, slots=True, repr=False)
class CandidateFactsV2:
    candidate_id: str; source_place_id: str; provider_business_status: str; name: str; formatted_address: str
    phone: str | None; latitude: float | None; longitude: float | None; website: str | None
    opening_hours: tuple[str, ...]; types: tuple[str, ...]; rating: float | None; user_ratings_total: int | None
    query_provenance: QueryProvenanceV2; retrieved_at: datetime

    def __post_init__(self) -> None:
        _req(isinstance(self.source_place_id, str) and PLACE.fullmatch(self.source_place_id) is not None, "invalid Place ID")
        _req(self.candidate_id == evidence_id("candidate", {"source_place_id": self.source_place_id}), "invalid candidate ID")
        _req(isinstance(self.provider_business_status, str) and self.provider_business_status in {"OPERATIONAL", "CLOSED_TEMPORARILY", "CLOSED_PERMANENTLY", "UNKNOWN"}, "invalid provider status")
        _req(all(isinstance(item, str) and item for item in (self.name, self.formatted_address)) and (self.phone is None or isinstance(self.phone, str)), "invalid candidate text")
        _req(self.website is None or isinstance(self.website, str), "invalid candidate website")
        _req((self.latitude is None) == (self.longitude is None), "coordinates must be a pair")
        if self.latitude is not None:
            _req(type(self.latitude) in {int, float} and type(self.longitude) in {int, float} and -90 <= self.latitude <= 90 and -180 <= self.longitude <= 180, "invalid coordinates")
        _req(self.rating is None or (type(self.rating) in {int, float} and 0 <= self.rating <= 5), "invalid rating")
        if self.user_ratings_total is not None:
            _int(self.user_ratings_total, "rating count", 0)
        _req(isinstance(self.query_provenance, QueryProvenanceV2), "invalid query provenance")
        object.__setattr__(self, "opening_hours", _strs(self.opening_hours, "opening hours"))
        object.__setattr__(self, "types", _strs(self.types, "types"))
        object.__setattr__(self, "retrieved_at", _utc(self.retrieved_at))


@dataclass(frozen=True, slots=True, repr=False)
class ReviewSnippetV2:
    evidence_id: str; source_place_id: str; published_at: datetime; rating: int | None; text: str; provider_order_index: int

    def __post_init__(self) -> None:
        published = _utc(self.published_at, "review publication")
        _req(isinstance(self.source_place_id, str)
             and PLACE.fullmatch(self.source_place_id) is not None
             and isinstance(self.text, str) and len(self.text) <= 200
             and (_has_visible_text(self.text) or self.rating is not None),
             "invalid review")
        if self.rating is not None:
            _req(type(self.rating) is int and 1 <= self.rating <= 5, "invalid review rating")
        _int(self.provider_order_index, "provider review order", 0)
        payload = {"source_place_id": self.source_place_id, "published_at": _iso(published), "rating": self.rating, "text": self.text, "provider_order_index": self.provider_order_index}
        _req(self.evidence_id == evidence_id("review", payload), "invalid review ID")
        object.__setattr__(self, "published_at", published)


@dataclass(frozen=True, slots=True, repr=False)
class ReviewEvidenceV2:
    status: str; source_place_id: str; retrieved_at: datetime; review_selection: str; returned_count: int; total_user_rating_count: int | None; snippets: tuple[ReviewSnippetV2, ...]

    def __post_init__(self) -> None:
        snippets = tuple(self.snippets) if isinstance(self.snippets, (list, tuple)) else ()
        retrieved = _utc(self.retrieved_at)
        _req(isinstance(self.source_place_id, str)
             and PLACE.fullmatch(self.source_place_id) is not None
             and isinstance(self.snippets, (list, tuple))
             and isinstance(self.status, str)
             and self.status in {"success", "missing", "timeout", "http_429", "http_4xx", "http_5xx", "decode_error", "other_error", "not_requested"}
             and self.review_selection == "provider_relevance",
             "invalid review packet")
        _int(self.returned_count, "returned review count", 0)
        if self.total_user_rating_count is not None: _int(self.total_user_rating_count, "total user rating count", 0)
        _req(self.returned_count <= 5 and (self.total_user_rating_count is None or self.returned_count <= self.total_user_rating_count) and len(snippets) == self.returned_count and all(isinstance(item, ReviewSnippetV2) for item in snippets), "invalid review packet")
        _req(len({item.evidence_id for item in snippets}) == len(snippets)
             and tuple(item.provider_order_index for item in snippets)
             == tuple(range(self.returned_count))
             and all(item.source_place_id == self.source_place_id
                     and item.published_at <= retrieved for item in snippets),
             "reviews must retain provider order and dated source binding")
        _req(self.status == "success" or not snippets, "failed review packet has content")
        _req(self.status == "success" or self.total_user_rating_count is None,
             "failed review packet cannot report a Details total")
        object.__setattr__(self, "snippets", snippets)
        object.__setattr__(self, "retrieved_at", retrieved)


@dataclass(frozen=True, slots=True)
class ArchiveCDXCaptureRowV2:
    evidence_id: str; captured_at: datetime; original_url: str; mimetype: str; status_code: int; digest: str; length: int
    raw_original_url: str | None = None

    def __post_init__(self) -> None:
        captured = _utc(self.captured_at, "capture timestamp")
        _req(captured.microsecond == 0, "CDX timestamp must have second precision")
        _safe_url(self.original_url)
        _req(self.original_url == canonical_http_url(self.original_url),
             "CDX original URL must be canonical")
        if self.raw_original_url is not None:
            _req(canonical_http_url(self.raw_original_url) == self.original_url,
                 "raw CDX original URL must match its canonical URL")
        _req(self.mimetype == "text/html", "invalid CDX mimetype")
        _req(self.status_code == 200, "invalid CDX status code")
        _req(isinstance(self.digest, str) and CDX_SHA1.fullmatch(self.digest) is not None, "invalid CDX digest"); _int(self.length, "CDX length", 0)
        payload = {"captured_at": _iso(captured), "original_url": self.original_url, "mimetype": self.mimetype, "status_code": self.status_code, "digest": self.digest, "length": self.length}
        _req(self.evidence_id == evidence_id("cdx_capture_v2", payload), "invalid CDX capture ID")
        object.__setattr__(self, "captured_at", captured)


def capture_request_id(row: ArchiveCDXCaptureRowV2) -> str:
    _req(isinstance(row, ArchiveCDXCaptureRowV2), "capture request needs an authenticated CDX row")
    payload = {"capture_id": row.evidence_id, "captured_at": _iso(row.captured_at), "original_url": row.original_url, "mimetype": row.mimetype, "status_code": row.status_code, "digest": row.digest, "length": row.length}
    return evidence_id("capture_request_v2", payload)


def parse_cdx_response_v2(body: bytes) -> tuple[
        tuple[ArchiveCDXCaptureRowV2, ...], str | None]:
    """Parse exact CDX JSON bytes into canonical retained rows and resume digest.

    The fixed server-side filters remain an untrusted response boundary, so any
    structural, header, row, or resume failure rejects the entire response. The
    returned digest authenticates, without retaining, the resume key.
    """
    _req(isinstance(body, bytes), "invalid CDX response bytes")
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError("invalid CDX JSON") from exc
    _req(isinstance(decoded, list) and decoded
         and isinstance(decoded[0], list)
         and all(isinstance(item, str) for item in decoded[0]),
         "invalid CDX table")
    header = tuple(decoded[0])
    required = ("timestamp", "original", "mimetype", "statuscode", "digest",
                "length")
    _req(len(header) == len(set(header)) and set(header) == set(required),
         "invalid CDX header")
    rows_raw = list(decoded[1:])
    resume_digest: str | None = None
    if (len(rows_raw) >= 2 and rows_raw[-2] == []
            and isinstance(rows_raw[-1], list) and len(rows_raw[-1]) == 1
            and isinstance(rows_raw[-1][0], str) and rows_raw[-1][0]):
        resume_digest = hashlib.sha256(rows_raw[-1][0].encode("utf-8")).hexdigest()
        rows_raw = rows_raw[:-2]
    elif any(item == [] for item in rows_raw):
        raise EvidenceValidationError("invalid CDX resume record")
    elif (rows_raw and isinstance(rows_raw[-1], list)
          and len(rows_raw[-1]) == 1 and isinstance(rows_raw[-1][0], str)):
        raise EvidenceValidationError("orphan CDX resume key")
    retained: dict[str, ArchiveCDXCaptureRowV2] = {}
    for raw in rows_raw:
        _req(isinstance(raw, list) and len(raw) == len(header),
             "malformed CDX data row")
        values = dict(zip(header, raw, strict=True))
        _req(all(isinstance(values[item], str) for item in required),
             "malformed CDX field type")
        original = values["original"]
        _req(re.match(r"(?i)^https?://", original) is not None,
             "invalid CDX original URL")
        try:
            canonical_original = canonical_http_url(original)
        except EvidenceValidationError as exc:
            raise EvidenceValidationError(
                "unsafe URL in exact CDX response"
            ) from exc
        if (re.fullmatch(r"\d{14}", values["timestamp"]) is None
                or values["statuscode"] != "200"
                or re.fullmatch(r"(?:0|[1-9]\d*)", values["length"]) is None):
            raise EvidenceValidationError("invalid exact CDX field")
        try:
            captured = datetime.strptime(values["timestamp"],
                                         "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            status_code = int(values["statuscode"])
            length = int(values["length"])
            payload = {
                "captured_at": _iso(captured),
                "original_url": canonical_original,
                "mimetype": values["mimetype"],
                "status_code": status_code,
                "digest": values["digest"],
                "length": length,
            }
            row = ArchiveCDXCaptureRowV2(
                evidence_id("cdx_capture_v2", payload), captured,
                canonical_original, values["mimetype"], status_code,
                values["digest"], length,
                raw_original_url=(original if original != canonical_original else None),
            )
        except (EvidenceValidationError, ValueError, TypeError) as exc:
            raise EvidenceValidationError("invalid exact CDX data row") from exc
        retained[row.evidence_id] = row
    rows = tuple(sorted(
        retained.values(),
        key=lambda row: (row.captured_at, canonical_http_url(row.original_url),
                         row.digest, row.length, row.evidence_id),
    ))
    return rows, resume_digest


@dataclass(frozen=True, slots=True, repr=False)
class ArchiveCDXPageV2:
    """Commit to CDX bytes, with explicit direct or extracted-response provenance.

    For ``outscraper_ai``, ``response_body`` is the extracted CDX document and
    ``http_status`` is the provider/adaptor status. Neither represents an
    authenticated original Wayback HTTP response; the sidecar records that limit.
    """
    page_number: int; status: str; attempts: int; retrieved_at: datetime
    http_status: int | None; response_sha256: str | None; response_length: int | None
    returned_rows: int; returned_years: tuple[int, ...]
    incoming_resume_key_digest: str | None; outgoing_resume_key_digest: str | None
    pagination_complete: bool; cache_provenance: str
    capture_rows: tuple[ArchiveCDXCaptureRowV2, ...]
    response_body: bytes | None = None
    retrieval_source: str = "direct_wayback"
    provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_archive_retrieval(self.retrieval_source, self.provenance_sha256)
        _req(isinstance(self.capture_rows, (tuple, list)), "invalid CDX capture rows")
        _req(isinstance(self.returned_years, (tuple, list)), "invalid returned years")
        rows = tuple(self.capture_rows)
        years = tuple(self.returned_years)
        _int(self.page_number, "CDX page number", 1); _int(self.attempts, "CDX attempts", 1)
        _req(isinstance(self.status, str) and self.status in RESPONSE
             and type(self.pagination_complete) is bool, "invalid CDX page")
        _opaque(self.cache_provenance, "wayback_cache_v2", "cache provenance")
        _req(self.incoming_resume_key_digest is None or
             (isinstance(self.incoming_resume_key_digest, str)
              and SHA.fullmatch(self.incoming_resume_key_digest) is not None),
             "invalid incoming resume digest")
        _req(self.outgoing_resume_key_digest is None or
             (isinstance(self.outgoing_resume_key_digest, str)
              and SHA.fullmatch(self.outgoing_resume_key_digest) is not None),
             "invalid outgoing resume digest")
        _req(all(isinstance(row, ArchiveCDXCaptureRowV2) for row in rows) and len({row.evidence_id for row in rows}) == len(rows), "invalid CDX capture rows")
        if self.status == "success":
            _req(self.http_status == 200 and self.response_sha256 is not None
                 and type(self.response_length) is int and self.response_length >= 0,
                 "invalid successful CDX response")
            _int(self.returned_rows, "returned rows", 0)
            derived_years = tuple(sorted({row.captured_at.year for row in rows}))
            _req(self.returned_rows == len(rows) and years == derived_years, "CDX summaries must derive from capture rows")
            _req(all(row.captured_at <= _utc(self.retrieved_at)
                     for row in rows),
                 "CDX response cannot contain future captures")
            _req(not rows or self.response_length > 0,
                 "nonempty CDX rows require a nonempty response")
            _req((self.pagination_complete and self.outgoing_resume_key_digest is None)
                 or (not self.pagination_complete
                     and self.outgoing_resume_key_digest is not None),
                 "invalid CDX resume state")
        elif self.status == "decode_error":
            _req(type(self.http_status) is int and 200 <= self.http_status < 300
                 and self.response_sha256 is not None and type(self.response_length) is int
                 and self.response_length >= 0, "invalid decode-error CDX response")
            _req(not rows and self.returned_rows == 0 and not years
                 and not self.pagination_complete and self.outgoing_resume_key_digest is None,
                 "invalid decode-error rows")
        elif self.status in {"http_429", "http_4xx", "http_5xx"}:
            _req(type(self.http_status) is int
                 and ((self.status == "http_429" and self.http_status == 429)
                      or (self.status == "http_4xx" and 400 <= self.http_status < 500
                          and self.http_status != 429)
                      or (self.status == "http_5xx" and 500 <= self.http_status < 600))
                 and self.response_sha256 is not None and type(self.response_length) is int
                 and self.response_length >= 0, "invalid HTTP CDX response")
            _req(not rows and self.returned_rows == 0 and not years
                 and not self.pagination_complete and self.outgoing_resume_key_digest is None,
                 "invalid HTTP CDX rows")
        elif self.status == "timeout":
            _req(self.http_status is None and self.response_sha256 is None
                 and self.response_length is None and not rows
                 and self.returned_rows == 0 and not years
                 and not self.pagination_complete and self.outgoing_resume_key_digest is None,
                 "invalid timeout CDX provenance")
        else:
            no_response = (self.http_status is None and self.response_sha256 is None
                           and self.response_length is None)
            has_response = (type(self.http_status) is int
                            and 100 <= self.http_status < 400
                            and self.response_sha256 is not None
                            and type(self.response_length) is int
                            and self.response_length >= 0)
            _req((no_response or has_response) and not rows and self.returned_rows == 0
                 and not years and not self.pagination_complete
                 and self.outgoing_resume_key_digest is None,
                 "invalid other-error CDX provenance")
        if self.response_sha256 is not None:
            _hash(self.response_sha256, "CDX response hash")
            _req(isinstance(self.response_body, bytes)
                 and len(self.response_body) == self.response_length
                 and hashlib.sha256(self.response_body).hexdigest()
                 == self.response_sha256,
                 "CDX response bytes do not match their commitment")
        else:
            _req(self.response_body is None,
                 "response-free CDX state cannot carry bytes")
        if self.status == "success":
            parsed_rows, parsed_resume = parse_cdx_response_v2(self.response_body)
            retained_ids = {row.evidence_id for row in rows}
            ordered_subset = tuple(
                row for row in parsed_rows if row.evidence_id in retained_ids
            )
            _req(rows == ordered_subset,
                 "CDX capture rows are not an ordered response subset")
            _req(self.outgoing_resume_key_digest == parsed_resume
                 and self.pagination_complete == (parsed_resume is None),
                 "CDX pagination does not derive from exact response bytes")
        object.__setattr__(self, "capture_rows", rows); object.__setattr__(self, "returned_years", years); object.__setattr__(self, "retrieved_at", _utc(self.retrieved_at))

    def to_private_dict(self) -> dict[str, Any]:
        return {"page_number": self.page_number, "status": self.status,
                "attempts": self.attempts, "retrieved_at": _iso(self.retrieved_at),
                "http_status": self.http_status, "response_sha256": self.response_sha256,
                "response_length": self.response_length,
                "returned_rows": self.returned_rows,
                "returned_years": list(self.returned_years),
                "incoming_resume_key_digest": self.incoming_resume_key_digest,
                "outgoing_resume_key_digest": self.outgoing_resume_key_digest,
                "pagination_complete": self.pagination_complete,
                "cache_provenance": self.cache_provenance,
                "capture_rows": [{"evidence_id": row.evidence_id,
                                  "captured_at": _iso(row.captured_at),
                                  "original_url": row.original_url,
                                  "mimetype": row.mimetype,
                                  "status_code": row.status_code,
                                  "digest": row.digest, "length": row.length,
                                  **({"raw_original_url": row.raw_original_url}
                                     if row.raw_original_url is not None else {})}
                                 for row in self.capture_rows],
                **_archive_retrieval_metadata(self.retrieval_source,
                                             self.provenance_sha256)}

    def to_public_dict(self) -> dict[str, Any]:
        return {"page_number": self.page_number, "status": self.status, "attempts": self.attempts, "retrieved_at": _iso(self.retrieved_at), "returned_rows": self.returned_rows, "returned_years": list(self.returned_years), "pagination_complete": self.pagination_complete,
                **_archive_retrieval_metadata(self.retrieval_source,
                                             self.provenance_sha256)}


@dataclass(frozen=True, slots=True, repr=False)
class ArchiveSourceSnapshotV2:
    evidence_id: str; capture_request_id: str; captured_at: datetime; replay_url: str; original_url: str; archived_page_text: str; body_truncated: bool; text_truncated: bool
    retrieval_source: str = "direct_wayback"
    provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_archive_retrieval(self.retrieval_source, self.provenance_sha256)
        captured = _utc(self.captured_at, "source capture timestamp")
        _req(captured.microsecond == 0,
             "source timestamp must have second precision")
        _opaque(self.capture_request_id, "capture_request_v2", "capture request ID")
        _safe_url(self.replay_url, replay=True)
        _safe_url(self.original_url)
        _req(self.original_url == canonical_http_url(self.original_url),
             "source original URL must be canonical")
        _req(isinstance(self.archived_page_text, str) and 0 < len(self.archived_page_text) <= 5000 and type(self.body_truncated) is bool and type(self.text_truncated) is bool, "invalid bounded archive text")
        page_hash = hashlib.sha256(self.archived_page_text.encode()).hexdigest()
        payload = {"capture_request_id": self.capture_request_id, "captured_at": _iso(captured), "replay_url": self.replay_url, "original_url": self.original_url, "page_text_sha256": page_hash}
        _req(self.evidence_id == evidence_id("archive_source_v2", payload), "source ID must use only Wayback fields")
        object.__setattr__(self, "captured_at", captured)

    @property
    def page_text_sha256(self) -> str:
        return hashlib.sha256(self.archived_page_text.encode()).hexdigest()

    def to_private_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "capture_request_id": self.capture_request_id, "captured_at": _iso(self.captured_at), "replay_url": self.replay_url, "original_url": self.original_url, "archived_page_text": self.archived_page_text, "page_text_sha256": self.page_text_sha256, "body_truncated": self.body_truncated, "text_truncated": self.text_truncated,
                **_archive_retrieval_metadata(self.retrieval_source,
                                             self.provenance_sha256)}

    def to_public_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "captured_at": _iso(self.captured_at), "body_truncated": self.body_truncated, "text_truncated": self.text_truncated,
                **_archive_retrieval_metadata(self.retrieval_source,
                                             self.provenance_sha256)}


@dataclass(frozen=True, slots=True)
class ArchiveFetchRecordV2:
    capture_request_id: str; source_evidence_id: str | None; status: str; attempts: int; retrieved_at: datetime; http_status: int | None; response_sha256: str | None; cache_provenance: str
    retrieval_source: str = "direct_wayback"
    provenance_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_archive_retrieval(self.retrieval_source, self.provenance_sha256)
        _opaque(self.capture_request_id, "capture_request_v2", "capture request ID"); _req(isinstance(self.status, str) and self.status in RESPONSE, "invalid fetch status"); _int(self.attempts, "fetch attempts", 1); _opaque(self.cache_provenance, "wayback_cache_v2", "cache provenance")
        if self.status == "success":
            _req(isinstance(self.source_evidence_id, str)
                 and re.fullmatch(r"archive_source_v2_[0-9a-f]{64}",
                                  self.source_evidence_id) is not None
                 and self.http_status == 200
                 and self.response_sha256 is not None,
                 "invalid successful fetch")
        elif self.status == "decode_error":
            _req(self.source_evidence_id is None and type(self.http_status) is int and 200 <= self.http_status < 300 and self.response_sha256 is not None, "invalid decode-error fetch")
        elif self.status in {"http_429", "http_4xx", "http_5xx"}:
            _req(self.source_evidence_id is None and type(self.http_status) is int and ((self.status == "http_429" and self.http_status == 429) or (self.status == "http_4xx" and 400 <= self.http_status < 500 and self.http_status != 429) or (self.status == "http_5xx" and 500 <= self.http_status < 600)) and self.response_sha256 is not None, "invalid HTTP fetch")
        elif self.status == "timeout":
            _req(self.source_evidence_id is None and self.http_status is None
                 and self.response_sha256 is None, "invalid timeout fetch")
        else:
            _req(self.source_evidence_id is None, "invalid other-error source")
            _req((self.http_status is None and self.response_sha256 is None)
                 or (type(self.http_status) is int and 100 <= self.http_status < 400
                     and self.response_sha256 is not None),
                 "invalid other-error fetch")
        if self.response_sha256 is not None: _hash(self.response_sha256, "fetch response hash")
        object.__setattr__(self, "retrieved_at", _utc(self.retrieved_at))

    def to_private_dict(self) -> dict[str, Any]:
        return {"capture_request_id": self.capture_request_id, "source_evidence_id": self.source_evidence_id, "status": self.status, "attempts": self.attempts, "retrieved_at": _iso(self.retrieved_at), "http_status": self.http_status, "response_sha256": self.response_sha256, "cache_provenance": self.cache_provenance,
                **_archive_retrieval_metadata(self.retrieval_source,
                                             self.provenance_sha256)}

    def to_public_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.to_private_dict().items() if key not in {"response_sha256", "cache_provenance"}}


@dataclass(frozen=True, slots=True)
class ArchiveOpeningClaimV2:
    evidence_id: str; source_evidence_id: str; year: int; excerpt: str; page_text_sha256: str; extractor_rule: str; extractor_version: str

    def __post_init__(self) -> None:
        _int(self.year, "opening year", 1600); _req(isinstance(self.excerpt, str) and 0 < len(self.excerpt) <= 200 and str(self.year) in self.excerpt, "invalid opening excerpt")
        _req(DISALLOWED_OPENING_CONTEXT.search(self.excerpt) is None
             and DISALLOWED_OPENING_SOURCE_CONTEXT.search(self.excerpt) is None,
             "negated, future, or uncertain opening claim")
        matches = tuple(OPENING.finditer(self.excerpt))
        years = {
            int(next(item for item in match.groups() if item is not None))
            for match in matches
        }
        _req(bool(matches) and years == {self.year}
             and ASSERTIVE_OPENING.fullmatch(self.excerpt) is not None,
             "invalid assertive relational opening claim")
        _hash(self.page_text_sha256, "page hash")
        _req(isinstance(self.source_evidence_id, str)
             and re.fullmatch(r"archive_source_v2_[0-9a-f]{64}",
                              self.source_evidence_id) is not None,
             "invalid claim provenance")
        _req((self.extractor_rule, self.extractor_version)
             == (OPENING_EXTRACTOR_RULE, OPENING_EXTRACTOR_VERSION),
             "unauthenticated extractor provenance")
        payload = {"source_evidence_id": self.source_evidence_id, "year": self.year, "excerpt": self.excerpt, "page_text_sha256": self.page_text_sha256, "extractor_rule": self.extractor_rule, "extractor_version": self.extractor_version}
        _req(self.evidence_id == evidence_id("opening_claim_v2", payload), "invalid opening-claim ID")

    def to_private_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "source_evidence_id": self.source_evidence_id, "year": self.year, "excerpt": self.excerpt, "page_text_sha256": self.page_text_sha256, "extractor_rule": self.extractor_rule, "extractor_version": self.extractor_version}

    def to_public_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "source_evidence_id": self.source_evidence_id, "year": self.year, "extractor_rule": self.extractor_rule, "extractor_version": self.extractor_version}


@dataclass(frozen=True, slots=True, repr=False)
class ArchiveMatchAssessmentV2:
    source_evidence_id: str; normalized_address_match: bool; exact_phone_match: bool; compatible_name_city_match: bool
    def __post_init__(self) -> None: _req(isinstance(self.source_evidence_id, str) and re.fullmatch(r"archive_source_v2_[0-9a-f]{64}", self.source_evidence_id) is not None and all(type(value) is bool for value in (self.normalized_address_match, self.exact_phone_match, self.compatible_name_city_match)), "invalid source assessment")
    @property
    def evidence_scope(self) -> str:
        return "outlet_specific" if self.normalized_address_match or self.exact_phone_match else ("probable_outlet" if self.compatible_name_city_match else "domain_or_brand_only")


@dataclass(frozen=True, slots=True, repr=False)
class ArchiveClaimAssessmentV2:
    claim_evidence_id: str; normalized_address_match: bool; exact_phone_match: bool; compatible_name_city_match: bool
    def __post_init__(self) -> None: _req(isinstance(self.claim_evidence_id, str) and re.fullmatch(r"opening_claim_v2_[0-9a-f]{64}", self.claim_evidence_id) is not None and all(type(value) is bool for value in (self.normalized_address_match, self.exact_phone_match, self.compatible_name_city_match)), "invalid claim assessment")
    @property
    def evidence_scope(self) -> str:
        return "outlet_specific" if self.normalized_address_match or self.exact_phone_match else ("probable_outlet" if self.compatible_name_city_match else "domain_or_brand_only")


def _derived_query_status(pages: tuple[ArchiveCDXPageV2, ...], cap_status: str) -> str:
    if not pages:
        return ""
    successful = [page for page in pages if page.status == "success"]
    failures = [page for page in pages if page.status != "success"]
    if not successful:
        statuses = {page.status for page in pages}
        return next(iter(statuses)) if len(statuses) == 1 else "other_error"
    complete = (not failures and cap_status == "not_capped"
                and pages[-1].pagination_complete
                and all(not page.pagination_complete
                        and page.outgoing_resume_key_digest is not None
                        for page in pages[:-1]))
    if not complete:
        return "partial"
    if not any(page.returned_rows for page in pages):
        return "success_no_capture"
    return "success"


def _derived_page_status(pages: tuple[ArchiveCDXPageV2, ...], fetches: tuple[ArchiveFetchRecordV2, ...]) -> str:
    if not pages or not any(page.returned_rows for page in pages) or not fetches:
        return "not_attempted"
    successful = sum(fetch.status == "success" for fetch in fetches)
    return "success" if successful == len(fetches) else ("partial_success" if successful else "all_failed")


@dataclass(frozen=True, slots=True)
class ArchiveAnnualSelectionV2:
    """Authenticated oldest-first replay outcome for one returned CDX year."""
    year: int
    max_attempts: int
    attempted_capture_request_ids: tuple[str, ...]
    selected_source_evidence_id: str | None
    stop_reason: str

    def __post_init__(self) -> None:
        _int(self.year, "annual selection year", 1600)
        _int(self.max_attempts, "annual maximum attempts", 1)
        _req(isinstance(self.attempted_capture_request_ids, (tuple, list)),
             "invalid annual attempt container")
        attempted = tuple(self.attempted_capture_request_ids)
        _req(len(attempted) == len(set(attempted))
             and all(isinstance(item, str)
                     and re.fullmatch(r"capture_request_v2_[0-9a-f]{64}", item)
                     for item in attempted), "invalid annual replay requests")
        _req(self.selected_source_evidence_id is None
             or (isinstance(self.selected_source_evidence_id, str)
                 and re.fullmatch(r"archive_source_v2_[0-9a-f]{64}",
                                  self.selected_source_evidence_id)),
             "invalid annual source")
        _req(self.stop_reason in {"first_usable", "exhausted", "max_attempts",
                                  "not_attempted"}, "invalid annual stop reason")
        _req(len(attempted) <= self.max_attempts,
             "annual attempts exceed their maximum")
        if self.stop_reason == "not_attempted":
            _req(not attempted and self.selected_source_evidence_id is None,
                 "invalid deferred annual selection")
        elif self.stop_reason == "first_usable":
            _req(bool(attempted) and self.selected_source_evidence_id is not None,
                 "first-usable selection needs attempts and a source")
        elif self.stop_reason == "max_attempts":
            _req(len(attempted) == self.max_attempts
                 and self.selected_source_evidence_id is None,
                 "invalid maximum-attempt selection")
        else:
            _req(bool(attempted) and self.selected_source_evidence_id is None,
                 "invalid exhausted selection")
        object.__setattr__(self, "attempted_capture_request_ids", attempted)

    def to_private_dict(self) -> dict[str, Any]:
        return {"year": self.year, "max_attempts": self.max_attempts,
                "attempted_capture_request_ids": list(self.attempted_capture_request_ids),
                "selected_source_evidence_id": self.selected_source_evidence_id,
                "stop_reason": self.stop_reason}

    def to_public_dict(self) -> dict[str, Any]:
        return {"year": self.year, "attempted_count": len(self.attempted_capture_request_ids),
                "selected": self.selected_source_evidence_id is not None,
                "stop_reason": self.stop_reason}


@dataclass(frozen=True, slots=True)
class ArchiveEvidenceV2:
    query_status: str; page_status: str; cap_status: str; query_key_digest: str | None; query_scope: str; retrieved_at: datetime
    max_cdx_pages: int; max_cdx_records: int; cdx_response_chain_sha256: str | None
    cdx_pages: tuple[ArchiveCDXPageV2, ...]
    annual_selections: tuple[ArchiveAnnualSelectionV2, ...]
    fetch_records: tuple[ArchiveFetchRecordV2, ...]
    source_snapshots: tuple[ArchiveSourceSnapshotV2, ...]
    opening_claims: tuple[ArchiveOpeningClaimV2, ...]

    def __post_init__(self) -> None:
        containers = (self.cdx_pages, self.annual_selections, self.fetch_records,
                      self.source_snapshots, self.opening_claims)
        _req(all(isinstance(item, (tuple, list)) for item in containers),
             "invalid archive collection container")
        pages = tuple(self.cdx_pages)
        selections = tuple(self.annual_selections)
        fetches = tuple(self.fetch_records)
        sources = tuple(self.source_snapshots)
        claims = tuple(self.opening_claims)
        _int(self.max_cdx_pages, "maximum CDX pages", 1)
        _int(self.max_cdx_records, "maximum CDX records", 1)
        _req(isinstance(self.query_status, str) and self.query_status in QUERY
             and isinstance(self.page_status, str) and self.page_status in PAGE
             and isinstance(self.cap_status, str) and self.cap_status in CAP
             and isinstance(self.query_scope, str)
             and self.query_scope in {"candidate_website", "alternate_candidate_url"},
             "invalid archive aggregate")
        _req(all(isinstance(item, ArchiveCDXPageV2) for item in pages)
             and all(isinstance(item, ArchiveAnnualSelectionV2) for item in selections)
             and all(isinstance(item, ArchiveFetchRecordV2) for item in fetches)
             and all(isinstance(item, ArchiveSourceSnapshotV2) for item in sources)
             and all(isinstance(item, ArchiveOpeningClaimV2) for item in claims),
             "invalid archive members")
        if not pages:
            _req(self.query_status in {"missing_url", "invalid_url"}
                 and self.page_status == "not_attempted"
                 and self.cap_status == "not_applicable"
                 and self.query_key_digest is None
                 and self.cdx_response_chain_sha256 is None
                 and not selections and not fetches and not sources and not claims,
                 "invalid empty archive state")
        else:
            _opaque(self.query_key_digest, "wayback_query_v2", "query key digest")
            _hash(self.cdx_response_chain_sha256, "CDX response-chain hash")
            response_bodies = tuple(page.response_body for page in pages
                                    if page.response_body is not None)
            _req(self.cdx_response_chain_sha256
                 == cdx_response_chain_hash(*response_bodies),
                 "CDX response-chain hash does not match exact page bytes")
            _req(self.cap_status != "not_applicable", "attempted archive cannot be not-applicable")
            _req(tuple(page.page_number for page in pages) == tuple(range(1, len(pages) + 1)), "CDX pages must be ordered and unique")
            _req(pages[0].incoming_resume_key_digest is None,
                 "first CDX page cannot have an incoming resume key")
            for index, page in enumerate(pages):
                if index:
                    _req(page.incoming_resume_key_digest
                         == pages[index - 1].outgoing_resume_key_digest,
                         "CDX incoming resume chain is not authenticated")
                if index < len(pages) - 1:
                    _req(page.status == "success" and not page.pagination_complete
                         and page.outgoing_resume_key_digest is not None,
                         "completed or failed CDX page is terminal")
                _req(page.outgoing_resume_key_digest is None
                     or page.outgoing_resume_key_digest
                     != page.incoming_resume_key_digest,
                     "CDX resume chain has a cycle")
            resume_values = [page.outgoing_resume_key_digest for page in pages
                             if page.outgoing_resume_key_digest is not None]
            _req(len(resume_values) == len(set(resume_values)),
                 "CDX resume chain has a cycle")
            seen_response_rows: set[str] = set()
            for page in pages:
                if page.status != "success":
                    continue
                parsed_rows, _ = parse_cdx_response_v2(page.response_body)
                expected_retained = tuple(
                    row for row in parsed_rows
                    if row.evidence_id not in seen_response_rows
                )
                _req(page.capture_rows == expected_retained,
                     "CDX page may drop only prior-page duplicates")
                seen_response_rows.update(row.evidence_id for row in parsed_rows)
            capture_rows = tuple(row for page in pages for row in page.capture_rows)
            canonical_rows = tuple(sorted(
                capture_rows,
                key=lambda row: (row.captured_at, canonical_http_url(row.original_url),
                                 row.digest, row.length, row.evidence_id),
            ))
            _req(capture_rows == canonical_rows, "CDX capture rows must be canonical")
            total_rows = len(capture_rows)
            _req(len(pages) <= self.max_cdx_pages,
                 "CDX page limit exceeded")
            _req(total_rows <= self.max_cdx_records,
                 "CDX record limit exceeded")
            last = pages[-1]
            cap_terminal = (last.status == "success" and not last.pagination_complete
                            and last.outgoing_resume_key_digest is not None)
            if self.cap_status == "max_pages":
                _req(cap_terminal and len(pages) == self.max_cdx_pages
                     and total_rows < self.max_cdx_records,
                     "invalid page-cap proof")
            elif self.cap_status == "max_records":
                _req(cap_terminal and total_rows >= self.max_cdx_records,
                     "invalid record-cap proof")
            else:
                _req(self.cap_status == "not_capped", "invalid attempted cap state")
                _req(not cap_terminal,
                     "unconsumed CDX resume requires a declared cap or next page")

            expected_query = _derived_query_status(pages, self.cap_status)
            _req(self.query_status == expected_query,
                 "aggregate query status must derive from CDX facts")
            request_ids = {capture_request_id(row) for row in capture_rows}
            _req(len({row.evidence_id for row in capture_rows}) == len(capture_rows), "duplicate CDX capture row")
            _req(len({item.capture_request_id for item in fetches}) == len(fetches), "duplicate selected replay request")
            _req(all(item.capture_request_id in request_ids for item in fetches), "fetch must reference a selected CDX capture")

            returned_years = tuple(sorted({row.captured_at.year for row in capture_rows}))
            _req(tuple(item.year for item in selections) == returned_years,
                 "annual selections must cover returned years exactly")
            capture_by_year: dict[int, tuple[ArchiveCDXCaptureRowV2, ...]] = {
                year: tuple(row for row in canonical_rows if row.captured_at.year == year)
                for year in returned_years
            }
            flattened_attempts: list[str] = []
            fetch_by_request = {item.capture_request_id: item for item in fetches}
            any_attempted = any(item.attempted_capture_request_ids for item in selections)
            _req(not any_attempted
                 or all(item.stop_reason != "not_attempted" for item in selections),
                 "annual replay is either wholly deferred or covers every year")
            for selection in selections:
                eligible_ids = tuple(capture_request_id(row)
                                     for row in capture_by_year[selection.year])
                attempted = selection.attempted_capture_request_ids
                _req(attempted == eligible_ids[:len(attempted)]
                     and len(attempted) <= selection.max_attempts,
                     "annual attempts must be an oldest-first prefix")
                _req(all(item in fetch_by_request for item in attempted),
                     "annual attempt lacks a replay-fetch record")
                records = tuple(fetch_by_request[item] for item in attempted)
                successes = tuple(item for item in records if item.status == "success")
                if selection.stop_reason == "not_attempted":
                    _req(not attempted and selection.selected_source_evidence_id is None,
                         "invalid deferred annual selection")
                elif selection.stop_reason == "first_usable":
                    _req(bool(records) and len(successes) == 1
                         and records[-1].status == "success"
                         and selection.selected_source_evidence_id
                         == records[-1].source_evidence_id,
                         "invalid first-usable annual selection")
                elif selection.stop_reason == "exhausted":
                    _req(attempted == eligible_ids and not successes
                         and len(eligible_ids) <= selection.max_attempts
                         and selection.selected_source_evidence_id is None,
                         "invalid exhausted annual selection")
                else:
                    _req(len(attempted) == selection.max_attempts
                         and len(attempted) < len(eligible_ids) and not successes
                         and selection.selected_source_evidence_id is None,
                         "invalid max-attempt annual selection")
                flattened_attempts.extend(attempted)
            _req(tuple(flattened_attempts)
                 == tuple(item.capture_request_id for item in fetches),
                 "annual selections and replay records differ")
            expected_page = _derived_page_status(pages, fetches)
            _req(self.page_status == expected_page,
                 "aggregate page status must derive from replay facts")

            _req(len({item.evidence_id for item in sources}) == len(sources) and len({item.capture_request_id for item in sources}) == len(sources), "duplicate source snapshot")
            successful_records = tuple(item for item in fetches if item.status == "success")
            _req(len({item.source_evidence_id for item in successful_records}) == len(successful_records), "duplicate successful source fetch")
            successful = {item.source_evidence_id: item for item in successful_records}
            _req(None not in successful and set(successful) == {item.evidence_id for item in sources}, "sources require exactly one successful fetch")
            _req(all(successful[item.evidence_id].capture_request_id == item.capture_request_id for item in sources), "source/fetch request mismatch")
            _req(all((successful[item.evidence_id].retrieval_source,
                      successful[item.evidence_id].provenance_sha256)
                     == (item.retrieval_source, item.provenance_sha256)
                     for item in sources), "source/fetch retrieval provenance mismatch")
            capture_by_request = {capture_request_id(row): row for row in capture_rows}
            def exact_capture_binding(item: ArchiveSourceSnapshotV2) -> bool:
                row = capture_by_request[item.capture_request_id]
                replay_timestamp, replay_target = _parse_replay_url(item.replay_url)
                canonical_target = canonical_http_url(row.original_url)
                return (row.captured_at == item.captured_at
                        and canonical_http_url(item.original_url) == canonical_target
                        and replay_timestamp == row.captured_at.strftime("%Y%m%d%H%M%S")
                        and replay_target == canonical_target)
            _req(all(exact_capture_binding(item) for item in sources), "source must exactly bind selected CDX row")
            _req(len({item.evidence_id for item in claims}) == len(claims), "duplicate opening claim")
            source_by_id = {item.evidence_id: item for item in sources}
            _req(all(item.source_evidence_id in source_by_id
                     and item.page_text_sha256
                     == source_by_id[item.source_evidence_id].page_text_sha256
                     and opening_claim_has_assertive_context(
                         source_by_id[item.source_evidence_id].archived_page_text,
                         item.excerpt,
                     )
                     and item.year
                     <= source_by_id[item.source_evidence_id].captured_at.year
                     for item in claims), "claim not backed by saved source context")
        object.__setattr__(self, "cdx_pages", pages)
        object.__setattr__(self, "annual_selections", selections)
        object.__setattr__(self, "fetch_records", fetches)
        object.__setattr__(self, "source_snapshots", sources)
        object.__setattr__(self, "opening_claims", claims)
        object.__setattr__(self, "retrieved_at", _utc(self.retrieved_at))

    def to_private_dict(self) -> dict[str, Any]:
        return {"query_status": self.query_status, "page_status": self.page_status,
                "cap_status": self.cap_status, "query_key_digest": self.query_key_digest,
                "query_scope": self.query_scope, "retrieved_at": _iso(self.retrieved_at),
                "max_cdx_pages": self.max_cdx_pages,
                "max_cdx_records": self.max_cdx_records,
                "cdx_response_chain_sha256": self.cdx_response_chain_sha256,
                "cdx_pages": [item.to_private_dict() for item in self.cdx_pages],
                "annual_selections": [item.to_private_dict()
                                      for item in self.annual_selections],
                "fetch_records": [item.to_private_dict() for item in self.fetch_records],
                "source_snapshots": [item.to_private_dict()
                                     for item in self.source_snapshots],
                "opening_claims": [item.to_private_dict() for item in self.opening_claims]}

    def to_public_dict(self) -> dict[str, Any]:
        return {"query_status": self.query_status, "page_status": self.page_status,
                "cap_status": self.cap_status, "query_scope": self.query_scope,
                "retrieved_at": _iso(self.retrieved_at),
                "cdx_pages": [item.to_public_dict() for item in self.cdx_pages],
                "annual_selections": [item.to_public_dict()
                                      for item in self.annual_selections],
                "fetch_records": [item.to_public_dict() for item in self.fetch_records],
                "source_snapshots": [item.to_public_dict()
                                     for item in self.source_snapshots],
                "opening_claims": [item.to_public_dict() for item in self.opening_claims]}


@dataclass(frozen=True, slots=True, repr=False)
class WaybackCollectionV2:
    archive_evidence: ArchiveEvidenceV2; archive_match_assessments: tuple[ArchiveMatchAssessmentV2, ...]; archive_claim_assessments: tuple[ArchiveClaimAssessmentV2, ...]
    def __post_init__(self) -> None:
        _req(isinstance(self.archive_match_assessments, (tuple, list))
             and isinstance(self.archive_claim_assessments, (tuple, list)),
             "invalid Wayback assessment containers")
        matches = tuple(self.archive_match_assessments)
        claims = tuple(self.archive_claim_assessments)
        _req(isinstance(self.archive_evidence, ArchiveEvidenceV2) and all(isinstance(item, ArchiveMatchAssessmentV2) for item in matches) and all(isinstance(item, ArchiveClaimAssessmentV2) for item in claims), "invalid Wayback collection")
        _req({item.source_evidence_id for item in matches} == {item.evidence_id for item in self.archive_evidence.source_snapshots} and len({item.source_evidence_id for item in matches}) == len(matches), "source assessments must be exact")
        _req({item.claim_evidence_id for item in claims} == {item.evidence_id for item in self.archive_evidence.opening_claims} and len({item.claim_evidence_id for item in claims}) == len(claims), "claim assessments must be exact")
        object.__setattr__(self, "archive_match_assessments", matches); object.__setattr__(self, "archive_claim_assessments", claims)

    @property
    def outlet_opening_year_state(self) -> str:
        claim_by_id = {item.evidence_id: item
                       for item in self.archive_evidence.opening_claims}
        years = {
            claim_by_id[item.claim_evidence_id].year
            for item in self.archive_claim_assessments
            if item.evidence_scope == "outlet_specific"
        }
        return "none" if not years else ("single" if len(years) == 1 else "conflict")


_CONTRACT = {"naics_vintage": "2017", "target_naics_code": "446110", "target_title": "Pharmacies and Drug Stores", "target_definition": "One current, public-facing prescription-dispensing pharmacy outlet or pharmacy department named by the listing.", "classification_rules": ("Use authenticated evidence only.", "Classify the listing unit, not its parent grocery store, clinic, hospital, or health system.", "A pharmacy department can qualify when the listing and evidence identify that public-facing dispensing outlet.", "A Conflicting target-construct judgment requires candidate evidence plus a successful independent review or appropriately outlet-linked archive citation.", "An assessable identity judgment requires candidate evidence plus a successful independent review or appropriately outlet-linked archive citation."), "schema_version": "v2.1", "prompt_template_version": "2026-08-31"}


@dataclass(frozen=True, slots=True, repr=False)
class ClassifierContractV2:
    naics_vintage: str; target_naics_code: str; target_title: str; target_definition: str; classification_rules: tuple[str, ...]; schema_version: str; prompt_template_version: str; contract_hash: str
    @classmethod
    def from_project_constants(cls) -> "ClassifierContractV2":
        return cls(**_CONTRACT, contract_hash=evidence_id("classifier_contract_v2", _CONTRACT))
    def __post_init__(self) -> None:
        expected = (*_CONTRACT.values(), evidence_id("classifier_contract_v2", _CONTRACT))
        actual = (self.naics_vintage, self.target_naics_code, self.target_title, self.target_definition, tuple(self.classification_rules) if isinstance(self.classification_rules, (tuple, list)) else self.classification_rules, self.schema_version, self.prompt_template_version, self.contract_hash)
        _req(actual == expected, "unauthenticated classifier contract")
        object.__setattr__(self, "classification_rules", tuple(self.classification_rules))


@dataclass(frozen=True, slots=True, repr=False)
class ClassifierInputV2:
    candidate: CandidateFactsV2; review_evidence: ReviewEvidenceV2; wayback_collection: WaybackCollectionV2; contract: ClassifierContractV2
    def __post_init__(self) -> None:
        _req(isinstance(self.candidate, CandidateFactsV2)
             and isinstance(self.review_evidence, ReviewEvidenceV2)
             and isinstance(self.wayback_collection, WaybackCollectionV2)
             and isinstance(self.contract, ClassifierContractV2)
             and self.review_evidence.source_place_id
             == self.candidate.source_place_id,
             "invalid classifier input")
        if self.review_evidence.status == "success":
            _req(self.review_evidence.retrieved_at == self.candidate.retrieved_at
                 and self.review_evidence.total_user_rating_count
                 == self.candidate.user_ratings_total,
                 "candidate and reviews must come from one dated Details packet")
        else:
            _req(self.review_evidence.retrieved_at >= self.candidate.retrieved_at,
                 "review attempt predates its candidate")


def _decision_maps(confidence: Any, citations: Any, hashes: Any) -> tuple[Mapping[str, str], Mapping[str, tuple[str, ...]], Mapping[str, str]]:
    conf, cites, reason = freeze_json(confidence), freeze_json(citations), freeze_json(hashes)
    _req(isinstance(conf, Mapping) and set(conf) == CLAIMS and all(isinstance(value, str) and value in CONFIDENCE for value in conf.values()), "invalid claim confidence")
    _req(isinstance(cites, Mapping) and set(cites) == CLAIMS
         and all(isinstance(value, tuple)
                 and all(isinstance(item, str)
                         and EVIDENCE_REF.fullmatch(item) is not None
                         for item in value)
                 and len(value) == len(set(value)) for value in cites.values()),
         "invalid claim evidence IDs")
    _req(isinstance(reason, Mapping) and set(reason) == CLAIMS, "invalid reasoning hashes")
    for value in reason.values(): _hash(value, "reasoning hash")
    return conf, cites, reason


@dataclass(frozen=True, slots=True, repr=False)
class LLMDecisionV2:
    calculated_naics: str | None; target_construct_status: str; business_status: str; candidate_plausibility: str; identity_consistency: str; historical_year_value: int | None; historical_year_type: str; claim_confidence: Mapping[str, str]; claim_evidence_ids: Mapping[str, tuple[str, ...]]; reasoning_hashes: Mapping[str, str]
    def __post_init__(self) -> None:
        _req(self.calculated_naics is None or (isinstance(self.calculated_naics, str) and NAICS.fullmatch(self.calculated_naics) is not None), "invalid NAICS")
        _req(isinstance(self.target_construct_status, str) and self.target_construct_status in TARGET_CONSTRUCT, "invalid target construct status")
        _req(isinstance(self.business_status, str) and self.business_status in BUSINESS, "invalid business status")
        _req(isinstance(self.candidate_plausibility, str) and self.candidate_plausibility in PLAUSIBILITY, "invalid candidate plausibility")
        _req(isinstance(self.identity_consistency, str) and self.identity_consistency in IDENTITY, "invalid identity consistency")
        _req(isinstance(self.historical_year_type, str) and self.historical_year_type in YEAR_TYPES, "invalid historical year type")
        null_year_type = self.historical_year_type == "unknown"
        _req(null_year_type == (self.historical_year_value is None),
             "historical year/type contract failed")
        if self.historical_year_value is not None: _int(self.historical_year_value, "historical year", 1600)
        confidence, citations, hashes = _decision_maps(self.claim_confidence, self.claim_evidence_ids, self.reasoning_hashes)
        citation_required = {
            "calculated_naics": self.calculated_naics is not None,
            "target_construct_status": self.target_construct_status != "Unresolved",
            "business_status": self.business_status != "Unknown",
            "candidate_plausibility": self.candidate_plausibility
            in {"Supported", "Refuted", "Conflicting"},
            "identity_consistency": self.identity_consistency != "NotAssessable",
            "historical_year": self.historical_year_type != "unknown",
        }
        _req(all(not required or citations[claim]
                 for claim, required in citation_required.items()),
             "assessable decision claims require evidence IDs")
        has_prefix = lambda claim, prefix: any(
            item.startswith(prefix) for item in citations[claim]
        )
        if self.calculated_naics is not None:
            _req(has_prefix("calculated_naics", "candidate_"),
                 "NAICS decision requires candidate evidence")
        if self.target_construct_status != "Unresolved":
            _req(has_prefix("target_construct_status", "candidate_"),
                 "target construct decision requires candidate evidence")
        if self.target_construct_status == "Conflicting":
            _req(any(not item.startswith("candidate_")
                     for item in citations["target_construct_status"]),
                 "target construct conflict requires noncandidate evidence")
        if self.business_status != "Unknown":
            _req(has_prefix("business_status", "candidate_"),
                 "business status requires candidate evidence")
        if self.candidate_plausibility in {"Supported", "Refuted", "Conflicting"}:
            _req(has_prefix("candidate_plausibility", "candidate_"),
                 "candidate plausibility requires candidate evidence")
        if self.candidate_plausibility == "Conflicting":
            _req(any(not item.startswith("candidate_")
                     for item in citations["candidate_plausibility"]),
                 "conflict requires noncandidate evidence")
        if self.identity_consistency != "NotAssessable":
            _req(has_prefix("identity_consistency", "candidate_")
                 and any(not item.startswith("candidate_")
                         for item in citations["identity_consistency"]),
                 "assessable identity requires candidate and independent evidence")
        if self.historical_year_type == "active_by_upper_bound":
            _req(has_prefix("historical_year", "archive_source_v2_"),
                 "active-by year requires archive-source evidence")
        elif self.historical_year_type == "explicit_archived_opening_claim":
            _req(has_prefix("historical_year", "opening_claim_v2_"),
                 "explicit year requires an opening claim")
        object.__setattr__(self, "claim_confidence", confidence); object.__setattr__(self, "claim_evidence_ids", citations); object.__setattr__(self, "reasoning_hashes", hashes)


def validate_llm_decision(packet: ClassifierInputV2, raw: Mapping[str, Any]) -> LLMDecisionV2:
    _req(isinstance(packet, ClassifierInputV2) and type(raw) is dict, "invalid decision packet")
    fields = {"calculated_naics", "target_construct_status", "business_status", "candidate_plausibility", "identity_consistency", "historical_year_value", "historical_year_type", "claim_confidence", "claim_evidence_ids", "reasoning"}
    _req(set(raw) == fields, "undeclared decision fields")
    naics, target_construct, business, plausibility, identity, year, year_type = raw["calculated_naics"], raw["target_construct_status"], raw["business_status"], raw["candidate_plausibility"], raw["identity_consistency"], raw["historical_year_value"], raw["historical_year_type"]
    _req(naics is None or (isinstance(naics, str) and NAICS.fullmatch(naics) is not None), "invalid NAICS")
    _req(isinstance(target_construct, str) and target_construct in TARGET_CONSTRUCT, "invalid target construct status"); _req(isinstance(business, str) and business in BUSINESS, "invalid business status"); _req(isinstance(plausibility, str) and plausibility in PLAUSIBILITY, "invalid candidate plausibility"); _req(isinstance(identity, str) and identity in IDENTITY, "invalid identity consistency"); _req(isinstance(year_type, str) and year_type in YEAR_TYPES, "invalid historical year type")
    null_year_type = year_type == "unknown"
    _req(null_year_type == (year is None), "historical year/type contract failed")
    if year is not None: _int(year, "historical year", 1600)
    _req(type(raw["claim_confidence"]) is dict and type(raw["claim_evidence_ids"]) is dict and type(raw["reasoning"]) is dict, "malformed decision maps")
    _req(all(isinstance(value, str) and 0 < len(value) <= 500 for value in raw["reasoning"].values()), "invalid reasoning")
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in raw["reasoning"].items()}
    decision = LLMDecisionV2(naics, target_construct, business, plausibility, identity, year, year_type, raw["claim_confidence"], raw["claim_evidence_ids"], hashes)
    archive = packet.wayback_collection.archive_evidence
    sources = {item.evidence_id: item for item in archive.source_snapshots}; claims = {item.evidence_id: item for item in archive.opening_claims}
    known = {packet.candidate.candidate_id} | {item.evidence_id for item in packet.review_evidence.snippets} | set(sources) | set(claims)
    _req(all(set(ids).issubset(known) for ids in decision.claim_evidence_ids.values()), "unknown evidence ID")
    candidate_required = (("calculated_naics", naics is not None), ("target_construct_status", target_construct != "Unresolved"), ("business_status", business in {"Active", "Inactive"}), ("candidate_plausibility", plausibility in {"Supported", "Refuted", "Conflicting"}))
    for claim, required in candidate_required:
        if required: _req(packet.candidate.candidate_id in decision.claim_evidence_ids[claim], f"{claim} requires candidate evidence")
    matches = {item.source_evidence_id: item for item in packet.wayback_collection.archive_match_assessments}
    assessed_claims = {item.claim_evidence_id: item for item in packet.wayback_collection.archive_claim_assessments}
    successful_reviews = ({item.evidence_id for item in packet.review_evidence.snippets}
                          if packet.review_evidence.status == "success" else set())
    outlet_sources = {
        item_id for item_id, assessment in matches.items()
        if assessment.evidence_scope in {"probable_outlet", "outlet_specific"}
    }
    outlet_claims = {
        item_id for item_id, assessment in assessed_claims.items()
        if assessment.evidence_scope in {"probable_outlet", "outlet_specific"}
    }
    independent_ids = successful_reviews | outlet_sources | outlet_claims
    if target_construct == "Conflicting":
        noncandidate = set(decision.claim_evidence_ids["target_construct_status"]) - {packet.candidate.candidate_id}
        _req(bool(noncandidate & independent_ids),
             "target construct conflict requires successful independent evidence")
    if plausibility == "Conflicting":
        noncandidate = set(decision.claim_evidence_ids["candidate_plausibility"]) - {packet.candidate.candidate_id}
        _req(any(item in sources or item in claims or item in successful_reviews for item in noncandidate), "conflict requires successful non-candidate evidence")
    if identity != "NotAssessable":
        independent = (set(decision.claim_evidence_ids["identity_consistency"])
                       - {packet.candidate.candidate_id})
        _req(bool(independent & independent_ids),
             "assessable identity requires successful independent evidence")
    outlet_claim_ids = {
        item_id for item_id, assessment in assessed_claims.items()
        if assessment.evidence_scope == "outlet_specific"
    }
    outlet_years = {claims[item_id].year for item_id in outlet_claim_ids}
    if packet.wayback_collection.outlet_opening_year_state == "conflict":
        _req(year_type == "unknown",
             "conflicting outlet opening years must map to public unknown")
    if year_type == "active_by_upper_bound":
        eligible = [item for item in sources.values() if matches[item.evidence_id].evidence_scope == "outlet_specific"]
        _req(eligible, "active-by requires outlet-specific successful source")
        earliest = min(item.captured_at.year for item in eligible)
        _req(year == earliest and any(item in sources and matches[item].evidence_scope == "outlet_specific" and sources[item].captured_at.year == earliest for item in decision.claim_evidence_ids["historical_year"]), "invalid active-by evidence")
    elif year_type == "explicit_archived_opening_claim":
        _req(len(outlet_years) == 1 and year in outlet_years
             and any(item in claims
                     and assessed_claims[item].evidence_scope == "outlet_specific"
                     and claims[item].year == year
                     for item in decision.claim_evidence_ids["historical_year"]),
             "explicit year requires one consistent outlet-specific claim year")
    return decision
