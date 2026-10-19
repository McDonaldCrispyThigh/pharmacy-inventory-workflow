"""Exact-snapshot page transport with independently authenticated discovery.

No discovery request can be sent to Outscraper through this adapter. Direct HTML
is parsed locally; outsourced text is labelled derived page text. The durable
provider client remains unchanged and owns paid-job admission, polling and resume.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

from archive_transport_binding import snapshot_identity, urls_bind
from outscraper_wayback import OutscraperFallback, PROMPT_VERSION


VERSION = "nets-ai-exact-snapshot-page-v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_METHODS = {"cdx", "timemap", "availability"}
_MAX_BODY = 8_000_000
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
_SKIP = {"script", "style", "noscript", "template", "svg", "canvas", "head"}


class ArchivePageRejected(ValueError):
    """Terminal page-validation outcome; never interpret as archive absence."""

    def __init__(self, reason):
        if not isinstance(reason, str) or re.fullmatch(r"[a-z0-9_]{1,100}", reason) is None:
            raise ValueError("invalid page rejection reason")
        self.reason = reason
        super().__init__("Archived page rejected: " + reason)


def _value(response, key, default=None):
    return response.get(key, default) if isinstance(response, Mapping) else getattr(response, key, default)


def _identity(url):
    try:
        return snapshot_identity(url)
    except (ValueError, TypeError, KeyError) as exc:
        raise ArchivePageRejected("invalid_exact_snapshot_url") from exc


def _same_capture(left, right):
    return left["timestamp"] == right["timestamp"] and urls_bind(left["original_url"], right["original_url"])


def _snapshot(candidate_url, snapshot):
    if not isinstance(snapshot, Mapping):
        raise ArchivePageRejected("missing_official_discovery_proof")
    if snapshot.get("method") not in _METHODS:
        raise ArchivePageRejected("unofficial_discovery_method")
    digest = snapshot.get("provenance_sha256", snapshot.get("official_discovery_provenance_sha256"))
    if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
        raise ArchivePageRejected("missing_official_discovery_provenance")
    alias = snapshot.get("official_discovery_provenance_sha256", digest)
    if alias != digest:
        raise ArchivePageRejected("conflicting_official_discovery_provenance")
    identity = _identity(snapshot.get("exact_url"))
    if (identity["timestamp"] != snapshot.get("timestamp")
            or not urls_bind(identity["original_url"], snapshot.get("original_url"))
            or not urls_bind(candidate_url, identity["original_url"])):
        raise ArchivePageRejected("snapshot_candidate_binding_failed")
    return {**identity, "method": snapshot["method"], "provenance_sha256": digest}


class _VisibleText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.redirect = False
        self.challenge = False

    def handle_starttag(self, tag, attrs):
        attrs = {key.lower(): value or "" for key, value in attrs}
        identity = attrs.get("id", "").lower()
        style = re.sub(r"\s+", "", attrs.get("style", "").lower())
        hidden = (tag in _SKIP or any(item[1] for item in self.stack)
                  or "hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true"
                  or "display:none" in style or "visibility:hidden" in style
                  or identity.startswith(("wm-ipp", "wayback-toolbar")))
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            self.redirect = True
        if identity.startswith(("cf-chl-", "cf-browser-verification", "challenge-running")):
            self.challenge = True
        if not hidden and tag in {"br", "p", "div", "li", "section", "article", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")
        if tag not in _VOID:
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in {"p", "div", "li", "section", "article", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not any(item[1] for item in self.stack):
            self.parts.append(data)


def _clean_text(text):
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _usable_text(text):
    if not text or not any(char.isalnum() for char in text):
        raise ArchivePageRejected("empty_archived_page_text")
    beginning = re.sub(r"\s+", " ", text).strip().lower()[:1500]
    failures = (
        r"^(?:error\s*)?(?:400 bad request|401 unauthorized|403 forbidden|404 not found|429 too many requests|500 internal server error|502 bad gateway|503 service unavailable|504 gateway timeout)\b",
        r"^(?:access denied|request blocked|page not found|site not found|this site can.t be reached|this page (?:isn.t|is not) available)\b",
        r"(?:wayback machine (?:has not|hasn.t|does not|doesn.t) (?:archived|have)|this url has been excluded from the wayback machine|cannot be crawled or displayed due to robots\.txt)",
        r"^(?:i (?:cannot|can.t|could not|was unable to)|unable to|failed to) (?:access|retrieve|fetch|read|load)\b",
        r"^(?:just a moment|checking your browser|verify (?:that )?you are human|please complete the security check)\b",
        r"^(?:the )?(?:requested )?(?:url|page|snapshot|content) (?:could not be|cannot be|was not) (?:retrieved|accessed|loaded|fetched)\b",
    )
    if any(re.search(pattern, beginning, re.I) for pattern in failures):
        raise ArchivePageRejected("archive_error_or_interstitial")
    return text


def _decode(body, content_type):
    if not isinstance(body, bytes) or not body or len(body) > _MAX_BODY:
        raise ArchivePageRejected("invalid_archived_response_size")
    charset = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9._-]+)", content_type, re.I)
    encoding = charset[1] if charset else "utf-8-sig"
    try:
        text = body.decode(encoding)
    except LookupError as exc:
        raise ArchivePageRejected("unknown_archived_text_encoding") from exc
    except UnicodeDecodeError as exc:
        if charset:
            raise ArchivePageRejected("invalid_archived_text_encoding") from exc
        try:
            text, encoding = body.decode("windows-1252"), "windows-1252"
        except UnicodeDecodeError as second:
            raise ArchivePageRejected("invalid_archived_text_encoding") from second
    if "\x00" in text:
        raise ArchivePageRejected("binary_archived_response")
    return text, encoding


def _text_payload(text, max_text_chars):
    if type(max_text_chars) is not int or not 1 <= max_text_chars <= 100_000:
        raise ValueError("max_text_chars must be an integer in [1,100000]")
    text = _usable_text(_clean_text(text))
    retained = text[:max_text_chars]
    return {
        "text": retained,
        "text_sha256": hashlib.sha256(retained.encode("utf-8")).hexdigest(),
        "full_visible_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "visible_text_characters": len(text), "text_truncated": len(text) > len(retained),
    }


def _result(identity, *, source_type, body, provenance_sha256, text_payload, retrieved_at=None):
    if not isinstance(provenance_sha256, str) or _SHA.fullmatch(provenance_sha256) is None:
        raise ArchivePageRejected("missing_page_provenance")
    if isinstance(retrieved_at, datetime):
        retrieved_at = retrieved_at.isoformat().replace("+00:00", "Z")
    return {
        "schema_version": VERSION, "status": "success", "source_type": source_type,
        "source_fidelity": "original_http" if source_type == "direct_raw" else "derived_page_text",
        "timestamp": identity["timestamp"], "original_url": identity["original_url"], "exact_url": identity["exact_url"],
        "official_discovery_method": identity["method"],
        "official_discovery_provenance_sha256": identity["provenance_sha256"],
        "provenance_sha256": provenance_sha256, "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_bytes": len(body), "retrieved_at": retrieved_at, **text_payload,
        "lossless_archive_response": source_type == "direct_raw",
        "derived_text_alone_can_establish_exact_opening_year": False,
    }


def validate_direct_page(response, candidate_url, snapshot, max_text_chars=5000):
    """Validate one already-recorded exact HTTP replay; never acquire a page.

    The runner supplies a parsed official discovery proof and the direct raw
    envelope digest. This pure function validates identity, HTTP/media boundaries
    and visible content; it does not read or mutate the provenance store.
    """
    identity = _snapshot(candidate_url, snapshot)
    if _value(response, "status") != 200:
        raise ArchivePageRejected("direct_replay_http_failure")
    if not _same_capture(identity, _identity(_value(response, "final_url"))):
        raise ArchivePageRejected("direct_replay_capture_changed")
    headers = _value(response, "headers", {})
    if not isinstance(headers, Mapping):
        raise ArchivePageRejected("invalid_direct_replay_headers")
    headers = {str(key).lower(): str(value) for key, value in headers.items()}
    if _value(response, "body_truncated", False) is True or _value(response, "oversized", False) is True:
        raise ArchivePageRejected("truncated_archived_response")
    if "memento-datetime" in headers:
        try:
            dated = parsedate_to_datetime(headers["memento-datetime"])
            if dated.tzinfo is None or dated.utcoffset().total_seconds() != 0 or dated.strftime("%Y%m%d%H%M%S") != identity["timestamp"]:
                raise ValueError("capture datetime")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ArchivePageRejected("direct_replay_memento_datetime_changed") from exc
    content_type = headers.get("content-type", "")
    media = content_type.split(";", 1)[0].strip().lower()
    if media not in {"text/html", "application/xhtml+xml", "text/plain"}:
        raise ArchivePageRejected("unsupported_archived_media_type")
    body = _value(response, "body")
    text, encoding = _decode(body, content_type)
    if media != "text/plain":
        parser = _VisibleText()
        try:
            parser.feed(text)
            parser.close()
        except (ValueError, AssertionError) as exc:
            raise ArchivePageRejected("archived_html_parser_failure") from exc
        if parser.redirect:
            raise ArchivePageRejected("archived_html_redirect")
        if parser.challenge:
            raise ArchivePageRejected("archive_error_or_interstitial")
        text = "".join(parser.parts)
    payload = _text_payload(text, max_text_chars)
    expected_body_hash = _value(response, "body_sha256", _value(response, "response_sha256"))
    if expected_body_hash is not None and expected_body_hash != hashlib.sha256(body).hexdigest():
        raise ArchivePageRejected("direct_replay_body_hash_mismatch")
    result = _result(identity, source_type="direct_raw", body=body,
                     provenance_sha256=_value(response, "provenance_sha256"), text_payload=payload,
                     retrieved_at=_value(response, "retrieved_at"))
    return {**result, "http_status": 200, "media_type": media, "decoded_encoding": encoding,
            "final_url": _value(response, "final_url"), "discovery_proof_validation": "parsed_by_caller"}


def _official_endpoint(url, method, candidate_url, requested_timestamp=None):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (ValueError, TypeError):
        return False
    if (parsed.scheme not in {"http", "https"} or parsed.username is not None or parsed.password is not None
            or port not in {None, 80, 443} or parsed.fragment):
        return False
    query = parse_qs(parsed.query, keep_blank_values=True)
    if method == "cdx":
        values = query.get("url", [])
        return parsed.hostname == "web.archive.org" and parsed.path == "/cdx/search/cdx" and len(values) == 1 and urls_bind(candidate_url, values[0])
    if method == "availability":
        values, times = query.get("url", []), query.get("timestamp", [])
        return (parsed.hostname == "archive.org" and parsed.path == "/wayback/available" and len(values) == 1
                and urls_bind(candidate_url, values[0]) and times == ([] if requested_timestamp is None else [str(requested_timestamp)]))
    match = re.fullmatch(r"/(?:web/)?timemap/(?:link/)?(?:[0-9]+/)?(https?://.+)", parsed.path)
    original = (match[1] + (("?" + parsed.query) if parsed.query else "")) if match else None
    return parsed.hostname == "web.archive.org" and match is not None and urls_bind(candidate_url, original)


def _official_timemap_transport_key(url):
    """Normalize only the outer Archive authority, never the embedded site URL."""
    try:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or parsed.hostname != "web.archive.org"
                or parsed.username is not None or parsed.password is not None
                or parsed.port not in {None, 80 if parsed.scheme == "http" else 443}
                or parsed.fragment):
            return None
        match = re.fullmatch(r"(?i:https?)://[^/?#]+(.*)", url)
        return "https://web.archive.org" + match[1] if match else None
    except (ValueError, TypeError):
        return None


def verify_official_snapshot(store, candidate_url, snapshot):
    """Reparse the content-addressed official receipt before any paid request."""
    identity = _snapshot(candidate_url, snapshot)
    try:
        record = store.read_record(identity["provenance_sha256"])
    except (ValueError, OSError, RuntimeError) as exc:
        raise ArchivePageRejected("official_discovery_provenance_unreadable") from exc
    requested = snapshot.get("requested_timestamp")
    if (record.get("provider") != "direct_wayback" or record.get("response_status") != 200
            or not _official_endpoint(record.get("request_url", ""), identity["method"], candidate_url, requested)
            or record.get("final_url") != record.get("request_url")
            or record.get("metadata", {}).get("body_truncated") is True
            or record.get("metadata", {}).get("oversized") is True):
        raise ArchivePageRejected("official_discovery_endpoint_unverified")
    try:
        body = base64.b64decode(record["body_base64"], validate=True)
        if len(body) != record["response_bytes"] or hashlib.sha256(body).hexdigest() != record["response_sha256"]:
            raise ValueError("body commitment")
    except (ValueError, TypeError, KeyError) as exc:
        raise ArchivePageRejected("official_discovery_body_unverified") from exc
    from archive_official_discovery import parse_availability, parse_cdx, parse_timemap
    if identity["method"] == "cdx":
        parsed = parse_cdx(body, candidate_url)
    elif identity["method"] == "timemap":
        parsed = parse_timemap(body, candidate_url)
    else:
        parsed = parse_availability(body, candidate_url, requested)
    if identity["method"] == "timemap":
        self_key = _official_timemap_transport_key(parsed.get("self_url"))
        request_key = _official_timemap_transport_key(record["request_url"])
        if self_key is None or self_key != request_key:
            raise ArchivePageRejected("official_timemap_self_url_mismatch")
    if parsed.get("status") != "success" or not any(_same_capture(identity, _identity(row.get("exact_url"))) for row in parsed.get("snapshots", [])):
        raise ArchivePageRejected("snapshot_not_in_official_discovery_response")
    return identity


def _extraction_record(store, response, identity, scope):
    digest = _value(response, "provenance_sha256")
    if not isinstance(digest, str) or _SHA.fullmatch(digest) is None:
        raise ArchivePageRejected("outscraper_missing_or_unavailable")
    try:
        record = store.read_record(digest)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArchivePageRejected("outscraper_provenance_unreadable") from exc
    metadata = record.get("metadata", {})
    request = urlsplit(record.get("request_url", ""))
    if (record.get("provider") != "outscraper_ai" or record.get("response_status") not in {200, 202}
            or record.get("request_url") != record.get("final_url")
            or request.scheme != "https" or request.netloc != "api.outscraper.cloud"
            or not (request.path == "/ai-scraper" or re.fullmatch(r"/requests/[A-Za-z0-9_-]{1,128}", request.path))
            or metadata.get("kind") != "replay" or metadata.get("job_scope") != scope
            or metadata.get("target_url") != identity["exact_url"]
            or metadata.get("prompt_version") != PROMPT_VERSION
            or metadata.get("source_fidelity") != "structural_only"
            or metadata.get("body_truncated") is not False or metadata.get("oversized") is not False):
        raise ArchivePageRejected("outscraper_response_binding_failed")
    try:
        raw = base64.b64decode(record["body_base64"], validate=True)
        if len(raw) != record["response_bytes"] or hashlib.sha256(raw).hexdigest() != record["response_sha256"]:
            raise ValueError("body commitment")
        payload = json.loads(raw)
    except (ValueError, TypeError, KeyError) as exc:
        raise ArchivePageRejected("outscraper_invalid_provider_envelope") from exc
    if not isinstance(payload, dict) or payload.get("status") != "Success" or (payload.get("id") is not None and payload["id"] != metadata.get("request_id")):
        raise ArchivePageRejected("outscraper_unsuccessful_or_mismatched_job")
    data = payload.get("data")
    for _ in range(2):
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
    if not isinstance(data, dict) or "retrieval_error" not in data or data["retrieval_error"] is not None:
        raise ArchivePageRejected("outscraper_reported_retrieval_failure")
    if not _same_capture(identity, _identity(data.get("target_url"))) or not _same_capture(identity, _identity(data.get("final_url"))):
        raise ArchivePageRejected("outscraper_extracted_capture_changed")
    if type(data.get("archive_timestamp")) not in {int, str} or str(data["archive_timestamp"]) != identity["timestamp"]:
        raise ArchivePageRejected("outscraper_extracted_timestamp_changed")
    if not isinstance(data.get("page_text"), str):
        raise ArchivePageRejected("outscraper_missing_extracted_text")
    return data, record, digest


def _derived_result(store, response, identity, scope, max_text_chars):
    data, record, digest = _extraction_record(store, response, identity, scope)
    body = data["page_text"].encode("utf-8")
    if len(body) > _MAX_BODY:
        raise ArchivePageRejected("invalid_extracted_response_size")
    text = _text_payload(data["page_text"], max_text_chars)
    result = _result(identity, source_type="outscraper_extracted", body=body,
                     provenance_sha256=digest, text_payload=text,
                     retrieved_at=record.get("retrieved_at"))
    # Recover this diagnostic from the frozen provider validation, not a mutable
    # retained page field. A local binding repair may accept the same receipt.
    adapter_status = 502 if record["metadata"].get("validation", {}).get("status") == "rejected" else 200
    return {**result, "provider_http_status": record["response_status"],
            "provider_response_sha256": record["response_sha256"],
            "legacy_provider_source_fidelity": record["metadata"]["source_fidelity"],
            "provider_adapter_status": adapter_status,
            "reported_target_url": data["target_url"], "reported_final_url": data["final_url"],
            "discovery_proof_validation": "official_raw_response_reparsed",
            "independently_verified_archived_statement": False}


def validate_retained_source(store, candidate_url, page, *, scope, max_text_chars=5000):
    """Rebuild retained evidence from immutable receipts without provider calls.

    Only accepted page fields are compared; runner metadata is kept untouched.
    Recomputing a checkpoint hash cannot legitimize altered source text, identity,
    provenance, fidelity or an invented claim about independent text verification.
    """
    if not isinstance(page, Mapping) or page.get("status") != "success":
        raise ArchivePageRejected("retained_source_is_not_successful")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("retained source scope must be explicit")
    proof = page.get("official_snapshot")
    identity = verify_official_snapshot(store, candidate_url, proof)
    source_type = page.get("source_type")
    digest = page.get("provenance_sha256")
    if source_type == "direct_raw":
        try:
            record = store.read_record(digest)
            body = base64.b64decode(record["body_base64"], validate=True)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            raise ArchivePageRejected("retained_direct_provenance_unreadable") from exc
        metadata = record.get("metadata", {})
        if (record.get("provider") != "direct_wayback" or metadata.get("scope") != scope
                or not _same_capture(identity, _identity(record.get("request_url")))):
            raise ArchivePageRejected("retained_direct_request_binding_failed")
        response = {"status": record["response_status"], "body": body,
                    "headers": record.get("headers", {}), "final_url": record["final_url"],
                    "provenance_sha256": digest, "retrieved_at": record.get("retrieved_at"),
                    "body_sha256": record["response_sha256"],
                    "body_truncated": metadata.get("body_truncated", False),
                    "oversized": metadata.get("oversized", False)}
        regenerated = validate_direct_page(response, candidate_url, proof, max_text_chars)
        expected_route = "direct"
    elif source_type == "outscraper_extracted":
        regenerated = _derived_result(store, {"provenance_sha256": digest}, identity, scope, max_text_chars)
        expected_route = "outscraper"
    else:
        raise ArchivePageRejected("retained_source_type_invalid")
    if "route" in page and page["route"] != expected_route:
        raise ArchivePageRejected("retained_source_route_mismatch")
    try:
        for key, value in regenerated.items():
            if key not in page or json.dumps(page[key], sort_keys=True, ensure_ascii=False, allow_nan=False) != json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False):
                raise ArchivePageRejected("retained_source_fields_changed")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ArchivePageRejected("retained_source_fields_changed") from exc
    return None


class ExactSnapshotOutscraper:
    def __init__(self, store, key, scope, *, transport=None, clock=None,
                 max_text_chars=5000, **provider_options):
        self.store, self.scope, self.max_text_chars = store, scope, max_text_chars
        self.provider = OutscraperFallback(store, key, scope=scope, transport=transport,
                                           clock=clock, **provider_options)

    def fetch(self, candidate_url, snapshot, *, kind="replay"):
        if kind != "replay":
            raise ArchivePageRejected("outscraper_discovery_forbidden")
        identity = verify_official_snapshot(self.store, candidate_url, snapshot)
        # WaybackDeferred intentionally propagates without catching or relabelling.
        response = self.provider.fetch("replay", identity["exact_url"], ())
        return _derived_result(self.store, response, identity, self.scope, self.max_text_chars)


__all__ = ["VERSION", "ArchivePageRejected", "validate_direct_page", "verify_official_snapshot", "validate_retained_source", "ExactSnapshotOutscraper"]
