"""Pure discovery parsers for retained official Internet Archive responses.

Callers must first authenticate the HTTP endpoint, status, final URL and retained
body. A CDX table does not echo the queried URL, so even a header-only table needs
that transport binding before it can support an empty-result observation.

Formats: Internet Archive's wayback-cdx-server README (JSON/resumption keys),
https://archive.org/help/wayback_api.php (closest-capture Availability), and
https://www.rfc-editor.org/rfc/rfc7089.html sections 5 and 5.1 (link TimeMaps).
No parser performs requests, reads credentials or trusts Outscraper output.
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime
import json
import re
from urllib.parse import urlsplit

from archive_transport_binding import snapshot_identity, urls_bind

MAX_BODY_BYTES = 16_000_000
MAX_ENTRIES = 100_000
MAX_CURSOR_CHARS = 16_384
CDX_FIELDS = ("timestamp", "original", "mimetype", "statuscode", "digest", "length")
_REQUIRED_CDX = {"timestamp", "original", "mimetype", "statuscode"}
_TOKEN = r"[!#$%&'*+.^_`|~0-9A-Za-z-]+"


class _ParseFailure(ValueError):
    pass


class _BindingFailure(ValueError):
    pass


def _result(status, reason, *, snapshots=None, complete=False, next_cursor=None, **extra):
    return {"status": status, "snapshots": snapshots or [], "complete": complete,
            "next_cursor": next_cursor, "reason": reason, **extra}


def _text(body):
    if not isinstance(body, bytes) or not body or len(body) > MAX_BODY_BYTES:
        raise _ParseFailure("empty_invalid_or_oversized_response_body")
    try:
        return body.decode("utf-8")
    except UnicodeError as exc:
        raise _ParseFailure("response_is_not_utf8") from exc


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise _ParseFailure("duplicate_json_object_key")
        result[name] = value
    return result


def _reject_constant(value):
    raise _ParseFailure("nonfinite_json_number")


def _json(body):
    try:
        return json.loads(_text(body), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except _ParseFailure:
        raise
    except (ValueError, RecursionError) as exc:
        raise _ParseFailure("invalid_json_response") from exc


def _candidate(candidate_url):
    if not isinstance(candidate_url, str) or not urls_bind(candidate_url, candidate_url):
        raise _BindingFailure("invalid_candidate_url")


def _timestamp(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9]{14}", value) is None:
        raise _ParseFailure("invalid_capture_timestamp_type_or_width")
    try:
        datetime.strptime(value, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise _ParseFailure("invalid_capture_timestamp_calendar") from exc
    return value


def _http_status(value):
    if type(value) not in (str, int) or re.fullmatch(r"[1-5][0-9]{2}", str(value)) is None:
        raise _ParseFailure("invalid_archived_http_status")
    return int(value)


def _snapshot(url, candidate_url, *, method):
    try:
        identity = snapshot_identity(url)
    except (ValueError, TypeError) as exc:
        raise _BindingFailure("invalid_or_nonofficial_snapshot_url") from exc
    if not urls_bind(candidate_url, identity["original_url"]):
        raise _BindingFailure("snapshot_original_does_not_bind_candidate")
    return {**identity, "method": method}


def parse_cdx(body: bytes, candidate_url: str) -> dict:
    """Read official CDX JSON, retaining rows and the exact opaque resume key.

    Four identity/filter columns are required; digest and length are optional
    metadata. Missing digests (including '-') never invalidate official capture
    identity. All present row cells remain strict strings. Only archived HTTP 200
    text/html rows become replay candidates; every row must bind the candidate.
    """
    try:
        _candidate(candidate_url)
        payload = _json(body)
        if (not isinstance(payload, list) or not payload or not isinstance(payload[0], list)
                or not payload[0] or not all(isinstance(value, str) for value in payload[0])):
            raise _ParseFailure("invalid_cdx_table")
        header = payload[0]
        if (len(header) != len(set(header)) or not _REQUIRED_CDX.issubset(header)
                or not set(header).issubset(CDX_FIELDS)):
            raise _ParseFailure("invalid_cdx_header")
        rows = payload[1:]
        if len(rows) > MAX_ENTRIES + 2:
            raise _ParseFailure("cdx_row_bound_exceeded")
        cursor = None
        if len(rows) >= 2 and rows[-2] == []:
            marker = rows[-1]
            if (not isinstance(marker, list) or len(marker) != 1
                    or not isinstance(marker[0], str) or not marker[0]
                    or len(marker[0]) > MAX_CURSOR_CHARS
                    or any(ord(char) < 32 or ord(char) == 127 for char in marker[0])):
                raise _ParseFailure("invalid_cdx_resume_key")
            cursor, rows = marker[0], rows[:-2]
        if any(row == [] or (isinstance(row, list) and len(row) == 1) for row in rows):
            raise _ParseFailure("invalid_cdx_resume_trailer")
        if len(rows) > MAX_ENTRIES:
            raise _ParseFailure("cdx_row_bound_exceeded")
        snapshots, skipped = [], 0
        for row_number, raw_row in enumerate(rows, 1):
            if (not isinstance(raw_row, list) or len(raw_row) != len(header)
                    or not all(isinstance(value, str) for value in raw_row)):
                raise _ParseFailure("invalid_cdx_row_shape_or_type")
            row = dict(zip(header, raw_row))
            timestamp = _timestamp(row["timestamp"])
            status = _http_status(row["statuscode"])
            original = row["original"]
            if not urls_bind(candidate_url, original):
                raise _BindingFailure("cdx_original_does_not_bind_candidate")
            length = row.get("length")
            if length is not None and length != "-" and re.fullmatch(r"[0-9]+", length) is None:
                raise _ParseFailure("invalid_cdx_length")
            if length is not None and len(length) > 20:
                raise _ParseFailure("invalid_cdx_length")
            if not row["mimetype"] or any(ord(char) < 32 for char in row["mimetype"]):
                raise _ParseFailure("invalid_cdx_mimetype")
            if status != 200 or row["mimetype"].lower() != "text/html":
                skipped += 1
                continue
            snapshot = _snapshot(f"https://web.archive.org/web/{timestamp}id_/{original}",
                                 candidate_url, method="cdx")
            snapshot.update(statuscode=status, mimetype=row["mimetype"],
                            digest=row.get("digest"),
                            length=int(length) if length not in (None, "-") else None,
                            cdx_row=row, row_number=row_number)
            snapshots.append(snapshot)
        complete = cursor is None
        status = "success" if snapshots or not complete else "empty"
        reason = ("cdx_page_requires_continuation" if not complete else
                  "cdx_capture_candidates" if snapshots else "completed_cdx_no_eligible_captures")
        return _result(status, reason, snapshots=snapshots, complete=complete, next_cursor=cursor,
                       returned_rows=len(rows), skipped_non_html_or_non_200_rows=skipped,
                       header=list(header), transport_binding_required=True)
    except _BindingFailure as exc:
        return _result("binding_failure", str(exc))
    except _ParseFailure as exc:
        return _result("parser_failure", str(exc))


def _split_links(text):
    """Split commas outside quoted attributes and angle-bracket URI targets."""
    parts, start, quoted, angled, escaped = [], 0, False, False, False
    for index, char in enumerate(text):
        if ord(char) < 32 and char not in "\r\n\t":
            raise _ParseFailure("invalid_timemap_control_character")
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif angled:
            if char == ">":
                angled = False
            elif char == "<":
                raise _ParseFailure("invalid_timemap_link_target")
        elif char == '"':
            quoted = True
        elif char == "<":
            angled = True
        elif char == ">":
            raise _ParseFailure("invalid_timemap_link_target")
        elif char == ",":
            parts.append(text[start:index].strip())
            start = index + 1
            if len(parts) > MAX_ENTRIES:
                raise _ParseFailure("timemap_link_bound_exceeded")
    if quoted or angled or escaped:
        raise _ParseFailure("unterminated_timemap_link")
    parts.append(text[start:].strip())
    if not all(parts) or len(parts) > MAX_ENTRIES:
        raise _ParseFailure("empty_or_excess_timemap_link")
    return parts


def _link(text):
    match = re.match(r"^<([^<>\s]+)>", text)
    if match is None:
        raise _ParseFailure("invalid_timemap_link_target")
    target, position, attributes = match[1], match.end(), {}
    while position < len(text):
        parameter = re.match(r"\s*;\s*(" + _TOKEN + r")\s*", text[position:])
        if parameter is None:
            raise _ParseFailure("invalid_timemap_link_parameter")
        name = parameter[1].lower()
        position += parameter.end()
        if name in attributes:
            raise _ParseFailure("duplicate_timemap_link_parameter")
        if position >= len(text) or text[position] != "=":
            attributes[name] = None
            continue
        position += 1
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text) and text[position] == '"':
            position += 1
            chars = []
            while position < len(text) and text[position] != '"':
                if text[position] == "\\":
                    position += 1
                    if position >= len(text):
                        raise _ParseFailure("unterminated_timemap_quoted_parameter")
                chars.append(text[position])
                position += 1
            if position >= len(text):
                raise _ParseFailure("unterminated_timemap_quoted_parameter")
            position += 1
            attributes[name] = "".join(chars)
        else:
            token = re.match(_TOKEN, text[position:])
            if token is None:
                raise _ParseFailure("invalid_timemap_unquoted_parameter")
            attributes[name] = token[0]
            position += token.end()
        if not text[position:].strip():
            position = len(text)
    relation = attributes.get("rel")
    if not isinstance(relation, str) or not relation.split():
        raise _ParseFailure("timemap_link_missing_relation")
    return {"target": target, "attributes": attributes, "relations": relation.lower().split()}


def _timemap_identity(url, candidate_url):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise _BindingFailure("invalid_timemap_page_url") from exc
    if (parsed.scheme not in {"http", "https"} or parsed.hostname != "web.archive.org"
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or port not in (None, 80 if parsed.scheme == "http" else 443)):
        raise _BindingFailure("nonofficial_timemap_page_url")
    match = re.fullmatch(r"/(?:web/)?timemap/(?:link/)?(?:[0-9]+/)?(https?://.+)", parsed.path)
    if match is None:
        raise _BindingFailure("invalid_timemap_page_endpoint")
    original = match[1] + ("?" + parsed.query if parsed.query else "")
    if not urls_bind(candidate_url, original):
        raise _BindingFailure("timemap_page_original_does_not_bind_candidate")
    return url


def _memento_datetime(value):
    if not isinstance(value, str):
        raise _ParseFailure("memento_datetime_missing")
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("not GMT")
        if format_datetime(parsed.astimezone(timezone.utc), usegmt=True) != value:
            raise ValueError("not canonical RFC1123")
    except (ValueError, TypeError, OverflowError) as exc:
        raise _ParseFailure("invalid_memento_datetime") from exc
    return f"{parsed.year:04d}{parsed.month:02d}{parsed.day:02d}{parsed.hour:02d}{parsed.minute:02d}{parsed.second:02d}"


def parse_timemap(body: bytes, candidate_url: str) -> dict:
    """Parse official link TimeMaps without splitting quoted date commas.

    A bound rel=original and rel=self are mandatory. rel=next and index/paging
    rel=timemap links are retained as continuations; their absence is necessary
    for complete=True. The caller must track visited pages, follow every returned
    next_cursors entry and enforce its own finite request/page budget.
    """
    try:
        _candidate(candidate_url)
        links = [_link(part) for part in _split_links(_text(body))]
        originals = [link for link in links if "original" in link["relations"]]
        selves = [link for link in links if "self" in link["relations"]]
        if len(originals) != 1 or len(selves) != 1:
            raise _BindingFailure("timemap_requires_one_original_and_self")
        if not urls_bind(candidate_url, originals[0]["target"]):
            raise _BindingFailure("timemap_original_does_not_bind_candidate")
        self_url = _timemap_identity(selves[0]["target"], candidate_url)
        if selves[0]["attributes"].get("type") not in (None, "application/link-format"):
            raise _ParseFailure("unsupported_self_timemap_serialization")
        snapshots, next_pages, seen = [], [], set()
        for link in links:
            target, attributes, relations = link["target"], link["attributes"], set(link["relations"])
            anchor = attributes.get("anchor")
            if anchor is not None and not (anchor == self_url if "self" in relations else urls_bind(candidate_url, anchor)):
                raise _BindingFailure("timemap_link_context_does_not_bind_candidate")
            if "memento" in relations:
                timestamp = _memento_datetime(attributes.get("datetime"))
                snapshot = _snapshot(target, candidate_url, method="timemap")
                if snapshot["timestamp"] != timestamp:
                    raise _BindingFailure("memento_datetime_and_url_timestamp_disagree")
                if snapshot["exact_url"] not in seen:
                    seen.add(snapshot["exact_url"])
                    snapshots.append({**snapshot, "reported_url": target,
                                      "datetime": attributes["datetime"],
                                      "mimetype": None, "statuscode": None,
                                      "replay_http_and_content_type_verification_required": True})
            if ("timemap" in relations or "next" in relations) and "memento" not in relations:
                linked = _timemap_identity(target, candidate_url)
                if attributes.get("type") not in (None, "application/link-format"):
                    raise _ParseFailure("unsupported_linked_timemap_serialization")
                if linked == self_url and "next" in relations:
                    raise _ParseFailure("cyclic_timemap_next_link")
                if linked != self_url and linked not in next_pages:
                    next_pages.append(linked)
        complete = not next_pages
        status = "success" if snapshots or not complete else "empty"
        reason = ("timemap_requires_continuation" if next_pages else
                  "timemap_capture_candidates" if snapshots else "completed_timemap_without_mementos")
        return _result(status, reason, snapshots=snapshots, complete=complete,
                       next_cursor=next_pages[0] if next_pages else None,
                       next_cursors=next_pages, self_url=self_url,
                       original_url=originals[0]["target"], transport_binding_required=True)
    except _BindingFailure as exc:
        return _result("binding_failure", str(exc))
    except _ParseFailure as exc:
        return _result("parser_failure", str(exc))


def parse_availability(body: bytes, candidate_url: str, requested_timestamp: str | None = None) -> dict:
    """Discover the returned nearest snapshot; never equate it to the requested time.

    A valid empty response means no currently accessible closest snapshot was
    reported for this lookup. All Availability results have complete=False.
    """
    request = {"requested_timestamp": requested_timestamp}
    try:
        _candidate(candidate_url)
        if requested_timestamp is not None and (not isinstance(requested_timestamp, str)
                or re.fullmatch(r"[0-9]{1,14}", requested_timestamp) is None):
            raise _ParseFailure("invalid_requested_availability_timestamp")
        payload = _json(body)
        if not isinstance(payload, dict) or not isinstance(payload.get("archived_snapshots"), dict):
            raise _ParseFailure("invalid_availability_envelope")
        if "url" in payload and not urls_bind(candidate_url, payload["url"]):
            raise _BindingFailure("availability_response_query_url_mismatch")
        if "timestamp" in payload and (type(payload["timestamp"]) not in (str, int)
                or requested_timestamp is None or str(payload["timestamp"]) != requested_timestamp):
            raise _BindingFailure("availability_response_query_timestamp_mismatch")
        entries = payload["archived_snapshots"]
        if not entries:
            return _result("empty", "availability_no_currently_accessible_snapshot", **request)
        if set(entries) != {"closest"} or not isinstance(entries["closest"], dict):
            raise _ParseFailure("invalid_availability_snapshot_collection")
        closest = entries["closest"]
        if type(closest.get("available")) is not bool:
            raise _ParseFailure("invalid_availability_available_flag")
        if closest["available"] is False:
            # Optional identity fields still cannot smuggle an unrelated snapshot.
            details = {"returned_available": False}
            if "timestamp" in closest:
                details["returned_timestamp"] = _timestamp(closest["timestamp"])
            if "status" in closest:
                details["returned_status"] = _http_status(closest["status"])
            if "url" in closest:
                identity = _snapshot(closest["url"], candidate_url, method="availability")
                if "returned_timestamp" in details and details["returned_timestamp"] != identity["timestamp"]:
                    raise _BindingFailure("availability_returned_timestamp_and_url_disagree")
                details["returned_url"] = closest["url"]
            return _result("empty", "availability_snapshot_reported_inaccessible", **request, **details)
        timestamp = _timestamp(closest.get("timestamp"))
        statuscode = _http_status(closest.get("status"))
        snapshot = _snapshot(closest.get("url"), candidate_url, method="availability")
        if snapshot["timestamp"] != timestamp:
            raise _BindingFailure("availability_returned_timestamp_and_url_disagree")
        details = {**request, "returned_timestamp": timestamp, "returned_url": closest["url"],
                   "returned_status": statuscode, "returned_available": True}
        if statuscode != 200:
            return _result("empty", "availability_snapshot_not_http_200", **details)
        snapshot.update(requested_timestamp=requested_timestamp, reported_url=closest["url"],
                        statuscode=statuscode, mimetype=None,
                        replay_http_and_content_type_verification_required=True)
        return _result("success", "availability_nearest_snapshot_candidate", snapshots=[snapshot], **details)
    except _BindingFailure as exc:
        return _result("binding_failure", str(exc), **request)
    except _ParseFailure as exc:
        return _result("parser_failure", str(exc), **request)


__all__ = ["parse_cdx", "parse_timemap", "parse_availability"]
