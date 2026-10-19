"""Conservative site binding for the separately repaired archive transport.

These are comparison keys, not acquisition URLs or proofs of archived content.
Callers must retain original URLs, response bytes, timestamps and provenance.
No network or provider modules are imported.
"""
from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from urllib.parse import SplitResult, urlsplit, urlunsplit


_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PERCENT = re.compile(r"%([0-9A-Fa-f]{2})")
_BAD_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_TRACKING_KEYS = frozenset({"gclid", "dclid", "fbclid", "msclkid", "gbraid", "wbraid", "mc_cid", "mc_eid", "_ga", "_gl"})
_REPLAY = re.compile(r"/web/([0-9]{14})(?:id_)?/((?i:https?)://.+)\Z")


def _validated_parts(value: str) -> SplitResult:
    if (not isinstance(value, str) or not value
            or any(not char.isprintable() or char.isspace() for char in value)
            or "\\" in value or _BAD_PERCENT.search(value)):
        raise ValueError("URL must be a printable absolute HTTP(S) URL")
    try:
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.netloc or not parts.hostname
                or parts.username is not None or parts.password is not None
                or "%" in parts.hostname or parts.netloc.endswith(":")):
            raise ValueError("invalid HTTP(S) URL authority")
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError("invalid HTTP(S) URL port")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid HTTP(S) URL authority") from exc
    return parts


def _host_key(host: str) -> str:
    if ":" in host:
        try:
            return "[" + ipaddress.IPv6Address(host).compressed + "]"
        except ValueError as exc:
            raise ValueError("invalid IPv6 URL host") from exc
    # Require URI-form ASCII hosts (IDNs may use their explicit xn-- spelling).
    # Python's legacy IDNA codec can otherwise collapse distinct modern domains,
    # for example fa\N{LATIN SMALL LETTER SHARP S}.example into fass.example.
    if not host.isascii():
        raise ValueError("URL host must use its explicit ASCII URI spelling")
    host = host.lower()
    labels = host.removesuffix(".").split(".")
    if len(host) > 254 or any(_HOST_LABEL.fullmatch(label) is None for label in labels):
        raise ValueError("invalid URL host")
    # Collapse the standard single www prefix; preserve nested subdomain names.
    if host.startswith("www.") and not host.startswith("www.www."):
        host = host[4:]
    return host


def _percent_key(value: str) -> str:
    def replace(match):
        decoded = chr(int(match[1], 16))
        return decoded if decoded in _UNRESERVED else "%" + match[1].upper()
    return _PERCENT.sub(replace, value)


def _remove_dot_segments(path: str) -> str:
    """RFC-style dot removal without merging repeated or encoded separators."""
    result = ""
    while path:
        if path.startswith("../"):
            path = path[3:]
        elif path.startswith("./"):
            path = path[2:]
        elif path.startswith("/./"):
            path = path[2:]
        elif path == "/.":
            path = "/"
        elif path.startswith("/../"):
            path = path[3:]
            result = result.rpartition("/")[0]
        elif path == "/..":
            path = "/"
            result = result.rpartition("/")[0]
        elif path in {".", ".."}:
            path = ""
        else:
            boundary = path.find("/", 1 if path.startswith("/") else 0)
            if boundary < 0:
                result += path
                break
            result += path[:boundary]
            path = path[boundary:]
    return result


def _tracking_value_key(value: str) -> str:
    # Preserve reserved percent escapes: %2F stays encoded and uppercase.
    return "".join(piece if _PERCENT.fullmatch(piece) else piece.casefold()
                   for piece in re.split(r"(%[0-9A-F]{2})", value))


def _query_key(query: str) -> str:
    if not query:
        return ""
    records = []
    for field in query.split("&"):
        key, separator, value = field.partition("=")
        key, value = _percent_key(key), _percent_key(value)
        lower_key = key.casefold()
        if lower_key.startswith("utm_"):
            key, value = _tracking_value_key(key), _tracking_value_key(value)
        elif lower_key in _TRACKING_KEYS:
            key = lower_key
        records.append((key, separator, value))
    # Query-key order is irrelevant, but repeated values retain their order.
    # Blank fields and the distinction between ?flag and ?flag= are retained.
    return "&".join(key + separator + value for key, separator, value in sorted(records, key=lambda row: row[0]))


def normalize_site_url(url: str) -> str:
    """Return an idempotent site comparison URL, raising ValueError if invalid.

    HTTP/HTTPS and a single leading www are equivalent. Default ports are
    interpreted under the input scheme; explicit nondefault ports remain bound.
    Path/query case, parameters, encoded separators, duplicate-value order and
    fragments remain meaningful. Only utm_* values are case folded; other
    tracking identifiers retain their opaque case-sensitive values.
    """
    parts = _validated_parts(url)
    host = _host_key(parts.hostname)
    port = parts.port
    if port == (80 if parts.scheme == "http" else 443):
        port = None
    # Keep an explicit HTTP:443 distinct from default HTTPS:443 while making
    # repeated normalization stable. HTTPS:80 is already nondefault under HTTPS.
    scheme = "http" if parts.scheme == "http" and port == 443 else "https"
    authority = host + (f":{port}" if port is not None else "")
    path = _remove_dot_segments(_percent_key(parts.path or "/")) or "/"
    if path != "/" and path.endswith("/") and not path.endswith("//"):
        path = path[:-1]
    return urlunsplit((scheme, authority, path, _query_key(parts.query), _percent_key(parts.fragment)))


def urls_bind(candidate_url: str, archive_original_url: str) -> bool:
    """Compare retained URLs without weakening failures into a positive match."""
    try:
        return normalize_site_url(candidate_url) == normalize_site_url(archive_original_url)
    except (ValueError, TypeError):
        return False


def snapshot_identity(replay_url: str) -> dict[str, str]:
    """Validate an official exact replay and preserve its original URL spelling.

    Both normal and id_ replays are accepted. The returned exact_url always uses
    HTTPS and id_; the original URL is validated, never replaced by its site key.
    This establishes URL structure only, not capture existence or text fidelity.
    """
    parts = _validated_parts(replay_url)
    if (parts.hostname.lower() != "web.archive.org" or parts.fragment
            or parts.port not in {None, 80 if parts.scheme == "http" else 443}):
        raise ValueError("snapshot must use the official Wayback host")
    tail = parts.path + ("?" + parts.query if "?" in replay_url else "")
    match = _REPLAY.fullmatch(tail)
    if match is None:
        raise ValueError("snapshot requires an exact 14-digit normal or id_ replay")
    timestamp, original = match.groups()
    try:
        datetime.strptime(timestamp, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ValueError("snapshot timestamp is not a valid calendar timestamp") from exc
    normalize_site_url(original)
    return {
        "timestamp": timestamp,
        "original_url": original,
        "exact_url": f"https://web.archive.org/web/{timestamp}id_/{original}",
    }


__all__ = ["normalize_site_url", "urls_bind", "snapshot_identity"]
