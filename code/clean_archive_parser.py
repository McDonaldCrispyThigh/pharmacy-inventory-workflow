"""Deterministic local parser for independently acquired archived HTML bytes.

No I/O, provider clients, or historical collector imports. Scientific ancestry:
``skills/wayback_agent.py`` identity routes and ``v2_evidence_models.py`` opening
contexts, with the listing-unit definition in ``v2_classifier.py``. Capture dates
never become opening dates. Offsets are Unicode character offsets in the exact
``visible_text`` returned here, not byte offsets in the HTML; each span binds to
both the text digest and original response digest and crawler provenance.

This is a conservative HTML text extractor, not a browser/CSS renderer. Scripts,
head metadata, inline-hidden elements, and archive toolbars are excluded. External
stylesheets are never fetched. Unrecognized wording remains visible for review.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

VERSION = "clean_archive_parser_v1"
_BLOCK = set("address article aside blockquote br dd div dl dt figcaption figure footer h1 h2 h3 h4 h5 h6 header hr li main nav ol p section td th tr ul".split())
_VOID = set("area base br col embed hr img input link meta param source track wbr".split())
_SKIP = set("head script style noscript template svg canvas iframe".split())
_OPENING = re.compile(r"\b(?:opened|established|founded)\s+(?:in\s+)?(\d{4})\b|\b(?:in\s+business|at\s+this\s+location|serving\s+(?:this\s+community|this\s+location))\s+since\s+(\d{4})\b", re.I)
_DOUBT = re.compile(r"n['’]t\b|\b(?:not|never|no|neither|nor|without|false|will|would|may|might|could|if|whether|unless|assuming|supposing|unknown|probably|apparently|appears?|seems?|plan(?:ned|ning)?|expected|scheduled|proposed|future|possibly|perhaps|reportedly|allegedly|purportedly|supposedly|disputed|uncertain|unconfirmed|rumou?r(?:ed|s)?|denied|debunked|refuted|rejected|believed|claim(?:ed|s|ing)?|intend(?:ed|s|ing)?)\b|\?", re.I)
_SECONDARY = re.compile(r"\b(?:review(?:er|s|ed)?|comment(?:s|ed)?|article|post(?:ed)?|copyright|published|publication|opening\s+hours?)\b|©", re.I)
_CORPORATE = re.compile(r"\b(?:company|corporate|corporation|brand|chain|organization|headquarters|nationwide|all\s+(?:our\s+)?(?:stores|locations|pharmacies))\b", re.I)
_OUTLET = re.compile(r"\b(?:this\s+(?:outlet|location|store|pharmacy)|our\s+(?:outlet|location|store|pharmacy)|at\s+this\s+location)\b", re.I)
_SERVICE = re.compile(r"\b(?:prescription\s+(?:refills?|dispensing|pickup|pick-up|services?)|(?:refill|fill|transfer|dispense|dispensing|filling)\s+(?:(?:your|a|the)\s+)?prescriptions?|pharmacy\s+hours|(?:retail|community|public-facing)\s+pharmacy|walk[- ]in\s+pharmacy)\b", re.I)
_CONTRARY = re.compile(r"\b(?:permanently\s+closed|temporarily\s+closed|no\s+longer\s+(?:open|operating|dispensing)|(?:does|do)\s+not\s+(?:fill|dispense)|closed\s+(?:since|in|on)|coming\s+soon|opening\s+soon|(?:will|plans?\s+to)\s+open|not\s+open\s+to\s+(?:the\s+)?public|(?:mail[- ]order|compounding|inpatient|institutional)\s+only)\b", re.I)
_PHONE = re.compile(r"(?<![\w+])(?<!\d[ .\-/])(?:\+1[ .-]?)?(?:\d{10}|\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}|\d{3} \d{3} \d{4}|\(\d{3}\) ?\d{3}-\d{4})(?!\w)")
_STREET = re.compile(r"\b\d{1,6}\s+(?:[\w.'-]+\s+){1,7}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Parkway|Pkwy|Way|Court|Ct)\b", re.I)


class _VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.challenge = False
        self.refresh = False
        self.secondary_text = []

    def handle_starttag(self, tag, attrs):
        attrs = {key.lower(): value or "" for key, value in attrs}
        identity = attrs.get("id", "").lower()
        style = re.sub(r"\s+", "", attrs.get("style", "").lower())
        hidden = (tag in _SKIP or any(x[1] for x in self.stack)
                  or "hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true"
                  or "display:none" in style or "visibility:hidden" in style
                  or identity.startswith(("wm-ipp", "wayback-toolbar")))
        secondary = (tag in {"blockquote", "q"}
                     or bool(re.search(r"review|comment|testimonial", identity + " " + attrs.get("class", ""), re.I))
                     or any(x[2] for x in self.stack))
        if identity.startswith(("cf-chl-", "cf-browser-verification", "challenge-running")):
            self.challenge = True
        if tag == "meta" and attrs.get("http-equiv", "").lower() == "refresh":
            self.refresh = True
        if tag in _BLOCK and not hidden:
            self.parts.append("\n")
        if tag not in _VOID:
            self.stack.append((tag, hidden, secondary))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        hidden = any(x[1] for x in self.stack)
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in _BLOCK and not hidden:
            self.parts.append("\n")

    def handle_data(self, data):
        if not any(x[1] for x in self.stack):
            self.parts.append(data)
            if any(x[2] for x in self.stack):
                self.secondary_text.append(data)


def _normalize_with_offsets(value):
    letters, positions = [], []
    for index, char in enumerate(value):
        for folded in unicodedata.normalize("NFKD", char).casefold():
            if unicodedata.category(folded).startswith("M"):
                continue
            folded = folded if folded.isalnum() else " "
            if folded == " " and (not letters or letters[-1] == " "):
                continue
            letters.append(folded)
            positions.append(index)
    return "".join(letters), positions


def _phrase_spans(phrase, text):
    needle = _normalize_with_offsets(phrase)[0].strip()
    normalized, positions = _normalize_with_offsets(text)
    if not needle:
        return []
    return [(positions[m.start()], positions[m.end() - 1] + 1)
            for m in re.finditer(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", normalized)]


def _digits(value):
    digits = re.sub(r"\D", "", value)
    return digits[1:] if len(digits) == 11 and digits[0] == "1" else digits


def _decode(body, headers):
    content_types = [v for k, v in headers if k.casefold() == "content-type"]
    if len(set(content_types)) > 1:
        raise ValueError("conflicting_content_type_headers")
    content_type = content_types[0] if content_types else ""
    mime = content_type.split(";", 1)[0].strip().casefold()
    if mime and mime not in {"text/html", "application/xhtml+xml"}:
        raise ValueError("non_html_content_type")
    declared = re.search(r"charset\s*=\s*[\"']?([A-Za-z0-9._-]+)", content_type, re.I)
    meta = re.search(br"<meta\b[^>]*charset\s*=\s*[\"']?\s*([A-Za-z0-9._-]+)", body[:4096], re.I)
    if body.startswith(codecs.BOM_UTF8):
        encoding, source = "utf-8-sig", "bom"
    elif body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding, source = "utf-16", "bom"
    elif declared:
        encoding, source = declared[1], "http_header"
    elif meta:
        encoding, source = meta[1].decode("ascii"), "html_meta"
    else:
        encoding, source = "utf-8", "default"
    try:
        text = body.decode(encoding, errors="strict")
    except LookupError as exc:
        raise ValueError("unknown_text_encoding") from exc
    except UnicodeDecodeError as exc:
        if source != "default":
            raise ValueError("invalid_text_encoding") from exc
        try:
            text, encoding, source = body.decode("windows-1252"), "windows-1252", "fallback"
        except UnicodeDecodeError as second:
            raise ValueError("invalid_text_encoding") from second
    if "\x00" in text or any(ord(c) < 32 and c not in "\t\r\n\f" for c in text):
        raise ValueError("binary_response")
    if not re.search(r"<(?:!doctype\s+html|html|body|head|p|div|main|h[1-6]|article|span)\b", text, re.I):
        raise ValueError("html_markup_missing")
    return text, encoding, source


def _snapshot_verified(provenance):
    timestamp = provenance.get("confirmed_snapshot_timestamp")
    if not isinstance(timestamp, str) or not re.fullmatch(r"\d{14}", timestamp):
        return False, "missing_or_invalid_confirmed_timestamp"
    try:
        datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        url = urlsplit(provenance.get("final_url", ""))
    except (ValueError, TypeError):
        return False, "invalid_snapshot_provenance"
    match = re.match(r"/web/(\d{14})(?:[a-z]+_)?/(https?://.+)", url.path)
    if url.scheme not in {"http", "https"} or url.hostname != "web.archive.org" or not match:
        return False, "final_url_not_exact_archive_replay"
    if match[1] != timestamp:
        return False, "snapshot_timestamp_mismatch"
    return True, None


def parse_archive_page(body: bytes, headers, candidate: dict, provenance: dict) -> dict:
    """Return private, JSON-serializable evidence; failures never mean no archive.

    ``sha256`` must hash exactly ``body``. ``confirmed_snapshot_timestamp`` is the
    crawler-confirmed 14-digit timestamp and must agree with ``final_url``.
    Transport/discovery authentication remains the crawler's responsibility.
    A missing timestamp allows text review but prevents all historical eligibility.
    """
    if not isinstance(body, bytes) or not isinstance(candidate, dict) or not isinstance(provenance, dict):
        raise TypeError("body must be bytes; candidate and provenance must be dictionaries")
    provenance = json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))
    header_pairs = list(headers.items()) if isinstance(headers, dict) else list(headers)
    if any(not isinstance(x, (tuple, list)) or len(x) != 2 or not all(isinstance(v, str) for v in x) for x in header_pairs):
        raise TypeError("headers must contain string pairs")
    digest = hashlib.sha256(body).hexdigest()
    provenance_digest = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    verified, timestamp_reason = _snapshot_verified(provenance)
    result = dict(schema_version=VERSION, raw_bytes_present=bool(body), raw_bytes=len(body),
                  raw_sha256=digest, raw_hash_verified=provenance.get("sha256") == digest,
                  raw_success=False, parse_status="collection_failure", reason=None,
                  substantive_parse=False, visible_text="", visible_text_sha256=None,
                  text_truncated=False, scope="unclear", matches={"name": [], "address": [], "phone": []},
                  service_evidence=[], contrary_evidence=[], opening_claims=[],
                  identity_ambiguities=[], active_by_eligible=False, active_by_year=None,
                  opening_year_state="none", explicit_opening_year=None,
                  snapshot_verified=verified, snapshot_verification_reason=timestamp_reason,
                  provenance=provenance, provenance_sha256=provenance_digest,
                  limitations=["Static HTML text; external CSS visibility is not evaluated.",
                               "Eligibility is conservative deterministic evidence screening, not a classifier judgment."])
    if not result["raw_hash_verified"]:
        result["reason"] = "raw_sha256_mismatch"
        return result
    if not body:
        result["reason"] = "empty_body"
        return result
    result["raw_success"] = True
    try:
        decoded, encoding, source = _decode(body, header_pairs)
        parser = _VisibleHTML()
        parser.feed(decoded)
        parser.close()
    except (ValueError, AssertionError) as exc:
        result["reason"] = str(exc) if isinstance(exc, ValueError) else "html_parser_error"
        return result
    text = "\n".join(line for raw in "".join(parser.parts).splitlines()
                     if (line := re.sub(r"[^\S\n]+", " ", raw).strip()))
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    result.update(visible_text=text, visible_text_sha256=text_digest, encoding=encoding, encoding_source=source)
    beginning = text[:1800]
    error_pattern = (r"^(?:error\s*)?(?:40[034]|429|50[0234])\b|^(?:access denied|request blocked|page not found|site not found|just a moment|checking your browser|verify (?:that )?you are human|please complete the security check)\b|"
                     r"wayback machine (?:has not|hasn't|does not|doesn't) (?:archived|have)|this url has been excluded from the wayback machine|cannot be crawled or displayed due to robots\.txt|"
                     r"(?:snapshot|page|url|content) (?:could not be|cannot be|was not) (?:retrieved|accessed|loaded|fetched)")
    if parser.challenge or re.search(error_pattern, beginning, re.I):
        result["reason"] = "archive_error_or_interstitial"
        return result
    if parser.refresh:
        result["reason"] = "meta_refresh_interstitial"
        return result
    if not any(c.isalnum() for c in text):
        result["reason"] = "empty_visible_text"
        return result
    result.update(parse_status="success", reason=None, substantive_parse=True)

    def span(start, end):
        return dict(start=start, end=end, text=text[start:end], raw_sha256=digest,
                    visible_text_sha256=text_digest, provenance_sha256=provenance_digest)

    name = candidate.get("displayName", "")
    name = name.get("text", "") if isinstance(name, dict) else name
    name = name if isinstance(name, str) else ""
    address = candidate.get("formattedAddress") or ""
    street = address.split(",")[0].strip()
    postal_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
    postals = list(re.finditer(r"(?<!\d)" + postal_match[1] + r"(?!\d)", text)) if postal_match else []
    names = _phrase_spans(name, text)
    streets = _phrase_spans(street, text)
    addresses = []
    for start, end in streets:
        # ZIP and street must share one address block, not unrelated page areas.
        for postal in postals:
            left, right = min(start, postal.start()), max(end, postal.end())
            if right - left <= 180 and "\n" not in text[left:right]:
                addresses.append((left, right))
                break
    phone_values = {_digits(candidate.get(key) or "") for key in ("internationalPhoneNumber", "nationalPhoneNumber")}
    phone_values = {x for x in phone_values if len(x) == 10}
    all_phones = list(_PHONE.finditer(text))
    phones = [(m.start(), m.end()) for m in all_phones if _digits(m[0]) in phone_values]
    matches = {"name": names, "address": addresses, "phone": phones}
    result["matches"] = {kind: [span(*pair) for pair in rows] for kind, rows in matches.items()}
    anchors = addresses + phones
    result["scope"] = "outlet_specific" if anchors else "broader_brand" if names else "unclear"
    foreign_streets = [m for m in _STREET.finditer(text) if not any(a <= m.start() < b for a, b in streets)]
    foreign_phones = [m for m in all_phones if phone_values and _digits(m[0]) not in phone_values]
    for kind, rows in (("other_address", foreign_streets), ("other_phone", foreign_phones)):
        result["identity_ambiguities"].extend({"kind": kind, "span": span(m.start(), m.end())} for m in rows)
    single_outlet = bool(anchors) and not result["identity_ambiguities"]
    blocks = [(m.start(), m.end(), m[0]) for m in re.finditer(r"[^\n]+", text)]
    for index, (start, end, block) in enumerate(blocks):
        context = block
        if index and (blocks[index - 1][2].endswith(":") or _CORPORATE.search(blocks[index - 1][2]) or _SECONDARY.search(blocks[index - 1][2])):
            context = blocks[index - 1][2] + "\n" + block
        local = any(start <= a < b <= end for a, b in anchors)
        corporate = bool(_CORPORATE.search(context))
        # A generic service statement can belong to a unique identified outlet;
        # explicit opening claims additionally need an outlet subject or local ID.
        tied = (local or single_outlet) and not corporate
        local_foreign = any(start <= m.start() < end for m in foreign_streets + foreign_phones)
        tied = tied and not local_foreign
        quoted = any(piece.strip() and piece.strip() in block for piece in parser.secondary_text)
        doubtful = bool(_DOUBT.search(context) or _SECONDARY.search(context) or quoted)
        negatives = list(_CONTRARY.finditer(block))
        for m in negatives:
            result["contrary_evidence"].append(dict(span=span(start + m.start(), start + m.end()),
                                                   context_span=span(start, end), outlet_tied=tied))
        for m in _SERVICE.finditer(block):
            eligible = tied and not doubtful and not negatives and verified
            result["service_evidence"].append(dict(span=span(start + m.start(), start + m.end()),
                context_span=span(start, end), outlet_tied=tied, eligible=eligible,
                reason=None if eligible else "unverified_ambiguous_nonoperational_or_unbound_service"))
        for m in _OPENING.finditer(block):
            year = int(next(x for x in m.groups() if x is not None))
            prefix = block[:m.start()]
            subject_pattern = (r"\b(?:(?:this|the|our)\s+(?:outlet|location|store|pharmacy)|it|we)"
                               r"\s+(?:(?:was|is|were|have|has|been|had|did|not|never|"
                               r"allegedly|reportedly|possibly|might|may|could|wasn't|wasn’t)\s+)*$")
            own_subject = bool(re.search(subject_pattern, prefix, re.I))
            named_subject = bool(name and re.search(re.escape(name) + r"\s+(?:(?:was|is)\s+)?$", prefix, re.I))
            # Two opening verbs coordinated under one outlet subject remain
            # separate claims; a new named subject cannot borrow a nearby address.
            coordinated = bool(_OUTLET.search(prefix) and re.search(r"\band\s+(?:was\s+|is\s+)?$", prefix, re.I))
            bare_local = local and not prefix.strip()
            own_subject = own_subject or named_subject or coordinated or bare_local
            opening_tied = tied and own_subject
            reason = ("broader_brand_claim" if corporate else "outlet_claim_not_bound" if not opening_tied
                      else "ambiguous_negated_or_secondary_context" if doubtful
                      else "snapshot_unverified" if not verified
                      else "year_after_snapshot" if year > int(provenance["confirmed_snapshot_timestamp"][:4])
                      else "implausible_year" if year < 1600 else None)
            result["opening_claims"].append(dict(year=year, span=span(start + m.start(), start + m.end()),
                context_span=span(start, end), scope="broader_brand" if corporate else "outlet_specific" if opening_tied else "unclear",
                eligible=reason is None, reason=reason))
    years = sorted({x["year"] for x in result["opening_claims"] if x["eligible"]})
    relevant_rejected = any(x["scope"] == "outlet_specific" and not x["eligible"] for x in result["opening_claims"])
    state = "conflict" if len(years) > 1 or (years and relevant_rejected) else "single" if years else "none"
    result["opening_year_state"] = state
    result["explicit_opening_year"] = years[0] if state == "single" else None
    active = any(x["eligible"] for x in result["service_evidence"])
    active = (active and not any(x["outlet_tied"] for x in result["contrary_evidence"])
              and state != "conflict" and not relevant_rejected)
    result["active_by_eligible"] = active
    result["active_by_year"] = int(provenance["confirmed_snapshot_timestamp"][:4]) if active else None
    return result
