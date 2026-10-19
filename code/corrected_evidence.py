"""Content-complete, privacy-safe evidence freezes for the corrected empirical run.

Only this explicitly authorized run serializes transient Google packets. Full
returned review text is retained; the established classifier still receives its
existing normalized 200-character snippets in provider order. Reviewer attribution
and profile links are unused by that design and are removed. Checksums supplement
the retained content; none is a provider signature or a substitute for the content.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

import v2_evidence_models as models
from forced_choice_classifier import (
    DECISION_SCHEMA, PROMPT_TEMPLATE_PAYLOAD, PROMPT_TEMPLATE_SHA256,
    PROMPT_VERSION, SYSTEM_INSTRUCTIONS, build_forced_choice_messages,
)
from forced_choice_models import (
    FORCED_CHOICE_CONTRACT, FORCED_CHOICE_CONTRACT_SHA256, FORCED_CHOICE_SCHEMA_VERSION,
)
from v2_classifier import _escaped_json, _wayback_prompt, build_candidate_facts, build_review_evidence

PACKET_SCHEMA = "nets-ai-corrected-evidence-packet-v1"
EVIDENCE_POLICY_VERSION = "2026-09-05-corrected-archive-evidence-v1"
MODEL = "gpt-5-mini-2025-08-07"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_BOARD_KEYS = {"board", "board_record", "board_match", "board_records", "license_nbr",
               "board_link", "board_linkage", "neighborhood", "tract", "proximity"}
_DETAIL_FIELDS = {"id", "displayName", "formattedAddress", "nationalPhoneNumber",
                  "internationalPhoneNumber", "location", "websiteUri", "regularOpeningHours",
                  "types", "rating", "userRatingCount", "businessStatus", "reviews",
                  "addressComponents"}
_REVIEW_PII_FIELDS = {"authorAttribution", "author_name", "author_url", "profile_photo_url",
                      "googleMapsUri", "name", "flagContentUri"}
_TYPE_NAMES = (
    "QueryProvenanceV2", "CandidateFactsV2", "ReviewSnippetV2", "ReviewEvidenceV2",
    "ArchiveCDXCaptureRowV2", "ArchiveCDXPageV2", "ArchiveSourceSnapshotV2",
    "ArchiveFetchRecordV2", "ArchiveOpeningClaimV2", "ArchiveMatchAssessmentV2",
    "ArchiveClaimAssessmentV2", "ArchiveAnnualSelectionV2", "ArchiveEvidenceV2",
    "WaybackCollectionV2", "ClassifierContractV2", "ClassifierInputV2",
)
_TYPES = {name: getattr(models, name) for name in _TYPE_NAMES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("packet contains an invalid JSON value") from exc


def packet_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _iso(value: datetime) -> str:
    _require(isinstance(value, datetime) and value.tzinfo is not None
             and value.utcoffset() is not None, "packet timestamp must be timezone aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _time(value: Any) -> datetime:
    _require(isinstance(value, str), "invalid retained packet timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid retained packet timestamp") from exc
    _require(_iso(parsed) == value, "retained packet timestamps must be canonical UTC")
    return parsed


def _hash(value: Any) -> None:
    _require(isinstance(value, str) and _SHA.fullmatch(value) is not None,
             "invalid packet provenance hash")


def _no_board_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(isinstance(key, str) and key.casefold() not in _BOARD_KEYS,
                     "downstream reference fields are forbidden in acquisition packets")
            _no_board_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _no_board_fields(item)


def sanitize_details(details: Mapping[str, Any]) -> dict[str, Any]:
    """Keep every requested observation; remove only unused reviewer identifiers."""
    _require(isinstance(details, Mapping), "Details must be a JSON object")
    value = _copy(dict(details))
    _no_board_fields(value)
    _require(set(value).issubset(_DETAIL_FIELDS), "undeclared Details response fields")
    rows = value.get("reviews", [])
    _require(isinstance(rows, list) and len(rows) <= 5, "invalid complete returned review subset")
    for row in rows:
        _require(isinstance(row, dict), "invalid returned review row")
        for key in _REVIEW_PII_FIELDS:
            row.pop(key, None)
    return value


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        name = type(value).__name__
        _require(name in _TYPES and type(value) is _TYPES[name], "undeclared evidence model")
        return {"__type__": name,
                "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, datetime):
        return {"__datetime__": _iso(value)}
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return _copy(value)


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "__datetime__" in value:
        _require(set(value) == {"__datetime__"}, "invalid datetime envelope")
        return _time(value["__datetime__"])
    if "__bytes_base64__" in value:
        _require(set(value) == {"__bytes_base64__"}, "invalid bytes envelope")
        try:
            return base64.b64decode(value["__bytes_base64__"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid retained evidence bytes") from exc
    if "__type__" in value:
        _require(set(value) == {"__type__", "fields"}
                 and isinstance(value["__type__"], str) and value["__type__"] in _TYPES,
                 "unknown typed evidence envelope")
        cls = _TYPES[value["__type__"]]
        data = value["fields"]
        _require(isinstance(data, dict) and set(data) == {field.name for field in fields(cls)},
                 "retained model fields differ from the strict contract")
        try:
            return cls(**{key: _decode(item) for key, item in data.items()})
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("retained typed evidence failed validation") from exc
    _require(not any(key.startswith("__") for key in value), "invalid evidence marker")
    return {key: _decode(item) for key, item in value.items()}


def _verification_records(value: Any, collection: models.WaybackCollectionV2) -> list[dict[str, Any]]:
    sources = {item.evidence_id: item for item in collection.archive_evidence.source_snapshots}
    if value is None:
        return []
    if isinstance(value, dict):
        _require(all(isinstance(item, dict) for item in value.values()), "invalid source verification record")
        records = [{"source_evidence_id": key, **item} for key, item in value.items()]
    else:
        _require(isinstance(value, list), "archive verifications must be records")
        records = _copy(value)
    _no_board_fields(records)
    seen = set()
    for record in records:
        _require(isinstance(record, dict) and record.get("source_evidence_id") in sources,
                 "archive verification does not bind a retained source")
        key = record["source_evidence_id"]
        _require(key not in seen, "duplicate archive verification")
        seen.add(key)
        source = sources[key]
        if record.get("schema_version") == "availability_snapshot_verification_v1":
            _require(record.get("source_provenance_sha256") == source.provenance_sha256
                     and record.get("source_retrieval_source") == source.retrieval_source == "outscraper_ai"
                     and record.get("requested_replay_url") == source.replay_url
                     and record.get("original_url") == source.original_url
                     and record.get("requested_timestamp") == source.captured_at.strftime("%Y%m%d%H%M%S")
                     and record.get("verification_scope") == "snapshot_identity_and_timestamp_only"
                     and record.get("independently_verified_text") is False
                     and record.get("opening_statement_verified") is False,
                     "Availability record changed its source or textual evidence boundary")
            _require(record.get("status") in {"confirmed", "mismatch", "unavailable", "transport_failure"}
                     and type(record.get("snapshot_identity_confirmed")) is bool
                     and record["snapshot_identity_confirmed"] == (record["status"] == "confirmed"),
                     "invalid Availability verification state")
            _time(record.get("checked_at"))
            if record.get("provenance_sha256") is not None:
                _require(isinstance(record.get("response"), dict),
                         "Availability HTTP provenance must retain its response envelope")
                wrapper = {"provenance_sha256": record["provenance_sha256"], "response": record["response"]}
                resolved = _official_response(wrapper)
                _require(resolved[0]["request_url"] == record.get("request_url"),
                         "Availability response changed its requested URL")
            else:
                _require(record["status"] == "transport_failure" and record.get("response") is None,
                         "Availability confirmation requires retained official response content")
            checked = _snapshot_check(source, record)
            _require((checked == "verified_official_snapshot") == record["snapshot_identity_confirmed"],
                     "Availability state differs from its retained response")
    return _copy(records)


def _verification_for(source_id: str, records: Any) -> dict[str, Any]:
    if not records:
        return {}
    if isinstance(records, dict):
        return records.get(source_id, {})
    return next((item for item in records if item.get("source_evidence_id") == source_id), {})


def _official_response(wrapper: Any) -> tuple[dict[str, Any], bytes] | None:
    if wrapper is None:
        return None
    _require(isinstance(wrapper, dict) and set(wrapper) == {"provenance_sha256", "response"},
             "independent response requires its complete committed envelope")
    _hash(wrapper["provenance_sha256"])
    value = wrapper["response"]
    _require(isinstance(value, dict) and packet_sha256(value) == wrapper["provenance_sha256"],
             "independent response envelope commitment mismatch")
    _require(value.get("provider") == "direct_wayback", "independent verification must use official Wayback")
    try:
        body = base64.b64decode(value["body_base64"], validate=True)
        _require(hashlib.sha256(body).hexdigest() == value["response_sha256"]
                 and len(body) == value["response_bytes"], "independent response body commitment mismatch")
        _time(value["retrieved_at"])
        endpoint = urlparse(value["request_url"])
        _require(endpoint.scheme == "https"
                 and (endpoint.netloc == "web.archive.org"
                      or (endpoint.netloc == "archive.org" and endpoint.path == "/wayback/available")),
                 "independent verification endpoint is not official Wayback")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid independent archive response") from exc
    return value, body


def _original_statement_text(source: models.ArchiveSourceSnapshotV2, verification: dict) -> str | None:
    resolved = _official_response(verification.get("original_statement"))
    if resolved is None:
        return None
    response, body = resolved
    _require(response["request_url"] == source.replay_url
             and response.get("final_url") == source.replay_url,
             "independent statement must bind the exact selected replay")
    if response.get("response_status") != 200:
        return None
    headers = {key.lower(): value for key, value in response.get("headers", {}).items()}
    content_type = headers.get("content-type", "")
    _require(re.match(r"(?i)^(?:text/html|application/xhtml\+xml)(?:;|$)", content_type) is not None,
             "independent statement requires an original HTML response")
    try:
        from wayback_agent import _html_to_text
    except ImportError:
        from skills.wayback_agent import _html_to_text
    return _html_to_text(body, content_type, 5000)[0]


def _snapshot_check(source: models.ArchiveSourceSnapshotV2, verification: dict) -> str:
    if verification.get("original_statement") is not None:
        if _original_statement_text(source, verification) is not None:
            return "verified_official_replay"
    wrapper = verification.get("snapshot_check")
    if verification.get("schema_version") == "availability_snapshot_verification_v1" and verification.get("provenance_sha256"):
        wrapper = {"provenance_sha256": verification["provenance_sha256"], "response": verification.get("response")}
    resolved = _official_response(wrapper)
    if resolved is None:
        return "not_independently_verified"
    response, body = resolved
    if response.get("response_status") != 200:
        return "official_check_unavailable"
    if response.get("final_url") != response["request_url"]:
        return "official_check_unavailable"
    parsed = urlparse(response["request_url"])
    if parsed.path == "/cdx/search/cdx":
        rows, _ = models.parse_cdx_response_v2(body)
        matches = any(item.captured_at == source.captured_at and item.original_url == source.original_url
                      for item in rows)
    elif parsed.path == "/wayback/available":
        _require(parse_qs(parsed.query).get("timestamp") == [source.captured_at.strftime("%Y%m%d%H%M%S")],
                 "Availability request changed the selected capture timestamp")
        try:
            payload = json.loads(body)
            closest = payload.get("archived_snapshots", {}).get("closest", {})
            _require(isinstance(closest, dict), "invalid Availability snapshot")
        except (ValueError, TypeError, AttributeError):
            return "official_check_unavailable"
        replay = closest.get("url", "")
        match = re.fullmatch(r"https?://web\.archive\.org/web/(\d{14})(?:id_)?/(https?://.+)", replay)
        matches = (closest.get("available") is True and str(closest.get("status")) == "200"
                   and closest.get("timestamp") == source.captured_at.strftime("%Y%m%d%H%M%S")
                   and match is not None and match.group(1) == closest["timestamp"]
                   and models.canonical_http_url(match.group(2)) == source.original_url)
    else:
        raise ValueError("unsupported official archive snapshot check")
    target = parse_qs(parsed.query).get("url", [])
    _require(len(target) == 1 and models.canonical_http_url(target[0]) == source.original_url,
             "official snapshot check requested a different original URL")
    return "verified_official_snapshot" if matches else "official_snapshot_not_confirmed"


def apply_archive_policy(collection: models.WaybackCollectionV2,
                         archive_verifications: Any = None) -> models.WaybackCollectionV2:
    """Retain observed evidence separately; withhold unverified derived opening claims."""
    _require(isinstance(collection, models.WaybackCollectionV2), "a typed Wayback collection is required")
    sources = {item.evidence_id: item for item in collection.archive_evidence.source_snapshots}
    kept = []
    for claim in collection.archive_evidence.opening_claims:
        src = sources[claim.source_evidence_id]
        if src.retrieval_source == "direct_wayback":
            kept.append(claim)
            continue
        verification = _verification_for(src.evidence_id, archive_verifications)
        original = _original_statement_text(src, verification)
        if original is not None and claim.excerpt in original and models.opening_claim_has_assertive_context(original, claim.excerpt):
            kept.append(claim)
    keep_ids = {item.evidence_id for item in kept}
    return models.WaybackCollectionV2(
        replace(collection.archive_evidence, opening_claims=tuple(kept)),
        collection.archive_match_assessments,
        tuple(item for item in collection.archive_claim_assessments if item.claim_evidence_id in keep_ids),
    )


def summarize_archive(collection: models.WaybackCollectionV2,
                      archive_verifications: Any = None) -> dict[str, Any]:
    archive = collection.archive_evidence
    sources = archive.source_snapshots
    scopes = {item.source_evidence_id: item.evidence_scope for item in collection.archive_match_assessments}
    captures = [row for page in archive.cdx_pages for row in page.capture_rows]
    if archive.query_status in {"missing_url", "invalid_url"}:
        outcome = "no_website"
    elif any(item.retrieval_source == "direct_wayback" for item in sources):
        outcome = "direct_raw"
    elif sources:
        outcome = "outscraper_extracted"
    elif captures:
        outcome = "structural_only"
    elif archive.query_status == "success_no_capture":
        outcome = "no_archive"
    else:
        outcome = "collection_failure"
    per_source = []
    for source in sources:
        scope = scopes[source.evidence_id]
        per_source.append({"source_evidence_id": source.evidence_id,
                           "retrieval_source": source.retrieval_source,
                           "evidence_scope": scope, "substantive": True,
                           "active_by_eligible": scope == "outlet_specific",
                           "independent_snapshot_check": _snapshot_check(
                               source, _verification_for(source.evidence_id, archive_verifications))})
    return {"outcome": outcome, "query_status": archive.query_status,
            "page_status": archive.page_status, "cap_status": archive.cap_status,
            "substantive_source_count": len(sources), "structural_capture_count": len(captures),
            "outlet_specific_archive_evidence_count": sum(value == "outlet_specific" for value in scopes.values()),
            "broader_brand_archive_evidence_count": sum(value == "domain_or_brand_only" for value in scopes.values()),
            "unclear_scope_archive_evidence_count": sum(value == "probable_outlet" for value in scopes.values()),
            "active_by_evidence_count": sum(value == "outlet_specific" for value in scopes.values()),
            "explicit_opening_year_evidence_count": sum(item.evidence_scope == "outlet_specific"
                                                       for item in collection.archive_claim_assessments),
            "sources": per_source}


def build_corrected_messages(packet: models.ClassifierInputV2,
                             archive_verifications: Any = None) -> list[dict[str, str]]:
    """Keep the final system/schema/core design and label the corrected source blocks."""
    filtered = apply_archive_policy(packet.wayback_collection, archive_verifications)
    _require(filtered == packet.wayback_collection, "classifier packet contains a withheld derived claim")
    messages = build_forced_choice_messages(packet)
    block = _wayback_prompt(packet)
    block["evidence_policy_version"] = EVIDENCE_POLICY_VERSION
    block["acquisition_summary"] = summarize_archive(filtered, archive_verifications)
    sources = {item.evidence_id: item for item in filtered.archive_evidence.source_snapshots}
    for entry in block["sources"]:
        source = sources[entry["evidence_id"]]
        derived = source.retrieval_source == "outscraper_ai"
        entry.update(retrieval_source=source.retrieval_source,
                     content_type=("archive-derived extracted text" if derived else "text from direct archived HTML"),
                     byte_fidelity=("not_byte_faithful" if derived else "direct_response_retained"),
                     exact_opening_year_rule="only supplied independently verified opening_claims may support an exact year")
    if not block["sources"]:
        block["structural_captures"] = [
            {"capture_evidence_id": row.evidence_id, "captured_at": _iso(row.captured_at),
             "original_url": row.original_url, "evidence_kind": "snapshot metadata only",
             "supports_archived_text_or_historical_year": False}
            for page in filtered.archive_evidence.cdx_pages for row in page.capture_rows
        ]
    opening, closing = "<UNTRUSTED_WAYBACK_JSON>\n", "\n</UNTRUSTED_WAYBACK_JSON>"
    prefix, remainder = messages[1]["content"].split(opening, 1)
    _, suffix = remainder.split(closing, 1)
    messages[1]["content"] = prefix + opening + _escaped_json(block) + closing + suffix
    _no_board_fields(block)
    _require(messages[0] == {"role": "system", "content": SYSTEM_INSTRUCTIONS}, "classifier system changed")
    return messages


def _bind_archive_to_candidate(candidate: models.CandidateFactsV2,
                               collection: models.WaybackCollectionV2) -> None:
    archive = collection.archive_evidence
    _require(archive.query_scope == "candidate_website", "corrected archive must use the candidate website")
    if candidate.website is None or not candidate.website.strip():
        _require(archive.query_status == "missing_url", "missing website disagrees with its archive query")
        return
    try:
        try:
            from wayback_agent import _canonical_candidate_url
        except ImportError:
            from skills.wayback_agent import _canonical_candidate_url
        canonical = _canonical_candidate_url(candidate.website)
    except models.EvidenceValidationError:
        _require(archive.query_status == "invalid_url", "invalid website disagrees with its archive query")
        return
    expected = models.evidence_id("wayback_query_v2", {
        "query_definition": "exact_html_cdx_v2", "canonical_url": canonical,
    })
    _require(archive.query_key_digest == expected,
             "archive query provenance does not bind the retained candidate website")


def make_packet(details: Mapping[str, Any], provenance: models.QueryProvenanceV2,
                retrieved_at: datetime, collection: models.WaybackCollectionV2, *,
                frame_entry: dict[str, str], fixed_frame_sha256: str,
                model: str = MODEL, archive_verifications: Any = None) -> dict[str, Any]:
    _hash(fixed_frame_sha256)
    _require(model in {MODEL, "gpt-5-mini"}, "corrected packet model differs from the authorized family")
    _require(isinstance(provenance, models.QueryProvenanceV2), "typed query provenance is required")
    _require(isinstance(frame_entry, dict) and set(frame_entry) == {"place_id", "query_definition_id", "source_zip"},
             "complete frozen frame entry is required")
    retained = sanitize_details(details)
    _require(frame_entry == {"place_id": retained.get("id"),
                             "query_definition_id": provenance.query_definition_id,
                             "source_zip": provenance.source_zip}, "Details changed the frozen frame binding")
    candidate = build_candidate_facts(retained, provenance, retrieved_at)
    reviews = build_review_evidence(retained, retrieved_at)
    _require(reviews.status == "success", "the complete returned review subset failed validation")
    _require(isinstance(collection, models.WaybackCollectionV2), "typed Wayback provenance is required")
    _bind_archive_to_candidate(candidate, collection)
    verifications = _verification_records(archive_verifications, collection)
    filtered = apply_archive_policy(collection, verifications)
    packet = models.ClassifierInputV2(candidate, reviews, filtered,
                                      models.ClassifierContractV2.from_project_constants())
    messages = build_corrected_messages(packet, verifications)
    record = {
        "schema_version": PACKET_SCHEMA, "evidence_policy_version": EVIDENCE_POLICY_VERSION,
        "candidate_id": candidate.candidate_id, "place_id": candidate.source_place_id,
        "frame_entry": _copy(frame_entry), "fixed_frame_sha256": fixed_frame_sha256,
        "model": model, "retrieved_at": _iso(retrieved_at),
        "query_provenance": _encode(provenance), "google_details": retained,
        "google_details_sha256": packet_sha256(retained),
        "wayback_observed": _encode(collection), "archive_verifications": verifications,
        "classifier_input": _encode(packet), "messages": messages,
        "messages_sha256": packet_sha256(messages),
        "archive_summary": summarize_archive(filtered, verifications),
        "decision_schema": _copy(DECISION_SCHEMA), "decision_schema_sha256": packet_sha256(DECISION_SCHEMA),
        "classifier_schema_version": FORCED_CHOICE_SCHEMA_VERSION,
        "classifier_contract": _copy(FORCED_CHOICE_CONTRACT),
        "classifier_contract_sha256": FORCED_CHOICE_CONTRACT_SHA256,
        "base_prompt_template": _copy(PROMPT_TEMPLATE_PAYLOAD),
        "base_prompt_template_sha256": PROMPT_TEMPLATE_SHA256, "base_prompt_version": PROMPT_VERSION,
        "privacy_policy": {"reviewer_attribution_retained": False, "full_returned_review_text_retained": True,
                           "classifier_review_normalization": "existing NFKC/control/whitespace normalization and 200-character limit"},
    }
    record["wayback_observed_sha256"] = packet_sha256(record["wayback_observed"])
    record["classifier_input_sha256"] = packet_sha256(record["classifier_input"])
    record["archive_verifications_sha256"] = packet_sha256(verifications)
    record["packet_sha256"] = packet_sha256(record)
    return record


def restore_packet(record: Mapping[str, Any]) -> models.ClassifierInputV2:
    """Validate every typed constructor and reproduce all retained messages/content."""
    _require(isinstance(record, Mapping), "retained packet must be an object")
    value = _copy(dict(record))
    expected_hash = value.pop("packet_sha256", None)
    _hash(expected_hash)
    _require(packet_sha256(value) == expected_hash, "complete evidence packet commitment mismatch")
    try:
        observed = _decode(value["wayback_observed"])
        provenance = _decode(value["query_provenance"])
        restored = _decode(value["classifier_input"])
        _require(isinstance(restored, models.ClassifierInputV2), "invalid retained classifier model")
        rebuilt = make_packet(value["google_details"], provenance, _time(value["retrieved_at"]), observed,
                              frame_entry=value["frame_entry"], fixed_frame_sha256=value["fixed_frame_sha256"],
                              model=value["model"], archive_verifications=value["archive_verifications"])
        _require(rebuilt == {**value, "packet_sha256": expected_hash},
                 "retained packet differs from reconstructed evidence, policy, or messages")
        return restored
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete retained evidence packet") from exc


__all__ = ["PACKET_SCHEMA", "EVIDENCE_POLICY_VERSION", "MODEL", "make_packet", "restore_packet",
           "build_corrected_messages", "sanitize_details", "apply_archive_policy", "summarize_archive", "packet_sha256"]
