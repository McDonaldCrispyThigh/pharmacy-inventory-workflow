"""Read-only Phase 1 failure and extracted-source provenance audit.

Frozen outcomes are never changed. A failed packet often retains only the direct
failure; a scope-bound SQLite snapshot indexes the final fallback and all its raw
attempts. Such recovered associations are disclosed separately from packet SHA
commitments. This module has no provider client and performs no filesystem writes.
"""
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qsl, urlsplit

from corrected_archive_verification import assess_availability
from corrected_evidence import _decode, _snapshot_check, restore_packet
from corrected_run_store import canonical, file_sha, read_sealed, sha, unit_key
from outscraper_wayback import PROMPT_VERSION, _extract, _target
from v2_evidence_models import EvidenceValidationError, canonical_http_url, evidence_id
from wayback_state import WaybackStateError, WaybackStateStore

SCHEMA = "nets-ai-corrected-failure-audit-v1"
OUTCOMES = ("direct_raw", "outscraper_extracted", "structural_only", "no_archive",
            "no_website", "collection_failure")
PROJECT = Path(__file__).resolve().parents[1]
_MAX_ROWS = 100_000
_MAX_ENVELOPE_BYTES = 48_000_000
_EXTRACTION_ERRORS = {
    "unexpected_extraction_shape", "reported_retrieval_error", "target_mismatch",
    "missing_cdx_response", "cdx_original_target_mismatch",
    "cdx_page_completeness_unverified", "archive_timestamp_mismatch", "missing_replay_text",
}
_PARSER_MESSAGES = {
    "invalid CDX JSON", "invalid CDX table", "invalid CDX header",
    "invalid CDX resume record", "orphan CDX resume key", "malformed CDX data row",
    "malformed CDX field type", "invalid CDX original URL", "unsafe URL in exact CDX response",
    "invalid exact CDX field", "invalid exact CDX data row", "invalid CDX mimetype",
    "invalid CDX status code", "invalid CDX digest", "invalid CDX length",
}


def _hash_bytes(body):
    return hashlib.sha256(body).hexdigest()


def _references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"provenance_sha256", "source_provenance_sha256"} and item:
                yield item
            elif key == "raw_response_sha256" and isinstance(item, list):
                yield from item
            else:
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


def _request_key(kind, target):
    if kind == "cdx":
        parsed = urlsplit(target)
        endpoint = target.split("?", 1)[0]
        params = [list(item) for item in parse_qsl(parsed.query, keep_blank_values=True)]
    elif kind == "replay":
        endpoint, params = target, []
    else:
        raise ValueError("unknown archive request kind")
    return evidence_id("wayback_cache_v2", {"schema_version": "wayback_request_v3",
                       "kind": kind, "endpoint": endpoint, "ordered_parameters": params})


def _job_key(scope, kind, target):
    return _hash_bytes(json.dumps([PROMPT_VERSION, scope, kind, target],
                                 ensure_ascii=False, separators=(",", ":")).encode())


def _state_snapshot(directory, scope):
    """Use SQLite's read transaction, never the store's mutating connection API."""
    database = directory / "state.sqlite3"
    jobs, direct, controls, commitments = {}, {}, {}, []
    if not database.exists():
        return jobs, direct, controls, commitments
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        connection.execute("BEGIN")
        rows = connection.execute(
            "SELECT namespace,key,payload,sha256 FROM state WHERE namespace IN (?,?,?,?) ORDER BY namespace,key",
            ("outscraper_jobs", "direct_requests", "cooldown", "outscraper_control"))
        for index, (namespace, key, body, digest) in enumerate(rows):
            if index >= _MAX_ROWS:
                raise ValueError("failure audit state row bound exceeded")
            if _hash_bytes(body) != digest:
                raise WaybackStateError("failure audit state checkpoint hash mismatch")
            value = json.loads(body)
            if not isinstance(value, dict):
                raise WaybackStateError("failure audit state checkpoint shape mismatch")
            if namespace == "outscraper_jobs":
                if value.get("scope") != scope:
                    continue
                if key != _job_key(scope, value["kind"], value["target_url"]):
                    raise WaybackStateError("failure audit job identity mismatch")
                request_key = _request_key(value["kind"], value["target_url"])
                jobs[request_key] = {**value, "job_key": key,
                                     "checkpoint_sha256": digest,
                                     "association": "scope_and_exact_request_checkpoint"}
            elif namespace == "direct_requests":
                direct[key] = value
            else:
                controls[namespace + "/" + key] = value
            commitments.append({"namespace": namespace, "key": key, "sha256": digest})
    finally:
        connection.rollback()
        connection.close()
    return jobs, direct, controls, commitments


def _empty_data(value):
    # Only explicit empty lists, including [[]], are the empty-extraction cause.
    for _ in range(3):
        if value == []:
            return True
        if not isinstance(value, list) or len(value) != 1:
            return False
        value = value[0]
    return False


def _diagnose(raw):
    """Return safe, deterministic technical labels, never provider free text."""
    metadata = raw.get("metadata", {})
    status = raw.get("response_status")
    result = {"provider": raw["provider"], "http_status": status,
              "response_received": status is not None,
              "retrieved_at": raw.get("retrieved_at"),
              "raw_body_sha256": raw["response_sha256"]}
    if status is None:
        name = metadata.get("error_type", "")
        reason = "transport_timeout" if name in {"TimeoutError", "timeout"} else "transport_failure"
    elif raw.get("final_url") != raw.get("request_url"):
        reason = "provider_response_url_mismatch"
    elif status == 429:
        reason = "http_429"
    elif status in (401, 402):
        reason = "authentication_failed" if status == 401 else "billing_blocked"
    elif metadata.get("oversized") or metadata.get("body_truncated"):
        reason = "response_body_limit_exceeded"
    elif status not in (200, 202):
        reason = "http_5xx" if status >= 500 else "http_other_failure"
    elif raw["provider"] != "outscraper_ai":
        reason = "direct_http_response_received"
    else:
        try:
            payload = json.loads(base64.b64decode(raw["body_base64"]))
        except (ValueError, UnicodeError):
            payload = None
        if not isinstance(payload, dict):
            reason = "invalid_provider_envelope"
        elif payload.get("status") in {"Pending", "InProgress", "Processing"}:
            reason = "provider_job_pending"
        elif payload.get("status") in {"Failure", "Failed", "Error"}:
            reason = "provider_job_failed"
        elif payload.get("status") != "Success":
            reason = "unrecognized_provider_status"
        elif _empty_data(payload.get("data")):
            reason = "outscraper_empty_extraction"
        else:
            try:
                target, original, timestamp = _target(metadata["kind"], metadata["target_url"], ())
                body, _, _ = _extract(metadata["kind"], target, original, timestamp, payload)
            except EvidenceValidationError as exc:
                reason = "cdx_parser_validation_failed"
                result["validator_error_sha256"] = _hash_bytes(str(exc).encode())
                if str(exc) in _PARSER_MESSAGES:
                    result["validator_reason"] = str(exc).lower().replace(" ", "_")
                if exc.__cause__ is not None and str(exc.__cause__) in _PARSER_MESSAGES:
                    result["validator_underlying_reason"] = str(exc.__cause__).lower().replace(" ", "_")
            except (KeyError, ValueError, TypeError, UnicodeError) as exc:
                message = str(exc)
                reason = ("outscraper_" + message if message in _EXTRACTION_ERRORS
                          else "outscraper_extraction_validation_failed")
                result["validator_error_sha256"] = _hash_bytes(message.encode())
            else:
                actual_hash = _hash_bytes(body)
                result["extracted_sha256"] = actual_hash
                reason = ("accepted_extraction" if metadata.get("extracted_sha256") == actual_hash
                          else "extracted_provenance_hash_mismatch")
    result["reason"] = reason
    return result


class _Reader:
    def __init__(self, directory):
        self.store = WaybackStateStore(directory)
        self.refs = set()
        self.diagnostics = {}

    def raw(self, digest):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise WaybackStateError("invalid failure audit raw reference")
        path = self.store.root / "responses" / digest[:2] / (digest + ".json")
        if path.stat().st_size > _MAX_ENVELOPE_BYTES:
            raise ValueError("failure audit raw envelope byte bound exceeded")
        value = self.store.read_record(digest)
        self.refs.add(digest)
        return value

    def diagnostic(self, digest):
        if digest not in self.diagnostics:
            self.diagnostics[digest] = {"provenance_sha256": digest, **_diagnose(self.raw(digest))}
        return self.diagnostics[digest]


def _scan_missing(reader, wanted, scope, jobs):
    """Bounded fallback when a response exists but its job checkpoint is absent."""
    if not wanted:
        return
    grouped = {}
    for index, path in enumerate(sorted((reader.store.root / "responses").glob("*/*.json"))):
        if index >= _MAX_ROWS:
            raise ValueError("failure audit fallback envelope scan bound exceeded")
        if path.stat().st_size > _MAX_ENVELOPE_BYTES:
            raise ValueError("failure audit raw envelope byte bound exceeded")
        raw = reader.store.read_record(path.stem)
        meta = raw.get("metadata", {})
        if raw["provider"] != "outscraper_ai" or meta.get("job_scope") != scope:
            continue
        target, kind = meta.get("target_url"), meta.get("kind")
        if not isinstance(target, str) or kind not in {"cdx", "replay"}:
            continue
        key = _request_key(kind, target)
        if key in wanted:
            grouped.setdefault(key, []).append((raw["retrieved_at"], path.stem, meta))
    for key, values in grouped.items():
        values.sort(key=lambda item: (item[0], item[1]))
        _, latest, meta = values[-1]
        jobs[key] = {"job_key": _job_key(scope, meta["kind"], meta["target_url"]),
                     "kind": meta["kind"], "scope": scope, "target_url": meta["target_url"],
                     "request_id": meta.get("request_id"), "state": "checkpoint_unavailable",
                     "provenance_sha256": latest, "raw_response_sha256": [item[1] for item in values],
                     "checkpoint_sha256": None, "association": "scope_and_exact_request_raw_metadata"}


def _job_row(job, reader):
    refs = list(dict.fromkeys(job.get("raw_response_sha256", []) +
                            ([job["provenance_sha256"]] if job.get("provenance_sha256") else [])))
    history = []
    for digest in refs:
        raw = reader.raw(digest)
        meta = raw.get("metadata", {})
        if raw["provider"] != "outscraper_ai":
            raise WaybackStateError("failure audit job references a different provider")
        # Account-wide authentication/billing failures may be deliberately reused
        # by the adapter. Their request is not relabelled as this candidate's job.
        account_shared = raw["response_status"] in (401, 402)
        if not account_shared and (meta.get("job_scope") != job["scope"]
                or meta.get("kind") != job["kind"] or meta.get("target_url") != job["target_url"]
                or (job.get("request_id") and meta.get("request_id") not in (None, job["request_id"]))):
            raise WaybackStateError("failure audit job raw target/scope identity mismatch")
        history.append({**reader.diagnostic(digest), "account_shared_response": account_shared})
    reason = history[-1]["reason"] if history else "fallback_response_not_retained"
    state = job.get("state")
    if state not in {"failed", "complete", "checkpoint_unavailable"}:
        reason = {"uncertain": "uncertain_submission", "submitting": "uncertain_submission",
                  "expired": "job_expired"}.get(state, "unresolved_fallback_job")
    elif not history and job.get("reason") in {"authentication_failed", "billing_blocked"}:
        reason = job["reason"]
    return {"job_key": job["job_key"], "provider_request_id": job.get("request_id"),
            "kind": job["kind"], "target_url_sha256": _hash_bytes(job["target_url"].encode()),
            "scope": job["scope"], "checkpoint_state": state, "checkpoint_sha256": job.get("checkpoint_sha256"),
            "association": job["association"], "reason": reason, "raw_attempts": history,
            "raw_response_sha256": refs, "candidate_ids": [], "collection_failure_candidate_ids": []}


def _source_row(record, source, reader, config, packet_file_sha):
    raw = reader.raw(source.provenance_sha256)
    metadata = raw.get("metadata", {})
    website = canonical_http_url(record["google_details"]["websiteUri"])
    bound = website == source.original_url
    issues = []
    try:
        target, original, timestamp = _target("replay", source.replay_url, ())
        bound = bound and canonical_http_url(original) == website and timestamp == source.captured_at.strftime("%Y%m%d%H%M%S")
        if (raw["provider"] != "outscraper_ai" or metadata.get("kind") != "replay"
                or metadata.get("target_url") != target or metadata.get("job_scope") != config["scope"]):
            issues.append("raw_source_target_scope_or_provider_mismatch")
        payload = json.loads(base64.b64decode(raw["body_base64"]))
        body, _, _ = _extract("replay", target, original, timestamp, payload)
        extracted = body.decode("utf-8").strip()
        limit = config["collector"]["max_text_chars"]
        if metadata.get("extracted_sha256") != _hash_bytes(body):
            issues.append("extracted_body_hash_mismatch")
        if (extracted[:limit] != source.archived_page_text
                or source.text_truncated != (len(extracted) > limit)
                or source.body_truncated):
            issues.append("retained_source_text_derivation_mismatch")
        if raw["response_status"] not in (200, 202) or raw["final_url"] != raw["request_url"]:
            issues.append("provider_response_binding_mismatch")
    except (ValueError, TypeError, KeyError, UnicodeError):
        issues.append("raw_source_extraction_validation_failed")
    derived = (metadata.get("ai_extracted") is True
               and metadata.get("lossless_archive_response") is False
               and metadata.get("source_fidelity") == "structural_only")
    if not derived:
        issues.append("derived_text_fidelity_disclosure_missing")
    verification = next((item for item in record["archive_verifications"]
                         if item["source_evidence_id"] == source.evidence_id), {})
    official_refs = sorted(set(_references(verification)))
    for digest in official_refs:
        reader.raw(digest)
    status = verification.get("status", "not_attempted")
    rechecked = _snapshot_check(source, verification)
    if verification.get("schema_version") == "availability_snapshot_verification_v1" and verification.get("provenance_sha256"):
        official = reader.raw(verification["provenance_sha256"])
        if official != verification.get("response"):
            raise WaybackStateError("embedded Availability response differs from retained raw envelope")
        if official.get("response_status") == 200 and official["final_url"] == official["request_url"]:
            assessed = assess_availability(base64.b64decode(official["body_base64"]), source.original_url,
                                           source.captured_at.strftime("%Y%m%d%H%M%S"))
            if assessed["status"] != status:
                issues.append("independent_snapshot_status_mismatch")
    return {"place_id": record["place_id"], "candidate_id": record["candidate_id"],
            "packet_sha256": record["packet_sha256"], "packet_file_sha256": packet_file_sha,
            "source_evidence_id": source.evidence_id, "source_provenance_sha256": source.provenance_sha256,
            "candidate_website_sha256": _hash_bytes(website.encode()), "candidate_url_binding_verified": bound,
            "retained_text_sha256": source.page_text_sha256,
            "raw_extracted_text_sha256": metadata.get("extracted_sha256"),
            "derived_extraction_not_raw_html": derived,
            "legacy_source_fidelity_label": metadata.get("source_fidelity"),
            "effective_content_kind": "outscraper_extracted_text",
            "source_derivation_verified": not issues, "issues": issues,
            "independent_verification_status": status, "independent_snapshot_check": rechecked,
            "independent_snapshot_confirmed": rechecked.startswith("verified_official"),
            "availability_verifies_text_or_opening_statement": False,
            "independent_verification_provenance_sha256": official_refs}


def build_failure_audit(root):
    """Return deterministic JSON data; partial frames are disclosed, never promoted.

    The maximum frame is 1,226, SQLite and fallback response scans are bounded at
    100,000 entries, and raw envelopes at 48 MB. Integrity failures raise. Counts
    refer to frozen candidates only; un-frozen candidates remain separately listed.
    """
    root = Path(root)
    scope = read_sealed(root / "run_scope.json")
    config = read_sealed(root / "archive_configuration.json")
    if config["scope"] != scope["run_id"]:
        raise ValueError("failure audit archive scope mismatch")
    frame = {item["place_id"]: item for item in scope["frame"]["places"]}
    if not 1 <= len(frame) <= 1226 or len(frame) != len(scope["frame"]["places"]):
        raise ValueError("failure audit candidate frame bound or uniqueness failed")
    directory = Path(config["shared_state_directory"])
    reader = _Reader(directory if directory.is_absolute() else PROJECT / directory)
    jobs, direct, controls, state_refs = _state_snapshot(reader.store.root, config["scope"])
    checkpoint_states = Counter(job["state"] for job in jobs.values())
    paths = sorted((root / "private/evidence_packets").glob("*.json"))
    if len(paths) > len(frame):
        raise ValueError("failure audit packet count exceeds frame")
    records, wanted, packet_files = [], set(), {}
    for path in paths:
        value = read_sealed(path)
        restore_packet(value)
        ident = value["place_id"]
        if (value["frame_entry"] != frame.get(ident) or path.stem != unit_key(ident)
                or value["fixed_frame_sha256"] != scope["frame"]["fixed_frame_sha256"]):
            raise ValueError("failure audit packet frame binding mismatch")
        observed = _decode(value["wayback_observed"]).archive_evidence
        for item in (*observed.cdx_pages, *observed.fetch_records):
            if item.status != "success" and item.cache_provenance not in jobs:
                wanted.add(item.cache_provenance)
        packet_files[path.relative_to(root).as_posix()] = file_sha(path)
        records.append((value, observed, packet_files[path.relative_to(root).as_posix()]))
    _scan_missing(reader, wanted, config["scope"], jobs)
    outcomes = Counter({key: 0 for key in OUTCOMES})
    website_counts, primary_counts, secondary_counts, parser_counts = Counter(), Counter(), Counter(), Counter()
    failures, candidates, sources, job_rows = [], [], [], {}
    direct_history_refs = set()
    for record, observed, packet_file_sha in records:
        for digest in set(_references(record["wayback_observed"])) | set(_references(record["archive_verifications"])):
            reader.raw(digest)
        outcome = record["archive_summary"]["outcome"]
        if outcome not in OUTCOMES:
            raise ValueError("failure audit unknown frozen outcome")
        outcomes[outcome] += 1
        returned = bool(str(record["google_details"].get("websiteUri") or "").strip())
        website_counts["returned_nonempty_website_url"] += int(returned)
        website_counts["usable_website_url"] += int(observed.query_status not in {"missing_url", "invalid_url"})
        website_counts["no_website_missing_url"] += int(observed.query_status == "missing_url")
        website_counts["no_website_invalid_url"] += int(observed.query_status == "invalid_url")
        limitations, events = set(), []
        failed_cdx = failed_replays = 0
        for kind, items in (("cdx", observed.cdx_pages), ("replay", observed.fetch_records)):
            for item in items:
                key = item.cache_provenance
                direct_refs = list(dict.fromkeys(direct.get(key, {}).get("raw_response_sha256", [])))
                direct_attempts = [reader.diagnostic(digest) for digest in direct_refs]
                direct_history_refs.update(direct_refs)
                job = jobs.get(key)
                row = None
                if job:
                    row = job_rows.setdefault(job["job_key"], _job_row(job, reader))
                    if record["candidate_id"] not in row["candidate_ids"]:
                        row["candidate_ids"].append(record["candidate_id"])
                    if outcome == "collection_failure" and record["candidate_id"] not in row["collection_failure_candidate_ids"]:
                        row["collection_failure_candidate_ids"].append(record["candidate_id"])
                if item.status == "success":
                    continue
                failed_cdx += int(kind == "cdx")
                failed_replays += int(kind == "replay")
                fallback_reason = row["reason"] if row else None
                reason = fallback_reason or ({"timeout": "transport_timeout", "http_429": "http_429",
                          "decode_error": "cdx_parser_validation_failed" if kind == "cdx" else "replay_text_decode_failed"}.get(item.status, "direct_" + item.status))
                if reason == "accepted_extraction":
                    last = reader.raw(row["raw_response_sha256"][-1])
                    meta = last["metadata"]
                    body = _extract(meta["kind"], *_target(meta["kind"], meta["target_url"], ()),
                                    json.loads(base64.b64decode(last["body_base64"])))[0]
                    reason = ("collector_response_body_limit" if len(body) > config["collector"]["max_body_bytes"]
                              else "collector_" + kind + "_validation_rejection")
                if not item.provenance_sha256:
                    limitations.add("failed_request_missing_packet_raw_reference")
                event = {"kind": kind, "request_cache_key": key, "frozen_status": item.status,
                         "packet_provenance_sha256": item.provenance_sha256, "reason": reason,
                         "job_key": row["job_key"] if row else None,
                         "fallback_association": row["association"] if row else "no_matching_fallback_ledger",
                         "fallback_raw_response_sha256": row["raw_response_sha256"] if row else [],
                         "direct_raw_attempts": direct_attempts}
                if item.provenance_sha256:
                    event["packet_raw_diagnostic"] = reader.diagnostic(item.provenance_sha256)
                events.append(event)
                limitations.add(reason)
        if failed_replays:
            limitations.add("all_replay_attempts_failed" if not observed.source_snapshots
                            else "some_replay_attempts_failed")
        if failed_cdx and observed.source_snapshots:
            limitations.add("partial_cdx_retrieval")
        if observed.cap_status not in {"not_capped", "not_applicable"}:
            limitations.add("collection_cap_" + observed.cap_status)
        base = {"place_id": record["place_id"], "candidate_id": record["candidate_id"],
                "packet_sha256": record["packet_sha256"], "packet_file_sha256": packet_file_sha,
                "frozen_outcome": outcome, "query_status": observed.query_status,
                "page_status": observed.page_status, "returned_website_url": returned,
                "failed_cdx_requests": failed_cdx, "failed_replay_requests": failed_replays,
                "secondary_limitations": sorted(limitations), "failed_requests": events}
        if outcome == "collection_failure":
            # A failed CDX page blocks capture selection. Use its first ordered
            # failure, preferring the final scope-bound fallback over direct errors.
            selected = next((event for event in events if event["kind"] == "cdx"), None)
            selected = selected or (events[0] if events else None)
            primary = selected["reason"] if selected else "collection_failure_without_request_diagnostic"
            primary_counts[primary] += 1
            if primary == "cdx_parser_validation_failed" and selected and selected["job_key"]:
                diagnostic = job_rows[selected["job_key"]]["raw_attempts"][-1]
                parser_counts[diagnostic.get("validator_underlying_reason",
                              diagnostic.get("validator_reason", "unspecified_parser_rejection"))] += 1
            failures.append({**base, "primary_reason": primary,
                             "primary_job_key": selected["job_key"] if selected else None})
        for limitation in limitations:
            secondary_counts[limitation] += 1
        candidates.append(base)
        for source in observed.source_snapshots:
            if source.retrieval_source == "outscraper_ai":
                sources.append(_source_row(record, source, reader, config, packet_file_sha))
    global_refs = sorted(set(_references(controls)))
    global_attempts = [reader.diagnostic(digest) for digest in global_refs]
    for job in jobs.values():
        if job["job_key"] not in job_rows:
            job_rows[job["job_key"]] = _job_row(job, reader)
    unresolved = [{"job_key": row["job_key"], "checkpoint_state": row["checkpoint_state"],
                   "reason": row["reason"], "provider_request_id": row["provider_request_id"],
                   "raw_response_sha256": row["raw_response_sha256"]}
                  for row in job_rows.values()
                  if row["checkpoint_state"] not in {"complete", "failed", "checkpoint_unavailable"}]
    unique_reasons = Counter(row["reason"] for row in job_rows.values() if row["reason"] != "accepted_extraction")
    failure_unique = Counter(row["reason"] for row in job_rows.values()
                             if row["collection_failure_candidate_ids"] and row["reason"] != "accepted_extraction")
    source_checks = Counter(row["independent_verification_status"] for row in sources)
    confirmed_candidates = {row["candidate_id"] for row in sources if row["independent_snapshot_confirmed"]}
    extracted_candidates = {row["candidate_id"] for row in sources}
    substantive = outcomes["direct_raw"] + outcomes["outscraper_extracted"]
    website_total = website_counts["returned_nonempty_website_url"]
    present = {record["place_id"] for record, _, _ in records}
    return {"schema_version": SCHEMA, "run_id": scope["run_id"], "frame_count": len(frame),
            "packet_count": len(records), "complete_frame": present == set(frame),
            "missing_packet_place_ids": sorted(set(frame) - present),
            "summary": {"wayback_outcomes": dict(outcomes), "website_counts": dict(website_counts),
                        "substantive_archive_candidates": substantive,
                        "substantive_coverage_among_returned_websites": substantive / website_total if website_total else None,
                        "collection_failure_primary_reason_counts": dict(sorted(primary_counts.items())),
                        "collection_failure_parser_subreason_counts": dict(sorted(parser_counts.items())),
                        "failed_outscraper_unique_job_reason_counts": dict(sorted(unique_reasons.items())),
                        "collection_failure_unique_job_reason_counts": dict(sorted(failure_unique.items())),
                        "secondary_limitation_candidate_counts": dict(sorted(secondary_counts.items())),
                        "outscraper_unique_jobs": len(job_rows), "outscraper_source_count": len(sources),
                        "outscraper_source_candidate_count": len(extracted_candidates),
                        "all_scoped_checkpoint_state_counts": dict(sorted(checkpoint_states.items())),
                        "unresolved_scoped_job_count": len(unresolved),
                        "matched_direct_request_attempt_status_counts": dict(sorted(Counter(
                            reader.diagnostic(digest)["reason"] for digest in direct_history_refs).items())),
                        "shared_global_direct_status_counts": dict(sorted(Counter(
                            row["reason"] for row in global_attempts if row["provider"] == "direct_wayback").items())),
                        "source_url_binding_failures": sum(not row["candidate_url_binding_verified"] for row in sources),
                        "source_derivation_failures": sum(not row["source_derivation_verified"] for row in sources),
                        "independent_verification_status_counts": dict(sorted(source_checks.items())),
                        "independently_confirmed_sources": sum(row["independent_snapshot_confirmed"] for row in sources),
                        "independently_confirmed_candidate_count": len(confirmed_candidates),
                        "not_independently_confirmed_candidate_count": len(extracted_candidates - confirmed_candidates)},
            "unresolved_scoped_jobs": sorted(unresolved, key=lambda row: row["job_key"]),
            "failure_rows": sorted(failures, key=lambda row: row["place_id"]),
            "candidate_rows": sorted(candidates, key=lambda row: row["place_id"]),
            "job_rows": sorted(job_rows.values(), key=lambda row: row["job_key"]),
            "source_rows": sorted(sources, key=lambda row: (row["place_id"], row["source_evidence_id"])),
            "global_provider_state_response_diagnostics": global_attempts,
            "global_route_limitation": "Global cooldown evidence is shared admission context; it does not prove a direct attempt occurred for every candidate.",
            "diagnostic_rule": "First failed CDX request, otherwise first failed replay; final same-scope exact-request fallback diagnosis takes precedence. Frozen outcomes are unchanged.",
            "counting_limit": "Candidate counts are weighted by candidates sharing a request. Unique jobs are scope-and-target job identities; provider request IDs are reported separately. Secondary causes overlap and do not sum to the outcome partition.",
            "evidence_limit": "Empty/failed extraction is not archive absence. AI extraction is not original archived HTML. Availability establishes timestamp/identity only and does not verify the text or an opening statement.",
            "archive_provenance_sha256": sorted(reader.refs),
            "input_commitments": {"packet_files": packet_files, "run_scope_sha256": file_sha(root / "run_scope.json"),
                                  "archive_configuration_sha256": file_sha(root / "archive_configuration.json"),
                                  "state_snapshot_rows_sha256": sha(state_refs), "state_snapshot_rows": state_refs,
                                  "implementation_sha256": file_sha(Path(__file__))}}
