"""Durable private inputs for the explicitly authorized corrected empirical run.

No search capability is exposed. Review authors/profile links are omitted; all
returned review text, order, dates and ratings are retained for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from skills.google_maps import GooglePlacesNewDetailsAgent
from v2_classifier import build_candidate_facts, build_review_evidence
from v2_evidence_models import QueryProvenanceV2


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal_json(path: Path, payload: Any) -> str:
    """Never replace an already frozen record, even with a new valid payload."""
    if path.exists():
        if read_sealed(path) != payload:
            raise ValueError("immutable record differs")
        return file_sha(path)
    atomic_json(path, {"schema": "nets-ai-corrected-sealed-json-v1", "sha256": sha(payload), "payload": payload})
    return file_sha(path)


def read_sealed(path: Path) -> Any:
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {"schema", "sha256", "payload"} or record["schema"] != "nets-ai-corrected-sealed-json-v1" or sha(record["payload"]) != record["sha256"]:
        raise ValueError("frozen record hash/schema mismatch")
    return record["payload"]


def unit_key(place_id: str) -> str:
    return hashlib.sha256(place_id.encode("utf-8")).hexdigest()


def validate_frame_ids(ids: list[str], old_ids: list[str]) -> None:
    if len(ids) != 1226 or len(old_ids) != 1226 or len(set(ids)) != 1226 or len(set(old_ids)) != 1226 or set(ids) != set(old_ids):
        raise ValueError("candidate frame must equal the exact frozen 1226 Place IDs")


def run_resumable_queue(items, work, *, max_active=24, clock=time.time, sleep=time.sleep, on_event=None):
    """Fairly poll a bounded set of accepted jobs without serial 20-second waits.

    Provider-level durable state owns task IDs and retries. This queue only
    schedules candidates, and never treats an unresolved submission as absence.
    """
    import heapq
    from wayback_state import WaybackDeferred
    if max_active < 1:
        raise ValueError("max_active must be positive")
    source = iter(items)
    queue = []
    serial = 0
    results = []
    def admit():
        nonlocal serial
        try:
            item = next(source)
        except StopIteration:
            return
        heapq.heappush(queue, (clock(), serial, item))
        serial += 1
    for _ in range(max_active):
        admit()
    while queue:
        due, _, item = heapq.heappop(queue)
        while due > clock():
            sleep(min(due - clock(), 30))
        try:
            result = work(item)
        except WaybackDeferred as exc:
            if on_event:
                on_event(item, {"status": "pending", **exc.to_dict()})
            if exc.retry_at is None:
                raise
            heapq.heappush(queue, (max(clock() + 1, exc.retry_at), serial, item))
            serial += 1
        else:
            results.append(result)
            if on_event:
                on_event(item, {"status": "complete", "completed": len(results)})
            admit()
    return results


def sanitize_details(details: dict) -> dict:
    retained = json.loads(json.dumps(details))
    for review in retained.get("reviews", []):
        if isinstance(review, dict):
            for key in ("authorAttribution", "name", "googleMapsUri", "flagContentUri"):
                review.pop(key, None)
    return retained


def acquire_details(private_root: Path, frame_entry: dict, api_key: str, transport: Any, *, max_attempts: int = 10, sleep=time.sleep) -> dict:
    place_id = frame_entry["place_id"]
    key = unit_key(place_id)
    path = private_root / "details" / (key + ".json")
    if path.exists():
        record = read_sealed(path)
        if record["place_id"] != place_id or record["frame_entry"] != frame_entry:
            raise ValueError("saved Details frame binding mismatch")
        return record
    attempt_root = private_root / "details_attempts" / key
    prior = sorted(attempt_root.glob("*.json")) if attempt_root.exists() else []
    # Recover a successful received response after interruption before publishing
    # the final Details unit. Never reacquire a success merely to rebuild a file.
    for p in prior:
        event = read_sealed(p)
        if event.get("status") == "success":
            seal_json(path, event)
            return event
    provenance = QueryProvenanceV2(frame_entry["query_definition_id"], frame_entry["source_zip"], "places_search")
    for attempt in range(len(prior) + 1, max_attempts + 1):
        requested = utcnow()
        event = {"place_id": place_id, "frame_entry": frame_entry, "attempt": attempt, "requested_at": requested, "endpoint": f"{GooglePlacesNewDetailsAgent.endpoint}/{place_id}", "field_mask": GooglePlacesNewDetailsAgent.field_mask, "status": "failure"}
        try:
            status, body = transport.get(event["endpoint"], {"X-Goog-Api-Key": api_key, "X-Goog-FieldMask": GooglePlacesNewDetailsAgent.field_mask, "Accept": "application/json"}, 30.0)
            event.update(http_status=status, retrieved_at=utcnow(), received_body_sha256=hashlib.sha256(body).hexdigest(), received_bytes=len(body))
            if status != 200 or len(body) > 2_000_000:
                raise ValueError("provider_http_or_response_size")
            details = json.loads(body.decode("utf-8"))
            if not isinstance(details, dict) or details.get("id") != place_id:
                raise ValueError("provider_identity_mismatch")
            event["details"] = sanitize_details(details)
            retrieved = datetime.fromisoformat(event["retrieved_at"].replace("Z", "+00:00"))
            build_candidate_facts(event["details"], provenance, retrieved)
            reviews = build_review_evidence(event["details"], retrieved)
            if reviews.status != "success":
                raise ValueError("review_packet_invalid")
            event.update(status="success", retained_details_sha256=sha(event["details"]), review_count=reviews.returned_count, privacy_transform="omit_review_author_and_profile_fields_only")
        except Exception as exc:
            # Exception text can contain provider content or request headers.
            event["error_type"] = type(exc).__name__
            event.setdefault("retrieved_at", utcnow())
        seal_json(attempt_root / f"{attempt:02}.json", event)
        if event["status"] == "success":
            seal_json(path, event)
            return event
        if event.get("http_status") in {400, 401, 403, 404}:
            break
        if attempt < max_attempts:
            sleep(min(2 ** (attempt - 1), 30))
    failure = {"place_id": place_id, "status": "failure", "attempts": len(list(attempt_root.glob('*.json'))), "recorded_at": utcnow()}
    atomic_json(private_root / "details_failures" / (key + ".json"), failure)
    return failure
