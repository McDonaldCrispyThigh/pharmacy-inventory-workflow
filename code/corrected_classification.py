"""Resumable classification of authenticated corrected-run evidence packets.

Every received model output is sealed before parsing or validation. Only the
existing forced-choice validator and eligibility reducer determine decisions.
This module never acquires evidence, changes messages, or reads an older run.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

from corrected_evidence import MODEL, restore_packet
from corrected_preflight import require_classification_promotion
from corrected_run_store import canonical, file_sha, read_sealed, seal_json, unit_key, utcnow
from forced_choice_classifier import DECISION_SCHEMA, _usage, contains_board_data
from forced_choice_eligibility import reduce_forced_choice_eligibility
from forced_choice_models import validate_forced_choice_decision
from run_forced_choice_workflow import _location_record
from v2_downstream_models import candidate_location_from_details

SCHEMA = "nets-ai-corrected-classification-v1"
_RETRYABLE = frozenset({"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError"})


class CorrectedClassificationAbort(RuntimeError):
    """A model or API contract failed; do not promote or switch models."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__("Corrected classification aborted: " + reason)


@contextmanager
def _unit_lock(private_root, key):
    directory = private_root / "classification_locks"
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / (key + ".sqlite3"), timeout=1)
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise CorrectedClassificationAbort("candidate_already_running") from exc
        yield
    finally:
        connection.rollback()
        connection.close()


def _binding(record, attempt):
    return {"schema_version": SCHEMA, "candidate_id": record["candidate_id"],
            "place_id": record["place_id"], "packet_sha256": record["packet_sha256"],
            "messages_sha256": record["messages_sha256"], "model": MODEL, "attempt": attempt}


def _bound(event, record, attempt):
    if not isinstance(event, dict) or any(event.get(key) != value for key, value in _binding(record, attempt).items()):
        raise ValueError("classification attempt does not match the frozen packet")
    return event


def _json_metadata(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_metadata(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {key: _json_metadata(item) for key, item in vars(value).items() if not key.startswith("_")}
    return None


def _received(record, request, response):
    output = getattr(response, "output_text", None)
    output = output if isinstance(output, str) else None
    return {**_binding(record, request["attempt"]), "status": "received",
            "requested_at": request["requested_at"], "received_at": utcnow(),
            "output_text": output,
            "response_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest() if output is not None else None,
            "returned_model": _json_metadata(getattr(response, "model", None)),
            "response_id": _json_metadata(getattr(response, "id", None)),
            "system_fingerprint": _json_metadata(getattr(response, "system_fingerprint", None)),
            "usage": _json_metadata(getattr(response, "usage", None)),
            "token_usage": _usage(response), "provider_status": _json_metadata(getattr(response, "status", None))}


def _validate_received(event, packet, record, attempt):
    _bound(event, record, attempt)
    output = event.get("output_text")
    expected = hashlib.sha256(output.encode("utf-8")).hexdigest() if isinstance(output, str) else None
    if event.get("status") != "received" or event.get("response_sha256") != expected:
        raise RuntimeError("received classification output commitment mismatch")
    if event.get("returned_model") != MODEL:
        raise CorrectedClassificationAbort("returned_model_mismatch")
    if not isinstance(output, str):
        raise ValueError("missing structured model output")
    return validate_forced_choice_decision(packet, json.loads(output))


def _validation_event(record, attempt, response_path, *, status, decision=None, reason=None):
    return {**_binding(record, attempt), "status": status, "reason": reason,
            "received_response_file_sha256": file_sha(response_path),
            "decision_sha256": decision.decision_sha256 if decision is not None else None,
            "recorded_at": utcnow()}


def _persist_validation(path, event):
    if path.exists():
        prior = read_sealed(path)
        if {key: value for key, value in prior.items() if key != "recorded_at"} != {
                key: value for key, value in event.items() if key != "recorded_at"}:
            raise ValueError("retained classification validation outcome changed")
    else:
        seal_json(path, event)


def _provider_metadata(record, event, decision):
    return {"status": "success", "requested_model": MODEL, "returned_model": event["returned_model"],
            "response_id": event["response_id"], "requested_at": event["requested_at"],
            "received_at": event["received_at"], "system_fingerprint": event["system_fingerprint"],
            "token_usage": event["token_usage"], "usage": event["usage"],
            "response_sha256": event["response_sha256"], "attempt_count": event["attempt"],
            "schema_version": record["classifier_schema_version"],
            "prompt_template_version": record["base_prompt_version"],
            "prompt_template_sha256": record["base_prompt_template_sha256"],
            "classifier_contract_sha256": record["classifier_contract_sha256"],
            "messages_sha256": record["messages_sha256"], "packet_sha256": record["packet_sha256"],
            "evidence_policy_version": record["evidence_policy_version"],
            "decision_sha256": decision.decision_sha256, "store": False}


def _completed(record, location, event, decision, response_path, private_root, completed_at):
    eligibility = reduce_forced_choice_eligibility(decision, place_id=record["place_id"],
                                                   candidate_id=record["candidate_id"])
    return {"schema_version": SCHEMA, "status": "completed", "candidate_id": record["candidate_id"],
            "place_id": record["place_id"], "location": location,
            "decision": decision.to_public_dict(), "decision_sha256": decision.decision_sha256,
            "packet_sha256": record["packet_sha256"], "messages_sha256": record["messages_sha256"],
            "provider_metadata": _provider_metadata(record, event, decision),
            "model": MODEL, "returned_model": event["returned_model"],
            "eligibility": eligibility.to_public_dict(), "attempt_count": event["attempt"],
            "received_response_path": response_path.relative_to(private_root).as_posix(),
            "received_response_file_sha256": file_sha(response_path), "completed_at": completed_at}


def _api_error(exc):
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    body = getattr(exc, "body", None)
    if code is None and isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code")
    code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", code) else None
    http_status = getattr(exc, "status_code", None)
    http_status = http_status if type(http_status) is int else None
    if http_status == 404 or name == "NotFoundError" or code in {"model_not_found", "model_not_available", "invalid_model"}:
        reason, retryable = "requested_model_unavailable", False
    elif (http_status in {400, 401, 403, 422}
          or code in {"insufficient_quota", "billing_hard_limit_reached", "invalid_api_key"}
          or name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError", "UnprocessableEntityError"}):
        reason, retryable = "api_configuration_failure", False
    else:
        reason, retryable = "transient_api_failure" if name in _RETRYABLE else "nonretryable_api_failure", name in _RETRYABLE
    return {"status": "api_error", "error_type": name, "http_status": http_status,
            "error_code": code, "reason": reason, "retryable": retryable, "recorded_at": utcnow()}


def _attempts(attempt_root, record):
    grouped = {}
    for path in sorted(attempt_root.glob("*.json")) if attempt_root.exists() else ():
        match = re.fullmatch(r"(\d{2})\.(request|response|validation|error)\.json", path.name)
        if match is None:
            raise ValueError("unexpected classification attempt filename")
        attempt = int(match[1])
        event = _bound(read_sealed(path), record, attempt)
        grouped.setdefault(attempt, {})[match[2]] = (path, event)
    if sorted(grouped) != list(range(1, len(grouped) + 1)) or any("request" not in events for events in grouped.values()):
        raise ValueError("classification attempt sequence is incomplete")
    for events in grouped.values():
        request = events["request"][1]
        if request.get("status") != "requested" or request.get("decision_schema_sha256") != record["decision_schema_sha256"]:
            raise ValueError("classification request contract changed")
        if "response" in events and "error" in events:
            raise ValueError("classification attempt has conflicting response and API error")
    return grouped


def classify_frozen_candidate(packet_record, private_root, client, *, max_attempts=10):
    """Use exact frozen messages and seal a completed or technical-unresolved unit.

    Caller authenticates the complete 1,226-packet preflight before entering this
    function and configures the SDK with max_retries=0. The durable attempt limit
    counts responses.create invocations; ordinary uncertainty never causes retry.
    """
    require_classification_promotion(Path(private_root).parent)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError("classification attempts must be in [1,10]")
    record = json.loads(canonical(packet_record))
    packet = restore_packet(record)
    if record["model"] != MODEL:
        raise CorrectedClassificationAbort("requested_model_not_pinned")
    if record["decision_schema"] != DECISION_SCHEMA or contains_board_data(record["messages"]):
        raise ValueError("frozen classifier input violates its schema or Board boundary")
    location = _location_record(candidate_location_from_details(
        record["google_details"], packet.candidate.query_provenance, packet.candidate.retrieved_at,
    ))
    if location["place_id"] != record["place_id"] or location["candidate_id"] != record["candidate_id"]:
        raise ValueError("corrected Details location differs from its frozen candidate")
    private_root = Path(private_root)
    key = unit_key(record["place_id"])
    unit_path = private_root / "classifications" / (key + ".json")
    attempt_root = private_root / "classification_attempts" / key
    with _unit_lock(private_root, key):
        previous = _attempts(attempt_root, record)
        completed = read_sealed(unit_path) if unit_path.exists() else None
        if completed and any(completed.get(name) != record[name] for name in ("candidate_id", "place_id", "packet_sha256", "messages_sha256", "model")):
            raise ValueError("completed classification belongs to a different frozen packet")
        # Revalidate every received output before considering a new call. A
        # successful raw receipt survives interruption before validation or seal.
        for attempt, events in previous.items():
            if "error" in events:
                error = events["error"][1]
                if error.get("retryable") is not True:
                    raise CorrectedClassificationAbort(error["reason"])
                continue
            if "response" not in events:
                continue  # Interrupted request consumes one bounded attempt.
            response_path, event = events["response"]
            validation_path = attempt_root / f"{attempt:02}.validation.json"
            try:
                decision = _validate_received(event, packet, record, attempt)
            except CorrectedClassificationAbort as exc:
                _persist_validation(validation_path, _validation_event(
                    record, attempt, response_path, status="aborted", reason=exc.reason))
                raise
            except (ValueError, TypeError):
                _persist_validation(validation_path, _validation_event(
                    record, attempt, response_path, status="invalid_output", reason="forced_choice_validation_failed"))
                continue
            _persist_validation(validation_path, _validation_event(record, attempt, response_path, status="validated", decision=decision))
            result = _completed(record, location, event, decision, response_path, private_root,
                                completed["completed_at"] if completed else utcnow())
            if completed is not None and completed != result:
                raise ValueError("completed classification differs from its validated raw response")
            seal_json(unit_path, result)
            return result
        if completed is not None:
            if completed.get("status") != "technical_unresolved" or completed.get("attempt_count") != len(previous):
                raise ValueError("completed classification lacks its successful raw response")
            expected_reducer = reduce_forced_choice_eligibility(None, place_id=record["place_id"], candidate_id=record["candidate_id"]).to_public_dict()
            if completed.get("decision") is not None or completed.get("decision_sha256") is not None or completed.get("eligibility") != expected_reducer or completed.get("location") != location:
                raise ValueError("technical unresolved unit does not match its evidence and reducer")
            return completed
        for attempt in range(len(previous) + 1, max_attempts + 1):
            require_classification_promotion(private_root.parent)
            request = {**_binding(record, attempt), "status": "requested", "requested_at": utcnow(),
                       "decision_schema_sha256": record["decision_schema_sha256"], "store": False}
            request_path = attempt_root / f"{attempt:02}.request.json"
            response_path = attempt_root / f"{attempt:02}.response.json"
            validation_path = attempt_root / f"{attempt:02}.validation.json"
            seal_json(request_path, request)
            try:
                response = client.responses.create(
                    model=MODEL, input=record["messages"],
                    text={"format": {"type": "json_schema", "name": "nets_ai_forced_choice_decision",
                                     "strict": True, "schema": DECISION_SCHEMA}},
                    store=False,
                )
            except Exception as exc:
                error = {**_binding(record, attempt), **_api_error(exc)}
                seal_json(attempt_root / f"{attempt:02}.error.json", error)
                if not error["retryable"]:
                    raise CorrectedClassificationAbort(error["reason"]) from None
            else:
                event = _received(record, request, response)
                seal_json(response_path, event)
                try:
                    decision = _validate_received(event, packet, record, attempt)
                except CorrectedClassificationAbort as exc:
                    _persist_validation(validation_path, _validation_event(
                        record, attempt, response_path, status="aborted", reason=exc.reason))
                    raise
                except (ValueError, TypeError):
                    _persist_validation(validation_path, _validation_event(
                        record, attempt, response_path, status="invalid_output", reason="forced_choice_validation_failed"))
                else:
                    _persist_validation(validation_path, _validation_event(record, attempt, response_path, status="validated", decision=decision))
                    result = _completed(record, location, event, decision, response_path, private_root, utcnow())
                    seal_json(unit_path, result)
                    return result
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 30))
        result = {"schema_version": SCHEMA, "status": "technical_unresolved", "candidate_id": record["candidate_id"],
                  "place_id": record["place_id"], "location": location, "decision": None, "decision_sha256": None,
                  "packet_sha256": record["packet_sha256"], "messages_sha256": record["messages_sha256"],
                  "provider_metadata": None, "model": MODEL, "returned_model": None,
                  "eligibility": reduce_forced_choice_eligibility(None, place_id=record["place_id"],
                                                                   candidate_id=record["candidate_id"]).to_public_dict(),
                  "attempt_count": len(_attempts(attempt_root, record)),
                  "failure_reason": "technical_attempts_exhausted", "completed_at": utcnow()}
        seal_json(unit_path, result)
        return result


__all__ = ["classify_frozen_candidate", "CorrectedClassificationAbort"]
