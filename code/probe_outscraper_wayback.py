"""Bounded Outscraper compatibility experiment using two public Wayback URLs.

This is not a workflow adapter. AI-extracted content never qualifies as lossless
Archive evidence, even when its reported fields pass structural checks.
No raw responses, credentials, or empirical inputs are written by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from v2_evidence_models import parse_cdx_response_v2

API_ROOT = "https://api.outscraper.cloud"
CONTROL = "http://example.com/"
CAPTURE = "20130919044612"  # Public capture shown in the official Wayback API overview.
CDX_URL = "https://web.archive.org/cdx/search/cdx?" + urlencode([
    ("url", CONTROL), ("output", "json"),
    ("fl", "timestamp,original,mimetype,statuscode,digest,length"),
    ("matchType", "exact"), ("filter", "statuscode:200"),
    ("filter", "mimetype:text/html"), ("limit", "1"), ("gzip", "false"),
])
REPLAY_URL = f"https://web.archive.org/web/{CAPTURE}id_/{CONTROL}"
TARGETS = {"cdx": CDX_URL, "replay": REPLAY_URL}
MAX_BYTES = 1_000_000


def extraction_params(stage):
    if stage == "cdx":
        prompt = ("Read only the supplied URL. Put its complete, verbatim CDX JSON response "
                  "in cdx_json. Do not infer, summarize, reconstruct, or browse other URLs. "
                  "If unavailable, return null and put the observed error in retrieval_error.")
        fields = ["cdx_json", "retrieval_error"]
    else:
        prompt = ("Read only the supplied archived URL. Extract the exact visible page text "
                  "as page_text, the final URL as final_url, and its 14-digit archive "
                  "timestamp as archive_timestamp. Do not substitute the live website, "
                  "infer missing content, or follow page links. Return null for unavailable "
                  "fields and put the observed error in retrieval_error.")
        fields = ["page_text", "final_url", "archive_timestamp", "retrieval_error"]
    # The live GET endpoint rejects the documented JSON-string schema parameter
    # with "Not a valid mapping type". Its documented prompt-only mode works
    # without relying on that mapping parser; retain the same requested fields.
    prompt += " Return an object with only these fields: " + ", ".join(fields) + "."
    return (("query", TARGETS[stage]), ("prompt", prompt),
            ("limit", "1"), ("async", "true"))


def load_api_key(env_file=ROOT / ".env"):
    # Explicit provider names only; never try unrelated repository credentials.
    names = ("OUTSCRAPER_API_KEY", "OUTSCRAPER_API_TOKEN")
    values = dotenv_values(env_file, interpolate=False) if env_file.is_file() else {}
    for source in (os.environ, values):
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                value = value.strip()
                if not value.isascii() or any(ord(char) < 33 or ord(char) == 127 for char in value):
                    raise ValueError("invalid Outscraper credential format")
                return value
    return None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True, repr=False)
class Response:
    status: int
    body: bytes


class Transport:
    def __init__(self, key, timeout):
        self.key, self.timeout = key, timeout
        self.opener = build_opener(_NoRedirects())

    def get(self, path, params=()):
        if path != "/ai-scraper" and not re.fullmatch(r"/requests/[A-Za-z0-9_-]{1,128}", path):
            raise ValueError("invalid Outscraper endpoint")
        headers = {"Accept": "application/json", "User-Agent": "NETS-AI-public-control-probe/1.0"}
        if self.key:
            headers["X-API-KEY"] = self.key
        request = Request(API_ROOT + path + ("?" + urlencode(params) if params else ""), headers=headers)
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except HTTPError as exc:
            response = exc
        with response:
            return Response(response.code, response.read(MAX_BYTES + 1))


def assess_extraction(stage, data):
    # Provider schemas sometimes wrap a per-query result in a singleton list.
    for _ in range(2):
        if isinstance(data, list) and len(data) == 1:
            data = data[0]
    if not isinstance(data, dict):
        return {"status": "unexpected_data_shape"}
    result = {"status": "extracted_data_returned", "ai_extracted": True,
              "reported_retrieval_error": bool(data.get("retrieval_error"))}
    if stage == "cdx":
        value = data.get("cdx_json")
        result["reported_cdx_schema_valid"] = False
        result["reported_cdx_valid_after_default_port_normalization"] = False
        if isinstance(value, str):
            try:
                rows, _ = parse_cdx_response_v2(value.encode("utf-8"))
                result.update(reported_cdx_schema_valid=True, reported_capture_count=len(rows),
                              reported_targets_match=all(row.original_url == CONTROL for row in rows))
            except ValueError as exc:
                result["reported_cdx_validation_error"] = str(exc)
                # Diagnose a reversible URL-format mismatch separately. Never
                # relabel transformed bytes as the provider's raw CDX response.
                try:
                    table = json.loads(value)
                    column = table[0].index("original")
                    changed = 0
                    for row in table[1:]:
                        url = urlparse(row[column])
                        if ((url.scheme == "http" and url.port == 80)
                                or (url.scheme == "https" and url.port == 443)):
                            row[column] = url._replace(netloc=url.netloc.rsplit(":", 1)[0]).geturl()
                            changed += 1
                    if changed:
                        normalized_rows, _ = parse_cdx_response_v2(json.dumps(table).encode("utf-8"))
                        result.update(reported_cdx_valid_after_default_port_normalization=True,
                                      normalized_capture_count=len(normalized_rows),
                                      normalized_targets_match=all(row.original_url == CONTROL for row in normalized_rows))
                except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                    pass
    else:
        text = data.get("page_text")
        timestamp = data.get("archive_timestamp")
        result.update(reported_text_nonempty=isinstance(text, str) and bool(text.strip()),
                      reported_archive_timestamp_type=type(timestamp).__name__,
                      reported_archive_timestamp_matches=type(timestamp) in (str, int) and str(timestamp) == CAPTURE,
                      reported_final_url_matches=data.get("final_url") == REPLAY_URL)
        if isinstance(text, str):
            result["reported_text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            result["reported_text_chars"] = len(text)
    return result


def run_probe(transport, *, max_polls=3, sleep=time.sleep):
    report = {"schema_version": "outscraper_wayback_probe_v1", "status": "probe_complete",
              "checked_at": datetime.now(timezone.utc).isoformat(), "external_requests": 0,
              "max_extraction_tasks": 2, "study_candidate_requests": 0,
              "workflow_replacement_verified": False,
              "replacement_limitation": "AI extraction does not establish lossless CDX/replay provenance",
              "stages": {}}
    for stage in TARGETS:
        path, params = "/ai-scraper", extraction_params(stage)
        result = {"status": "not_attempted", "target_url": TARGETS[stage]}
        report["stages"][stage] = result
        for attempt in range(max_polls + 1):
            if attempt:
                sleep(20)
            report["external_requests"] += 1
            try:
                response = transport.get(path, params)
            except (OSError, URLError) as exc:
                cause = exc.reason if isinstance(exc, URLError) else exc
                result.update(status="transport_error", error_type=type(cause).__name__)
                report["status"] = "transport_error"
                return report
            result["http_status"] = response.status
            result["response_sha256"] = hashlib.sha256(response.body).hexdigest()
            result["response_bytes"] = len(response.body)
            if response.status not in (200, 202):
                result["status"] = {401: "authentication_failed", 402: "billing_blocked",
                                    429: "rate_limited"}.get(response.status, "http_error")
                report["status"] = result["status"]
                return report
            if len(response.body) > MAX_BYTES:
                result["status"] = "oversized_response"
                break
            try:
                payload = json.loads(response.body)
                if not isinstance(payload, dict):
                    raise ValueError("invalid envelope")
            except (ValueError, UnicodeError):
                result["status"] = "invalid_response"
                break
            request_id = payload.get("id")
            safe_id = isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id)
            if safe_id:
                result["request_id"] = request_id
            if payload.get("status") == "Success" and "data" in payload:
                result.update(assess_extraction(stage, payload["data"]))
                break
            if payload.get("status") in ("Pending", "InProgress", "Processing") and safe_id:
                # Never follow the untrusted results_location, especially with a credential.
                path, params = "/requests/" + request_id, ()
                result["status"] = "pending"
                continue
            result["status"] = "provider_error"
            break
    if any(row["status"] != "extracted_data_returned" for row in report["stages"].values()):
        report["status"] = "probe_incomplete"
    return report


def main(argv=None, *, stdout=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--auth-probe", action="store_true",
                        help="Allow one unauthenticated CDX task request when the key is missing")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--max-polls", type=int, default=3)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 60 or not 0 <= args.max_polls <= 6:
        parser.error("timeout must be in (0,60]; max-polls must be in [0,6]")
    stdout = stdout or sys.stdout
    if not args.live:
        report = {"status": "dry_run", "external_requests": 0, "targets": TARGETS,
                  "max_extraction_tasks": 2, "workflow_replacement_verified": False}
    else:
        key = load_api_key()
        if key is None and not args.auth_probe:
            report = {"status": "blocked_missing_credentials", "external_requests": 0,
                      "required_variable": "OUTSCRAPER_API_KEY", "workflow_replacement_verified": False}
        elif key is None:
            # This branch intentionally checks authentication only, never polls or
            # submits a second task even if the service accepts an anonymous request.
            transport = Transport(None, args.timeout)
            try:
                response = transport.get("/ai-scraper", extraction_params("cdx"))
                report = {"status": "authentication_failed" if response.status == 401 else "auth_probe_only",
                          "credential_configured": False, "http_status": response.status,
                          "external_requests": 1, "wayback_retrieval_verified": False,
                          "workflow_replacement_verified": False,
                          "response_bytes": len(response.body),
                          "response_sha256": hashlib.sha256(response.body).hexdigest()}
            except (OSError, URLError) as exc:
                cause = exc.reason if isinstance(exc, URLError) else exc
                report = {"status": "transport_error", "error_type": type(cause).__name__,
                          "external_requests": 1, "workflow_replacement_verified": False}
        else:
            report = run_probe(Transport(key, args.timeout), max_polls=args.max_polls)
    stdout.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return 0 if report["status"] in ("dry_run", "probe_complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
