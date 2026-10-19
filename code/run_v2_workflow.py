"""Command-line entry point for the user-authorized thesis-stage v2 workflow.

Dry-run mode constructs no network client and writes nothing.  Live-mode provider
objects are lazy.  Live thesis execution is bound to ``AGENTS.md`` and emits a
non-blocking compliance warning; it does not claim provider-issued permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "code"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from skills.google_maps import (  # noqa: E402
    GooglePlacesNewProvider,
    PlacesNewDetailsError,
)
from skills.wayback_agent import (  # noqa: E402
    WaybackHTTPResponse,
    WaybackServiceUnavailable,
)
from wayback_runtime import (  # noqa: E402
    LazyWaybackCollector, deferred_report, require_research_wayback_service,
)
from wayback_state import WaybackDeferred  # noqa: E402
from study_design import TARGET_ZIPS  # noqa: E402
from v2_classifier import (  # noqa: E402
    DECISION_SCHEMA,
    ClassifierResponseError,
    classify_v2,
)
from v2_delivery import (  # noqa: E402
    DeliveryInputsV2,
    authenticate_live_delivery_provenance,
    load_local_delivery_context,
    run_v2_delivery,
)
from v2_workflow import (  # noqa: E402
    AuthorizationGateError,
    V2Workflow,
    V2WorkflowConfig,
)


# Legacy external-permission pins remain available for non-thesis callers.  The CLI's
# thesis-stage path does not require either pin.
APPROVED_PERMISSION_SHA256: str | None = None
APPROVED_PILOT_PROMOTION_SHA256: str | None = None
COMPLIANCE_WARNING = (
    "Thesis-stage execution is user-authorized by AGENTS.md only; it does not "
    "claim provider-issued permission or authorize publication/repository release."
)


class _LazyFullDelivery:
    """Load authenticated local downstream inputs only after the live gate."""

    def __init__(
        self,
        input_manifest: Path,
        promotion_file: Path,
        target_zips: tuple[str, ...],
        model: str,
        pilot_delivery_root: Path | None = None,
        approved_permission_sha256: str | None = None,
        approved_promotion_sha256: str | None = None,
    ) -> None:
        self.input_manifest = input_manifest
        self.promotion_file = promotion_file
        self.pilot_delivery_root = pilot_delivery_root
        self.target_zips = tuple(target_zips)
        self.model = model
        self.approved_permission_sha256 = approved_permission_sha256
        self.approved_promotion_sha256 = approved_promotion_sha256
        self._context = None

    def preflight(self) -> None:
        if self._context is None:
            try:
                promotion = json.loads(
                    self.promotion_file.read_text(encoding="utf-8")
                )
                relative = Path(promotion["pilot_manifest_file"])
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError,
                    TypeError) as exc:
                raise ValueError("full delivery promotion is invalid") from exc
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("promoted pilot path is unsafe")
            pilot_manifest = (
                self.promotion_file.resolve().parent / relative
            ).resolve()
            promoted_root = pilot_manifest.parent
            if (
                pilot_manifest.name != "run_manifest.json"
                or self.pilot_delivery_root is not None
                and self.pilot_delivery_root.resolve() != promoted_root
            ):
                raise ValueError("CLI pilot package differs from promoted pilot")
            self._context = load_local_delivery_context(
                self.input_manifest,
                promoted_root,
                self.target_zips,
                project_root=ROOT,
                pilot_expectations={
                    "authorization_file_sha256":
                    promotion["authorization_file_sha256"],
                    "written_permission_sha256":
                    promotion["written_permission_sha256"],
                    "model": self.model,
                    "run_id": promotion["pilot_run_id"],
                    "manifest_sha256": promotion["pilot_manifest_sha256"],
                    "semantic_sha256":
                    promotion["pilot_semantic_manifest_sha256"],
                },
            )

    def __call__(self, candidates, config, output_root):
        self.preflight()
        if tuple(config.zips) != self.target_zips:
            raise ValueError("full delivery ZIP frame changed after preflight")
        context = self._context
        live_provenance = authenticate_live_delivery_provenance(
            output_root.parent,
            config.run_id,
            expected_authorization_file_sha256=hashlib.sha256(
                config.authorization_file.read_bytes()
            ).hexdigest(),
            expected_written_permission_sha256=self.approved_permission_sha256,
            expected_promotion_file_sha256=self.approved_promotion_sha256,
            expected_candidates=tuple(candidates),
        )
        inputs = DeliveryInputsV2(
            candidates=tuple(candidates),
            board=context.board,
            target_zips=self.target_zips,
            origins=context.origins,
            ablation_rows=context.ablation_rows,
            run_metadata={
                "run_id": config.run_id,
                "model": config.model,
                "executed_at": datetime.now(timezone.utc).isoformat()
                .replace("+00:00", "Z"),
                "provider_version": "google-places-api-new",
                "evidence_version": "v2.1-details-reviews-wayback",
            },
            empirical_v2_results_exist=True,
            source_label="authorized_live_full",
            source_hashes=context.source_hashes,
            live_provenance=live_provenance,
        )
        return run_v2_delivery(inputs, output_root)


class _GoogleOnlyRedirects(HTTPRedirectHandler):
    """Do not forward the Google API key outside the exact API host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if (parsed.scheme != "https"
                or (parsed.hostname or "").casefold()
                != "places.googleapis.com"):
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _GoogleHTTPTransport:
    """Small bounded urllib transport; it never logs headers, bodies, or URLs."""

    def __init__(self) -> None:
        self._opener = build_opener(_GoogleOnlyRedirects())

    def _send(self, request: Request, timeout: float) -> tuple[int, bytes]:
        try:
            with self._opener.open(request, timeout=timeout) as response:
                body = response.read(2_000_001)
                return int(response.status), body
        except HTTPError as exc:
            return int(exc.code), exc.read(2_000_001)

    def get(self, url: str, headers: dict[str, str], timeout: float
            ) -> tuple[int, bytes]:
        return self._send(Request(url, headers=headers, method="GET"), timeout)

    def post(self, url: str, headers: dict[str, str], body: bytes,
             timeout: float) -> tuple[int, bytes]:
        return self._send(
            Request(url, data=body, headers=headers, method="POST"), timeout,
        )


class _ArchiveOnlyRedirects(HTTPRedirectHandler):
    """Disable automatic redirects so the collector validates each Location."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _WaybackArchiveTransport:
    """Bounded Internet Archive transport with off-archive redirects disabled."""

    def __init__(self) -> None:
        self._opener = build_opener(_ArchiveOnlyRedirects())

    def get(self, url: str, params: tuple[tuple[str, str], ...],
            timeout: float, *, follow_redirects: bool) -> WaybackHTTPResponse:
        if follow_redirects is not False:
            raise ValueError("Wayback redirects must be handled by the collector")
        parsed = urlparse(url)
        if (parsed.scheme != "https"
                or (parsed.hostname or "").casefold() != "web.archive.org"):
            raise ValueError("Wayback transport accepts only the archive host")
        target = url + (("&" if parsed.query else "?") + urlencode(params)
                        if params else "")
        request = Request(
            target,
            headers={"Accept": "application/json,text/html;q=0.9",
                     "User-Agent": "NETS-AI-honors-thesis-v2/1.0"},
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                body = response.read(2_000_001)
                return WaybackHTTPResponse(
                    int(response.status), dict(response.headers.items()), body,
                    response.geturl(),
                )
        except HTTPError as exc:
            return WaybackHTTPResponse(
                int(exc.code), dict(exc.headers.items()), exc.read(2_000_001),
                exc.geturl(),
            )


class _LazyGoogleProvider:
    max_details_attempts = 10

    def __init__(self) -> None:
        self._provider: GooglePlacesNewProvider | None = None

    def _get(self) -> GooglePlacesNewProvider:
        if self._provider is None:
            key = os.getenv("GOOGLE_MAPS_API_KEY")
            if not key:
                raise RuntimeError("Google Places credential is unavailable")
            self._provider = GooglePlacesNewProvider(key, _GoogleHTTPTransport())
        return self._provider

    def list_candidates(self, **kwargs):
        return self._get().list_candidates(**kwargs)

    def get_place_details(self, place_id: str):
        for attempt in range(1, self.max_details_attempts + 1):
            try:
                return self._get().get_place_details(place_id)
            except PlacesNewDetailsError:
                if attempt == self.max_details_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 30))
        raise AssertionError("Places Details retry loop did not terminate")

    def preflight(self) -> None:
        self._get()


class _LazyClassifier:
    max_validation_attempts = 10
    retryable_exception_names = frozenset({
        "APIConnectionError", "APITimeoutError", "InternalServerError",
        "RateLimitError",
    })

    def __init__(self) -> None:
        self._client: Any | None = None

    def __call__(self, packet, model: str):
        for attempt in range(1, self.max_validation_attempts + 1):
            try:
                return classify_v2(packet, self._get(), model)
            except ClassifierResponseError:
                if attempt == self.max_validation_attempts:
                    raise
            except Exception as exc:
                if (
                    type(exc).__name__ not in self.retryable_exception_names
                    or attempt == self.max_validation_attempts
                ):
                    raise
                time.sleep(min(2 ** (attempt - 1), 30))
        raise AssertionError("classifier retry loop did not terminate")

    def _get(self):
        if self._client is None:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OpenAI credential is unavailable")
            from openai import OpenAI
            self._client = OpenAI(api_key=key)
        return self._client

    def evaluate_messages(self, messages, model: str):
        """Evaluate one already condition-bounded pilot prompt without persistence."""
        requested_at = datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        response = self._get().responses.create(
            model=model,
            input=messages,
            text={"format": {"type": "json_schema",
                             "name": "nets_ai_v2_decision",
                             "strict": True, "schema": DECISION_SCHEMA}},
            temperature=0,
            store=False,
        )
        output = getattr(response, "output_text", None)
        if not isinstance(output, str):
            raise RuntimeError("OpenAI pilot output is unavailable")
        value = json.loads(output)
        if not isinstance(value, dict):
            raise RuntimeError("OpenAI pilot output is not an object")
        usage = getattr(response, "usage", None)

        def usage_value(name: str):
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None) if usage is not None else None

        return value, {
            "returned_model": getattr(response, "model", None),
            "requested_at": requested_at,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "token_usage": {
                "input_tokens": usage_value("input_tokens"),
                "output_tokens": usage_value("output_tokens"),
                "total_tokens": usage_value("total_tokens"),
            },
            "request_count": 1,
        }

    def preflight(self) -> None:
        self._get()


def _read_zips(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return tuple(sorted(TARGET_ZIPS))
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError("ZIP file is unavailable") from exc
    values = tuple(item for item in re.split(r"[\s,]+", raw.strip()) if item)
    if (not values or len(set(values)) != len(values)
            or any(re.fullmatch(r"\d{5}", item) is None for item in values)):
        raise ValueError("ZIP file must contain unique five-digit ZIP codes")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NETS-AI Wayback/LLM v2 thesis workflow.",
    )
    parser.add_argument("--task", choices=("pharmacy",), required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--zips-file", type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "data" / "v2")
    parser.add_argument("--authorization-file", type=Path,
                        default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pilot-size", type=int, default=8)
    parser.add_argument("--run-mode", choices=("smoke", "pilot", "full"),
                        default="smoke")
    parser.add_argument("--promotion-file", type=Path)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=ROOT / "analysis" / "input_manifest.json",
    )
    parser.add_argument("--pilot-delivery-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, stdout=None, stderr=None) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = _parser().parse_args(argv)
    wayback_preflight_report = None
    try:
        zips = _read_zips(args.zips_file)
        authorization = args.authorization_file or ROOT / "AGENTS.md"
        if not args.dry_run and args.model is None:
            raise ValueError("live v2 mode requires an explicit model")
        model = args.model or "model-unset"
        effective_limit = args.limit
        if args.run_mode == "smoke" and effective_limit is None:
            effective_limit = 1
        if args.run_mode == "pilot" and effective_limit is None:
            effective_limit = 10
        config = V2WorkflowConfig(
            run_id=args.run_id, task=args.task, city=args.city, zips=zips,
            output_root=args.output_root,
            authorization_file=authorization, model=model,
            dry_run=args.dry_run, resume=args.resume, limit=effective_limit,
            run_mode=args.run_mode, promotion_file=args.promotion_file,
            pilot_size=args.pilot_size,
        )
        if args.dry_run:
            result = V2Workflow(
                None, None, None, allow_thesis_stage_live=True
            ).run(config)
            stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
            return 0
        stderr.write(f"Compliance warning: {COMPLIANCE_WARNING}\n")
        load_dotenv(dotenv_path=ROOT / ".env", override=False)
        execution_authorization_sha256 = hashlib.sha256(
            authorization.read_bytes()
        ).hexdigest()
        promotion_sha256 = (
            hashlib.sha256(args.promotion_file.read_bytes()).hexdigest()
            if args.run_mode == "full" and args.promotion_file is not None
            else None
        )
        cache_dir = args.output_root / args.run_id / "private" / "wayback_cache"
        state_dir = args.output_root / "private" / "wayback_state"
        google = _LazyGoogleProvider()
        classifier = _LazyClassifier()
        full_delivery = (
            _LazyFullDelivery(
                args.input_manifest,
                args.promotion_file,
                zips,
                model,
                args.pilot_delivery_root,
                execution_authorization_sha256,
                promotion_sha256,
            )
            if args.run_mode == "full" and args.promotion_file is not None
            else None
        )

        def live_preflight():
            nonlocal wayback_preflight_report
            # The authenticated local contract is checked before credentials are
            # read; V2Workflow invokes this after the local thesis-stage checks.
            if full_delivery is not None:
                full_delivery.preflight()
            wayback_preflight_report = require_research_wayback_service(
                state_dir=state_dir, direct_transport=_WaybackArchiveTransport(),
                envfile=ROOT / ".env",
            )
            google.preflight()
            classifier.preflight()

        runner = V2Workflow(
            google,
            LazyWaybackCollector(cache_dir, state_dir=state_dir,
                                 transport_factory=_WaybackArchiveTransport,
                                 envfile=ROOT / ".env"),
            classifier,
            approved_permission_sha256=execution_authorization_sha256,
            approved_promotion_sha256=promotion_sha256,
            allow_thesis_stage_live=True,
            live_preflight=live_preflight,
            ablation_evaluator=classifier.evaluate_messages,
            delivery_callback=full_delivery,
        )
        result = runner.run(config)
        stdout.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
        if str(result.get("status", "")).endswith("_paused_wayback"):
            return 3
        return 0 if result.get("status") in {
            "dry_run", "smoke_complete", "pilot_complete", "full_complete",
        } else 1
    except WaybackDeferred as exc:
        stderr.write(json.dumps(deferred_report(exc), sort_keys=True) + "\n")
        return 3
    except WaybackServiceUnavailable as exc:
        stderr.write("Wayback preflight failed: " + json.dumps(exc.report, sort_keys=True) + "\n")
        return 3
    except AuthorizationGateError:
        # Wayback availability and deferral retain their dedicated diagnostics.
        # Other local contract/dependency errors remain sanitized here.
        if wayback_preflight_report is not None and wayback_preflight_report["status"] != "ready":
            stderr.write("Wayback preflight failed: " + json.dumps(wayback_preflight_report, sort_keys=True) + "\n")
            return 3
        stderr.write(
            "Live v2 acquisition stopped: a local authorization, promotion, "
            "or dependency check failed.\n"
        )
        return 2
    except (OSError, TypeError, ValueError):
        stderr.write("The v2 workflow configuration or checkpoint is invalid.\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
