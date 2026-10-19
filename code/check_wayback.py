"""Public-control Availability/CDX/replay diagnostic; no credentials or file writes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.wayback_agent import (
    AVAILABILITY_API, CDX_API, WAYBACK_CONTROL_URL, WaybackHTTPResponse,
    probe_wayback_endpoint, probe_wayback_service,
)


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DiagnosticTransport:
    def __init__(self):
        self.opener = build_opener(_NoRedirects())

    def get(self, url, params, timeout, *, follow_redirects):
        parsed = urlparse(url)
        if (follow_redirects or parsed.scheme != "https" or parsed.username
                or parsed.password or parsed.port or parsed.fragment
                or not (url in {AVAILABILITY_API, CDX_API}
                        or (parsed.hostname == "web.archive.org" and parsed.path.startswith("/web/")))):
            raise ValueError("diagnostic transport accepts only declared Archive endpoints")
        request = Request(url + ("?" + urlencode(params) if params else ""), headers={
            "User-Agent": "NETS-AI-honors-thesis-wayback-diagnostic/1.0",
            "Accept": "application/json,text/html",
        })
        try:
            response = self.opener.open(request, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            return WaybackHTTPResponse(response.code, dict(response.headers),
                                       response.read(1_000_001), response.geturl())


def main(argv=None, *, transport=None, stdout=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    transport = transport if transport is not None else DiagnosticTransport()
    stdout = stdout if stdout is not None else sys.stdout
    availability, response = probe_wayback_endpoint(
        transport, AVAILABILITY_API, (("url", WAYBACK_CONTROL_URL),), args.timeout,
    )
    if response is not None:
        try:
            payload = json.loads(response.body)
            snapshots = payload["archived_snapshots"]
            if not isinstance(snapshots, dict):
                raise ValueError("invalid snapshots")
            closest = snapshots.get("closest", {})
            if not isinstance(closest, dict):
                raise ValueError("invalid closest snapshot")
            availability["status"] = (
                "snapshot_available" if closest.get("available") is True
                else "no_accessible_snapshot"
            )
        except (ValueError, KeyError, TypeError):
            availability["status"] = "invalid_response"
    report = probe_wayback_service(transport, timeout=args.timeout)
    report["availability"] = availability
    stdout.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return 0 if report["status"] == "ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
