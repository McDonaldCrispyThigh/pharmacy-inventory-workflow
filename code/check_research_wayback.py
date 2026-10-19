"""Resume a public CDX/replay control through the production acquisition path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'code')]

from run_v2_workflow import _WaybackArchiveTransport
from skills.wayback_agent import WaybackCollectorConfig, WaybackServiceUnavailable
from wayback_runtime import deferred_report, require_research_wayback_service
from wayback_state import WaybackDeferred, WaybackStateError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--state-dir', type=Path)
    args = parser.parse_args(argv)
    if not args.live:
        print(json.dumps({'status': 'dry_run', 'network_requests': 0,
                          'control': 'http://example.com/',
                          'strategy': 'direct_wayback_then_outscraper_ai'}))
        return 0
    if args.state_dir is None:
        parser.error('--state-dir is required for durable live checks')
    try:
        report = require_research_wayback_service(
            state_dir=args.state_dir, direct_transport=_WaybackArchiveTransport(),
            config=WaybackCollectorConfig(max_attempts=1),
        )
    except WaybackDeferred as exc:
        print(json.dumps(deferred_report(exc), sort_keys=True))
        return 3
    except WaybackServiceUnavailable as exc:
        print(json.dumps(exc.report, sort_keys=True))
        return 3
    except WaybackStateError:
        print(json.dumps({'status': 'integrity_error', 'retryable': False}))
        return 4
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
