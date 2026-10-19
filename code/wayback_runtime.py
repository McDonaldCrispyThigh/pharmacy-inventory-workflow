"""Live Wayback wiring shared by both runners; no I/O occurs at import time."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from outscraper_wayback import OutscraperFallback, load_api_key
from skills.wayback_agent import WaybackEvidenceCollector
from wayback_state import WaybackDeferred, WaybackStateStore


def build_wayback_collector(
    cache_dir: Path, *, state_dir: Path, direct_transport: Any,
    envfile: Path | None = None, config: Any = None, clock: Any = None,
    sleep: Any = None, random: Any = None,
) -> WaybackEvidenceCollector:
    """Use direct Archive requests first and one shared durable fallback store.

    ``state_dir`` must be shared across empirical run directories. The caller must
    invoke this only in live mode after authenticating its local run contract.
    """
    options = {}
    state_options = {}
    fallback_options = {
        "scope": hashlib.sha256(str(cache_dir.resolve()).encode()).hexdigest(),
    }
    if clock is not None:
        options["clock"] = clock
        state_options["clock"] = clock

        def epoch_clock():
            value = clock()
            return value.timestamp() if isinstance(value, datetime) else float(value)

        fallback_options["clock"] = epoch_clock
    for name, value in (("config", config), ("sleep", sleep), ("random", random)):
        if value is not None:
            options[name] = value
    store = WaybackStateStore(state_dir, **state_options)
    key = load_api_key(envfile) if envfile is not None else load_api_key()
    fallback = OutscraperFallback(store, key, **fallback_options) if key else None
    return WaybackEvidenceCollector(
        direct_transport, cache_dir, state_store=store, fallback=fallback, **options
    )


def require_research_wayback_service(
    *, state_dir: Path, direct_transport: Any, envfile: Path | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Probe CDX and replay through the same direct/fallback acquisition path.

    Health cache and asynchronous jobs survive a retry before an empirical run
    directory exists. Deferred jobs propagate instead of becoming missing evidence.
    """
    collector = build_wayback_collector(
        state_dir / "health_cache", state_dir=state_dir,
        direct_transport=direct_transport, envfile=envfile, **options,
    )
    return collector.check_health()


class LazyWaybackCollector:
    """Delay credentials and state creation until an authenticated live run."""

    def __init__(self, cache_dir: Path, *, state_dir: Path, transport_factory: Any,
                 envfile: Path | None = None):
        self.cache_dir = cache_dir
        self.state_dir = state_dir
        self.transport_factory = transport_factory
        self.envfile = envfile
        self._collector = None

    def collect(self, candidate):
        if self._collector is None:
            self._collector = build_wayback_collector(
                self.cache_dir, state_dir=self.state_dir,
                direct_transport=self.transport_factory(), envfile=self.envfile,
            )
        return self._collector.collect(candidate)


def deferred_report(error: WaybackDeferred) -> dict[str, Any]:
    """Expose controlled retry metadata, never provider response text or keys."""
    return {
        "status": "wayback_deferred", "retryable": True,
        "reason": error.reason, "retry_at": error.retry_at,
    }
