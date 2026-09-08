"""Prometheus exposition of the Card 4.1 drift flag, for 4.2 (alerting, Dilshana).

Registers a custom collector that re-reads ``monitoring/drift_status.json`` on every
``/metrics`` scrape — not a value cached at process start — so a stale reading is
never served between drift-job runs. Exposes one gauge per label schema that has an
``"ok"`` entry; a schema still on ``"insufficient_data"`` is simply absent from the
scrape rather than reported as "no drift" (Prometheus's `absent()` can alert on that
distinction if 4.2 wants to).

See ``docs/adr/005-drift-signal-format.md`` for why this is a gauge next to the
existing ``/metrics`` (Card 4.3) rather than a dedicated endpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from prometheus_client import REGISTRY
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

STATUS_PATH = Path(os.environ.get("DRIFT_STATUS_PATH", "monitoring/drift_status.json"))

METRIC_NAME = "drift_detected"
METRIC_HELP = (
    "1 if Card 4.1's Evidently job flagged data drift for this label schema, else 0. "
    "Absent for a schema whose last run was 'insufficient_data' — see drift_status.json."
)


class DriftStatusCollector(Collector):
    """Reads drift_status.json fresh on every collection instead of caching a value."""

    def collect(self):
        gauge = GaugeMetricFamily(METRIC_NAME, METRIC_HELP, labels=["schema"])
        status: dict = {}
        if STATUS_PATH.exists():
            try:
                status = json.loads(STATUS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                status = {}  # corrupt/mid-write file — report nothing rather than crash the scrape
        for schema, entry in status.items():
            if entry.get("status") == "ok":
                gauge.add_metric([schema], 1.0 if entry.get("drift_detected") else 0.0)
        yield gauge


def register(registry=REGISTRY) -> None:
    """Register the collector once. Safe to call repeatedly (e.g. under --reload)."""
    if any(isinstance(c, DriftStatusCollector) for c in list(registry._collector_to_names)):
        return
    registry.register(DriftStatusCollector())
