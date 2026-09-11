"""Drift-detection support for Card 4.1.

Modules:
- ``request_store``: SQLite log of ``/predict`` calls — the "current" dataset.
- ``datasets``: reference (training) vs. current (request-store) frames.
- ``drift_report``: the Evidently job and the ``drift_status.json`` flag.

See ``docs/adr/005-drift-signal-format.md`` for the format/backend decisions this
package implements.
"""
