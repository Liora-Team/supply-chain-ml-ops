"""Drift-detection support for Card 4.1.

Modules:
- ``request_store``: SQLite log of ``/predict`` calls — the "current" dataset.

See ``docs/adr/005-drift-signal-format.md`` for the format/backend decisions this
package implements.
"""
