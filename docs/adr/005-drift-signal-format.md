# ADR 005: Drift Signal Format, Request Store, and Comparison Window

**Context:**
Card 4.1 (drift detection) produces a signal that two other cards depend on: 4.2 (Prometheus/
Grafana alerting, Dilshana) reads it to fire an alert, and 4.4 (automated retraining, Luc)
reads it to trigger the orchestrator. Building against three different open questions
(flag format, request-store backend, comparison window) without agreement would force 4.2
and 4.4 to guess at an interface that might change. This ADR closes those three open
questions from `docs/TODO.md` Card 4.1, agreed with Luc and Dilshana on [дата].

**Decision:**

**1. Drift flag — single file as source of truth.**
`monitoring/drift_status.json` is written by the Evidently report job on every run
(scheduled and on-demand), fully **overwritten each time** — including back to
`"drift_detected": false` once drift clears, so a resolved shift doesn't leave 4.4 retraining
forever. Shape:

```json
{
  "drift_detected": true,
  "timestamp": "2026-09-05T14:30:00Z",
  "schema": "3-class",
  "n_rows": 500,
  "window_start": "2026-09-04T14:30:00Z",
  "window_end": "2026-09-05T14:30:00Z",
  "report_path": "monitoring/reports/2026-09-05T1430.html"
}
```

`n_rows` / `window_start` / `window_end` let 4.4 implement a retrain cooldown (e.g. don't
retrigger on a window that barely changed) without re-deriving it from the request store.

Consumers:
- **4.2 (Dilshana)** reads a Prometheus gauge (`drift_detected{schema=...}` 0/1) exposed
  next to the existing `/metrics` (`prometheus_fastapi_instrumentator`, Card 4.3) — no new
  scrape target needed, it's the same endpoint they already scrape.
- **4.4 (Luc)** reads the JSON file directly from the shared volume (see below) — Airflow
  and the drift job share a filesystem, so no HTTP call is needed for an in-network boolean.
- No dedicated `/drift-status` API endpoint is built. Revisit only if something outside the
  compose network needs to read the flag.

**2. Request store — SQLite.**
`monitoring/request_store.db`, one row per `/predict` call: `text`, `predicted_label`,
`confidence`, `model_version`, `schema`, `timestamp`. Opened with `PRAGMA journal_mode=WAL`
and a busy timeout, so concurrent writes from multiple `api` replicas (`--scale api=N`)
don't corrupt the file. A cleanup step (delete rows older than N days) runs as part of the
scheduled orchestrator job — this closes the retention/cleanup note already flagged in
`docs/MAINTENANCE.md`'s data-privacy section pending Card 4.1.

**Known limitation:** in the k8s deployment (Card 3.4, `sc-mlops-api` at >= 2 replicas, no
PVC), each pod gets its own local SQLite file rather than a shared one. Acceptable for the
compose/course demo; a shared volume or a networked store would be required before this
scales past a single-node deployment.

**3. Comparison window.**
"Current" data = requests from the last `DRIFT_WINDOW_HOURS` (default 24), capped at
`DRIFT_MAX_ROWS` most recent rows (default 500) — whichever limit is hit first. Below
`DRIFT_MIN_ROWS` (default 50), the job writes `"status": "insufficient_data"` instead of a
drift verdict, so a quiet period doesn't produce a false read either way. All three are env
vars, tunable without a code change.

**4. Shared filesystem.**
Neither `api` nor `airflow` currently mount anything under a common path (`api` has no
volumes at all; `airflow` mounts `./dags`, `./data`, `./models`, `./.dvc`, `./.git`). A new
bind mount, `./monitoring:/app/monitoring`, is added to both services in
`docker-compose.yml` — this is where `request_store.db` and `drift_status.json` live, and
is what makes "4.4 reads the file directly" actually possible.

**5. Dependency isolation.**
Evidently is added to the `orchestration` uv group (alongside `apache-airflow` and
`dvc[s3]`), not `api` — the API image stays torch-free and small per the project's existing
dependency-group split (`pyproject.toml`).

**Consequences:**

- 4.2 and 4.4 can build against a fixed interface (one gauge, one file) starting now,
  instead of waiting on 4.1's implementation details.
- The drift job, the request store, and the flag file all live under one new `monitoring/`
  module and one new compose bind mount — no new services, no new infrastructure beyond
  what Cards 3.1–3.4 already run.
- The retention note pending in `docs/MAINTENANCE.md` gets closed as part of this card
  instead of being deferred again.
- The k8s multi-replica SQLite limitation is accepted and documented rather than solved now;
  revisiting it is out of scope for this card.