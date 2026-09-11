# ADR 006: Automated Retraining Control Loop (Card 4.4)

**Status:** Accepted (implements Issue #23).

**Context:**
Card 4.4 closes the MLOps loop. It consumes the drift signal defined by Card 4.1
in **ADR 005** (`monitoring/drift_status.json`, per-schema, overwritten each run,
with `status`/`drift_detected`/`timestamp`/`n_rows`/window fields, plus the
`insufficient_data` guard) and triggers retraining automatically, without
bypassing the Card 3.2 promotion gate (`src/promotion.py`, strict candidate >
production macro-F1).

**Open questions (Issue #23) and decisions:**

1. **Trigger policy — flag, schedule, or both?**
   A short-schedule *poller* DAG (`auto_retrain_pipeline`, `RETRAIN_POLL_MINUTES`,
   default 15) reads the flag and acts only on actionable drift. A pure
   event-sensor is fragile on the single standalone Airflow container (ADR 002);
   a poller guarded by cooldown + dedupe is restart-safe and never retrains on
   every tick.

2. **Cooldown length?**
   `RETRAIN_COOLDOWN_HOURS`, default 24 (aligned with `DRIFT_WINDOW_HOURS`).
   Persisted in `monitoring/retrain_state.json` on the shared mount, surviving
   Airflow/API/container restarts. The same drift event is deduplicated via
   `last_handled_drift_ts`.

3. **How is the next category slice chosen?**
   A persistent, deterministic cursor over `RETRAIN_CATEGORIES`, wrapping at the
   list end. This implements DATA_SOURCES.md **Option 3** by driving the *data*
   pipeline with a real category slice (a new optional `category` param on
   `get_data.load_raw` / `data_pipeline`), NOT by varying row counts.

**Persistent drift after cooldown:** surfaced via logs (`[PERSISTENT-DRIFT]`) and
a `persistent_drift_blocked` flag — never a tight retrain loop.

**API adoption:** the API caches the production pipeline in-process (`@lru_cache`).
After a promotion, `docker compose restart api` serves the new model without
rebuilding the image (defence: "API serves it, no rebuild"). Card 4.4 does not
add a reload endpoint and does not give Airflow Docker-socket access.

**Ownership / no shared-file edits:** `retrain_state.json` lives beside
`drift_status.json` under `monitoring/` (the shared mount Card 4.1/PR #38 owns).
Card 4.4 does NOT edit `docker-compose.yml`, `Dockerfile`, or
`monitoring/.gitignore` (owned by PR #38/#39), so merge order between #38/#39/#23
is unaffected. The control DAG imports only stdlib + `src.retrain` — never
`monitoring` — so it is unaffected by PR #38's Dockerfile monitoring-copy blocker.

**Known limitation (honest scoping):** for the demo, each retrain uses a
*different, deterministic* category slice ("genuinely sees new data" per the
issue). A strict held-out-baseline guarantee and a fully unattended API reload
are documented as future improvements; they are out of scope for closing the
loop here. Airflow runs Linux-only (POSIX), so the loop is validated via CI on
Ubuntu and via the docker-compose airflow container, not on native Windows.

**Comparability note:** promotion compares candidate vs. production macro-F1 via
the existing Card 3.2 gate; Card 4.4 does not change the evaluation policy — it
only changes the training slice. Card 4.4 never forces `drift_status.json` to
false; the flag clears only when Card 4.1 next evaluates a non-drifted window.

**Consequences:** deterministic, restart-safe, observable, fully unit-tested
(`tests/test_retrain_decision.py`) without Airflow, Docker, or network.