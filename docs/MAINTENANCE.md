# Maintenance guide — operating the service (Card 4.3)

> Runbooks for the three things an operator actually does: **update** (new model or new
> code), **roll back**, and **retrain**. Written so someone who did not build the project
> can run it. Commands are canonical here; background lives in the README and ADRs —
> linked, not repeated.

**Prerequisites:** repo cloned, `make setup` done, `.env` filled in
([CONTRIBUTING.md](../CONTRIBUTING.md)), Docker running for anything compose-based.

---

## 1. Update

### 1a. Deploy a new model (no code change)

A model deploy is just moving the MLflow `production` alias — no rebuild, no new image
(model loading explained in [README — Model loading](../README.md#model-loading)).

1. Pick the approved run id in the MLflow UI (experiment `trustpilot-reviews`).
2. Promote it — registers the run's model and moves the `production` alias:

   ```bash
   uv run python scripts/register_model.py \
     --schema 3-class \
     --run-id <approved-run-id> \
     --tracking-uri "<mlflow-tracking-uri>"
   ```

3. Restart the API so it drops its in-process model cache and loads the new version:

   ```bash
   docker compose restart api                                   # compose stack
   kubectl -n sc-mlops rollout restart deployment/sc-mlops-api  # k8s
   ```

4. Verify — `source` should be `registry` and the registered name current:

   ```bash
   curl -sk https://localhost/models -H "Authorization: Bearer $TOKEN"
   ```

Details and the no-`--run-id` variant: [README — Promote a model](../README.md#promote-a-model).

### 1b. Deploy new code

1. Open a PR to `dev`, get a review, merge ([CONTRIBUTING.md](../CONTRIBUTING.md) — CI
   must be green).
2. The merge to `dev` triggers the `cicd-k8s` workflow (Card 3.4): it builds a
   SHA-tagged image on GHCR and rolls it out to the k8s deployment — nothing to run
   by hand. See [ADR 004](adr/004-cicd-k8s.md).
3. Compose stack (course demo host): rebuild and restart manually:

   ```bash
   make build && make up
   ```

---

## 2. Rollback

### 2a. Roll back a model

Move the `production` alias back to the previous version — nothing is deleted or
retrained. `scripts/register_model.py` prints the previous version and the exact restore
snippet at every promotion; the canonical copy is in
[README — Roll back a promotion](../README.md#roll-back-a-promotion).

1. Run the printed restore snippet (sets the alias back to `<previous-version>`).
2. Restart the API (`docker compose restart api` / `kubectl rollout restart`) — same cache reason as above.
3. Verify with `/models` as in 1a.

### 2b. Roll back code

1. Fastest (k8s): roll the deployment back to the previous image:

   ```bash
   kubectl -n sc-mlops rollout undo deployment/sc-mlops-api
   ```

2. Then revert the offending commit on `dev` (`git revert <sha>`, PR, merge) so the
   next `cicd-k8s` run does not redeploy the bad code.
3. Compose stack: `make build && make up` after the revert lands.

---

## 3. Retrain

### 3a. Orchestrated (Airflow — preferred)

The data pipeline (ingest → preprocess → DVC version) runs as the `data_pipeline` DAG
([ADR 002](adr/002-airflow-orchestration.md)); commands in
[README — Containers](../README.md#containers-compose):

```bash
make dag-up        # start Airflow (needs DAGSHUB_TOKEN in .env)
make dag-trigger   # trigger data_pipeline; watch it at http://127.0.0.1:8080
make dag-down
```

Afterwards review and commit the refreshed `data/processed/*.dvc` pointers.

Then trigger the model DAG (`model_pipeline`: train → evaluate → **promotion gate**,
Card 3.2):

```bash
docker compose --profile airflow exec airflow \
  airflow dags trigger model_pipeline
```

Its gate only moves the `production` alias when the new run's macro-F1 is strictly
greater than the current production model's; otherwise nothing is promoted. Details:
[README — Automated promotion gate](../README.md#automated-promotion-gate).

### 3b. Manual fallback

```bash
make data    # rebuild data splits
make train   # retrain the classical pipelines
make test    # sanity check
```

Then promote the winning run as in **1a** (the gate's judgement call — compare macro-F1
yourself before moving the alias).

---

## 4. Data privacy

The service is trained on a **public Trustpilot dataset** (HuggingFace; provenance and
scope in [docs/DATA_SOURCES.md](DATA_SOURCES.md)). Review text can contain personal
names the reviewers wrote themselves, but the service has **no user accounts and
collects no user PII**: `/predict` takes raw text and returns a rating, and requests are
authenticated with short-lived JWTs identifying a client, not a person
([ADR 003](adr/003-jwt-auth-and-self-signed-tls.md)).

Card 4.1 stores **`/predict` requests** for monitoring (text, predicted label, confidence,
model version, timestamp — no client identity) in a SQLite file under the shared
`./monitoring` compose mount. Retention: rows older than `REQUEST_STORE_RETENTION_DAYS`
(default 30) are purged automatically at the end of every scheduled drift-check run
(`scripts/run_drift_check.py`), not on a separate timer — see
[ADR 005](adr/005-drift-signal-format.md) for the store's format and the k8s
multi-replica caveat.


## Automated Retraining (Card 4.4)

### What triggers retraining
The `auto_retrain_pipeline` DAG polls every `RETRAIN_POLL_MINUTES` and reads the drift flag from
`DRIFT_STATUS_PATH` (defaults to `monitoring/drift_status.json` if unset). It triggers a retrain
only when a watched schema reports `drift_detected: true` with `status: "ok"`, the event is new
(not already handled), the report is not stale, and no cooldown is active.

### Drift-status contract (read-only for 4.4)
Per-schema keys: `status`, `drift_detected`, `timestamp`, `n_rows`,
`window_start`, `window_end`, `report_path`. `status: "insufficient_data"` never
triggers. Card 4.4 never writes this file (see ADR 005/006).

### Cooldown policy
`RETRAIN_COOLDOWN_HOURS` (default 24). A successful trigger writes
`cooldown_until` into `monitoring/retrain_state.json` on the shared mount, so it
survives Airflow/API/container restarts. The same drift event is deduplicated
via `last_handled_drift_ts`.

### Category-slice rotation (Option 3)
`RETRAIN_CATEGORIES` (ordered list). A persistent `slice_cursor` advances each
retrain and wraps at the end. It drives the data pipeline with a real category
slice (new `category` param).

**Important:** to preserve promotion-gate comparability (Card 3.2), category slices must extend
the training set while keeping the evaluation test set fixed.

### Inspect retraining state
Windows: `type monitoring\retrain_state.json`  
macOS/Linux: `cat monitoring/retrain_state.json`

### Manual trigger / replay
Airflow UI -> `auto_retrain_pipeline` -> Trigger. To seed a signal without
Card 4.1: `python scripts/seed_drift_status.py --drift`.

### How promotion works
The DAG triggers `data_pipeline` (next slice) then `model_pipeline`. The Card 3.2
gate (`src/promotion.py`, strict candidate > production macro-F1) is the SOLE
promotion authority. Card 4.4 never promotes.

### How the API picks up the promoted model
The API caches the production pipeline (`@lru_cache`). After a promotion, run
`docker compose restart api` to serve the new model **without rebuilding the
image**.

### Persistent drift after cooldown
If drift remains `true` after the cooldown window expires (i.e., a fresh drift report arrives
after `cooldown_until`), the loop logs `[PERSISTENT-DRIFT]` and sets
`persistent_drift_blocked=true` in `monitoring/retrain_state.json` to prevent a tight retraining loop.
This requires operator attention.

To resume automated retraining after investigation, edit `monitoring/retrain_state.json` and set
`persistent_drift_blocked` to `false` (or remove the key).

### Known limitations
- Airflow is Linux-only (POSIX); the loop runs in the docker-compose airflow
  container and is validated by CI on Ubuntu, not on native Windows.
- Kubernetes 2-replica setup has no shared PVC for the SQLite request store
  (Card 4.1) or the state file -> per-pod state (ADR 005). The demo runs on
  docker-compose where the `./monitoring` mount is shared.
