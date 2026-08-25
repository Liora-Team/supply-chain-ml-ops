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
   docker compose restart api
   ```

4. Verify — `source` should be `registry` and the registered name current:

   ```bash
   curl -sk https://localhost/models -H "Authorization: Bearer $TOKEN"
   ```

Details and the no-`--run-id` variant: [README — Promote a model](../README.md#promote-a-model).

### 1b. Deploy new code

1. Open a PR to `dev`, get a review, merge ([CONTRIBUTING.md](../CONTRIBUTING.md) — CI
   must be green).
2. Rebuild and restart the stack on the host:

   ```bash
   make build && make up
   ```

> The automated path (CI builds a SHA-tagged image on GHCR, k8s rolls it out) is
> Card 3.4 — **pending PR #35**. Once merged, step 2 becomes "merge and let the
> pipeline deploy".

---

## 2. Rollback

### 2a. Roll back a model

Move the `production` alias back to the previous version — nothing is deleted or
retrained. `scripts/register_model.py` prints the previous version and the exact restore
snippet at every promotion; the canonical copy is in
[README — Roll back a promotion](../README.md#roll-back-a-promotion).

1. Run the printed restore snippet (sets the alias back to `<previous-version>`).
2. `docker compose restart api` — same cache reason as above.
3. Verify with `/models` as in 1a.

### 2b. Roll back code

1. Revert the offending commit on `dev` (`git revert <sha>`, PR, merge).
2. `make build && make up` to serve the reverted code.

> With Card 3.4 (**pending PR #35**) this becomes an image-tag revert:
> `kubectl -n sc-mlops rollout undo deployment/sc-mlops-api`.

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

> The model DAG (`model_pipeline`: train → evaluate → **promotion gate**) is Card 3.2 —
> **pending PR #34**. Its gate only moves the `production` alias when the new run's
> macro-F1 is strictly greater than the current production model's; otherwise nothing
> is promoted.

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

Card 4.1 will start **storing `/predict` requests** for monitoring. When that lands,
this section must gain a retention/cleanup note (how long stored requests are kept and
how they are purged) — tracked in the Card 4.1 acceptance criteria.
