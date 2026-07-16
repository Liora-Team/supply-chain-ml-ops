# Milestones — technical playbook

The **north-star** for the build: what to make, in what order, who owns it, and how to know
each stage is done. One milestone per course phase. Each task uses the same shape so anyone
can pick it up cold:

> **Task** — what to do · **Why** — the MLOps concept it teaches (one sentence) ·
> **Owner** — who builds it (rotation → `docs/PROJECT_MANAGEMENT.md`) · **Files** — where ·
> **Run** — command(s) · **Done when** — a testable acceptance check ·
> **Gotcha** — the thing that'll bite you.

Project framing (what we predict and why) → [`docs/ML_CANVAS.md`](docs/ML_CANVAS.md).
All other docs → the **Docs map** in [`README.md`](README.md).

## Roadmap

| Phase | Focus | Target date | Project week(s) | Status |
|-------|-------|-------------|-----------------|--------|
| 1 | Foundations & Containerization | **Jul 10** | 1 | 🟢 done |
| 2 | Microservices, Tracking & Versioning | **Jul 24** | 2–3 | ⚪ next |
| 3 | Orchestration & Deployment | **Aug 7** | 4–5 | ⚪ |
| 4 | Monitoring & Maintenance | **Sep 4** | 6–7 | ⚪ |
| — | **Project freeze** — everything done & demoable | **Sep 9** | — | ⚪ |
| — | **Final defence** | **Sep 14** | — | ⚪ |

One project week per weekly check-in. (Not the course's multi-week "Sprint N" module
windows — those are mapped to cards in `docs/TODO.md`'s course calendar.)
**Freeze rule:** the project must be ready **≥5 days before the defence** — after Sep 9,
only rehearsal, slides and demo recording; zero feature work.

---

## Milestone 1 — Foundations & Containerization · Phase 1 · **Jul 10** · 🟢 done

**Goal:** a reproducible, tested, containerized base everyone runs identically.

Complete — the full task list with evidence lives in `docs/TASKS_DONE.md`:
env (`uv`/`make setup`, torch-free), Docker + compose, `scripts/get_data.py` (full 123k pull),
baseline pipelines (`make train`), 15 tests, `api/main.py`, CI workflow, docs.

**Done when:** teammate clones → `make setup && make test` green → `make api` answers `/predict`.
**Kickoff mapping:** Phase 1 "Foundations & Env setup".

---

## Milestone 2 — Microservices, Tracking & Versioning · Phase 2 · target **Jul 24** *(next build)*

**Goal:** trace every experiment, version data + models, split into services.

> 📋 **Board-ready breakdown** — per-member cards with subtask checklists, branch names and
> acceptance criteria: [`docs/TODO.md`](docs/TODO.md).

**What you'll learn:**
- **Experiment tracking** — never again wonder "which run produced this model?"
- **Data & model versioning** — give datasets and weights the same audit trail git gives code.
- **Service decomposition** — one container per responsibility, so parts scale and fail independently.

### Task 2.1 — Experiment tracking with MLflow · **Owner: Luc**

- **Why:** experiment tracking makes every training run reproducible and comparable — params,
  metrics and the produced model are recorded automatically instead of in someone's terminal history.
- **Files:** `scripts/build_pipelines.py` (`main()` loop, around each estimator fit); add
  `mlflow` to a `track` dependency group in `pyproject.toml`.
- **Do:** wrap each fit in `with mlflow.start_run(run_name=<model_id>):`, then
  `mlflow.log_params(best_params)`, `mlflow.log_metric("macro_f1", score)`,
  `mlflow.sklearn.log_model(pipe, "model")`.
- **Run:** `mlflow ui` (or the compose service from Task 2.4) → `http://localhost:5000`.
- **Done when:** a `make train` run shows up in the MLflow UI with params + macro-F1 + the artifact.
- **Gotcha:** point `MLFLOW_TRACKING_URI` at the server so runs land
  in one place, not local `./mlruns`.

### Task 2.2 — Model registry, API side · **Owner: Dilshana**

- **Why:** a model registry — a versioned catalog with promote/rollback — decouples
  "a new model exists" from "the API serves it": deployment becomes a one-step alias move.
- **Files:** `api/main.py`, `src/registry.py`, `tests/test_api.py`.
- **Do:** register the best run's model (with Luc); point a `production` **alias** at the
  chosen version (stage transitions are deprecated in MLflow 2.x — use aliases); have the
  API optionally load from the registry (`models:/<name>@production`) instead of the local
  joblib. Add tests for both load paths.
- **Done when:** moving the `production` alias in MLflow changes what the API serves (no code edit).
- **Gotcha:** keep the local-joblib path as a fallback so the API still
  boots offline (and CI stays green without an MLflow server).

### Task 2.3 — Data + model versioning with DVC · **Owner: Mykola**

- **Why:** data versioning gives datasets the audit trail git gives code — you can name the
  exact bytes a model trained on (data lineage) without bloating the repo.
- **Run:** `uv add --group dev dvc` → `dvc init` → `dvc add data/processed/train.csv
  data/processed/test.csv` → `dvc add` **every** `models/` weight (all `.joblib`, incl. the
  64 MB RF-5class; DistilBERT when it arrives) → `dvc remote add`.
- **Files:** commit the small `*.dvc` pointers; data and weights go to the remote. Team
  decision: **no model binaries in git** — git holds code + JSON metric sidecars, DVC holds
  data + weights, MLflow holds runs. CI gains a `dvc pull` step (token via repo secret).
- **Done when:** fresh clone + `make pull` restores data + all weights and tests pass;
  `git ls-files` contains no `.joblib`.
- **Gotcha:** the JSON metric sidecars stay in git (the leaderboard reads them without DVC);
  onboarding now needs one `make pull` — update the README quickstart claim. DistilBERT
  weights load via env vars (→ [CONTRIBUTING.md §7](CONTRIBUTING.md#7-data--model-handling)).

### Task 2.4 — Microservices split · **Owner: Marco**

*(Split on the board into 2.4-A day-1 shared setup — dev branch, DagsHub, secrets, hygiene —
and 2.4-B compose split; see `docs/TODO.md`.)*

- **Why:** isolating services (API / training / tracking / storage) is the architecture step
  that makes everything after — orchestration, scaling, independent deploys — possible.
- **Files:** `docker-compose.yml` → services: `api` (exists), `training` (runs
  `build_pipelines.py`), `mlflow` (tracking server), plus a store (`minio` for artifacts and/or
  `postgres` for the MLflow backend). This is also where the DistilBERT path becomes its own
  torch-only service (the README's "Phase 2" note), keeping the classical API image slim —
  weights via the `DISTILBERT_*` env vars ([CONTRIBUTING.md §7](CONTRIBUTING.md#7-data--model-handling)).
- **Done when:** `docker compose up` brings the whole stack; the API reads models the training
  service produced and MLflow tracked.
- **Gotcha:** services talk over the compose network — use service
  names (`http://mlflow:5000`), not `localhost`, inside containers.

### Task 2.5 — Internal Ops dashboard · **Owner: Marco**

- **Why:** the seven demo pages are customer-facing; the team needs one operator view of
  project state — a model-registry / data-versioning / tracking cockpit turns "is it working?"
  into a glance and gives each later phase a home to surface its tool (MLflow, Airflow, Grafana).
- **Files:** `pages/8_Ops.py` (new Streamlit page) + `app.py` (nav entry + "What's inside" row);
  reads `src.registry` + `src.eda` only. Board breakdown → `docs/TODO.md` Card 2.5.
- **Do:** a status strip (registry counts + serving model + dataset rows + MLflow-configured), a
  model-registry table (Serving ★ = the API default), a data & versioning panel (star balance,
  git SHA, DVC state), an MLflow config panel, a secrets-masked env expander, and deep links out.
- **Done when:** opening **Ops** on a fresh clone (nothing configured) renders every panel with no
  crash, ★ marks the API's default model, and no secret value is printed.
- **Gotcha:** keep it **in-process** — no HTTP ping of the API (that needs `requests`/`httpx`,
  absent from the `app` group); wrap every external read (git, `.dvc`, MLflow env) so missing
  Phase-2/3/4 tooling degrades to a note instead of an exception.

**Milestone done when:** an experiment is tracked in MLflow, `dvc pull` restores data/models,
the multi-service stack runs.
**Kickoff mapping:** Phase 2 "Microservices, tracking & version control".

---

## Milestone 3 — Orchestration & Deployment · Phase 3 · target **Aug 7**

**Goal:** end-to-end automated pipeline, secured + scalable deployment. *(Detail firms up during M2.)*

> 📋 **Board-ready breakdown** — per-member cards with subtask checklists, branch names and
> acceptance criteria: [`docs/TODO.md`](docs/TODO.md).

**What you'll learn:**
- **Pipeline orchestration** — a DAG (directed acyclic graph of steps) replaces "run these
  scripts in the right order by hand".
- **Continuous deployment & rollback** — shipping and un-shipping become routine, not events.
- **API hardening** — a public endpoint without auth and rate limits is an incident waiting to happen.

### Task 3.1 — Orchestrated pipeline, data half · **Owner: Marco**

- **Why:** orchestration encodes the pipeline's order and retries in code, so the whole chain
  runs from one trigger instead of tribal knowledge.
- **Do:** the ingest → preprocess steps of an **Airflow** DAG (mandatory course module,
  Sprint 3 Jul 20–31), DVC-aware (pull inputs, push outputs). Coordinates with Task 3.2
  (the model half of the same DAG).
- **Done when:** triggering the DAG produces fresh, versioned `data/processed/` without manual steps.
- **Gotcha:** the DAG runs in its own container — mount/pull data
  explicitly, don't assume the host's files.

### Task 3.2 — Orchestrated pipeline, model half · **Owner: Dilshana**

- **Why:** a conditional-promotion gate turns "we retrained" into "we retrained and deployed
  *only if* it's better" — automation with a quality bar.
- **Do:** the train → evaluate → conditionally-promote steps of the DAG: promote to
  `Production` only if macro-F1 beats the current baseline (→ `docs/ML_CANVAS.md`).
- **Done when:** one trigger runs the whole chain and a *worse* model is **not** promoted.
- **Gotcha:** compare against the *currently served* model's metric,
  not a hard-coded number.

### Task 3.3 — API security · **Owner: Mykola**

- **Why:** authentication, input validation and rate limiting are the minimum bar for exposing
  a model to the outside world.
- **Files:** `api/main.py`, `tests/test_api.py`.
- **Do:** add auth (API key / JWT) as a FastAPI dependency; validate/limit input size; add
  basic rate limiting. Tests for authorized/unauthorized/flooded calls.
- **Done when:** unauthenticated `/predict` is rejected (401/403) and tests prove it.
- **Gotcha:** keep `/health` unauthenticated — Docker healthchecks and
  Kubernetes probes need it.

### Task 3.4 — CI/CD + scalable deployment · **Owner: Luc**

- **Why:** continuous deployment with rollback means a bad release is a one-command revert,
  and replicas mean one crashed container doesn't take the service down.
- **Files:** `.github/workflows/ci.yml`, new `k8s/` manifests.
- **Do:** extend CI with a build+deploy job; keep the previous image tag for rollback.
  Write Kubernetes manifests (Deployment + Service) from the compose setup. *Course timing:*
  the k8s module opens Aug 27 (after the M3 target) — land the CI/CD half by Aug 7, do the
  k8s hands-on Aug 27 – Sep 4, hard stop at the Sep 9 freeze (agree scope with the mentor).
- **Done when:** a push deploys; a bad deploy reverts in one step; the API runs with >1 replica
  behind a service.
- **Gotcha:** tag images with the git SHA, not `latest` — `latest`
  makes rollback meaningless.

**Kickoff mapping:** Phase 3 "Orchestration & Deployment".

---

## Milestone 4 — Monitoring & Maintenance · Phase 4 · target **Sep 4**

**Goal:** detect degradation and react automatically. New-data options for drift/retraining
→ `docs/DATA_SOURCES.md`.

> 📋 **Board-ready breakdown** — per-member cards with subtask checklists, branch names and
> acceptance criteria: [`docs/TODO.md`](docs/TODO.md).

**What you'll learn:**
- **Observability** — metrics, dashboards and alerts turn "is it working?" into a glance.
- **Drift detection** — production data slowly stops looking like training data; catch it.
- **Closing the loop** — retraining triggered by monitoring is what makes an ML system
  self-maintaining.

### Task 4.1 — Drift detection with Evidently · **Owner: Mykola**

- **Why:** drift detection compares live inputs against the training distribution — the early
  warning that model quality is about to drop, before labels confirm it.
- **Do:** reference (train) vs current (recent `/predict` inputs) Evidently reports;
  data-quality checks. Simulate drift by replaying a skewed category slice
  (`docs/DATA_SOURCES.md`, Option 3).
- **Done when:** an injected shift produces a drift report/flag.
- **Gotcha:** you need to *store* incoming requests to have a
  "current" dataset — add that first.

### Task 4.2 — Dashboards + alerts with Prometheus & Grafana · **Owner: Dilshana**

- **Why:** dashboards and alert rules are how a team *operates* a model — latency, throughput
  and drift on one screen, with a ping when a threshold breaks.
- **Files:** `docker-compose.yml` (add `prometheus` + `grafana`), Grafana dashboard JSON,
  alert rules.
- **Do:** scrape the API's `/metrics` (Task 4.3); dashboards for latency, throughput, drift,
  system health; alert rules on top.
- **Done when:** a Grafana board shows live metrics and an alert fires on drift.
- **Gotcha:** Prometheus scrapes over the compose network — target
  `api:8000`, not `localhost`.

### Task 4.3 — Metrics endpoint, API docs & maintenance guide · **Owner: Marco**

- **Why:** instrumentation exposes what the service is doing; the maintenance guide is the
  course's "could someone else operate this?" success criterion.
- **Files:** `api/main.py` (add `/metrics` via `prometheus-fastapi-instrumentator`),
  `docs/` (maintenance guide), FastAPI `/docs` (auto — verify completeness).
- **Done when:** `/metrics` serves Prometheus format; the maintenance guide covers
  update/rollback/retrain procedures. *Nice-to-have:* polish the Streamlit demo
  (`make setup-full && make app`).

### Task 4.4 — Automated retraining · **Owner: Luc**

- **Why:** this closes the MLOps loop — monitoring detects decay, the orchestrator retrains,
  the gate promotes only improvements, unattended.
- **Do:** trigger the M3 DAG on drift signal (Task 4.1) or schedule; auto-promote only if the
  new model beats the current one; replay new data per `docs/DATA_SOURCES.md` Option 3.
- **Done when:** retraining runs unattended and the API picks up the new model.
- **Gotcha:** guard against retrain loops — a drift flag that never
  clears will retrain forever; add a cooldown.

**Kickoff mapping:** Phase 4 "Monitoring & Maintenance".

---

## Milestone 5 — Defence · **Sep 14** (freeze **Sep 9**)

Everything below happens in the freeze window Sep 9–14 — the system itself is finished and
demoable by Sep 9.

- End-to-end demo script: data → train (tracked) → deploy → predict → monitor → retrain.
- Slides: problem (→ `docs/ML_CANVAS.md`), architecture diagram, results (macro-F1 vs
  baseline), MLOps decisions (why uv/Docker/MLflow/DVC/Evidently), lessons learned.
- **Each member presents the track they owned** plus one slide on a track they rotated
  through earlier (rotation → `docs/PROJECT_MANAGEMENT.md`). Each owner documents their own
  component; Marco compiles.
- Team retro: what went well / to improve / next.

> ℹ️ First `make test` on a new machine downloads NLTK data (`stopwords`, `wordnet`, `omw-1.4`) — needs network once.
