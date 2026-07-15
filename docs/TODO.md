# Milestone 2 — Task board source · Phase 2 · target **Jul 24**

**Goal:** trace every experiment (MLflow), version data + models (DVC), split the stack into
services (compose). One card per member, matching the Phase-2 rotation
([PROJECT_MANAGEMENT.md §1](PROJECT_MANAGEMENT.md)): Luc → A (modeling/tracking),
Dilshana → D (serving), Mykola → C (data/versioning), Marco → B (infra).

This file is the **copy-paste source for the GitHub Project board**: create one issue per card
(title, labels, assignee below), paste the card body as the issue description, keep the
checklists as task lists so GitHub renders progress. The board — not this file — is the live
status ([PROJECT_MANAGEMENT.md §2](PROJECT_MANAGEMENT.md)). Roadmap context →
[MILESTONES.md](../MILESTONES.md) Milestone 2.

**Infra decisions (agreed Jul 15, 2026):**

- **DagsHub** hosts the shared MLflow tracking server **and** the DVC remote (S3). One free
  hosted platform, everyone sees the same runs and data; compose adds local-dev services only.
- **Model registry uses aliases** (`models:/<name>@production`) — stage transitions are
  deprecated in MLflow 2.x.
- **Git flow uses the `dev` integration branch** — all PRs target `dev`; `dev → main` merges
  once per milestone ([CONTRIBUTING.md §3](../CONTRIBUTING.md)).

**Shared env contract** (set in `.env`, documented in `.env.example`, values never committed):

- `MLFLOW_TRACKING_URI` — the DagsHub MLflow endpoint of the team repo (the repo URL with the
  `.mlflow` suffix). Unset ⇒ tools fall back to local `./mlruns`.
- `MLFLOW_TRACKING_USERNAME` — DagsHub username.
- `MLFLOW_TRACKING_PASSWORD` — DagsHub personal access token (acts as the password).

**Dependency order:**

```
2.4-A (Marco, day 1: dev branch + DagsHub + env plumbing)
   └─→ 2.1 (Luc: tracking)  ──→ 2.2 (Dilshana: registry needs runs to register)
   └─→ 2.3 (Mykola: DVC — parallel, only needs DagsHub creds)
   └─→ 2.4-B (Marco: compose services — parallel after A)
```

Everyone: branch from `dev`, PR to `dev`, squash-merge, `Closes #NN`, ≥1 review, CI green
([CONTRIBUTING.md](../CONTRIBUTING.md)). Definition of Done →
[PROJECT_MANAGEMENT.md §4](PROJECT_MANAGEMENT.md).

### Working in parallel on `dev` (avoid merge pain)

After Card 2.4 part A creates `dev` and the DagsHub access, **all four cards can be worked at
the same time on separate branches** — the design keeps each card's core logic in different
files (`build_pipelines.py`, `registry.py`/`api`, `.dvc/`, `docker-compose.yml`). A few files
are touched by more than one card; handle them so branches coexist:

| Shared file | Touched by | Rule |
|---|---|---|
| `.env.example` | 2.1, 2.2, 2.4 | Append your vars under a clearly-commented section header; don't reorder existing lines. |
| `README.md` | 2.1, 2.2, 2.3 | Each card edits a **different** section (train row / API section / quickstart) — edit only yours. |
| `pyproject.toml` + `uv.lock` | 2.1 (`track`), 2.3 (`dvc`) | Add your dependency group only. If `uv.lock` conflicts on merge, **regenerate it** (`uv lock`), never hand-resolve. |
| `CONTRIBUTING.md` | 2.3 (§7), 2.4 (§6) | Different sections — edit only yours. |

General: keep each PR small, **rebase on `dev` right before opening the PR** (pull in whatever
merged first), and re-run `make test` after the rebase. Whoever merges second on a shared file
rebases, not the first.

---

## Card 2.1 — Add MLflow experiment tracking to training

| | |
|---|---|
| **Assignee** | **Luc** |
| **Labels** | `phase-2` `type:model` `priority:high` |
| **Branch** | `feature/lu-mlflow-tracking` (from `dev`) |
| **Depends on** | Card 2.4 part A (DagsHub repo + token exist) |
| **Files** | `scripts/build_pipelines.py` · `pyproject.toml` + `uv.lock` · `.env.example` · `tests/` · README (`make train` row) |

**Why:** every training run becomes reproducible and comparable — params, macro-F1 and the
produced model recorded automatically instead of in someone's terminal history.

**Where:** `scripts/build_pipelines.py`, inside `main()` — the double loop over `TARGETS` ×
schemas. Everything worth logging already exists in the loop body: the model id (used for the
output filenames), the tuned hyperparameters read from the checkpoint sidecar, the fixed
TF-IDF settings (`TFIDF_KWARGS`) and random seed, the three test metrics (macro-F1,
weighted-F1, accuracy), and the fitted pipeline object itself.

### Subtasks

- [ ] Add `mlflow` to a new optional `track` dependency group via uv (keeps MLflow out of the
      torch-free API image); commit `pyproject.toml` and `uv.lock` together.
- [ ] At the start of `main()`, configure tracking once: take the tracking URI from
      `MLFLOW_TRACKING_URI` and fall back to a local `mlruns/` directory (already
      git-ignored) when unset — offline runs and CI must keep working. Set the experiment
      name to **`trustpilot-reviews`** (one experiment for the whole project). DagsHub
      authentication is picked up automatically from the two `MLFLOW_TRACKING_*` credential
      env vars.
- [ ] Wrap each fit+eval iteration in an MLflow run (`mlflow.start_run`), using the model id
      (algo + variant + schema) as the run name. Per run, log:
      - **tags** — algorithm, schema, and whether the XGBoost 5-class label shift applied;
      - **params** — the tuned hyperparameters, the TF-IDF settings (prefixed so they don't
        collide with estimator params), and the random seed;
      - **metrics** — macro-F1 (primary), weighted-F1, accuracy, plus the tuning-vs-pipeline
        parity delta already computed for the console table;
      - **the model** — via the sklearn model-logging API, with a small input example (a
        couple of lemmatised strings) so the artifact gets a signature. Use the artifact
        path **`model`** — Card 2.2 registers runs by that path.
- [ ] **Backfill DistilBERT eval** (repo-review fix): the two DistilBERT eval JSONs under
      `models/pipelines/` carry null weighted-F1/accuracy (only the 3-class macro-F1 0.6811
      is real, transcribed from a notebook not in the repo). Log the known values as one
      MLflow run tagged as a backfill; fill the JSON nulls if the eval can be re-run,
      otherwise add a note field in the JSON stating they are unavailable. Keep the structure
      `src/registry.py::_scan_bert` reads (metrics nested under the phase-2 key).
- [ ] Smoke test (new `tests/test_tracking.py`): point the tracking URI at a pytest tmp
      directory, run one tiny fit through the same logging path, assert a run exists with a
      macro-F1 metric and a model artifact. Must not need network — CI has no MLflow server.
- [ ] Docs: README `make train` row mentions tracking; `.env.example` documents the three
      `MLFLOW_*` vars (see the shared env contract at the top of this file).

### Done when

`make train` with `MLFLOW_TRACKING_URI` set produces one run per (algo × schema) in the
DagsHub MLflow UI — params + macro-F1 + loadable model artifact; with the env var unset it
still trains and logs locally; `uv run pytest` green without any server.

---

## Card 2.2 — Serve models from the MLflow registry (alias-based)

| | |
|---|---|
| **Assignee** | **Dilshana** |
| **Labels** | `phase-2` `type:api` `priority:high` |
| **Branch** | `feature/di-mlflow-registry` (from `dev`) |
| **Depends on** | Card 2.4 part A only (DagsHub repo + token). **Runs parallel to 2.1** — build the loader, API wiring and all three tests against a run you log yourself locally; only the *final* promotion of the team's real best model waits for 2.1 to merge. |
| **Files** | `api/main.py` · `src/registry.py` · new `scripts/register_model.py` · `tests/test_api.py` · `.env.example` · README API section |

**Why:** a model registry decouples "a new model exists" from "the API serves it" —
deployment becomes a one-step alias move, rollback is moving the alias back.

### Design

- Registered model names: **`reviews-classifier-3class`** and **`reviews-classifier-5class`**.
- Load precedence at API startup, per schema:
  1. if `MLFLOW_TRACKING_URI` is set, try loading the registry model addressed by name +
     the **`production` alias** (the `models:/<name>@production` URI form);
  2. on any failure (no server, no registered model, no alias, timeout) **or** when the env
     var is unset, use the existing local-joblib path (`src/registry.py` scan →
     `ModelEntry.joblib_path`), exactly as today. The API must always boot offline; CI has
     no MLflow server.
- The loaded object is the same TF-IDF + classifier pipeline either way — the downstream
  `src/inference.predict_classical` code path stays unchanged.

### Subtasks

- [ ] Promotion tooling (`scripts/register_model.py`): a
      small CLI that, given a schema and optionally a run id, finds the chosen run in the
      `trustpilot-reviews` experiment (default: latest run of the best model, currently
      LogReg 3-class), registers its `model` artifact under the schema's registry name, and
      points the `production` alias at the new version. Use the MLflow client API for alias
      assignment — aliases, not deprecated stages. Build and test it against a run **you log
      yourself** (log a quick LogReg pipeline to your local `mlruns/`); the *final* promotion
      of the team's real best model is the one step that waits for 2.1 to merge. Rollback =
      pointing the alias back at the
      previous version; document both flows.
- [ ] `src/registry.py`: add a `load_production(schema)` helper implementing the precedence
      above, returning the pipeline plus a source marker (registry vs local joblib). Import
      MLflow lazily inside the function so plain module import stays MLflow-free.
- [ ] `api/main.py`: resolve the served model through that helper at startup (or first
      request); extend the `/models` response with the source field so it's visible which
      path served each entry.
- [ ] Tests (`tests/test_api.py`), three cases:
      - fallback: env var unset → API boots and `/predict` serves local joblibs (protects CI);
      - registry hit: monkeypatch the MLflow model-loading call to return a real local
        pipeline, set a fake tracking URI, assert the source reports the registry;
      - registry failure: monkeypatch the load to raise, assert clean fallback to the local
        joblib source.
- [ ] Docs: README API section + `.env.example` explain the precedence and the alias-move
      deploy/rollback flow.

### Done when

Moving the `production` alias to a different model version in MLflow changes what the API
serves — **no code edit, no image rebuild** (restart allowed). Without the env var the API
boots offline and CI stays green.

---

## Card 2.3 — Version data + large models with DVC (DagsHub remote)

| | |
|---|---|
| **Assignee** | **Mykola** |
| **Labels** | `phase-2` `type:data` `priority:high` |
| **Branch** | `feature/mk-dvc-versioning` (from `dev`) |
| **Depends on** | Card 2.4 part A (DagsHub repo + token exist) — otherwise parallel |
| **Files** | `.dvc/config` · `*.dvc` pointers · `.gitignore` · `Makefile` · `CONTRIBUTING.md §7` · README quickstart |

**Why:** datasets and weights get the same audit trail git gives code — you can name the exact
bytes a model trained on without bloating the repo.

### What goes to DVC vs stays in git

| Artifact | Where | Why |
|---|---|---|
| `data/processed/train.csv` + `test.csv` (~123k rows) | **DVC** | large, regenerable via `make data` |
| `models/pipelines/RandomForest__…__5-class.joblib` (64 MB) | **DVC** | currently git-ignored, only its JSON sidecar committed |
| `models/pipelines/RandomForest__…__3-class.joblib` (10.8 MB) | **DVC** (migrate out of git) | repo-review fix — largest tracked file after `uv.lock` |
| all other served pipelines (few hundred KB) + all JSON sidecars | **git** | API/tests must work from clean clone without DVC |
| `data/processed/eda_sample.parquet` + `eda_summary.json` | **git** | small committed EDA artifacts, unchanged |

### Subtasks

- [ ] Add DVC with the S3 extra to the `dev` dependency group via uv (the DagsHub remote is
      S3-backed); commit `pyproject.toml` and `uv.lock`.
- [ ] Initialise DVC in the repo (creates `.dvc/` and `.dvcignore`), disable DVC analytics,
      commit what init stages.
- [ ] Configure the default remote to point at the team repo's DagsHub S3 endpoint (the repo
      URL with the `.s3` suffix). **Credentials go in via DVC's `--local` config flag only**
      — they land in the git-ignored `.dvc/config.local` (DagsHub token as both access key
      and secret); only the committed `.dvc/config` carries the url/endpoint.
- [ ] Data: regenerate the CSVs locally with `make data`, then put `train.csv` and
      `test.csv` under DVC tracking. The root `.gitignore` already ignores `data/` — let DVC
      manage its own ignore entries next to the data, and make sure the two committed EDA
      artifacts stay un-ignored as today.
- [ ] Models:
      - RF 5-class weight (rebuild with `make train` if absent locally — it's git-ignored):
        put it under DVC tracking;
      - RF 3-class weight: **remove it from git tracking while keeping the file on disk**
        (git's cached-removal mode), then put it under DVC tracking. Its JSON sidecar stays
        in git. Clean up the now-obsolete explicit ignore/un-ignore lines for these two
        joblibs in the root `.gitignore` — DVC writes its own.
- [ ] Publish in the right order: commit the `.dvc` pointer files, DVC-managed ignore files
      and `.dvc/config` to git; push the binaries to the remote with DVC **before** pushing
      the git branch, so reviewers can pull immediately.
- [ ] Graceful-degrade check: without a DVC pull, both RF entries lack a local joblib —
      `src/registry.py` already renders such entries leaderboard-only (`loadable` false, as
      the 5-class does today). Run the test suite on a clone without pulling — must stay green.
- [ ] `Makefile`: add a `pull` target wrapping DVC pull, plus a `make help` line.
- [ ] Docs: `CONTRIBUTING.md §7` gets the concrete day-to-day loop (track new artifacts with
      `dvc add`, refresh re-generated ones with `dvc commit`, push data before pushing git,
      pull data after pulling git); README quickstart notes the optional `make pull` for
      full-size data/models.

### Done when

Fresh clone + credentials in `.dvc/config.local` + `make pull` restores both CSVs and both RF
weights; `make test` and `make api` pass on a clean clone **without** DVC; no git-tracked
file over ~1 MB remains except `uv.lock`.

---

## Card 2.4 — Shared infra: dev branch, DagsHub, microservices compose split

| | |
|---|---|
| **Assignee** | **Marco** |
| **Labels** | `phase-2` `type:infra` `priority:high` |
| **Branch** | `feature/ma-compose-services` (from `dev`; part A is repo-settings work, day 1) |
| **Depends on** | — (part A unblocks everyone) |
| **Files** | `docker-compose.yml` · `Dockerfile` · `.env.example` · `.dockerignore` · `.pre-commit-config.yaml` (new) · `CONTRIBUTING.md` |

**Why:** one container per responsibility (API / training / tracking) is the architecture step
that makes orchestration, scaling and independent deploys possible — and part A is the shared
plumbing every other card needs on day 1.

### Subtasks — part A (day 1, unblocks 2.1/2.2/2.3)

- [ ] Create the `dev` branch from up-to-date `main` and push it to origin.
- [ ] GitHub settings (manual): make `dev` the default branch; add branch protection on
      **both** `main` and `dev` — require a PR, require the CI status check, require one
      approving review, forbid force pushes.
- [ ] Create the DagsHub repository **connected to** the GitHub repo
      (`Liora-Team/supply-chain-ml-ops`) via DagsHub's "connect a repository" flow; add all
      four members as collaborators.
- [ ] Each member generates a DagsHub personal access token (DagsHub settings → tokens);
      update `.env.example` with the three `MLFLOW_*` vars from the shared env contract at
      the top of this file. Tokens are never committed.

### Subtasks — part B (compose split)

- [ ] `docker-compose.yml` — one service per responsibility:
      - **api** (exists): add passthrough of the three `MLFLOW_*` env vars so Card 2.2's
        registry load works in-container;
      - **training**: one-shot job that runs the training script; built from a dedicated
        Dockerfile target that adds the `track` dependency group; gated behind a compose
        **profile** (e.g. `train`) so a plain `docker compose up` never retrains; mounts the
        `data/` and `models/` directories; gets the `MLFLOW_*` passthrough;
      - **mlflow**: local tracking UI for offline dev only (team truth lives on DagsHub) —
        official MLflow image, server on port 5000, run store on a named volume;
      - **bert**: DistilBERT as its own torch-only service, built from a Dockerfile target
        that installs the `bert` dependency group; behind its own profile; weights located
        via the `DISTILBERT_*` env vars ([CONTRIBUTING.md §7](../CONTRIBUTING.md)). Keeps the
        classical API image slim.
- [ ] `Dockerfile`: convert to multi-stage with named targets — the existing image becomes
      the default `api` target; add `training` and `bert` targets that differ only in which
      uv dependency groups they sync.
- [ ] Inside containers, services address each other by **service name** over the compose
      network (e.g. the mlflow service on port 5000), never `localhost`.
- [ ] Repo hygiene (repo-review fixes):
      - new `.pre-commit-config.yaml` running the same ruff + black checks as CI; add
        `pre-commit` to the `dev` dependency group and a `CONTRIBUTING.md §6` line telling
        contributors to install the hook once;
      - prune dead `.dockerignore` entries (`notebooks/`, `configs/`, `requirements.txt`,
        `.gstack/`).
- [ ] Verify: `docker compose up` → api + mlflow healthy (healthchecks pass); the train
      profile runs to completion and its run appears in the configured tracking server.

### Done when

`dev` is the default branch with protections on both long-lived branches; all members have
DagsHub access + tokens; `docker compose up` brings up `api` + `mlflow` healthy; the train
profile runs a tracked training whose model the API can then serve.

---

## Importing cards as GitHub issues

Create one issue per card — title, labels and assignee from the card's table, body = the card
section from this file (keep the checklists so GitHub renders progress). Use the web UI or the
`gh issue create` CLI. Then add the four issues to the Project board (**To Do**) and set
Iteration = current sprint ([PROJECT_MANAGEMENT.md §2](PROJECT_MANAGEMENT.md) for board and
label conventions).

## Milestone 2 exit checklist

- [ ] An experiment is tracked in MLflow (DagsHub UI shows params + macro-F1 + artifact)
- [ ] `dvc pull` on a fresh clone restores data + large models
- [ ] The multi-service stack runs (`docker compose up`)
- [ ] Moving the `production` alias changes what the API serves
- [ ] `dev → main` milestone merge done, tag `milestone-2`
