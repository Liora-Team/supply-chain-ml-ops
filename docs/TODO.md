# Milestones 2–4 — Task board source

**The big picture:** trace every experiment (MLflow), version data + models (DVC), split into
services (M2, **Jul 24**) → orchestrate the pipeline, secure and deploy (M3, **Aug 7**) →
monitor, detect drift, retrain automatically (M4, **Sep 4**) → **project freeze Sep 9**
(ready ≥5 days before the defence; Sep 9–14 is rehearsal/slides/demo-recording only) →
defence **Sep 14**.

This file is the **copy-paste source for the GitHub Project board**: one issue per card
(title, labels, assignee below), card body = issue description, checklists stay as task lists
so GitHub renders progress. The board — not this file — is the live status
([PROJECT_MANAGEMENT.md §2](PROJECT_MANAGEMENT.md)). Roadmap context →
[MILESTONES.md](../MILESTONES.md).

Owners follow the rotation matrix ([PROJECT_MANAGEMENT.md §1](PROJECT_MANAGEMENT.md)):

| | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|
| **Marco** | B — infra | C — data | D — serving |
| **Mykola** | C — data | D — serving | A — modeling |
| **Dilshana** | D — serving | A — modeling | B — infra |
| **Luc** | A — modeling | B — infra | C — data |

**Course module calendar** (plan work so the module lands just before you need it):

| Course sprint | Dates | Modules | Feeds our cards |
|---|---|---|---|
| Sprint 3 — Versioning and Deployment | Jul 20 – Jul 31 | **Airflow (mandatory)** · BentoML serving | 3.1, 3.2 (Airflow DAG) — module ends a week before the M3 target |
| Sprint 4 — Monitoring | Aug 3 – Aug 26 | **Drift Monitoring (mandatory)** · Prometheus & Grafana | 4.1, 4.2 — module arrives *before* M4; both can start during the M3 window if capacity allows |
| Sprint 5 — Scaling & MLOps platform | Aug 27 – Sep 18 | Kubernetes | 3.4's k8s half — module lands **after** the M3 target; k8s work itself must finish by **Sep 4** (the module window outlives the Sep 9 freeze and the Sep 14 defence); see card 3.4 |

Course "sprints" are the learning platform's multi-week module windows — not the project's
weekly check-in cadence (that one is the "Project week(s)" column in
[MILESTONES.md](../MILESTONES.md)).

**Architecture decisions (agreed Jul 15, 2026):**

- **No model binaries in git.** Git holds code + small metadata (JSON metric sidecars);
  **DVC** holds data + all model weights (every `.joblib`, DistilBERT); **MLflow** holds
  runs + the registry. CI restores weights with a DVC pull authenticated by a repo secret.
- **DagsHub** hosts the shared MLflow tracking server **and** the DVC remote (S3). One free
  hosted platform; compose adds local-dev services only.
- **Model registry uses aliases** (`models:/<name>@production`) — stage transitions are
  deprecated in MLflow 2.x.
- **Git flow uses the `dev` integration branch** — all PRs target `dev`; `dev → main` merges
  once per milestone, tagged ([CONTRIBUTING.md §3](../CONTRIBUTING.md)).
- **Reverse proxy = nginx.** A single public entry point fronts the stack — a `proxy` (nginx)
  service in compose (Card 2.4-B), `nginx-ingress` in k8s (Card 3.4). **TLS terminates at the
  proxy/ingress** (Card 3.3); internal services (`api`, `mlflow`) speak plain HTTP over the
  private network and are never published directly. This is the reverse-proxy + TLS role —
  **not** a load balancer (the k8s Service already balances replicas).

**Security thread** (where each security aspect lives — one row per defence-slide bullet):

| Aspect | Where it's handled |
|---|---|
| Secrets never in git | [CONTRIBUTING.md §7](../CONTRIBUTING.md) rule · DVC creds in git-ignored `--local` config (2.3) · `DAGSHUB_TOKEN` as Actions secret (2.4-A) |
| Secret scanning | gitleaks pre-commit hook + GitHub secret scanning & push protection (2.4-A) |
| Dependency vulnerabilities | Dependabot alerts (2.4-A) |
| Container hardening | non-root user, pinned base image, no secrets in layers (2.4-B) |
| Branch protection | PRs-only, CI required, review required on `main` + `dev` (2.4-A) |
| API security | auth, input validation, rate limiting (3.3) |
| TLS / HTTPS in transit | terminated at the nginx reverse proxy (2.4-B) / ingress (3.4); app speaks HTTP behind it (3.3) |
| Single public entry / internal services not exposed | nginx reverse proxy fronts the stack (2.4-B) / ingress (3.4) |
| Runtime secrets in prod | k8s Secrets, never baked into images (3.4) |
| Data privacy | public dataset; PII + request-retention note in the maintenance guide (4.3) |

**Shared env contract** (set in `.env`, documented in `.env.example`, values never committed):

- `MLFLOW_TRACKING_URI` — the DagsHub MLflow endpoint of the team repo (the repo URL with the
  `.mlflow` suffix). Unset ⇒ tools fall back to local `./mlruns`.
- `MLFLOW_TRACKING_USERNAME` — DagsHub username.
- `MLFLOW_TRACKING_PASSWORD` — DagsHub personal access token (acts as the password).

**Dependency order:**

```
M2: 2.4-A (Marco, day 1: dev branch + DagsHub + secrets)
      └─→ 2.1 (Luc: tracking) ─┐
      └─→ 2.2 (Dilshana: registry — parallel; final promote waits on 2.1)
      └─→ 2.3 (Mykola: DVC)    ├─→ M2 exit: dev → main, tag milestone-2
      └─→ 2.4-B (Marco: compose split + nginx reverse proxy)
      └─→ 2.5 (Marco: Ops dashboard — parallel, buildable now)
M3: 3.1 (data DAG) ↔ 3.2 (model DAG — same orchestrator, agreed interface)
    3.3 (API security + TLS at the proxy) and 3.4 (CI/CD + k8s incl. ingress) fully parallel
M4: 4.3 (/metrics) ─→ 4.2 (Prometheus/Grafana scrapes it)
    4.1 (drift) ─→ 4.4 (retrain trigger consumes the drift signal)
```

Everyone: branch from `dev`, PR to `dev`, squash-merge, `Closes #NN`, ≥1 review, CI green
([CONTRIBUTING.md](../CONTRIBUTING.md)). Definition of Done →
[PROJECT_MANAGEMENT.md §4](PROJECT_MANAGEMENT.md).

### Working in parallel on `dev` (avoid merge pain)

After Card 2.4-A creates `dev` and the DagsHub access, **all Phase-2 cards can be worked at
the same time on separate branches** — each card's core logic lives in different files
(`build_pipelines.py`, `registry.py`/`api`, `.dvc/`, `docker-compose.yml`). A few files are
touched by more than one card; handle them so branches coexist:

| Shared file | Touched by | Rule |
|---|---|---|
| `.env.example` | 2.1, 2.2, 2.4-A | Append your vars under a clearly-commented section header; don't reorder existing lines. |
| `README.md` | 2.1, 2.2, 2.3 | Each card edits a **different** section (train row / API section / quickstart) — edit only yours. |
| `pyproject.toml` + `uv.lock` | 2.1 (`track`), 2.3 (`dvc`) | Add your dependency group only. If `uv.lock` conflicts on merge, **regenerate it** (`uv lock`), never hand-resolve. |
| `CONTRIBUTING.md` | 2.3 (§7), 2.4-A (§6) | Different sections — edit only yours. |
| `.github/workflows/ci.yml` | 2.3 (dvc pull step) | Only 2.3 touches CI in Phase 2. |

General: keep each PR small, **rebase on `dev` right before opening the PR** (pull in whatever
merged first), and re-run `make test` after the rebase. Whoever merges second on a shared file
rebases, not the first.

---
---

# Milestone 2 — Microservices, Tracking & Versioning · target **Jul 24**

---

## Card 2.1 — Add MLflow experiment tracking to training

| | |
|---|---|
| **Assignee** | **Luc** |
| **Labels** | `phase-2` `type:model` `priority:high` |
| **Branch** | `feature/lu-mlflow-tracking` (from `dev`) |
| **Depends on** | Card 2.4-A (DagsHub repo + token exist) |
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
      MLflow run tagged as a backfill. The weights themselves are being DVC-tracked in Card
      2.3 (source: Marco's machine / team GDrive) — once restored, **re-run the eval on the
      test split** to fill the null fields properly; coordinate with Mykola. Keep the
      structure `src/registry.py::_scan_bert` reads (metrics nested under the phase-2 key).
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
| **Depends on** | Card 2.4-A only (DagsHub repo + token). **Runs parallel to 2.1** — build the loader, API wiring and all three tests against a run you log yourself locally; only the *final* promotion of the team's real best model waits for 2.1 to merge. |
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
     `ModelEntry.joblib_path`) — after Card 2.3 lands, that local file is the DVC-restored
     copy (`make pull`), no longer a git-committed one. The API must always boot offline
     once weights are pulled; CI has no MLflow server.
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
      pointing the alias back at the previous version; document both flows.
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
boots offline (weights present via `make pull`) and CI stays green.

---

## Card 2.3 — Move data + all model weights to DVC (DagsHub remote)

| | |
|---|---|
| **Assignee** | **Mykola** |
| **Labels** | `phase-2` `type:data` `priority:high` |
| **Branch** | `feature/mk-dvc-versioning` (from `dev`) |
| **Depends on** | Card 2.4-A (DagsHub repo + token + `DAGSHUB_TOKEN` Actions secret) |
| **Files** | `.dvc/config` · `*.dvc` pointers · `.gitignore` · `.github/workflows/ci.yml` · `Makefile` · `CONTRIBUTING.md §7` · README quickstart |

**Why:** the team decision is **no model binaries in git** — git holds code + small metadata,
DVC holds data + weights, MLflow holds runs. Datasets and weights get the same audit trail
git gives code, and the repo stays small.

### What goes to DVC vs stays in git

| Artifact | Where |
|---|---|
| `data/processed/train.csv` + `test.csv` (~123k rows) | **DVC** |
| **all** `models/pipelines/*.joblib` — the 9 currently committed (incl. the 10.8 MB RF 3-class) **plus** the git-ignored 64 MB RF 5-class | **DVC** (migrate out of git) |
| DistilBERT fine-tune dirs `models/distilbert_{3,5}class/final/` (2 × ~255 MB) | **DVC** — source: Marco's machine; interim backup: team GDrive `Liora - resources/supply_chain_resources/Archive.zip` (<https://drive.google.com/drive/folders/1IFYanbdAT6WlG7RQzFVC5C5fUyp8u3yQ>) |
| all JSON metric sidecars (`models/checkpoints/*.json`, `models/pipelines/*.json`) | **git** — small metadata, the leaderboard reads them without DVC |
| `data/processed/eda_sample.parquet` + `eda_summary.json` | **git** — small committed EDA artifacts |

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
      manage its own ignore entries next to the data, and keep the two committed EDA
      artifacts un-ignored as today.
- [ ] Models — migrate **every** `.joblib` out of git: for each of the 9 committed pipeline
      joblibs, remove it from git tracking while keeping the file on disk (git's
      cached-removal mode), then put it under DVC tracking. Also DVC-track the 64 MB RF
      5-class weight (rebuild with `make train` if absent locally). All JSON sidecars stay in
      git. Clean up the now-obsolete joblib ignore/un-ignore lines in the root `.gitignore` —
      DVC writes its own.
- [ ] **DistilBERT weights into DVC:** get the two fine-tune directories from Marco (source
      of truth is his machine; the team GDrive `Archive.zip` linked in the table above is
      the interim backup — Marco: verify the zip contains both `final/` dirs), place them at
      `models/distilbert_3class/final/` and `models/distilbert_5class/final/` (the exact
      default paths `src/registry.py` scans), then DVC-track both directories. Each dir must
      contain `config.json`, `model.safetensors` (~255 MB), `tokenizer.json`,
      `tokenizer_config.json` (fast tokenizer — no `vocab.txt` needed) and
      `eval_metrics.json`. Beware: in Marco's other project these dirs are **absolute-path
      symlinks** — copy the *resolved* directories, not the links. After `make pull` the
      registry should mark both bert entries loadable with no env vars set — the
      `DISTILBERT_*` vars stay only as an override for a custom local path (decision:
      DVC is the single model store; no HF-hub hosting).
- [ ] Publish in the right order: commit the `.dvc` pointer files, DVC-managed ignore files
      and `.dvc/config` to git; push the binaries to the remote with DVC **before** pushing
      the git branch, so reviewers can pull immediately.
- [ ] **CI** (`.github/workflows/ci.yml`): add a DVC pull step before pytest — install the
      dvc extra, write the DagsHub credentials from the **`DAGSHUB_TOKEN` repository secret**
      (created in Card 2.4-A) into the local DVC config, pull. Tests need the weights now
      that none are committed.
- [ ] `Makefile`: add a `pull` target wrapping DVC pull, plus a `make help` line.
- [ ] Docs — this changes the on-boarding story:
      - README quickstart: fresh clone now needs **one `make pull`** (with DagsHub creds)
        before `make test` / `make api`; update the "no dataset download and no training"
        claim accordingly;
      - `CONTRIBUTING.md §7`: the concrete day-to-day loop (track new artifacts with
        `dvc add`, refresh re-generated ones with `dvc commit`, push data before pushing
        git, pull data after pulling git) and the golden-rule wording (clean clone +
        `uv sync` + `make pull`).

### Done when

Fresh clone + credentials + `make pull` restores CSVs, all 10 classical weights **and both
DistilBERT dirs** (registry marks the bert entries loadable), then `make test` and `make api`
pass; CI is green pulling via the repo secret; `git ls-files` contains **zero `.joblib`** and
no tracked file over ~1 MB except `uv.lock`.

---

## Card 2.4-A — Day-1 shared infra setup (dev branch, DagsHub, secrets, hygiene)

| | |
|---|---|
| **Assignee** | **Marco** |
| **Labels** | `phase-2` `type:infra` `priority:high` |
| **Branch** | `feature/ma-infra-setup` (from `dev`, for the file changes; branch/settings work is done directly on GitHub) |
| **Depends on** | — (day 1, unblocks every other Phase-2 card) |
| **Files** | `.env.example` · `.pre-commit-config.yaml` (new) · `.dockerignore` · `CONTRIBUTING.md §6` |

**Why:** the shared plumbing every other card needs on day 1 — the integration branch, the
hosted MLflow/DVC platform, the credentials, and the local quality gate.

### Subtasks

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
- [ ] Add the **`DAGSHUB_TOKEN` repository secret** on GitHub (Settings → Secrets and
      variables → Actions) — Card 2.3's CI pull step authenticates with it.
- [ ] New `.pre-commit-config.yaml` running the same ruff + black checks as CI, **plus a
      gitleaks hook** (secret scanning before every commit — the cheapest insurance against
      a token landing in history); add `pre-commit` to the `dev` dependency group and a
      `CONTRIBUTING.md §6` line telling contributors to install the hook once.
- [ ] GitHub Settings (manual, free, one-time): enable **secret scanning + push
      protection** and **Dependabot alerts** under Security → Code security and analysis.
- [ ] Prune dead `.dockerignore` entries (`notebooks/`, `configs/`, `requirements.txt`,
      `.gstack/`).

### Done when

`dev` is the default branch with protections on both long-lived branches; all members have
DagsHub access + tokens; the `DAGSHUB_TOKEN` Actions secret exists; pre-commit runs ruff +
black locally on commit.

---

## Card 2.4-B — Microservices compose split

| | |
|---|---|
| **Assignee** | **Marco** |
| **Labels** | `phase-2` `type:infra` `priority:high` |
| **Branch** | `feature/ma-compose-services` (from `dev`) |
| **Depends on** | Card 2.4-A (env contract exists). Parallel to 2.1/2.2/2.3. |
| **Files** | `docker-compose.yml` · `Dockerfile` · `deploy/nginx/nginx.conf` (new) · README (compose notes) |

**Why:** one container per responsibility (API / training / tracking) is the architecture
step that makes orchestration, scaling and independent deploys possible.

### Subtasks

- [ ] `Dockerfile`: convert to multi-stage with named targets — the existing image becomes
      the default `api` target; add `training` and `bert` targets that differ only in which
      uv dependency groups they sync (`track` / `bert`).
- [ ] `docker-compose.yml` — one service per responsibility:
      - **api** (exists): add passthrough of the three `MLFLOW_*` env vars so Card 2.2's
        registry load works in-container;
      - **training**: one-shot job that runs the training script; built from the `training`
        Dockerfile target; gated behind a compose **profile** (e.g. `train`) so a plain
        `docker compose up` never retrains; mounts the `data/` and `models/` directories;
        gets the `MLFLOW_*` passthrough;
      - **mlflow**: local tracking UI for offline dev only (team truth lives on DagsHub) —
        official MLflow image, server on port 5000, run store on a named volume;
      - **bert**: DistilBERT as its own torch-only service from the `bert` Dockerfile
        target; behind its own profile; weights located via the `DISTILBERT_*` env vars
        ([CONTRIBUTING.md §7](../CONTRIBUTING.md)). Keeps the classical API image slim.
      - **proxy** (nginx): the **single public entry point** — publishes port 80 (443 when TLS
        lands in Card 3.3) and reverse-proxies `/` → `api:8000` and `/mlflow` → `mlflow:5000`
        over the compose network; `api` and `mlflow` stop being published directly. Config in
        `deploy/nginx/nginx.conf`. An `upstream` block lets it round-robin when `api` is scaled
        (`docker compose up --scale api=N`) — the LB comes free, but the point here is the
        single entry + a home for TLS.
- [ ] Inside containers, services address each other by **service name** over the compose
      network (e.g. the mlflow service on port 5000), never `localhost`.
- [ ] Dockerfile hardening: run as a **non-root user**, pin the base image to an exact tag
      (or digest), and keep secrets out of image layers — credentials arrive only via env
      passthrough at runtime.
- [ ] Verify: `docker compose up` → api + mlflow healthy (healthchecks pass) and reachable
      **through the proxy** (`curl localhost/health` hits the api via nginx); the train
      profile runs to completion and its run appears in the configured tracking server.

### Done when

`docker compose up` brings up `api` + `mlflow` healthy behind the `proxy` (nginx) as the single
published entry (`curl localhost/health` reaches the api through nginx); the train profile runs
a tracked training whose model the API can then serve; the bert profile builds and starts its
own torch image.

---

## Card 2.5 — Internal Ops dashboard (`pages/8_Ops.py`)

| | |
|---|---|
| **Assignee** | **Marco** |
| **Labels** | `phase-2` `type:infra` `priority:med` |
| **Branch** | `feature/ma-ops-dashboard` (from `dev`) |
| **Depends on** | — buildable now (registry + committed EDA artifact). MLflow/DVC panels light up after 2.1/2.2/2.3; Airflow/Grafana deep links after Phase 3/4. |
| **Files** | `pages/8_Ops.py` (new) · `app.py` (nav entry + "What's inside" row) |

**Why:** the seven existing pages are the customer-facing demo; the team has no single
operator view of project state. An internal cockpit — model registry, data & versioning,
tracking config — turns "is it working?" into one glance and gives every later phase a home
to surface its tool (MLflow, Airflow, Grafana) instead of scattering UIs.

### Subtasks

- [ ] New `pages/8_Ops.py`, registered in `app.py` (`st.navigation` entry + a row in the
      "What's inside" table). Cross-schema page — does **not** call `ui.sidebar_controls()`.
- [ ] **Status strip** (`st.columns` of `st.metric`): models registered, servable count,
      3-class serving model, dataset rows, MLflow configured (yes/no).
- [ ] **Model registry** table from `src.registry.all_models()` (both schemas): Model, Variant,
      Schema, macro-F1, Servable ✓, Serving ★, Kind. ★ resolves the API default the same way
      `api/main.py::default_model_id` does (env override → best loadable classical).
- [ ] **Data & versioning**: rows / categories / companies + a star-balance bar from
      `src.eda.summary()`; git short SHA; DVC status (`.dvc/` present? → configured / Card 2.3).
- [ ] **MLflow & tracking**: reflect `MLFLOW_TRACKING_URI` config only — **no HTTP ping** (that
      would pull `requests`/`httpx` into the `app` group). Info note when unset.
- [ ] **Environment / config** expander: `DEFAULT_MODEL_*`, `DISTILBERT_*`, `MLFLOW_TRACKING_URI`;
      **secrets masked** — username/password shown as `✓ set` / `—`, never the value.
- [ ] **Deep links**: `st.link_button` to the API `/docs`; MLflow/Airflow/Grafana greyed until
      their phase lands.
- [ ] In-process + graceful degradation: every external read (git, `.dvc`, MLflow env) wrapped so
      a fresh clone with nothing configured renders without a crash. **No new dependency.**
- [ ] `ruff` + `black` clean; `make test` stays green.

### Done when

Opening **Ops** in the Streamlit sidebar on a fresh clone (no MLflow/DVC set) renders every
panel with no crash, the registry table marks the API's default model with ★, the data panel
shows live rows + git commit, and no secret value is ever printed.

---

### Milestone 2 exit checklist

- [ ] An experiment is tracked in MLflow (DagsHub UI shows params + macro-F1 + artifact)
- [ ] `make pull` on a fresh clone restores data + all weights; CI pulls via the secret
- [ ] `git ls-files` contains no model binaries
- [ ] Moving the `production` alias changes what the API serves
- [ ] The multi-service stack runs (`docker compose up`)
- [ ] The `proxy` (nginx) fronts the stack — `curl localhost/health` reaches the api through nginx
- [ ] `dev → main` milestone merge done, tag `milestone-2`

---
---

# Milestone 3 — Orchestration & Deployment · target **Aug 7**

> ✏️ **Draft scope** — M3/M4 cards are written early to give the whole-project picture.
> Treat subtask checklists as guidelines, not contracts: scope locks at the phase-kickoff
> check-in, and each card's **Open questions** get answered first (record choices as short
> ADR notes under `docs/` — [CONTRIBUTING.md §8](../CONTRIBUTING.md)).

Two decisions already made by the course calendar (record both as ADR notes):
- **Orchestrator = Airflow** — the mandatory Sprint-3 module (Jul 20–31), ends a week before
  the M3 target. No Prefect debate needed.
- **Serving stays FastAPI** — Sprint 3 also teaches BentoML, but our FastAPI service is
  built, tested and containerized since Phase 1; the BentoML module is reference material,
  a swap is not planned.

---

## Card 3.1 — Orchestrated pipeline, data half

| | |
|---|---|
| **Assignee** | **Marco** |
| **Labels** | `phase-3` `type:data` `priority:med` |
| **Branch** | `feature/ma-dag-data` (from `dev`) |
| **Depends on** | M2 complete. Shares the orchestrator + DAG with 3.2 — agree the interface first (see below). |
| **Files** | new `dags/` (or `flows/`) · `docker-compose.yml` (orchestrator service) · `scripts/get_data.py` reuse |

**Why:** orchestration encodes the pipeline's order and retries in code, so the whole chain
runs from one trigger instead of tribal knowledge.

### Open questions (answer at M3 kickoff)

- Airflow deployment shape — the official docker-compose file (heavy, many services) or a
  slim custom setup?
- Where do the DAG containers get DVC/DagsHub credentials — mounted `.dvc/config.local`,
  env vars, or a compose secret?
- M3 scope: manual-trigger only, or already scheduled?

### Subtasks

- [ ] Orchestrator is **Airflow** (mandatory course module, Sprint 3 Jul 20–31 — see the
      calendar at the top of this file); one stack for both 3.1 and 3.2 (two halves of the
      same DAG). Record the one-line ADR.
- [ ] Agree the 3.1 ↔ 3.2 interface with Dilshana before coding: the data half ends by
      producing versioned `data/processed/` (fresh CSVs, DVC-tracked and pushed); the model
      half consumes exactly that. Write it down in the DAG module docstring.
- [ ] Add the orchestrator as a compose service (own profile, so the default stack stays
      light); UI port documented.
- [ ] DAG steps, data half: ingest (reuse `scripts/get_data.py` logic — import, don't
      shell out blindly) → preprocess → DVC add/commit + push of the refreshed
      `data/processed/`. The DAG runs in its own container: pull inputs explicitly at the
      start, never assume the host's files.
- [ ] Retries + failure alerts on each step (orchestrator-native settings are enough).
- [ ] Docs: how to trigger manually, where to watch progress.

### Done when

Triggering the DAG produces fresh, DVC-versioned `data/processed/` with no manual steps, and
the run is visible (with per-step status) in the orchestrator UI.

---

## Card 3.2 — Orchestrated pipeline, model half (train → eval → gated promote)

| | |
|---|---|
| **Assignee** | **Dilshana** |
| **Labels** | `phase-3` `type:model` `priority:med` |
| **Branch** | `feature/di-dag-model` (from `dev`) |
| **Depends on** | 3.1's interface (versioned `data/processed/`); same Airflow stack as 3.1. |
| **Files** | same `dags/`/`flows/` module tree as 3.1 · reuses `scripts/build_pipelines.py` + `scripts/register_model.py` (from 2.2) |

**Why:** a conditional-promotion gate turns "we retrained" into "we retrained and deployed
*only if* it's better" — automation with a quality bar.

### Open questions (answer at M3 kickoff)

- Promotion rule: strictly better macro-F1, or require a margin (avoid alias churn on
  noise-level improvements)?
- Alias move fully automated inside the DAG, or a manual-approval step before promotion?
- Retrain data in M3: same train/test splits, or already the Option-3 category slices
  (DATA_SOURCES.md)?

### Subtasks

- [ ] DAG steps, model half: pull the versioned data (from 3.1's output) → train (reuse the
      tracked training from 2.1, so every DAG run is an MLflow run) → evaluate on the test
      split → **conditionally promote**.
- [ ] The gate: promote (move the `production` alias via the 2.2 promotion tooling) only if
      the new run's macro-F1 beats the **currently served** model's — query the registry for
      the current production version's metric, never compare against a hard-coded number
      (baseline reference → `docs/ML_CANVAS.md`).
- [ ] Log the gate's verdict (promoted / rejected + both scores) where the team can see it —
      the run's tags plus the orchestrator log.
- [ ] Test the negative path deliberately: run the DAG with a crippled config (e.g. tiny
      training subset) and verify the worse model is **not** promoted.
- [ ] Docs: the promotion rule, how to override manually (registry alias move) if ever needed.

### Done when

One trigger runs data-pull → train → eval → gate end-to-end; a better model gets the
`production` alias automatically; a worse model is rejected — both cases demonstrated.

---

## Card 3.3 — API security (auth, validation, rate limiting)

| | |
|---|---|
| **Assignee** | **Mykola** |
| **Labels** | `phase-3` `type:api` `priority:med` |
| **Branch** | `feature/mk-api-security` (from `dev`) |
| **Depends on** | — fully parallel within M3. |
| **Files** | `api/main.py` · `tests/test_api.py` · `.env.example` · `deploy/nginx/nginx.conf` (TLS server block) · README API section |

**Why:** authentication, input validation and rate limiting are the minimum bar for exposing
a model endpoint to the outside world.

### Open questions (answer at M3 kickoff)

- Auth scheme: static API key (header-based, simplest for a course exam) or JWT (closer to
  production)? Record as ADR.
- Rate-limit store: in-memory (fine at course scale) or redis (survives restarts, works
  with >1 replica in 3.4)?
- How are API keys distributed — one team key in `.env`, or per-member keys?
- TLS cert: self-signed for the demo, or a real cert (mkcert locally / Let's Encrypt via
  cert-manager in k8s)? Record as ADR.

### Subtasks

- [ ] Implement auth as a FastAPI dependency on `/predict` and `/models`; key/secret comes
      from env (documented in `.env.example`), never hard-coded.
- [ ] **`/health` stays unauthenticated** — the Docker healthcheck and the k8s probes
      (Card 3.4) depend on it.
- [ ] Input hardening: request size limit, text length limit, reject empty/absurd payloads
      with clear 4xx errors (extend the existing pydantic models).
- [ ] Basic rate limiting per client (middleware or dependency; in-memory is fine at course
      scale — note the production caveat in the README).
- [ ] **TLS/HTTPS termination** at the nginx reverse proxy (Card 2.4-B): serve 443 with a cert
      (self-signed is acceptable for the course demo), redirect 80 → 443; the app stays plain
      HTTP behind the proxy. In k8s the same terminates at the ingress (Card 3.4).
- [ ] Tests: authorized call passes; missing/wrong key → 401/403; oversized payload → 4xx;
      burst beyond the limit → 429. Existing tests updated to authenticate.

### Done when

Unauthenticated `/predict` is rejected (401/403) and tests prove all four paths; `/health`
still answers bare (compose healthcheck stays green); external traffic is served over HTTPS
(TLS terminated at the nginx proxy / ingress).

---

## Card 3.4 — CI/CD + scalable deployment (k8s)

| | |
|---|---|
| **Assignee** | **Luc** |
| **Labels** | `phase-3` `type:infra` `priority:med` |
| **Branch** | `feature/lu-cicd-k8s` (from `dev`) |
| **Depends on** | 2.4-B's Dockerfile targets. Parallel to the rest of M3. |
| **Files** | `.github/workflows/` (build/deploy job) · new `k8s/` manifests (incl. `k8s/ingress.yaml`) · README deploy section |

**Why:** continuous deployment with rollback means a bad release is a one-command revert, and
replicas mean one crashed container doesn't take the service down.

**Course timing:** the Kubernetes module opens **Aug 27 — after the M3 target (Aug 7)**.
Land the CI/CD half (build, SHA tags, deploy, rollback) by the M3 target; do the k8s
hands-on **Aug 27 – Sep 4** (the module's opening week), hard stop at the **Sep 9 project
freeze**. If that's too tight, agree a reduced k8s scope with the mentor at M3 kickoff —
never let it slip past Sep 4. (The module window officially runs to Sep 18 — past the
freeze and the defence; the window is for *learning*, not for project work.)

### Open questions (answer at M3 kickoff)

- Deployment target: local cluster (kind/minikube — zero cost, demoable on any laptop) or
  a cloud cluster? Record as ADR.
- Image registry: GHCR (free for the repo) or something else?
- Deploy trigger: automatically on merge to `dev`, or manual workflow dispatch?
- Rollback mechanism: `kubectl rollout undo` or re-apply the previous SHA tag?
- Ingress controller: nginx-ingress via the standard manifest, or a minikube/kind addon?
  Record as ADR.

### Subtasks

- [ ] Extend CI with a build job: build the `api` image, **tag with the git SHA** (never
      `latest` — `latest` makes rollback meaningless), push to a registry (GHCR is free for
      the repo).
- [ ] Deploy job: apply the k8s manifests with the new tag on merge to `dev` (or manual
      dispatch — decide with the mentor); keep the previous tag recorded so rollback is one
      step.
- [ ] `k8s/` manifests translated from the compose setup: Deployment (**replicas ≥ 2**,
      resource requests/limits, liveness/readiness probes on `/health`) + Service; secrets
      for the API key (3.3) and `MLFLOW_*` via k8s Secrets, not baked into the image.
- [ ] **Ingress** (`k8s/ingress.yaml`, nginx-ingress controller): the single external entry
      point routing external HTTPS → the api Service; **TLS terminates here** (cert via a k8s
      Secret / optional cert-manager). The Service keeps ClusterIP L4 load-balancing across the
      ≥2 replicas — the ingress adds entry + TLS, not a second balancer.
- [ ] Rollback procedure: documented one-step revert (re-apply previous SHA tag / rollout
      undo) and actually rehearse it once.
- [ ] Docs: README deploy section — how a push becomes a deployment, how to roll back.

### Done when

A push deploys automatically; a bad deploy reverts in one step (rehearsed, not just
documented); the API runs with >1 replica behind a Service, reached through the nginx ingress
over HTTPS.

---

### Milestone 3 exit checklist

- [ ] One trigger runs the full DAG: fresh data → tracked training → gated promotion
- [ ] A worse model is demonstrably not promoted
- [ ] `/predict` requires auth; `/health` open; tests cover 401/403/429
- [ ] Image tagged by git SHA, deployed on k8s with ≥2 replicas, rollback rehearsed
- [ ] External traffic is HTTPS through the nginx reverse proxy (compose) / ingress (k8s)
- [ ] `dev → main` milestone merge done, tag `milestone-3`

---
---

# Milestone 4 — Monitoring & Maintenance · target **Sep 4**

> ✏️ **Draft scope** — same rule as M3: subtask checklists are guidelines; scope locks at
> the M4 kickoff check-in after each card's **Open questions** are answered (short ADR
> notes under `docs/`).

New-data options for drift/retraining → [DATA_SOURCES.md](DATA_SOURCES.md) (Option 3 —
category slicing of the existing 123k — is the agreed first pass).

---

## Card 4.1 — Drift detection with Evidently

| | |
|---|---|
| **Assignee** | **Mykola** |
| **Labels** | `phase-4` `type:model` `priority:low` |
| **Branch** | `feature/mk-drift-evidently` (from `dev`) |
| **Depends on** | M3's deployed API. 4.4 consumes this card's drift signal. |
| **Files** | `api/main.py` (request logging) · new `monitoring/` module · `docker-compose.yml` |

**Why:** drift detection compares live inputs against the training distribution — the early
warning that model quality is about to drop, before labels confirm it.

**Course timing:** the mandatory Drift Monitoring module is Sprint 4 (Aug 3–26) — it lands
well before the M4 target, so this card can start during the M3 window if capacity allows.

### Open questions (answer at M4 kickoff)

- Drift-flag format: file, endpoint, or Prometheus metric? Must be agreed three-ways with
  4.2 (alerting) and 4.4 (retrain trigger) before building.
- Request-store backend: append-only file, SQLite, or a volume-backed table?
- Comparison window: how many recent requests count as "current"?

### Subtasks

- [ ] **Prerequisite first:** persist incoming `/predict` requests (text + predicted label +
      timestamp) — without a stored "current" dataset there is nothing to compare. Simple
      append-only store is fine (file/SQLite/volume); mind size growth.
- [ ] Reference dataset = the training distribution (pull via DVC); current dataset = a
      recent window of stored requests.
- [ ] Evidently report job: data-drift + data-quality checks on the text-derived features;
      runnable on demand and on a schedule (hook into the M3 orchestrator).
- [ ] Expose the result as a machine-readable **drift flag** (file/endpoint/metric) — 4.4
      triggers retraining from it and 4.2 alerts on it. Agree the format with Luc and
      Dilshana before building.
- [ ] Simulate drift to prove it works: replay a skewed category slice
      (DATA_SOURCES Option 3) against the API and show the report flips.

### Done when

An injected distribution shift produces a drift report and raises the flag; the flag is
consumable by 4.2 (alert) and 4.4 (retrain trigger).

---

## Card 4.2 — Dashboards + alerts with Prometheus & Grafana

| | |
|---|---|
| **Assignee** | **Dilshana** |
| **Labels** | `phase-4` `type:infra` `priority:low` |
| **Branch** | `feature/di-monitoring-stack` (from `dev`) |
| **Depends on** | 4.3's `/metrics` endpoint (can start against a stub scrape target). |
| **Files** | `docker-compose.yml` (prometheus + grafana services) · prometheus config · Grafana dashboard JSON + alert rules |

**Why:** dashboards and alert rules are how a team *operates* a model — latency, throughput
and drift on one screen, with a ping when a threshold breaks.

**Course timing:** the Prometheus & Grafana module is Sprint 4 (Aug 3–26) — it lands well
before the M4 target, so this card can start during the M3 window if capacity allows.

### Open questions (answer at M4 kickoff)

- Alert routing: Grafana alerting UI only, or a webhook into the team channel?
- Dashboards provisioned from JSON committed in-repo (reproducible) — which directory
  layout?

### Subtasks

- [ ] Compose services: prometheus (scrape config targeting the API's metrics endpoint **by
      service name** over the compose network, never localhost) + grafana (provisioned
      datasource + dashboards from JSON, so the setup is reproducible from a clean clone).
- [ ] Dashboards: request latency (p50/p95), throughput, error rate, drift status (from
      4.1's flag/metric), container/system health.
- [ ] Alert rules on top: latency above threshold, error-rate spike, drift flag raised.
      Route alerts somewhere visible (Grafana alerting UI is enough; a webhook to the team
      channel is a bonus).
- [ ] Demonstrate one alert firing end-to-end (e.g. replay the 4.1 drift simulation).

### Done when

A Grafana board shows live metrics from real traffic and an alert fires when drift is
injected — reproducible from a clean clone via compose.

---

## Card 4.3 — Metrics endpoint, API docs & maintenance guide

| | |
|---|---|
| **Assignee** | **Marco** |
| **Labels** | `phase-4` `type:api` `priority:low` |
| **Branch** | `feature/ma-metrics-docs` (from `dev`) |
| **Depends on** | — start of M4 (4.2 scrapes this endpoint). |
| **Files** | `api/main.py` (`/metrics`) · new `docs/MAINTENANCE.md` · FastAPI `/docs` review |

**Why:** instrumentation exposes what the service is doing; the maintenance guide is the
course's "could someone else operate this?" success criterion.

### Open questions (answer at M4 kickoff)

- Metric naming convention for the dashboards — agree the exact names with 4.2 before
  either side builds.
- Is `/metrics` reachable only inside the compose/k8s network, or exposed (and if exposed,
  behind 3.3's auth)?

### Subtasks

- [ ] Add `/metrics` in Prometheus format via the FastAPI instrumentator library (request
      count/latency histograms out of the box); keep it unauthenticated **inside** the
      compose/k8s network but note the exposure caveat in the README.
- [ ] Coordinate the metric names 4.2 will dashboard/alert on (agree with Dilshana).
- [ ] `docs/MAINTENANCE.md`: the three operator procedures — **update** (deploy a new model:
      alias move; deploy new code: CI/CD flow), **rollback** (alias back-move; image tag
      revert), **retrain** (trigger the DAG manually; what the gate does) — each as a short
      numbered runbook with the exact commands.
- [ ] Data-privacy paragraph in the maintenance guide: public Trustpilot dataset; review
      text can contain personal names but the service collects no accounts or user PII;
      the stored `/predict` requests (4.1) get a documented retention/cleanup note.
- [ ] Review the auto-generated FastAPI `/docs`: every endpoint has descriptions, request/
      response examples, auth documented (3.3). Fix gaps in the pydantic models/docstrings.
- [ ] *Nice-to-have:* polish the Streamlit demo (`make setup-full && make app`) for the
      defence.

### Done when

`/metrics` serves Prometheus format (4.2 scrapes it successfully); MAINTENANCE.md covers
update/rollback/retrain as runnable runbooks; `/docs` is complete.

---

## Card 4.4 — Automated retraining (close the loop)

| | |
|---|---|
| **Assignee** | **Luc** |
| **Labels** | `phase-4` `type:data` `priority:low` |
| **Branch** | `feature/lu-auto-retrain` (from `dev`) |
| **Depends on** | 3.1+3.2's DAG · 4.1's drift flag. |
| **Files** | orchestrator schedule/trigger config · the drift-flag consumer · `docs/MAINTENANCE.md` (retrain section, with Marco) |

**Why:** this closes the MLOps loop — monitoring detects decay, the orchestrator retrains,
the gate promotes only improvements, unattended.

### Open questions (answer at M4 kickoff)

- Trigger policy: on drift flag, on schedule, or both?
- Cooldown length after a triggered retrain?
- How is the "next category slice" chosen/rotated across retrains (Option 3,
  DATA_SOURCES.md)?

### Subtasks

- [ ] Trigger the M3 DAG from the 4.1 drift flag (preferred) and/or on a schedule — consume
      the agreed flag format.
- [ ] New data for the retrain: replay the next category slice per DATA_SOURCES Option 3 so
      each retrain genuinely sees new data.
- [ ] The 3.2 gate stays the promotion authority — retraining never bypasses it.
- [ ] **Cooldown guard:** a drift flag that never clears must not retrain forever — after a
      triggered retrain, suppress further triggers for a cooldown window (and alert if drift
      persists through it, since that means retraining didn't fix it).
- [ ] End-to-end demo: inject drift → flag raises → DAG fires → better model promoted → API
      serves it → flag clears. Record the sequence for the defence.

### Done when

Retraining runs unattended from a drift signal, the API picks up the promoted model, and the
cooldown provably prevents a retrain loop.

---

### Milestone 4 exit checklist

- [ ] Injected drift produces a report + machine-readable flag
- [ ] Grafana shows live metrics; an alert fires on drift
- [ ] `/metrics` in Prometheus format; MAINTENANCE.md runbooks complete
- [ ] Drift signal triggers an unattended retrain; gate + cooldown behave
- [ ] `dev → main` milestone merge done, tag `milestone-4`
- [ ] **Sep 9 freeze respected** — demo rehearsed, no open feature PRs; Sep 9–14 is
      rehearsal/slides only

---
---

## Importing cards as GitHub issues

Create one issue per card — title, labels and assignee from the card's table, body = the card
section from this file (keep the checklists so GitHub renders progress). Use the web UI or the
`gh issue create` CLI. Add every issue to the Project board: current-milestone cards in
**To Do**, future-milestone cards in **Backlog**; set Iteration when a card is pulled into a
sprint ([PROJECT_MANAGEMENT.md §2](PROJECT_MANAGEMENT.md) for board and label conventions).
Bump a milestone's cards to `priority:high` at its kickoff.
