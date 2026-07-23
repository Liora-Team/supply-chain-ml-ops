# Supply Chain MLOps — Trustpilot Review Rating

> Productionizing an NLP model with a full MLOps lifecycle: reproducible environments,
> a tested inference API, containerization, experiment tracking, versioning, CI/CD, and monitoring.

This repository is the **MLOps engineering** of a Trustpilot review-rating model. Exploratory
data-science work (EDA notebooks, model exploration, tuning) is **out of scope by design** —
the repo holds only clean, reusable code plus the operational tooling required to run it
reliably: environments, tests, containers, tracking, versioning, CI/CD, monitoring.

---

## The problem

Predict a **Trustpilot review's star rating from its text**.

| Aspect | Detail |
|--------|--------|
| Task | Multiclass text classification |
| Target | Review star rating |
| Label schemas | **5-class** (raw 1–5★) and **3-class** (0 = negative 1–2★, 1 = neutral 3★, 2 = positive 4–5★) |
| Dataset | [`Kerassy/trustpilot-reviews-123k`](https://huggingface.co/datasets/Kerassy/trustpilot-reviews-123k) (~123k reviews, HuggingFace) |
| Primary metric | **macro-F1** (data is class-imbalanced) |
| Models | Classical (TF-IDF + sklearn/XGBoost, served) and a fine-tuned **DistilBERT** (Phase 2) |

Full framing — value proposition, cost model, baseline metric, monitoring plan — lives in the
**[ML Canvas](docs/ML_CANVAS.md)**.

## Status

The project runs in **four course phases** (Foundations → Tracking & Versioning →
Orchestration & Deployment → Monitoring). **Phase 1 is done; Phase 2 is next** — the roadmap
with dates and per-task owners is in [MILESTONES.md](MILESTONES.md).

## Quickstart

```bash
# 1. Reproducible dev env (torch-free: core + api + dev). ~30s.
make setup                       # = uv sync

# 2. Run the test suite (preprocessing + inference + API).
make test                        # 15 passing (first run downloads NLTK data once — needs network)

# 3. Serve the inference API.
make api                         # → http://localhost:8000/docs
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"terrible, broke in a day","schema":"3-class"}'
# → {"label_display":"Negative", ...}

# …or fully containerized (nginx is the single public entry — see "Containers" below):
make up                          # docker compose up --build -d
curl -s localhost/health         # → api via the nginx proxy on port 80
```

**That's the whole verification** — the trained pipelines are committed in `models/`, so a fresh
clone needs **no dataset download and no training** to run the tests and serve predictions.
(The only one-time network fetch is the small NLTK corpora on the first `make test`.)

The remaining targets are **optional**, only for specific jobs (`make help` lists everything):

| Target | When you need it |
|--------|------------------|
| `make data` | Rebuild `data/processed/` from the raw HuggingFace dataset (~123k reviews). Only needed before retraining — the API/tests never read it. |
| `make train` | Retrain all model pipelines from `data/processed/` (run `make data` first). Only when changing models or data. |
| `make setup-full` + `make app` | Run the Streamlit demo front-end. Installs the heavy torch stack — skip unless you want the UI. |

> **Note on scope:** the served API is **classical-only and torch-free** (keeps the env and
> image small). The DistilBERT path becomes its own service in Phase 2's microservices split
> (MILESTONES Task 2.4; weight handling → [CONTRIBUTING.md §7](CONTRIBUTING.md#7-data--model-handling)).

## Containers (compose)

One container per responsibility; the nginx **proxy** is the only published entry point
(port 80 — TLS on 443 lands with Card 3.3). Internal services talk over the compose
network by service name and are never published directly.

```
                 ┌────────► api:8000     (classical inference — /)
localhost:80 ─ proxy (nginx)
                 └────────► mlflow:5000  (local tracking UI — /mlflow/)
```

| Service | Runs | Notes |
|---------|------|-------|
| `proxy` | always | nginx, config in `deploy/nginx/nginx.conf`. Routes `/` → api, `/mlflow/` → mlflow. |
| `api` | always | Torch-free classical API (Dockerfile `api` target). Scale with `docker compose up -d --scale api=N` (then `docker compose restart proxy` to refresh its upstream pool). |
| `mlflow` | always | Local tracking UI for offline dev on a named volume — team truth lives on DagsHub. |
| `training` | `--profile train` | One-shot pipeline rebuild (`training` target) over bind-mounted `data/` + `models/`. A plain `docker compose up` never retrains. |
| `bert` | `--profile bert` | DistilBERT torch image (`bert` target); weights via `DISTILBERT_*` ([CONTRIBUTING.md §7](CONTRIBUTING.md#7-data--model-handling)). |

```bash
docker compose up -d                                  # proxy + api + mlflow
curl -s localhost/health                              # api through nginx
open http://localhost/mlflow/                         # local MLflow UI

# One-shot training, logged to the local MLflow (or drop the URI override to use .env/DagsHub):
MLFLOW_TRACKING_URI=http://mlflow:5000 docker compose --profile train up --build training

docker compose --profile bert up -d --build bert      # DistilBERT service (heavy: torch)
```

Credentials (`MLFLOW_*`) are passed through from `.env` at runtime — never baked into images.

## Repository structure

```
supply-chain-ml-ops/
├── README.md                  # this file
├── CONTRIBUTING.md            # collaborator rules: git flow, commits, PRs, style, tests
├── MILESTONES.md              # roadmap + per-phase technical tasks with owners
├── Makefile                   # make setup / test / lint / api / app / data / up
├── Dockerfile                 # multi-target: api (torch-free, default) / training / bert
├── docker-compose.yml         # proxy (nginx) + api + mlflow, plus train/bert profiles
├── deploy/                    # nginx reverse-proxy config (deploy/nginx/nginx.conf)
├── pyproject.toml             # uv project — core deps + optional groups (api/app/bert/data/dev)
├── docs/                      # ML_CANVAS, PROJECT_MANAGEMENT, TASKS_DONE, DATA_SOURCES
├── src/                       # core package: preprocessing, inference, registry, eda, explain, findings, ui
├── api/                       # FastAPI inference service (main.py)
├── scripts/                   # get_data.py (collect+preprocess) · build_pipelines.py (train)
│                              # · build_eda_artifacts.py (provenance of the committed EDA artifacts)
├── tests/                     # pytest: preprocessing, inference, api
├── models/                    # small sklearn pipelines + metric sidecars (large weights → DVC)
├── data/                      # big splits gitignored (DVC from Phase 2); two small committed
│                              # EDA artifacts: eda_sample.parquet + eda_summary.json
├── deploy/                    # nginx reverse-proxy config for compose (Phase 2) — planned
├── k8s/                       # Kubernetes manifests incl. ingress (Phase 3) — planned
└── .github/                   # issue/PR templates + CI workflow
```

## Team

**Marco, Mykola, Dilshana, Luc** — mentored by **Kilyan**. Each member owns one track
(modeling / infra / data / serving) and **roles rotate per phase**, so everyone touches every
MLOps area by the defence — rotation matrix and rationale in
[docs/PROJECT_MANAGEMENT.md](docs/PROJECT_MANAGEMENT.md).

## Docs map

One canonical home per topic — if you're about to write the same thing twice, link instead:

| Doc | What lives there (and only there) |
|-----|-----------------------------------|
| [docs/ML_CANVAS.md](docs/ML_CANVAS.md) | Project framing: the 10 canvas blocks, objectives, **baseline metric**, cost model |
| [MILESTONES.md](MILESTONES.md) | **Roadmap** + per-phase technical tasks: Owner, Why, commands, acceptance |
| [docs/PROJECT_MANAGEMENT.md](docs/PROJECT_MANAGEMENT.md) | Team & **rotation matrix**, board conventions, check-in template, **Definition of Done**, kick-off compliance map |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Env setup, **git/PR workflow**, style, data/model handling, the **golden rule** |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | Dataset detail + options for new data over time (drift/retraining) |
| [docs/TASKS_DONE.md](docs/TASKS_DONE.md) | Completed Phase-1 tasks, issue-ready for the board |
| [docs/TODO.md](docs/TODO.md) | **Milestones 2–4** cards — per-member checklists, branch names, acceptance criteria (source for the Project board) |

**New to the repo? Read in this order:**

1. This README — what the project is + run the quickstart
2. [docs/ML_CANVAS.md](docs/ML_CANVAS.md) — the problem, data, metric, and cost model (the *why* before the *how*)
3. [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — the dataset in depth
4. [MILESTONES.md](MILESTONES.md) — what gets built, phase by phase
5. [docs/PROJECT_MANAGEMENT.md](docs/PROJECT_MANAGEMENT.md) — how work is organized and reviewed
6. [CONTRIBUTING.md](CONTRIBUTING.md) — the rules, before your first PR
7. [docs/TASKS_DONE.md](docs/TASKS_DONE.md) — reference: what Phase 1 already delivered

Kickoff resources: [mlops_project_template](https://github.com/ssime-git/mlops_project_template) · [mlops-boiler-plate](https://github.com/ssime-git/mlops-boiler-plate) · example: [dec23_mlops_accidents](https://github.com/DataScientest-Studio/dec23_mlops_accidents)
