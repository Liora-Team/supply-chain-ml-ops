# Tasks done — Phase 1

> Completed work, formatted for the GitHub Project board: create one issue per entry
> (title + labels + body), close it immediately — the board then carries the full history.
> Ongoing and future tasks live only on the board; see `docs/PROJECT_MANAGEMENT.md`.

## How to import

For each entry below: `New issue` → paste title + body → add the labels → close as completed
→ add to the Project (Status: **Done**).

---

## 1. Define the ML Canvas, primary metric and baseline

**Labels:** `phase-1` `type:docs`

Frame the project: the 10 canvas blocks (prediction task, value proposition, decisions,
data, features, models, monitoring, …) → `docs/ML_CANVAS.md`. Pin the primary metric.

**Evidence:** canvas recorded in `docs/ML_CANVAS.md`; primary metric **macro-F1**;
baseline pinned (see task 7).

## 2. Set up the reproducible dev environment

**Labels:** `phase-1` `type:infra`

`pyproject.toml` + `uv.lock` with a torch-free default env and optional groups
(`api`/`app`/`bert`/`data`/`dev`); `Makefile` targets (`setup`/`test`/`lint`/`api`/`data`/`train`/`up`).

**Evidence:** clean-room check green — `.venv` wiped, `uv sync` from the lockfile alone
reproduces the env (~30s); `make lint` clean.

## 3. Containerize the inference API

**Labels:** `phase-1` `type:infra`

Slim, torch-free `Dockerfile` + `docker-compose.yml` for the `api` service.

**Evidence:** `docker compose up` serves `/predict` in-container (image ~2 GB).

## 4. Build the data collection + preprocessing script

**Labels:** `phase-1` `type:data`

`scripts/get_data.py` downloads the 123k HuggingFace dataset and preprocesses it to one
canonical `data/processed/` (clean → stopword-removed → lemmatised + 3-class label).

**Evidence:** full pull produced `train.csv` (98,544 rows) + `test.csv` (24,637 rows)
with columns `review, stars, review_lemma, label_3class`.

## 5. Add unit tests for the main-step functions

**Labels:** `phase-1` `type:data`

pytest coverage for `preprocess_text`, `collapse_to_3class`, `predict_classical`,
and the API endpoints.

**Evidence:** `make test` — **15 passing**, including a fresh NLTK download on a new machine.

## 6. Build the FastAPI inference service

**Labels:** `phase-1` `type:api`

`api/main.py` exposing `/health`, `/models`, `/predict` on top of `predict_classical`.

**Evidence:** `POST /predict` answers on host (`make api`) and in Docker
(e.g. `"terrible, broke in a day"` → `Negative 0.96`).

## 7. Train the baseline model pipelines

**Labels:** `phase-1` `type:model`

`scripts/build_pipelines.py` builds all classical pipelines (TF-IDF + sklearn/XGBoost)
as single self-contained sklearn `Pipeline` objects, with metric sidecars.

**Evidence:** `make train` rebuilds 10 pipelines from the fresh data pull under the pinned
sklearn; LogReg 3-class macro-F1 **0.687**; every model within 0.02 of its reference score
except XGBoost (improved).

## 8. Add CI (lint + tests on every PR)

**Labels:** `phase-1` `type:infra`

`.github/workflows/ci.yml`: `uv sync --frozen` → `ruff` + `black --check` → `pytest`,
on every pull request and push to `main`.

**Evidence:** workflow committed; runs automatically once the repo is on GitHub.

## 9. Write the project documentation

**Labels:** `phase-1` `type:docs`

`README.md` (overview + quickstart + docs map), `CONTRIBUTING.md` (git flow, commits, PRs,
style), `MILESTONES.md` (technical playbook), `docs/ML_CANVAS.md` (project framing),
`docs/PROJECT_MANAGEMENT.md` (board conventions), `docs/DATA_SOURCES.md` (future data
options), issue/PR templates.

**Evidence:** docs render, links resolve, quickstart verified from a clean clone.
