# Contributing

Rules for everyone working in this repo (team & role rotation →
[docs/PROJECT_MANAGEMENT.md](docs/PROJECT_MANAGEMENT.md)). Keeping these consistent is what
makes the project reproducible and the collaboration painless — it directly maps to the
course's **Reproducibility** and **Collaboration** challenges.

Read this once before your first PR. It takes 5 minutes and saves hours.

---

## 1. Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| Python | 3.12 | pinned in `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | fast, reproducible dependency management |
| Docker + Docker Compose | latest | identical runtime across machines |
| git | 2.30+ | version control |
| [gh CLI](https://cli.github.com/) | latest | PRs / issues from the terminal (optional but handy) |

## 2. Environment setup

```bash
uv sync          # creates .venv and installs the EXACT versions from uv.lock
```

`uv.lock` is committed on purpose: it guarantees every teammate has the same dependency tree.
**Never** hand-edit `uv.lock`. To add a dependency:

```bash
uv add <package>          # runtime dep
uv add --dev <package>    # dev-only dep (pytest, ruff, ...)
```

Then commit the updated `pyproject.toml` **and** `uv.lock` together.

## 3. Git workflow

Two long-lived branches, both **protected** (no direct pushes — everything lands via PR):

| Branch | Role |
|--------|------|
| `main` | Stable. Only receives `dev` merges at milestone boundaries (tagged, e.g. `milestone-2`). |
| `dev`  | Integration branch — **the default target for every PR**. CI must be green here at all times. |

Work branches are cut **from `dev`** and named:

```
feature/<initials>-<short-slug>     e.g. feature/mk-dockerfile
fix/<initials>-<short-slug>         e.g. fix/di-tfidf-path
docs/<initials>-<short-slug>        e.g. docs/ma-readme
```

Keep branches small and short-lived. Rebase on `dev` before opening a PR.
At the end of each milestone a maintainer merges `dev → main` and tags it.

## 4. Commit messages — Conventional Commits

Format: `type(scope): summary` (≤ 50 chars, imperative). Body only when the *why* isn't obvious.

```
feat(api): add /predict endpoint for classical model
fix(preprocessing): handle empty review text
docs(pm): add sprint-1 backlog
test(inference): cover collapse_to_3class
chore(ci): add lint job
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci`.

## 5. Pull Requests

1. Open a PR against `dev`; link the issue it closes (`Closes #12`).
2. Fill the PR description: **what**, **why**, **how tested**.
3. CI must be **green** (lint + tests).
4. **At least 1 review** required before merge (Collaboration challenge → regular reviews).
5. **Squash-merge**. Delete the branch after merge.

Keep PRs reviewable: aim for < ~400 changed lines. Split large work.

## 6. Code style & quality

- **Pre-commit hook** (install once, after `uv sync`): runs the same ruff + black checks as
  CI plus a detect-secrets scan on every commit.
  ```bash
  uv run pre-commit install
  ```
- **Formatter/linter**: `ruff` (lint) + `black` (format). Run before pushing:
  ```bash
  uv run ruff check . && uv run black --check .
  ```
- **Type hints** on public functions; **docstrings** on modules and public functions.
- **Tests are required** for every new pipeline main-step function (Phase 1 requirement).
  Put them in `tests/`, named `test_*.py`, run with:
  ```bash
  uv run pytest
  ```
- Reuse the existing `src/` package (`preprocessing`, `inference`, `registry`, ...) — don't
  duplicate logic. The TF-IDF vectorizer is **loaded, never refit** at serve time to avoid
  train/serve skew.

## 7. Data & model handling

- **Never commit large files.** `data/processed/{train,test}.csv` and **every** model weight
  (all `models/pipelines/*.joblib`, including the 64 MB RandomForest 5-class, and both
  DistilBERT fine-tune dirs `models/distilbert_{3,5}class/final/`, 2 × ~255 MB) are versioned
  with **DVC** — git holds only code, `*.dvc` pointer files, and the small JSON metric
  sidecars (`models/checkpoints/*.json`, `models/pipelines/*.json`). DVC is the single model
  store — no HF-hub hosting; the `DISTILBERT_3CLASS` / `DISTILBERT_5CLASS` env vars exist only
  to override with a custom local path (see `.env.example`).
- **Remotes.** The **default** remote is the team's DagsHub-hosted, S3-backed storage
  (`dagshub`, `s3://dvc` @ `https://dagshub.com/contact.marco.accardi/supply-chain-ml-ops.s3`) — this is
  what `make pull` / `make push` and CI use.
- **Credentials never go in `.dvc/config`** (that file is committed and only holds
  urls/endpoints). Set them locally with `--local`, which writes to the git-ignored
  `.dvc/config.local`:
  ```bash
  # DagsHub: the token acts as both access key and secret (get one from DagsHub → Settings → Tokens)
  uv run dvc remote modify --local dagshub access_key_id <DAGSHUB_TOKEN>
  uv run dvc remote modify --local dagshub secret_access_key <DAGSHUB_TOKEN>
  
  ```
  CI writes the same DagsHub config from the `DAGSHUB_TOKEN` repository secret (Card 2.4-A) —
  never commit real values.
- **Day-to-day loop:**
  ```bash
  make pull                          # start of day: restore data + weights (uv run dvc pull)
  # ...edit code, regenerate data/retrain models...
  dvc add data/processed/train.csv data/processed/test.csv   # new/changed dataset
  dvc add models/pipelines/<name>.joblib                     # new/changed model weight
  dvc commit                         # refresh hashes for artifacts regenerated in place
  dvc push                           # push binaries to the DagsHub remote FIRST
  git add . && git commit -m "..." && git push               # then push the git branch/pointers
  ```
  Pushing data before git (and pulling git before data — i.e. `git pull` then `make pull`) is
  what keeps `.dvc` pointer files in git always resolvable: a reviewer who pulls the branch can
  immediately `make pull` and get exactly the binaries the pointers describe.
- Small runtime artifacts (a few hundred KB: `eda_summary.json`, `eda_sample.parquet`)
  may be committed by explicitly un-ignoring them in `.gitignore`.
- **Document data transformations and features** as you add them (Data Management good
  practice). Validate data quality before it enters a pipeline.

## 8. Documenting decisions

When you make a non-obvious technical choice (a library, a threshold, an architecture split),
record it in a short note under `docs/` (a lightweight ADR: *context → decision → consequence*).
Future-you and the mentor review will thank you.

---

**Golden rule:** if a teammate can't reproduce your result from a clean clone + `uv sync` +
`make pull` (or `docker compose up` after `make pull`), it isn't done. See the Definition of Done in
[docs/PROJECT_MANAGEMENT.md](docs/PROJECT_MANAGEMENT.md).