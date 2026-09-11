# Multi-target image — one build target per service responsibility (Card 2.4-B):
#   api      — classical inference API, torch-free (the default target)
#   training — one-shot job that rebuilds the served pipelines (core deps only)
#   bert     — the same API plus the torch stack for DistilBERT (api + bert groups)
#   airflow  — single-container orchestrator, `airflow standalone` (Card 3.1)
# Targets differ only in which uv dependency groups they sync.

# --- base: pinned interpreter + uv + lockfile (shared by every target) ---
FROM python:3.12.12-slim-bookworm AS base

# uv for reproducible, lockfile-pinned installs.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/usr/share/nltk_data \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Non-root runtime user (container hardening). Fixed UID so bind-mount ownership is stable.
RUN useradd --uid 10001 --create-home app

# Dependencies first (cache-friendly). Secrets never enter layers — MLFLOW_* / DISTILBERT_*
# credentials arrive only via environment passthrough at runtime (docker-compose.yml).
COPY pyproject.toml uv.lock ./

# --- training: one-shot pipeline rebuild (scripts/build_pipelines.py) ---
# No NLTK corpora: data/processed/*.csv is already lemmatised by scripts/get_data.py.
# `--group track` installs mlflow so containerized runs can log to the compose
# mlflow service (or DagsHub) via MLFLOW_TRACKING_URI; without it build_pipelines.py
# silently trains untracked.
FROM base AS training

RUN uv sync --frozen --no-install-project --no-default-groups --group track

COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
USER app

# data/ and models/ are bind mounts (docker-compose.yml), never baked into the image.
CMD ["uv", "run", "--no-sync", "python", "scripts/build_pipelines.py"]

# --- bert: API image + torch stack, weights resolved via DISTILBERT_* (CONTRIBUTING.md §7) ---
FROM base AS bert

RUN uv sync --frozen --no-install-project --no-default-groups --group api --group bert

RUN uv run --no-sync python -c "import nltk; [nltk.download(p, quiet=True, download_dir='/usr/share/nltk_data') for p in ('stopwords','wordnet','omw-1.4')]"

# models/ is baked in (minus the DVC weights, see .dockerignore) so this image also serves the
# classical pipelines standalone; the compose bind mount shadows it when the stack runs.
COPY --chown=app:app src ./src
COPY --chown=app:app api ./api
COPY --chown=app:app models ./models
COPY --chown=app:app monitoring ./monitoring
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- airflow: single-container orchestrator — scheduler + dag-processor + UI in one
# `airflow standalone` process, SQLite metadata on a named volume (Card 3.1, ADR 002). ---
FROM base AS airflow

ENV AIRFLOW_HOME=/opt/airflow

RUN uv sync --frozen --no-install-project --no-default-groups --group orchestration --group data --group track

# The preprocess task lemmatises in-container, so corpora are baked like api/bert.
RUN uv run --no-sync python -c "import nltk; [nltk.download(p, quiet=True, download_dir='/usr/share/nltk_data') for p in ('stopwords','wordnet','omw-1.4')]"

# 0777: the airflow_home named volume inherits this dir's permissions on first use, and a
# Linux DOCKER_UID override (see docker-compose.yml `training`) must still be able to write it.
RUN mkdir -p /opt/airflow && chmod 0777 /opt/airflow

# dags/ is baked so the image is self-contained; the compose bind mount shadows it for
# edit-without-rebuild iteration.
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app dags ./dags
COPY --chown=app:app monitoring ./monitoring
USER app

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "airflow", "standalone"]

# --- api: slim torch-free classical inference API (last stage = default target) ---
FROM base AS api

RUN uv sync --frozen --no-install-project --no-default-groups --group api

# Pre-download the NLTK corpora so inference needs no network at runtime.
RUN uv run --no-sync python -c "import nltk; [nltk.download(p, quiet=True, download_dir='/usr/share/nltk_data') for p in ('stopwords','wordnet','omw-1.4')]"

# Application code + the small classical model artifacts.
# Ownership is set on the COPY itself: a `chown -R /app` would rewrite every file in
# /app/.venv into a duplicate layer. The runtime user only needs to read the venv.
COPY --chown=app:app src ./src
COPY --chown=app:app api ./api
COPY --chown=app:app models ./models
COPY --chown=app:app monitoring ./monitoring
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
