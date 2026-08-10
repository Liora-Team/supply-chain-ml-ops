# Multi-target image — one build target per service responsibility (Card 2.4-B):
#   api      — classical inference API, torch-free (the default target)
#   training — one-shot job that rebuilds the served pipelines (core deps only)
#   bert     — the same API plus the torch stack for DistilBERT (api + bert groups)
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
# Core deps only — build_pipelines.py imports nothing beyond pandas/joblib/sklearn/xgboost.
# Card 2.1 adds `--group track` here, together with the MLflow logging that needs it.
FROM base AS training

RUN uv sync --frozen --no-install-project --no-default-groups

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
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

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
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
