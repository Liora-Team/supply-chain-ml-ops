# Slim, torch-free image for the classical inference API (Phase 1 deliverable).
# The DistilBERT path is deliberately out of scope here — it becomes its own image in Phase 2.
FROM python:3.12-slim

# uv for reproducible, lockfile-pinned installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    NLTK_DATA=/usr/share/nltk_data \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# 1) Dependencies first (cache-friendly). Core + api groups only: no dev, no torch/transformers.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group api

# 2) Pre-download the NLTK corpora so inference needs no network at runtime.
RUN uv run --no-sync python -c "import nltk; [nltk.download(p, quiet=True, download_dir='/usr/share/nltk_data') for p in ('stopwords','wordnet','omw-1.4')]"

# 3) Application code + the small classical model artifacts.
COPY src ./src
COPY api ./api
COPY models ./models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
