# Supply Chain MLOps — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup setup-full pull push test lint fmt api app data train eda-artifacts certs up down build clean dag-up dag-down dag-trigger

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Reproducible dev env (core + api + dev, torch-free)
	uv sync

setup-full:  ## Full env incl. Streamlit app + DistilBERT (heavy: torch)
	uv sync --group app --group bert --group data

pull:  ## Restore data + model weights from the DVC remote (needs DagsHub creds)
	uv run dvc pull

pull-ci: ## Restore only CI-required artifacts
	uv run dvc pull data/processed/*.dvc models/pipelines/*.dvc

push:  ## Push data + weights to the DVC remote (default: DagsHub)
	uv run dvc push

test:  ## Run the unit tests
	uv run pytest

lint:  ## Lint + format check
	uv run ruff check . && uv run black --check .

fmt:  ## Auto-fix lint + format
	uv run ruff check --fix . && uv run black .

api:  ## Run the FastAPI inference service on :8000
	PYTHONPATH=. uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:  ## Run the Streamlit front-end (needs `make setup-full`)
	uv run --group app --group bert streamlit run app.py

data:  ## Download + preprocess the dataset into data/processed/
	uv run --group data python scripts/get_data.py

train:  ## Rebuild the served model pipelines (run `make data` first)
	uv run python scripts/build_pipelines.py

eda-artifacts:  ## Rebuild the committed EDA artefacts (needs the uncommitted featurised parquet)
	uv run --group data python scripts/build_eda_artifacts.py

certs:  ## Generate the self-signed TLS cert for the nginx proxy (Card 3.3)
	mkdir -p deploy/nginx/certs
	openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
	  -keyout deploy/nginx/certs/privkey.pem \
	  -out deploy/nginx/certs/fullchain.pem -subj "/CN=localhost"

up: certs  ## Build + start the compose stack (proxy + api + mlflow)
	docker compose up --build -d

down:  ## Stop the compose stack
	docker compose down

build:  ## Build the default (api) image — train/bert are profile-gated
	docker compose build

dag-up:  ## Start Airflow (UI: http://127.0.0.1:8080) — profile-gated, default stack unaffected
	docker compose --profile airflow up -d --build airflow

# `compose down` is project-wide and would take proxy/api/mlflow with it; `rm -sf` stops
# and removes just this container, keeping the airflow_home volume (metadata + history).
dag-down:  ## Stop + remove the Airflow service only (default stack untouched)
	docker compose --profile airflow rm -sf airflow

dag-trigger:  ## Trigger the data_pipeline DAG from the CLI (needs `make dag-up` first)
	docker compose exec airflow uv run --no-sync airflow dags trigger data_pipeline

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__