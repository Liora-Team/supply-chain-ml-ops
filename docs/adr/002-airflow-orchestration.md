# ADR 002: Airflow as a Single Standalone Container

**Context:**
Cards 3.1 (data half) and 3.2 (model half) need an orchestrator so the pipeline's order and
retries live in code and the whole chain runs from one trigger. Airflow is the course-mandated
module and the industry default, and both halves must share one stack. The official Airflow
docker-compose file ships ~7 services (Postgres, Redis, scheduler, workers, triggerer, …),
which would double this repo's service count for a manually-triggered course DAG.

**Decision:**
Run Airflow 3.3 as **one compose service** (`airflow` profile, Dockerfile `airflow` target)
executing `airflow standalone` — scheduler, dag-processor and UI in a single process, SQLite
metadata on the `airflow_home` named volume. The image extends the repo's shared `base` stage
and syncs the `orchestration` + `data` uv groups, so Airflow is pinned by `uv.lock` like every
other dependency (no Airflow constraints files).

Two deliberate deviations, both container-scoped:

- **UI port `127.0.0.1:8080`** is published directly instead of routing through nginx. The
  proxy is always-on but `airflow` is profile-gated, and nginx fails at startup when an
  upstream doesn't resolve — the same reason `training` isn't proxied. Binding to localhost
  keeps the "nothing new publicly exposed" spirit of the single-entry rule; the UI runs
  login-free (`SIMPLE_AUTH_MANAGER_ALL_ADMINS`) only because of that binding.
- **DVC runs inside the container** for the DAG's final `dvc add` + `dvc push` step. The
  container mounts `./data` (rw), `./.dvc` (rw — shared config *and* cache, so hashes computed
  in-container are valid on the host) and `./.git` read-only (DVC refuses to run outside a git
  repo; add/push only read it). Credentials arrive as `DAGSHUB_TOKEN` from `.env`, mapped to
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (the S3 remote falls back to boto3's env chain)
  — never baked into the image. The refreshed `.dvc` pointers appear on the host for a human
  `git commit`; no git credentials ever enter the container.

**Consequences:**

- The default `docker compose up` stack is unchanged; Airflow is opt-in via
  `--profile airflow` (`make dag-up`).
- One process and SQLite mean no HA and no parallel scheduler — fine for a manual-trigger
  course pipeline, not a production deployment shape.
- Card 3.2's model DAG drops into the same `dags/` tree and container with no new
  infrastructure.
- Versioning a DAG run end-to-end still takes one human step (committing the pointer files),
  by design — automation stops where git credentials would have to enter a container.
