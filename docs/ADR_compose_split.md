# ADR — Card 2.4-B: compose split, nginx entry, MLflow prefix, CPU torch pin

Lightweight decision record (context → decision → consequence) for the non-obvious choices
in the microservices compose split ([Card 2.4-B](TODO.md), PR #30).

## 1. One multi-target Dockerfile instead of one Dockerfile per service

**Context.** api / training / bert differ only in which uv dependency groups they sync and
which files they copy; the base (pinned Python, uv, lockfile, non-root user) is identical.

**Decision.** A single `Dockerfile` with a shared `base` stage and named targets
(`api` — default, `training`, `bert`); compose selects via `build.target`.

**Consequence.** Zero duplicated base setup and one place to harden; the trade-off is that
all images rebuild their dependency layer when `uv.lock` changes.

## 2. MLflow behind the proxy sub-path: `--static-prefix /mlflow`

**Context.** nginx is the single public entry; the MLflow UI lives at `/mlflow/`. Without a
prefix the UI's absolute asset links (`/static-files/…`) 404 through the proxy.

**Decision.** Run the server with `--static-prefix /mlflow` and proxy `/mlflow/` without
stripping. Side effects handled: the REST API moves under the prefix too, so the
**in-network tracking URI is `http://mlflow:5000/mlflow`** (mirrors DagsHub's `….mlflow`
suffix); MLflow 3.x's DNS-rebinding guard 403s service-name clients, so
`--allowed-hosts localhost,localhost:*,mlflow,mlflow:*` is set.

**Consequence.** One URL shape everywhere (`<entry>/mlflow`), UI and API both reachable
through the single entry; anyone pointing tools at the local server must include the
`/mlflow` path suffix.

## 3. CPU-only torch wheels — but only for arm64 Linux

**Context.** Default PyPI torch wheels for Linux bundle the multi-GB nvidia CUDA stack. The
bert container is CPU-only (built/run on Apple-silicon laptops → linux/aarch64), where those
wheels are dead weight (~10 GB image) and can exhaust disk at build time. DistilBERT
retraining, however, happens on a separate GPU machine and must keep CUDA-capable wheels.

**Decision.** `[tool.uv.sources]` pins `torch`/`torchvision` to the PyTorch CPU index with
marker `sys_platform == 'linux' and platform_machine == 'aarch64'` only.

**Consequence.** bert image is 3.6 GB; macOS (`uv sync --group bert`) and x86_64 Linux
(GPU boxes, most CI runners) resolve exactly as before from the same lockfile.

## 4. bert weights default to the mounted DVC location

**Context.** Compose passes `DISTILBERT_*` through even when unset, and an empty string
overrides `src/registry.py`'s default paths. Relative fallbacks containing `/` would trip
the registry's hub-id heuristic and be marked loadable while missing.

**Decision.** The bert service defaults the vars to the absolute in-container paths
`/app/models/distilbert_{3,5}class/final` inside the `models/` bind mount (where Card 2.3's
`dvc pull` restores the weights).

**Consequence.** With weights absent the service still starts (registry marks the entries
not loadable — classical serving unaffected); once DVC restores them, no configuration is
needed. Override via `.env` for a custom path or HF hub id.

## 5. Non-root images, but the training job runs as the host user

**Context.** Every target runs as UID 10001 for container hardening. `training` is the one
target that *writes* — `scripts/build_pipelines.py` emits into the host-owned `models/` bind
mount. On Linux those files belong to UID 1000, so the job gets `EACCES`; Docker Desktop and
OrbStack remap mount ownership, so the failure is invisible on macOS.

Ownership also can't be handed over with `chown -R app:app /app`: `UV_PROJECT_ENVIRONMENT`
puts the venv under `/app`, so a recursive chown rewrites every dependency file into a
duplicate image layer — the entire torch stack twice in the bert image.

**Decision.** Ownership is set on the `COPY` (`--chown=app:app`) and never recursively; the
venv stays root-owned and world-readable, which is all `uv run --no-sync` needs. The
`training` service alone takes `user: "${DOCKER_UID:-10001}:${DOCKER_GID:-10001}"`, defaulting
to the image user, plus `HOME=/tmp` because an overridden UID has no `/etc/passwd` entry.

The vars are **not** named `UID`/`GID`: bash and zsh both mark those readonly, so the obvious
`UID=$(id -u) docker compose …` aborts with `UID: readonly variable` before compose runs. CI
caught this — the first version of this decision shipped that broken incantation in the README.

**Consequence.** Images keep a non-root default and stop carrying a duplicated venv. Linux
contributors prefix the train profile with `DOCKER_UID=$(id -u) DOCKER_GID=$(id -g)`; macOS
needs nothing. CI asserts all of it on a real Linux runner, including a control proving UID
10001 still cannot write the mount.

## 6. The bert image still bakes `models/`

**Context.** The bert service bind-mounts `./models`, so the copy baked at build time is
shadowed at runtime — dead weight on the face of it.

**Decision.** Keep it. The bert target runs the same `api.main:app` and serves the classical
pipelines too, so baking them keeps the image runnable standalone (`docker run` with no
mount). `.dockerignore` already excludes `models/distilbert_*/`, so the cost is a few MB of
joblib, not the 510 MB of weights.

**Consequence.** Slight redundancy under compose, no footgun outside it.
