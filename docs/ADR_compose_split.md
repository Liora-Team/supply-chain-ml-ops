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
