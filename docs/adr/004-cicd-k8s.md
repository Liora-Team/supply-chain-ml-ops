# ADR 004 — CI/CD & Kubernetes target for Phase 3 (Card 3.4)

## Context
Card 3.4 requires: SHA-tagged images, k8s deployment with >=2 replicas, probes on `/health`, Secrets for API key + MLFLOW_*, ingress-nginx with TLS termination, and a one-step rollback.

## Decision
- **Image registry**: GHCR (free, integrated with GitHub).
- **Deployment target (demo & CI)**: Local cluster (`kind`).
- **Deploy trigger**: Automatic on push to `dev`. The full `kind` rehearsal runs on merges to `dev`. This is intentional to limit CI runtime and avoid publishing images to GHCR on every PR, while PRs remain gated by the standard validation workflow (`ci.yml`).
- **Rollback**: `kubectl rollout undo deployment/sc-mlops-api`.
- **Ingress & TLS**: `ingress-nginx` controller with Kubernetes TLS Secret (self-signed for local demo).

## Consequences
- Fully reproducible on any developer laptop and in CI.
- A shared cloud cluster can be added later without changing the image/tagging strategy.
