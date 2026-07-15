<!-- Target `dev`, never `main` (CONTRIBUTING §3). Keep < ~400 changed lines — split large work. -->

## What
<!-- Bullet the changes. -->

Closes # <!-- or "Part of #NN" when the issue spans several PRs -->

## Why
<!-- Motivation / the card it serves. One or two sentences. -->

## How tested
<!-- Commands + results, e.g.:
     - `make test` (15 passed)
     - `uv run pre-commit run --all-files` (ruff, black, detect-secrets pass)
     - `make api` + curl /predict -->

## Not in this PR
<!-- Deliberately out of scope / follow-ups + where they're tracked. Delete if N/A. -->

## Checklist
- [ ] Branch `feature|fix|docs/<initials>-<slug>` cut from `dev`; PR targets `dev` ([CONTRIBUTING §3](../CONTRIBUTING.md#3-git-workflow))
- [ ] Conventional Commit title, e.g. `feat(api): ...` — it becomes the squash commit (§4)
- [ ] Lint + tests green locally: `uv run ruff check . && uv run black --check .` · `uv run pytest` (§6)
- [ ] No large files or secrets — data/weights go via DVC (§7); pre-commit installed (§6)
- [ ] Deps changed? `pyproject.toml` + `uv.lock` committed together (§2)
- [ ] Non-obvious decision made? Short ADR note under `docs/` (§8)
- [ ] Meets the [Definition of Done](../docs/PROJECT_MANAGEMENT.md#4-definition-of-done)
