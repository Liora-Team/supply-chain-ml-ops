---
name: Task card
about: A unit of work for the MLOps board — same card shape as docs/TODO.md
title: "Card <M.N> — <action-oriented title>"
labels: []
assignees: []
---

<!-- One clear deliverable per card. If it needs two PRs, split it (like 2.4-A / 2.4-B). -->

| | |
|---|---|
| **Assignee** | |
| **Labels** | `phase-N` · `type:data\|model\|api\|infra\|docs` · `priority:high\|med\|low` |
| **Milestone** | M2 / M3 / M4 |
| **Branch** | `feature/<initials>-<short-slug>` (from `dev` — [CONTRIBUTING §3](../../CONTRIBUTING.md#3-git-workflow)) |
| **Depends on** | #NN / none — note what can run in parallel |
| **Files** | paths this card touches (helps avoid branch collisions) |

**Why:** <!-- the MLOps concept this teaches / the objective it serves — one sentence -->

### Open questions

<!-- Delete this section if scope is already locked. Draft-scope cards (M3/M4): answer at
     the phase kickoff, record choices as short ADR notes under docs/ (CONTRIBUTING §8). -->

### Subtasks

- [ ]

### Done when

<!-- A testable acceptance check, e.g. "moving the production alias changes what the API
     serves — no code edit". -->

- [ ] <acceptance check>
- [ ] Meets the [Definition of Done](../../docs/PROJECT_MANAGEMENT.md#4-definition-of-done)
- [ ] Reproducible from a clean clone (`make setup`; + `make pull` once DVC lands — Card 2.3)
