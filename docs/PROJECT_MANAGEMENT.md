# Project Management

How this project is planned, tracked, and reviewed. Read this to know **what to work on
next, who owns it, and how we report progress** at the weekly check-in.
All other docs → the **Docs map** in [`README.md`](../README.md).

The **GitHub Project board is the single source of truth for tasks and status.**
Roadmap + per-task technical detail (commands, acceptance, gotchas) → [`MILESTONES.md`](../MILESTONES.md).

---

## 1. Team, roles & rotation

- Team: **Marco, Mykola, Dilshana, Luc**
- Mentor: **Kilyan** — weekly check-ins, 1h max
- Whole team owns the backlog: if a task stalls, it's everyone's problem.

Work is split into four **tracks**, and each member rotates to a new track every **phase**:

| Track | Covers |
|-------|--------|
| **A — Modeling & tracking** | training pipelines, MLflow, evaluation, drift analysis |
| **B — Infra & CI** | Docker/compose, reverse proxy (nginx/ingress), CI/CD, Kubernetes, Prometheus/Grafana plumbing |
| **C — Data & versioning** | data pipelines, DVC, ingestion, retraining data |
| **D — Serving & API** | FastAPI, registry integration, auth, instrumentation |

| | Sprint 1 (done) | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| **Marco** | A | B | C | D |
| **Mykola** | B | C | D | A |
| **Dilshana** | C | D | A | B |
| **Luc** | D | A | B | C |

**Why per phase, not per sprint:** phases span two sprints, and rotating mid-phase would hand
off half-built work (an MLflow server, a DAG). Rotating per phase keeps continuity inside each
deliverable while still guaranteeing that **by the defence every member has owned all four
MLOps areas** — which is exactly what the mentoring format (each member presents their own
tasks) rewards. The per-task assignments live in the milestone task headers
(`MILESTONES.md`, "Owner:" on every Phase 2–4 task).

## 2. How we track work — GitHub Issues + Project board

Everything actionable is a **GitHub Issue**. Issues move across a **GitHub Project board**:

```
Backlog  →  To Do  →  In Progress  →  In Review  →  Done
```

- **Backlog** — captured but not scheduled.
- **To Do** — pulled into the current sprint, has an owner.
- **In Progress** — someone is actively working it (their branch is open, cut from `dev`).
- **In Review** — PR opened against `dev`, waiting on review + CI.
- **Done** — PR squash-merged into `dev`, CI green, docs updated (see Definition of Done).
  `dev → main` merges happen once per milestone ([CONTRIBUTING.md §3](../CONTRIBUTING.md)).

### Labels

| Group | Labels |
|-------|--------|
| Phase | `phase-1` `phase-2` `phase-3` `phase-4` |
| Type | `type:data` `type:model` `type:infra` `type:api` `type:docs` |
| Priority | `priority:high` `priority:med` `priority:low` |

### Issue hygiene
- Title = action (`Add Dockerfile for the API`), not a topic (`Docker`).
- One deliverable per issue. If it needs two PRs, it's two issues.
- Assign an owner and a phase label the moment it enters **To Do**.
- Link the PR to the issue with `Closes #NN`.

### Board setup (one-time)
Use GitHub's **"Iterative development"** Project template — its weekly **Iteration** field maps
straight to our weekly check-in, and one Project holds several views:
- **Board** (group by Status) — daily flow, the columns above.
- **Iteration** (group by Iteration) — what's committed for this week's check-in.
- **Roadmap** (group by a `Phase` field, using the phase target dates) — the 4-phase timeline.

Seed it: create the labels above, import `docs/TASKS_DONE.md` as closed issues (board
history), then **one issue per Phase 2–4 task in `MILESTONES.md`**, with the issue assignee =
the task's **Owner**.

## 3. Weekly check-in (mentoring session)

Cadence: **weekly, 1 hour max**, on Zoom/Teams. **Every member presents — no exceptions.**
Each person answers three points:

```markdown
### <Name> — week of <date>
1. Tasks done since last check-in:
   - #NN ...
2. Successes / challenges / blockers:
   - Win: ...
   - Blocker: ... (need help with ...)
3. Remaining backlog / next up:
   - #NN ...
```

At the end of each session the mentor may set collective or individual objectives (by
completion or by date) to balance workload. Capture them as new issues before you close the
call.

## 4. Definition of Done

A task is **Done** only when all that apply are true:

**Code**
- [ ] Function has a unit test (main-step functions are required to)
- [ ] `ruff` + `black` clean
- [ ] CI green
- [ ] Reviewed & approved by ≥1 teammate
- [ ] Reproducible from a clean clone (`uv sync` + `make pull` / `docker compose up` after `make pull`)
- [ ] Docs updated if behavior/interface changed

**Data / model**
- [ ] Tracked with DVC (Phase 2+), not committed raw
- [ ] Transformation / feature documented

**Docs**
- [ ] Renders correctly, links resolve, no TODO left in the merged version

## 5. Kick-off compliance map

Where each course requirement is satisfied in this repo — the checklist for mentor
**control points** (technical / MLOps / documentation reviews):

| Kick-off requirement | Where |
|----------------------|-------|
| 4 phases with objectives & key components | `MILESTONES.md` roadmap + Milestones 1–4 ("Kickoff mapping" line each) |
| Project framing (ML Canvas) | `docs/ML_CANVAS.md` (10 blocks, Dorard v1.1) |
| Every member presents tasks / challenges / backlog | §3 check-in template; per-member Owners in `MILESTONES.md` |
| Team owns the backlog | §1 + the Project board |
| Reproducibility challenge | `CONTRIBUTING.md` (env, lockfile, golden rule) + CI |
| Collaboration challenge | `CONTRIBUTING.md` (git flow, reviews) + this file |
| Production challenge | Milestones 3–4 (deployment, monitoring, retraining) |
| Success criteria: functional pipeline, automated processes, monitored deployment | Milestones 2–4 "Done when" gates |
| Success criteria: clear docs, API docs, maintenance guides | Task 4.3 + the Docs map in `README.md` |

---

**When in doubt about what to do next:** pick the highest-priority unblocked issue in **To Do**
that matches your current track (§1), assign yourself, move it to **In Progress**.
