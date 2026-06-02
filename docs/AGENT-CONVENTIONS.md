# AGENT-CONVENTIONS.md

> How any agent (or human) works on this repo across many sessions over many months.
> Read this first, every session. It is short on purpose.

This project will span many sessions and many agents. Continuity comes from a small
set of living documents, not from memory. These conventions keep the repo coherent
and never broken.

> **Location:** the spine lives in `docs/` (this file, `VISION.md`, `DESIGN.md`, `STATUS.md`,
> `DECISIONS.md`). The root `README.md` is the public front door. Detailed per-component design
> docs also go in `docs/` (e.g. `docs/cli.md`).

---

## The documentation spine

Four+1 living documents. Keep them current; they are the source of truth.

| File | Role | Mutability |
|------|------|------------|
| `VISION.md` | The long-term arc, the wedge, the north star. *Why* this exists. | Rarely changes. Changing it is a deliberate, flagged event. |
| `DESIGN.md` | Architecture, the memory model, locked scope per version, decisions. *What* and *how*. | Evolves per phase. |
| `STATUS.md` | The handoff baton: where we are right now, what's next, what's blocked. | Updated **every** session. |
| `DECISIONS.md` | Append-only rationale log: `date · decision · alternatives · why`. | Append only. Never edit past entries. |
| `AGENT-CONVENTIONS.md` | This file. The session ritual. | Rarely changes. |

If any of these is missing, recreate it before doing other work.

---

## Session ritual

### START (before any work)
1. Read `VISION.md`, then `STATUS.md`, then `DESIGN.md`.
2. **Restate the north star** in one sentence to confirm alignment.
3. Read the last few entries of `DECISIONS.md` so you don't re-litigate settled calls.
4. Pick up from "Next actions" in `STATUS.md`.

### DURING
- Stay inside the **locked scope** for the current version (see `DESIGN.md`). New ideas
  go to a "Stretch / later" list, not into the build.
- **Vision check:** before any meaningful decision, ask "does this change the long-term
  vision in `VISION.md`?" If yes, **stop and flag it to the human** — make it explicit;
  do not silently decide. Log the question in `DECISIONS.md`.
- Correctness over coverage. This tool's credibility is accuracy; a wrong number is worse
  than an unsupported model. When unsure, mark a value `APPROXIMATE` and say why.
- Never leave the repo broken. If you can't finish, leave it building/working and write
  down the exact next step.

### END (before you stop)
1. Update `STATUS.md`: what changed, current state, next actions, open questions/blockers.
2. Append any decisions to `DECISIONS.md` (`date · decision · alternatives · why`).
3. Confirm the repo is in a working state (docs consistent; if code exists, it builds/tests pass).
4. Leave one clear "NEXT:" line at the top of `STATUS.md`.

---

## Decision log format (`DECISIONS.md`)

Append entries like:

```
## YYYY-MM-DD — <short decision title>
- **Decision:** what we chose.
- **Alternatives:** what we rejected.
- **Why:** the reasoning / evidence.
- **Vision impact:** none | flagged-to-human | changes-VISION.md
```

---

## Working principles
- **Standalone-first.** `vramcheck` must be fully useful with nothing else installed and
  must never depend on a sibling project. Interop is optional, never required.
- **Adoption is the #1 goal.** Decisions that lower adoption friction win ties.
- **Honesty over polish.** Show the memory breakdown and the error band. Never hide the
  uncertainty that makes capacity planning hard.
- **Thinnest thing that proves the thesis.** Each version ships the smallest artifact that
  demonstrably moves the north-star metric.

\* Shipping name **`vramcheck`** (locked 2026-06-01); repo `github.com/ayuan153/canirun`. See `DECISIONS.md`.
