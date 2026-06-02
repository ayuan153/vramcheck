# DECISIONS.md — append-only rationale log

> Format: `date · decision · alternatives · why · vision impact`. **Never edit past entries.**

## 2026-06-01 — Establish documentation spine
- **Decision:** Adopt VISION / DESIGN / STATUS / DECISIONS / AGENT-CONVENTIONS as the living spine; follow the session ritual.
- **Alternatives:** Single README; issue tracker only.
- **Why:** Project spans many agent sessions over months; continuity must live in files, not memory.
- **Vision impact:** none.

## 2026-06-01 — Name `canirun` is not viable for distribution
- **Decision:** Do not publish under `canirun`. Recommend **`vramcheck`** (clear on PyPI + GitHub); human to confirm, then re-verify before first publish. Docs use `canirun` as a placeholder meanwhile.
- **Alternatives:** `canirun` (PyPI taken by a competing tool; `canirun.ai` active in same niche; org taken), `llmfit` (PyPI taken, same niche), `willitrun` (GitHub taken).
- **Why:** Direct user confusion + can't claim the pip name or the obvious domain — fatal for an adoption-first launch.
- **Vision impact:** flagged-to-human (brand/launch); vision itself is name-independent.

## 2026-06-01 — Confirm the competitive gap is real
- **Decision:** Proceed; the gap is genuine. Wedge = attention-aware KV (incl. MLA) + PagedAttention waste + activation/overhead + an actionable batch×context capacity table.
- **Alternatives:** Abandon (a good tool already exists) — rejected after feature-matrix review of HF gaunernst, apxml, Smirnov, gpu_poor, hf-vram-calc.
- **Why:** No tool models PagedAttention/prefix-cache/chunked-prefill or emits a capacity sweep; the only pip CLI uses a blunt `×1.2` overhead fudge.
- **Vision impact:** none (confirms wedge).

## 2026-06-01 — Memory model definition
- **Decision:** Model peak resident VRAM with vLLM semantics: weights + attention-aware KV (MHA/GQA/MQA via `num_kv_heads`; MLA via `kv_lora_rank + qk_rope_head_dim`, no factor 2) + PagedAttention block rounding (block_size=16) + activation peak + CUDA overhead; budget = `util×VRAM − weights − activation − overhead`; `max_batch = floor(budget / kv_per_seq)`.
- **Alternatives:** Naive `params×2 + ×1.2` fudge (rejected as inaccurate); training-style accounting (rejected, inference-specific).
- **Why:** Correctness is the credibility; the fudge-factor tools are exactly what we beat.
- **Vision impact:** none (this *is* the core).

## 2026-06-01 — Validation via vLLM `# GPU blocks` log (crash-free ground truth)
- **Decision:** Use vLLM's startup-logged `# GPU blocks: N` as precise KV-budget ground truth, plus an OOM binary-search cross-check, on one GPU.
- **Alternatives:** OOM-crash sweeps only (slower, wasteful); external load tester (not required, standalone-first).
- **Why:** Cheap, exact, isolates the two biggest error sources (activation + overhead) for calibration without crashing.
- **Vision impact:** none (serves the north-star proof).

## 2026-06-01 — Lock v0.1 scope
- **Decision:** 4 GQA models (Llama-3.1-8B/70B, Mistral-7B, Qwen2.5-32B); GPUs A100-40/80, H100-80, RTX 4090; outputs = verdict + breakdown + max-batch@ctx + OOM line + capacity sweep; CLI + Pyodide web. MLA formula in, accuracy validation deferred to v0.2. Ship gate ≤10% (stretch ≤5%).
- **Alternatives:** Include MLA validation + multi-GPU + prefix-cache now (rejected — not the thinnest thing that proves accuracy; balloons validation cost).
- **Why:** v0.1 = thinnest artifact that proves accuracy on the dominant r/LocalLLaMA case (dense GQA models).
- **Vision impact:** flagged — MLA is named in the wedge but its *validation* is deferred; confirm acceptable.

## 2026-06-01 — Tech stack (minor)
- **Decision:** Python; CLI via Typer + Rich; web = one-page static site running the same Python `core/` via Pyodide on GitHub Pages (single source of truth); packaging via `pyproject.toml`; vendored numeric config snapshot for offline/gated models with HF fetch fallback.
- **Alternatives:** Port math to JS for the web (rejected — divergence risk); serverless API (rejected — hosting + cost); Rust/Go CLI (rejected — audience is Python-native, higher friction).
- **Why:** Lowest adoption friction; CLI and web can never produce different numbers.
- **Vision impact:** flagged-to-human (web approach is open decision §2).

## 2026-06-01 — Phase 0 approved; four decisions locked
- **Decision:** (1) Name = `vramcheck`; (2) web = Pyodide single-source on GitHub Pages; (3) validation infra = on-demand **A100-80GB** cloud rental (no hardware purchase, ~$10-20 total); (4) accuracy validation is a **mandatory MVP gate**, and MLA is validated in the MVP via DeepSeek-V2-Lite (not deferred).
- **Alternatives:** keep `canirun` (taken); JS port / serverless web (divergence / cost); buy a GPU (unnecessary capex); defer MLA validation to v0.2 (weaker on the wedge + against "accuracy in MVP").
- **Why:** human confirmed name + web approach; accuracy is the product, and DeepSeek-V2-Lite makes MLA validation cheap on the same rented GPU.
- **Vision impact:** none — strengthens the north-star proof (now 5 models incl. one MLA).

## 2026-06-01 — Engineering broken into granular phases
- **Decision:** P1 core lib → P2 CLI → P3 validation (MVP accuracy gate) → P4 web → P5 launch. MVP = P1-P4; launch = P5. (DESIGN §11)
- **Alternatives:** monolithic "build then validate" — rejected; validation must gate the MVP, not trail it.
- **Why:** per human direction to break engineering phases more granularly with validation inside the MVP.
- **Vision impact:** none.

## 2026-06-01 — Name re-verified after human pushback; `vramcheck` stands
- **Decision:** Keep `vramcheck`. Human preferred `canirun` "if viable" and couldn't find it on pip, so we re-verified independently against the PyPI JSON API.
- **Evidence:** `pypi.org/project/canirun/` → 200, `canirun` 1.0.1 (PythonicVarun, Jan 2026, "check if you can run a HF model locally"); GitHub `PythonicVarun/canirun` real (31 commits); `canirun.ai` live competitor. npm `canirun` free but irrelevant (Python package).
- **Why:** Not just a name clash — a direct competitor in our exact niche/audience; using `canirun` would invite "isn't this just canirun?" and kill the adoption wedge.
- **Vision impact:** flagged-to-human (brand). Vision unaffected.

## 2026-06-01 — P1 core library: pure-Python + zero-install tests
- **Decision:** Implement `core/` with no third-party deps (stdlib only); ship tests as stdlib `unittest`, runnable via `python3 -m unittest` (no pytest/network). Vendor model configs as plain dataclasses; live HF fetch deferred.
- **Alternatives:** numpy/pydantic (needless weight); pytest (install friction for a zero-dep core); fetch HF configs at runtime now (network + gated-token friction).
- **Why:** the core is simple arithmetic; minimal deps = lowest adoption friction and trivially testable in any environment.
- **Vision impact:** none (serves standalone-first + adoption).

## 2026-06-01 — Design documentation consolidated under docs/
- **Decision:** Move the spine (VISION/DESIGN/STATUS/DECISIONS/AGENT-CONVENTIONS) into `docs/`; `README.md` stays at root as the GitHub front door; per-component design docs (e.g. `docs/cli.md`) live in `docs/` too.
- **Alternatives:** keep everything at root (cluttered); split design vs. logs across root and docs/ (inconsistent).
- **Why:** human asked to keep design documentation in `docs/`; siblings keep cross-references intact.
- **Vision impact:** none.

## 2026-06-01 — P2 CLI uses stdlib argparse (Rich/Typer deferred)
- **Decision:** Build the CLI with stdlib `argparse` + a plain-text table; no runtime dependencies. Rich/Typer deferred to launch polish as an optional `[pretty]` extra. Revises the DESIGN §7 "Typer + Rich" choice.
- **Alternatives:** Typer + Rich now (2 deps + install/network friction, not testable in a bare env); argparse + Rich (1 cosmetic dep).
- **Why:** adoption-first — `pip install vramcheck` pulls zero transitive deps and the CLI runs/tests anywhere; the product is the accurate number + table, not box-drawing. Presentation-only, reversible.
- **Vision impact:** none (serves adoption + standalone-first).

## 2026-06-01 — P4 web tool loads the real core via Pyodide
- **Decision:** The one-page web tool fetches the actual `vramcheck/core/*.py` files into Pyodide's FS and imports them, rather than porting math to JS or standing up an API. Hosted as static files on GitHub Pages (serve repo root so `web/` can fetch `../vramcheck/`).
- **Alternatives:** JS reimplementation (divergence risk); micropip-install a published wheel (needs publish + build); serverless API (cost + infra).
- **Why:** guarantees the web numbers equal the CLI numbers (single source of truth) with zero backend.
- **Vision impact:** none (implements locked DESIGN §7).
