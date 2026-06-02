# STATUS.md — handoff baton

> Update this **every** session (END ritual). One clear `NEXT:` line at the top.

**NEXT:** Begin **P1 — core library** (`core/`: kv, memory, gpus, models, report) with unit tests
against hand-computed values. Phase 0 approved 2026-06-01; all four decisions locked (DESIGN §10).

---

## Where we are
- **Phase:** 0 (research + design) approved. **Now starting Phase 1 (core lib).**
- **North star (restated):** know exactly what you can run before you rent a GPU — predicted-vs-actual
  OOM within ≤10% (stretch ≤5%) for the 5 v0.1 models (4 GQA + 1 MLA) on one A100-80GB.

## Done this session (2026-06-01)
- Documentation spine created: `AGENT-CONVENTIONS.md`, `VISION.md`, `DESIGN.md`, `STATUS.md`, `DECISIONS.md`.
- Name check: **`canirun` not viable** (PyPI taken by competing tool; `canirun.ai` active in-niche;
  GitHub org taken). Recommended fallback **`vramcheck`** (clear on PyPI + GitHub).
- Gap confirmed real: no tool models PagedAttention waste / prefix-cache / chunked prefill, none
  emits a batch×context capacity table; only pip CLI (`hf-vram-calc`) uses a `×1.2` fudge.
- Memory model closed: weights + attention-aware KV (MHA/GQA/MQA + MLA) + PagedAttention rounding +
  activation + CUDA overhead; configs from HF `config.json` (vendored snapshot for offline/gated).
  Ranked error sources documented.
- Validation method: use vLLM's logged `# GPU blocks: N` as crash-free ground truth, plus an OOM
  binary-search cross-check. One GPU, a few GPU-hours.
- v0.1 scope locked (4 GQA models, 4 GPUs, breakdown + max-batch + OOM line + sweep table; OUT list).
- Ship gate: ≤10% MUST / ≤5% stretch. Stretch (deployment planner) + broadcast draft written.

## Decisions locked (2026-06-01)
1. **Name:** `vramcheck` (re-verify PyPI / domain / trademark before publish).
2. **Web:** Pyodide single-source on GitHub Pages.
3. **Validation infra:** rent on-demand A100-80GB (no purchase; ~$10-20); HF token needed at validation.
4. **Accuracy:** validation is a mandatory MVP gate; MLA validated in MVP via DeepSeek-V2-Lite (5th model).

## Next actions (granular phases — see DESIGN §11)
- **P1 (now):** scaffold `core/` + `pyproject.toml`; implement `kv.py` + `memory.py`; unit tests vs
  hand-computed values (Llama-3.1-70B @8k = 2.68 GB KV; DeepSeek-V2-Lite MLA; etc.).
- **P2:** Typer + Rich CLI over `core/`; sweep table + `--json`.
- **P3 (MVP accuracy gate):** rent A100-80GB; vLLM `# GPU blocks` + OOM search; fill §6 table for all
  5 models; calibrate activation `k` + overhead. Gate ≤10% (stretch ≤5%).
- **P4:** Pyodide web one-pager on GitHub Pages. **P5:** publish to PyPI + broadcast.

## Repo state
- Working state: docs only, consistent. No code yet. Git initialized; committed + pushed at this checkpoint.
- Remote: `github.com/ayuan153/canirun`. Package / CLI / domain = `vramcheck`.
