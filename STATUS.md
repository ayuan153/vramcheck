# STATUS.md — handoff baton

> Update this **every** session (END ritual). One clear `NEXT:` line at the top.

**NEXT:** **P2 — CLI** (Typer + Rich over `core/`): verdict + breakdown + max-batch@ctx + OOM line +
sweep table + `--json`. P1 core library is done (12/12 unit tests green).

---

## Where we are
- **Phase:** P1 (core memory-model library) **complete — 12/12 tests green.** Next: P2 (CLI).
- **North star (restated):** know exactly what you can run before you rent a GPU — predicted-vs-actual
  OOM within ≤10% (stretch ≤5%) for the 5 v0.1 models (4 GQA + 1 MLA) on one A100-80GB.

## Done — Phase 0 (2026-06-01)
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

## Done — P1 core library (2026-06-01)
- Pure-Python, dependency-free package `vramcheck/core/`: `models.py` (5 vendored configs),
  `gpus.py` (4 GPUs), `kv.py` (MHA/GQA/MQA + MLA + PagedAttention rounding), `memory.py`
  (weights / activation / overhead / budget / max_batch / fits / sweep). `pyproject.toml` + README.
- `tests/test_core.py`: **12/12 green** (`python3 -m unittest discover -s tests -t .`) — exact
  hand-computed values incl. Llama-70B 327,680 B/token, 2.5 GiB/seq @8k, GQA=8×MHA, MLA=31,104
  B/token, block-16 rounding, fp8=½fp16, budget/max_batch=54, OOM→0.
- activation (10% of weights) + overhead (1 GiB) are placeholders flagged for P3 calibration.

## Decisions locked (2026-06-01)
1. **Name:** `vramcheck` (re-verify PyPI / domain / trademark before publish).
2. **Web:** Pyodide single-source on GitHub Pages.
3. **Validation infra:** rent on-demand A100-80GB (no purchase; ~$10-20); HF token needed at validation.
4. **Accuracy:** validation is a mandatory MVP gate; MLA validated in MVP via DeepSeek-V2-Lite (5th model).

## Next actions (granular phases — see DESIGN §11)
- ✅ **P1 done:** core library + 12 unit tests green.
- **P2 (next):** Typer + Rich CLI over `core/`; verdict + breakdown + sweep table + `--json`.
- **P3 (MVP accuracy gate):** rent A100-80GB; vLLM `# GPU blocks` + OOM search; fill §6 table for all
  5 models; calibrate activation `k` + overhead. Gate ≤10% (stretch ≤5%).
- **P4:** Pyodide web one-pager on GitHub Pages. **P5:** publish to PyPI + broadcast.

## Repo state
- Working state: P1 core library + tests (green). Git: P1 checkpoint committed + pushed.
- Remote: `github.com/ayuan153/canirun`. Package / CLI / domain = `vramcheck`.
