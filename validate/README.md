# validate/ — P3 accuracy harness (the MVP gate)

Measures the **real** KV budget vLLM allocates per model on one GPU, then calibrates the two
approximate knobs in `vramcheck/core/memory.py` (`DEFAULT_ACT_FRACTION`, `DEFAULT_OVERHEAD_GIB`)
and reports predicted-vs-measured error. Ship gate: **≤10% (stretch ≤5%)** on all 5 models
(see `docs/DESIGN.md` §4, §6).

> Status: **ready to run, not yet run** (needs a GPU). The pure logic (`parse.py`, the calibration
> math, `calibrate.analyze`) is unit-tested off-GPU in `tests/test_validate.py`. Only `run.py`
> touches vLLM/torch, via lazy imports.

## How it works
vLLM profiles each model at startup and fixes `# GPU blocks` — a **crash-free** measurement of the
KV budget. We read it (engine API, falling back to parsing the startup log), convert to bytes via
`core.per_token_kv_bytes`, and recover the true `activation + overhead = usable − weights − KV`.
A 2-parameter least-squares fit across the 5 models gives the calibrated defaults. When vLLM logs
its "Memory profiling results" (model weights / activation / non-torch / KV reserved), the harness
uses those **measured** values directly instead of approximate `num_params`, and prints suggested
`num_params` corrections for `vramcheck/core/models.py`.

## Prerequisites (on the rented box)
- 1× **A100-80GB** (cloud rental; no purchase). ~$10–20, a couple of hours.
- `pip install vllm` (pulls a matching torch/CUDA).
- A **Hugging Face token** with access to gated Llama weights: `huggingface-cli login` or `export HF_TOKEN=...`.
- `trust_remote_code` is already set for DeepSeek-V2-Lite in `config.py`.

## Run
```sh
# 1) measure KV budgets (writes validate/results.json, checkpointing after each model)
python -m validate.run --util 0.90 --out validate/results.json
#    subset while iterating:
python -m validate.run --models llama-3.1-8b,deepseek-v2-lite

# 2) calibrate + see predicted vs measured (no GPU needed for this step)
python -m validate.calibrate validate/results.json --ctx 8192
```
Then set the two printed defaults in `vramcheck/core/memory.py`, re-run `python -m unittest
discover -s tests -t .` (adjust the budget/max_batch expectations that depend on those defaults),
and paste the predicted-vs-measured table into `docs/DESIGN.md` §6.

## ⚠️ The 70B caveat (discuss in infra setup)
Llama-3.1-70B at fp16 is ~141 GB and **cannot load on one 80GB GPU** (our tool correctly reports
OOM). To validate the 70B GQA path on a single GPU, `config.py` defaults it to an **AWQ-int4**
checkpoint (~35 GB). We should confirm the exact repo and quant together.

## Infra setup — things to settle together
- **Provider/instance:** RunPod vs Lambda vs Vast; confirm a true A100-80GB (not 40GB/MIG).
- **HF access:** token with Llama-3.1 gating approved; DeepSeek/Qwen need `trust_remote_code`.
- **70B:** confirm the AWQ-int4 repo id (or choose fp8); set matching `weight_dtype` in `config.py`.
- **vLLM version drift:** if `# GPU blocks` isn't found, capture startup log
  (`python -m validate.run ... 2> startup.log`) — `parse.parse_gpu_blocks` handles the known formats;
  we extend the patterns if the wording changed.
- **Block size / dtype:** default block_size=16, KV fp16; revisit if testing fp8 KV.
- **Optional OOM cross-check:** the `# GPU blocks` budget is the primary signal; an empirical
  max-concurrency probe can be added if we want a second confirmation of the OOM line.
