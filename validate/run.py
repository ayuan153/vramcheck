"""Measure the real KV budget vLLM allocates per model on ONE GPU, write results.json.

Requires a GPU + `pip install vllm`. vLLM/torch are imported lazily inside functions so this
module imports fine off-GPU (only `run`/`measure` need the GPU).

Usage (on the rented A100-80GB):
  python -m validate.run --util 0.90 --out validate/results.json
  python -m validate.run --models llama-3.1-8b,deepseek-v2-lite   # subset

Loading a model triggers vLLM's startup profiling, which fixes `# GPU blocks` (the KV budget).
We read it from the engine API, falling back to parsing the captured startup log if the API
attribute moved in this vLLM version.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys

from . import config
from .parse import (
    parse_gpu_blocks, parse_kv_cache_tokens,
    parse_weight_gib, parse_activation_gib, parse_nontorch_gib, parse_kv_reserved_gib,
)


def _num_gpu_blocks(llm) -> int | None:
    """vLLM's attribute path for num_gpu_blocks has moved across versions — try the known ones."""
    candidates = [
        lambda: llm.llm_engine.cache_config.num_gpu_blocks,
        lambda: llm.llm_engine.model_executor.cache_config.num_gpu_blocks,
        lambda: llm.llm_engine.scheduler[0].block_manager.num_total_gpu_blocks,
    ]
    for c in candidates:
        try:
            v = c()
            if v:
                return int(v)
        except Exception:
            pass
    return None


def _total_gpu_memory_bytes() -> int | None:
    try:
        import torch
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return None


def measure(target: config.Target, util: float, max_model_len: int = 8192,
            block_size: int = 16) -> dict:
    from vllm import LLM  # lazy: only needed on the GPU box

    # Capture vLLM's INFO logs so we can parse the KV budget if the API attribute moved.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    for name in ("", "vllm"):
        logging.getLogger(name).addHandler(handler)
    try:
        llm = LLM(model=target.hf_id, gpu_memory_utilization=util,
                  max_model_len=max_model_len, **target.vllm_kwargs)
    finally:
        for name in ("", "vllm"):
            logging.getLogger(name).removeHandler(handler)

    log = buf.getvalue()
    blocks = _num_gpu_blocks(llm)
    if blocks is None:
        blocks = parse_gpu_blocks(log)
    return {
        "key": target.key, "hf_id": target.hf_id,
        "weight_dtype": target.weight_dtype, "kv_dtype": target.kv_dtype,
        "util": util, "block_size": block_size,
        "num_gpu_blocks": blocks,
        "kv_tokens": parse_kv_cache_tokens(log),
        # vLLM's own memory profiler (when present) — measured, not inferred:
        "weight_gib": parse_weight_gib(log),
        "activation_gib": parse_activation_gib(log),
        "nontorch_gib": parse_nontorch_gib(log),
        "kv_reserved_gib": parse_kv_reserved_gib(log),
        "total_gpu_memory_bytes": _total_gpu_memory_bytes(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validate.run")
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--models", help="comma-separated core keys (default: all targets)")
    ap.add_argument("--out", default="validate/results.json")
    a = ap.parse_args(argv)

    keys = a.models.split(",") if a.models else list(config.TARGETS)
    results = []
    for k in keys:
        target = config.TARGETS[k]
        print(f"[validate] loading {k} ({target.hf_id}) ...", file=sys.stderr)
        try:
            row = measure(target, a.util, a.max_model_len)
        except Exception as e:  # one model failing must not lose the others
            row = {"key": k, "hf_id": target.hf_id, "error": str(e)}
            print(f"[validate] {k} FAILED: {e}", file=sys.stderr)
        results.append(row)
        with open(a.out, "w") as f:  # checkpoint after each model
            json.dump(results, f, indent=2)
        print(f"[validate] {k}: blocks={row.get('num_gpu_blocks', row.get('error'))}", file=sys.stderr)
    print(f"[validate] wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
