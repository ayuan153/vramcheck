"""Measure the real KV budget vLLM allocates per model on ONE GPU, write results.json.

Requires a GPU + `pip install vllm`. vLLM/torch are imported lazily inside functions so this
module imports fine off-GPU (only `run`/`measure` need the GPU).

Usage (on the rented A100-80GB):
  python -m validate.run --util 0.92 --out validate/results.json
  python -m validate.run --models llama-3.1-8b,deepseek-v2-lite   # subset

Loading a model triggers vLLM's startup profiling, which fixes the KV budget (logged in v0.22 as
"GPU KV cache size: N tokens" / "Available KV cache memory: X GiB"). We read num_gpu_blocks from
the engine API, falling back to parsing the captured startup log if the attribute moved.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys

from . import config
from .parse import (
    parse_gpu_blocks, parse_kv_cache_tokens, parse_kv_reserved_gib,
    parse_weight_gib, parse_activation_gib, parse_nontorch_gib, parse_cudagraph_gib,
)


def _num_gpu_blocks(llm) -> int | None:
    """vLLM's attribute path for num_gpu_blocks has moved across versions — try the known ones."""
    candidates = [
        lambda: llm.llm_engine.vllm_config.cache_config.num_gpu_blocks,   # vLLM v0.22 (V1)
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

    # Capture vLLM logs at DEBUG — in v0.22 the per-component memory breakdown
    # ("...GiB for weight / peak activation / non-torch / CUDAGraph memory") is logged at DEBUG.
    # The KV budget ("GPU KV cache size", "Available KV cache memory") is INFO.
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    vllm_logger = logging.getLogger("vllm")
    prev_level = vllm_logger.level
    vllm_logger.setLevel(logging.DEBUG)
    for name in ("", "vllm"):
        logging.getLogger(name).addHandler(handler)
    try:
        llm = LLM(model=target.hf_id, gpu_memory_utilization=util,
                  max_model_len=max_model_len, **target.vllm_kwargs)
    finally:
        for name in ("", "vllm"):
            logging.getLogger(name).removeHandler(handler)
        vllm_logger.setLevel(prev_level)

    log = buf.getvalue()
    blocks = _num_gpu_blocks(llm)
    if blocks is None:
        blocks = parse_gpu_blocks(log)
    return {
        "key": target.key, "hf_id": target.hf_id,
        "weight_dtype": target.weight_dtype, "kv_dtype": target.kv_dtype,
        "util": util, "block_size": block_size,
        "num_gpu_blocks": blocks,
        "kv_tokens": parse_kv_cache_tokens(log),          # "GPU KV cache size: N tokens"
        # vLLM's own memory profiler (when present) — measured, not inferred:
        "kv_reserved_gib": parse_kv_reserved_gib(log),    # "Available KV cache memory: X GiB"
        "weight_gib": parse_weight_gib(log),
        "activation_gib": parse_activation_gib(log),
        "nontorch_gib": parse_nontorch_gib(log),
        "cudagraph_gib": parse_cudagraph_gib(log),
        "total_gpu_memory_bytes": _total_gpu_memory_bytes(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validate.run")
    ap.add_argument("--util", type=float, default=0.92)
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
