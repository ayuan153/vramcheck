"""Total VRAM budget and max-batch math.

Total resident VRAM = weights + KV cache + activation peak + CUDA/framework overhead
(DESIGN.md §3.1). `activation` and `overhead` are APPROXIMATE in v0.1 and get
calibrated against real vLLM runs in P3 (the MVP accuracy gate).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .gpus import GPUSpec
from .kv import DTYPE_BYTES, DEFAULT_BLOCK_SIZE, kv_bytes_per_seq
from .models import ModelConfig

GiB = 1024 ** 3

# APPROXIMATE defaults — calibrated in P3.
DEFAULT_ACT_FRACTION = 0.10  # activation peak ≈ 10% of weights
DEFAULT_OVERHEAD_GIB = 1.0   # CUDA/framework fixed cost, single GPU


def weight_bytes(cfg: ModelConfig, weight_dtype: str = "fp16") -> float:
    return cfg.num_params * DTYPE_BYTES[weight_dtype]


def usable_bytes(gpu: GPUSpec, util: float | None = None) -> float:
    """VRAM vLLM makes available for weights + KV (= util × total)."""
    return (gpu.default_util if util is None else util) * gpu.memory_gib * GiB


@dataclass(frozen=True)
class Plan:
    gpu: str
    model: str
    ctx: int
    weights_bytes: float
    activation_bytes: float
    overhead_bytes: float
    kv_budget_bytes: float
    kv_per_seq_bytes: float
    max_batch: int


def plan(
    gpu: GPUSpec,
    cfg: ModelConfig,
    ctx: int,
    *,
    weight_dtype: str = "fp16",
    kv_dtype: str = "fp16",
    util: float | None = None,
    act_fraction: float = DEFAULT_ACT_FRACTION,
    overhead_gib: float = DEFAULT_OVERHEAD_GIB,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Plan:
    w = weight_bytes(cfg, weight_dtype)
    act = act_fraction * w
    oh = overhead_gib * GiB
    budget = usable_bytes(gpu, util) - w - act - oh
    per_seq = kv_bytes_per_seq(cfg, ctx, kv_dtype, block_size)
    mb = max(0, math.floor(budget / per_seq)) if budget > 0 else 0
    return Plan(gpu.name, cfg.name, ctx, w, act, oh, budget, per_seq, mb)


def max_batch(gpu: GPUSpec, cfg: ModelConfig, ctx: int, **kw) -> int:
    """Largest number of concurrent sequences of length `ctx` that fit."""
    return plan(gpu, cfg, ctx, **kw).max_batch


def fits(gpu: GPUSpec, cfg: ModelConfig, ctx: int, batch: int, **kw) -> bool:
    p = plan(gpu, cfg, ctx, **kw)
    return batch * p.kv_per_seq_bytes <= p.kv_budget_bytes


def sweep(gpu: GPUSpec, cfg: ModelConfig, contexts, **kw) -> list[tuple[int, int]]:
    """(context_length, max_batch) pairs — the headline capacity table."""
    return [(c, max_batch(gpu, cfg, c, **kw)) for c in contexts]
