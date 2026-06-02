"""GPU VRAM table for the v0.1 supported set.

`memory_gib` is nominal (binary GiB). vLLM reserves `default_util` of it for
model + KV (default 0.90). Nominal vs. truly-usable bytes is a known error source
(DESIGN.md §3.9), reconciled against torch.cuda.mem_get_info() in P3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    memory_gib: float
    default_util: float = 0.90


GPUS: dict[str, GPUSpec] = {
    "a100-40gb": GPUSpec("A100-40GB", 40),
    "a100-80gb": GPUSpec("A100-80GB", 80),
    "h100-80gb": GPUSpec("H100-80GB", 80),
    "rtx-4090": GPUSpec("RTX 4090", 24),
}
