"""Pure-Python memory model — the single source of truth (CLI and web both call this)."""

from .models import MODELS, ModelConfig
from .gpus import GPUS, GPUSpec
from .kv import (
    DTYPE_BYTES,
    DEFAULT_BLOCK_SIZE,
    per_token_kv_bytes,
    padded_tokens,
    kv_bytes_per_seq,
    kv_waste_bytes,
)
from .memory import (
    GiB,
    Plan,
    weight_bytes,
    usable_bytes,
    plan,
    max_batch,
    fits,
    sweep,
)

__all__ = [
    "MODELS", "ModelConfig", "GPUS", "GPUSpec",
    "DTYPE_BYTES", "DEFAULT_BLOCK_SIZE", "per_token_kv_bytes", "padded_tokens",
    "kv_bytes_per_seq", "kv_waste_bytes",
    "GiB", "Plan", "weight_bytes", "usable_bytes", "plan", "max_batch", "fits", "sweep",
]
