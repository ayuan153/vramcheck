"""KV-cache size math: MHA/GQA/MQA + MLA, with PagedAttention block rounding.

Reference engine: vLLM. Standard KV per token = 2 * L * num_kv_heads * head_dim * dtype
(GQA/MQA fall out of num_kv_heads). MLA caches one compressed latent + a decoupled
RoPE key, so KV per token = L * (kv_lora_rank + qk_rope_head_dim) * dtype — no factor
of 2. See DESIGN.md §3.3-3.4.
"""

from __future__ import annotations

import math

from .models import ModelConfig

# Bytes per cached element, by dtype.
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}

# vLLM PagedAttention default block size (tokens per block).
DEFAULT_BLOCK_SIZE = 16


def per_token_kv_bytes(cfg: ModelConfig, kv_dtype: str = "fp16") -> float:
    """KV-cache bytes for ONE token, summed across all layers."""
    b = DTYPE_BYTES[kv_dtype]
    if cfg.attention == "mla":
        if cfg.kv_lora_rank is None or cfg.qk_rope_head_dim is None:
            raise ValueError(f"{cfg.name}: MLA requires kv_lora_rank + qk_rope_head_dim")
        return cfg.num_hidden_layers * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * b
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * b


def padded_tokens(ctx: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Tokens rounded up to the next PagedAttention block boundary."""
    return math.ceil(ctx / block_size) * block_size


def kv_bytes_per_seq(
    cfg: ModelConfig, ctx: int, kv_dtype: str = "fp16", block_size: int = DEFAULT_BLOCK_SIZE
) -> float:
    """KV-cache bytes for one sequence of length `ctx`, including block-rounding waste."""
    return padded_tokens(ctx, block_size) * per_token_kv_bytes(cfg, kv_dtype)


def kv_waste_bytes(
    cfg: ModelConfig, ctx: int, kv_dtype: str = "fp16", block_size: int = DEFAULT_BLOCK_SIZE
) -> float:
    """Bytes wasted per sequence by rounding `ctx` up to a block boundary."""
    return (padded_tokens(ctx, block_size) - ctx) * per_token_kv_bytes(cfg, kv_dtype)
