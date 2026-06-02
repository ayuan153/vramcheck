"""Vendored configs for the v0.1 supported model set.

Only the numeric fields the memory model needs (no weights). KV-cache math depends
on layers / heads / head_dim / dtype, which are exact here. Parameter counts are
canonical/approximate and will be reconciled against HF config.json + safetensors
during P3 validation (DESIGN.md §3.8, §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_params: int
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention: str = "gqa"  # "mha" | "gqa" | "mqa" | "mla"
    # MLA-only (DeepSeek): KV = L * (kv_lora_rank + qk_rope_head_dim) * dtype
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None


MODELS: dict[str, ModelConfig] = {
    "llama-3.1-8b": ModelConfig(
        "Llama-3.1-8B-Instruct", 8_030_000_000, 32, 4096, 32, 8, 128, "gqa"),
    "llama-3.1-70b": ModelConfig(
        "Llama-3.1-70B-Instruct", 70_600_000_000, 80, 8192, 64, 8, 128, "gqa"),
    "mistral-7b": ModelConfig(
        "Mistral-7B-Instruct-v0.3", 7_248_000_000, 32, 4096, 32, 8, 128, "gqa"),
    "qwen2.5-32b": ModelConfig(
        "Qwen2.5-32B-Instruct", 32_500_000_000, 64, 5120, 40, 8, 128, "gqa"),
    # MLA: num_key_value_heads is unused for the KV formula; kv_lora_rank + rope drive it.
    "deepseek-v2-lite": ModelConfig(
        "DeepSeek-V2-Lite", 15_700_000_000, 27, 2048, 16, 16, 128, "mla",
        kv_lora_rank=512, qk_rope_head_dim=64),
}
