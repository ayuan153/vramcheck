"""Validation targets: our core model keys → HF repo id + vLLM launch params.

IMPORTANT: Models too big for one 80GB GPU at bf16 are validated quantized to honour the
single-GPU constraint: Llama-3.1-70B (~141 GB fp16) → AWQ-int4 (~35 GB); Qwen2.5-32B (~65 GB bf16,
no KV headroom) → official AWQ-int4 (~18 GB). HF ids + single-80GB loadability verified 2026-06-04.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    key: str            # must match a vramcheck.core.MODELS key
    hf_id: str          # Hugging Face repo id to load in vLLM
    weight_dtype: str   # dtype core uses to predict weights: fp16/bf16/fp8/int4
    kv_dtype: str = "fp16"
    vllm_kwargs: dict = field(default_factory=dict)


TARGETS: dict[str, Target] = {
    "llama-3.1-8b": Target(
        "llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "bf16",
        vllm_kwargs={"dtype": "bfloat16"}),
    "mistral-7b": Target(
        "mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "bf16",
        vllm_kwargs={"dtype": "bfloat16"}),
    # Qwen-32B bf16 (~65 GB) leaves no KV headroom on one 80GB GPU → validate the official AWQ-int4.
    "qwen2.5-32b": Target(
        "qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct-AWQ", "int4",
        vllm_kwargs={"quantization": "awq"}),
    "deepseek-v2-lite": Target(
        "deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite", "bf16",
        vllm_kwargs={"dtype": "bfloat16", "trust_remote_code": True}),
    # 70B fp16 won't fit on one 80GB GPU → validate AWQ-int4 (~35 GB). Confirm repo in infra setup.
    "llama-3.1-70b": Target(
        "llama-3.1-70b", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "int4",
        vllm_kwargs={"quantization": "awq"}),
}
