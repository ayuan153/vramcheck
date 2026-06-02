"""Validation targets: our core model keys → HF repo id + vLLM launch params.

IMPORTANT: Llama-3.1-70B at fp16 is ~141 GB and does NOT fit on one 80GB GPU. To honour the
single-GPU constraint it is validated quantized (AWQ-int4, ~35 GB). HF ids / quant below are
sensible defaults to CONFIRM during infra setup — edit freely.
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
    "qwen2.5-32b": Target(
        "qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "bf16",
        vllm_kwargs={"dtype": "bfloat16"}),
    "deepseek-v2-lite": Target(
        "deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite", "bf16",
        vllm_kwargs={"dtype": "bfloat16", "trust_remote_code": True}),
    # 70B fp16 won't fit on one 80GB GPU → validate AWQ-int4 (~35 GB). Confirm repo in infra setup.
    "llama-3.1-70b": Target(
        "llama-3.1-70b", "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4", "int4",
        vllm_kwargs={"quantization": "awq"}),
}
