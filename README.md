# vramcheck

**Know exactly what you can run before you rent a GPU.**

An accurate KV-cache / VRAM capacity planner for local LLM serving: given a model, a
context length, and a GPU, it tells you the max batch size that fits, the memory
breakdown, and the OOM line — no GPU required.

Unlike basic VRAM calculators, `vramcheck` models what actually moves the number:
GQA/MQA/MLA attention variants, PagedAttention block rounding, activation and CUDA
overhead — not just `params × 2`.

> Status: **early development (v0.1).** See [`DESIGN.md`](DESIGN.md) for the locked scope,
> the memory model, and the validation plan, and [`VISION.md`](VISION.md) for the arc.

## v0.1 supported set
- Models: Llama-3.1-8B/70B, Mistral-7B, Qwen2.5-32B, DeepSeek-V2-Lite (MLA)
- GPUs: A100-40/80GB, H100-80GB, RTX 4090

Accuracy is the product: predictions are validated against real vLLM runs to within ≤10%
(stretch ≤5%) before launch.
