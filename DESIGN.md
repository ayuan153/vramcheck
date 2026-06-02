# DESIGN.md — Phase 0 output

> Architecture, memory model, locked v0.1 scope, and key decisions for **`vramcheck`** — the
> KV-cache / VRAM capacity planner (repo: `github.com/ayuan153/canirun`). Phase 0 (research +
> design) is **complete and approved (2026-06-01)**; all four open decisions are now locked (§10).
>
> North star: *know exactly what you can run before you rent a GPU* — predicted-vs-actual OOM
> within ≤10% (stretch ≤5%) for 3-4 popular models on one GPU.

---

## 1. Name check — ✅ RESOLVED: ship as `vramcheck` (2026-06-01)

`canirun` is **not viable** as shipped. Verified 2026-06-01:

| Channel | Verdict | Detail |
|---|---|---|
| PyPI `canirun` | 🚫 TAKEN | `PythonicVarun/canirun` v1.0.1 (Jan 2026) — a **directly competing** "estimate HW requirements for HF models" CLI. |
| `canirun.ai` | 🚫 TAKEN | Active product in the **exact same niche** ("Can your machine run AI models?"). |
| GitHub org `canirun` | ⚠️ TAKEN | Org claimed (low activity), plus the notable `PythonicVarun/canirun` repo. |
| `canirun.dev` / `.io` / `.app` | ❓ UNCLEAR | Not confirmed; need WHOIS. `canirun.xyz` appears registered. |
| Trademark / confusion | 🚫 HIGH RISK | PyPI name + `.ai` product both in-niche → direct user confusion. |

**The current GitHub repo `github.com/ayuan153/canirun` works for code hosting, but the
*distribution* name (PyPI package + web domain + the word we launch with) cannot be `canirun`.**
This is the one decision that touches the brand, the launch post, and the install command — so
it is flagged, not silently chosen.

**Fallbacks investigated:**

| Candidate | Spirit | PyPI | GitHub | Note |
|---|---|---|---|---|
| `llmfit` | "will this LLM fit?" | 🚫 TAKEN (same niche) | — | Rejected. |
| `willitrun` | conversational | ❓ unclear | ⚠️ taken (ML-fit CLI + org) | Risky. |
| **`vramcheck`** | "check VRAM vs model" | ✅ clear | ✅ clear | **Recommended.** Short, literal, unambiguous. |

**Decision (locked 2026-06-01):** ship as **`vramcheck`** — pip `pip install vramcheck`, CLI
`vramcheck`, target domain `vramcheck.dev`. Repo stays `github.com/ayuan153/canirun` for now
(optional cosmetic rename later). **Still TODO before first publish:** re-verify `vramcheck` on
PyPI, confirm `vramcheck.dev` is registrable, and a quick trademark pass. Docs below use `vramcheck`.

> Historical note: `canirun` was rejected because the PyPI name and `canirun.ai` are both taken by
> directly competing in-niche tools — fatal confusion risk for an adoption-first launch.

---

## 2. Competitive gap — confirmed real (2026-06-01)

No existing tool combines accurate attention-aware KV math, PagedAttention waste, activation +
overhead accounting, and an *actionable* capacity table. Feature matrix:

| Feature | HF gaunernst | apxml | Smirnov | gpu_poor | hf-vram-calc (CLI) | **vramcheck (target)** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| GQA/MQA KV reduction | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| MLA (DeepSeek) | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| PagedAttention block waste | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Prefix-cache hit rate | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (v0.2) |
| Chunked prefill | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (v0.2) |
| Activation memory (inference) | ❌ | partial | train-only | ✅ | ❌ | ✅ |
| CUDA/framework overhead | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |
| **Batch×context sweep table** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Installable CLI (pip) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Actively maintained | ❌ | ✅ | ✅ | ❌ | ⚠️ | ✅ |

**Verdict.** The only pip CLI (`hf-vram-calc`) is GQA-aware but uses a blunt `×1.2` overhead
fudge, models no PagedAttention, no MLA, no activation memory, and emits no capacity sweep. The
best web tool (apxml) is closed-source and still produces no sweep. **No tool answers: "given
80 GB, what is my max concurrency at 4k / 8k / 16k / 32k / 128k context?"** That table is our wedge.

---

## 3. The memory model (credibility core)

We model the **peak resident VRAM** of an inference server (vLLM semantics as the reference) and
ask: at a given context length, how many concurrent sequences (batch) fit before OOM?

### 3.1 Total budget
```
usable_VRAM      = gpu_total_VRAM × gpu_memory_utilization        (vLLM default 0.90)
KV_budget        = usable_VRAM − model_weights − activation_peak − overhead
max_batch(ctx)   = floor( KV_budget / kv_bytes_per_seq(ctx) )
OOM_line(batch)  = the (ctx, batch) pair where KV_bytes_per_seq × batch first exceeds KV_budget
```

### 3.2 Model weights
```
model_weights = num_params × bytes_per_param
bytes_per_param: fp16/bf16=2, fp8/int8=1, int4=0.5
```
`num_params` comes from the HF config (or a known constant for the supported set).

### 3.3 KV cache — attention-variant aware
**Standard MHA / GQA / MQA:**
```
kv_bytes_per_seq(ctx) = 2 × L × num_kv_heads × head_dim × ctx × kv_dtype_bytes
```
- `2` = K and V. `num_kv_heads` (not `num_attention_heads`) is the GQA/MQA lever.
- Llama-3-70B: 64 attn heads, **8 kv heads** → 8× smaller KV than naive MHA. MQA → 1 kv head.
- `head_dim = hidden_size / num_attention_heads` (or explicit `head_dim` in config).
- `kv_dtype_bytes`: 2 (fp16/bf16) or 1 (fp8 KV cache).

**MLA (DeepSeek-V2/V3):** caches one compressed latent + a decoupled RoPE key; **no factor of 2**:
```
kv_bytes_per_seq(ctx) = L × (kv_lora_rank + qk_rope_head_dim) × ctx × kv_dtype_bytes
```
≈ 93% smaller than the MHA equivalent. (Formula **and** accuracy validation both in v0.1/MVP,
via DeepSeek-V2-Lite — §5, §6.)

### 3.4 PagedAttention block rounding
vLLM allocates KV in fixed blocks (default **block_size = 16 tokens**). Each sequence rounds up:
```
blocks_per_seq    = ceil(ctx / block_size)
kv_bytes_per_seq  = blocks_per_seq × block_size × per_token_kv_bytes   (per-token form of §3.3)
waste_per_seq     = (blocks_per_seq × block_size − ctx) × per_token_kv_bytes   ( ≤ 15 tokens )
```
Small per sequence, but at high concurrency it is real (e.g. 256 seqs × 15 tokens adds up). We
report it explicitly rather than ignoring it.

### 3.5 Activation peak — the biggest unknown
- **Decode** (1 token/step): small, ~O(batch × hidden).
- **Prefill** (prompt processing): peak ≈ `prefill_tokens × hidden × c`. Without chunked prefill
  a long prompt spikes this by several GB. With chunked prefill it is bounded by `chunk_size`.
- v0.1 estimate: `activation_peak ≈ k × model_weights` with `k ≈ 0.05–0.15`, **calibrated** from
  the validation runs (§6) rather than guessed. Reported as a range with an `APPROXIMATE` flag.

### 3.6 CUDA / framework overhead
Fixed cost not in the naive formula: CUDA context, cuBLAS workspace, allocator, CUDA graphs,
(multi-GPU) NCCL buffers.
- Single GPU: **~0.5–1.5 GB**. Multi-GPU (TP): **~1–3 GB per GPU** (out of v0.1 scope).
- v0.1 uses a calibrated single-GPU constant, flagged `APPROXIMATE`.

### 3.7 Prefix cache (v0.2)
Effective new KV per request = `(1 − hit_rate) × full_kv`. A shared system prompt at high hit
rate frees KV for more concurrency. Modeled in v0.2.

### 3.8 Where configs come from
Primary: HF `config.json` via `huggingface_hub` (`num_hidden_layers`, `num_key_value_heads`,
`num_attention_heads`, `hidden_size`, `head_dim`, `vocab_size`, MLA fields `kv_lora_rank`,
`qk_rope_head_dim`). Many top models (Llama) are **gated** and need a token. To keep the tool
zero-friction and offline-capable, v0.1 **bundles a tiny vendored snapshot** of *just the numeric
config fields* (no weights) for the supported set; live HF fetch is the fallback for arbitrary models.

### 3.9 Known error sources (ranked) — honesty caveat
1. **Prefill activation memory** — biggest unknown; long prompts without chunked prefill spike it.
2. **CUDA / framework overhead** — config- and version-dependent; CUDA graphs add 1–2 GB.
3. **PagedAttention rounding waste** — bounded but real at high concurrency.
4. **KV dtype surprises** — accidental fp32 KV doubles/quadruples the cache.
5. **TP / NCCL buffers** — out of v0.1 scope; will matter for multi-GPU.

We treat "within ~5% of actual OOM" as the **core technical risk**, not a footnote. Every output
carries a breakdown and an honest band.

---

## 4. Ground-truth / validation methodology (cheap, 1 GPU)

The key insight: **we don't need to OOM-crash to get ground truth.** vLLM profiles the model at
startup and **logs the exact KV budget it found** as `# GPU blocks: N`. That gives a precise,
crash-free measurement of `usable_VRAM − weights − activation_peak − overhead`.

**Procedure (per model, on one rented GPU):**
1. Start vLLM with a known model + GPU + `gpu_memory_utilization=0.90`.
2. Read the logged `# GPU blocks: N`. Then:
   `measured_KV_budget = N × block_size × per_token_kv_bytes`.
3. Compare to our predicted `KV_budget`. The residual calibrates `activation_peak` + `overhead`
   (§3.5–3.6) — directly attacking error sources #1 and #2.
4. Cross-check the **OOM line**: binary-search `--max-num-seqs` / context until vLLM refuses or
   OOMs; confirm it matches our predicted `max_batch(ctx)` within band.
5. Repeat at 2-3 context lengths per model.

Cost: a few GPU-hours on a spot/community instance (one A100-80GB or one 4090). No sibling tool
required; an external load tester *may* corroborate but is never needed.

---

## 5. Locked v0.1 scope

**In scope:**
- **Models (5, attention-diverse, all popular on r/LocalLLaMA):**
  - Llama-3.1-8B-Instruct (GQA, 8 kv heads)
  - Llama-3.1-70B-Instruct (GQA, 8 kv heads)
  - Mistral-7B-Instruct (GQA)
  - Qwen2.5-32B-Instruct (GQA)
  - DeepSeek-V2-Lite (MLA) — small (15.7B, ~31 GB fp16), fits one A100-80GB, so MLA accuracy is
    **validated in the MVP** (per "accuracy is part of MVP"), not deferred.
- **GPUs:** A100-40GB, A100-80GB, H100-80GB, RTX 4090-24GB. (Validation runs on whichever single
  GPU we can rent; other GPUs are pure arithmetic on the validated per-token + overhead model.)
- **Precision:** weights fp16/bf16/fp8/int4; KV cache fp16 or fp8.
- **Outputs:**
  1. **Verdict:** fits / doesn't fit for a given (model, ctx, batch, GPU).
  2. **Memory breakdown:** weights / KV / activation / overhead / PagedAttention waste (GB + %).
  3. **Max batch @ context** and the **OOM line**.
  4. **Capacity sweep table:** rows = context lengths, cols = GPUs (or batch), cells = max
     concurrent sequences. This table is the headline artifact.
- **Surfaces:** `pip`-installable CLI (single source of truth) **and** a one-page hosted web tool.
- **Engine semantics:** vLLM (PagedAttention, `gpu_memory_utilization=0.90`, block_size=16).

**Explicitly OUT of v0.1:**
- Multi-GPU / tensor & pipeline parallelism (TP/PP) and NCCL buffer modeling.
- Prefix-cache and chunked-prefill *modeling in the headline number* (v0.2).
- Training / fine-tuning / LoRA memory.
- Throughput, latency, tokens/sec predictions (that's the v2 "deployment planner").
- llama.cpp / GGUF / TGI / TensorRT-LLM engine-specific accounting.
- Arbitrary/unknown models in the *validated* set (live HF fetch works, but only the 4 above
  carry an accuracy guarantee).

---

## 6. Validation plan & "good enough to ship" (north star, in numbers)

**Deliverable:** a *predicted vs. actual* table, produced on one GPU.

| Model | GPU | Context | Predicted max batch | Measured max batch | Error % |
|---|---|---|---|---|---|
| Llama-3.1-8B | A100-80GB | 8k / 32k | … | from vLLM `# GPU blocks` + OOM search | … |
| Llama-3.1-70B | A100-80GB | 8k / 32k | … | … | … |
| Mistral-7B | A100-80GB | 8k / 32k | … | … | … |
| Qwen2.5-32B | A100-80GB | 8k / 32k | … | … | … |
| DeepSeek-V2-Lite (MLA) | A100-80GB | 8k / 32k | … | … | … |

**Ship gate (numbers):**
- **MUST:** ≤ **10%** error on `max_batch(ctx)` for all 5 models at the tested contexts, on the
  validated GPU; and KV-budget prediction within ≤ **10%** of the logged `# GPU blocks` budget.
- **STRETCH:** ≤ **5%** on both.
- If we cannot hit ≤10% after calibration, we **do not ship a confident number** — we ship the
  breakdown with the honest band and flag it (per VISION "what would change this vision").

---

## 7. Architecture

```
vramcheck/
├─ core/                      # pure-Python memory model — the single source of truth
│  ├─ models.py               # supported-model configs (vendored snapshot) + HF fetch fallback
│  ├─ gpus.py                 # GPU VRAM table (nominal + usable + util default)
│  ├─ kv.py                   # MHA/GQA/MQA + MLA KV math, PagedAttention rounding
│  ├─ memory.py               # weights + activation + overhead + KV → budget, max_batch, OOM line
│  └─ report.py               # breakdown + sweep table data structures
├─ cli/                       # thin CLI over core (Typer + Rich tables)
└─ web/                       # one-page tool; runs core in-browser via Pyodide (single source of truth)
```

**Decisions (minor, recorded in DECISIONS.md):**
- **Language: Python.** Audience is Python-native; lowest adoption friction; reuses `huggingface_hub`.
- **CLI:** Typer + Rich (clean tables/colors for the screenshot that sells the launch).
- **Web (single source of truth):** static one-page site that loads the *same* Python core via
  **Pyodide**, hosted free on GitHub Pages — no server, no API keys, and the web number can never
  drift from the CLI number. (Alternative: port math to JS — rejected, risks divergence; or a tiny
  serverless function — rejected, adds hosting + cost.) **Locked 2026-06-01.**
- **Packaging:** `pyproject.toml`, PEP 621, console-script entry point.
- **Offline-first model configs:** vendored numeric snapshot avoids gated-model token friction.

---

## 8. Stretch note — from calculator to deployment planner (sketch, do NOT build)

The arithmetic that answers "will it fit?" inverts into "what *should* I run?":
- **Tensor-parallel degree:** smallest TP that fits the target (ctx, batch) within per-GPU VRAM,
  net of NCCL buffers — search over TP ∈ {1,2,4,8}.
- **Quantization recommendation:** cheapest precision (int4 < fp8 < fp16) that meets a fit/quality
  constraint; surface the capacity won per precision step.
- **Batch config:** recommend `max_num_seqs` / `max_num_batched_tokens` given an SLA and the
  prefix-cache/chunked-prefill model.
- **Workflow integration:** emit machine-readable output (JSON) → drop into Terraform / K8s HPA
  capacity planning. The CLI becomes a planning primitive, not just an answer box.

This is the v2 north star; v0.1 deliberately ships none of it.

---

## 9. Broadcast draft (for launch, not now)

**r/LocalLLaMA post**
> **Title:** I built a tool that tells you *exactly* what you can run before you rent a GPU
>
> Tired of renting an A100, loading Llama-70B, and OOM-crashing to find the context limit? I
> built a free CLI + web tool that gives a model-specific answer with a full memory breakdown —
> no GPU needed. Unlike the existing calculators it actually models GQA/MQA/MLA, PagedAttention
> block waste, activation + CUDA overhead — not just `params × 2`. Output is an *actionable* table:
> max concurrent sequences at 4k/8k/16k/32k/128k for your GPU, and the exact OOM line. Validated
> against real vLLM runs to within ~X% (breakdown + error band shown — I'm honest about what's hard).
> `pip install vramcheck` or paste a model + GPU at vramcheck.dev. Feedback welcome.

**Show HN title**
> Show HN: vramcheck — know exactly what you can run before you rent a GPU

**Lead screenshot (the thing that sells it):** the CLI printing a clean Rich table —
```
Llama-3.1-70B (fp16) on A100-80GB · vLLM · util=0.90
┌──────────┬──────────────┬───────────────┐
│ Context  │ Max batch    │ KV / Weights / Act / OH │
├──────────┼──────────────┼───────────────┤
│   4,096  │      42      │  ... breakdown bars ... │
│   8,192  │      19      │                         │
│  32,768  │       4      │                         │
│ 128,000  │   ✗ OOM      │  weights alone = 140 GB │
└──────────┴──────────────┴───────────────┘
verdict: 70B fits to ~32k @ batch 4 on one A100-80GB; 128k needs ≥2 GPUs.
```

---

## 10. Decisions — RESOLVED (2026-06-01)
1. **Name:** ✅ `vramcheck` (pip / CLI / domain). Re-verify PyPI + domain + trademark before publish. (§1)
2. **Web approach:** ✅ Pyodide single-source-of-truth on GitHub Pages (no JS port, no server). (§7)
3. **Validation infra:** ✅ rent an on-demand **A100-80GB** cloud container (RunPod / Lambda Labs);
   **no hardware purchase**; ~$10-20 total. An HF token is needed at validation time to download
   gated Llama weights.
4. **MLA / accuracy:** ✅ accuracy validation is a **mandatory MVP gate** (we do not launch on
   unvalidated numbers). MLA is validated in the MVP via DeepSeek-V2-Lite on the same A100-80GB.

## 11. Engineering phases (v0.1 / MVP) — granular

Validation (P3) is a hard MVP gate; P1→P4 = MVP, P5 = launch.

| Phase | Deliverable | Done-gate |
|---|---|---|
| **P1 — core lib** | `core/` (kv, memory, gpus, models, report) + vendored config snapshot | unit tests pass vs hand-computed values (e.g. Llama-3.1-70B @8k = 2.68 GB KV) |
| **P2 — CLI** | Typer+Rich CLI: verdict, breakdown, max-batch@ctx, OOM line, sweep table, `--json` | renders the §9 launch table |
| **P3 — validation (MVP accuracy gate)** | rent A100-80GB; run vLLM; capture `# GPU blocks` + OOM search; fill §6 table for all 5 models; calibrate activation `k` + overhead constant | ≤10% error (stretch ≤5%) on all 5 models |
| **P4 — web** | one-page Pyodide site on GitHub Pages running the same `core/` | web numbers match CLI on spot checks |
| **P5 — launch** | publish `vramcheck` to PyPI; README; broadcast (§9) | clean-env install works; post live |
