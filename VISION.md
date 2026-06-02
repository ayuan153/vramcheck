# VISION.md

> The long-term arc. Read this first, every session. Grounds every decision below it.

## North star (one sentence)
**Know exactly what you can run before you rent a GPU** — instantly tell anyone whether
model X fits at context Y on GPU Z, and at what batch size, with a trustworthy memory
breakdown and an honest error band, without spinning up a GPU.

**North-star proof:** a *predicted vs. actual OOM* table for the 5 v0.1 models (4 GQA + 1 MLA) on
a single A100-80GB where our prediction lands within **≤10%** of the real limit (stretch **≤5%**). Accuracy
is the product. If we are not measurably accurate, we have nothing.

## Who hurts and why
"Can I run Llama-3-70B at 128K context on 2× A100?" is the single most recurring question
on r/LocalLLaMA. Today people guess with napkin math, rent a GPU at \$1-3/hr, and OOM-crash
to discover the limit. Existing answers (HF's `kv-cache-calculator` Space, assorted VRAM web
calculators, blog formulas) do basic arithmetic and ignore the things that actually move the
number. The pain is real, recurring, and currently paid for in wasted GPU-hours.

## The wedge (why we win)
Be the **instant, accurate** answer — a `pip`-installable CLI **and** a one-page hosted web
tool — that gives a definitive, model-specific verdict with a memory breakdown, no GPU
required. We win on the things competitors ignore:
- GQA / MQA / **MLA** attention variants (per-head KV math, not a fudge factor)
- **PagedAttention** block granularity and rounding waste
- **Activation** memory and **CUDA / framework** overhead
- **Prefix-cache** hit rates and **chunked prefill** effects
- An **actionable** output: max batch size at each context length + the OOM line, not a single number

## The 2-year arc

### v0.1 — now: the KV/VRAM planner (thinnest thing that proves accuracy)
CLI + tiny web tool for popular models on A100 / H100 / 4090. Outputs: max batch size at a
range of context lengths, full memory breakdown, and the OOM line. Locked scope in `DESIGN.md`.
Success = the north-star proof table.

### ~6 months: become the default "what can I run?" answer
Broaden model and GPU coverage. Model prefix-cache and chunked-prefill effects in the headline
number. Become the tool linked in every r/LocalLLaMA "what can I run?" thread.

### ~2 years: a full inference deployment planner
Not just "will it fit?" but "what *should* my serving config be?" — recommend tensor-parallel
degree, quantization, and batch configuration; integrable into capacity-planning workflows
(Terraform / K8s HPA). The default answer to "what should my serving config be?"

## Standalone-first (non-negotiable)
`vramcheck` must be fully useful with nothing else installed and must never depend on any sibling
project. Optional interop only: its predictions *can* be sanity-checked against any load
tester's real measurements, but it never *requires* one.

## What would change this vision
The vision changes only if: (a) accuracy proves unreachable within a useful error band on a
single GPU, or (b) an equally accurate, actionable, open standalone tool emerges first. Either
event must be flagged to the human and logged in `DECISIONS.md` before pivoting.

> **Naming note:** shipping name is **`vramcheck`** (locked 2026-06-01; `canirun` was taken on
> PyPI and as `canirun.ai` by competing in-niche tools). Repo: `github.com/ayuan153/canirun`.
