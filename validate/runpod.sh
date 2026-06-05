#!/usr/bin/env bash
# validate/runpod.sh — one-command P3 validation runner for a single-GPU box
# (RunPod / Lambda / any CUDA machine). Plug-and-play: provide a Hugging Face token,
# then run. It installs vLLM, measures the real KV budget for the v0.1 models, and
# calibrates the memory model.
#
# SAFETY: this runs the job on whatever box you are ALREADY on. It does NOT provision
# or terminate any cloud resource — so it can never leave a GPU billing in the background.
# You rent/stop the pod yourself.
#
# ── Usage (on the rented GPU pod) ────────────────────────────────────────────
#   # 1) drop your HF token in (either way):
#   export HF_TOKEN=hf_xxx                      #   ...or:
#   echo hf_xxx > validate/.hf_token            #   (gitignored; never committed)
#   # 2) run:
#   bash validate/runpod.sh                     # from inside the repo
#
#   # Bootstrap from a bare pod (clones the repo first):
#   curl -fsSL https://raw.githubusercontent.com/ayuan153/canirun/main/validate/runpod.sh \
#     | HF_TOKEN=hf_xxx bash
#
# ── Optional env knobs (all have sane defaults) ──────────────────────────────
#   HF_HOME       weight cache dir            (default /workspace/hf — the big volume)
#   UTIL          gpu_memory_utilization      (default 0.92, vLLM v0.22 default)
#   MODELS        comma-list subset for full  (default: all 5 targets)
#   SMOKE_MODEL   cheap pre-flight model      (default llama-3.1-8b — also tests the token)
#   SKIP_SMOKE=1  skip the pre-flight gate
#   SKIP_INSTALL=1  skip `pip install vllm` (already installed)
#   REPO_URL      git URL to clone if not in the repo
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/ayuan153/canirun.git}"
HF_HOME="${HF_HOME:-/workspace/hf}"
UTIL="${UTIL:-0.92}"
SMOKE_MODEL="${SMOKE_MODEL:-llama-3.1-8b}"
export HF_HOME

log() { printf '\n\033[1m[runpod.sh] %s\033[0m\n' "$*"; }

# ── locate or fetch the repo ─────────────────────────────────────────────────
if [ -f "validate/run.py" ]; then
  REPO_DIR="$(pwd)"
elif [ -f "/workspace/canirun/validate/run.py" ]; then
  REPO_DIR="/workspace/canirun"
else
  REPO_DIR="${WORKDIR:-/workspace}/canirun"
  log "cloning $REPO_URL -> $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true   # best-effort refresh; ignore on shallow/detached
log "repo: $REPO_DIR"

# ── HF token (the only "key" you need) ───────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -f validate/.hf_token ]; then
  HF_TOKEN="$(tr -d '[:space:]' < validate/.hf_token)"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  log "HF token loaded."
else
  log "WARNING: no HF token (env HF_TOKEN or validate/.hf_token). Gated Llama models will 401."
fi

# ── sanity: are we actually on a GPU box? ────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  log "WARNING: nvidia-smi not found — runpod.sh is meant to run on a CUDA GPU box."
fi

# ── install vLLM (skip if already importable) ────────────────────────────────
if [ "${SKIP_INSTALL:-0}" != "1" ] && ! python -c "import vllm" 2>/dev/null; then
  log "installing vLLM (pulls a matching torch/CUDA) ..."
  python -m pip install -q --upgrade pip
  python -m pip install -q vllm
fi
python -c "import vllm; print('vLLM', vllm.__version__)" || { log "vLLM import failed"; exit 1; }

# ── smoke test: smallest gated model only (cheap; catches token / load / log-format breaks) ──
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  log "SMOKE: measuring '$SMOKE_MODEL' only — pre-flight before the large downloads ..."
  python -m validate.run --models "$SMOKE_MODEL" --util "$UTIL" --out validate/results-smoke.json
  if ! python -m validate.calibrate validate/results-smoke.json; then
    log "SMOKE FAILED — KV budget unreadable (HF token, model load, or vLLM log-format drift)."
    log "Aborting BEFORE the ~100GB downloads. Inspect validate/results-smoke.json."
    exit 1
  fi
  log "SMOKE OK."
fi

# ── full run + calibration ───────────────────────────────────────────────────
log "FULL: measuring all targets (first run downloads ~100GB of weights) ..."
if [ -n "${MODELS:-}" ]; then
  python -m validate.run --util "$UTIL" --models "$MODELS" --out validate/results.json
else
  python -m validate.run --util "$UTIL" --out validate/results.json
fi

log "CALIBRATE:"
python -m validate.calibrate validate/results.json --ctx 8192 | tee validate/calibration-report.txt

log "DONE."
log "Send back:  validate/results.json  +  validate/calibration-report.txt"
log "Then DELETE the pod (not just stop) to end billing."
