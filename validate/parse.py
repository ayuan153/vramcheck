"""Pure helpers for P3 validation — no vllm/torch imports, so they unit-test off-GPU.

Covers: parsing vLLM's KV-budget log lines, KV-capacity math, error %, and a 2-parameter
least-squares fit of (activation + overhead) against model weight size.
"""

from __future__ import annotations

import re
from typing import Optional

# vLLM has reported its KV budget under a few wordings across versions.
_BLOCK_PATTERNS = [
    re.compile(r"#\s*GPU blocks:\s*([\d,]+)"),
    re.compile(r"GPU blocks:\s*([\d,]+)"),
]
_TOKEN_PATTERNS = [
    # vLLM V1 (v0.22): "GPU KV cache size: 12,345 tokens" = total KV capacity in tokens.
    re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
]


def _first_int(patterns, text: str) -> Optional[int]:
    for p in patterns:
        m = p.search(text)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def parse_gpu_blocks(text: str) -> Optional[int]:
    """Extract `# GPU blocks: N` from a captured vLLM startup log."""
    return _first_int(_BLOCK_PATTERNS, text)


def parse_kv_cache_tokens(text: str) -> Optional[int]:
    """Extract a 'GPU KV cache size: N tokens' style line (newer vLLM)."""
    return _first_int(_TOKEN_PATTERNS, text)


def kv_capacity_tokens(num_gpu_blocks: int, block_size: int = 16) -> int:
    return num_gpu_blocks * block_size


# vLLM's startup "Memory profiling results" line reports these directly (wording varies by
# version, so the regexes are tolerant). Each returns GiB or None when absent.
_F = r"([\d.]+)"
# Cover both the older "Memory profiling results" wording and vLLM v0.22 (V1 engine), which logs
# e.g. "...14.96 GiB for weight, 1.23 GiB for peak activation, 0.45 GiB for non-torch memory, and
# 2.10 GiB for CUDAGraph memory" (DEBUG) and "Available KV cache memory: X GiB" (INFO).
_WEIGHT_PATTERNS = [
    re.compile(r"model weights take\s*" + _F + r"\s*GiB", re.IGNORECASE),
    re.compile(_F + r"\s*GiB for weight", re.IGNORECASE),
]
_ACT_PATTERNS = [
    re.compile(r"activation peak memory takes\s*" + _F + r"\s*GiB", re.IGNORECASE),
    re.compile(_F + r"\s*GiB for peak activation", re.IGNORECASE),
]
_NONTORCH_PATTERNS = [
    re.compile(r"non[_-]?torch[\w ]{0,16}?takes\s*" + _F + r"\s*GiB", re.IGNORECASE),
    re.compile(_F + r"\s*GiB for non-torch memory", re.IGNORECASE),
]
_CUDAGRAPH_PATTERNS = [
    re.compile(_F + r"\s*GiB for CUDAGraph memory", re.IGNORECASE),
]
_KVRES_PATTERNS = [
    re.compile(r"Available KV cache memory:\s*" + _F + r"\s*GiB", re.IGNORECASE),  # v0.22 (INFO)
    re.compile(r"reserved for KV Cache is\s*" + _F + r"\s*GiB", re.IGNORECASE),     # older
]


def _first_float(patterns, text: str) -> Optional[float]:
    for p in patterns:
        m = p.search(text)
        if m:
            return float(m.group(1))
    return None


def parse_weight_gib(text: str) -> Optional[float]:
    return _first_float(_WEIGHT_PATTERNS, text)


def parse_activation_gib(text: str) -> Optional[float]:
    return _first_float(_ACT_PATTERNS, text)


def parse_nontorch_gib(text: str) -> Optional[float]:
    return _first_float(_NONTORCH_PATTERNS, text)


def parse_cudagraph_gib(text: str) -> Optional[float]:
    return _first_float(_CUDAGRAPH_PATTERNS, text)


def parse_kv_reserved_gib(text: str) -> Optional[float]:
    """KV budget in GiB — vLLM v0.22 'Available KV cache memory: X GiB' (or older wording)."""
    return _first_float(_KVRES_PATTERNS, text)


def error_pct(predicted: float, measured: float) -> float:
    return abs(predicted - measured) / measured * 100.0 if measured else float("inf")


def fit_activation_overhead(samples: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares fit of nonkv ≈ act_fraction * weight_bytes + overhead_bytes.

    `samples` = [(weight_bytes, nonkv_bytes), ...]. Returns (act_fraction, overhead_bytes).
    With <2 distinct weight values the slope is unidentifiable → returns (0.0, mean nonkv).
    """
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    xs = [w for w, _ in samples]
    ys = [y for _, y in samples]
    if len(set(xs)) < 2:
        return 0.0, sum(ys) / n
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(w * y for w, y in samples)
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return slope, intercept
