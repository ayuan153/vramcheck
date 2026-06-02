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
    re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    re.compile(r"Maximum concurrency.*?for\s*([\d,]+)\s*tokens", re.IGNORECASE),
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
