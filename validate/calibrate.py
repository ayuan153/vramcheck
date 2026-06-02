"""Calibrate the memory model from validate/results.json and print the §6 predicted-vs-measured table.

No GPU needed — operates on the JSON produced by `validate.run`. From each measured KV budget we
recover the true (activation + overhead), fit it across models, and report the resulting accuracy.

Usage: python -m validate.calibrate validate/results.json [--ctx 8192]
"""

from __future__ import annotations

import argparse
import json

from vramcheck import core

from .parse import error_pct, fit_activation_overhead, kv_capacity_tokens


def _capacity_tokens(row: dict):
    if row.get("num_gpu_blocks"):
        return kv_capacity_tokens(row["num_gpu_blocks"], row.get("block_size", 16))
    return row.get("kv_tokens")


def analyze(rows: list[dict]):
    """Returns (act_fraction, overhead_bytes, enriched_rows). Each enriched row carries the
    measured KV budget, usable bytes, weights, and the recovered nonkv (= activation + overhead)."""
    samples, enriched = [], []
    for r in rows:
        tokens = _capacity_tokens(r)
        if r.get("error") or tokens is None:
            continue
        cfg = core.MODELS[r["key"]]
        per_token = core.per_token_kv_bytes(cfg, r["kv_dtype"])
        measured_kv = tokens * per_token
        total = r.get("total_gpu_memory_bytes")
        usable = r["util"] * total if total else core.usable_bytes(core.GPUS["a100-80gb"], r["util"])
        weights = core.weight_bytes(cfg, r["weight_dtype"])
        nonkv = usable - weights - measured_kv
        samples.append((weights, nonkv))
        enriched.append({"cfg": cfg, "row": r, "measured_kv": measured_kv,
                         "usable": usable, "weights": weights, "nonkv": nonkv})
    act_fraction, overhead = fit_activation_overhead(samples)
    return act_fraction, overhead, enriched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="validate.calibrate")
    ap.add_argument("results", nargs="?", default="validate/results.json")
    ap.add_argument("--ctx", type=int, default=8192,
                    help="context length for the predicted-vs-measured max_batch comparison")
    a = ap.parse_args(argv)

    with open(a.results) as f:
        rows = json.load(f)
    act_fraction, overhead, enriched = analyze(rows)

    if not enriched:
        print("No usable measurements in results (all errored or missing block counts).")
        return 1

    print(f"Fitted core defaults:  DEFAULT_ACT_FRACTION = {act_fraction:.4f}   "
          f"DEFAULT_OVERHEAD_GIB = {overhead / core.GiB:.3f}")
    print(f"(measured GPU total reconciles GPUS table; current nominal = 80 GiB)\n")
    print(f"{'model':<24}{'measured_KV_GiB':>16}{'pred_maxb':>11}{'meas_maxb':>11}{'err%':>8}")
    for e in enriched:
        cfg, r = e["cfg"], e["row"]
        per_seq = core.kv_bytes_per_seq(cfg, a.ctx, r["kv_dtype"])
        meas_mb = int(e["measured_kv"] // per_seq)
        budget = e["usable"] - e["weights"] - act_fraction * e["weights"] - overhead
        pred_mb = max(0, int(budget // per_seq))
        print(f"{cfg.name:<24}{e['measured_kv'] / core.GiB:>16.2f}"
              f"{pred_mb:>11}{meas_mb:>11}{error_pct(pred_mb, meas_mb):>8.1f}")
    print(f"\nApply: set the two defaults in vramcheck/core/memory.py, then re-run the unit tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
