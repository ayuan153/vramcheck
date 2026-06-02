"""Thin argparse CLI over vramcheck.core. Parses args + formats; all numbers come from core."""

from __future__ import annotations

import argparse
import json
import sys

from . import core, report

DEFAULT_CONTEXTS = [4096, 8192, 16384, 32768, 131072]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vramcheck",
        description="Know exactly what you can run before you rent a GPU.",
    )
    p.add_argument("model", nargs="?", help="model key (see --list)")
    p.add_argument("gpu", nargs="?", help="GPU key (see --list)")
    p.add_argument("--ctx", type=int, help="context length")
    p.add_argument("--batch", type=int, help="concurrency; with --ctx gives a fits/OOM verdict")
    p.add_argument("--contexts", help="comma-separated context lengths for the sweep")
    p.add_argument("--weight-dtype", default="fp16")
    p.add_argument("--kv-dtype", default="fp16")
    p.add_argument("--util", type=float, help="gpu_memory_utilization (default: GPU's 0.90)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--list", action="store_true", dest="list_", help="list supported models + GPUs")
    return p


def _emit(obj: dict, as_json: bool, text: str) -> int:
    print(json.dumps(obj, indent=2) if as_json else text)
    return 0


def _err(msg: str) -> int:
    print(f"vramcheck: {msg}", file=sys.stderr)
    return 2


def _breakdown_dict(p: core.Plan) -> dict:
    return {
        "weights": p.weights_bytes,
        "activation": p.activation_bytes,
        "overhead": p.overhead_bytes,
        "kv_budget": p.kv_budget_bytes,
    }


def main(argv=None) -> int:
    a = _parser().parse_args(argv)

    if a.list_:
        models = {k: m.name for k, m in core.MODELS.items()}
        gpus = {k: g.name for k, g in core.GPUS.items()}
        text = "Models:\n" + "\n".join(f"  {k:<18} {v}" for k, v in models.items())
        text += "\n\nGPUs:\n" + "\n".join(f"  {k:<18} {v}" for k, v in gpus.items())
        return _emit({"models": models, "gpus": gpus}, a.json, text)

    if not a.model or not a.gpu:
        return _err("model and gpu are required (or use --list)")
    if a.model not in core.MODELS:
        return _err(f"unknown model '{a.model}'. Choices: {', '.join(core.MODELS)}")
    if a.gpu not in core.GPUS:
        return _err(f"unknown gpu '{a.gpu}'. Choices: {', '.join(core.GPUS)}")
    for label, dt in (("--weight-dtype", a.weight_dtype), ("--kv-dtype", a.kv_dtype)):
        if dt not in core.DTYPE_BYTES:
            return _err(f"unknown {label} '{dt}'. Choices: {', '.join(core.DTYPE_BYTES)}")
    if a.batch is not None and a.ctx is None:
        return _err("--batch requires --ctx")

    cfg, gpu = core.MODELS[a.model], core.GPUS[a.gpu]
    util = gpu.default_util if a.util is None else a.util
    kw = dict(weight_dtype=a.weight_dtype, kv_dtype=a.kv_dtype, util=a.util)

    common = {
        "model": cfg.name, "gpu": gpu.name,
        "weight_dtype": a.weight_dtype, "kv_dtype": a.kv_dtype, "util": util,
    }

    # Verdict mode: explicit context + batch.
    if a.ctx is not None and a.batch is not None:
        p = core.plan(gpu, cfg, a.ctx, **kw)
        used = a.batch * p.kv_per_seq_bytes
        ok = used <= p.kv_budget_bytes
        obj = {**common, "ctx": a.ctx, "batch": a.batch, "fits": ok,
               "kv_used_bytes": used, "breakdown_bytes": _breakdown_dict(p)}
        verdict = "FITS" if ok else "OOM"
        text = (f"{report.header(p, a.weight_dtype, util)}\n{report.breakdown_line(p)}\n\n"
                f"batch {a.batch:,} @ ctx {a.ctx:,}: {verdict} "
                f"(KV {report.gib(used):.1f} GiB of {report.gib(p.kv_budget_bytes):.1f} GiB budget)")
        return _emit(obj, a.json, text)

    # Max-batch mode: context only.
    if a.ctx is not None:
        p = core.plan(gpu, cfg, a.ctx, **kw)
        obj = {**common, "ctx": a.ctx, "max_batch": p.max_batch,
               "breakdown_bytes": _breakdown_dict(p)}
        tail = "✗ OOM (model does not fit)" if p.max_batch == 0 else f"max batch = {p.max_batch:,}"
        text = (f"{report.header(p, a.weight_dtype, util)}\n{report.breakdown_line(p)}\n\n"
                f"ctx {a.ctx:,}: {tail}")
        return _emit(obj, a.json, text)

    # Sweep mode (default).
    contexts = ([int(c) for c in a.contexts.split(",")] if a.contexts else DEFAULT_CONTEXTS)
    rows = core.sweep(gpu, cfg, contexts, **kw)
    p_ref = core.plan(gpu, cfg, contexts[0], **kw)
    obj = {**common, "sweep": [{"ctx": c, "max_batch": b} for c, b in rows],
           "breakdown_bytes": _breakdown_dict(p_ref)}
    text = (f"{report.header(p_ref, a.weight_dtype, util)}\n{report.breakdown_line(p_ref)}\n"
            f"{report.sweep_table(rows)}")
    return _emit(obj, a.json, text)
