"""Plain-text formatting for the CLI. Pure presentation — no memory math lives here."""

from __future__ import annotations

from .core import GiB, Plan


def gib(nbytes: float) -> float:
    return nbytes / GiB


def header(p: Plan, weight_dtype: str, util: float) -> str:
    return f"{p.model} ({weight_dtype}) on {p.gpu} · util={util:.2f}"


def breakdown_line(p: Plan) -> str:
    return (
        f"weights {gib(p.weights_bytes):.1f} GiB · "
        f"activation ~{gib(p.activation_bytes):.1f} GiB · "
        f"overhead ~{gib(p.overhead_bytes):.1f} GiB · "
        f"KV budget {gib(p.kv_budget_bytes):.1f} GiB   "
        f"(~approx: activation, overhead — P3)"
    )


def sweep_table(rows: list[tuple[int, int]]) -> str:
    lines = ["", f"  {'context':>9}    {'max batch':>9}"]
    for ctx, batch in rows:
        cell = "OOM" if batch == 0 else f"{batch:,}"
        lines.append(f"  {ctx:>9,}    {cell:>9}")
    return "\n".join(lines)
