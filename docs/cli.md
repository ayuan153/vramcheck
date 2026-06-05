# docs/cli.md — P2: the `vramcheck` CLI

> Design for the command-line surface over `core/`. The CLI is a **thin presenter**: all numbers
> come from `vramcheck.core`; the CLI only parses args and formats output. See `DESIGN.md` §7, §11.

## Dependency choice — revised from the locked "Typer + Rich"

**Decision: stdlib `argparse` + a plain-text table. Zero runtime dependencies.** Rich/Typer are
deferred to launch polish (P5).

Why the change (logged in `DECISIONS.md`):
- **Adoption is the #1 goal** → `pip install vramcheck` pulling **zero** transitive deps is the
  lowest-friction install possible, and the CLI then runs/tests in any environment (incl. this one).
- The product is the *accurate number and the table*, not box-drawing characters. A clean aligned
  ASCII table reads fine in a terminal and in a Reddit code block.
- Rich is purely cosmetic for the launch screenshot; it can be added later as an optional extra
  (`vramcheck[pretty]`) without changing the core or the CLI contract.

This is a reversible, presentation-only deviation; the commands and outputs below are unchanged.

## Invocation

```
vramcheck <model> <gpu> [options]      # also: python3 -m vramcheck ...
vramcheck --list                       # list supported models + GPUs
```

`<model>` and `<gpu>` are keys from `core.MODELS` / `core.GPUS` (e.g. `llama-3.1-70b`, `a100-80gb`).
Unknown keys print the valid set and exit non-zero.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--ctx N` | (none) | Context length. Without it, the default sweep runs. |
| `--batch N` | (none) | Concurrency. With `--ctx`, produces a fits/doesn't verdict. |
| `--contexts a,b,c` | `4096,8192,16384,32768,131072` | Context lengths for the sweep table. |
| `--weight-dtype` | `fp16` | `fp16\|bf16\|fp8\|int4` — weight precision. |
| `--kv-dtype` | `fp16` | `fp16\|bf16\|fp8` — KV-cache precision. |
| `--util F` | GPU default (0.92) | `gpu_memory_utilization`. |
| `--json` | off | Machine-readable output (for scripting / the v2 workflow integration). |

## Output modes (precedence)

1. `--list` → two short lists of keys + display names.
2. `--ctx` **and** `--batch` → **verdict**: `FITS`/`OOM` + memory breakdown for that exact config.
3. `--ctx` only → **max batch** at that context + breakdown.
4. neither → **capacity sweep table**: rows = contexts, col = max concurrent sequences, + breakdown.

Every non-list mode prints a header (`<Model> (<weight-dtype>) on <GPU> · util=<u>`) and a memory
breakdown (weights / activation / overhead / KV budget, in GiB). Activation + overhead are flagged
`~approx (P3)` until calibrated.

`--json` emits the same data as a single JSON object (header fields, breakdown bytes, and either a
`verdict`, a `max_batch`, or a `sweep` list).

## Example (sweep)

```
$ vramcheck llama-3.1-8b a100-80gb
Llama-3.1-8B-Instruct (fp16) on A100-80GB · util=0.92
weights 15.0 GiB · activation ~1.5 GiB · overhead ~1.0 GiB · KV budget 56.1 GiB   (~approx: activation, overhead — P3)

  context    max batch
    4,096          112
    8,192           56
   16,384           28
   32,768           14
  131,072            3
```

## Out of scope for P2
Rich/color output, prefix-cache/chunked-prefill flags, multi-GPU/TP, live HF model fetch. These
are later phases (see `DESIGN.md` §5 OUT and §11).
