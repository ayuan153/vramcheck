# docs/web.md — P4: the one-page web tool

> A single static page (`web/index.html`) that runs the **real** `vramcheck.core` in the browser
> via [Pyodide](https://pyodide.org). No server, no API, no build step. See `DESIGN.md` §7, §11.

## Why Pyodide (single source of truth)

The locked decision (DESIGN §7) is that the web tool must never compute different numbers from the
CLI. Instead of porting the memory model to JavaScript (divergence risk), the page **fetches the
actual `vramcheck/core/*.py` files and imports them in an in-browser Python runtime**. The KV /
PagedAttention / budget math is literally the same code the CLI runs.

## How it works

1. The page loads Pyodide from a CDN.
2. JS fetches the six core source files (relative URL `../vramcheck/...`) and writes them into
   Pyodide's virtual filesystem under `vramcheck/`.
3. It imports `vramcheck.core` and defines two helpers: `options()` (model/GPU lists for the
   dropdowns) and `compute(...)` (returns the header, memory breakdown, and the sweep as JSON).
4. On any input change it calls `compute(...)` and renders the breakdown + capacity table.

The dropdowns are populated **from `core.MODELS` / `core.GPUS`**, so adding a model/GPU to the
package automatically appears in the web UI.

> Note: `PKG_FILES` in `index.html` lists the core modules to load. If `core/` gains a module,
> add it there. (The page intentionally loads only `core/` — not `cli.py`/`report.py`.)

## Run locally

```sh
# from the repo root (so ../vramcheck/ resolves from web/)
python3 -m http.server 8000
# open http://localhost:8000/web/index.html
```

## Deploy on GitHub Pages (free)

Serve the repo root so both `/web/index.html` and `/vramcheck/...` are reachable:

- Settings → Pages → Build from a branch → `main` / `/ (root)`.
- The tool is then at `https://<user>.github.io/<repo>/web/index.html`.

Because the page fetches `../vramcheck/...`, the package must be served alongside the page (it is,
when Pages serves the repo root). A custom domain (`vramcheck.dev`) is a later/launch step (P5).

## Out of scope (P4)
Styling polish, sharable permalink state, prefix-cache/chunked-prefill controls, multi-GPU. The
math accuracy itself is earned in P3 (calibration), independent of this presentation layer.
