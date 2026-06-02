"""P3 validation harness — measures real vLLM KV budget and calibrates the memory model.

GPU/vLLM code lives only in `run.py` (lazy imports). `parse.py`, `config.py`, and the
calibration math in `calibrate.py` import nothing GPU-specific and are unit-tested off-GPU.
"""
