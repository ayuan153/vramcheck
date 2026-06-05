"""Off-GPU tests for the P3 validation harness (no vllm/torch needed)."""

import unittest

from validate import calibrate
from validate.parse import (
    parse_gpu_blocks, parse_kv_cache_tokens, kv_capacity_tokens,
    error_pct, fit_activation_overhead,
    parse_weight_gib, parse_activation_gib, parse_nontorch_gib, parse_kv_reserved_gib,
    parse_cudagraph_gib,
)
from vramcheck import core


class TestParse(unittest.TestCase):
    def test_parse_gpu_blocks_variants(self):
        self.assertEqual(parse_gpu_blocks("INFO 06-01 # GPU blocks: 12,345, # CPU blocks: 678"), 12345)
        self.assertEqual(parse_gpu_blocks("... GPU blocks: 999 ..."), 999)
        self.assertIsNone(parse_gpu_blocks("no budget logged here"))

    def test_parse_kv_cache_tokens(self):
        self.assertEqual(parse_kv_cache_tokens("GPU KV cache size: 432,944 tokens"), 432944)
        self.assertIsNone(parse_kv_cache_tokens("nothing"))

    def test_capacity_and_error(self):
        self.assertEqual(kv_capacity_tokens(100, 16), 1600)
        self.assertAlmostEqual(error_pct(54, 60), 10.0)
        self.assertEqual(error_pct(5, 0), float("inf"))


    def test_parse_memory_profile_fields(self):
        line = ("Memory profiling results: total=79.21GiB model weights take 14.99GiB; "
                "non_torch_memory takes 0.12GiB; PyTorch activation peak memory takes 1.20GiB; "
                "the rest of the memory reserved for KV Cache is 56.34 GiB")
        self.assertAlmostEqual(parse_weight_gib(line), 14.99)
        self.assertAlmostEqual(parse_activation_gib(line), 1.20)
        self.assertAlmostEqual(parse_nontorch_gib(line), 0.12)
        self.assertAlmostEqual(parse_kv_reserved_gib(line), 56.34)
        self.assertIsNone(parse_weight_gib("nothing here"))

    def test_parse_vllm_v022_fields(self):
        # v0.22 V1 engine: KV budget at INFO; per-component breakdown at DEBUG.
        info = "Available KV cache memory: 35.42 GiB ... GPU KV cache size: 1,234,560 tokens"
        debug = ("Actual usage is 14.96 GiB for weight, 1.23 GiB for peak activation, "
                 "0.45 GiB for non-torch memory, and 2.10 GiB for CUDAGraph memory.")
        self.assertEqual(parse_kv_cache_tokens(info), 1234560)
        self.assertAlmostEqual(parse_kv_reserved_gib(info), 35.42)
        self.assertAlmostEqual(parse_weight_gib(debug), 14.96)
        self.assertAlmostEqual(parse_activation_gib(debug), 1.23)
        self.assertAlmostEqual(parse_nontorch_gib(debug), 0.45)
        self.assertAlmostEqual(parse_cudagraph_gib(debug), 2.10)


class TestFit(unittest.TestCase):
    def test_exact_linear_recovery(self):
        slope, intercept = fit_activation_overhead([(10.0, 2.0), (20.0, 3.0), (30.0, 4.0)])
        self.assertAlmostEqual(slope, 0.1)
        self.assertAlmostEqual(intercept, 1.0)

    def test_degenerate_returns_mean(self):
        slope, intercept = fit_activation_overhead([(5.0, 2.0), (5.0, 4.0)])
        self.assertEqual(slope, 0.0)
        self.assertAlmostEqual(intercept, 3.0)


class TestCalibrate(unittest.TestCase):
    def _row(self, key, blocks, total=85_000_000_000):
        return {"key": key, "weight_dtype": "bf16", "kv_dtype": "fp16",
                "util": 0.9, "block_size": 16, "num_gpu_blocks": blocks,
                "total_gpu_memory_bytes": total}

    def test_analyze_recovers_nonkv(self):
        rows = [self._row("llama-3.1-8b", 26226), self._row("qwen2.5-32b", 3000),
                {"key": "mistral-7b", "error": "OOM"}]  # errored row must be skipped
        act, overhead, enriched = calibrate.analyze(rows)
        self.assertEqual(len(enriched), 2)
        e = enriched[0]
        # nonkv == usable - weights - measured_kv (the recovered activation+overhead)
        self.assertAlmostEqual(e["nonkv"], e["usable"] - e["weights"] - e["measured_kv"], places=3)
        self.assertGreater(e["measured_kv"], 0)
        self.assertIsInstance(act, float)
        self.assertIsInstance(overhead, float)


if __name__ == "__main__":
    unittest.main()
