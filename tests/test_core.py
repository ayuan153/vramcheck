"""Unit tests for the vramcheck core memory model.

Zero-install: run `python3 -m unittest discover -s tests -t .` from the repo root.
Expected values are hand-computed from the formulas in DESIGN.md §3.
"""

import unittest
from dataclasses import replace

from vramcheck.core import (
    MODELS, GPUS, ModelConfig, GPUSpec, GiB,
    per_token_kv_bytes, kv_bytes_per_seq, kv_waste_bytes, padded_tokens,
    weight_bytes, plan, max_batch, fits, sweep,
)


class TestKVMath(unittest.TestCase):
    def test_gqa_per_token_llama70b(self):
        # 2 (K+V) * 80 layers * 8 kv heads * 128 head_dim * 2 bytes (fp16)
        self.assertEqual(per_token_kv_bytes(MODELS["llama-3.1-70b"], "fp16"), 327_680)

    def test_per_seq_block_aligned_matches_design(self):
        cfg = MODELS["llama-3.1-70b"]
        # 8192 is block-aligned -> no waste; 2.68 GB figure in DESIGN §3.2.
        self.assertEqual(kv_bytes_per_seq(cfg, 8192, "fp16"), 2_684_354_560)
        self.assertAlmostEqual(kv_bytes_per_seq(cfg, 8192) / GiB, 2.5, places=6)
        self.assertEqual(kv_waste_bytes(cfg, 8192), 0)

    def test_gqa_is_8x_smaller_than_mha(self):
        cfg = MODELS["llama-3.1-70b"]  # 64 attn / 8 kv
        mha = replace(cfg, num_key_value_heads=cfg.num_attention_heads)
        self.assertEqual(per_token_kv_bytes(mha) / per_token_kv_bytes(cfg), 8)

    def test_mqa_single_kv_head(self):
        cfg = replace(MODELS["llama-3.1-70b"], num_key_value_heads=1, attention="mqa")
        self.assertEqual(per_token_kv_bytes(cfg, "fp16"), 2 * 80 * 1 * 128 * 2)

    def test_mla_deepseek(self):
        cfg = MODELS["deepseek-v2-lite"]
        # 27 layers * (kv_lora_rank 512 + qk_rope 64) * 2 bytes, no factor of 2.
        self.assertEqual(per_token_kv_bytes(cfg, "fp16"), 31_104)
        self.assertEqual(kv_bytes_per_seq(cfg, 8192, "fp16"), 31_104 * 8192)

    def test_paged_attention_rounding(self):
        self.assertEqual(padded_tokens(8193, 16), 8208)  # rounds up one block
        cfg = MODELS["llama-3.1-8b"]
        self.assertEqual(
            kv_waste_bytes(cfg, 8193, "fp16", 16), 15 * per_token_kv_bytes(cfg, "fp16")
        )

    def test_fp8_halves_fp16(self):
        cfg = MODELS["llama-3.1-8b"]
        self.assertEqual(per_token_kv_bytes(cfg, "fp8") * 2, per_token_kv_bytes(cfg, "fp16"))


class TestMemory(unittest.TestCase):
    # Synthetic clean config so the budget test is independent of vendored param counts.
    SYN = ModelConfig("syn-8B", 8_000_000_000, 32, 4096, 32, 8, 128, "gqa")
    GPU = GPUSpec("A100-80GB", 80)
    KW = dict(util=0.9, act_fraction=0.10, overhead_gib=1.0)

    def test_weight_bytes(self):
        self.assertEqual(weight_bytes(self.SYN, "fp16"), 16_000_000_000)
        self.assertEqual(weight_bytes(self.SYN, "int4"), 4_000_000_000)

    def test_budget_and_max_batch(self):
        p = plan(self.GPU, self.SYN, 8192, **self.KW)
        # syn-8B KV/seq @8192 = 2*32*8*128*2 * 8192 = exactly 1 GiB.
        self.assertEqual(p.kv_per_seq_bytes, 1_073_741_824)
        # usable 0.9*80GiB - 16e9 weights - 1.6e9 act - 1GiB oh = 58,635,669,504 -> //1GiB = 54
        self.assertEqual(p.kv_budget_bytes, 58_635_669_504)
        self.assertEqual(p.max_batch, 54)

    def test_fits_boundary(self):
        self.assertTrue(fits(self.GPU, self.SYN, 8192, 54, **self.KW))
        self.assertFalse(fits(self.GPU, self.SYN, 8192, 55, **self.KW))

    def test_max_batch_monotonic_in_context(self):
        batches = [b for _, b in sweep(GPUS["a100-80gb"], MODELS["llama-3.1-8b"],
                                       [4096, 8192, 16384, 32768])]
        self.assertEqual(batches, sorted(batches, reverse=True))

    def test_oom_returns_zero(self):
        # 70B fp16 weights (~141 GB) exceed one 80GB GPU -> budget < 0 -> 0.
        self.assertEqual(max_batch(GPUS["a100-80gb"], MODELS["llama-3.1-70b"], 8192), 0)


if __name__ == "__main__":
    unittest.main()
