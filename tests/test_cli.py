"""Smoke tests for the CLI. Zero-install: `python3 -m unittest discover -s tests -t .`."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from vramcheck.cli import main


def run(args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(args)
    return rc, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def test_list(self):
        rc, out, _ = run(["--list"])
        self.assertEqual(rc, 0)
        self.assertIn("llama-3.1-70b", out)
        self.assertIn("A100-80GB", out)

    def test_sweep_default(self):
        rc, out, _ = run(["llama-3.1-8b", "a100-80gb"])
        self.assertEqual(rc, 0)
        self.assertIn("max batch", out)
        self.assertIn("4,096", out)

    def test_sweep_json(self):
        rc, out, _ = run(["llama-3.1-8b", "a100-80gb", "--json"])
        self.assertEqual(rc, 0)
        obj = json.loads(out)
        self.assertEqual(len(obj["sweep"]), 5)
        self.assertEqual(set(obj["breakdown_bytes"]), {"weights", "activation", "overhead", "kv_budget"})

    def test_verdict_fits_and_oom(self):
        _, out_ok, _ = run(["llama-3.1-8b", "a100-80gb", "--ctx", "8192", "--batch", "10", "--json"])
        self.assertTrue(json.loads(out_ok)["fits"])
        _, out_no, _ = run(["llama-3.1-8b", "a100-80gb", "--ctx", "8192", "--batch", "200", "--json"])
        self.assertFalse(json.loads(out_no)["fits"])

    def test_max_batch_mode(self):
        rc, out, _ = run(["llama-3.1-8b", "a100-80gb", "--ctx", "8192", "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["max_batch"], 54)

    def test_model_too_big_is_oom(self):
        _, out, _ = run(["llama-3.1-70b", "a100-80gb", "--ctx", "8192", "--json"])
        self.assertEqual(json.loads(out)["max_batch"], 0)

    def test_unknown_model_errors(self):
        rc, _, err = run(["nope", "a100-80gb"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown model", err)

    def test_batch_requires_ctx(self):
        rc, _, err = run(["llama-3.1-8b", "a100-80gb", "--batch", "4"])
        self.assertEqual(rc, 2)
        self.assertIn("--batch requires --ctx", err)

    def test_unknown_dtype_errors(self):
        rc, _, err = run(["llama-3.1-8b", "a100-80gb", "--weight-dtype", "fp9"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown --weight-dtype", err)


if __name__ == "__main__":
    unittest.main()
