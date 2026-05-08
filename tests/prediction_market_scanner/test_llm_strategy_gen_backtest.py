"""Tests for :mod:`llm_strategy_gen.backtest_runner`."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_strategy_gen.backtest_runner import (
    DEFAULT_GATE_THRESHOLD,
    BacktestResult,
    BacktestRunner,
    _validate_proposal_safety,
)
from llm_strategy_gen.feature_proposal import FeatureProposal


def _make_proposal(*, name="atr_z", python_code=None) -> FeatureProposal:
    if python_code is None:
        python_code = (
            "def feature(df):\n"
            "    return (df['atr'] - df['atr'].mean()) / df['atr'].std()"
        )
    return FeatureProposal(
        name=name,
        description="test proposal",
        python_code=python_code,
        expected_lift_sharpe=0.15,
        risk_notes="none",
        proposed_at_utc="2026-05-08T00:00:00Z",
    )


class ValidateProposalSafetyTests(unittest.TestCase):
    def test_safe_code_passes(self):
        ok, reason = _validate_proposal_safety(
            "def feature(df):\n    return df.rolling(10).mean()"
        )
        self.assertTrue(ok, msg=reason)
        self.assertEqual(reason, "")

    def test_imports_pandas_passes(self):
        ok, reason = _validate_proposal_safety(
            "import pandas as pd\n"
            "def feature(df):\n"
            "    return pd.Series(df['x']).rolling(5).mean()"
        )
        self.assertTrue(ok, msg=reason)

    def test_os_import_rejected(self):
        ok, reason = _validate_proposal_safety("import os\nos.system('ls')")
        self.assertFalse(ok)
        self.assertIn("os", reason)

    def test_subprocess_import_rejected(self):
        ok, reason = _validate_proposal_safety(
            "import subprocess\nsubprocess.run(['ls'])"
        )
        self.assertFalse(ok)
        self.assertIn("subprocess", reason)

    def test_dunder_import_call_rejected(self):
        ok, reason = _validate_proposal_safety(
            "x = __import__('os')\nx.system('ls')"
        )
        self.assertFalse(ok)
        self.assertIn("__import__", reason)

    def test_exec_call_rejected(self):
        ok, reason = _validate_proposal_safety("exec('print(1)')")
        self.assertFalse(ok)
        self.assertIn("exec", reason)

    def test_eval_call_rejected(self):
        ok, reason = _validate_proposal_safety("eval('1+1')")
        self.assertFalse(ok)
        self.assertIn("eval", reason)

    def test_open_call_rejected(self):
        ok, reason = _validate_proposal_safety("open('/etc/passwd').read()")
        self.assertFalse(ok)
        self.assertIn("open", reason)

    def test_attribute_access_to_forbidden_root(self):
        # Even if `os` isn't imported in this snippet — defense in depth.
        ok, reason = _validate_proposal_safety("os.system('ls')")
        self.assertFalse(ok)
        self.assertIn("os", reason)

    def test_from_subprocess_rejected(self):
        ok, reason = _validate_proposal_safety(
            "from subprocess import run\nrun(['ls'])"
        )
        self.assertFalse(ok)
        self.assertIn("subprocess", reason)

    def test_syntax_error_rejected(self):
        ok, reason = _validate_proposal_safety("def feature(df:")
        self.assertFalse(ok)
        self.assertIn("syntax_error", reason)

    def test_pickle_rejected(self):
        ok, reason = _validate_proposal_safety("import pickle\npickle.loads(b'x')")
        self.assertFalse(ok)
        self.assertIn("pickle", reason)


class BacktestRunnerTests(unittest.TestCase):
    def test_safe_proposal_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            runner = BacktestRunner(dataset_path=d / "ds.parquet")
            result = runner.run(_make_proposal())
            self.assertIsInstance(result, BacktestResult)
            self.assertEqual(result.gate_threshold, DEFAULT_GATE_THRESHOLD)
            self.assertEqual(result.sharpe_baseline, 0.0)
            # Stub Sharpe is in [-0.1, +0.4] on top of baseline 0.
            self.assertGreaterEqual(result.sharpe_with_proposal, -0.1)
            self.assertLessEqual(result.sharpe_with_proposal, 0.4)
            self.assertEqual(
                result.sharpe_lift,
                result.sharpe_with_proposal - result.sharpe_baseline,
            )
            self.assertIn("STUB", result.notes)

    def test_unsafe_proposal_rejected_with_zero_sharpe(self):
        bad = _make_proposal(
            name="bad",
            python_code="import os\nos.system('rm -rf /')",
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(dataset_path=Path(tmp) / "ds.parquet")
            result = runner.run(bad)
            self.assertFalse(result.passed_gate)
            self.assertEqual(result.sharpe_with_proposal, 0.0)
            self.assertEqual(result.sharpe_baseline, 0.0)
            self.assertIn("safety_reject", result.notes)

    def test_baseline_meta_loaded_from_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            meta_path = d / "baseline.json"
            meta_path.write_text(json.dumps({"sharpe_baseline": 1.5}), encoding="utf-8")
            runner = BacktestRunner(dataset_path=d / "ds.parquet")
            result = runner.run(_make_proposal(), baseline_meta_path=meta_path)
            self.assertEqual(result.sharpe_baseline, 1.5)
            self.assertAlmostEqual(
                result.sharpe_lift,
                result.sharpe_with_proposal - 1.5,
                places=6,
            )

    def test_missing_baseline_meta_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            runner = BacktestRunner(dataset_path=d / "ds.parquet")
            result = runner.run(
                _make_proposal(),
                baseline_meta_path=d / "no-such.json",
            )
            self.assertEqual(result.sharpe_baseline, 0.0)

    def test_corrupt_baseline_meta_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            meta = d / "baseline.json"
            meta.write_text("{ not json", encoding="utf-8")
            runner = BacktestRunner(dataset_path=d / "ds.parquet")
            result = runner.run(_make_proposal(), baseline_meta_path=meta)
            self.assertEqual(result.sharpe_baseline, 0.0)

    def test_deterministic_for_same_proposal(self):
        """Same proposal text → same stub Sharpe (used by tests + audit)."""
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(dataset_path=Path(tmp) / "ds.parquet")
            r1 = runner.run(_make_proposal(name="repro"))
            r2 = runner.run(_make_proposal(name="repro"))
            self.assertEqual(r1.sharpe_with_proposal, r2.sharpe_with_proposal)

    def test_gate_threshold_respected(self):
        # With an absurdly high gate, even good proposals fail.
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(
                dataset_path=Path(tmp) / "ds.parquet",
                gate_threshold=10.0,
            )
            r = runner.run(_make_proposal())
            self.assertFalse(r.passed_gate)
            self.assertEqual(r.gate_threshold, 10.0)

        # With a permissive gate (well below the stub min), it might pass.
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(
                dataset_path=Path(tmp) / "ds.parquet",
                gate_threshold=-1.0,
            )
            r = runner.run(_make_proposal())
            self.assertTrue(r.passed_gate)

    def test_to_dict_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(dataset_path=Path(tmp) / "ds.parquet")
            r = runner.run(_make_proposal())
            d = r.to_dict()
            self.assertIn("proposal", d)
            self.assertIn("sharpe_lift", d)
            self.assertIn("passed_gate", d)
            self.assertIn("notes", d)
            # Re-serializable as JSON.
            json.dumps(d)


if __name__ == "__main__":
    unittest.main()
