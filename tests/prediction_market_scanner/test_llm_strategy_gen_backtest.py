"""Tests for :mod:`llm_strategy_gen.backtest_runner`."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from llm_strategy_gen.backtest_runner import (
    DEFAULT_GATE_THRESHOLD,
    BacktestResult,
    BacktestRunner,
    _extract_sharpe,
    _materialize_proposal_feature,
    _slug_for_feature_col,
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


class SlugForFeatureColTests(unittest.TestCase):
    def test_slug_alnum_safe(self):
        self.assertEqual(
            _slug_for_feature_col("atr-z (rolling)"), "llm_proposed_atr_z_rolling"
        )

    def test_slug_caps_length(self):
        long = "X" * 200
        out = _slug_for_feature_col(long)
        # llm_proposed_ + at most 40 chars from the name
        self.assertLessEqual(len(out), len("llm_proposed_") + 40)

    def test_slug_handles_empty(self):
        self.assertEqual(_slug_for_feature_col(""), "llm_proposed_feature")


class ExtractSharpeTests(unittest.TestCase):
    def test_pulls_sim_sharpe(self):
        self.assertEqual(_extract_sharpe({"sim_sharpe": 1.7}), 1.7)

    def test_missing_defaults_zero(self):
        self.assertEqual(_extract_sharpe({}), 0.0)

    def test_nan_defaults_zero(self):
        self.assertEqual(_extract_sharpe({"sim_sharpe": float("nan")}), 0.0)

    def test_inf_defaults_zero(self):
        self.assertEqual(_extract_sharpe({"sim_sharpe": float("inf")}), 0.0)

    def test_non_numeric_defaults_zero(self):
        self.assertEqual(_extract_sharpe({"sim_sharpe": "not-a-number"}), 0.0)


class MaterializeProposalFeatureTests(unittest.TestCase):
    def test_returns_callable_feature_function(self):
        proposal = _make_proposal(
            python_code=(
                "def feature(df):\n"
                "    return df['x'] * 2"
            )
        )
        fn = _materialize_proposal_feature(proposal)
        df = pd.DataFrame({"x": [1, 2, 3]})
        out = fn(df)
        self.assertTrue((out.to_numpy() == np.array([2, 4, 6])).all())

    def test_rejects_proposal_without_feature_name(self):
        proposal = _make_proposal(
            python_code="def other_fn(df):\n    return df['x']"
        )
        with self.assertRaises(ValueError):
            _materialize_proposal_feature(proposal)

    def test_restricted_globals_blocks_open_at_exec(self):
        # ``open`` is not exposed in the restricted builtins; a call should
        # raise NameError when the function executes (even though the AST
        # check also catches this earlier).
        proposal = _make_proposal(
            python_code=(
                "def feature(df):\n"
                "    open('/tmp/x')\n"
                "    return df['x']"
            )
        )
        fn = _materialize_proposal_feature(proposal)
        with self.assertRaises(NameError):
            fn(pd.DataFrame({"x": [1]}))


class BacktestRunnerLiveFailClosedTests(unittest.TestCase):
    """live=True paths that should fail closed (passed_gate=False) before
    spending any training time."""

    def test_missing_dataset_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(
                dataset_path=Path(tmp) / "nope.parquet", live=True
            )
            result = runner.run(_make_proposal())
        self.assertFalse(result.passed_gate)
        self.assertIn("dataset_missing", result.notes)

    def test_safety_reject_still_fires_in_live_mode(self):
        # Safety reject lives in run(), so live=True must NOT bypass it.
        with tempfile.TemporaryDirectory() as tmp:
            runner = BacktestRunner(
                dataset_path=Path(tmp) / "any.parquet", live=True
            )
            bad = _make_proposal(
                python_code=(
                    "import os\n"
                    "def feature(df):\n"
                    "    return df['x']"
                )
            )
            result = runner.run(bad)
        self.assertFalse(result.passed_gate)
        self.assertIn("safety_reject", result.notes)
        # safety_reject path returns 0/0 — we never reached the live machinery.
        self.assertEqual(result.sharpe_baseline, 0.0)

    def test_feature_call_raising_fails_closed(self):
        # Synthesize a 540-row dataset so train() doesn't choke, then a
        # proposal whose feature() raises at call time.
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds.parquet"
            _build_synthetic_parquet(ds, n=540)
            runner = BacktestRunner(dataset_path=ds, live=True)
            bad = _make_proposal(
                python_code=(
                    "def feature(df):\n"
                    "    raise RuntimeError('boom')"
                )
            )
            result = runner.run(bad)
        self.assertFalse(result.passed_gate)
        self.assertIn("feature_call_failed", result.notes)

    def test_feature_returning_wrong_length_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds.parquet"
            _build_synthetic_parquet(ds, n=540)
            runner = BacktestRunner(dataset_path=ds, live=True)
            bad = _make_proposal(
                python_code=(
                    "def feature(df):\n"
                    "    return [0.0, 1.0, 2.0]"
                )
            )
            result = runner.run(bad)
        self.assertFalse(result.passed_gate)
        self.assertIn("feature_length_mismatch", result.notes)

    def test_feature_returning_non_numeric_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds.parquet"
            _build_synthetic_parquet(ds, n=540)
            runner = BacktestRunner(dataset_path=ds, live=True)
            bad = _make_proposal(
                python_code=(
                    "def feature(df):\n"
                    "    return ['hello'] * len(df)"
                )
            )
            result = runner.run(bad)
        self.assertFalse(result.passed_gate)
        # Either feature_not_numeric (cast fails) OR train_failed (XGBoost
        # rejects). Both are fail-closed; assert one of them.
        self.assertTrue(
            "feature_not_numeric" in result.notes
            or "train_failed" in result.notes,
            msg=f"unexpected notes: {result.notes}",
        )


def _build_synthetic_parquet(path: Path, *, n: int = 540, seed: int = 7) -> None:
    """Build a small parquet with the schema train() expects: timestamp +
    numeric feature columns + binary ``label``. Used to exercise the
    live-mode backtest without depending on the real datasets."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, size=n)
    f2 = rng.normal(0, 1, size=n)
    f3 = rng.normal(0, 1, size=n)
    # Label has some signal so isotonic calibration doesn't collapse to a
    # constant — needed because train() chokes on degenerate distributions
    # in some sklearn versions.
    score = 0.8 * f1 + 0.4 * f2 + 0.1 * rng.normal(0, 1, size=n)
    label = (score > np.median(score)).astype(int)
    ts = pd.date_range("2026-01-01", periods=n, freq="1min").astype(str)
    df = pd.DataFrame({
        "timestamp": ts, "f1": f1, "f2": f2, "f3": f3, "label": label,
    })
    df.to_parquet(path, index=False)


class BacktestRunnerLiveSmokeTests(unittest.TestCase):
    """End-to-end live mode: trains baseline + with-proposal models on a
    synthetic dataset and asserts the result has finite Sharpes and a
    completed-vs-failed disposition."""

    def test_live_mode_runs_walk_forward_and_returns_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds.parquet"
            _build_synthetic_parquet(ds, n=540, seed=11)
            runner = BacktestRunner(
                dataset_path=ds, gate_threshold=999.0, live=True
            )  # gate=999 so we cleanly assert "did not pass"
            proposal = _make_proposal(
                python_code=(
                    "def feature(df):\n"
                    "    return df['f1'] - df['f1'].mean()"
                )
            )
            result = runner.run(proposal)
        # The result should be a fully-populated BacktestResult, not the
        # fail-closed shape: notes should describe the live walk-forward.
        self.assertIn("live_walk_forward", result.notes)
        # Sharpe values must be finite numbers.
        self.assertTrue(np.isfinite(result.sharpe_baseline))
        self.assertTrue(np.isfinite(result.sharpe_with_proposal))
        self.assertTrue(np.isfinite(result.sharpe_lift))
        self.assertFalse(result.passed_gate)  # gate=999 is unreachable

    def test_live_mode_low_gate_can_pass_on_meaningful_feature(self):
        # With a forgiving gate, a feature that actually correlates with
        # the label should not get rejected. We're not asserting it passes
        # (small synthetic data → noisy Sharpe), only that the gate check
        # logic runs to completion without a fail-closed note.
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "ds.parquet"
            _build_synthetic_parquet(ds, n=600, seed=42)
            runner = BacktestRunner(
                dataset_path=ds, gate_threshold=-100.0, live=True
            )
            proposal = _make_proposal(
                python_code=(
                    "def feature(df):\n"
                    "    return df['f1'].rolling(10, min_periods=1).mean()"
                )
            )
            result = runner.run(proposal)
        self.assertIn("live_walk_forward", result.notes)


if __name__ == "__main__":
    unittest.main()
