"""Tests for alpha_lab.nightly_runner.

Covers the end-to-end mocked flow (miner -> gate -> daily summary) as well
as the CLI argument parsing. The miner and gate are mocked so we don't
duplicate coverage of their internals — those are tested elsewhere.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
from unittest import mock

from alpha_lab.auto_promotion_gate import AutoPromotionGate, PromotionCandidate
from alpha_lab.correlation_miner import (
    CorrelationMiner,
    CorrelationResult,
    FeaturePair,
)
from alpha_lab.nightly_runner import (
    NightlyRunner,
    _parse_horizons,
    build_arg_parser,
    main,
)


def _pair(suffix: str = "x") -> FeaturePair:
    return FeaturePair(
        feature_a=f"signal_{suffix}",
        feature_b=f"target_{suffix}",
        horizon_bars=5,
        asset_class_a="spot_crypto",
        asset_class_b="spot_crypto",
    )


def _result(pair: FeaturePair, rank_ic: float) -> CorrelationResult:
    return CorrelationResult(
        pair=pair,
        rank_ic=rank_ic,
        n_samples=200,
        pvalue=0.01,
        computed_at_utc="2026-05-08T00:00:00+00:00",
    )


class NightlyRunnerEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name)
        self.queue_path = self.output_dir / "promotion_queue.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _runner_with_canned_results(
        self, results: List[CorrelationResult]
    ) -> NightlyRunner:
        # Mock the miner so we don't need real DataFrames; the runner only
        # calls miner.mine() once and treats the return value opaquely.
        miner = mock.create_autospec(CorrelationMiner, instance=True)
        miner.mine.return_value = results
        gate = AutoPromotionGate(
            threshold_rank_ic=0.05,
            min_samples=3,
            promotion_queue_path=self.queue_path,
        )
        return NightlyRunner(miner=miner, gate=gate, output_dir=self.output_dir)

    def test_run_once_writes_daily_summary_to_tmpdir(self) -> None:
        results = [_result(_pair("a"), 0.10), _result(_pair("b"), 0.02)]
        runner = self._runner_with_canned_results(results)
        anchor = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)
        counts = runner.run_once(now_utc=anchor)

        self.assertEqual(counts["results_mined"], 2)
        self.assertEqual(counts["run_date_utc"], "2026-05-08")
        # Two results, but min_samples=3 -> no promotion candidate yet.
        self.assertEqual(counts["promotions_emitted"], 0)

        summary_path = self.output_dir / "2026-05-08.json"
        self.assertTrue(summary_path.exists())
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["counts"]["results_mined"], 2)
        self.assertEqual(payload["counts"]["promotions_emitted"], 0)
        self.assertEqual(payload["window_days"], 30)
        # Top results sorted by |rank_ic| desc — the runner takes whatever
        # the miner gave it, in order.
        self.assertEqual(len(payload["top_results"]), 2)
        self.assertEqual(payload["top_results"][0]["rank_ic"], 0.10)

    def test_run_once_emits_candidate_after_enough_samples(self) -> None:
        pair = _pair("hot")
        # Drive the gate to 3 records of high rank_ic so a candidate emits.
        miner = mock.create_autospec(CorrelationMiner, instance=True)
        gate = AutoPromotionGate(
            threshold_rank_ic=0.05,
            min_samples=3,
            promotion_queue_path=self.queue_path,
        )
        runner = NightlyRunner(miner=miner, gate=gate, output_dir=self.output_dir)

        for day in range(3):
            anchor = datetime(2026, 5, 6, tzinfo=timezone.utc) + timedelta(days=day)
            miner.mine.return_value = [_result(pair, 0.10)]
            counts = runner.run_once(now_utc=anchor)

        # On the third run, the gate has 3 samples -> 1 promotion.
        self.assertEqual(counts["promotions_emitted"], 1)
        # And the JSONL queue has at least one row (it can have more if the
        # gate re-emits across runs because we never dedupe).
        self.assertTrue(self.queue_path.exists())
        lines = self.queue_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 1)
        # The most recent summary lists the candidate.
        summary_path = self.output_dir / "2026-05-08.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["promotion_candidates"]), 1)
        self.assertEqual(
            payload["promotion_candidates"][0]["pair"]["feature_a"], "signal_hot"
        )

    def test_run_once_top_n_capped_at_five(self) -> None:
        # Build 10 results; summary should only embed the top 5.
        results = [_result(_pair(str(i)), 0.20 - 0.01 * i) for i in range(10)]
        runner = self._runner_with_canned_results(results)
        runner.run_once(now_utc=datetime(2026, 5, 8, tzinfo=timezone.utc))
        summary_path = self.output_dir / "2026-05-08.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["top_results"]), 5)

    def test_run_once_empty_results_writes_empty_summary(self) -> None:
        runner = self._runner_with_canned_results([])
        counts = runner.run_once(now_utc=datetime(2026, 5, 8, tzinfo=timezone.utc))
        self.assertEqual(counts["results_mined"], 0)
        self.assertEqual(counts["promotions_emitted"], 0)
        summary_path = self.output_dir / "2026-05-08.json"
        self.assertTrue(summary_path.exists())
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["top_results"], [])
        self.assertEqual(payload["promotion_candidates"], [])

    def test_run_once_naive_anchor_is_normalized_to_utc(self) -> None:
        runner = self._runner_with_canned_results([])
        anchor = datetime(2026, 5, 8, 12, 0)  # tzinfo=None
        counts = runner.run_once(now_utc=anchor)
        # Should not crash; the runner normalizes to UTC internally.
        self.assertEqual(counts["run_date_utc"], "2026-05-08")

    def test_summary_json_atomic_write_no_tmp_file_left(self) -> None:
        runner = self._runner_with_canned_results([_result(_pair(), 0.1)])
        runner.run_once(now_utc=datetime(2026, 5, 8, tzinfo=timezone.utc))
        # Atomic-rename: no .tmp file remains in the directory.
        files = sorted(p.name for p in self.output_dir.iterdir())
        self.assertIn("2026-05-08.json", files)
        for f in files:
            self.assertFalse(f.endswith(".tmp"), f"tmp file leaked: {f}")


class CLIArgParsingTests(unittest.TestCase):
    def test_parser_default_values(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.output_dir, Path("runs/alpha_lab/"))
        self.assertEqual(args.window_days, 30)
        self.assertEqual(args.threshold_rank_ic, 0.05)
        self.assertEqual(args.min_samples, 30)
        self.assertEqual(args.horizons, "5,15,60")
        self.assertEqual(args.log_level, "INFO")
        self.assertIsNone(args.redis_url)

    def test_parser_overrides(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--output-dir", "/tmp/foo",
                "--window-days", "7",
                "--threshold-rank-ic", "0.1",
                "--min-samples", "10",
                "--redis-url", "redis://localhost:6379/2",
                "--horizons", "1,5",
                "--log-level", "DEBUG",
            ]
        )
        self.assertEqual(args.output_dir, Path("/tmp/foo"))
        self.assertEqual(args.window_days, 7)
        self.assertAlmostEqual(args.threshold_rank_ic, 0.1)
        self.assertEqual(args.min_samples, 10)
        self.assertEqual(args.redis_url, "redis://localhost:6379/2")
        self.assertEqual(args.horizons, "1,5")
        self.assertEqual(args.log_level, "DEBUG")

    def test_parse_horizons_happy_path(self) -> None:
        self.assertEqual(_parse_horizons("5,15,60"), [5, 15, 60])
        self.assertEqual(_parse_horizons("1"), [1])
        self.assertEqual(_parse_horizons("  5 , 10 , 30 "), [5, 10, 30])

    def test_parse_horizons_rejects_invalid(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_horizons("abc")
        with self.assertRaises(SystemExit):
            _parse_horizons("0,5")
        with self.assertRaises(SystemExit):
            _parse_horizons("")

    def _build_feature_sources_module(
        self, build_fn
    ):  # type: ignore[no-untyped-def]
        """Inject a stand-in ``alpha_lab.feature_sources`` module into sys.modules.

        Commit 3 of the E2 skeleton ships ``nightly_runner.py`` BEFORE the
        real ``feature_sources.py`` (commit 4). The runner's ``main()`` lazy-
        imports the module so missing-import is non-fatal at runtime; tests
        that want to drive ``main()`` end-to-end install a fake module here.
        """
        import sys
        import types

        mod = types.ModuleType("alpha_lab.feature_sources")
        mod.build_default_feature_sources = build_fn  # type: ignore[attr-defined]
        sys.modules["alpha_lab.feature_sources"] = mod
        # Also register on the parent package so ``from alpha_lab.feature_sources``
        # imports inside ``main()`` resolve cleanly under Python's import system.
        import alpha_lab as parent

        parent.feature_sources = mod  # type: ignore[attr-defined]
        return mod

    def _cleanup_feature_sources_module(self) -> None:
        import sys

        sys.modules.pop("alpha_lab.feature_sources", None)
        import alpha_lab as parent

        if hasattr(parent, "feature_sources"):
            delattr(parent, "feature_sources")

    def test_main_without_sources_logs_warning_and_exits_zero(self) -> None:
        # When build_default_feature_sources returns empty (the skeleton
        # state), main() should log + return 0 rather than crashing.
        self._build_feature_sources_module(lambda: [])
        try:
            tmp = tempfile.TemporaryDirectory()
            try:
                rc = main(["--output-dir", tmp.name])
                self.assertEqual(rc, 0)
                # No summary written because we exited before run_once().
                self.assertEqual(list(Path(tmp.name).glob("*.json")), [])
            finally:
                tmp.cleanup()
        finally:
            self._cleanup_feature_sources_module()

    def test_main_with_mocked_sources_runs_full_cycle(self) -> None:
        # Inject a fake module exposing one trivial source so main() runs
        # end-to-end through the runner. The source returns None to keep
        # the miner cheap (no DataFrames needed).
        fake_source = mock.MagicMock()
        fake_source.name = "fake"
        fake_source.asset_class = "spot_crypto"
        fake_source.fetch_window.return_value = None  # miner skips empty

        self._build_feature_sources_module(lambda: [fake_source])
        try:
            tmp = tempfile.TemporaryDirectory()
            try:
                rc = main(["--output-dir", tmp.name, "--horizons", "5"])
                self.assertEqual(rc, 0)
                files = list(Path(tmp.name).glob("*.json"))
                # A daily summary file should be written even with an empty mine.
                self.assertEqual(len(files), 1)
                payload = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(payload["counts"]["results_mined"], 0)
            finally:
                tmp.cleanup()
        finally:
            self._cleanup_feature_sources_module()

    def test_main_when_feature_sources_module_missing_exits_zero(self) -> None:
        """Regression: nightly_runner must not crash if feature_sources.py
        isn't yet on disk. Commit 3 ships the runner before commit 4 ships
        the module — main() catches ImportError and logs a warning."""
        # Make sure no fake module is registered.
        self._cleanup_feature_sources_module()
        tmp = tempfile.TemporaryDirectory()
        try:
            rc = main(["--output-dir", tmp.name])
            self.assertEqual(rc, 0)
        finally:
            tmp.cleanup()


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()
