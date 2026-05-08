"""Tests for :mod:`llm_strategy_gen.weekly_job` and :mod:`pr_opener`."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from llm_strategy_gen.backtest_runner import BacktestResult, BacktestRunner
from llm_strategy_gen.feature_proposal import (
    FeatureProposal,
    FeatureProposalGenerator,
)
from llm_strategy_gen.outcome_analyst import OutcomeAnalyst, OutcomePattern
from llm_strategy_gen.pr_opener import (
    DEFAULT_BASE_BRANCH,
    PROpener,
    PROpenResult,
    _slugify,
)
from llm_strategy_gen.weekly_job import WeeklyJob


def _make_proposal(name="atr_z") -> FeatureProposal:
    return FeatureProposal(
        name=name,
        description="Z-scored ATR vs 240-bar baseline.",
        python_code=(
            "def feature(df):\n"
            "    return (df['atr'] - df['atr'].rolling(240).mean())"
        ),
        expected_lift_sharpe=0.18,
        risk_notes="Sensitive to ATR window gaps.",
        proposed_at_utc="2026-05-08T00:00:00Z",
    )


def _passing_result(proposal: FeatureProposal) -> BacktestResult:
    return BacktestResult(
        proposal=proposal,
        sharpe_baseline=1.0,
        sharpe_with_proposal=1.5,
        sharpe_lift=0.5,
        passed_gate=True,
        gate_threshold=0.2,
        notes="STUB",
    )


def _failing_result(proposal: FeatureProposal) -> BacktestResult:
    return BacktestResult(
        proposal=proposal,
        sharpe_baseline=1.0,
        sharpe_with_proposal=1.05,
        sharpe_lift=0.05,
        passed_gate=False,
        gate_threshold=0.2,
        notes="below_gate",
    )


class SlugifyTests(unittest.TestCase):
    def test_basic_lowercases(self):
        self.assertEqual(_slugify("ATR Z-Score"), "atr_z_score")

    def test_strips_non_alnum(self):
        self.assertEqual(_slugify("foo!!! bar??"), "foo_bar")

    def test_empty_falls_back(self):
        self.assertEqual(_slugify(""), "unnamed_feature")

    def test_caps_length(self):
        long = "a" * 200
        self.assertLessEqual(len(_slugify(long)), 48)


class PROpenerTests(unittest.TestCase):
    def test_writes_feature_and_test_files_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            opener = PROpener(repo_root=repo)
            proposal = _make_proposal()
            result = opener.open_pr(
                proposal, _passing_result(proposal), dry_run=True
            )
            self.assertIsInstance(result, PROpenResult)
            self.assertTrue(result.dry_run)
            self.assertTrue(result.feature_path.exists())
            self.assertTrue(result.test_path.exists())
            self.assertIsNone(result.pr_url)
            self.assertIsNone(result.branch_name)
            # Feature file contains the python_code:
            content = result.feature_path.read_text(encoding="utf-8")
            self.assertIn("def feature(df):", content)
            self.assertIn("LLM-proposed feature: atr_z", content)
            # Test stub file is a real Python module that imports unittest:
            test_content = result.test_path.read_text(encoding="utf-8")
            self.assertIn("import unittest", test_content)
            self.assertIn("self.skipTest", test_content)

    def test_dry_run_does_not_invoke_gh_runner(self):
        gh_runner = MagicMock(return_value={"url": "should-not-be-called"})
        with tempfile.TemporaryDirectory() as tmp:
            opener = PROpener(repo_root=Path(tmp), gh_runner=gh_runner)
            proposal = _make_proposal()
            opener.open_pr(proposal, _passing_result(proposal), dry_run=True)
            gh_runner.assert_not_called()

    def test_live_mode_invokes_gh_runner(self):
        gh_runner = MagicMock(return_value={"url": "https://github.com/foo/bar/pull/1"})
        with tempfile.TemporaryDirectory() as tmp:
            opener = PROpener(repo_root=Path(tmp), gh_runner=gh_runner)
            proposal = _make_proposal()
            result = opener.open_pr(
                proposal, _passing_result(proposal), dry_run=False
            )
            gh_runner.assert_called_once()
            argv = gh_runner.call_args[0][0]
            self.assertEqual(argv[0], "gh")
            self.assertIn("--draft", argv)
            self.assertIn("--base", argv)
            self.assertIn(DEFAULT_BASE_BRANCH, argv)
            self.assertEqual(result.pr_url, "https://github.com/foo/bar/pull/1")
            self.assertFalse(result.dry_run)
            self.assertEqual(result.branch_name, "e3/llm-proposed-atr_z")

    def test_live_mode_without_gh_runner_falls_back_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            opener = PROpener(repo_root=Path(tmp))  # no gh_runner
            proposal = _make_proposal()
            result = opener.open_pr(
                proposal, _passing_result(proposal), dry_run=False
            )
            # Falls back to dry-run-equivalent behavior; no crash.
            self.assertTrue(result.dry_run)
            self.assertIsNone(result.pr_url)

    def test_failed_gate_refuses_to_open_pr(self):
        with tempfile.TemporaryDirectory() as tmp:
            opener = PROpener(repo_root=Path(tmp))
            proposal = _make_proposal()
            result = opener.open_pr(
                proposal, _failing_result(proposal), dry_run=True
            )
            self.assertIsNone(result)
            # No artifacts written:
            features_dir = Path(tmp) / "src" / "llm_strategy_gen" / "proposed_features"
            self.assertFalse(features_dir.exists())

    def test_pr_body_contains_caveats(self):
        with tempfile.TemporaryDirectory() as tmp:
            opener = PROpener(repo_root=Path(tmp))
            proposal = _make_proposal()
            result = opener.open_pr(
                proposal, _passing_result(proposal), dry_run=True
            )
            body = result.gh_command[result.gh_command.index("--body") + 1]
            self.assertIn("CAVEATS", body.upper())
            self.assertIn("LLM", body)
            self.assertIn("merging", body.lower())
            self.assertIn("review", body.lower())

    def test_gh_runner_crash_yields_none_url(self):
        gh_runner = MagicMock(side_effect=RuntimeError("kaboom"))
        with tempfile.TemporaryDirectory() as tmp:
            opener = PROpener(repo_root=Path(tmp), gh_runner=gh_runner)
            proposal = _make_proposal()
            result = opener.open_pr(
                proposal, _passing_result(proposal), dry_run=False
            )
            self.assertIsNone(result.pr_url)
            self.assertFalse(result.dry_run)  # opener tried and failed; not a fallback


class WeeklyJobTests(unittest.TestCase):
    def _make_components(
        self,
        *,
        patterns,
        proposals,
        results,
        opener_returns=None,
    ):
        analyst = MagicMock(spec=OutcomeAnalyst)
        analyst.analyze.return_value = patterns

        generator = MagicMock(spec=FeatureProposalGenerator)
        generator.propose.return_value = proposals

        runner = MagicMock(spec=BacktestRunner)
        runner.run.side_effect = results

        opener = MagicMock(spec=PROpener)
        if opener_returns is None:
            # By default, return a non-None PROpenResult per call.
            opener.open_pr.side_effect = [
                PROpenResult(
                    feature_path=Path("/tmp/feat.py"),
                    test_path=Path("/tmp/test.py"),
                    branch_name=None, pr_url=None, gh_command=[], dry_run=True,
                )
                for _ in proposals
            ]
        else:
            opener.open_pr.side_effect = opener_returns
        return analyst, generator, runner, opener

    def test_full_pipeline_counts(self):
        patterns = [
            OutcomePattern(
                pattern_summary="p1", frequency=5, signal_quality_score=80.0,
                evidence_postmortem_ids=["a", "b"],
            ),
        ]
        proposals = [_make_proposal("a"), _make_proposal("b"), _make_proposal("c")]
        results = [
            _passing_result(proposals[0]),
            _failing_result(proposals[1]),
            _passing_result(proposals[2]),
        ]
        analyst, generator, runner, opener = self._make_components(
            patterns=patterns, proposals=proposals, results=results,
        )
        # opener should only be called for passing results (2x)
        opener.open_pr.side_effect = [
            PROpenResult(
                feature_path=Path("/tmp/feat.py"),
                test_path=Path("/tmp/test.py"),
                branch_name=None, pr_url=None, gh_command=[], dry_run=True,
            )
            for _ in range(2)
        ]
        job = WeeklyJob(analyst, generator, runner, opener, dry_run=True)
        counts = job.run()

        self.assertEqual(counts["patterns"], 1)
        self.assertEqual(counts["proposals"], 3)
        self.assertEqual(counts["backtests_run"], 3)
        self.assertEqual(counts["passed_gate"], 2)
        self.assertEqual(counts["prs_opened"], 2)
        self.assertEqual(opener.open_pr.call_count, 2)

    def test_no_patterns_returns_zero_counts(self):
        analyst, generator, runner, opener = self._make_components(
            patterns=[], proposals=[], results=[],
        )
        job = WeeklyJob(analyst, generator, runner, opener, dry_run=True)
        counts = job.run()
        self.assertEqual(counts["patterns"], 0)
        self.assertEqual(counts["proposals"], 0)
        # Generator should NOT be called when there are no patterns.
        generator.propose.assert_not_called()

    def test_no_proposals_returns_zero_downstream(self):
        patterns = [OutcomePattern(
            pattern_summary="p", frequency=1, signal_quality_score=50.0,
            evidence_postmortem_ids=["x"],
        )]
        analyst, generator, runner, opener = self._make_components(
            patterns=patterns, proposals=[], results=[],
        )
        job = WeeklyJob(analyst, generator, runner, opener, dry_run=True)
        counts = job.run()
        self.assertEqual(counts["patterns"], 1)
        self.assertEqual(counts["proposals"], 0)
        runner.run.assert_not_called()

    def test_dry_run_passed_through_to_opener(self):
        proposals = [_make_proposal()]
        analyst, generator, runner, opener = self._make_components(
            patterns=[OutcomePattern(
                pattern_summary="p", frequency=1, signal_quality_score=50.0,
                evidence_postmortem_ids=["x"],
            )],
            proposals=proposals,
            results=[_passing_result(proposals[0])],
        )
        opener.open_pr.side_effect = None
        opener.open_pr.return_value = PROpenResult(
            feature_path=Path("/tmp/feat.py"),
            test_path=Path("/tmp/test.py"),
            branch_name=None, pr_url=None, gh_command=[], dry_run=True,
        )
        job = WeeklyJob(analyst, generator, runner, opener, dry_run=True)
        job.run()
        self.assertEqual(opener.open_pr.call_args.kwargs.get("dry_run"), True)

    def test_analyzer_crash_does_not_propagate(self):
        analyst = MagicMock(spec=OutcomeAnalyst)
        analyst.analyze.side_effect = RuntimeError("synthetic")
        generator = MagicMock(spec=FeatureProposalGenerator)
        runner = MagicMock(spec=BacktestRunner)
        opener = MagicMock(spec=PROpener)
        job = WeeklyJob(analyst, generator, runner, opener, dry_run=True)
        counts = job.run()
        # Returns zero counts; downstream not called.
        self.assertEqual(counts["patterns"], 0)
        generator.propose.assert_not_called()

    def test_runner_crash_skips_proposal_keeps_others(self):
        proposals = [_make_proposal("a"), _make_proposal("b")]
        analyst = MagicMock(spec=OutcomeAnalyst)
        analyst.analyze.return_value = [OutcomePattern(
            pattern_summary="p", frequency=1, signal_quality_score=50.0,
            evidence_postmortem_ids=["x"],
        )]
        generator = MagicMock(spec=FeatureProposalGenerator)
        generator.propose.return_value = proposals
        runner = MagicMock(spec=BacktestRunner)
        runner.run.side_effect = [
            RuntimeError("crash on first"),
            _passing_result(proposals[1]),
        ]
        opener = MagicMock(spec=PROpener)
        opener.open_pr.return_value = PROpenResult(
            feature_path=Path("/tmp/f.py"),
            test_path=Path("/tmp/t.py"),
            branch_name=None, pr_url=None, gh_command=[], dry_run=True,
        )
        job = WeeklyJob(analyst, generator, runner, opener, dry_run=True)
        counts = job.run()
        self.assertEqual(counts["backtests_run"], 1)
        self.assertEqual(counts["passed_gate"], 1)
        self.assertEqual(counts["prs_opened"], 1)


class WeeklyJobCLITests(unittest.TestCase):
    def test_main_runs_with_dry_run(self):
        from llm_strategy_gen.weekly_job import main
        with tempfile.TemporaryDirectory() as tmp:
            # No postmortems → analyst returns []; pipeline is a no-op.
            rc = main(["--repo-root", tmp, "--dry-run"])
            self.assertEqual(rc, 0)

    def test_gate_threshold_resolution_from_env(self):
        from llm_strategy_gen.weekly_job import _resolve_gate_threshold
        # CLI value wins
        self.assertEqual(_resolve_gate_threshold(0.5), 0.5)


if __name__ == "__main__":
    unittest.main()
