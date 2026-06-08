"""Tests for :mod:`llm_strategy_gen.outcome_analyst`."""
from __future__ import annotations

import datetime as _dt
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_strategy_gen.outcome_analyst import (
    DEFAULT_SIGNAL_QUALITY_SCORE,
    MAX_LLM_CALLS_PER_ANALYZE,
    OutcomeAnalyst,
    OutcomePattern,
)


def _utcnow_minus(hours: int = 0, days: int = 0) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=hours, days=days)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_postmortem(
    dir_path: Path,
    *,
    trade_id: str,
    symbol: str,
    root_cause: str,
    triggered_at: str,
) -> None:
    payload = {
        "trade_id": trade_id,
        "symbol": symbol,
        "triggered_at_utc": triggered_at,
        "synthesizer": {"root_cause": root_cause, "summary": "test"},
        "findings": [],
    }
    (dir_path / f"{trade_id}.json").write_text(json.dumps(payload), encoding="utf-8")


class _FakeLLM:
    """Test-injected LLM caller; bypasses the env-var safety check."""

    _test_inject = True

    def __init__(self, *, response=None, raises=None):
        self.response = response
        self.raises = raises
        self.calls = 0

    def __call__(self, prompt, timeout_s=30):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.response


class OutcomeAnalystEmptyTests(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyst = OutcomeAnalyst(postmortems_dir=Path(tmp) / "does-not-exist")
            self.assertEqual(analyst.analyze(window_days=7), [])

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyst = OutcomeAnalyst(postmortems_dir=Path(tmp))
            self.assertEqual(analyst.analyze(window_days=7), [])

    def test_zero_window_days_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_postmortem(
                d, trade_id="t1", symbol="BTC-USD",
                root_cause="Signal", triggered_at=_utcnow_minus(hours=1),
            )
            analyst = OutcomeAnalyst(postmortems_dir=d)
            self.assertEqual(analyst.analyze(window_days=0), [])


class OutcomeAnalystClusteringTests(unittest.TestCase):
    def _build_dataset(self, dir_path: Path) -> None:
        # 10 postmortems mixing causes/symbols.
        recipes = [
            ("t01", "BTC-USD", "Signal"),
            ("t02", "BTC-USD", "Signal"),
            ("t03", "BTC-USD", "Signal"),
            ("t04", "BTC-USD", "Sizing"),
            ("t05", "ETH-USD", "Signal"),
            ("t06", "ETH-USD", "Execution"),
            ("t07", "ETH-USD", "Execution"),
            ("t08", "SOL-USD", "Mixed"),
            ("t09", "SOL-USD", "Sizing"),
            ("t10", "SOL-USD", "Sizing"),
        ]
        for tid, sym, rc in recipes:
            _write_postmortem(
                dir_path, trade_id=tid, symbol=sym,
                root_cause=rc, triggered_at=_utcnow_minus(hours=1),
            )

    def test_synthetic_fallback_without_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._build_dataset(d)
            analyst = OutcomeAnalyst(postmortems_dir=d)  # no llm_caller
            patterns = analyst.analyze(window_days=7)
            self.assertGreaterEqual(len(patterns), 1)
            for p in patterns:
                self.assertIsInstance(p, OutcomePattern)
                self.assertEqual(p.signal_quality_score, DEFAULT_SIGNAL_QUALITY_SCORE)
                self.assertGreaterEqual(p.frequency, 1)
                # Frequency-descending
            for i in range(1, len(patterns)):
                self.assertGreaterEqual(patterns[i - 1].frequency, patterns[i].frequency)
            # Top cluster should be BTC-USD/Signal (3 entries) — same freq
            # as SOL-USD/Sizing (3) but ties break by (root_cause,symbol)
            # alphabetically; both are length 3 so the top group has
            # frequency 3.
            self.assertEqual(patterns[0].frequency, 3)

    def test_excludes_postmortems_outside_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_postmortem(
                d, trade_id="recent", symbol="BTC-USD",
                root_cause="Signal", triggered_at=_utcnow_minus(hours=1),
            )
            _write_postmortem(
                d, trade_id="old", symbol="BTC-USD",
                root_cause="Signal", triggered_at=_utcnow_minus(days=30),
            )
            analyst = OutcomeAnalyst(postmortems_dir=d)
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(len(patterns), 1)
            self.assertEqual(patterns[0].evidence_postmortem_ids, ["recent"])

    def test_unparseable_timestamp_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_postmortem(
                d, trade_id="bad-ts", symbol="BTC-USD",
                root_cause="Signal", triggered_at="not-a-timestamp",
            )
            _write_postmortem(
                d, trade_id="good", symbol="BTC-USD",
                root_cause="Signal", triggered_at=_utcnow_minus(hours=1),
            )
            analyst = OutcomeAnalyst(postmortems_dir=d)
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(sum(p.frequency for p in patterns), 1)

    def test_unreadable_json_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "broken.json").write_text("{ not json", encoding="utf-8")
            _write_postmortem(
                d, trade_id="good", symbol="BTC-USD",
                root_cause="Signal", triggered_at=_utcnow_minus(hours=1),
            )
            analyst = OutcomeAnalyst(postmortems_dir=d)
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(sum(p.frequency for p in patterns), 1)


class OutcomeAnalystLLMTests(unittest.TestCase):
    def test_llm_called_and_caps_at_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # 5 distinct (root_cause, symbol) clusters → would call 5x
            # without the cap.
            for i, (rc, sym) in enumerate(
                [("Signal", "BTC-USD"), ("Sizing", "BTC-USD"),
                 ("Execution", "ETH-USD"), ("Process", "ETH-USD"),
                 ("Mixed", "SOL-USD")]
            ):
                _write_postmortem(
                    d, trade_id=f"t{i}", symbol=sym, root_cause=rc,
                    triggered_at=_utcnow_minus(hours=1),
                )

            fake = _FakeLLM(
                response={"pattern_summary": "LLM-refined summary",
                         "signal_quality_score": 87}
            )
            analyst = OutcomeAnalyst(postmortems_dir=d, llm_caller=fake)
            patterns = analyst.analyze(window_days=7)

            self.assertEqual(len(patterns), 5)
            self.assertEqual(fake.calls, MAX_LLM_CALLS_PER_ANALYZE)
            # First 3 patterns should have the LLM-refined summary;
            # the remaining 2 fall back to synthetic.
            llm_refined = [p for p in patterns if p.pattern_summary == "LLM-refined summary"]
            self.assertEqual(len(llm_refined), 3)
            for p in llm_refined:
                self.assertEqual(p.signal_quality_score, 87.0)

    def test_llm_timeout_graceful_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_postmortem(
                d, trade_id="t1", symbol="BTC-USD", root_cause="Signal",
                triggered_at=_utcnow_minus(hours=1),
            )
            fake = _FakeLLM(raises=TimeoutError("simulated timeout"))
            analyst = OutcomeAnalyst(postmortems_dir=d, llm_caller=fake)
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(len(patterns), 1)
            # Falls back to synthetic summary + default score.
            self.assertEqual(patterns[0].signal_quality_score, DEFAULT_SIGNAL_QUALITY_SCORE)
            self.assertIn("Signal", patterns[0].pattern_summary)
            self.assertIn("BTC-USD", patterns[0].pattern_summary)

    def test_llm_returns_garbage_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_postmortem(
                d, trade_id="t1", symbol="BTC-USD", root_cause="Signal",
                triggered_at=_utcnow_minus(hours=1),
            )
            fake = _FakeLLM(response="not-a-dict")
            analyst = OutcomeAnalyst(postmortems_dir=d, llm_caller=fake)
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(len(patterns), 1)
            self.assertEqual(patterns[0].signal_quality_score, DEFAULT_SIGNAL_QUALITY_SCORE)

    def test_missing_api_key_no_inject_skips_llm(self):
        # _test_inject=False simulates a real production caller that
        # consults env. With no GEMINI_API_KEY in env we should bail
        # out before invoking it.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_postmortem(
                d, trade_id="t1", symbol="BTC-USD", root_cause="Signal",
                triggered_at=_utcnow_minus(hours=1),
            )

            class _ProdLLM:
                _test_inject = False
                calls = 0

                def __call__(self, prompt, timeout_s=30):
                    self.calls += 1
                    return {"pattern_summary": "should-not-be-called", "signal_quality_score": 99}

            prod = _ProdLLM()
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GEMINI_API_KEY", None)
                os.environ.pop("GOOGLE_API_KEY", None)
                analyst = OutcomeAnalyst(postmortems_dir=d, llm_caller=prod)
                patterns = analyst.analyze(window_days=7)
            self.assertEqual(prod.calls, 0)
            self.assertEqual(patterns[0].signal_quality_score, DEFAULT_SIGNAL_QUALITY_SCORE)


class OutcomeAnalystAuditLogTests(unittest.TestCase):
    def test_performance_audit_folded_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            audit_path = d / "performance_audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "reviews": [
                            {
                                "trade_id": "legacy-1",
                                "symbol": "BTC-USD",
                                "settled_at": _utcnow_minus(hours=2),
                                "outcome_review": {"matrix_classification": "Good Failure"},
                            },
                            {
                                "trade_id": "legacy-old",
                                "symbol": "BTC-USD",
                                "settled_at": _utcnow_minus(days=60),
                                "outcome_review": {},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pm_dir = d / "postmortems"
            pm_dir.mkdir()
            analyst = OutcomeAnalyst(
                postmortems_dir=pm_dir,
                performance_audit_path=audit_path,
            )
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(len(patterns), 1)
            self.assertEqual(patterns[0].evidence_postmortem_ids, ["legacy-1"])

    def test_missing_audit_path_no_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pm_dir = d / "postmortems"
            pm_dir.mkdir()
            _write_postmortem(
                pm_dir, trade_id="t1", symbol="BTC-USD",
                root_cause="Signal", triggered_at=_utcnow_minus(hours=1),
            )
            analyst = OutcomeAnalyst(
                postmortems_dir=pm_dir,
                performance_audit_path=d / "no-such-file.json",
            )
            patterns = analyst.analyze(window_days=7)
            self.assertEqual(len(patterns), 1)


if __name__ == "__main__":
    unittest.main()
