import logging
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from calibration_agent import ml_service
from calibration_agent.ml_service import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    RESEARCH_FEATURE_COLUMNS,
    extract_research_features,
    get_xgboost_probability,
)
from models import Market
from news_research_agent.models import NewsResearchReport
from reddit_research_agent.models import RedditResearchReport


class CalibrationBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset module-level caches so tests don't leak state between runs.
        ml_service._MODEL = None
        ml_service._MODEL_LOADED = False
        ml_service._MOCK_WARNED = False
        os.environ.pop("XGBOOST_MODEL_PATH", None)

    def tearDown(self) -> None:
        ml_service._MODEL = None
        ml_service._MODEL_LOADED = False
        ml_service._MOCK_WARNED = False
        os.environ.pop("XGBOOST_MODEL_PATH", None)

    def _market(self, *, implied_prob=0.52) -> Market:
        return Market(
            market_id="mkt-1",
            title="Test Market",
            category="Politics",
            implied_prob=implied_prob,
            bid_price=max(0.0, implied_prob - 0.01),
            ask_price=min(1.0, implied_prob + 0.01),
            volume_24h=15000.0,
            price_history={"1h": 0.01, "6h": 0.02, "24h": 0.03},
            open_interest=40000.0,
            resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if the event occurs.",
        )

    def test_returns_jittered_probability_from_market(self):
        with patch("calibration_agent.ml_service.random.uniform", return_value=0.02):
            probability = get_xgboost_probability(self._market(implied_prob=0.52))

        self.assertAlmostEqual(probability, 0.54)

    def test_clamps_probability_to_zero_and_one(self):
        with patch("calibration_agent.ml_service.random.uniform", return_value=0.02):
            self.assertEqual(get_xgboost_probability(self._market(implied_prob=0.99)), 1.0)

        with patch("calibration_agent.ml_service.random.uniform", return_value=-0.02):
            self.assertEqual(get_xgboost_probability(self._market(implied_prob=0.01)), 0.0)

    def test_rejects_non_market_inputs(self):
        with self.assertRaisesRegex(TypeError, "Market"):
            get_xgboost_probability({"implied_prob": 0.5})


class MockWarningTests(unittest.TestCase):
    """Verify the loud-mock warning fires exactly once per process."""

    def setUp(self) -> None:
        ml_service._MODEL = None
        ml_service._MODEL_LOADED = False
        ml_service._MOCK_WARNED = False
        os.environ.pop("XGBOOST_MODEL_PATH", None)

    def tearDown(self) -> None:
        ml_service._MODEL = None
        ml_service._MODEL_LOADED = False
        ml_service._MOCK_WARNED = False
        os.environ.pop("XGBOOST_MODEL_PATH", None)

    def _market(self) -> Market:
        return Market(
            market_id="mkt-warn",
            title="Warning market",
            category="Politics",
            implied_prob=0.5,
            bid_price=0.49,
            ask_price=0.51,
            volume_24h=1000.0,
            price_history={},
            open_interest=0.0,
            resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if the event occurs.",
        )

    def test_first_call_emits_loud_warning(self):
        with self.assertLogs("calibration_agent.ml_service", level="DEBUG") as ctx:
            get_xgboost_probability(self._market())

        warnings = [
            record for record in ctx.records if record.levelno == logging.WARNING
        ]
        self.assertEqual(len(warnings), 1)
        self.assertIn("mock XGBoost baseline", warnings[0].getMessage())
        self.assertIn("NOT trustworthy", warnings[0].getMessage())

    def test_subsequent_calls_log_at_debug_only(self):
        # First call to flip the once-per-process flag.
        get_xgboost_probability(self._market())

        with self.assertLogs("calibration_agent.ml_service", level="DEBUG") as ctx:
            get_xgboost_probability(self._market())
            get_xgboost_probability(self._market())

        warnings = [r for r in ctx.records if r.levelno == logging.WARNING]
        debugs = [r for r in ctx.records if r.levelno == logging.DEBUG]
        self.assertEqual(warnings, [])
        self.assertGreaterEqual(len(debugs), 2)


class _StubModel:
    """Picklable stub returning a deterministic positive-class probability."""

    POSITIVE_PROBA = 0.73

    def predict_proba(self, features):
        return [[1.0 - self.POSITIVE_PROBA, self.POSITIVE_PROBA]]


class RealModelLoadTests(unittest.TestCase):
    """Cover the XGBOOST_MODEL_PATH happy path and failure-fallback path."""

    def setUp(self) -> None:
        ml_service._MODEL = None
        ml_service._MODEL_LOADED = False
        ml_service._MOCK_WARNED = False
        os.environ.pop("XGBOOST_MODEL_PATH", None)

    def tearDown(self) -> None:
        ml_service._MODEL = None
        ml_service._MODEL_LOADED = False
        ml_service._MOCK_WARNED = False
        os.environ.pop("XGBOOST_MODEL_PATH", None)

    def _market(self, *, implied_prob=0.4) -> Market:
        return Market(
            market_id="mkt-real",
            title="Real market",
            category="Politics",
            implied_prob=implied_prob,
            bid_price=max(0.0, implied_prob - 0.01),
            ask_price=min(1.0, implied_prob + 0.01),
            volume_24h=2500.0,
            price_history={"1h": 0.0, "6h": 0.0, "24h": 0.0},
            open_interest=0.0,
            resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if the event occurs.",
        )

    def test_real_model_used_when_path_points_to_valid_joblib(self):
        import joblib

        model_path = os.path.join(self._tmp_dir(), "stub_model.joblib")
        joblib.dump(_StubModel(), model_path)
        os.environ["XGBOOST_MODEL_PATH"] = model_path

        # Patch random so any accidental fallback to the mock would be obvious.
        with patch("calibration_agent.ml_service.random.uniform", return_value=0.99):
            probability = get_xgboost_probability(self._market(implied_prob=0.4))

        self.assertAlmostEqual(probability, _StubModel.POSITIVE_PROBA)
        # Cached model object should be the real model, and the mock-warning
        # flag must remain unflipped because the mock path was never taken.
        self.assertTrue(ml_service._MODEL_LOADED)
        self.assertIsNotNone(ml_service._MODEL)
        self.assertFalse(ml_service._MOCK_WARNED)

    def test_real_model_load_failure_falls_back_to_mock(self):
        bad_path = os.path.join(self._tmp_dir(), "this_is_not_a_real_pickle.joblib")
        with open(bad_path, "wb") as fh:
            fh.write(b"not a valid joblib payload")
        os.environ["XGBOOST_MODEL_PATH"] = bad_path

        with self.assertLogs("calibration_agent.ml_service", level="ERROR") as ctx:
            with patch(
                "calibration_agent.ml_service.random.uniform", return_value=0.0
            ):
                probability = get_xgboost_probability(self._market(implied_prob=0.4))

        # Mock fallback returns implied_prob + jitter (jitter forced to 0).
        self.assertAlmostEqual(probability, 0.4)
        # An ERROR was logged describing the load failure.
        errors = [r for r in ctx.records if r.levelno == logging.ERROR]
        self.assertTrue(any("Failed to load XGBoost model" in r.getMessage() for r in errors))

    def test_real_model_missing_file_falls_back_to_mock(self):
        os.environ["XGBOOST_MODEL_PATH"] = "/nonexistent/path/model.joblib"
        with self.assertLogs("calibration_agent.ml_service", level="ERROR") as ctx:
            with patch(
                "calibration_agent.ml_service.random.uniform", return_value=0.0
            ):
                probability = get_xgboost_probability(self._market(implied_prob=0.6))

        self.assertAlmostEqual(probability, 0.6)
        errors = [r for r in ctx.records if r.levelno == logging.ERROR]
        self.assertTrue(any("does not point to an existing file" in r.getMessage() for r in errors))

    def _tmp_dir(self) -> str:
        # unittest doesn't give us tmp_path like pytest, so manage one ourselves
        # and clean up in tearDown.
        import tempfile

        if not hasattr(self, "_tmp"):
            self._tmp = tempfile.mkdtemp(prefix="ml_service_test_")
            self.addCleanup(self._cleanup_tmp)
        return self._tmp

    def _cleanup_tmp(self) -> None:
        import shutil

        shutil.rmtree(self._tmp, ignore_errors=True)


class ExtractResearchFeaturesTests(unittest.TestCase):
    """Cover extract_research_features for full / None / partial inputs."""

    def _full_news_report(self) -> NewsResearchReport:
        return NewsResearchReport(
            timeline=["2026-04-20: Headline"],
            key_facts=["Source-backed fact"],
            source_quality_score=8,
            bullish_thesis="Headline supports YES.",
            bearish_thesis="Coverage may reverse.",
            evidence_quality_score=72,
            misinformation_risk_score=18,
            sentiment_score=35,
            key_sources=["https://example.com/a"],
            summary="Coverage is factual and recent.",
        )

    def _full_reddit_report(self) -> RedditResearchReport:
        return RedditResearchReport(
            bullish_thesis="Primary-source upside case.",
            bearish_thesis="Consensus may already price this in.",
            key_evidence=["Reddit signal"],
            key_assumptions=["The signal is real"],
            conviction_score=7,
            evidence_quality_score=84,
            misinformation_risk_score=12,
            sentiment_score=-22,
            key_sources=["u/sourcehound"],
            summary="Threads converge on a fresh primary-source signal.",
            pricing_assessment="underpriced",
            assessment_reasoning="Discussion contains fresher evidence than the market move.",
        )

    def test_constants_in_expected_order(self):
        # ALL_FEATURE_COLUMNS is the documented model contract: 8 market
        # columns first, then 6 research columns.
        self.assertEqual(len(FEATURE_COLUMNS), 8)
        self.assertEqual(len(RESEARCH_FEATURE_COLUMNS), 6)
        self.assertEqual(ALL_FEATURE_COLUMNS, FEATURE_COLUMNS + RESEARCH_FEATURE_COLUMNS)
        self.assertEqual(
            RESEARCH_FEATURE_COLUMNS,
            (
                "news_sentiment_score",
                "news_evidence_quality_score",
                "news_misinformation_risk_score",
                "reddit_sentiment_score",
                "reddit_evidence_quality_score",
                "reddit_misinformation_risk_score",
            ),
        )

    def test_full_reports_populate_every_field(self):
        feats = extract_research_features(
            self._full_news_report(), self._full_reddit_report()
        )
        self.assertEqual(feats["news_sentiment_score"], 35.0)
        self.assertEqual(feats["news_evidence_quality_score"], 72.0)
        self.assertEqual(feats["news_misinformation_risk_score"], 18.0)
        self.assertEqual(feats["reddit_sentiment_score"], -22.0)
        self.assertEqual(feats["reddit_evidence_quality_score"], 84.0)
        self.assertEqual(feats["reddit_misinformation_risk_score"], 12.0)
        # All values are floats so the model layer can rely on the contract.
        for value in feats.values():
            self.assertIsInstance(value, float)
        # Keys cover every research column exactly.
        self.assertEqual(set(feats.keys()), set(RESEARCH_FEATURE_COLUMNS))

    def test_none_reports_default_to_neutral_zero(self):
        feats = extract_research_features(None, None)
        self.assertEqual(set(feats.keys()), set(RESEARCH_FEATURE_COLUMNS))
        for column in RESEARCH_FEATURE_COLUMNS:
            self.assertEqual(feats[column], 0.0, msg=f"{column} should default to 0.0")

    def test_partial_dict_report_defaults_missing_fields_to_zero(self):
        # Legacy callsite passes raw dicts; missing fields fall back to 0.0.
        partial_news = {"sentiment_score": 50}  # only one of the three news fields
        feats = extract_research_features(partial_news, None)
        self.assertEqual(feats["news_sentiment_score"], 50.0)
        self.assertEqual(feats["news_evidence_quality_score"], 0.0)
        self.assertEqual(feats["news_misinformation_risk_score"], 0.0)
        self.assertEqual(feats["reddit_sentiment_score"], 0.0)
        self.assertEqual(feats["reddit_evidence_quality_score"], 0.0)
        self.assertEqual(feats["reddit_misinformation_risk_score"], 0.0)

    def test_only_one_report_present(self):
        feats = extract_research_features(None, self._full_reddit_report())
        self.assertEqual(feats["news_sentiment_score"], 0.0)
        self.assertEqual(feats["news_evidence_quality_score"], 0.0)
        self.assertEqual(feats["news_misinformation_risk_score"], 0.0)
        self.assertEqual(feats["reddit_sentiment_score"], -22.0)
        self.assertEqual(feats["reddit_evidence_quality_score"], 84.0)
        self.assertEqual(feats["reddit_misinformation_risk_score"], 12.0)

    def test_non_numeric_field_collapses_to_zero(self):
        # Defensive: arbitrary objects with a non-numeric field don't crash.
        class _StubReport:
            sentiment_score = "not a number"
            evidence_quality_score = None
            misinformation_risk_score = 40

        feats = extract_research_features(_StubReport(), _StubReport())
        self.assertEqual(feats["news_sentiment_score"], 0.0)
        self.assertEqual(feats["news_evidence_quality_score"], 0.0)
        self.assertEqual(feats["news_misinformation_risk_score"], 40.0)
        self.assertEqual(feats["reddit_sentiment_score"], 0.0)
        self.assertEqual(feats["reddit_evidence_quality_score"], 0.0)
        self.assertEqual(feats["reddit_misinformation_risk_score"], 40.0)


class FullFeatureVectorTests(unittest.TestCase):
    """Cover _full_feature_vector ordering and length contract."""

    def _market(self) -> Market:
        return Market(
            market_id="mkt-vec",
            title="Vector Market",
            category="Politics",
            implied_prob=0.42,
            bid_price=0.41,
            ask_price=0.43,
            volume_24h=18000.0,
            price_history={"1h": 0.01, "6h": 0.02, "24h": 0.03},
            open_interest=22000.0,
            resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if event occurs.",
        )

    def test_vector_length_matches_all_feature_columns(self):
        vector = ml_service._full_feature_vector(self._market(), None, None)
        self.assertEqual(len(vector), len(ALL_FEATURE_COLUMNS))

    def test_research_signals_appear_in_vector_at_expected_positions(self):
        news = NewsResearchReport(
            timeline=["2026-04-20: Headline"],
            key_facts=["Fact"],
            source_quality_score=7,
            bullish_thesis="Up.",
            bearish_thesis="Down.",
            evidence_quality_score=66,
            misinformation_risk_score=22,
            sentiment_score=11,
            key_sources=["https://example.com/a"],
            summary="Summary.",
        )
        reddit = RedditResearchReport(
            bullish_thesis="Up.",
            bearish_thesis="Down.",
            key_evidence=["Evidence"],
            key_assumptions=["Assumption"],
            conviction_score=6,
            evidence_quality_score=55,
            misinformation_risk_score=33,
            sentiment_score=-44,
            key_sources=["u/source"],
            summary="Summary.",
            pricing_assessment="underpriced",
            assessment_reasoning="Reason.",
        )
        vector = ml_service._full_feature_vector(self._market(), news, reddit)
        # Research columns sit at indices 8..13 (in RESEARCH_FEATURE_COLUMNS order).
        self.assertEqual(vector[8], 11.0)   # news_sentiment_score
        self.assertEqual(vector[9], 66.0)   # news_evidence_quality_score
        self.assertEqual(vector[10], 22.0)  # news_misinformation_risk_score
        self.assertEqual(vector[11], -44.0) # reddit_sentiment_score
        self.assertEqual(vector[12], 55.0)  # reddit_evidence_quality_score
        self.assertEqual(vector[13], 33.0)  # reddit_misinformation_risk_score

    def test_market_features_compat_shim_returns_full_vector_with_zero_research(self):
        # _market_features is now a shim over _full_feature_vector(market, None, None)
        # so the vector still has len(ALL_FEATURE_COLUMNS) entries but with the
        # last 6 zeroed out.
        vector = ml_service._market_features(self._market())
        self.assertEqual(len(vector), len(ALL_FEATURE_COLUMNS))
        for value in vector[len(FEATURE_COLUMNS):]:
            self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
