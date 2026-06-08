import unittest
from datetime import datetime, timezone

from calibration_agent.analyzer import CalibrationAgent, SYSTEM_PROMPT
from calibration_agent.models import CalibrationReport
from models import Market
from news_research_agent.models import NewsResearchReport


class FakeHttpOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeGenerationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponse:
    def __init__(self, parsed=None, text=""):
        self.parsed = parsed
        self.text = text


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


class FakeClient:
    def __init__(self, response):
        self.models = FakeModels(response)


class FakeClientFactoryModule:
    def __init__(self):
        self.calls = []

    def Client(self, *, api_key):
        self.calls.append(api_key)
        return FakeClient(FakeResponse(parsed={}))


class FakeTypesModule:
    def HttpOptions(self, **kwargs):
        return FakeHttpOptions(**kwargs)

    def GenerateContentConfig(self, **kwargs):
        return FakeGenerationConfig(**kwargs)


class CalibrationAgentTests(unittest.TestCase):
    def _response_payload(self) -> CalibrationReport:
        return CalibrationReport(
            xgboost_prob=0.56,
            llm_adjustment_pct_points=1.5,
            calibrated_true_prob=0.575,
            confidence_score=68,
            key_drivers=["News flow is supportive.", "The research surfaced a credible data point."],
            key_uncertainties=["Late polling could reverse the move.", "Evidence quality is still moderate."],
            edge_vs_market=0.035,
            action="monitor",
            reasoning="The baseline moved slightly higher because the qualitative evidence is supportive but not decisive.",
        )

    def _market(self) -> Market:
        return Market(
            market_id="mkt-1",
            title="Market Alpha",
            category="Politics",
            implied_prob=0.54,
            bid_price=0.53,
            ask_price=0.55,
            volume_24h=12000.0,
            price_history={"1h": 0.01, "6h": 0.02, "24h": 0.03},
            open_interest=25000.0,
            resolution_date=datetime(2026, 11, 5, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if Alpha occurs.",
        )

    def _news_report(self) -> NewsResearchReport:
        return NewsResearchReport(
            timeline=["2026-04-20T10:00:00+00:00 | Headline A"],
            key_facts=["Source-backed fact"],
            source_quality_score=7,
            bullish_thesis="Primary-source headline supports the YES side resolving.",
            bearish_thesis="Coverage volume is thin and a late counter-headline could reverse the setup.",
            evidence_quality_score=70,
            misinformation_risk_score=15,
            sentiment_score=20,
            key_sources=["https://example.com/a"],
            summary="Coverage is factual and recent.",
        )

    def test_calibrate_returns_calibration_report(self):
        client = FakeClient(FakeResponse(parsed=self._response_payload()))
        agent = CalibrationAgent(api_key="test-key", client=client, types_module=FakeTypesModule())

        report = agent.calibrate(
            market=self._market(),
            news_report=self._news_report(),
            xgboost_prob=0.56,
            reddit_report={"pricing_assessment": "underpriced", "evidence_quality_score": 8},
        )

        self.assertEqual(report.action, "monitor")
        self.assertEqual(len(client.models.calls), 1)
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-2.5-pro")
        self.assertEqual(call["config"].kwargs["system_instruction"], SYSTEM_PROMPT)
        self.assertEqual(call["config"].kwargs["response_mime_type"], "application/json")
        self.assertIs(call["config"].kwargs["response_schema"], CalibrationReport)
        self.assertEqual(call["config"].kwargs["temperature"], 0.2)
        self.assertEqual(call["config"].kwargs["http_options"].kwargs["timeout"], 30)
        self.assertIn('"market"', call["contents"])
        self.assertIn('"news_research_report"', call["contents"])
        self.assertIn('"reddit_research_report"', call["contents"])
        self.assertIn('"title": "Market Alpha"', call["contents"])
        self.assertIn('"xgboost_prob": 0.56', call["contents"])
        self.assertIn('"pricing_assessment": "underpriced"', call["contents"])

    def test_calibrate_uses_placeholder_news_report_when_missing(self):
        client = FakeClient(FakeResponse(parsed=self._response_payload()))
        agent = CalibrationAgent(api_key="test-key", client=client, types_module=FakeTypesModule())

        agent.calibrate(
            market=self._market(),
            news_report=None,
            xgboost_prob=0.56,
        )

        call = client.models.calls[0]
        self.assertIn("No news research report was provided for Market Alpha", call["contents"])
        self.assertIn('"source_quality_score": 0', call["contents"])

    def test_constructor_uses_new_client_factory(self):
        fake_genai = FakeClientFactoryModule()
        agent = CalibrationAgent(
            api_key="test-key",
            genai_module=fake_genai,
            types_module=FakeTypesModule(),
        )

        self.assertEqual(fake_genai.calls, ["test-key"])
        self.assertIsNotNone(agent.client)

    def test_invalid_inputs_raise(self):
        agent = CalibrationAgent(api_key="test-key", client=FakeClient(FakeResponse(parsed=self._response_payload())), types_module=FakeTypesModule())

        with self.assertRaisesRegex(TypeError, "Market"):
            agent.calibrate(
                market={"title": "not-a-market"},
                news_report=self._news_report(),
                xgboost_prob=0.5,
            )

        with self.assertRaisesRegex(ValueError, "xgboost_prob"):
            agent.calibrate(
                market=self._market(),
                news_report=self._news_report(),
                xgboost_prob=1.2,
            )

        with self.assertRaisesRegex(ValueError, "market_features"):
            agent.calibrate_probability(
                market_features={},
                research_summaries={"news": self._news_report().model_dump()},
                xgboost_baseline=0.5,
                market_implied_prob=0.5,
            )


class CalibrationXGBoostConstraintTests(unittest.TestCase):
    """Characterize the relationship between ``xgboost_prob`` and
    ``calibrated_true_prob`` under different ``llm_adjustment_pct_points``
    values.

    Intended constraint (per spec):
        ``calibrated_true_prob >= xgboost_prob`` whenever
        ``llm_adjustment_pct_points`` is non-negative.

    The invariant is enforced by ``_validate_invariants`` in
    ``src/calibration_agent/analyzer.py`` (called from ``_coerce_report``):
    a non-negative ``llm_adjustment_pct_points`` combined with
    ``calibrated_true_prob < xgboost_prob`` raises ``ValueError``.

    Test cases:
      (a) positive adjustment + calibrated >= baseline → passes.
      (b) zero adjustment + calibrated == baseline → passes (boundary).
      (c) NEGATIVE adjustment + calibrated < baseline → permitted (per spec
          the invariant only applies for non-negative adjustments).
      (d) positive adjustment + calibrated < baseline → raises ValueError
          (spec violation; this is the bug the validator catches).
    """

    def _market(self) -> Market:
        return Market(
            market_id="mkt-cal-cons",
            title="Constraint test market",
            category="Politics",
            implied_prob=0.50,
            bid_price=0.49,
            ask_price=0.51,
            volume_24h=15_000.0,
            price_history={"1h": 0.0, "6h": 0.0, "24h": 0.0},
            open_interest=20_000.0,
            resolution_date=datetime(2026, 11, 5, tzinfo=timezone.utc),
            rules_text="Resolves to YES if event occurs.",
        )

    def _news_report(self) -> NewsResearchReport:
        return NewsResearchReport(
            timeline=["2026-04-20T10:00:00+00:00 | Headline"],
            key_facts=["Source-backed fact"],
            source_quality_score=7,
            bullish_thesis="Coverage is supportive.",
            bearish_thesis="Counter-headline could reverse.",
            evidence_quality_score=70,
            misinformation_risk_score=15,
            sentiment_score=20,
            key_sources=["https://example.com/a"],
            summary="Coverage is factual and recent.",
        )

    def _agent_with_payload(self, payload: CalibrationReport) -> CalibrationAgent:
        client = FakeClient(FakeResponse(parsed=payload))
        return CalibrationAgent(api_key="test-key", client=client, types_module=FakeTypesModule())

    def _calibration_payload(
        self,
        *,
        xgboost_prob: float,
        llm_adjustment_pct_points: float,
        calibrated_true_prob: float,
    ) -> CalibrationReport:
        return CalibrationReport(
            xgboost_prob=xgboost_prob,
            llm_adjustment_pct_points=llm_adjustment_pct_points,
            calibrated_true_prob=calibrated_true_prob,
            confidence_score=70,
            key_drivers=["A driver"],
            key_uncertainties=["An uncertainty"],
            edge_vs_market=calibrated_true_prob - 0.50,
            action="monitor",
            reasoning="Constraint characterization stub.",
        )

    def test_a_positive_llm_adjustment_yields_calibrated_ge_baseline(self):
        """Case (a): with a strictly-positive LLM adjustment the calibrated
        probability is >= the XGBoost baseline. This is the happy path.
        """
        payload = self._calibration_payload(
            xgboost_prob=0.50,
            llm_adjustment_pct_points=2.5,
            calibrated_true_prob=0.525,
        )
        agent = self._agent_with_payload(payload)

        report = agent.calibrate(
            market=self._market(),
            news_report=self._news_report(),
            xgboost_prob=0.50,
        )

        self.assertGreaterEqual(
            report.calibrated_true_prob,
            report.xgboost_prob,
            msg="Positive adjustment must keep calibrated >= baseline.",
        )

    def test_b_zero_llm_adjustment_yields_calibrated_equal_baseline(self):
        """Case (b): with a zero LLM adjustment the calibrated probability
        equals the baseline (boundary of the >= constraint).
        """
        payload = self._calibration_payload(
            xgboost_prob=0.50,
            llm_adjustment_pct_points=0.0,
            calibrated_true_prob=0.50,
        )
        agent = self._agent_with_payload(payload)

        report = agent.calibrate(
            market=self._market(),
            news_report=self._news_report(),
            xgboost_prob=0.50,
        )

        self.assertGreaterEqual(report.calibrated_true_prob, report.xgboost_prob)
        self.assertAlmostEqual(report.calibrated_true_prob, report.xgboost_prob, places=6)

    def test_c_negative_llm_adjustment_below_baseline_is_permitted(self):
        """Case (c): a *negative* LLM adjustment is permitted to drop the
        calibrated probability below the baseline. The spec invariant only
        constrains non-negative adjustments, so this case must NOT raise.
        """
        payload = self._calibration_payload(
            xgboost_prob=0.50,
            llm_adjustment_pct_points=-3.0,
            calibrated_true_prob=0.47,
        )
        agent = self._agent_with_payload(payload)

        report = agent.calibrate(
            market=self._market(),
            news_report=self._news_report(),
            xgboost_prob=0.50,
        )

        self.assertIsInstance(report, CalibrationReport)
        self.assertLess(report.calibrated_true_prob, report.xgboost_prob)
        self.assertLess(report.llm_adjustment_pct_points, 0.0)

    def test_d_positive_adjustment_below_baseline_raises(self):
        """Case (d): a non-negative adjustment combined with calibrated
        < baseline violates the spec invariant. ``_validate_invariants``
        must raise ``ValueError`` rather than silently passing through
        the invalid LLM payload.
        """
        payload = self._calibration_payload(
            xgboost_prob=0.50,
            llm_adjustment_pct_points=5.0,
            calibrated_true_prob=0.40,
        )
        agent = self._agent_with_payload(payload)

        with self.assertRaisesRegex(ValueError, "calibrated_true_prob"):
            agent.calibrate(
                market=self._market(),
                news_report=self._news_report(),
                xgboost_prob=0.50,
            )


if __name__ == "__main__":
    unittest.main()
