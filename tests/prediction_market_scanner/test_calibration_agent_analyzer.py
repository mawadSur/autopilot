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


if __name__ == "__main__":
    unittest.main()
