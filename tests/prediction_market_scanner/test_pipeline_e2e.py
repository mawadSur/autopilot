"""End-to-end characterization test for the multi-agent pipeline.

This test wires a single Market through the full pipeline:
  Reddit research -> News research -> Synthesis -> Calibration -> Risk

The goal is to prove serialization compatibility between stages: each
stage's output must validate against the next stage's input via the
actual pydantic ``model_validate`` paths. Gemini calls are stubbed via
the same FakeClient/FakeTypesModule pattern used in the existing
analyzer-level tests (see ``test_synthesis_agent_analyzer`` and
``test_outcome_review_agent``).
"""
from __future__ import annotations

import asyncio
import logging
import unittest
from datetime import datetime, timezone

from calibration_agent.analyzer import CalibrationAgent
from calibration_agent.models import CalibrationReport
from models import Market
from news_research_agent.models import NewsResearchReport
from reddit_research_agent.analyzer import RedditAgent
from reddit_research_agent.models import RedditResearchReport
from risk_management_agent.models import RiskAssessment
from risk_management_agent.risk_engine import RiskCalculator
from synthesis_agent.analyzer import SynthesisAgent
from synthesis_agent.models import SynthesisReport

from orchestrator import _build_risk_assessment


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


class FakeTypesModule:
    def HttpOptions(self, **kwargs):
        return FakeHttpOptions(**kwargs)

    def GenerateContentConfig(self, **kwargs):
        return FakeGenerationConfig(**kwargs)


def _build_reddit_payload() -> RedditResearchReport:
    return RedditResearchReport(
        bullish_thesis="Threads converge on a fresher primary-source signal supporting YES.",
        bearish_thesis="Counter-arguments raise the chance the consensus already prices it in.",
        key_evidence=["A leaked memo cited in r/politics", "A district-level poll snippet"],
        key_assumptions=["The leaked memo is authentic", "The poll is representative"],
        conviction_score=7,
        evidence_quality_score=72,
        misinformation_risk_score=20,
        sentiment_score=25,
        key_sources=["r/politics/u/sourcehound", "r/PoliticalDiscussion"],
        summary="Reddit signal points to a slight underpricing relative to the latest evidence.",
        pricing_assessment="underpriced",
        assessment_reasoning="Fresh primary-source evidence has not yet been reflected in the price.",
    )


def _build_news_payload() -> NewsResearchReport:
    return NewsResearchReport(
        timeline=["2026-04-20T10:00:00+00:00 | Headline A", "2026-04-21T12:00:00+00:00 | Headline B"],
        key_facts=["Source-backed fact A", "Source-backed fact B"],
        source_quality_score=8,
        bullish_thesis="Fresh coverage points to material evidence supporting the YES side.",
        bearish_thesis="Coverage volume is moderate; reversal on a single headline is plausible.",
        evidence_quality_score=78,
        misinformation_risk_score=18,
        sentiment_score=20,
        key_sources=["https://example.com/a", "https://example.com/b"],
        summary="Coverage is factual, recent, and moderately supportive of YES.",
    )


def _build_synthesis_payload() -> SynthesisReport:
    return SynthesisReport(
        implied_probability=0.42,
        narrative_direction="bullish",
        has_unique_evidence=True,
        reasons_market_is_right=[
            "The market has already absorbed the polling trend.",
            "Resolution criteria are clear and conservative.",
            "Sample size of evidence is still modest.",
        ],
        reasons_market_is_wrong=[
            "Reddit surfaced fresh primary-source evidence not in the news cycle.",
            "Recent news shifted the trajectory faster than price followed.",
            "Crowd attention may be on irrelevant headlines.",
        ],
        verdict="stale",
        explanation="Latest evidence appears to outpace the current market price.",
    )


def _build_calibration_payload() -> CalibrationReport:
    return CalibrationReport(
        xgboost_prob=0.45,
        llm_adjustment_pct_points=2.0,
        calibrated_true_prob=0.47,
        confidence_score=70,
        key_drivers=["Reddit surfaced a credible underpricing signal.", "News coverage is consistent."],
        key_uncertainties=["Sample size of fresh evidence is small.", "Late-breaking headlines could reverse."],
        edge_vs_market=0.05,
        action="paper-trade candidate",
        reasoning="Modest upward adjustment supported by consistent qualitative evidence.",
    )


def _build_market() -> Market:
    return Market(
        market_id="mkt-e2e-1",
        title="Will Candidate X win?",
        category="Politics",
        implied_prob=0.42,
        bid_price=0.41,
        ask_price=0.43,
        volume_24h=25000.0,
        price_history={"1h": 0.005, "6h": 0.01, "24h": 0.02},
        open_interest=40000.0,
        resolution_date=datetime(2026, 11, 5, tzinfo=timezone.utc),
        rules_text="Resolves to YES if Candidate X wins.",
    )


class PipelineEndToEndTests(unittest.TestCase):
    """Characterize that all stages serialize to the next stage's schema."""

    def test_full_pipeline_produces_risk_assessment_without_serialization_breakage(self):
        market = _build_market()

        # --- Stage 1: Reddit research ---
        reddit_payload = _build_reddit_payload()
        reddit_agent = RedditAgent(
            api_key="test-key",
            client=FakeClient(FakeResponse(parsed=reddit_payload)),
            types_module=FakeTypesModule(),
        )

        # --- Stage 2: News research ---
        news_payload = _build_news_payload()

        # --- Stage 3: Synthesis ---
        synthesis_payload = _build_synthesis_payload()
        synthesis_agent = SynthesisAgent(
            "test-key",
            client=FakeClient(FakeResponse(parsed=synthesis_payload)),
            types_module=FakeTypesModule(),
        )

        # --- Stage 4: Calibration ---
        calibration_payload = _build_calibration_payload()
        calibration_agent = CalibrationAgent(
            api_key="test-key",
            client=FakeClient(FakeResponse(parsed=calibration_payload)),
            types_module=FakeTypesModule(),
        )

        # --- Stage 5: Risk ---
        risk_calculator = RiskCalculator()

        # Capture WARNING+ records via a custom handler attached to the root logger.
        warning_records: list[logging.LogRecord] = []

        class _WarningPlusHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.levelno >= logging.WARNING:
                    warning_records.append(record)

        capture_handler = _WarningPlusHandler(level=logging.WARNING)
        root_logger = logging.getLogger()
        root_logger.addHandler(capture_handler)
        try:
            # 1. Reddit -> RedditResearchReport
            reddit_report = asyncio.run(
                reddit_agent.analyze_discussion(
                    market_title=market.title,
                    implied_prob=market.implied_prob,
                    reddit_context=(
                        "REDDIT DISCUSSION CONTEXT\n"
                        "THREAD 1 | r/politics | score=120 | comments=40 | author=u/example\n"
                        "TITLE: Will Candidate X win?\n"
                        "TOP COMMENTS:\n"
                        "- u/sourcehound | score=80: A leaked memo backs YES."
                    ),
                )
            )
            self.assertIsInstance(reddit_report, RedditResearchReport)
            # Cross-stage validation: round-trip Reddit through model_validate.
            RedditResearchReport.model_validate(reddit_report.model_dump())

            # 2. News -> NewsResearchReport (already shaped; round-trip to exercise schema).
            news_report = NewsResearchReport.model_validate(news_payload.model_dump())
            self.assertIsInstance(news_report, NewsResearchReport)

            # 3. Synthesis -> SynthesisReport
            synthesis_report = synthesis_agent.synthesize_edge(
                market_data={
                    "market_id": market.market_id,
                    "title": market.title,
                    "category": market.category,
                    "implied_probability": market.implied_prob,
                },
                social_context=reddit_report.summary or "Reddit discussion summary",
                news_context=news_report.summary,
            )
            self.assertIsInstance(synthesis_report, SynthesisReport)
            SynthesisReport.model_validate(synthesis_report.model_dump())

            # 4. Calibration -> CalibrationReport (consumes Reddit + News reports)
            calibration_report = calibration_agent.calibrate(
                market=market,
                news_report=news_report,
                xgboost_prob=0.45,
                reddit_report=reddit_report,
            )
            self.assertIsInstance(calibration_report, CalibrationReport)
            CalibrationReport.model_validate(calibration_report.model_dump())

            # 5. Risk -> RiskAssessment (consumes Calibration)
            risk_metrics = risk_calculator.calculate_base_metrics(
                market=market,
                calibrated_true_prob=calibration_report.calibrated_true_prob,
                bankroll=10_000.0,
            )
            risk_assessment = _build_risk_assessment(
                risk_metrics=risk_metrics,
                calibration=calibration_report,
            )
        finally:
            root_logger.removeHandler(capture_handler)

        self.assertIsInstance(risk_assessment, RiskAssessment)
        # Round-trip the final assessment to confirm schema compliance.
        RiskAssessment.model_validate(risk_assessment.model_dump())

        # All required RiskAssessment fields are present and well-typed.
        self.assertIsInstance(risk_assessment.allow_trade, bool)
        self.assertIsInstance(risk_assessment.simulated_position_size_pct, float)
        self.assertIsInstance(risk_assessment.max_loss_if_wrong, float)
        self.assertIsInstance(risk_assessment.expected_value_estimate, float)
        self.assertGreaterEqual(len(risk_assessment.top_risk_reasons), 1)
        self.assertIsInstance(risk_assessment.kill_switch_triggered, bool)
        self.assertIn(
            risk_assessment.final_recommendation,
            {"reject", "small", "medium", "high-conviction paper trade"},
        )
        self.assertTrue(risk_assessment.risk_logic_summary)

        # No WARNING+ records should have been emitted by the pipeline itself.
        self.assertEqual(
            warning_records,
            [],
            msg=f"Pipeline emitted unexpected WARNING+ records: {[r.getMessage() for r in warning_records]}",
        )


if __name__ == "__main__":
    unittest.main()
