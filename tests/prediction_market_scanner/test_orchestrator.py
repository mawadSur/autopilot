import io
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from calibration_agent.models import CalibrationReport
from models import Market
from orchestrator import (
    analyze_top_markets,
    build_news_search_query,
    build_reddit_search_query,
    print_alpha_assessment_table,
    run_final_risk_gate,
)
from pydantic import BaseModel
from reddit_research_agent.models import RedditResearchReport
from risk_management_agent.risk_engine import RiskCalculator


class FakeNewsResearchReport(BaseModel):
    timeline: list[str]
    key_facts: list[str]
    source_quality_score: int
    bullish_thesis: str
    bearish_thesis: str
    evidence_quality_score: int
    misinformation_risk_score: int
    sentiment_score: int
    key_sources: list[str]
    summary: str


class FakeRedditDiver:
    calls = []

    def __init__(self, search_query, subreddits=None):
        self.search_query = search_query
        self.subreddits = list(subreddits or [])
        type(self).calls.append({"query": search_query, "subreddits": self.subreddits})

    def fetch_threads(self):
        return f"reddit::{self.search_query}"


class FakeNewsAggregator:
    calls = []

    def __init__(self, search_query):
        self.search_query = search_query
        type(self).calls.append({"query": search_query})

    def fetch_news_context(self):
        return f"news::{self.search_query}"


class FakeRedditAgent:
    def __init__(self):
        self.calls = []

    async def analyze_discussion(self, market_title, implied_prob, reddit_context, **_):
        self.calls.append(
            {
                "market_title": market_title,
                "implied_prob": implied_prob,
                "reddit_context": reddit_context,
            }
        )
        return RedditResearchReport(
            bullish_thesis="Primary-source upside case.",
            bearish_thesis="Consensus may already price this in.",
            key_evidence=["Reddit signal"],
            key_assumptions=["The signal is real"],
            conviction_score=7,
            evidence_quality_score=80,
            misinformation_risk_score=15,
            sentiment_score=30,
            key_sources=["u/sourcehound"],
            summary="Reddit threads converge on a fresher primary-source signal.",
            pricing_assessment="underpriced",
            assessment_reasoning="The discussion contains fresher evidence than the market move.",
        )


class FakeNewsAgent:
    def __init__(self):
        self.calls = []

    async def analyze_news(self, market_title, implied_prob, news_context, **_):
        self.calls.append(
            {
                "market_title": market_title,
                "implied_prob": implied_prob,
                "news_context": news_context,
            }
        )
        return FakeNewsResearchReport(
            timeline=["2026-04-01: Headline"],
            key_facts=["Source-backed fact"],
            source_quality_score=8,
            bullish_thesis="Headline supports the YES side resolving.",
            bearish_thesis="Coverage is thin and could reverse on a single late headline.",
            evidence_quality_score=70,
            misinformation_risk_score=20,
            sentiment_score=25,
            key_sources=["https://example.com/a"],
            summary="Coverage is factual and recent.",
        )


class FakeCalibrationAgent:
    def __init__(self):
        self.calls = []

    def calibrate(self, market, news_report, xgboost_prob, reddit_report=None):
        self.calls.append(
            {
                "market": market,
                "news_report": news_report,
                "reddit_report": reddit_report,
                "xgboost_prob": xgboost_prob,
            }
        )
        action = "paper-trade candidate" if market.market_id == "mkt-2" else "monitor"
        return CalibrationReport(
            xgboost_prob=xgboost_prob,
            llm_adjustment_pct_points=2.5 if action == "paper-trade candidate" else 0.5,
            calibrated_true_prob=0.605 if action == "paper-trade candidate" else 0.415,
            confidence_score=82 if action == "paper-trade candidate" else 61,
            key_drivers=[
                "Reddit surfaced a credible leading signal.",
                "Recent news flow is directionally supportive.",
            ],
            key_uncertainties=[
                "The evidence set is still small.",
                "Late headlines could reverse the setup.",
            ],
            edge_vs_market=0.085 if action == "paper-trade candidate" else 0.005,
            action=action,
            reasoning=f"Calibration for {market.title}.",
        )


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        FakeRedditDiver.calls = []
        FakeNewsAggregator.calls = []
        self.baseline_calls = []

    def _markets(self):
        return [
            Market(
                market_id="mkt-1",
                title="Market Alpha",
                category="Politics",
                implied_prob=0.41,
                bid_price=0.40,
                ask_price=0.42,
                volume_24h=12000.0,
                price_history={"1h": 0.01, "6h": 0.02, "24h": 0.05},
                open_interest=25000.0,
                resolution_date=datetime(2026, 11, 5, tzinfo=timezone.utc),
                rules_text="Resolves to Yes if Alpha occurs.",
            ),
            Market(
                market_id="mkt-2",
                title="Market Beta",
                category="Politics",
                implied_prob=0.52,
                bid_price=0.51,
                ask_price=0.53,
                volume_24h=22000.0,
                price_history={"1h": 0.02, "6h": 0.04, "24h": 0.08},
                open_interest=40000.0,
                resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
                rules_text="Resolves to Yes if Beta occurs.",
            ),
        ]

    def _baseline_probability(self, market):
        self.baseline_calls.append(market)
        if market.market_id == "mkt-2":
            return 0.58
        return 0.41

    def test_build_query_helpers_include_title_category_and_suffix(self):
        market = self._markets()[0]
        self.assertEqual(build_reddit_search_query(market), "Market Alpha Politics reddit discussion")
        self.assertEqual(build_news_search_query(market), "Market Alpha Politics news")

    async def test_analyze_top_markets_runs_calibration_pipeline(self):
        markets = self._markets()
        reddit_agent = FakeRedditAgent()
        news_agent = FakeNewsAgent()
        calibration_agent = FakeCalibrationAgent()

        def fake_fetch_markets_fn(**_):
            return markets

        def fake_build_scan_results_fn(*, fetch_markets_fn=None, **kwargs):
            available_markets = list(
                fetch_markets_fn(
                    min_volume_24h=kwargs.get("min_volume_24h"),
                    page_size=kwargs.get("page_size"),
                    max_pages=kwargs.get("max_pages"),
                )
            )
            return [
                {
                    "market_id": available_markets[1].market_id,
                    "title": available_markets[1].title,
                    "category": available_markets[1].category,
                    "implied_prob": available_markets[1].implied_prob,
                    "research_priority": 98,
                },
                {
                    "market_id": available_markets[0].market_id,
                    "title": available_markets[0].title,
                    "category": available_markets[0].category,
                    "implied_prob": available_markets[0].implied_prob,
                    "research_priority": 91,
                },
            ]

        calibrations = await analyze_top_markets(
            top_n=2,
            build_scan_results_fn=fake_build_scan_results_fn,
            fetch_markets_fn=fake_fetch_markets_fn,
            reddit_diver_cls=FakeRedditDiver,
            reddit_agent=reddit_agent,
            news_aggregator_cls=FakeNewsAggregator,
            news_agent=news_agent,
            calibration_agent=calibration_agent,
            xgboost_probability_fn=self._baseline_probability,
            subreddits=["politics"],
        )

        self.assertEqual(len(calibrations), 2)
        self.assertEqual(FakeRedditDiver.calls[0], {"query": "Market Beta Politics reddit discussion", "subreddits": ["politics"]})
        self.assertEqual(FakeNewsAggregator.calls[0], {"query": "Market Beta Politics news"})
        self.assertEqual(reddit_agent.calls[0]["reddit_context"], "reddit::Market Beta Politics reddit discussion")
        self.assertEqual(news_agent.calls[0]["news_context"], "news::Market Beta Politics news")
        self.assertEqual(self.baseline_calls[0].market_id, "mkt-2")
        self.assertAlmostEqual(self.baseline_calls[0].spread, 0.020000000000000018)
        self.assertEqual(calibration_agent.calls[0]["xgboost_prob"], 0.58)
        self.assertEqual(calibration_agent.calls[0]["market"].market_id, "mkt-2")
        self.assertEqual(calibration_agent.calls[0]["reddit_report"].pricing_assessment, "underpriced")
        self.assertEqual(calibration_agent.calls[0]["news_report"].source_quality_score, 8)
        self.assertEqual(calibrations[0].action, "paper-trade candidate")
        self.assertEqual(calibrations[1].action, "monitor")

    async def test_analyze_top_markets_uses_default_news_fetcher_and_agent(self):
        markets = self._markets()
        calibration_agent = FakeCalibrationAgent()

        def fake_fetch_markets_fn(**_):
            return markets

        def fake_build_scan_results_fn(*, fetch_markets_fn=None, **kwargs):
            available_markets = list(fetch_markets_fn(**kwargs))
            return [
                {
                    "market_id": available_markets[1].market_id,
                    "title": available_markets[1].title,
                    "category": available_markets[1].category,
                    "implied_prob": available_markets[1].implied_prob,
                    "research_priority": 98,
                }
            ]

        fake_feed = type(
            "Feed",
            (),
            {
                "entries": [
                    {
                        "title": "Headline A",
                        "link": "https://example.com/a",
                        "published": "2026-04-20T10:00:00+00:00",
                        "summary": "Summary A",
                    },
                    {
                        "title": "Headline B",
                        "link": "https://example.com/b",
                        "published": "2026-04-21T11:00:00+00:00",
                        "summary": "Summary B",
                    },
                ]
            },
        )()

        with patch("news_research_agent.fetcher.feedparser.parse", return_value=fake_feed):
            calibrations = await analyze_top_markets(
                top_n=1,
                build_scan_results_fn=fake_build_scan_results_fn,
                fetch_markets_fn=fake_fetch_markets_fn,
                reddit_diver_cls=FakeRedditDiver,
                reddit_agent=FakeRedditAgent(),
                calibration_agent=calibration_agent,
                xgboost_probability_fn=self._baseline_probability,
            )

        self.assertEqual(len(calibrations), 1)
        self.assertEqual(calibration_agent.calls[0]["news_report"].timeline[0], "2026-04-20T10:00:00+00:00 | Headline A")
        self.assertGreaterEqual(calibration_agent.calls[0]["news_report"].source_quality_score, 4)

    async def test_print_alpha_assessment_table_renders_summary_table(self):
        markets = self._markets()

        def fake_fetch_markets_fn(**_):
            return markets

        def fake_build_scan_results_fn(*, fetch_markets_fn=None, **kwargs):
            available_markets = list(fetch_markets_fn(**kwargs))
            return [
                {
                    "market_id": available_markets[1].market_id,
                    "title": available_markets[1].title,
                    "category": available_markets[1].category,
                    "implied_prob": available_markets[1].implied_prob,
                    "research_priority": 98,
                }
            ]

        calibrations = await analyze_top_markets(
            top_n=1,
            build_scan_results_fn=fake_build_scan_results_fn,
            fetch_markets_fn=fake_fetch_markets_fn,
            reddit_diver_cls=FakeRedditDiver,
            reddit_agent=FakeRedditAgent(),
            news_aggregator_cls=FakeNewsAggregator,
            news_agent=FakeNewsAgent(),
            calibration_agent=FakeCalibrationAgent(),
            xgboost_probability_fn=self._baseline_probability,
        )

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rendered = print_alpha_assessment_table(calibrations)
        output = buffer.getvalue()

        self.assertIn("Market Beta", rendered)
        self.assertIn("Alpha Assessment", rendered)
        self.assertIn("XGBoost Prob", rendered)
        self.assertIn("LLM Adj", rendered)
        self.assertIn("Calibrated Prob", rendered)
        self.assertIn("Edge vs Market", rendered)
        self.assertIn("paper-trade candidate", rendered)
        self.assertIn("Market Beta", rendered)
        self.assertIn("\033[1m\033[32m", rendered)
        self.assertIn(rendered, output)

    async def test_run_final_risk_gate_writes_execution_log_and_assessment(self):
        market = self._markets()[1]
        calibration = CalibrationReport(
            xgboost_prob=0.58,
            llm_adjustment_pct_points=2.5,
            calibrated_true_prob=0.62,
            confidence_score=84,
            key_drivers=["Signal"],
            key_uncertainties=["Uncertainty"],
            edge_vs_market=0.10,
            action="paper-trade candidate",
            reasoning="Strong setup.",
        )
        scanner_row = {
            "market_id": market.market_id,
            "title": market.title,
            "category": market.category,
            "implied_prob": market.implied_prob,
            "research_priority": 99,
        }

        reddit_report = RedditResearchReport(
            bullish_thesis="Bullish thesis.",
            bearish_thesis="Bearish thesis.",
            key_evidence=["Evidence"],
            key_assumptions=["Assumption"],
            conviction_score=7,
            evidence_quality_score=82,
            misinformation_risk_score=11,
            sentiment_score=27,
            key_sources=["u/sourcehound"],
            summary="Bullish sentiment",
            pricing_assessment="underpriced",
            assessment_reasoning="Reasoning.",
        )
        news_report = FakeNewsResearchReport(
            timeline=["2026-04-20: Headline"],
            key_facts=["Fact"],
            source_quality_score=8,
            bullish_thesis="Up.",
            bearish_thesis="Down.",
            evidence_quality_score=64,
            misinformation_risk_score=19,
            sentiment_score=33,
            key_sources=["https://example.com/a"],
            summary="Supportive headlines",
        )

        execution = run_final_risk_gate(
            calibration=calibration,
            market=market,
            scanner_row=scanner_row,
            reddit_report=reddit_report,
            news_report=news_report,
            bankroll=10_000.0,
            risk_calculator=RiskCalculator(),
        )

        self.assertIn("risk_assessment", execution)
        self.assertIn("risk_metrics", execution)
        self.assertTrue(execution["log_path"].exists())
        self.assertIn("scanner", execution["event_payload"])
        self.assertIn("research", execution["event_payload"])
        self.assertIn("calibration", execution["event_payload"])
        self.assertIn("risk", execution["event_payload"])

        # market_outcome is the XGBoost training label ("did the market
        # resolve YES?"). It must start as None and only be filled in once
        # the market settles, distinct from final_outcome (the side-aware
        # "did the trade win?" answer).
        self.assertIn("market_outcome", execution["event_payload"])
        self.assertIsNone(execution["event_payload"]["market_outcome"])
        self.assertIsNone(execution["event_payload"]["final_outcome"])

        # features_window must be populated at decision time so that, once
        # trades settle, the (features, outcome) pairs train the real XGBoost
        # baseline. Asserts the slot stops being None.
        features_window = execution["event_payload"]["features_window"]
        self.assertIsInstance(features_window, dict)
        for column in (
            "implied_prob",
            "spread",
            "volume_24h",
            "open_interest",
            "days_to_resolution",
            "price_change_1h",
            "price_change_6h",
            "price_change_24h",
            "captured_at_utc",
        ):
            self.assertIn(column, features_window, msg=f"features_window missing column {column!r}")
        self.assertEqual(features_window["implied_prob"], market.implied_prob)

        # Research-signal features must also be merged into features_window so
        # the calibration model can learn from sentiment / evidence / misinfo.
        for column in (
            "news_sentiment_score",
            "news_evidence_quality_score",
            "news_misinformation_risk_score",
            "reddit_sentiment_score",
            "reddit_evidence_quality_score",
            "reddit_misinformation_risk_score",
        ):
            self.assertIn(
                column,
                features_window,
                msg=f"features_window missing research column {column!r}",
            )
        self.assertEqual(features_window["news_sentiment_score"], 33.0)
        self.assertEqual(features_window["news_evidence_quality_score"], 64.0)
        self.assertEqual(features_window["news_misinformation_risk_score"], 19.0)
        self.assertEqual(features_window["reddit_sentiment_score"], 27.0)
        self.assertEqual(features_window["reddit_evidence_quality_score"], 82.0)
        self.assertEqual(features_window["reddit_misinformation_risk_score"], 11.0)

        # Canonical schema: orchestrator-written logs must self-identify via
        # the ``source`` field so build_dataset.py can fork on provenance
        # (orchestrator vs backfill vs shadow). ``notes`` stays None for the
        # orchestrator path; only backfill populates it with caveats.
        self.assertIn("source", execution["event_payload"])
        self.assertEqual(execution["event_payload"]["source"], "orchestrator")
        self.assertIn("notes", execution["event_payload"])
        self.assertIsNone(execution["event_payload"]["notes"])

        # PnL tracking: decision-time fields must be populated (entry_price /
        # position_size_usd / max_loss_usd) so settlement can derive realized
        # PnL. exit_price + realized_pnl_usd start as None and are filled by
        # mark_trade_settled.py.
        payload = execution["event_payload"]
        for key in (
            "entry_price",
            "position_size_usd",
            "exit_price",
            "realized_pnl_usd",
            "max_loss_usd",
        ):
            self.assertIn(key, payload, msg=f"event_payload missing PnL key {key!r}")
        self.assertIsNone(payload["exit_price"])
        self.assertIsNone(payload["realized_pnl_usd"])
        self.assertGreater(payload["entry_price"], 0.0)
        self.assertGreaterEqual(payload["position_size_usd"], 0.0)
        self.assertGreaterEqual(payload["max_loss_usd"], 0.0)
        execution["log_path"].unlink(missing_ok=True)

    async def test_analyze_top_markets_isolates_per_market_failures(self):
        markets = self._markets()
        reddit_agent = FakeRedditAgent()
        news_agent = FakeNewsAgent()

        class FailingForOneMarketCalibrationAgent(FakeCalibrationAgent):
            def calibrate(self, market, news_report, xgboost_prob, reddit_report=None):
                if market.market_id == "mkt-1":
                    raise RuntimeError("synthetic calibration failure")
                return super().calibrate(
                    market,
                    news_report,
                    xgboost_prob,
                    reddit_report=reddit_report,
                )

        calibration_agent = FailingForOneMarketCalibrationAgent()

        def fake_fetch_markets_fn(**_):
            return markets

        def fake_build_scan_results_fn(*, fetch_markets_fn=None, **kwargs):
            available_markets = list(
                fetch_markets_fn(
                    min_volume_24h=kwargs.get("min_volume_24h"),
                    page_size=kwargs.get("page_size"),
                    max_pages=kwargs.get("max_pages"),
                )
            )
            return [
                {
                    "market_id": available_markets[1].market_id,
                    "title": available_markets[1].title,
                    "category": available_markets[1].category,
                    "implied_prob": available_markets[1].implied_prob,
                    "research_priority": 98,
                },
                {
                    "market_id": available_markets[0].market_id,
                    "title": available_markets[0].title,
                    "category": available_markets[0].category,
                    "implied_prob": available_markets[0].implied_prob,
                    "research_priority": 91,
                },
            ]

        with self.assertLogs("orchestrator", level="WARNING") as log_capture:
            calibrations = await analyze_top_markets(
                top_n=2,
                build_scan_results_fn=fake_build_scan_results_fn,
                fetch_markets_fn=fake_fetch_markets_fn,
                reddit_diver_cls=FakeRedditDiver,
                reddit_agent=reddit_agent,
                news_aggregator_cls=FakeNewsAggregator,
                news_agent=news_agent,
                calibration_agent=calibration_agent,
                xgboost_probability_fn=self._baseline_probability,
                subreddits=["politics"],
            )

        # The healthy market still calibrates and is returned.
        self.assertEqual(len(calibrations), 1)
        self.assertEqual(calibrations[0].action, "paper-trade candidate")

        # The failing market is captured in the new failed_markets attribute.
        failed_markets = getattr(calibrations, "failed_markets", None)
        self.assertIsNotNone(failed_markets)
        self.assertEqual(len(failed_markets), 1)
        self.assertEqual(failed_markets[0]["market_id"], "mkt-1")
        self.assertIn("RuntimeError", failed_markets[0]["error"])
        self.assertIn("synthetic calibration failure", failed_markets[0]["error"])

        # The failure is logged at WARNING with the market id.
        warning_records = [
            record for record in log_capture.records if record.levelname == "WARNING"
        ]
        self.assertTrue(
            any("mkt-1" in record.getMessage() for record in warning_records),
            msg=f"expected mkt-1 in WARNING logs, got {[r.getMessage() for r in warning_records]}",
        )


if __name__ == "__main__":
    unittest.main()
