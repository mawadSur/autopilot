"""End-to-end tests for the multi-market async fan-out path in ``analyze_top_markets``.

These tests intentionally drive *three* markets through the orchestrator
simultaneously to verify the parallel fan-out is doing actual work per market
(not silently collapsing to one), and that a single failing market does not
poison the rest of the batch.

Stubs follow the same shape as ``test_orchestrator.py``: each agent records its
own per-call list and returns deterministic, *per-market* responses so we can
prove the orchestrator wired the right inputs to the right outputs for each
market.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from calibration_agent.models import CalibrationReport
from models import Market
from orchestrator import analyze_top_markets
from pydantic import BaseModel
from reddit_research_agent.models import RedditResearchReport


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
    calls: list[dict] = []

    def __init__(self, search_query, subreddits=None):
        self.search_query = search_query
        self.subreddits = list(subreddits or [])
        type(self).calls.append({"query": search_query, "subreddits": self.subreddits})

    def fetch_threads(self):
        return f"reddit::{self.search_query}"


class FakeNewsAggregator:
    calls: list[dict] = []

    def __init__(self, search_query):
        self.search_query = search_query
        type(self).calls.append({"query": search_query})

    def fetch_news_context(self):
        return f"news::{self.search_query}"


class PerMarketRedditAgent:
    """Returns a different RedditResearchReport per market title."""

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
        # Use the market title as a deterministic per-market signal so we can
        # assert that the calibration agent saw the right report per market.
        return RedditResearchReport(
            bullish_thesis=f"Bull case for {market_title}.",
            bearish_thesis=f"Bear case for {market_title}.",
            key_evidence=[f"Evidence for {market_title}"],
            key_assumptions=[f"Assumption for {market_title}"],
            conviction_score=7,
            evidence_quality_score=80,
            misinformation_risk_score=15,
            sentiment_score=30,
            key_sources=[f"u/source-{market_title.replace(' ', '-')}"],
            summary=f"Reddit summary for {market_title}.",
            pricing_assessment="underpriced",
            assessment_reasoning=f"Reasoning for {market_title}.",
        )


class PerMarketNewsAgent:
    """Returns a different NewsResearchReport per market title."""

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
            timeline=[f"2026-04-01: Headline about {market_title}"],
            key_facts=[f"Fact about {market_title}"],
            source_quality_score=8,
            bullish_thesis=f"News bull case for {market_title}.",
            bearish_thesis=f"News bear case for {market_title}.",
            evidence_quality_score=70,
            misinformation_risk_score=20,
            sentiment_score=25,
            key_sources=[f"https://example.com/{market_title.replace(' ', '-')}"],
            summary=f"News summary for {market_title}.",
        )


class PerMarketCalibrationAgent:
    """Returns a different CalibrationReport per market_id."""

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
        # Anchor the calibrated_true_prob on the implied probability to give us
        # a per-market deterministic signal we can assert on.
        action_by_id = {
            "mkt-1": "monitor",
            "mkt-2": "paper-trade candidate",
            "mkt-3": "monitor",
        }
        action = action_by_id.get(market.market_id, "monitor")
        return CalibrationReport(
            xgboost_prob=xgboost_prob,
            llm_adjustment_pct_points=2.5 if action == "paper-trade candidate" else 0.5,
            calibrated_true_prob=market.implied_prob + 0.05,
            confidence_score=80 if action == "paper-trade candidate" else 60,
            key_drivers=[f"Driver for {market.market_id}"],
            key_uncertainties=[f"Uncertainty for {market.market_id}"],
            edge_vs_market=0.07 if action == "paper-trade candidate" else 0.01,
            action=action,
            reasoning=f"Calibration for {market.title}.",
        )


def _three_markets() -> list[Market]:
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
        Market(
            market_id="mkt-3",
            title="Market Gamma",
            category="Politics",
            implied_prob=0.33,
            bid_price=0.32,
            ask_price=0.34,
            volume_24h=15000.0,
            price_history={"1h": 0.005, "6h": 0.01, "24h": 0.02},
            open_interest=18000.0,
            resolution_date=datetime(2026, 10, 10, tzinfo=timezone.utc),
            rules_text="Resolves to Yes if Gamma occurs.",
        ),
    ]


def _build_scan_results_three(*, fetch_markets_fn=None, **kwargs):
    """Returns scanner rows for all three markets in fixed order."""
    available_markets = list(
        fetch_markets_fn(
            min_volume_24h=kwargs.get("min_volume_24h"),
            page_size=kwargs.get("page_size"),
            max_pages=kwargs.get("max_pages"),
        )
    )
    rows = []
    priorities = {"mkt-1": 91, "mkt-2": 98, "mkt-3": 85}
    for market in available_markets:
        rows.append(
            {
                "market_id": market.market_id,
                "title": market.title,
                "category": market.category,
                "implied_prob": market.implied_prob,
                "research_priority": priorities.get(market.market_id, 50),
            }
        )
    return rows


class MultiMarketParallelFanOutTests(unittest.IsolatedAsyncioTestCase):
    """Test 1 — multi-market parallel async fan-out actually runs per-market."""

    def setUp(self):
        FakeRedditDiver.calls = []
        FakeNewsAggregator.calls = []
        self.baseline_calls: list[Market] = []

    def _xgboost_baseline(self, market: Market) -> float:
        # Per-market deterministic baselines so we can prove the calibration
        # agent received the right baseline per market.
        self.baseline_calls.append(market)
        return {"mkt-1": 0.41, "mkt-2": 0.58, "mkt-3": 0.35}.get(market.market_id, 0.5)

    async def test_three_markets_run_in_parallel_through_full_fan_out(self):
        markets = _three_markets()
        reddit_agent = PerMarketRedditAgent()
        news_agent = PerMarketNewsAgent()
        calibration_agent = PerMarketCalibrationAgent()

        def fake_fetch_markets_fn(**_):
            return markets

        results = await analyze_top_markets(
            top_n=3,
            build_scan_results_fn=_build_scan_results_three,
            fetch_markets_fn=fake_fetch_markets_fn,
            reddit_diver_cls=FakeRedditDiver,
            reddit_agent=reddit_agent,
            news_aggregator_cls=FakeNewsAggregator,
            news_agent=news_agent,
            calibration_agent=calibration_agent,
            xgboost_probability_fn=self._xgboost_baseline,
            subreddits=["politics"],
        )

        # All 3 markets land in the returned _CalibrationResults list.
        self.assertEqual(len(results), 3)

        # Each agent was called exactly 3 times (one per market) — proves
        # the fan-out is real, not a collapsed single call.
        self.assertEqual(len(reddit_agent.calls), 3)
        self.assertEqual(len(news_agent.calls), 3)
        self.assertEqual(len(calibration_agent.calls), 3)
        self.assertEqual(len(FakeRedditDiver.calls), 3)
        self.assertEqual(len(FakeNewsAggregator.calls), 3)
        self.assertEqual(len(self.baseline_calls), 3)

        # No failures should be recorded for the happy path.
        self.assertEqual(getattr(results, "failed_markets", None), [])

        # The returned object IS-A list — preserves backward compat for callers
        # that iterate / index / list() it.
        self.assertIsInstance(results, list)
        as_list = list(results)
        self.assertEqual(len(as_list), 3)
        for entry in as_list:
            self.assertIsInstance(entry, CalibrationReport)

        # Per-market deterministic responses prove the orchestrator wired
        # the right Reddit/News/baseline inputs into the right calibration
        # output for each market (not just identical responses fanned out).
        seen_market_ids = {call["market"].market_id for call in calibration_agent.calls}
        self.assertEqual(seen_market_ids, {"mkt-1", "mkt-2", "mkt-3"})

        for call in calibration_agent.calls:
            mkt_id = call["market"].market_id
            mkt_title = call["market"].title
            # The reddit/news report attached to each calibration call should
            # carry the per-market signal injected by PerMarket{Reddit,News}Agent.
            self.assertIn(mkt_title, call["reddit_report"].summary)
            self.assertIn(mkt_title, call["news_report"].summary)

            expected_baseline = {"mkt-1": 0.41, "mkt-2": 0.58, "mkt-3": 0.35}[mkt_id]
            self.assertEqual(call["xgboost_prob"], expected_baseline)


class MultiMarketFailureIsolationTests(unittest.IsolatedAsyncioTestCase):
    """Test 2 — a single failing market doesn't poison the batch."""

    def setUp(self):
        FakeRedditDiver.calls = []
        FakeNewsAggregator.calls = []
        self.baseline_calls: list[Market] = []

    def _xgboost_baseline(self, market: Market) -> float:
        self.baseline_calls.append(market)
        return {"mkt-1": 0.41, "mkt-2": 0.58, "mkt-3": 0.35}.get(market.market_id, 0.5)

    async def test_second_market_failure_does_not_kill_batch(self):
        markets = _three_markets()
        reddit_agent = PerMarketRedditAgent()
        news_agent = PerMarketNewsAgent()

        class CalibrationFailsOnSecondMarket(PerMarketCalibrationAgent):
            def calibrate(self, market, news_report, xgboost_prob, reddit_report=None):
                if market.market_id == "mkt-2":
                    raise ValueError("synthetic mid-batch calibration failure")
                return super().calibrate(
                    market,
                    news_report,
                    xgboost_prob,
                    reddit_report=reddit_report,
                )

        calibration_agent = CalibrationFailsOnSecondMarket()

        def fake_fetch_markets_fn(**_):
            return markets

        with self.assertLogs("orchestrator", level="WARNING") as log_capture:
            results = await analyze_top_markets(
                top_n=3,
                build_scan_results_fn=_build_scan_results_three,
                fetch_markets_fn=fake_fetch_markets_fn,
                reddit_diver_cls=FakeRedditDiver,
                reddit_agent=reddit_agent,
                news_aggregator_cls=FakeNewsAggregator,
                news_agent=news_agent,
                calibration_agent=calibration_agent,
                xgboost_probability_fn=self._xgboost_baseline,
                subreddits=["politics"],
            )

        # Markets 1 and 3 still complete — the failure didn't kill the batch.
        self.assertEqual(len(results), 2)
        completed_ids = {
            call["market"].market_id
            for call in calibration_agent.calls
            if call["market"].market_id != "mkt-2"
        }
        self.assertEqual(completed_ids, {"mkt-1", "mkt-3"})

        # The failed market is captured in `failed_markets` with the
        # exception type name + message.
        failed = getattr(results, "failed_markets", None)
        self.assertIsNotNone(failed)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["market_id"], "mkt-2")
        self.assertIn("ValueError", failed[0]["error"])
        self.assertIn("synthetic mid-batch calibration failure", failed[0]["error"])

        # A WARNING was emitted naming the failed market id.
        warning_messages = [
            record.getMessage()
            for record in log_capture.records
            if record.levelname == "WARNING"
        ]
        self.assertTrue(
            any("mkt-2" in msg for msg in warning_messages),
            msg=f"expected 'mkt-2' in WARNING logs, got {warning_messages!r}",
        )


if __name__ == "__main__":
    unittest.main()
