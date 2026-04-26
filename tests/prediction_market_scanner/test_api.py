"""Happy-path + bad-input tests for src/api/main.py.

Exercises every endpoint exposed by the prediction-market FastAPI without
hitting any real LLM or network. Uses ``fastapi.testclient.TestClient`` plus
``unittest.mock`` to stub the scanner / research / calibration / risk agents.

Filesystem isolation: each test patches ``AUTOPILOT_TRADE_STORE`` to a
``tempfile.TemporaryDirectory`` so trade logs and ``performance_audit.json``
never bleed across runs.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
from unittest.mock import patch

from fastapi.testclient import TestClient

from calibration_agent.models import CalibrationReport
from models import Market
from reddit_research_agent.models import RedditResearchReport
from risk_management_agent.models import RiskAssessment, RiskMetrics

import src.api.main as api_main
from src.api.dependencies import TRADE_STORE_ENV_VAR
from storage import (
    SQLITE_PATH_ENV_VAR,
    SQLiteStore,
    reset_default_store,
    sync_audit_to_sqlite,
    sync_trade_logs_to_sqlite,
)


def _build_market(
    market_id: str = "mkt-1",
    *,
    title: str = "Will Alpha resolve YES?",
    category: str = "Politics",
) -> Market:
    return Market(
        market_id=market_id,
        title=title,
        category=category,
        implied_prob=0.45,
        bid_price=0.44,
        ask_price=0.46,
        volume_24h=20_000.0,
        price_history={"1h": 0.01, "6h": 0.02, "24h": 0.04},
        open_interest=30_000.0,
        resolution_date=datetime(2026, 12, 1, tzinfo=timezone.utc),
        rules_text="Resolves YES if Alpha event occurs by 2026-12-01.",
    )


def _build_reddit_report() -> RedditResearchReport:
    return RedditResearchReport(
        bullish_thesis="Strong primary-source signal of Alpha.",
        bearish_thesis="Crowd may be over-discounting reversal risk.",
        key_evidence=["Reddit thread links primary source"],
        key_assumptions=["Source remains credible"],
        conviction_score=7,
        evidence_quality_score=70,
        misinformation_risk_score=15,
        sentiment_score=20,
        key_sources=["u/sourcehound"],
        summary="Reddit threads converge on Alpha resolving YES.",
        pricing_assessment="underpriced",
        assessment_reasoning="Discussion contains fresher evidence than the price.",
    )


def _build_news_report_dict() -> Dict[str, Any]:
    """Pydantic-compatible news report payload (constructed via NewsResearchReport in tests)."""

    return {
        "timeline": ["2026-04-20: Headline A"],
        "key_facts": ["Source-backed fact"],
        "source_quality_score": 7,
        "bullish_thesis": "Headlines back YES resolution.",
        "bearish_thesis": "Coverage thin and reversible.",
        "evidence_quality_score": 65,
        "misinformation_risk_score": 18,
        "sentiment_score": 30,
        "key_sources": ["https://example.com/a"],
        "summary": "Coverage tilts bullish.",
    }


def _build_calibration_report(
    *,
    action: str = "paper-trade candidate",
    calibrated_true_prob: float = 0.55,
) -> CalibrationReport:
    return CalibrationReport(
        xgboost_prob=0.50,
        llm_adjustment_pct_points=5.0,
        calibrated_true_prob=calibrated_true_prob,
        confidence_score=78,
        key_drivers=["Reddit signal", "News coverage"],
        key_uncertainties=["Late headline reversal possible"],
        edge_vs_market=calibrated_true_prob - 0.45,
        action=action,
        reasoning="Calibrated using mocked agents.",
    )


class _StubScanner:
    """Tiny stub for ``main.build_scan_results`` that returns the rows for the markets it was given."""

    def __init__(self, markets: Sequence[Market]) -> None:
        self.markets = list(markets)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls.append(kwargs)
        rows: List[Dict[str, Any]] = []
        for index, market in enumerate(self.markets):
            rows.append(
                {
                    "market_id": market.market_id,
                    "title": market.title,
                    "category": market.category,
                    "implied_prob": float(market.implied_prob),
                    "spread": 0.02,
                    "volume_24h": float(market.volume_24h),
                    "move_24h": 0.04,
                    "days_to_resolution": 30.0,
                    "clarity_score": 70 - index,
                    "anomaly_flags": [],
                    "research_priority": 99 - index,
                }
            )
        return rows


class _StubFetcher:
    """Stub for ``fetch_active_markets`` returning canned ``Market`` instances."""

    def __init__(self, markets: Sequence[Market]) -> None:
        self.markets = list(markets)

    def __call__(self, **_: Any) -> Iterator[Market]:
        yield from self.markets


class _StubRedditDiver:
    instances: List["_StubRedditDiver"] = []

    def __init__(self, search_query: str, *, subreddits: Optional[Sequence[str]] = None, **_: Any) -> None:
        self.search_query = search_query
        self.subreddits = list(subreddits or [])
        type(self).instances.append(self)

    def fetch_threads(self) -> str:
        return f"reddit-stub::{self.search_query}"


class _StubNewsAggregator:
    instances: List["_StubNewsAggregator"] = []

    def __init__(self, search_query: str, *_: Any, **__: Any) -> None:
        self.search_query = search_query
        type(self).instances.append(self)

    def fetch_news_context(self) -> str:
        return f"news-stub::{self.search_query}"


class _StubRedditAgent:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def analyze_discussion(self, **kwargs: Any) -> RedditResearchReport:
        self.calls.append(kwargs)
        return _build_reddit_report()


class _StubNewsAgent:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def analyze_news(self, **kwargs: Any) -> Any:
        from news_research_agent.models import NewsResearchReport

        self.calls.append(kwargs)
        return NewsResearchReport(**_build_news_report_dict())


class _StubCalibrationAgent:
    def __init__(self, *, action: str = "paper-trade candidate") -> None:
        self.calls: List[Dict[str, Any]] = []
        self.action = action

    def calibrate(self, **kwargs: Any) -> CalibrationReport:
        self.calls.append(kwargs)
        return _build_calibration_report(action=self.action)


def _patch_agents(
    *,
    markets: Sequence[Market],
    scanner: Optional[_StubScanner] = None,
    research_mock: bool = False,
    calibration_action: str = "paper-trade candidate",
):
    """Return a list of context managers that swap real agents for stubs."""

    fetcher = _StubFetcher(markets)
    scanner_stub = scanner or _StubScanner(markets)
    reddit_agent = _StubRedditAgent()
    news_agent = _StubNewsAgent()
    calibration_agent = _StubCalibrationAgent(action=calibration_action)
    _StubRedditDiver.instances = []
    _StubNewsAggregator.instances = []

    env_patch_kwargs = {"AUTOPILOT_TRADE_STORE": os.environ.get("AUTOPILOT_TRADE_STORE", "")}
    if research_mock:
        env_patch_kwargs["RESEARCH_MOCK"] = "true"

    return {
        "patches": [
            patch.object(api_main, "build_scan_results", side_effect=scanner_stub),
            patch.object(api_main, "fetch_active_markets", side_effect=fetcher),
            patch.object(api_main, "RedditDeepDiver", _StubRedditDiver),
            patch.object(api_main, "GoogleNewsRSSFetcher", _StubNewsAggregator),
            patch.object(api_main, "RedditAgent", lambda: reddit_agent),
            patch.object(api_main, "NewsAgent", lambda: news_agent),
            patch.object(api_main, "CalibrationAgent", lambda: calibration_agent),
            patch.object(api_main, "get_xgboost_probability", lambda market: 0.50),
        ],
        "scanner": scanner_stub,
        "fetcher": fetcher,
        "reddit_agent": reddit_agent,
        "news_agent": news_agent,
        "calibration_agent": calibration_agent,
    }


class _PatchScope:
    """Apply a list of ``mock.patch`` objects together; release them on exit."""

    def __init__(self, patches: Sequence[Any]) -> None:
        self.patches = list(patches)
        self.entered: List[Any] = []

    def __enter__(self) -> "_PatchScope":
        for patcher in self.patches:
            self.entered.append(patcher.__enter__())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for patcher in reversed(self.patches):
            patcher.__exit__(exc_type, exc, tb)


class APIBaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trade_store = Path(self._tmp.name)
        self._env_patch = patch.dict(
            os.environ,
            {TRADE_STORE_ENV_VAR: str(self.trade_store)},
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)
        self.client = TestClient(api_main.app)


class HealthEndpointTests(APIBaseTestCase):
    def test_health_returns_ok(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "autopilot-prediction-market")
        self.assertEqual(body["version"], api_main.APP_VERSION)


class ScanEndpointTests(APIBaseTestCase):
    def test_scan_returns_results_from_scanner(self) -> None:
        markets = [_build_market("mkt-1"), _build_market("mkt-2", title="Will Beta resolve YES?")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post(
                "/scan",
                json={"top_n": 5, "category": "Politics", "min_volume_24h": 1000.0},
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual({row["market_id"] for row in body["results"]}, {"mkt-1", "mkt-2"})
        self.assertTrue(body["scan_id"].startswith("scan-"))
        # Scanner was called with the request kwargs.
        self.assertGreaterEqual(len(wired["scanner"].calls), 1)
        call = wired["scanner"].calls[0]
        self.assertEqual(call["category"], "Politics")
        self.assertEqual(call["min_volume_24h"], 1000.0)

    def test_scan_empty_body_uses_defaults(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post("/scan", json={})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], 1)


class ResearchEndpointTests(APIBaseTestCase):
    def test_research_with_mock_data(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets, research_mock=True)
        with patch.dict(os.environ, {"RESEARCH_MOCK": "true"}, clear=False), _PatchScope(wired["patches"]):
            response = self.client.post("/research", json={"market_id": "mkt-1"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["market_id"], "mkt-1")
        self.assertIn("Will Alpha resolve YES?", body["reddit_query"])
        # Reddit/News reports came back as serializable dicts with the stub payloads.
        self.assertEqual(body["reddit_report"]["pricing_assessment"], "underpriced")
        self.assertEqual(body["news_report"]["source_quality_score"], 7)


class PredictEndpointTests(APIBaseTestCase):
    def test_predict_returns_calibration_report(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post("/predict", json={"market_id": "mkt-1"})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("calibrated_true_prob", body)
        self.assertIn("action", body)
        self.assertEqual(body["action"], "paper-trade candidate")

    def test_predict_unknown_market_returns_404(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post("/predict", json={"market_id": "does-not-exist"})
        self.assertEqual(response.status_code, 404)


class RiskEndpointTests(APIBaseTestCase):
    def _calibration_payload(self) -> Dict[str, Any]:
        return _build_calibration_report().model_dump()

    def test_risk_returns_assessment(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post(
                "/risk",
                json={
                    "market_id": "mkt-1",
                    "calibration": self._calibration_payload(),
                    "bankroll": 5_000.0,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["market_id"], "mkt-1")
        self.assertIn("risk_metrics", body)
        self.assertIn("risk_assessment", body)
        # Risk assessment fields are present.
        assessment = body["risk_assessment"]
        self.assertIn("allow_trade", assessment)
        self.assertIn("final_recommendation", assessment)

    def test_risk_with_invalid_bankroll_returns_422(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post(
                "/risk",
                json={
                    "market_id": "mkt-1",
                    "calibration": self._calibration_payload(),
                    "bankroll": -1.0,
                },
            )
        # Pydantic v2 enforces bankroll >= 0.0 at request validation time → 422.
        self.assertEqual(response.status_code, 422)


class PaperTradeEndpointTests(APIBaseTestCase):
    def test_paper_trade_writes_log_to_store_dir(self) -> None:
        markets = [_build_market("mkt-1")]
        wired = _patch_agents(markets=markets)
        with _PatchScope(wired["patches"]):
            response = self.client.post(
                "/paper-trade",
                json={"market_id": "mkt-1", "top_n": 3, "category": "Politics"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["market_id"], "mkt-1")
        log_path = Path(body["trade_log_path"])
        self.assertTrue(log_path.is_file(), msg=f"missing trade log at {log_path}")
        # Log lives inside the configured trade store dir, not the repo root.
        self.assertEqual(log_path.parent.resolve(), self.trade_store.resolve())
        # Round-trip: payload contains the canonical orchestrator schema.
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "open")
        self.assertEqual(payload["source"], "orchestrator")
        self.assertIn("features_window", payload)


class SettleEndpointTests(APIBaseTestCase):
    def _seed_open_trade(self, market_id: str = "mkt-1") -> Path:
        log_path = self.trade_store / f"trade_execution_{market_id}.json"
        log_path.write_text(
            json.dumps(
                {
                    "event_id": market_id,
                    "trade_id": market_id,
                    "status": "open",
                    "created_at_utc": "2026-04-20T12:00:00+00:00",
                    "settled_at": None,
                    "final_outcome": None,
                    "market_outcome": None,
                    "post_settlement_news": None,
                    "scanner": {"market_id": market_id, "implied_prob": 0.45},
                    "features_window": {},
                    "calibration": None,
                    "risk": None,
                    "entry_price": 0.46,
                    "position_size_usd": 230.0,
                    "exit_price": None,
                    "realized_pnl_usd": None,
                    "max_loss_usd": 230.0,
                    "source": "orchestrator",
                    "notes": None,
                }
            ),
            encoding="utf-8",
        )
        return log_path

    def test_settle_marks_log_as_settled(self) -> None:
        log_path = self._seed_open_trade("mkt-7")
        response = self.client.post(
            "/settle",
            json={
                "market_id": "mkt-7",
                "outcome": "win",
                "market_outcome": "yes",
                "news": "Alpha confirmed.",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["payload"]["status"], "settled")
        self.assertTrue(body["payload"]["final_outcome"])
        self.assertTrue(body["payload"]["market_outcome"])
        # File on disk also mutated.
        on_disk = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["status"], "settled")

    def test_settle_unknown_market_returns_404(self) -> None:
        response = self.client.post(
            "/settle",
            json={"market_id": "missing", "outcome": "win"},
        )
        self.assertEqual(response.status_code, 404)


class TradesEndpointTests(APIBaseTestCase):
    def _seed(self, market_id: str, *, status: str, source: str = "orchestrator") -> Path:
        path = self.trade_store / f"trade_execution_{market_id}.json"
        path.write_text(
            json.dumps(
                {
                    "trade_id": market_id,
                    "status": status,
                    "source": source,
                    "scanner": {"market_id": market_id},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_trades_filters_by_status(self) -> None:
        self._seed("mkt-open", status="open")
        self._seed("mkt-settled", status="settled")
        response = self.client.get("/trades?status=open")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["trades"][0]["trade_id"], "mkt-open")

    def test_trades_filters_by_source(self) -> None:
        self._seed("mkt-open", status="open", source="orchestrator")
        self._seed("mkt-shadow", status="open", source="shadow")
        response = self.client.get("/trades?source=shadow")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["trades"][0]["trade_id"], "mkt-shadow")


class PostmortemsEndpointTests(APIBaseTestCase):
    def _seed_audit(self) -> None:
        audit_payload = {
            "reviews": [
                {
                    "trade_id": "mkt-1",
                    "outcome_review": {"matrix_classification": "Deserved Success", "good_process": True, "good_outcome": True},
                    "reviewed_at": "2026-04-20T12:00:00+00:00",
                },
                {
                    "trade_id": "mkt-2",
                    "outcome_review": {"matrix_classification": "Good Failure", "good_process": True, "good_outcome": False},
                    "reviewed_at": "2026-04-21T12:00:00+00:00",
                },
            ],
            "aggregates": {"review_count": 2, "process_health_pct": 100.0, "win_rate_pct": 50.0},
        }
        (self.trade_store / "performance_audit.json").write_text(
            json.dumps(audit_payload), encoding="utf-8"
        )

    def test_postmortems_reads_audit_file(self) -> None:
        self._seed_audit()
        response = self.client.get("/postmortems")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual({r["trade_id"] for r in body["reviews"]}, {"mkt-1", "mkt-2"})
        self.assertEqual(body["aggregates"]["review_count"], 2)

    def test_postmortems_with_no_audit_returns_empty(self) -> None:
        response = self.client.get("/postmortems")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["reviews"], [])


class SQLiteBackedEndpointTests(APIBaseTestCase):
    """When ``AUTOPILOT_SQLITE_PATH`` is set the read endpoints serve SQLite rows.

    The hard test: seed 2 trade logs into SQLite, then *delete* the source
    JSON files. The endpoint must still return both rows because they're
    now living in the SQLite mirror, proving the fallback was bypassed.
    """

    def setUp(self) -> None:
        super().setUp()
        # Each test gets its own db file so the singleton cache doesn't leak across cases.
        self.db_path = self.trade_store / "autopilot.sqlite"
        self._sqlite_env_patch = patch.dict(
            os.environ,
            {SQLITE_PATH_ENV_VAR: str(self.db_path)},
            clear=False,
        )
        self._sqlite_env_patch.start()
        self.addCleanup(self._sqlite_env_patch.stop)
        # Force a fresh singleton (and tear it down) so neighbour tests aren't poisoned.
        reset_default_store()
        self.addCleanup(reset_default_store)

    def _seed_json_trade(
        self,
        market_id: str,
        *,
        status: str = "open",
        source: str = "orchestrator",
    ) -> Path:
        path = self.trade_store / f"trade_execution_{market_id}.json"
        path.write_text(
            json.dumps(
                {
                    "event_id": market_id,
                    "trade_id": market_id,
                    "status": status,
                    "source": source,
                    "scanner": {"market_id": market_id, "title": f"market {market_id}"},
                    "features_window": {"market_implied_prob": 0.45},
                    "calibration": {"calibrated_true_prob": 0.55},
                    "risk": {"risk_assessment": {"allow_trade": True}},
                    "research": {"reddit_query": "demo"},
                    "entry_price": 0.45,
                    "exit_price": None,
                    "position_size_usd": 200.0,
                    "realized_pnl_usd": None,
                    "max_loss_usd": 200.0,
                    "notes": None,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_trades_endpoint_uses_sqlite_when_enabled(self) -> None:
        path_a = self._seed_json_trade("mkt-sql-a", status="open")
        path_b = self._seed_json_trade("mkt-sql-b", status="open")
        # Mirror both files into SQLite using a private store, then close it so
        # the endpoint can open its own singleton against the same db file.
        with SQLiteStore(self.db_path) as store:
            result = sync_trade_logs_to_sqlite(self.trade_store, store=store)
        self.assertEqual(result["synced"], 2, msg=result)

        # Delete the source JSON files. Any read that falls back to the JSON
        # walk will return 0 rows; the SQLite path must still see both trades.
        path_a.unlink()
        path_b.unlink()

        response = self.client.get("/trades?limit=10")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(
            {row["trade_id"] for row in body["trades"]},
            {"mkt-sql-a", "mkt-sql-b"},
        )

    def test_postmortems_endpoint_uses_sqlite_when_enabled(self) -> None:
        audit_path = self.trade_store / "performance_audit.json"
        audit_payload = {
            "reviews": [
                {
                    "trade_id": "mkt-sql-a",
                    "source_file": "/tmp/trade_execution_mkt-sql-a.json",
                    "trade_key": "/tmp/trade_execution_mkt-sql-a.json:mkt-sql-a",
                    "outcome_review": {
                        "matrix_classification": "Deserved Success",
                        "good_process": True,
                        "good_outcome": True,
                    },
                    "reviewed_at": "2026-04-20T12:00:00+00:00",
                    "settled_at": "2026-04-20T11:00:00+00:00",
                    "final_outcome": True,
                },
                {
                    "trade_id": "mkt-sql-b",
                    "source_file": "/tmp/trade_execution_mkt-sql-b.json",
                    "trade_key": "/tmp/trade_execution_mkt-sql-b.json:mkt-sql-b",
                    "outcome_review": {
                        "matrix_classification": "Good Failure",
                        "good_process": True,
                        "good_outcome": False,
                    },
                    "reviewed_at": "2026-04-21T12:00:00+00:00",
                    "settled_at": "2026-04-21T11:00:00+00:00",
                    "final_outcome": False,
                },
            ],
            "aggregates": {"review_count": 2, "process_health_pct": 100.0, "win_rate_pct": 50.0},
        }
        audit_path.write_text(json.dumps(audit_payload), encoding="utf-8")

        with SQLiteStore(self.db_path) as store:
            result = sync_audit_to_sqlite(audit_path, store=store)
        self.assertEqual(result["synced"], 2, msg=result)

        # Delete the audit JSON. SQLite must still serve the reviews.
        audit_path.unlink()

        response = self.client.get("/postmortems?limit=10")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["count"], 2)
        self.assertEqual(
            {row["trade_id"] for row in body["reviews"]},
            {"mkt-sql-a", "mkt-sql-b"},
        )


if __name__ == "__main__":
    unittest.main()
