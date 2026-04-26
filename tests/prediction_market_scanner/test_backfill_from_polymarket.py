"""Tests for ``calibration_agent.backfill_from_polymarket`` and the
companion ``fetcher.fetch_resolved_markets`` helper.

The backfill module must:

* Convert ``(Market, market_outcome)`` pairs into synthetic trade execution
  logs that match the canonical orchestrator schema (``source="backfill"``,
  ``notes`` populated, ``status="settled"``, ``market_outcome`` set).
* Always emit the full ``features_window`` (14 numeric columns +
  ``captured_at_utc``), with research-signal columns defaulting to ``0.0``
  because backfilled rows have no research context.
* Be safely importable without touching the network (the real fetcher must
  be lazy-imported inside ``backfill``).

The fetcher tests cover the resolution-outcome parser end-to-end via a
stubbed HTTP session.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import patch

import requests

from calibration_agent.backfill_from_polymarket import (
    BACKFILL_NOTES,
    backfill,
)
from calibration_agent.ml_service import ALL_FEATURE_COLUMNS, RESEARCH_FEATURE_COLUMNS
from fetcher import fetch_resolved_markets
from models import Market


def _make_market(
    *,
    market_id: str,
    implied_prob: float = 0.42,
    volume_24h: float = 8000.0,
    resolution_date: datetime = datetime(2026, 4, 1, tzinfo=timezone.utc),
) -> Market:
    return Market(
        market_id=market_id,
        title=f"Market {market_id}",
        category="Politics",
        implied_prob=implied_prob,
        bid_price=max(0.0, implied_prob - 0.01),
        ask_price=min(1.0, implied_prob + 0.01),
        volume_24h=volume_24h,
        price_history={"1h": 0.01, "6h": 0.02, "24h": 0.03},
        open_interest=15_000.0,
        resolution_date=resolution_date,
        rules_text="Resolves to Yes if condition is met.",
    )


def _capture_stderr(func, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


class BackfillFromPolymarketTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def _fake_fetcher(
        self, pairs: List[Tuple[Market, bool]]
    ):
        captured: Dict[str, Any] = {"calls": []}

        def fetcher(**kwargs: Any) -> List[Tuple[Market, bool]]:
            captured["calls"].append(dict(kwargs))
            return pairs

        return fetcher, captured

    def test_writes_one_log_per_pair_with_canonical_schema(self) -> None:
        pairs = [
            (_make_market(market_id="mkt-1"), True),
            (_make_market(market_id="mkt-2"), False),
            (_make_market(market_id="mkt-3"), True),
        ]
        fetcher, captured = self._fake_fetcher(pairs)

        summary, _ = _capture_stderr(
            backfill,
            self.tmp_path,
            fetcher=fetcher,
            limit=10,
            min_volume_24h=1.0,
        )

        # Summary counts match.
        self.assertEqual(summary["markets_fetched"], 3)
        self.assertEqual(summary["trade_logs_written"], 3)
        self.assertEqual(summary["skipped_ambiguous_resolution"], 0)
        self.assertEqual(summary["output_dir"], str(self.tmp_path))

        # The fake fetcher was called exactly once with our kwargs forwarded.
        self.assertEqual(len(captured["calls"]), 1)
        self.assertEqual(captured["calls"][0]["min_volume_24h"], 1.0)
        self.assertIsNone(captured["calls"][0]["days_back"])

        written_files = sorted(self.tmp_path.glob("trade_execution_*.json"))
        self.assertEqual(
            [path.name for path in written_files],
            [
                "trade_execution_mkt-1.json",
                "trade_execution_mkt-2.json",
                "trade_execution_mkt-3.json",
            ],
        )

        for path, (market, expected_outcome) in zip(written_files, pairs):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["source"], "backfill")
            self.assertEqual(payload["notes"], BACKFILL_NOTES)
            self.assertEqual(payload["status"], "settled")
            self.assertEqual(payload["market_outcome"], expected_outcome)
            self.assertEqual(payload["final_outcome"], expected_outcome)
            self.assertEqual(payload["event_id"], market.market_id)
            self.assertEqual(payload["trade_id"], market.market_id)
            self.assertIsNone(payload["model_meta"])
            self.assertIsNone(payload["research"])
            self.assertIsNone(payload["calibration"])
            self.assertIsNone(payload["risk"])
            self.assertIsNone(payload["post_settlement_news"])
            self.assertEqual(payload["created_at_utc"], payload["settled_at"])

            scanner = payload["scanner"]
            self.assertEqual(scanner["market_id"], market.market_id)
            self.assertEqual(scanner["title"], market.title)
            self.assertEqual(scanner["category"], market.category)
            self.assertAlmostEqual(scanner["implied_prob"], market.implied_prob)

            features_window = payload["features_window"]
            for column in ALL_FEATURE_COLUMNS:
                self.assertIn(column, features_window, msg=f"missing {column!r}")
            self.assertIn("captured_at_utc", features_window)
            for column in RESEARCH_FEATURE_COLUMNS:
                self.assertEqual(
                    features_window[column],
                    0.0,
                    msg=f"{column!r} must default to 0.0 when no research is available",
                )
            self.assertEqual(features_window["implied_prob"], market.implied_prob)

    def test_empty_fetcher_returns_empty_summary(self) -> None:
        fetcher, _ = self._fake_fetcher([])

        summary, _ = _capture_stderr(
            backfill,
            self.tmp_path,
            fetcher=fetcher,
            limit=10,
        )

        self.assertEqual(summary["markets_fetched"], 0)
        self.assertEqual(summary["trade_logs_written"], 0)
        self.assertEqual(summary["skipped_ambiguous_resolution"], 0)
        self.assertEqual(list(self.tmp_path.glob("trade_execution_*.json")), [])

    def test_limit_caps_number_of_logs_written(self) -> None:
        pairs = [
            (_make_market(market_id=f"mkt-{idx}"), bool(idx % 2 == 0))
            for idx in range(5)
        ]
        fetcher, _ = self._fake_fetcher(pairs)

        summary, _ = _capture_stderr(
            backfill,
            self.tmp_path,
            fetcher=fetcher,
            limit=2,
            min_volume_24h=1.0,
        )

        self.assertEqual(summary["markets_fetched"], 5)
        self.assertEqual(summary["trade_logs_written"], 2)
        self.assertEqual(len(list(self.tmp_path.glob("trade_execution_*.json"))), 2)

    def test_module_import_does_not_touch_network(self) -> None:
        # Re-import the module from scratch with requests.Session patched to
        # raise. If the module imports the real fetcher eagerly (which would
        # in turn instantiate a Session at call time, but we also want to
        # guard against module-level network IO), this test will explode.
        with patch.object(
            requests.Session,
            "get",
            side_effect=AssertionError(
                "backfill module must not touch the network at import time"
            ),
        ):
            import importlib

            module = importlib.import_module(
                "calibration_agent.backfill_from_polymarket"
            )
            importlib.reload(module)
            self.assertTrue(hasattr(module, "backfill"))


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any, headers: Dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _FakeSession:
    def __init__(self, responses: List[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.headers: Dict[str, str] = {}
        self.calls: List[Dict[str, Any]] = []
        self.closed = False

    def get(self, url: str, params: Dict[str, Any] | None = None, timeout: int | None = None) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        if not self.responses:
            raise AssertionError("No queued response left for _FakeSession.get")
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _closed_market_payload(
    *,
    market_id: str,
    outcome_prices: List[str],
    volume: float = 12_000.0,
    end_date: str = "2026-04-01T00:00:00Z",
    closed: bool = True,
) -> Dict[str, Any]:
    """Minimal closed-market payload for fetch_resolved_markets tests."""

    return {
        "id": market_id,
        "question": f"Resolved market {market_id}?",
        "description": "Resolved.",
        "active": False,
        "closed": closed,
        "bestBid": float(outcome_prices[0] if outcome_prices else 0.0),
        "bestAsk": float(outcome_prices[0] if outcome_prices else 0.0),
        "volumeNum": volume,
        "outcomePrices": outcome_prices,
        "endDate": end_date,
        "events": [{"openInterest": 5000.0, "description": "Event."}],
    }


class FetchResolvedMarketsTests(unittest.TestCase):
    @patch("fetcher.random.uniform", return_value=0.0)
    @patch("fetcher.time.sleep")
    def test_parses_outcomes_and_skips_ambiguous(self, _sleep_mock, _uniform_mock) -> None:
        page = [
            _closed_market_payload(market_id="m-yes", outcome_prices=["1", "0"]),
            _closed_market_payload(market_id="m-no", outcome_prices=["0", "1"]),
            # Ambiguous: refunded / void / split.
            _closed_market_payload(market_id="m-ambig", outcome_prices=["0.5", "0.5"]),
            # Also ambiguous: within the deadband but not extreme.
            _closed_market_payload(market_id="m-mid", outcome_prices=["0.7", "0.3"]),
        ]
        session = _FakeSession([_FakeResponse(200, page), _FakeResponse(200, [])])

        with self.assertLogs("fetcher", level="INFO") as log_capture:
            results = fetch_resolved_markets(
                min_volume_24h=1_000.0,
                page_size=10,
                session=session,
            )

        self.assertEqual(len(results), 2)
        market_outcomes = [(market.market_id, outcome) for market, outcome in results]
        self.assertIn(("m-yes", True), market_outcomes)
        self.assertIn(("m-no", False), market_outcomes)
        # Ambiguous markets must not appear.
        for market_id, _ in market_outcomes:
            self.assertNotIn(market_id, {"m-ambig", "m-mid"})

        # Query params must request closed markets, ordered by endDate desc.
        first_call = session.calls[0]
        self.assertEqual(first_call["params"]["active"], "false")
        self.assertEqual(first_call["params"]["closed"], "true")
        self.assertEqual(first_call["params"]["order"], "endDate")
        self.assertEqual(first_call["params"]["ascending"], "false")
        self.assertEqual(first_call["params"]["offset"], 0)

        # The skip count is logged at INFO so operators see fidelity loss.
        self.assertTrue(
            any("ambiguous" in record.lower() for record in log_capture.output),
            msg=f"expected ambiguous-skip INFO log, got {log_capture.output!r}",
        )

    @patch("fetcher.random.uniform", return_value=0.0)
    @patch("fetcher.time.sleep")
    def test_filters_low_volume_and_pages(self, _sleep_mock, _uniform_mock) -> None:
        page_one = [
            _closed_market_payload(market_id="hi-vol", outcome_prices=["1", "0"], volume=20_000.0),
            _closed_market_payload(market_id="lo-vol", outcome_prices=["1", "0"], volume=100.0),
        ]
        page_two = [
            _closed_market_payload(market_id="hi-vol-2", outcome_prices=["0", "1"], volume=15_000.0),
        ]
        session = _FakeSession(
            [
                _FakeResponse(200, page_one),
                _FakeResponse(200, page_two),
            ]
        )

        results = fetch_resolved_markets(
            min_volume_24h=5_000.0,
            page_size=2,
            session=session,
        )

        self.assertEqual(
            sorted([market.market_id for market, _ in results]),
            ["hi-vol", "hi-vol-2"],
        )
        self.assertEqual(session.calls[0]["params"]["offset"], 0)
        self.assertEqual(session.calls[1]["params"]["offset"], 2)

    @patch("fetcher.random.uniform", return_value=0.0)
    @patch("fetcher.time.sleep")
    def test_days_back_skips_old_markets(self, _sleep_mock, _uniform_mock) -> None:
        recent_dt = datetime.now(timezone.utc).replace(microsecond=0)
        old_iso = "2020-01-01T00:00:00Z"
        recent_iso = recent_dt.isoformat().replace("+00:00", "Z")
        page = [
            _closed_market_payload(
                market_id="recent",
                outcome_prices=["1", "0"],
                end_date=recent_iso,
            ),
            _closed_market_payload(
                market_id="old",
                outcome_prices=["1", "0"],
                end_date=old_iso,
            ),
        ]
        session = _FakeSession([_FakeResponse(200, page), _FakeResponse(200, [])])

        results = fetch_resolved_markets(
            min_volume_24h=1_000.0,
            page_size=10,
            session=session,
            days_back=30,
        )

        self.assertEqual([market.market_id for market, _ in results], ["recent"])

    @patch("fetcher.random.uniform", return_value=0.0)
    @patch("fetcher.time.sleep")
    def test_skips_markets_marked_not_closed(self, _sleep_mock, _uniform_mock) -> None:
        page = [
            _closed_market_payload(
                market_id="actually-open",
                outcome_prices=["1", "0"],
                closed=False,
            ),
            _closed_market_payload(
                market_id="closed-yes",
                outcome_prices=["1", "0"],
            ),
        ]
        session = _FakeSession([_FakeResponse(200, page), _FakeResponse(200, [])])

        results = fetch_resolved_markets(
            min_volume_24h=1_000.0,
            page_size=10,
            session=session,
        )

        self.assertEqual([market.market_id for market, _ in results], ["closed-yes"])


if __name__ == "__main__":
    unittest.main()
