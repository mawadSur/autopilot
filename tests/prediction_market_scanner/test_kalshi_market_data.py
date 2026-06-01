"""Tests for src/exchanges/kalshi_market_data.py.

Hermetic — never makes a real HTTPS call. We patch ``session.get`` to return
a stub response whose ``.json()`` yields canned Kalshi-shaped payloads, and
assert on parsing, cents->probability conversion, error wrapping, and
missing-field tolerance.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, Optional
from unittest import mock

import requests

from exchanges.kalshi_market_data import (
    KalshiAPIError,
    KalshiMarketDataClient,
    normalize_market,
)


# ---------------------------------------------------------------------------
# Stub HTTP scaffolding
# ---------------------------------------------------------------------------


class _StubResponse:
    """Minimal ``requests.Response`` shaped stub."""

    def __init__(
        self,
        json_data: Any = None,
        status_code: int = 200,
        raise_on_json: Optional[Exception] = None,
    ) -> None:
        self._json_data = json_data
        self.status_code = status_code
        self._raise_on_json = raise_on_json

    def raise_for_status(self) -> None:
        if not (200 <= int(self.status_code) < 300):
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        if self._raise_on_json is not None:
            raise self._raise_on_json
        return self._json_data


# ---------------------------------------------------------------------------
# Sample payloads (realistic Kalshi shapes; prices in cents 1-99)
# ---------------------------------------------------------------------------


def _markets_payload() -> Dict[str, Any]:
    return {
        "cursor": "next_page_token",
        "markets": [
            {
                "ticker": "PRESWIN-24-DEM",
                "title": "Will the Democratic candidate win?",
                "yes_bid": 58,
                "yes_ask": 62,
                "no_bid": 38,
                "no_ask": 42,
                "last_price": 60,
                "volume": 12345,
                "close_time": "2024-11-05T23:59:59Z",
                "status": "open",
            },
            {
                "ticker": "RATE-HIKE-Q1",
                "title": "Will the Fed hike rates in Q1?",
                "yes_bid": 10,
                "yes_ask": 14,
                "no_bid": 86,
                "no_ask": 90,
                "last_price": 12,
                "volume": 678,
                "close_time": "2025-03-31T23:59:59Z",
                "status": "open",
            },
        ],
    }


def _single_market_payload() -> Dict[str, Any]:
    return {
        "market": {
            "ticker": "PRESWIN-24-DEM",
            "title": "Will the Democratic candidate win?",
            "yes_bid": 58,
            "yes_ask": 62,
            "no_bid": 38,
            "no_ask": 42,
            "last_price": 60,
            "volume": 12345,
            "close_time": "2024-11-05T23:59:59Z",
            "status": "open",
        }
    }


def _orderbook_payload() -> Dict[str, Any]:
    return {
        "orderbook": {
            # Kalshi books: [price_in_cents, quantity] levels per side.
            "yes": [[58, 1000], [57, 2500], [56, 4000]],
            "no": [[42, 1200], [43, 1800], [44, 3000]],
        }
    }


def _make_client(get_mock: mock.Mock) -> KalshiMarketDataClient:
    session = mock.Mock(spec=requests.Session)
    session.get = get_mock
    return KalshiMarketDataClient(session=session)


# ---------------------------------------------------------------------------
# get_markets: parse + normalize
# ---------------------------------------------------------------------------


class GetMarketsTest(unittest.TestCase):
    def test_get_markets_parses_and_normalizes(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_markets_payload()))
        client = _make_client(get_mock)

        markets = client.get_markets(limit=50, status="open")

        self.assertEqual(len(markets), 2)
        first = markets[0]
        # Common schema keys present.
        self.assertEqual(
            set(first.keys()),
            {
                "ticker",
                "title",
                "yes_bid",
                "yes_ask",
                "no_bid",
                "no_ask",
                "last_price",
                "volume",
                "close_ts",
                "implied_prob",
            },
        )
        self.assertEqual(first["ticker"], "PRESWIN-24-DEM")
        self.assertEqual(first["close_ts"], "2024-11-05T23:59:59Z")
        self.assertEqual(first["volume"], 12345)

    def test_get_markets_passes_query_params(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_markets_payload()))
        client = _make_client(get_mock)

        client.get_markets(limit=25, status="open", cursor="abc123")

        _, kwargs = get_mock.call_args
        self.assertEqual(kwargs["params"]["limit"], 25)
        self.assertEqual(kwargs["params"]["status"], "open")
        self.assertEqual(kwargs["params"]["cursor"], "abc123")
        self.assertEqual(kwargs["timeout"], 10.0)

    def test_get_markets_handles_empty_payload(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({}))
        client = _make_client(get_mock)
        self.assertEqual(client.get_markets(), [])


# ---------------------------------------------------------------------------
# cents -> probability conversion
# ---------------------------------------------------------------------------


class CentsToProbabilityTest(unittest.TestCase):
    def test_yes_ask_60c_to_060(self) -> None:
        raw = {"ticker": "X", "yes_ask": 60}
        norm = normalize_market(raw)
        self.assertAlmostEqual(norm["yes_ask"], 0.60)

    def test_all_price_fields_scaled_to_unit_interval(self) -> None:
        raw = {
            "ticker": "X",
            "yes_bid": 58,
            "yes_ask": 62,
            "no_bid": 38,
            "no_ask": 42,
            "last_price": 60,
        }
        norm = normalize_market(raw)
        self.assertAlmostEqual(norm["yes_bid"], 0.58)
        self.assertAlmostEqual(norm["yes_ask"], 0.62)
        self.assertAlmostEqual(norm["no_bid"], 0.38)
        self.assertAlmostEqual(norm["no_ask"], 0.42)
        self.assertAlmostEqual(norm["last_price"], 0.60)

    def test_implied_prob_is_yes_midpoint(self) -> None:
        # mid of 58c/62c -> (0.58 + 0.62) / 2 = 0.60
        raw = {"ticker": "X", "yes_bid": 58, "yes_ask": 62, "last_price": 99}
        norm = normalize_market(raw)
        self.assertAlmostEqual(norm["implied_prob"], 0.60)

    def test_implied_prob_falls_back_to_ask_then_bid_then_last(self) -> None:
        ask_only = normalize_market({"ticker": "X", "yes_ask": 70})
        self.assertAlmostEqual(ask_only["implied_prob"], 0.70)

        bid_only = normalize_market({"ticker": "X", "yes_bid": 30})
        self.assertAlmostEqual(bid_only["implied_prob"], 0.30)

        last_only = normalize_market({"ticker": "X", "last_price": 45})
        self.assertAlmostEqual(last_only["implied_prob"], 0.45)


# ---------------------------------------------------------------------------
# Error handling: network / HTTP errors raise KalshiAPIError
# ---------------------------------------------------------------------------


class ErrorHandlingTest(unittest.TestCase):
    def test_network_error_raises_kalshi_api_error(self) -> None:
        get_mock = mock.Mock(
            side_effect=requests.ConnectionError("connection refused")
        )
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError) as ctx:
            client.get_markets()
        # Context preserved, original on __cause__, not a bare requests error.
        self.assertIn("get_markets", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, requests.ConnectionError)

    def test_timeout_raises_kalshi_api_error(self) -> None:
        get_mock = mock.Mock(side_effect=requests.Timeout("timed out"))
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError):
            client.get_market("PRESWIN-24-DEM")

    def test_http_500_raises_kalshi_api_error_with_ticker_context(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({}, status_code=500))
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError) as ctx:
            client.get_orderbook("RATE-HIKE-Q1")
        msg = str(ctx.exception)
        self.assertIn("get_orderbook", msg)
        self.assertIn("RATE-HIKE-Q1", msg)
        self.assertIsInstance(ctx.exception.__cause__, requests.HTTPError)

    def test_non_json_body_raises_kalshi_api_error(self) -> None:
        get_mock = mock.Mock(
            return_value=_StubResponse(raise_on_json=ValueError("no json"))
        )
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError):
            client.get_markets()

    def test_empty_ticker_rejected(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_single_market_payload()))
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError):
            client.get_market("")
        with self.assertRaises(KalshiAPIError):
            client.get_orderbook("")


# ---------------------------------------------------------------------------
# get_market / get_orderbook happy paths
# ---------------------------------------------------------------------------


class GetMarketAndOrderbookTest(unittest.TestCase):
    def test_get_market_normalizes_single(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_single_market_payload()))
        client = _make_client(get_mock)

        market = client.get_market("PRESWIN-24-DEM")
        self.assertEqual(market["ticker"], "PRESWIN-24-DEM")
        self.assertAlmostEqual(market["yes_ask"], 0.62)
        self.assertAlmostEqual(market["implied_prob"], 0.60)

        # URL path includes the ticker.
        args, _ = get_mock.call_args
        self.assertIn("/markets/PRESWIN-24-DEM", args[0])

    def test_get_market_missing_market_object_raises(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({"unexpected": True}))
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError):
            client.get_market("PRESWIN-24-DEM")

    def test_get_orderbook_returns_raw_book(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse(_orderbook_payload()))
        client = _make_client(get_mock)

        book = client.get_orderbook("PRESWIN-24-DEM", depth=5)
        self.assertIn("yes", book)
        self.assertIn("no", book)
        self.assertEqual(book["yes"][0], [58, 1000])

        _, kwargs = get_mock.call_args
        self.assertEqual(kwargs["params"]["depth"], 5)

    def test_get_orderbook_missing_book_raises(self) -> None:
        get_mock = mock.Mock(return_value=_StubResponse({"nope": 1}))
        client = _make_client(get_mock)
        with self.assertRaises(KalshiAPIError):
            client.get_orderbook("PRESWIN-24-DEM")


# ---------------------------------------------------------------------------
# normalize_market: missing-field tolerance
# ---------------------------------------------------------------------------


class NormalizeMissingFieldsTest(unittest.TestCase):
    def test_empty_dict_does_not_crash(self) -> None:
        norm = normalize_market({})
        self.assertIsNone(norm["ticker"])
        self.assertIsNone(norm["title"])
        self.assertIsNone(norm["yes_bid"])
        self.assertIsNone(norm["yes_ask"])
        self.assertIsNone(norm["implied_prob"])
        self.assertEqual(norm["volume"], 0)
        self.assertIsNone(norm["close_ts"])

    def test_partial_payload_preserves_known_fields(self) -> None:
        norm = normalize_market({"ticker": "ONLY", "yes_ask": 55})
        self.assertEqual(norm["ticker"], "ONLY")
        self.assertAlmostEqual(norm["yes_ask"], 0.55)
        self.assertAlmostEqual(norm["implied_prob"], 0.55)
        self.assertIsNone(norm["no_bid"])

    def test_non_numeric_price_degrades_to_none(self) -> None:
        norm = normalize_market({"ticker": "X", "yes_ask": "not-a-number"})
        self.assertIsNone(norm["yes_ask"])

    def test_non_numeric_volume_degrades_to_zero(self) -> None:
        norm = normalize_market({"ticker": "X", "volume": "garbage"})
        self.assertEqual(norm["volume"], 0)

    def test_alt_volume_and_close_keys_supported(self) -> None:
        norm = normalize_market(
            {"ticker": "X", "volume_24h": 999, "close_ts": "2030-01-01T00:00:00Z"}
        )
        self.assertEqual(norm["volume"], 999)
        self.assertEqual(norm["close_ts"], "2030-01-01T00:00:00Z")

    def test_non_dict_input_raises(self) -> None:
        with self.assertRaises(KalshiAPIError):
            normalize_market(["not", "a", "dict"])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
