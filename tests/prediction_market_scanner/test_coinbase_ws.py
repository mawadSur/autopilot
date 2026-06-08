"""Tests for src/exchanges/coinbase_ws.py and its integration with CoinbaseExchange.

All tests are hermetic — no real WebSocket is opened; we drive
``handle_message`` directly with synthesized Coinbase Advanced Trade
payloads and verify cache state.
"""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest import mock

from exchanges.coinbase_ws import (
    CoinbaseMarketStream,
    _from_product_id,
    _to_product_id,
)


def _ticker_frame(
    product_id: str = "ETH-USD",
    *,
    price: str = "2200.50",
    bid: str = "2200.40",
    ask: str = "2200.60",
    volume: str = "1234.5",
    ts_iso: str = "2026-05-17T12:00:00.000000Z",
) -> Dict[str, Any]:
    return {
        "channel": "ticker",
        "timestamp": ts_iso,
        "events": [
            {
                "type": "update",
                "tickers": [
                    {
                        "type": "ticker",
                        "product_id": product_id,
                        "price": price,
                        "volume_24_h": volume,
                        "best_bid": bid,
                        "best_ask": ask,
                    }
                ],
            }
        ],
    }


def _candle_frame(
    product_id: str = "ETH-USD",
    *,
    bars: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if bars is None:
        bars = [
            {
                "product_id": product_id,
                "start": str(int(time.time()) // 60 * 60),
                "open": "2200.0",
                "high": "2201.0",
                "low": "2199.0",
                "close": "2200.5",
                "volume": "12.3",
            }
        ]
    return {"channel": "candles", "events": [{"type": "snapshot", "candles": bars}]}


# ---------------------------------------------------------------------------
# CoinbaseMarketStream unit tests
# ---------------------------------------------------------------------------


class TestSymbolHelpers(unittest.TestCase):
    def test_to_product_id_handles_both_styles(self) -> None:
        self.assertEqual(_to_product_id("ETH/USD"), "ETH-USD")
        self.assertEqual(_to_product_id("eth-usd"), "ETH-USD")

    def test_from_product_id_returns_slash_form(self) -> None:
        self.assertEqual(_from_product_id("ETH-USD"), "ETH/USD")
        self.assertEqual(_from_product_id("BTC-USD"), "BTC/USD")


class TestTickerCache(unittest.TestCase):
    def test_handle_ticker_populates_cache(self) -> None:
        stream = CoinbaseMarketStream()
        stream.handle_message(_ticker_frame())
        cached = stream.get_ticker("ETH/USD")
        self.assertIsNotNone(cached)
        assert cached is not None  # for mypy
        self.assertEqual(cached["symbol"], "ETH/USD")
        self.assertAlmostEqual(cached["bid"], 2200.40)
        self.assertAlmostEqual(cached["ask"], 2200.60)
        self.assertAlmostEqual(cached["last"], 2200.50)
        self.assertAlmostEqual(cached["volume_24h_base"], 1234.5)
        self.assertEqual(cached["as_of_utc"], "2026-05-17T12:00:00.000000Z")

    def test_get_ticker_returns_none_when_stale(self) -> None:
        stream = CoinbaseMarketStream(ticker_freshness_s=0.0)
        stream.handle_message(_ticker_frame())
        # Freshness window is 0s -> any read after the insert is stale.
        time.sleep(0.001)
        self.assertIsNone(stream.get_ticker("ETH/USD"))

    def test_get_ticker_returns_none_for_unknown_symbol(self) -> None:
        stream = CoinbaseMarketStream()
        stream.handle_message(_ticker_frame(product_id="ETH-USD"))
        self.assertIsNone(stream.get_ticker("BTC/USD"))

    def test_ticker_with_missing_bid_falls_back_to_last(self) -> None:
        stream = CoinbaseMarketStream()
        frame = _ticker_frame(bid="", ask="")
        stream.handle_message(frame)
        cached = stream.get_ticker("ETH/USD")
        # When bid/ask are empty strings (Coinbase auth-gated case) the
        # stream should fall back to last for both sides, mirroring REST.
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertAlmostEqual(cached["bid"], 2200.50)
        self.assertAlmostEqual(cached["ask"], 2200.50)

    def test_ticker_with_no_usable_price_drops_frame(self) -> None:
        stream = CoinbaseMarketStream()
        stream.handle_message(_ticker_frame(price="", bid="", ask=""))
        self.assertIsNone(stream.get_ticker("ETH/USD"))


class TestCandlesCache(unittest.TestCase):
    def _now_min(self) -> int:
        return int(datetime.now(timezone.utc).timestamp()) // 60 * 60

    def test_handle_candles_appends_in_order(self) -> None:
        stream = CoinbaseMarketStream()
        now = self._now_min()
        bars = [
            {"product_id": "ETH-USD", "start": str(now - 120), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
            {"product_id": "ETH-USD", "start": str(now - 60),  "open": "2", "high": "2", "low": "2", "close": "2", "volume": "2"},
            {"product_id": "ETH-USD", "start": str(now),       "open": "3", "high": "3", "low": "3", "close": "3", "volume": "3"},
        ]
        stream.handle_message(_candle_frame(bars=bars))
        candles = stream.get_candles("ETH/USD", limit=3)
        self.assertIsNotNone(candles)
        assert candles is not None
        self.assertEqual([c["open"] for c in candles], [1.0, 2.0, 3.0])

    def test_handle_candles_overwrites_in_progress_bar(self) -> None:
        stream = CoinbaseMarketStream()
        now = self._now_min()
        stream.handle_message(
            _candle_frame(bars=[
                {"product_id": "ETH-USD", "start": str(now), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
            ])
        )
        stream.handle_message(
            _candle_frame(bars=[
                {"product_id": "ETH-USD", "start": str(now), "open": "1", "high": "5", "low": "1", "close": "5", "volume": "9"},
            ])
        )
        candles = stream.get_candles("ETH/USD", limit=1)
        self.assertIsNotNone(candles)
        assert candles is not None
        self.assertEqual(len(candles), 1)
        self.assertAlmostEqual(candles[0]["close"], 5.0)
        self.assertAlmostEqual(candles[0]["volume"], 9.0)

    def test_get_candles_returns_none_when_too_short(self) -> None:
        stream = CoinbaseMarketStream()
        now = self._now_min()
        stream.handle_message(
            _candle_frame(bars=[
                {"product_id": "ETH-USD", "start": str(now), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
            ])
        )
        # Cache has 1 bar; caller wants 5.
        self.assertIsNone(stream.get_candles("ETH/USD", limit=5))

    def test_get_candles_returns_none_when_stale(self) -> None:
        stream = CoinbaseMarketStream(candle_freshness_s=0.0)
        now = self._now_min()
        stream.handle_message(
            _candle_frame(bars=[
                {"product_id": "ETH-USD", "start": str(now - 600), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"},
            ])
        )
        # Newest bar is 10 min old; freshness floor is 0 -> stale.
        self.assertIsNone(stream.get_candles("ETH/USD", limit=1))

    def test_handle_candles_prunes_to_max(self) -> None:
        stream = CoinbaseMarketStream(max_candles=3)
        now = self._now_min()
        for i in range(5):
            stream.handle_message(
                _candle_frame(bars=[{
                    "product_id": "ETH-USD",
                    "start": str(now - (4 - i) * 60),
                    "open": str(i), "high": str(i), "low": str(i),
                    "close": str(i), "volume": str(i),
                }])
            )
        candles = stream.get_candles("ETH/USD", limit=3)
        self.assertIsNotNone(candles)
        assert candles is not None
        # Should be the 3 newest (i=2,3,4)
        self.assertEqual([c["open"] for c in candles], [2.0, 3.0, 4.0])


class TestSubscriptionTracking(unittest.TestCase):
    def test_ensure_subscribed_records_symbol(self) -> None:
        stream = CoinbaseMarketStream()
        stream.ensure_subscribed("ETH/USD")
        stream.ensure_subscribed("BTC-USD")
        # Both should be tracked, deduped, in product-id form.
        self.assertEqual(stream._tracked_products, {"ETH-USD", "BTC-USD"})

    def test_ensure_subscribed_is_idempotent(self) -> None:
        stream = CoinbaseMarketStream()
        stream.ensure_subscribed("ETH/USD")
        stream.ensure_subscribed("ETH/USD")
        self.assertEqual(stream._tracked_products, {"ETH-USD"})


# ---------------------------------------------------------------------------
# CoinbaseExchange integration -- stream replaces REST when available.
# ---------------------------------------------------------------------------


class _FakeStream:
    """Drop-in stream mock with controllable responses."""

    def __init__(
        self,
        *,
        ticker: Optional[Dict[str, Any]] = None,
        candles: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.ticker = ticker
        self.candles = candles
        self.ensure_calls: List[str] = []

    def ensure_subscribed(self, symbol: str) -> None:
        self.ensure_calls.append(symbol)

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.ticker

    def get_candles(self, symbol: str, *, limit: int) -> Optional[List[Dict[str, Any]]]:
        if self.candles is None or len(self.candles) < limit:
            return None
        return self.candles[-limit:]

    def stop(self, *, timeout_s: float = 5.0) -> None:
        pass


class _MinimalCcxt:
    """Stand-in ccxt module so CoinbaseExchange can construct without network."""

    class coinbase:
        def __init__(self, params: Dict[str, Any]) -> None:
            self.params = params

        def set_sandbox_mode(self, on: bool) -> None:
            pass

        def load_markets(self) -> None:
            pass


class TestExchangeStreamWiring(unittest.TestCase):
    def setUp(self) -> None:
        from exchanges import CoinbaseExchange

        self.CoinbaseExchange = CoinbaseExchange

    def test_get_ticker_uses_stream_when_fresh(self) -> None:
        ex = self.CoinbaseExchange(sandbox=True, ccxt_module=_MinimalCcxt())
        ex._stream = _FakeStream(
            ticker={
                "symbol": "ETH/USD",
                "bid": 2100.0,
                "ask": 2101.0,
                "last": 2100.5,
                "volume_24h_base": 99.0,
                "as_of_utc": "2026-05-17T12:00:00Z",
            }
        )
        with mock.patch("exchanges.coinbase.requests.get") as get:
            ticker = ex.get_ticker("ETH/USD")
        self.assertEqual(ticker.bid, 2100.0)
        self.assertEqual(ticker.ask, 2101.0)
        get.assert_not_called()
        # Stream subscription is ensured for the normalised symbol.
        self.assertIn("ETH/USD", ex._stream.ensure_calls)  # type: ignore[union-attr]

    def test_get_ticker_falls_back_to_rest_when_stream_empty(self) -> None:
        ex = self.CoinbaseExchange(sandbox=True, ccxt_module=_MinimalCcxt())
        ex._stream = _FakeStream(ticker=None)
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "best_bid_price": "2100.0",
            "best_ask_price": "2101.0",
            "price": "2100.5",
            "volume_24h": "99.0",
        }
        with mock.patch(
            "exchanges.coinbase.requests.get", return_value=fake_response
        ) as get:
            ticker = ex.get_ticker("ETH/USD")
        self.assertEqual(ticker.bid, 2100.0)
        get.assert_called_once()

    def test_fetch_recent_candles_uses_stream_when_enough_bars(self) -> None:
        ex = self.CoinbaseExchange(sandbox=True, ccxt_module=_MinimalCcxt())
        bars = [
            {
                "timestamp": "2026-05-17T11:58:00+00:00",
                "_unix": 1747483080,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
            },
            {
                "timestamp": "2026-05-17T11:59:00+00:00",
                "_unix": 1747483140,
                "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 2.0,
            },
        ]
        ex._stream = _FakeStream(candles=bars)
        with mock.patch("exchanges.coinbase.requests.get") as get:
            rows = ex.fetch_recent_candles("ETH/USD", limit=2)
        get.assert_not_called()
        self.assertEqual(len(rows), 2)
        # The internal _unix key must be stripped to match REST contract.
        self.assertNotIn("_unix", rows[0])
        self.assertNotIn("_unix", rows[1])
        self.assertEqual(rows[0]["open"], 1.0)
        self.assertEqual(rows[1]["close"], 2.0)

    def test_fetch_recent_candles_falls_back_when_stream_short(self) -> None:
        ex = self.CoinbaseExchange(sandbox=True, ccxt_module=_MinimalCcxt())
        ex._stream = _FakeStream(candles=[])  # empty -> too short
        fake_response = mock.MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "candles": [
                {
                    "start": "1747483140",
                    "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1",
                }
            ]
        }
        with mock.patch(
            "exchanges.coinbase.requests.get", return_value=fake_response
        ) as get:
            rows = ex.fetch_recent_candles("ETH/USD", limit=1)
        get.assert_called_once()
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
