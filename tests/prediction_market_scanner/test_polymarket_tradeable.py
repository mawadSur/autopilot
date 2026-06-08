"""Tests for src/exchanges/adapters/polymarket_tradeable.py — Lane D D2, Commit 1.

Hermetic: a fake fetcher returns canned market data; trade-execution
logs are written into a tempdir. No real Polymarket Gamma API calls.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from exchanges.adapters import PolymarketTradeable
from exchanges.coinbase import OrderResult, Ticker
from protocols import AssetClass, FeeModel, RiskAttributes, Tradeable


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeMarket:
    """Quacks like ``models.Market`` for the fields PolymarketTradeable reads."""

    market_id: str
    bid_price: float = 0.0
    ask_price: float = 0.0
    implied_prob: float = 0.5
    volume_24h: float = 0.0


class _FakeFetcher:
    """Minimal Polymarket fetcher stub.

    Surfaces only the optional ``fetch_market`` / ``get_balances`` shims
    PolymarketTradeable probes for. Tests instantiate with the canned
    market and balances they want to return.
    """

    def __init__(
        self,
        *,
        market: Optional[_FakeMarket] = None,
        balances: Optional[Dict[str, float]] = None,
        raise_on_fetch: Optional[Exception] = None,
        raise_on_balances: Optional[Exception] = None,
    ) -> None:
        self._market = market
        self._balances = balances
        self.fetch_calls: List[str] = []
        self.balances_calls = 0
        self._raise_on_fetch = raise_on_fetch
        self._raise_on_balances = raise_on_balances

    def fetch_market(self, market_id: str) -> Optional[_FakeMarket]:
        self.fetch_calls.append(market_id)
        if self._raise_on_fetch is not None:
            raise self._raise_on_fetch
        return self._market

    def get_balances(self) -> Dict[str, float]:
        self.balances_calls += 1
        if self._raise_on_balances is not None:
            raise self._raise_on_balances
        return dict(self._balances or {})


class _NoMethodFetcher:
    """Fetcher with NEITHER fetch_market NOR get_balances. Stub-fallback path."""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPolymarketTradeableProtocolConformance(unittest.TestCase):
    def test_runtime_isinstance_check(self) -> None:
        fetcher = _FakeFetcher()
        adapter = PolymarketTradeable("mkt-1", fetcher)
        self.assertIsInstance(adapter, Tradeable)

    def test_static_metadata(self) -> None:
        adapter = PolymarketTradeable("mkt-42", _FakeFetcher())
        self.assertEqual(adapter.symbol, "polymarket:mkt-42")
        self.assertEqual(adapter.market_id, "mkt-42")
        self.assertEqual(adapter.asset_class, AssetClass.PREDICTION_BINARY)
        self.assertEqual(adapter.tick_size, 0.01)
        self.assertEqual(adapter.min_size, 1.0)

    def test_default_fee_model(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        fm = adapter.fee_model
        self.assertIsInstance(fm, FeeModel)
        self.assertEqual(fm.maker, 0.0)
        self.assertEqual(fm.taker, 0.0)
        self.assertEqual(fm.settlement_fee_bps, 200)
        self.assertEqual(fm.gas_fee_usd, 0.0)

    def test_fee_model_overrides(self) -> None:
        adapter = PolymarketTradeable(
            "mkt-1", _FakeFetcher(), fee_bps=350, gas_fee_usd=0.42
        )
        self.assertEqual(adapter.fee_model.settlement_fee_bps, 350)
        self.assertEqual(adapter.fee_model.gas_fee_usd, 0.42)

    def test_empty_market_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PolymarketTradeable("", _FakeFetcher())


class TestPolymarketTradeableGetTicker(unittest.TestCase):
    def test_uses_fetcher_market_when_available(self) -> None:
        market = _FakeMarket(
            market_id="mkt-1",
            bid_price=0.62,
            ask_price=0.66,
            implied_prob=0.64,
            volume_24h=12345.0,
        )
        fetcher = _FakeFetcher(market=market)
        adapter = PolymarketTradeable("mkt-1", fetcher)
        ticker = adapter.get_ticker()
        self.assertIsInstance(ticker, Ticker)
        self.assertEqual(ticker.symbol, "polymarket:mkt-1")
        self.assertEqual(ticker.bid, 0.62)
        self.assertEqual(ticker.ask, 0.66)
        self.assertEqual(ticker.last, 0.64)
        self.assertEqual(ticker.volume_24h_base, 12345.0)
        self.assertEqual(fetcher.fetch_calls, ["mkt-1"])

    def test_returns_stub_when_fetcher_returns_none(self) -> None:
        fetcher = _FakeFetcher(market=None)
        adapter = PolymarketTradeable("mkt-1", fetcher)
        ticker = adapter.get_ticker()
        # Stub: zero quote sides, last=0.5 (max-uncertainty marker).
        self.assertEqual(ticker.bid, 0.0)
        self.assertEqual(ticker.ask, 0.0)
        self.assertEqual(ticker.last, 0.5)

    def test_returns_stub_when_fetcher_has_no_method(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _NoMethodFetcher())
        ticker = adapter.get_ticker()
        self.assertEqual(ticker.last, 0.5)
        self.assertEqual(ticker.symbol, "polymarket:mkt-1")

    def test_returns_stub_when_fetcher_raises(self) -> None:
        fetcher = _FakeFetcher(raise_on_fetch=RuntimeError("boom"))
        adapter = PolymarketTradeable("mkt-1", fetcher)
        ticker = adapter.get_ticker()
        # Best-effort: error logged, stub returned.
        self.assertEqual(ticker.last, 0.5)


class TestPolymarketTradeableGetBalances(unittest.TestCase):
    def test_returns_empty_when_fetcher_has_no_method(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _NoMethodFetcher())
        self.assertEqual(adapter.get_balances(), {})

    def test_returns_balances_when_fetcher_provides(self) -> None:
        fetcher = _FakeFetcher(balances={"USDC": 4321.5})
        adapter = PolymarketTradeable("mkt-1", fetcher)
        self.assertEqual(adapter.get_balances(), {"USDC": 4321.5})
        self.assertEqual(fetcher.balances_calls, 1)

    def test_returns_empty_when_fetcher_raises(self) -> None:
        fetcher = _FakeFetcher(raise_on_balances=RuntimeError("auth"))
        adapter = PolymarketTradeable("mkt-1", fetcher)
        self.assertEqual(adapter.get_balances(), {})


class TestPolymarketTradeablePlaceMarketOrder(unittest.TestCase):
    def test_writes_trade_execution_log_and_returns_open_order(self) -> None:
        market = _FakeMarket(
            market_id="mkt-1",
            bid_price=0.64,
            ask_price=0.66,
            implied_prob=0.65,
        )
        fetcher = _FakeFetcher(market=market)
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            adapter = PolymarketTradeable(
                "mkt-1", fetcher, trade_store_dir=store_dir
            )
            result = adapter.place_market_order(
                side="buy", quote_size_usd=100.0
            )

            log_path = store_dir / "trade_execution_mkt-1.json"
            self.assertTrue(log_path.exists())
            payload = json.loads(log_path.read_text())
            self.assertEqual(payload["event_id"], "mkt-1")
            self.assertEqual(payload["status"], "open")
            self.assertEqual(payload["source"], "supervisor_polymarket_tradeable")
            self.assertEqual(payload["entry_price"], 0.66)
            self.assertEqual(payload["position_size_usd"], 100.0)

            self.assertIsInstance(result, OrderResult)
            self.assertEqual(result.order_id, "trade_execution_mkt-1.json")
            self.assertEqual(result.status, "open")
            self.assertEqual(result.symbol, "polymarket:mkt-1")
            self.assertEqual(result.filled_quote_usd, 100.0)
            self.assertAlmostEqual(result.filled_base, 100.0 / 0.66)
            self.assertEqual(result.avg_fill_price, 0.66)

    def test_uses_log_writer_seam(self) -> None:
        captured: Dict[str, Any] = {}

        def writer(*, event_payload: Dict[str, Any], market_id: str) -> Path:
            captured["payload"] = event_payload
            captured["market_id"] = market_id
            return Path("/tmp/fake_trade_execution.json")

        market = _FakeMarket(
            market_id="mkt-1",
            bid_price=0.40,
            ask_price=0.42,
            implied_prob=0.41,
        )
        adapter = PolymarketTradeable(
            "mkt-1", _FakeFetcher(market=market), log_writer=writer
        )
        result = adapter.place_market_order(side="buy", base_size=10.0)
        self.assertEqual(captured["market_id"], "mkt-1")
        self.assertEqual(captured["payload"]["event_id"], "mkt-1")
        self.assertEqual(result.order_id, "fake_trade_execution.json")
        # base_size=10 at ask=0.42 => quote = 4.2
        self.assertAlmostEqual(result.filled_quote_usd, 4.2, places=6)

    def test_requires_size_argument(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        with self.assertRaises(ValueError):
            adapter.place_market_order(side="buy")

    def test_sell_uses_bid_side(self) -> None:
        market = _FakeMarket(
            market_id="mkt-1",
            bid_price=0.55,
            ask_price=0.60,
            implied_prob=0.575,
        )
        with tempfile.TemporaryDirectory() as tmp:
            adapter = PolymarketTradeable(
                "mkt-1", _FakeFetcher(market=market), trade_store_dir=Path(tmp)
            )
            result = adapter.place_market_order(
                side="sell", quote_size_usd=55.0
            )
            self.assertEqual(result.avg_fill_price, 0.55)


class TestPolymarketTradeableLimitAndCancel(unittest.TestCase):
    def test_place_limit_order_raises_not_implemented(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        with self.assertRaises(NotImplementedError):
            adapter.place_limit_order(side="buy", base_size=1.0, limit_price=0.5)

    def test_cancel_order_raises_not_implemented(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        with self.assertRaises(NotImplementedError):
            adapter.cancel_order("ord-1")


class TestPolymarketTradeableGetOpenOrders(unittest.TestCase):
    def test_returns_empty_when_no_log_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = PolymarketTradeable(
                "mkt-1", _FakeFetcher(), trade_store_dir=Path(tmp)
            )
            self.assertEqual(adapter.get_open_orders(), [])

    def test_returns_open_log_entry(self) -> None:
        market = _FakeMarket(
            market_id="mkt-1",
            bid_price=0.40,
            ask_price=0.42,
            implied_prob=0.41,
        )
        with tempfile.TemporaryDirectory() as tmp:
            adapter = PolymarketTradeable(
                "mkt-1", _FakeFetcher(market=market), trade_store_dir=Path(tmp)
            )
            adapter.place_market_order(side="buy", quote_size_usd=42.0)
            open_orders = adapter.get_open_orders()
            self.assertEqual(len(open_orders), 1)
            self.assertEqual(open_orders[0].symbol, "polymarket:mkt-1")
            self.assertEqual(open_orders[0].status, "open")
            self.assertEqual(open_orders[0].order_id, "trade_execution_mkt-1.json")

    def test_skips_settled_log_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp)
            payload = {
                "event_id": "mkt-1",
                "status": "settled",
                "position_size_usd": 100.0,
                "entry_price": 0.5,
            }
            (store_dir / "trade_execution_mkt-1.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            adapter = PolymarketTradeable(
                "mkt-1", _FakeFetcher(), trade_store_dir=store_dir
            )
            self.assertEqual(adapter.get_open_orders(), [])


class TestPolymarketTradeableRiskAttributes(unittest.TestCase):
    def test_basic_kelly_divisor_and_notional(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        ra = adapter.risk_attributes(
            side="buy", size_base=100.0, entry_price=0.65
        )
        self.assertIsInstance(ra, RiskAttributes)
        # p*(1-p) = 0.65 * 0.35 = 0.2275
        self.assertAlmostEqual(ra.kelly_divisor, 0.2275, places=6)
        # notional = 100 * 0.65 = 65
        self.assertAlmostEqual(ra.notional_exposure_usd, 65.0, places=6)
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_edge_low(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        ra = adapter.risk_attributes(
            side="buy", size_base=10.0, entry_price=0.02
        )
        # 0.02 * 0.98 = 0.0196 — small but positive.
        self.assertGreater(ra.kelly_divisor, 0.0)
        self.assertAlmostEqual(ra.kelly_divisor, 0.0196, places=6)

    def test_edge_high(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        ra = adapter.risk_attributes(
            side="buy", size_base=10.0, entry_price=0.98
        )
        # 0.98 * 0.02 = 0.0196 — symmetric to the low edge.
        self.assertGreater(ra.kelly_divisor, 0.0)
        self.assertAlmostEqual(ra.kelly_divisor, 0.0196, places=6)

    def test_p_at_half_max_divisor(self) -> None:
        adapter = PolymarketTradeable("mkt-1", _FakeFetcher())
        ra = adapter.risk_attributes(side="buy", size_base=1.0, entry_price=0.5)
        # 0.5 * 0.5 = 0.25 (the maximum of p*(1-p))
        self.assertAlmostEqual(ra.kelly_divisor, 0.25, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
