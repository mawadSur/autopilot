"""Tests for src/exchanges/adapters/coinbase_tradeable.py — Lane D D1, Commit 2.

Hermetic: a fake CoinbaseExchange records call shapes and returns canned
responses. No real ccxt or HTTP calls are made.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple

from exchanges.adapters import CoinbaseTradeable
from exchanges.coinbase import Balance, OrderResult, Ticker
from protocols import AssetClass, FeeModel, RiskAttributes, Tradeable


# ---------------------------------------------------------------------------
# Fake CoinbaseExchange
# ---------------------------------------------------------------------------


class _FakeCoinbaseExchange:
    """Records call shapes; returns canned responses. Quacks like CoinbaseExchange."""

    def __init__(self) -> None:
        self.market_calls: List[Tuple[str, str, Dict[str, Any]]] = []
        self.limit_calls: List[Tuple[str, str, Dict[str, Any]]] = []
        self.cancel_calls: List[Tuple[str, str]] = []
        self.open_orders_calls: List[Optional[str]] = []
        self.get_ticker_calls: List[str] = []
        self.get_balances_calls = 0

        self.canned_order = OrderResult(
            order_id="ord-1",
            symbol="ETH/USD",
            side="buy",
            type="market",
            quote_size_usd=100.0,
            base_size=0.05,
            limit_price=None,
            status="filled",
            filled_base=0.05,
            filled_quote_usd=100.0,
            avg_fill_price=2000.0,
            fee_usd=0.6,
            created_at_utc="2026-05-08T00:00:00+00:00",
            raw_payload={},
        )
        self.canned_ticker = Ticker(
            symbol="ETH/USD",
            bid=1999.0,
            ask=2001.0,
            last=2000.0,
            volume_24h_base=10000.0,
            as_of_utc="2026-05-08T00:00:00+00:00",
        )
        self.canned_balances: List[Balance] = [
            Balance(currency="USD", free=900.0, locked=100.0, total=1000.0),
            Balance(currency="ETH", free=0.5, locked=0.0, total=0.5),
        ]
        self.canned_open_orders: List[OrderResult] = []

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        self.market_calls.append(
            (symbol, side, {"quote_size_usd": quote_size_usd, "base_size": base_size})
        )
        return self.canned_order

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        *,
        base_size: float,
        limit_price: float,
    ) -> OrderResult:
        self.limit_calls.append(
            (symbol, side, {"base_size": base_size, "limit_price": limit_price})
        )
        return self.canned_order

    def cancel_order(self, order_id: str, symbol: str) -> OrderResult:
        self.cancel_calls.append((order_id, symbol))
        return self.canned_order

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        self.open_orders_calls.append(symbol)
        return list(self.canned_open_orders)

    def get_ticker(self, symbol: str) -> Ticker:
        self.get_ticker_calls.append(symbol)
        return self.canned_ticker

    def get_balances(self) -> List[Balance]:
        self.get_balances_calls += 1
        return list(self.canned_balances)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoinbaseTradeable(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeCoinbaseExchange()
        self.adapter = CoinbaseTradeable(self.fake, "ETH/USD")  # type: ignore[arg-type]

    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(self.adapter, Tradeable)

    def test_static_metadata(self) -> None:
        self.assertEqual(self.adapter.symbol, "ETH/USD")
        self.assertEqual(self.adapter.asset_class, AssetClass.SPOT_CRYPTO)
        self.assertGreater(self.adapter.tick_size, 0.0)
        self.assertGreater(self.adapter.min_size, 0.0)

    def test_default_fee_model(self) -> None:
        fm = self.adapter.fee_model
        self.assertIsInstance(fm, FeeModel)
        self.assertGreater(fm.taker, 0.0)
        self.assertGreaterEqual(fm.taker, fm.maker)
        self.assertEqual(fm.settlement_fee_bps, 0)
        self.assertEqual(fm.gas_fee_usd, 0.0)

    def test_fee_model_override(self) -> None:
        custom = FeeModel(maker=0.0, taker=0.001)
        adapter = CoinbaseTradeable(self.fake, "BTC/USD", fee_model=custom)  # type: ignore[arg-type]
        self.assertIs(adapter.fee_model, custom)

    def test_tick_and_min_size_overrides(self) -> None:
        adapter = CoinbaseTradeable(
            self.fake, "BTC/USD", tick_size=0.5, min_size=0.0001  # type: ignore[arg-type]
        )
        self.assertEqual(adapter.tick_size, 0.5)
        self.assertEqual(adapter.min_size, 0.0001)

    # ---- delegation -----------------------------------------------------

    def test_place_market_order_delegates(self) -> None:
        result = self.adapter.place_market_order(side="buy", quote_size_usd=100.0)
        self.assertEqual(result.order_id, "ord-1")
        self.assertEqual(len(self.fake.market_calls), 1)
        symbol, side, kwargs = self.fake.market_calls[0]
        self.assertEqual(symbol, "ETH/USD")
        self.assertEqual(side, "buy")
        self.assertEqual(kwargs["quote_size_usd"], 100.0)
        self.assertIsNone(kwargs["base_size"])

    def test_place_market_order_base_size(self) -> None:
        self.adapter.place_market_order(side="sell", base_size=0.25)
        self.assertEqual(len(self.fake.market_calls), 1)
        _, side, kwargs = self.fake.market_calls[0]
        self.assertEqual(side, "sell")
        self.assertEqual(kwargs["base_size"], 0.25)
        self.assertIsNone(kwargs["quote_size_usd"])

    def test_place_limit_order_delegates(self) -> None:
        self.adapter.place_limit_order(side="buy", base_size=0.1, limit_price=1995.0)
        self.assertEqual(len(self.fake.limit_calls), 1)
        symbol, side, kwargs = self.fake.limit_calls[0]
        self.assertEqual(symbol, "ETH/USD")
        self.assertEqual(side, "buy")
        self.assertEqual(kwargs["base_size"], 0.1)
        self.assertEqual(kwargs["limit_price"], 1995.0)

    def test_cancel_order_delegates(self) -> None:
        self.adapter.cancel_order("ord-42")
        self.assertEqual(self.fake.cancel_calls, [("ord-42", "ETH/USD")])

    def test_get_open_orders_delegates_with_symbol(self) -> None:
        self.fake.canned_open_orders = [self.fake.canned_order]
        result = self.adapter.get_open_orders()
        self.assertEqual(len(result), 1)
        # Adapter MUST scope to its own symbol (callers shouldn't see other markets).
        self.assertEqual(self.fake.open_orders_calls, ["ETH/USD"])

    def test_get_ticker_delegates_with_symbol(self) -> None:
        ticker = self.adapter.get_ticker()
        self.assertEqual(ticker.symbol, "ETH/USD")
        self.assertEqual(self.fake.get_ticker_calls, ["ETH/USD"])

    def test_get_balances_collapses_to_dict(self) -> None:
        balances = self.adapter.get_balances()
        # Tradeable contract is Dict[str, float] keyed by currency, with totals
        self.assertEqual(balances, {"USD": 1000.0, "ETH": 0.5})
        self.assertEqual(self.fake.get_balances_calls, 1)

    # ---- risk attributes ------------------------------------------------

    def test_risk_attributes_basic(self) -> None:
        ra = self.adapter.risk_attributes(
            side="buy", size_base=0.5, entry_price=2000.0
        )
        self.assertIsInstance(ra, RiskAttributes)
        self.assertEqual(ra.notional_exposure_usd, 1000.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_risk_attributes_zero_size(self) -> None:
        ra = self.adapter.risk_attributes(
            side="sell", size_base=0.0, entry_price=2000.0
        )
        self.assertEqual(ra.notional_exposure_usd, 0.0)
        self.assertEqual(ra.kelly_divisor, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
