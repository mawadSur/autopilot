"""Tests for src/exchanges/adapters/hyperliquid_tradeable.py — Lane D D1, Commit 3.

Hermetic: a fake HyperliquidExchange records call shapes and returns canned
responses. No real HTTP calls are made.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional

# Import via the submodule explicitly — D1 commit 3 deliberately does NOT
# touch src/exchanges/adapters/__init__.py (D2 will rewire exports). The
# adapter is fully usable via its module path today.
from exchanges.adapters.hyperliquid_tradeable import HyperliquidTradeable
from exchanges.coinbase import Balance, OrderResult, Ticker
from exchanges.hyperliquid import PerpPosition
from protocols import AssetClass, FeeModel, RiskAttributes, Tradeable


# ---------------------------------------------------------------------------
# Fake HyperliquidExchange
# ---------------------------------------------------------------------------


class _FakeHyperliquidExchange:
    """Records call shapes; returns canned responses. Quacks like HyperliquidExchange."""

    def __init__(self) -> None:
        self.get_ticker_calls: List[str] = []
        self.get_balances_calls = 0
        self.open_orders_calls: List[Optional[str]] = []
        self.open_positions_calls = 0

        self.canned_ticker = Ticker(
            symbol="ETH",
            bid=1999.0,
            ask=2001.0,
            last=2000.0,
            volume_24h_base=10000.0,
            as_of_utc="2026-05-08T00:00:00+00:00",
        )
        self.canned_balances: List[Balance] = [
            Balance(currency="USDC", free=900.0, locked=100.0, total=1000.0),
        ]
        self.canned_open_orders: List[OrderResult] = []
        self.canned_open_positions: List[PerpPosition] = []
        self.open_positions_should_raise = False

    def get_ticker(self, symbol: str) -> Ticker:
        self.get_ticker_calls.append(symbol)
        return self.canned_ticker

    def get_balances(self) -> List[Balance]:
        self.get_balances_calls += 1
        return list(self.canned_balances)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResult]:
        self.open_orders_calls.append(symbol)
        return list(self.canned_open_orders)

    def get_open_positions(self) -> List[PerpPosition]:
        self.open_positions_calls += 1
        if self.open_positions_should_raise:
            raise RuntimeError("clearinghouseState requires wallet")
        return list(self.canned_open_positions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHyperliquidTradeable(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeHyperliquidExchange()
        self.adapter = HyperliquidTradeable(self.fake, "ETH")  # type: ignore[arg-type]

    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(self.adapter, Tradeable)

    def test_static_metadata(self) -> None:
        self.assertEqual(self.adapter.symbol, "ETH")
        self.assertEqual(self.adapter.asset_class, AssetClass.PERP_CRYPTO)
        self.assertGreater(self.adapter.tick_size, 0.0)
        self.assertGreater(self.adapter.min_size, 0.0)

    def test_default_fee_model(self) -> None:
        fm = self.adapter.fee_model
        self.assertIsInstance(fm, FeeModel)
        self.assertGreater(fm.taker, 0.0)
        self.assertGreaterEqual(fm.taker, fm.maker)
        self.assertEqual(fm.settlement_fee_bps, 0)

    def test_fee_model_override(self) -> None:
        custom = FeeModel(maker=0.0, taker=0.0001)
        adapter = HyperliquidTradeable(self.fake, "BTC", fee_model=custom)  # type: ignore[arg-type]
        self.assertIs(adapter.fee_model, custom)

    # ---- read-only delegation ------------------------------------------

    def test_get_ticker_delegates(self) -> None:
        ticker = self.adapter.get_ticker()
        self.assertEqual(ticker.symbol, "ETH")
        self.assertEqual(self.fake.get_ticker_calls, ["ETH"])

    def test_get_balances_collapses_to_dict(self) -> None:
        balances = self.adapter.get_balances()
        self.assertEqual(balances, {"USDC": 1000.0})
        self.assertEqual(self.fake.get_balances_calls, 1)

    def test_get_open_orders_delegates_with_symbol(self) -> None:
        result = self.adapter.get_open_orders()
        self.assertEqual(result, [])
        self.assertEqual(self.fake.open_orders_calls, ["ETH"])

    # ---- write methods raise -------------------------------------------

    def test_place_market_order_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.adapter.place_market_order(side="buy", quote_size_usd=100.0)

    def test_place_market_order_with_base_size_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.adapter.place_market_order(side="sell", base_size=0.1)

    def test_place_limit_order_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.adapter.place_limit_order(
                side="buy", base_size=0.1, limit_price=1995.0
            )

    def test_cancel_order_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.adapter.cancel_order("ord-1")

    # ---- risk attributes ------------------------------------------------

    def test_risk_attributes_no_open_position(self) -> None:
        ra = self.adapter.risk_attributes(
            side="buy", size_base=0.5, entry_price=2000.0
        )
        self.assertIsInstance(ra, RiskAttributes)
        self.assertEqual(ra.notional_exposure_usd, 1000.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        # No matching position exists, so liq_price / margin remain None.
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)
        self.assertEqual(self.fake.open_positions_calls, 1)

    def test_risk_attributes_with_open_position(self) -> None:
        self.fake.canned_open_positions = [
            PerpPosition(
                symbol="ETH",
                side="long",
                size_base=1.0,
                entry_price=1900.0,
                mark_price=2000.0,
                unrealized_pnl_usd=100.0,
                liquidation_price=1500.0,
                leverage=5.0,
                margin_used_usd=380.0,
            )
        ]
        ra = self.adapter.risk_attributes(
            side="buy", size_base=0.5, entry_price=2000.0
        )
        self.assertEqual(ra.notional_exposure_usd, 1000.0)
        self.assertEqual(ra.liquidation_price, 1500.0)
        self.assertEqual(ra.margin_used_usd, 380.0)

    def test_risk_attributes_position_on_other_symbol_ignored(self) -> None:
        self.fake.canned_open_positions = [
            PerpPosition(
                symbol="BTC",
                side="long",
                size_base=0.1,
                entry_price=50000.0,
                mark_price=51000.0,
                unrealized_pnl_usd=100.0,
                liquidation_price=40000.0,
                leverage=3.0,
                margin_used_usd=1666.0,
            )
        ]
        ra = self.adapter.risk_attributes(
            side="buy", size_base=0.5, entry_price=2000.0
        )
        # Adapter is bound to ETH; the BTC position must NOT leak.
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_risk_attributes_swallows_clearinghouse_failure(self) -> None:
        # When the client throws (e.g. wallet address not configured), risk
        # sizing for notional exposure should still succeed — only the
        # perp-only fields drop to None.
        self.fake.open_positions_should_raise = True
        ra = self.adapter.risk_attributes(
            side="buy", size_base=0.25, entry_price=4000.0
        )
        self.assertEqual(ra.notional_exposure_usd, 1000.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
