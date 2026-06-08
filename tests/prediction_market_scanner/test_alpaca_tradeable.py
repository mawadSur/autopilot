"""Tests for src/exchanges/adapters/alpaca_tradeable.py — Phase P3 commit 2.

Hermetic: a fake AlpacaExchange records call shapes and returns canned
responses. No real HTTP calls are made.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

from exchanges.adapters import AlpacaTradeable
from exchanges.adapters.alpaca_tradeable import AlpacaTradeable as DirectAlpacaTradeable
from exchanges.coinbase import OrderResult, Ticker
from protocols import AssetClass, FeeModel, RiskAttributes, Tradeable


# ---------------------------------------------------------------------------
# Fake AlpacaExchange
# ---------------------------------------------------------------------------


class _FakeAlpacaExchange:
    """Records call shapes; returns canned responses. Quacks like AlpacaExchange."""

    def __init__(self) -> None:
        self.get_ticker_calls: List[str] = []
        self.get_balances_calls = 0
        self.get_open_orders_calls = 0
        self.place_market_calls: List[Dict[str, Any]] = []
        self.place_limit_calls: List[Dict[str, Any]] = []
        self.cancel_order_calls: List[str] = []

        self.canned_ticker = Ticker(
            symbol="AAPL",
            bid=180.00,
            ask=180.05,
            last=180.02,
            volume_24h_base=1500.0,
            as_of_utc="2026-05-08T15:00:00+00:00",
        )
        self.canned_balances: Dict[str, float] = {"USD": 25000.0}
        self.canned_open_orders: List[OrderResult] = []
        self.canned_market_order = OrderResult(
            order_id="ord-99",
            symbol="AAPL",
            side="buy",
            type="market",
            base_size=10.0,
            status="open",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc="2026-05-08T15:00:00+00:00",
            raw_payload={},
        )
        self.canned_limit_order = OrderResult(
            order_id="ord-lim-1",
            symbol="AAPL",
            side="sell",
            type="limit",
            base_size=5.0,
            limit_price=200.0,
            status="open",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc="2026-05-08T15:00:00+00:00",
            raw_payload={},
        )
        self.canned_cancel_result = OrderResult(
            order_id="ord-1",
            symbol="",
            side="buy",
            type="market",
            status="cancelled",
            filled_base=0.0,
            filled_quote_usd=0.0,
            avg_fill_price=None,
            fee_usd=0.0,
            created_at_utc="2026-05-08T15:00:00+00:00",
            raw_payload={},
        )

        # If True, write methods raise NotImplementedError to simulate
        # the connector's flag-gated behaviour.
        self.flag_enabled: bool = False

    # ---- the gating mirrors AlpacaExchange's behaviour --------------

    def _check_flag(self) -> None:
        if not self.flag_enabled:
            raise NotImplementedError(
                "Alpaca trading is feature-flag-gated; set ALPACA_TRADING_ENABLED=true"
            )

    # ---- read paths -------------------------------------------------

    def get_ticker(self, symbol: str) -> Ticker:
        self.get_ticker_calls.append(symbol)
        return self.canned_ticker

    def get_balances(self) -> Dict[str, float]:
        self.get_balances_calls += 1
        return dict(self.canned_balances)

    def get_open_orders(self) -> List[OrderResult]:
        self.get_open_orders_calls += 1
        return list(self.canned_open_orders)

    # ---- write paths -----------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        self._check_flag()
        self.place_market_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "quote_size_usd": quote_size_usd,
                "base_size": base_size,
            }
        )
        return self.canned_market_order

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        *,
        base_size: float,
        limit_price: float,
    ) -> OrderResult:
        self._check_flag()
        self.place_limit_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "base_size": base_size,
                "limit_price": limit_price,
            }
        )
        return self.canned_limit_order

    def cancel_order(self, order_id: str) -> OrderResult:
        self._check_flag()
        self.cancel_order_calls.append(order_id)
        return self.canned_cancel_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlpacaTradeable(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeAlpacaExchange()
        # Use the package-level export to confirm the __init__.py wiring.
        self.adapter = AlpacaTradeable(self.fake, "aapl")  # type: ignore[arg-type]

    def test_init_export_alias_matches_direct_import(self) -> None:
        # Both import paths must resolve to the same class.
        self.assertIs(AlpacaTradeable, DirectAlpacaTradeable)

    def test_protocol_conformance(self) -> None:
        self.assertIsInstance(self.adapter, Tradeable)

    def test_static_metadata(self) -> None:
        # symbol is namespaced; raw_symbol is bare and uppercased.
        self.assertEqual(self.adapter.symbol, "alpaca:AAPL")
        self.assertEqual(self.adapter.raw_symbol, "AAPL")
        self.assertEqual(self.adapter.asset_class, AssetClass.SPOT_EQUITY)
        self.assertEqual(self.adapter.tick_size, 0.01)
        self.assertEqual(self.adapter.min_size, 1.0)

    def test_asset_class_enum_value(self) -> None:
        # Confirm the enum value string itself.
        self.assertEqual(AssetClass.SPOT_EQUITY.value, "spot_equity")

    def test_default_fee_model_is_zero(self) -> None:
        fm = self.adapter.fee_model
        self.assertIsInstance(fm, FeeModel)
        self.assertEqual(fm.maker, 0.0)
        self.assertEqual(fm.taker, 0.0)
        self.assertEqual(fm.settlement_fee_bps, 0)
        self.assertEqual(fm.gas_fee_usd, 0.0)

    def test_fee_model_override(self) -> None:
        custom = FeeModel(maker=0.0, taker=0.0001)
        adapter = AlpacaTradeable(self.fake, "TSLA", fee_model=custom)  # type: ignore[arg-type]
        self.assertIs(adapter.fee_model, custom)

    def test_min_size_override_for_fractional(self) -> None:
        adapter = AlpacaTradeable(
            self.fake, "AAPL", min_size=0.0001  # type: ignore[arg-type]
        )
        self.assertEqual(adapter.min_size, 0.0001)

    def test_invalid_symbol_raises(self) -> None:
        with self.assertRaises(ValueError):
            AlpacaTradeable(self.fake, "")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AlpacaTradeable(self.fake, None)  # type: ignore[arg-type]

    def test_exchange_property_returns_underlying(self) -> None:
        self.assertIs(self.adapter.exchange, self.fake)

    # ---- read-only delegation ------------------------------------------

    def test_get_ticker_delegates_with_raw_symbol(self) -> None:
        ticker = self.adapter.get_ticker()
        self.assertEqual(ticker.symbol, "AAPL")
        # Adapter passes the bare ticker, not the namespaced form.
        self.assertEqual(self.fake.get_ticker_calls, ["AAPL"])

    def test_get_balances_returns_dict(self) -> None:
        balances = self.adapter.get_balances()
        self.assertEqual(balances, {"USD": 25000.0})
        self.assertEqual(self.fake.get_balances_calls, 1)

    def test_get_open_orders_delegates(self) -> None:
        result = self.adapter.get_open_orders()
        self.assertEqual(result, [])
        self.assertEqual(self.fake.get_open_orders_calls, 1)

    # ---- write methods: gating owned by AlpacaExchange -----------------

    def test_place_market_order_propagates_not_implemented_when_flag_unset(self) -> None:
        # Flag default = False.
        self.fake.flag_enabled = False
        with self.assertRaises(NotImplementedError) as cm:
            self.adapter.place_market_order(side="buy", base_size=10.0)
        self.assertIn("ALPACA_TRADING_ENABLED", str(cm.exception))
        # Adapter should not have recorded a call (raised before delegating
        # body — actually delegated, but the connector raised).
        self.assertEqual(self.fake.place_market_calls, [])

    def test_place_market_order_succeeds_when_flag_enabled(self) -> None:
        self.fake.flag_enabled = True
        result = self.adapter.place_market_order(side="buy", base_size=10.0)
        self.assertIsInstance(result, OrderResult)
        self.assertEqual(result.order_id, "ord-99")
        self.assertEqual(self.fake.place_market_calls, [
            {
                "symbol": "AAPL",
                "side": "buy",
                "quote_size_usd": None,
                "base_size": 10.0,
            }
        ])

    def test_place_market_order_with_notional(self) -> None:
        self.fake.flag_enabled = True
        self.adapter.place_market_order(side="buy", quote_size_usd=1500.0)
        call = self.fake.place_market_calls[-1]
        self.assertEqual(call["quote_size_usd"], 1500.0)
        self.assertIsNone(call["base_size"])

    def test_place_limit_order_gated_then_succeeds(self) -> None:
        self.fake.flag_enabled = False
        with self.assertRaises(NotImplementedError):
            self.adapter.place_limit_order(
                side="sell", base_size=5.0, limit_price=200.0
            )
        self.fake.flag_enabled = True
        result = self.adapter.place_limit_order(
            side="sell", base_size=5.0, limit_price=200.0
        )
        self.assertEqual(result.order_id, "ord-lim-1")
        call = self.fake.place_limit_calls[-1]
        self.assertEqual(call["symbol"], "AAPL")
        self.assertEqual(call["side"], "sell")
        self.assertEqual(call["limit_price"], 200.0)

    def test_cancel_order_gated_then_succeeds(self) -> None:
        self.fake.flag_enabled = False
        with self.assertRaises(NotImplementedError):
            self.adapter.cancel_order("ord-1")
        self.fake.flag_enabled = True
        result = self.adapter.cancel_order("ord-1")
        self.assertEqual(result.status, "cancelled")
        self.assertEqual(self.fake.cancel_order_calls, ["ord-1"])

    # ---- risk attributes ------------------------------------------------

    def test_risk_attributes_basic(self) -> None:
        ra = self.adapter.risk_attributes(
            side="buy", size_base=10, entry_price=180
        )
        self.assertIsInstance(ra, RiskAttributes)
        self.assertEqual(ra.notional_exposure_usd, 1800.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        # Cash equity has no leverage / liquidation in V1.
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_risk_attributes_short_side_same_shape(self) -> None:
        ra = self.adapter.risk_attributes(
            side="sell", size_base=4.5, entry_price=200.0
        )
        self.assertAlmostEqual(ra.notional_exposure_usd, 900.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        # Short selling on Alpaca is margin-only — but margin estimation
        # is not yet wired (see module docstring TODO).
        self.assertIsNone(ra.margin_used_usd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
