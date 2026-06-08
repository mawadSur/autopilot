"""Tests for src/protocols/tradeable.py — Lane D Sub-agent D1, Commit 1.

Covers the AssetClass enum, FeeModel + RiskAttributes dataclasses, and the
runtime_checkable behaviour of the Tradeable Protocol. Hermetic — no I/O.
"""

from __future__ import annotations

import dataclasses
import unittest
from typing import Dict, List, Literal, Optional

from protocols import (
    AssetClass,
    FeeModel,
    RiskAttributes,
    Tradeable,
)
from protocols.tradeable import Tradeable as TradeableImport


class _StubTradeable:
    """Minimal stand-in implementing every Tradeable attribute.

    Used to verify ``isinstance(obj, Tradeable)`` returns True for any
    object with the documented surface. Method bodies return canned values
    sufficient to satisfy attribute presence checks.
    """

    @property
    def symbol(self) -> str:
        return "ETH/USD"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.SPOT_CRYPTO

    @property
    def tick_size(self) -> float:
        return 0.01

    @property
    def min_size(self) -> float:
        return 0.0001

    @property
    def fee_model(self) -> FeeModel:
        return FeeModel(maker=0.001, taker=0.0015)

    def get_ticker(self):  # type: ignore[no-untyped-def]
        return None

    def get_balances(self) -> Dict[str, float]:
        return {}

    def place_market_order(
        self,
        side: Literal["buy", "sell"],
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ):  # type: ignore[no-untyped-def]
        return None

    def place_limit_order(
        self,
        side: Literal["buy", "sell"],
        *,
        base_size: float,
        limit_price: float,
    ):  # type: ignore[no-untyped-def]
        return None

    def cancel_order(self, order_id: str):  # type: ignore[no-untyped-def]
        return None

    def get_open_orders(self) -> List:
        return []

    def risk_attributes(
        self,
        *,
        side: Literal["buy", "sell"],
        size_base: float,
        entry_price: float,
    ) -> RiskAttributes:
        return RiskAttributes(kelly_divisor=1.0, notional_exposure_usd=0.0)


class _IncompleteTradeable:
    """Missing several Tradeable methods on purpose."""

    @property
    def symbol(self) -> str:
        return "ETH/USD"

    # Missing: asset_class, tick_size, min_size, fee_model, all order ops, ...


class TestAssetClassEnum(unittest.TestCase):
    def test_three_expected_values(self) -> None:
        # D1's original three asset classes.
        self.assertEqual(AssetClass.SPOT_CRYPTO.value, "spot_crypto")
        self.assertEqual(AssetClass.PERP_CRYPTO.value, "perp_crypto")
        self.assertEqual(AssetClass.PREDICTION_BINARY.value, "prediction_binary")
        # Phase P3 added SPOT_EQUITY for the Alpaca stocks adapter.
        self.assertEqual(AssetClass.SPOT_EQUITY.value, "spot_equity")
        self.assertEqual(len(list(AssetClass)), 4)

    def test_distinct_members(self) -> None:
        self.assertNotEqual(AssetClass.SPOT_CRYPTO, AssetClass.PERP_CRYPTO)
        self.assertNotEqual(AssetClass.PERP_CRYPTO, AssetClass.PREDICTION_BINARY)
        self.assertNotEqual(AssetClass.SPOT_EQUITY, AssetClass.SPOT_CRYPTO)


class TestFeeModel(unittest.TestCase):
    def test_basic_round_trip(self) -> None:
        fm = FeeModel(maker=0.001, taker=0.0015)
        self.assertEqual(fm.maker, 0.001)
        self.assertEqual(fm.taker, 0.0015)
        self.assertEqual(fm.settlement_fee_bps, 0)
        self.assertEqual(fm.gas_fee_usd, 0.0)

    def test_settlement_fee_and_gas_set(self) -> None:
        fm = FeeModel(
            maker=0.0,
            taker=0.0,
            settlement_fee_bps=200,
            gas_fee_usd=1.25,
        )
        self.assertEqual(fm.settlement_fee_bps, 200)
        self.assertEqual(fm.gas_fee_usd, 1.25)

    def test_frozen(self) -> None:
        fm = FeeModel(maker=0.001, taker=0.0015)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            fm.maker = 0.999  # type: ignore[misc]

    def test_equality(self) -> None:
        a = FeeModel(maker=0.001, taker=0.0015)
        b = FeeModel(maker=0.001, taker=0.0015)
        c = FeeModel(maker=0.001, taker=0.002)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


class TestRiskAttributes(unittest.TestCase):
    def test_spot_shape_defaults(self) -> None:
        ra = RiskAttributes(kelly_divisor=1.0, notional_exposure_usd=1000.0)
        self.assertEqual(ra.kelly_divisor, 1.0)
        self.assertEqual(ra.notional_exposure_usd, 1000.0)
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_perp_shape(self) -> None:
        ra = RiskAttributes(
            kelly_divisor=1.0,
            notional_exposure_usd=5000.0,
            liquidation_price=1500.0,
            margin_used_usd=500.0,
        )
        self.assertEqual(ra.liquidation_price, 1500.0)
        self.assertEqual(ra.margin_used_usd, 500.0)

    def test_binary_shape(self) -> None:
        # Variance of Bernoulli with p=0.6 is 0.24
        ra = RiskAttributes(kelly_divisor=0.24, notional_exposure_usd=100.0)
        self.assertAlmostEqual(ra.kelly_divisor, 0.24)
        self.assertIsNone(ra.liquidation_price)
        self.assertIsNone(ra.margin_used_usd)

    def test_frozen(self) -> None:
        ra = RiskAttributes(kelly_divisor=0.5, notional_exposure_usd=100.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ra.kelly_divisor = 0.99  # type: ignore[misc]


class TestTradeableProtocol(unittest.TestCase):
    def test_protocol_is_runtime_checkable(self) -> None:
        # Sanity: the import path from the package and the module are the same.
        self.assertIs(Tradeable, TradeableImport)

        stub = _StubTradeable()
        self.assertIsInstance(stub, Tradeable)

    def test_incomplete_class_fails_isinstance(self) -> None:
        incomplete = _IncompleteTradeable()
        self.assertNotIsInstance(incomplete, Tradeable)

    def test_arbitrary_object_fails_isinstance(self) -> None:
        self.assertNotIsInstance("not a tradeable", Tradeable)
        self.assertNotIsInstance(object(), Tradeable)

    def test_stub_methods_are_callable(self) -> None:
        stub = _StubTradeable()
        # Spot-check a few attributes / methods exist and behave
        self.assertEqual(stub.symbol, "ETH/USD")
        self.assertEqual(stub.asset_class, AssetClass.SPOT_CRYPTO)
        self.assertIsInstance(stub.fee_model, FeeModel)
        ra = stub.risk_attributes(side="buy", size_base=0.5, entry_price=2000.0)
        self.assertIsInstance(ra, RiskAttributes)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
