"""Tests for ``RiskCalculator.passes_market_filters`` (P1 #10).

The extreme-price band was tightened from [0.01, 0.99] to [0.02, 0.98]
because Kelly sizing explodes faster than naive intuition past p=0.95.
Centralising the check in the risk engine means the orchestrator no
longer needs its own duplicated kill switch.
"""

from __future__ import annotations

import unittest

from risk_management_agent.risk_engine import RiskCalculator


class PassesMarketFiltersTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calculator = RiskCalculator()

    def test_price_098_passes(self) -> None:
        self.assertTrue(self.calculator.passes_market_filters(0.98))

    def test_price_099_rejects(self) -> None:
        self.assertFalse(self.calculator.passes_market_filters(0.99))

    def test_price_002_passes(self) -> None:
        self.assertTrue(self.calculator.passes_market_filters(0.02))

    def test_price_001_rejects(self) -> None:
        self.assertFalse(self.calculator.passes_market_filters(0.01))

    def test_safe_interior_price_passes(self) -> None:
        self.assertTrue(self.calculator.passes_market_filters(0.5))

    def test_zero_and_one_reject(self) -> None:
        self.assertFalse(self.calculator.passes_market_filters(0.0))
        self.assertFalse(self.calculator.passes_market_filters(1.0))

    def test_non_numeric_rejects(self) -> None:
        self.assertFalse(self.calculator.passes_market_filters("not-a-number"))

    def test_caller_supplied_band_overrides(self) -> None:
        # Wider explicit band lets 0.99 through.
        self.assertTrue(
            self.calculator.passes_market_filters(0.99, p_min=0.0, p_max=0.999)
        )

    def test_band_can_be_set_via_constructor(self) -> None:
        loose = RiskCalculator(price_floor=0.005, price_ceil=0.995)
        self.assertTrue(loose.passes_market_filters(0.99))
        self.assertFalse(loose.passes_market_filters(0.999))


if __name__ == "__main__":
    unittest.main()
