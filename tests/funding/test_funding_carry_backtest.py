"""Tests for the funding-carry backtest (READ-ONLY, hermetic — no network)."""
from __future__ import annotations

import unittest
from funding_carry_backtest import (
    normalize_funding_history, backtest_symbol, SymbolBacktest,
)

H = 3_600_000  # 1h in ms


def _hist(rates, start=1_000_000_000_000):
    return [{"timestamp": start + i * H, "fundingRate": r} for i, r in enumerate(rates)]


class NormalizeTests(unittest.TestCase):
    def test_filters_and_sorts(self):
        raw = [
            {"timestamp": 200, "fundingRate": 0.0002},
            {"timestamp": 100, "fundingRate": "0.0001"},
            {"timestamp": 300, "fundingRate": None},
            "junk",
        ]
        out = normalize_funding_history(raw)
        self.assertEqual([h["fundingRate"] for h in out], [0.0001, 0.0002])  # sorted, None dropped


class BacktestSymbolTests(unittest.TestCase):
    def test_persistent_positive_funding_short_side(self):
        bt = backtest_symbol("X", _hist([0.0001] * 24), period_hours=1.0,
                             round_trip_bps=20.0, basis_buffer_annual=0.05)
        self.assertEqual(bt.side, "short_perp")
        self.assertAlmostEqual(bt.realized_total, 0.0024, places=9)  # 24 * 0.0001
        self.assertEqual(bt.favorable_pct, 1.0)
        self.assertAlmostEqual(bt.worst_period, 0.0001, places=9)
        # gross annualized > 0, net < gross (cost subtracted)
        self.assertGreater(bt.gross_annual, 0)
        self.assertLess(bt.net_annual, bt.gross_annual)

    def test_persistent_negative_funding_long_side(self):
        bt = backtest_symbol("Y", _hist([-0.0001] * 24), period_hours=1.0)
        self.assertEqual(bt.side, "long_perp")
        # long earns -r each period; r=-0.0001 -> +0.0001/period
        self.assertAlmostEqual(bt.realized_total, 0.0024, places=9)
        self.assertEqual(bt.favorable_pct, 1.0)

    def test_flipping_funding_exposes_whip(self):
        # half +, half - : as a short, the negative periods are LOSSES
        rates = [0.0002] * 12 + [-0.0003] * 12
        bt = backtest_symbol("Z", _hist(rates), period_hours=1.0)
        # mean = (0.0024 - 0.0036)/24 < 0 -> long side; +0.0002 periods become losses
        self.assertEqual(bt.side, "long_perp")
        self.assertLess(bt.favorable_pct, 1.0)   # some periods adverse
        self.assertLess(bt.worst_period, 0.0)    # at least one losing period

    def test_empty_history(self):
        self.assertIsNone(backtest_symbol("E", [], period_hours=1.0))


if __name__ == "__main__":
    unittest.main()
