"""Tests for the crypto funding-carry scanner (READ-ONLY, hermetic — no network).

Pins the HONEST cost math (the discipline that prevents repeating the crypto-1m
edge<cost death) and the net-of-cost ranking.
"""
from __future__ import annotations

import unittest

from funding_carry_scanner import (
    annualize_funding, amortized_round_trip_annual, net_carry_annual,
    normalize_funding_rates, scan_carry, DEFAULT_PERIOD_HOURS,
)


class CostMathTests(unittest.TestCase):
    def test_annualize_hourly(self):
        # 0.0001/hr * 24 * 365 = 0.876 (87.6%/yr)
        self.assertAlmostEqual(annualize_funding(0.0001, 1.0), 0.876, places=6)

    def test_annualize_8h(self):
        self.assertAlmostEqual(annualize_funding(0.0001, 8.0), 0.0001 * 3 * 365, places=9)

    def test_zero_period_is_safe(self):
        self.assertEqual(annualize_funding(0.01, 0.0), 0.0)

    def test_amortized_cost_shrinks_with_hold(self):
        # 20bps round trip over 14 days, annualized.
        self.assertAlmostEqual(amortized_round_trip_annual(20.0, 14.0), 0.002 * 365 / 14, places=9)
        # Longer hold = cheaper per year.
        self.assertLess(amortized_round_trip_annual(20.0, 90.0), amortized_round_trip_annual(20.0, 14.0))

    def test_net_subtracts_cost_and_buffer_and_uses_abs(self):
        # gross 50%/yr, 20bps/14d (~5.21%) + 5% buffer -> net ~39.8%
        net = net_carry_annual(0.50, round_trip_bps=20.0, hold_days=14.0, basis_buffer_annual=0.05)
        self.assertAlmostEqual(net, 0.50 - (0.002 * 365 / 14) - 0.05, places=6)
        # negative funding is equally harvestable: abs() -> same net for -0.50.
        self.assertAlmostEqual(net_carry_annual(-0.50, round_trip_bps=20.0, hold_days=14.0,
                                                basis_buffer_annual=0.05), net, places=9)

    def test_thin_carry_can_go_negative(self):
        # 4%/yr gross can't clear the hurdle -> negative net (don't trade).
        self.assertLess(net_carry_annual(0.04, round_trip_bps=20.0, hold_days=14.0,
                                         basis_buffer_annual=0.05), 0.0)


class NormalizeTests(unittest.TestCase):
    def test_keeps_numeric_rows_and_mark(self):
        raw = {
            "HYPE/USDC:USDC": {"fundingRate": 0.0001325, "markPrice": 40.0},
            "BAD/USDC:USDC": {"fundingRate": None},
            "STR/USDC:USDC": {"fundingRate": "0.00005"},  # string ok
            "junk": "notadict",
        }
        rows = normalize_funding_rates(raw, period_hours=1.0)
        syms = {r["symbol"] for r in rows}
        self.assertEqual(syms, {"HYPE/USDC:USDC", "STR/USDC:USDC"})
        hype = next(r for r in rows if r["symbol"].startswith("HYPE"))
        self.assertEqual(hype["mark_price"], 40.0)
        self.assertEqual(hype["period_hours"], 1.0)


class ScanTests(unittest.TestCase):
    def _rows(self):
        return normalize_funding_rates({
            "BIG/USDC:USDC": {"fundingRate": 0.0001},     # +87.6%/yr -> short_perp
            "NEG/USDC:USDC": {"fundingRate": -0.00008},   # -70.1%/yr -> long_perp
            "THIN/USDC:USDC": {"fundingRate": 0.000004},  # +3.5%/yr -> below hurdle
        }, period_hours=1.0)

    def test_ranks_by_net_filters_thin_and_assigns_side(self):
        ranked = scan_carry(self._rows(), round_trip_bps=20.0, hold_days=14.0,
                            basis_buffer_annual=0.05, min_net_annual=0.10)
        syms = [c.symbol for c in ranked]
        self.assertEqual(syms, ["BIG/USDC:USDC", "NEG/USDC:USDC"])   # THIN filtered, sorted by net
        self.assertEqual(ranked[0].side, "short_perp")
        self.assertEqual(ranked[1].side, "long_perp")
        self.assertGreater(ranked[0].net_annual, ranked[1].net_annual)

    def test_min_net_floor_can_empty(self):
        self.assertEqual(scan_carry(self._rows(), min_net_annual=2.0), [])


if __name__ == "__main__":
    unittest.main()
