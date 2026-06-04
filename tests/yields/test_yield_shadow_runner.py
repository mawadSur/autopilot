"""Tests for the stablecoin yield SHADOW runner (NO network; temp ledger).

Hermetic: prices are injected into ``run_scan`` so nothing touches Kraken/ccxt.
Covers the pure accrual/peg helpers, the open-then-accrue ledger lifecycle, depeg
alerting, and a restart-safe fold round-trip.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from yield_shadow_runner import (
    yield_accrued, peg_deviation, is_depeg, fold_ledger, run_scan, SECONDS_PER_YEAR,
)

H_MS = 3_600_000.0


class YieldAccruedTests(unittest.TestCase):
    def test_one_full_year(self):
        self.assertAlmostEqual(yield_accrued(0.045, 10_000.0, SECONDS_PER_YEAR), 450.0, places=6)

    def test_half_year(self):
        self.assertAlmostEqual(yield_accrued(0.045, 10_000.0, SECONDS_PER_YEAR / 2.0), 225.0, places=6)

    def test_scales_with_seconds(self):
        one_hour = yield_accrued(0.045, 10_000.0, 3600.0)
        two_hours = yield_accrued(0.045, 10_000.0, 7200.0)
        self.assertAlmostEqual(two_hours, 2.0 * one_hour, places=9)
        self.assertGreater(one_hour, 0.0)

    def test_zero_seconds_zero_accrual(self):
        self.assertEqual(yield_accrued(0.045, 10_000.0, 0.0), 0.0)


class PegHelperTests(unittest.TestCase):
    def test_peg_deviation(self):
        self.assertAlmostEqual(peg_deviation(0.997), -0.003, places=9)
        self.assertAlmostEqual(peg_deviation(1.002), 0.002, places=9)

    def test_not_depeg_within_threshold(self):
        self.assertFalse(is_depeg(peg_deviation(0.997), 0.005))  # -0.003 within 0.5%

    def test_depeg_when_breached(self):
        self.assertTrue(is_depeg(peg_deviation(0.99), 0.005))    # -0.01 breaches 0.5%
        self.assertTrue(is_depeg(peg_deviation(1.02), 0.005))    # symmetric on the high side


class RunScanTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._d.name, "yield.jsonl")
        self.t0 = 1_000_000_000_000.0

    def tearDown(self):
        self._d.cleanup()

    def _scan(self, prices, now_ms):
        return run_scan(
            self.path, prices, now_ms=now_ms, apy=0.045, notional=10_000.0,
            stablecoins=["USDC", "USDT"], depeg_threshold=0.005,
        )

    def test_first_scan_opens_positions(self):
        res = self._scan({"USDC/USD": 0.9997, "USDT/USD": 0.9989}, self.t0)
        self.assertEqual(res["action"], "opened")
        self.assertEqual(res["n"], 2)
        self.assertEqual(res["alerts"], [])
        st = fold_ledger(self.path)
        self.assertEqual(set(st), {"USDC", "USDT"})
        self.assertEqual(st["USDC"]["realized_usd"], 0.0)  # no accrual at open (no look-ahead)
        self.assertEqual(st["USDC"]["n_accruals"], 0)
        self.assertEqual(st["USDC"]["last_ts_ms"], self.t0)
        self.assertAlmostEqual(st["USDC"]["apy"], 0.045, places=9)
        self.assertAlmostEqual(st["USDC"]["notional"], 10_000.0, places=9)

    def test_second_scan_accrues_one_hour(self):
        self._scan({"USDC/USD": 0.9997, "USDT/USD": 0.9989}, self.t0)
        res = self._scan({"USDC/USD": 0.9998, "USDT/USD": 0.9990}, self.t0 + H_MS)
        self.assertEqual(res["action"], "accrued")
        self.assertEqual(res["n"], 2)
        self.assertEqual(res["alerts"], [])  # both within peg
        st = fold_ledger(self.path)
        # 1h of 4.5% APY on $10,000 = 0.045 * 10000 * 3600 / 31557600
        expected = yield_accrued(0.045, 10_000.0, 3600.0)
        self.assertAlmostEqual(st["USDC"]["realized_usd"], expected, places=9)
        self.assertAlmostEqual(st["USDT"]["realized_usd"], expected, places=9)
        self.assertEqual(st["USDC"]["n_accruals"], 1)
        # peg + deviation recorded from the injected price
        self.assertAlmostEqual(st["USDC"]["last_price"], 0.9998, places=9)
        self.assertAlmostEqual(st["USDC"]["last_deviation"], -0.0002, places=9)

    def test_depeg_price_raises_alert(self):
        self._scan({"USDC/USD": 0.9997, "USDT/USD": 0.9989}, self.t0)
        res = self._scan({"USDC/USD": 0.985, "USDT/USD": 0.9990}, self.t0 + H_MS)
        self.assertEqual(res["n"], 2)
        self.assertEqual(len(res["alerts"]), 1)
        alert = res["alerts"][0]
        self.assertEqual(alert["symbol"], "USDC")
        self.assertAlmostEqual(alert["price"], 0.985, places=9)
        self.assertAlmostEqual(alert["deviation"], -0.015, places=9)
        # the depeg still accrues yield (the deposit is still earning until pulled)
        st = fold_ledger(self.path)
        self.assertGreater(st["USDC"]["realized_usd"], 0.0)
        self.assertAlmostEqual(st["USDC"]["last_deviation"], -0.015, places=9)

    def test_missing_quote_skips_without_fabricating(self):
        self._scan({"USDC/USD": 0.9997, "USDT/USD": 0.9989}, self.t0)
        # USDT has no quote this scan -> it must NOT accrue.
        res = self._scan({"USDC/USD": 0.9998}, self.t0 + H_MS)
        self.assertEqual(res["n"], 1)
        st = fold_ledger(self.path)
        self.assertGreater(st["USDC"]["realized_usd"], 0.0)
        self.assertEqual(st["USDT"]["realized_usd"], 0.0)
        self.assertEqual(st["USDT"]["n_accruals"], 0)


class FoldRoundTripTests(unittest.TestCase):
    def test_fold_round_trips_realized_and_last_ts(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        path = os.path.join(d.name, "yield.jsonl")
        t0 = 2_000_000_000_000.0
        run_scan(path, {"USDC/USD": 1.0001}, now_ms=t0, apy=0.045, notional=10_000.0,
                 stablecoins=["USDC"], depeg_threshold=0.005)
        run_scan(path, {"USDC/USD": 1.0002}, now_ms=t0 + 2 * H_MS, apy=0.045,
                 notional=10_000.0, stablecoins=["USDC"], depeg_threshold=0.005)
        st = fold_ledger(path)
        self.assertAlmostEqual(
            st["USDC"]["realized_usd"], yield_accrued(0.045, 10_000.0, 7200.0), places=9)
        self.assertEqual(st["USDC"]["last_ts_ms"], t0 + 2 * H_MS)
        self.assertEqual(st["USDC"]["n_accruals"], 1)
        self.assertAlmostEqual(st["USDC"]["last_price"], 1.0002, places=9)


if __name__ == "__main__":
    unittest.main()
