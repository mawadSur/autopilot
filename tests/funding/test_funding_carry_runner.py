"""Tests for the funding-carry shadow runner (NO network; temp ledger)."""
from __future__ import annotations

import os, tempfile, unittest
from funding_carry_runner import accrue_delta, fold_ledger, run_scan
from funding_carry_scanner import normalize_funding_rates

H_MS = 3_600_000.0


class AccrueDeltaTests(unittest.TestCase):
    def test_short_receives_positive_funding(self):
        self.assertAlmostEqual(accrue_delta(1.0, 0.0001, 1000.0, 1.0), 0.1, places=9)
    def test_long_receives_negative_funding(self):
        # long (side -1) on r=-0.0001 -> +0.1 ; on r=+0.0001 -> -0.1 (pays)
        self.assertAlmostEqual(accrue_delta(-1.0, -0.0001, 1000.0, 1.0), 0.1, places=9)
        self.assertAlmostEqual(accrue_delta(-1.0, 0.0001, 1000.0, 1.0), -0.1, places=9)
    def test_periods_scale(self):
        self.assertAlmostEqual(accrue_delta(1.0, 0.0001, 1000.0, 3.0), 0.3, places=9)


class RunScanTests(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._d.name, "carry.jsonl")
        self.rows = normalize_funding_rates({
            "AAA/USDC:USDC": {"fundingRate": 0.0001},    # +87.6%/yr -> short
            "BBB/USDC:USDC": {"fundingRate": -0.00008},  # -70%/yr  -> long
            "THIN/USDC:USDC": {"fundingRate": 0.000003}, # ~2.6%/yr -> filtered
        }, period_hours=1.0)
    def tearDown(self): self._d.cleanup()

    def test_first_scan_opens_basket(self):
        res = run_scan(self.path, self.rows, now_ms=1_000_000_000_000.0, period_hours=1.0,
                       notional=1000.0, top_k=8, min_net_annual=0.10, round_trip_bps=20.0,
                       basis_buffer_annual=0.05, hold_days=14.0)
        self.assertEqual(res["action"], "opened")
        st = fold_ledger(self.path)
        self.assertEqual(set(st), {"AAA/USDC:USDC", "BBB/USDC:USDC"})  # THIN filtered
        self.assertEqual(st["AAA/USDC:USDC"]["side"], "short_perp")
        self.assertEqual(st["BBB/USDC:USDC"]["side"], "long_perp")
        # one-time entry cost recorded (20bps of 1000 = $2)
        self.assertAlmostEqual(st["AAA/USDC:USDC"]["rt_cost_usd"], 2.0, places=6)

    def test_second_scan_accrues_one_hour(self):
        t0 = 1_000_000_000_000.0
        run_scan(self.path, self.rows, now_ms=t0, period_hours=1.0, notional=1000.0,
                 top_k=8, min_net_annual=0.10, round_trip_bps=20.0, basis_buffer_annual=0.05, hold_days=14.0)
        run_scan(self.path, self.rows, now_ms=t0 + H_MS, period_hours=1.0, notional=1000.0,
                 top_k=8, min_net_annual=0.10, round_trip_bps=20.0, basis_buffer_annual=0.05, hold_days=14.0)
        st = fold_ledger(self.path)
        # AAA short on +0.0001 over 1 period * $1000 = +$0.10 realized
        self.assertAlmostEqual(st["AAA/USDC:USDC"]["realized_usd"], 0.10, places=6)
        # BBB long on -0.00008 over 1 period -> +$0.08
        self.assertAlmostEqual(st["BBB/USDC:USDC"]["realized_usd"], 0.08, places=6)
        self.assertEqual(st["AAA/USDC:USDC"]["n_accruals"], 1)


if __name__ == "__main__":
    unittest.main()
