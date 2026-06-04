"""Tests for hedge feasibility (READ-ONLY, hermetic — no network)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from funding_hedge_feasibility import (
    base_token, required_hedge, find_spot_venues, all_in_net, assess,
)


def _c(symbol, side, net):  # CarryRow-like
    return SimpleNamespace(symbol=symbol, side=side, net_annual=net)


class PureTests(unittest.TestCase):
    def test_base_token(self):
        self.assertEqual(base_token("XMR/USDC:USDC"), "XMR")
        self.assertEqual(base_token("hype/usdc:usdc"), "HYPE")
        self.assertEqual(base_token("SOL"), "SOL")

    def test_required_hedge(self):
        self.assertEqual(required_hedge("short_perp"), "buy_spot")
        self.assertEqual(required_hedge("long_perp"), "short_spot")

    def test_find_spot_venues(self):
        m = {"XMR": {"kraken"}, "SOL": {"coinbase", "kraken"}}
        self.assertEqual(find_spot_venues("xmr", m), ["kraken"])
        self.assertEqual(find_spot_venues("SOL", m), ["coinbase", "kraken"])
        self.assertEqual(find_spot_venues("NONE", m), [])

    def test_all_in_net(self):
        self.assertAlmostEqual(all_in_net(0.30, "buy_spot"), 0.30, places=9)         # fee in scanner
        self.assertAlmostEqual(all_in_net(0.30, "short_spot", borrow_annual=0.20), 0.10, places=9)


class AssessTests(unittest.TestCase):
    def test_classifies_and_sorts(self):
        cands = [
            _c("XMR/USDC:USDC", "short_perp", 0.30),   # buy_spot + spot -> harvestable, all-in 30%
            _c("HID/USDC:USDC", "short_perp", 0.50),   # buy_spot + NO spot -> no_spot
            _c("MEME/USDC:USDC", "long_perp", 0.40),   # short_spot -> needs_borrow, all-in 40-20=20%
        ]
        spot = {"XMR": {"kraken"}, "MEME": {"coinbase"}}
        res = assess(cands, spot, borrow_annual=0.20)
        by = {a.symbol: a for a in res}
        self.assertEqual(by["XMR/USDC:USDC"].verdict, "harvestable")
        self.assertEqual(by["XMR/USDC:USDC"].spot_venues, ["kraken"])
        self.assertEqual(by["HID/USDC:USDC"].verdict, "no_spot")
        self.assertEqual(by["MEME/USDC:USDC"].verdict, "needs_borrow")
        self.assertAlmostEqual(by["MEME/USDC:USDC"].all_in_net_annual, 0.20, places=9)
        # sorted by all-in desc: XMR(.30) > MEME(.20) > HID(.50 buy_spot, all-in .50 but no_spot still ranks by all-in)
        self.assertEqual(res[0].symbol, "HID/USDC:USDC")  # all-in 0.50 (but verdict no_spot)
        # harvestable-only ordering check
        harv = [a for a in res if a.verdict == "harvestable"]
        self.assertEqual([a.symbol for a in harv], ["XMR/USDC:USDC"])


if __name__ == "__main__":
    unittest.main()
