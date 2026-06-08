"""Unit tests for the model-free arbitrage detector (``arb_detector``).

These pin down the intra-market arbitrage accounting identity (buy YES + NO,
guaranteed $1 payout), the honest cost netting (settlement fee on the $1 payout +
gas), the scan/sort/shadow-ledger closed loop, and the directional-EV helper.

Run:
    env PYTHONPATH=src ./.venv/bin/python -m unittest \
        tests.prediction_market_scanner.test_arb_detector -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

from arb_detector import (
    intramarket_arb,
    market_row_to_arb_input,
    net_directional_edge,
    scan_intramarket_arbs,
)
from state.pnl_ledger import PnlLedger


class IntramarketArbTests(unittest.TestCase):
    def test_arb_returned_with_thin_fee_correct_math(self) -> None:
        # yes_ask + no_ask = 0.97 < 1 -> gross edge 0.03; zero fee/gas keeps it.
        arb = intramarket_arb(0.48, 0.49, settlement_fee_bps=0.0, gas_usd=0.0)
        self.assertIsNotNone(arb)
        assert arb is not None  # for type-checkers
        self.assertAlmostEqual(arb["cost_basis"], 0.97)
        self.assertAlmostEqual(arb["gross_edge"], 0.03)
        self.assertAlmostEqual(arb["net_edge_per_pair"], 0.03)
        self.assertAlmostEqual(arb["settlement_fee_usd"], 0.0)
        self.assertAlmostEqual(arb["gas_usd"], 0.0)
        # net_edge_pct = net / cost_basis = 0.03 / 0.97.
        self.assertAlmostEqual(arb["net_edge_pct"], 0.03 / 0.97)
        # est_profit_usd is 0 when size is not supplied.
        self.assertEqual(arb["est_profit_usd"], 0.0)
        # Legs carry both sides at their ask prices.
        self.assertEqual(
            arb["legs"],
            [{"side": "YES", "price": 0.48}, {"side": "NO", "price": 0.49}],
        )

    def test_default_fee_haircut_still_arb(self) -> None:
        # Same 0.03 gross, but with the default 200bps fee ($0.02) netted out.
        arb = intramarket_arb(0.48, 0.49)  # default settlement_fee_bps=200
        self.assertIsNotNone(arb)
        assert arb is not None
        self.assertAlmostEqual(arb["settlement_fee_usd"], 0.02)
        self.assertAlmostEqual(arb["net_edge_per_pair"], 0.03 - 0.02)
        self.assertAlmostEqual(arb["net_edge_pct"], (0.03 - 0.02) / 0.97)

    def test_no_arb_when_sum_at_or_above_one(self) -> None:
        # yes_ask + no_ask = 1.05 >= 1 -> no gross edge at all.
        self.assertIsNone(intramarket_arb(0.55, 0.50, settlement_fee_bps=0.0))
        # Exactly 1.00 is also not an arb (gross_edge == 0, not > 0).
        self.assertIsNone(intramarket_arb(0.50, 0.50, settlement_fee_bps=0.0))

    def test_gross_edge_eaten_by_fee_and_gas_returns_none(self) -> None:
        # cost_basis 0.99 -> gross 0.01. 200bps fee = $0.02 alone exceeds it.
        self.assertIsNone(intramarket_arb(0.49, 0.50, settlement_fee_bps=200.0))
        # Even with a smaller fee, gas can eat the remaining edge.
        # cost 0.98 -> gross 0.02; 50bps fee = $0.005, gas $0.02 -> net -0.005.
        self.assertIsNone(
            intramarket_arb(0.49, 0.49, settlement_fee_bps=50.0, gas_usd=0.02)
        )

    def test_size_usd_produces_correct_est_profit(self) -> None:
        # cost 0.80 -> gross 0.20, zero fee -> net 0.20 per pair.
        # size_usd 500 = 500 pairs (each pays $1) -> est profit 0.20 * 500 = 100.
        arb = intramarket_arb(0.40, 0.40, settlement_fee_bps=0.0, size_usd=500.0)
        self.assertIsNotNone(arb)
        assert arb is not None
        self.assertAlmostEqual(arb["net_edge_per_pair"], 0.20)
        self.assertAlmostEqual(arb["est_profit_usd"], 100.0)

    def test_input_validation_rejects_out_of_range_prices(self) -> None:
        for bad in (0.0, -0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                intramarket_arb(bad, 0.5)
            with self.assertRaises(ValueError):
                intramarket_arb(0.5, bad)


class ScanIntramarketArbsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "arb_ledger.jsonl")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sorts_by_net_edge_pct_and_logs_shadow_records(self) -> None:
        ledger = PnlLedger(self.path)
        markets = [
            # net_pct ~ (0.03 - 0.02)/0.97 = 0.01031 -> below 0.005? no, above.
            {"market_id": "small-edge", "yes_ask": 0.48, "no_ask": 0.49},
            # net_pct ~ (0.20 - 0.02)/0.80 = 0.225 -> big edge.
            {"market_id": "big-edge", "yes_ask": 0.40, "no_ask": 0.40},
            # No arb: sum >= 1, must be skipped (no record, not in results).
            {"market_id": "no-edge", "yes_ask": 0.55, "no_ask": 0.50},
            # Missing no_ask -> skipped silently.
            {"market_id": "incomplete", "yes_ask": 0.40},
        ]

        results = scan_intramarket_arbs(markets, ledger=ledger, min_net_edge_pct=0.005)

        # Two real arbs, sorted by net_edge_pct descending.
        self.assertEqual([r["market_id"] for r in results], ["big-edge", "small-edge"])
        self.assertGreater(results[0]["net_edge_pct"], results[1]["net_edge_pct"])
        for r in results:
            self.assertEqual(r["strategy"], "intramarket_arb")

        # Exactly two SHADOW open records were logged, with the expected fields.
        open_positions = ledger.open_positions()
        self.assertEqual(len(open_positions), 2)
        by_market = {r.market_id: r for r in open_positions}
        self.assertEqual(set(by_market), {"big-edge", "small-edge"})
        for rec in open_positions:
            self.assertEqual(rec.status, "open")
            self.assertEqual(rec.venue, "polymarket")
            self.assertEqual(rec.strategy, "intramarket_arb")
            self.assertEqual(rec.side, "YES+NO")
            self.assertIn("SHADOW", rec.notes)
            self.assertIn("NO order placed", rec.notes)
        # entry_price is the cost basis of the pair.
        self.assertAlmostEqual(by_market["big-edge"].entry_price, 0.80)
        self.assertAlmostEqual(by_market["small-edge"].entry_price, 0.97)
        # fees_usd is the settlement fee on $1 payout (default 200bps = $0.02).
        self.assertAlmostEqual(by_market["big-edge"].fees_usd, 0.02)

    def test_min_net_edge_pct_filters_out_thin_arbs(self) -> None:
        # gross 0.03, 200bps fee -> net 0.01, net_pct ~ 0.0103.
        # A 0.05 (5%) threshold filters it out; no record logged.
        ledger = PnlLedger(self.path)
        markets = [{"market_id": "thin", "yes_ask": 0.48, "no_ask": 0.49}]
        results = scan_intramarket_arbs(markets, ledger=ledger, min_net_edge_pct=0.05)
        self.assertEqual(results, [])
        self.assertEqual(ledger.open_positions(), [])

    def test_size_usd_propagates_to_records_and_profit(self) -> None:
        ledger = PnlLedger(self.path)
        markets = [{"id": "sized", "yes_ask": 0.40, "no_ask": 0.40}]
        results = scan_intramarket_arbs(
            markets, ledger=ledger, min_net_edge_pct=0.005, size_usd=500.0
        )
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["est_profit_usd"], 100.0 - 500.0 * 0.02)
        rec = ledger.open_positions()[0]
        self.assertEqual(rec.market_id, "sized")  # tolerant of 'id' key
        self.assertAlmostEqual(rec.size, 500.0)

    def test_tolerates_field_name_variants(self) -> None:
        # camelCase / alternate keys must still be read.
        markets = [{"ticker": "ALT", "yesAsk": 0.40, "noAsk": 0.40}]
        results = scan_intramarket_arbs(markets, min_net_edge_pct=0.005)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["market_id"], "ALT")


class NetDirectionalEdgeTests(unittest.TestCase):
    def test_positive_pre_fee_edge(self) -> None:
        # fair 0.60 vs ask 0.50 with zero fee: EV = 0.60 - 0.50 = +0.10.
        ev = net_directional_edge(0.60, 0.50, settlement_fee_bps=0.0)
        self.assertAlmostEqual(ev, 0.10)

    def test_fee_haircut_reduces_edge(self) -> None:
        # 200bps fee on the $1 payout, weighted by win prob 0.60:
        # EV = 0.60 * (1 - 0.02) - 0.50 = 0.588 - 0.50 = 0.088 < 0.10.
        ev = net_directional_edge(0.60, 0.50, settlement_fee_bps=200.0)
        self.assertAlmostEqual(ev, 0.088)
        self.assertLess(ev, 0.10)

    def test_negative_edge_when_ask_above_fair(self) -> None:
        # Buying at 0.70 when fair is only 0.55 is a losing bet pre-fee.
        ev = net_directional_edge(0.55, 0.70, settlement_fee_bps=0.0)
        self.assertAlmostEqual(ev, 0.55 - 0.70)
        self.assertLess(ev, 0.0)

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            net_directional_edge(1.2, 0.5)  # prob out of [0,1]
        with self.assertRaises(ValueError):
            net_directional_edge(0.5, 1.0)  # ask not in (0,1)


class MarketRowMappingTests(unittest.TestCase):
    def test_returns_none_without_no_ask(self) -> None:
        # A bare scanner row (implied_prob only) cannot prove an arb on its own.
        row = {"market_id": "m1", "implied_prob": 0.42, "spread": 0.03}
        self.assertIsNone(market_row_to_arb_input(row))

    def test_maps_with_explicit_no_ask(self) -> None:
        row = {"market_id": "m2", "implied_prob": 0.42}
        mapped = market_row_to_arb_input(row, no_ask=0.50)
        self.assertEqual(mapped["market_id"], "m2")
        self.assertAlmostEqual(mapped["yes_ask"], 0.42)
        self.assertAlmostEqual(mapped["no_ask"], 0.50)


if __name__ == "__main__":
    unittest.main()
