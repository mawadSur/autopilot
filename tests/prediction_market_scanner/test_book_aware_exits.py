"""Tests for W1 book-aware exit fills (SHADOW-ONLY).

The early-exit sweep DECIDES on the mark but BOOKS the realized P/L at the price
it would actually realize liquidating into the bid book — the honest exit number
the top-of-book mark was missing. Covered here:

  * ``vwap_sell_price`` — walks the bid side (best price first) for the position's
    units, handles partial fills on a thin book, and degrades on empty/zero input;
  * ``apply_exit_rules(fill_fn=...)`` — take-profit fires on the mark but the P/L
    is booked at the (lower) book fill, not the mark;
  * ``make_whale_fill_fn`` — parses ``asset=`` from notes and walks that token's
    book (get_order_book patched; no network).
"""

from __future__ import annotations

import os
import tempfile
import unittest

import whale_follow_runner as runner
from exit_rules import apply_exit_rules
from exchanges import polymarket_market_data as pmd
from exchanges.polymarket_market_data import vwap_sell_price
from state.pnl_ledger import PnlLedger, TradeRecord


class VwapSellPriceTests(unittest.TestCase):
    def test_walks_best_bid_first(self) -> None:
        # Unsorted bids: sell into the highest price first.
        bids = [{"price": "0.48", "size": "200"}, {"price": "0.50", "size": "100"}]
        vwap, filled = vwap_sell_price(bids, 150)
        # 100 @ 0.50 + 50 @ 0.48 = 50 + 24 = 74 over 150 units.
        self.assertAlmostEqual(vwap, 74.0 / 150.0, places=6)
        self.assertAlmostEqual(filled, 150.0, places=6)

    def test_partial_fill_on_thin_book(self) -> None:
        bids = [{"price": "0.50", "size": "100"}, {"price": "0.48", "size": "200"}]
        vwap, filled = vwap_sell_price(bids, 400)  # depth is only 300
        self.assertAlmostEqual(filled, 300.0, places=6)  # flags the thin book
        self.assertAlmostEqual(vwap, (50.0 + 96.0) / 300.0, places=6)

    def test_bare_pair_levels(self) -> None:
        vwap, filled = vwap_sell_price([[0.60, 50]], 50)
        self.assertAlmostEqual(vwap, 0.60, places=6)

    def test_empty_and_nonpositive(self) -> None:
        self.assertIsNone(vwap_sell_price([], 100))
        self.assertIsNone(vwap_sell_price(None, 100))
        self.assertIsNone(vwap_sell_price([{"price": "0.5", "size": "10"}], 0))
        self.assertIsNone(vwap_sell_price([{"price": "0.5", "size": "0"}], 10))


class ApplyExitRulesFillTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        # entry 0.40, size 100 -> 250 units.
        self.ledger.append(TradeRecord(
            trade_id="t1", ts_utc="2026-06-02T00:00:00+00:00", venue="polymarket",
            market_id="0xM", side="Yes", entry_price=0.40, size=100.0, fees_usd=0.0,
            slippage_bps=0.0, strategy="whale_convergence", status="open",
            notes="outcomeIndex=0; asset=999",
        ))

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_fill_used_for_pnl_not_the_mark(self) -> None:
        # Mark 0.96 -> take-profit-price 0.95 fires. But the book only fills at 0.50.
        res = apply_exit_rules(
            self.ledger, lambda r: 0.96,
            stop_loss_pct=0.20, take_profit_pct=0.30, take_profit_price=0.95,
            fill_fn=lambda r: 0.50, now_iso="2026-06-03T00:00:00+00:00",
        )
        self.assertEqual(res["take_profit"], 1)
        settled = self.ledger.settled()[0]
        # P/L booked at the FILL (0.50): 250*0.50*0.98 - 100 = +22.5, not +135.2.
        self.assertAlmostEqual(settled.realized_pnl_usd, 22.5, places=3)
        self.assertAlmostEqual(settled.exit_price, 0.50, places=6)

    def test_fill_none_falls_back_to_mark(self) -> None:
        res = apply_exit_rules(
            self.ledger, lambda r: 0.96,
            stop_loss_pct=0.20, take_profit_pct=0.30, take_profit_price=0.95,
            fill_fn=lambda r: None, now_iso="2026-06-03T00:00:00+00:00",
        )
        self.assertEqual(res["take_profit"], 1)
        settled = self.ledger.settled()[0]
        self.assertAlmostEqual(settled.exit_price, 0.96, places=6)  # mark fallback


class MakeWhaleFillFnTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = pmd.get_order_book

    def tearDown(self) -> None:
        pmd.get_order_book = self._orig

    def _rec(self, notes: str, entry=0.40, size=100.0) -> TradeRecord:
        return TradeRecord(
            trade_id="t", ts_utc="2026-06-02T00:00:00+00:00", venue="polymarket",
            market_id="0xM", side="Yes", entry_price=entry, size=size, fees_usd=0.0,
            slippage_bps=0.0, strategy="whale_convergence", status="open", notes=notes,
        )

    def test_parses_asset_and_walks_book(self) -> None:
        seen = {}
        def fake_book(token_id, session=None):
            seen["token"] = token_id
            return {"bids": [{"price": "0.55", "size": "10000"}]}
        pmd.get_order_book = fake_book
        fill = runner.make_whale_fill_fn()
        price = fill(self._rec("title=Yanks v Sox; asset=12345; outcomeIndex=0"))
        self.assertEqual(seen["token"], "12345")
        self.assertAlmostEqual(price, 0.55, places=6)

    def test_no_asset_marker_returns_none(self) -> None:
        fill = runner.make_whale_fill_fn()
        self.assertIsNone(fill(self._rec("outcomeIndex=0")))  # pre-W1 record

    def test_book_error_returns_none(self) -> None:
        def boom(token_id, session=None):
            raise RuntimeError("network")
        pmd.get_order_book = boom
        fill = runner.make_whale_fill_fn()
        self.assertIsNone(fill(self._rec("asset=12345")))


if __name__ == "__main__":
    unittest.main()
