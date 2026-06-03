"""Tests for W2 book-aware ENTRY pricing + per-market depth cap (SHADOW-ONLY).

W2 mirrors W1 (book-aware exits) on the ENTRY side and adds a depth cap:

  * ``vwap_buy_price`` — walks the ask side (cheapest price first) for the units
    we'd buy, handles partial fills on a thin book, and degrades on empty/zero
    input (mirror of ``vwap_sell_price``);
  * book-aware ENTRY — with a patched ask book, the logged ``entry_price`` is the
    ASK VWAP (the price we'd actually lift), NOT the last ``/trades`` mark; with
    no ``asset`` (or a book error) it falls back to the mark;
  * per-market DEPTH CAP — a market too thin to take our size (size_usd > 5% of
    near-mid ASK depth) is skipped (``n_thin``, marked ``'thin_book'``); a deep
    book is logged; a book error fails OPEN (does not drop the candidate);
  * regression — book_entry default off + max_book_frac default 0.0 leaves
    ``_log_candidates`` behaving exactly as before for callers that don't pass the
    new args.

No network: a fake data-api client returns canned ``get_trades`` prices for the
``/trades`` mark, and ``polymarket_market_data.get_order_book`` is patched for the
ASK book.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import whale_follow_runner as runner
from exchanges import polymarket_market_data as pmd
from exchanges.polymarket_market_data import vwap_buy_price
from state.pnl_ledger import PnlLedger


class FakePricingClient:
    """get_trades(market=...) returns one trade per (cid, outcomeIndex) price."""

    def __init__(self, prices: Dict[tuple, float]) -> None:
        self.prices = prices  # {(conditionId, outcomeIndex): price}

    def get_trades(self, market: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        rows = []
        for (cid, oi), price in self.prices.items():
            if cid == market:
                rows.append(
                    {"conditionId": cid, "outcomeIndex": oi, "price": price, "timestamp": 1}
                )
        return rows


def _cand(
    cid: str,
    outcome: str,
    oi: int,
    *,
    asset: Optional[str] = None,
    wallets=("0xA", "0xB", "0xC"),
) -> Dict[str, Any]:
    cand: Dict[str, Any] = {
        "conditionId": cid,
        "outcome": outcome,
        "outcomeIndex": oi,
        "wallets": list(wallets),
        "n_target_holders": len(wallets),
        "title": f"{outcome} market",
    }
    if asset is not None:
        cand["asset"] = asset
    return cand


class VwapBuyPriceTests(unittest.TestCase):
    def test_walks_cheapest_ask_first(self) -> None:
        # Unsorted asks: buy the lowest price first.
        asks = [{"price": "0.52", "size": "200"}, {"price": "0.50", "size": "100"}]
        vwap, filled = vwap_buy_price(asks, 150)
        # 100 @ 0.50 + 50 @ 0.52 = 50 + 26 = 76 over 150 units.
        self.assertAlmostEqual(vwap, 76.0 / 150.0, places=6)
        self.assertAlmostEqual(filled, 150.0, places=6)

    def test_partial_fill_on_thin_book(self) -> None:
        asks = [{"price": "0.50", "size": "100"}, {"price": "0.52", "size": "200"}]
        vwap, filled = vwap_buy_price(asks, 400)  # depth is only 300
        self.assertAlmostEqual(filled, 300.0, places=6)  # flags the thin book
        self.assertAlmostEqual(vwap, (50.0 + 104.0) / 300.0, places=6)

    def test_bare_pair_levels(self) -> None:
        vwap, filled = vwap_buy_price([[0.60, 50]], 50)
        self.assertAlmostEqual(vwap, 0.60, places=6)
        self.assertAlmostEqual(filled, 50.0, places=6)

    def test_empty_and_nonpositive(self) -> None:
        self.assertIsNone(vwap_buy_price([], 100))
        self.assertIsNone(vwap_buy_price(None, 100))
        self.assertIsNone(vwap_buy_price([{"price": "0.5", "size": "10"}], 0))
        self.assertIsNone(vwap_buy_price([{"price": "0.5", "size": "0"}], 10))


class BookAwareEntryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        self._orig = pmd.get_order_book

    def tearDown(self) -> None:
        pmd.get_order_book = self._orig
        self._dir.cleanup()

    def test_entry_priced_off_ask_book_not_trades_mark(self) -> None:
        # /trades mark is 0.50, but the ask book lifts at a higher VWAP. With
        # book_entry the logged entry must be the ASK VWAP, not the mark.
        seen = {}

        def fake_book(token_id, session=None):
            seen["token"] = token_id
            # size 50 @ entry estimate; deep enough that depth cap (off) is moot.
            return {"asks": [{"price": "0.50", "size": "100"}, {"price": "0.60", "size": "100"}]}

        pmd.get_order_book = fake_book
        client = FakePricingClient({("0xM", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xM", "Yes", 0, asset="999")],
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            book_entry=True,
        )
        self.assertEqual(seen["token"], "999")
        self.assertEqual(len(logged), 1)
        rec = self.ledger.open_positions()[0]
        # best_ask 0.50 -> units = 100/0.50 = 200. Buy 100@0.50 + 100@0.60 = 110
        # over 200 -> VWAP 0.55. That is what must be recorded, not the 0.50 mark.
        self.assertAlmostEqual(rec.entry_price, 0.55, places=6)
        self.assertIn("ask-book VWAP", rec.notes)

    def test_no_asset_falls_back_to_trades_mark(self) -> None:
        # No 'asset' on the candidate -> book entry can't run -> use the mark.
        def boom(token_id, session=None):  # must not even be called without asset
            raise AssertionError("get_order_book called without an asset")

        pmd.get_order_book = boom
        client = FakePricingClient({("0xM", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xM", "Yes", 0)],  # no asset
            size_usd=100.0, mark_entry=True, dedup=False, book_entry=True,
        )
        self.assertEqual(len(logged), 1)
        rec = self.ledger.open_positions()[0]
        self.assertAlmostEqual(rec.entry_price, 0.50, places=6)  # mark fallback
        self.assertIn("latest /trades mark", rec.notes)

    def test_book_error_falls_back_to_trades_mark(self) -> None:
        def boom(token_id, session=None):
            raise RuntimeError("network")

        pmd.get_order_book = boom
        client = FakePricingClient({("0xM", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xM", "Yes", 0, asset="999")],
            size_usd=100.0, mark_entry=True, dedup=False, book_entry=True,
        )
        self.assertEqual(len(logged), 1)
        rec = self.ledger.open_positions()[0]
        self.assertAlmostEqual(rec.entry_price, 0.50, places=6)  # mark fallback
        self.assertIn("latest /trades mark", rec.notes)


class DepthCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        self._orig = pmd.get_order_book

    def tearDown(self) -> None:
        pmd.get_order_book = self._orig
        self._dir.cleanup()

    def test_thin_book_skipped(self) -> None:
        # Near-mid ASK depth ~= 50 @ 0.50 = $25. size_usd 100 >> 5% * 25 = $1.25.
        def thin_book(token_id, session=None):
            return {"asks": [{"price": "0.50", "size": "50"}]}

        pmd.get_order_book = thin_book
        client = FakePricingClient({("0xTHIN", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xTHIN", "Yes", 0, asset="111")],
            size_usd=100.0, mark_entry=True, dedup=False,
            # book_entry off so the mark drives the entry; only the cap is tested.
            book_entry=False, max_book_frac=0.05, depth_band=0.05,
        )
        self.assertEqual(logged, [])
        self.assertEqual(len(self.ledger.open_positions()), 0)

    def test_thin_book_marked_and_counted(self) -> None:
        def thin_book(token_id, session=None):
            return {"asks": [{"price": "0.50", "size": "50"}]}

        pmd.get_order_book = thin_book
        client = FakePricingClient({("0xTHIN", 0): 0.50})
        cands = [_cand("0xTHIN", "Yes", 0, asset="111")]
        runner._log_candidates(
            ledger=self.ledger, client=client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False,
            book_entry=False, max_book_frac=0.05, depth_band=0.05,
        )
        self.assertEqual(cands[0].get("skipped"), "thin_book")

    def test_deep_book_logged(self) -> None:
        # Near-mid depth: 100000 @ 0.50 = $50k. size 100 << 5% * 50k = $2.5k -> kept.
        def deep_book(token_id, session=None):
            return {"asks": [{"price": "0.50", "size": "100000"}]}

        pmd.get_order_book = deep_book
        client = FakePricingClient({("0xDEEP", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xDEEP", "Yes", 0, asset="222")],
            size_usd=100.0, mark_entry=True, dedup=False,
            book_entry=False, max_book_frac=0.05, depth_band=0.05,
        )
        self.assertEqual(len(logged), 1)
        self.assertEqual(len(self.ledger.open_positions()), 1)

    def test_depth_band_excludes_far_levels(self) -> None:
        # Best ask 0.50; a huge far level at 0.90 is OUTSIDE depth_band (0.05), so
        # near-mid depth is only the 50 @ 0.50 = $25 -> size 100 is too thin.
        def banded_book(token_id, session=None):
            return {"asks": [
                {"price": "0.50", "size": "50"},
                {"price": "0.90", "size": "1000000"},  # far from mid, excluded
            ]}

        pmd.get_order_book = banded_book
        client = FakePricingClient({("0xBAND", 0): 0.50})
        cands = [_cand("0xBAND", "Yes", 0, asset="333")]
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False,
            book_entry=False, max_book_frac=0.05, depth_band=0.05,
        )
        self.assertEqual(logged, [])
        self.assertEqual(cands[0].get("skipped"), "thin_book")

    def test_book_error_fails_open(self) -> None:
        # A book read error must NOT drop the candidate (fail open).
        def boom(token_id, session=None):
            raise RuntimeError("network")

        pmd.get_order_book = boom
        client = FakePricingClient({("0xERR", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xERR", "Yes", 0, asset="444")],
            size_usd=100.0, mark_entry=True, dedup=False,
            book_entry=False, max_book_frac=0.05, depth_band=0.05,
        )
        self.assertEqual(len(logged), 1)  # proceeded on the mark, not skipped
        self.assertEqual(len(self.ledger.open_positions()), 1)


class DepthCapDefaultOffRegressionTests(unittest.TestCase):
    """book_entry default off + max_book_frac default 0.0 == prior behavior."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        self._orig = pmd.get_order_book

    def tearDown(self) -> None:
        pmd.get_order_book = self._orig
        self._dir.cleanup()

    def test_defaults_never_touch_the_book(self) -> None:
        # With the new args defaulted off, get_order_book must never be called and
        # the entry must come straight from the /trades mark — exactly as before.
        def boom(token_id, session=None):
            raise AssertionError("get_order_book called with new args defaulted off")

        pmd.get_order_book = boom
        client = FakePricingClient({("0xM", 0): 0.50})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client,
            candidates=[_cand("0xM", "Yes", 0, asset="999")],
            size_usd=100.0, mark_entry=True, dedup=False,
            # book_entry / max_book_frac NOT passed -> function defaults (off).
        )
        self.assertEqual(len(logged), 1)
        rec = self.ledger.open_positions()[0]
        self.assertAlmostEqual(rec.entry_price, 0.50, places=6)
        self.assertIn("latest /trades mark", rec.notes)


if __name__ == "__main__":
    unittest.main()
