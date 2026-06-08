"""Tests for the win-rate filters in whale_follow_runner (SHADOW-ONLY).

Two filters added to delete structurally -EV trades before they reach the ledger:

  * ENTRY-BAND ``[min_entry_price, max_entry_price]`` — skip entries near $0 (the
    losing side) or near $1 (fee-drag, near-decided market). Only enter where the
    outcome is genuinely uncertain.
  * DIRECTIONAL READ (``require_directional``) — drop any market where convergence
    fired on more than one outcome (target wallets split both sides = no
    directional edge to follow).

No network: a fake client returns canned ``get_trades`` prices. These exercise
:func:`_log_candidates` directly (the shared back-end for both runner modes).
"""

from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any, Dict, List, Optional

import whale_follow_runner as runner
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


def _cand(cid: str, outcome: str, oi: int, wallets=("0xA", "0xB", "0xC")) -> Dict[str, Any]:
    return {
        "conditionId": cid,
        "outcome": outcome,
        "outcomeIndex": oi,
        "wallets": list(wallets),
        "n_target_holders": len(wallets),
        "title": f"{outcome} market",
    }


class EntryBandFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_in_band_kept_out_of_band_skipped(self) -> None:
        # Distinct markets so the directional filter is irrelevant here.
        client = FakePricingClient({
            ("0xMID", 0): 0.50,    # in band -> logged
            ("0xHIGH", 0): 0.97,   # near $1 -> fee drag -> skipped
            ("0xLOW", 0): 0.03,    # near $0 -> losing side -> skipped
        })
        cands = [_cand("0xMID", "Mid", 0), _cand("0xHIGH", "High", 0), _cand("0xLOW", "Low", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=cands,
            mark_entry=True, dedup=False, min_confidence=0.0,
            min_entry_price=0.15, max_entry_price=0.85, require_directional=False,
        )
        kept = {c["conditionId"] for c in logged}
        self.assertEqual(kept, {"0xMID"})
        self.assertEqual(len(self.ledger.open_positions()), 1)
        self.assertEqual(self.ledger.open_positions()[0].side, "Mid")

    def test_unmarkable_entry_is_out_of_band(self) -> None:
        # No price for this market -> entry 0.0 -> below the floor -> skipped.
        client = FakePricingClient({})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=[_cand("0xNONE", "X", 0)],
            mark_entry=True, dedup=False, min_entry_price=0.15, max_entry_price=0.85,
        )
        self.assertEqual(logged, [])
        self.assertEqual(len(self.ledger.open_positions()), 0)

    def test_band_off_keeps_extremes(self) -> None:
        # Default band (0..1) is off -> a $0.97 entry is still logged.
        client = FakePricingClient({("0xHIGH", 0): 0.97})
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=[_cand("0xHIGH", "High", 0)],
            mark_entry=True, dedup=False,  # min_entry_price/max default to 0.0/1.0
        )
        self.assertEqual(len(logged), 1)


class DirectionalFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_split_market_both_sides_dropped(self) -> None:
        # Convergence fired on BOTH outcomes of one market -> no directional read.
        client = FakePricingClient({})
        cands = [_cand("0xGAME", "Yes", 0), _cand("0xGAME", "No", 1)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=cands,
            mark_entry=False, dedup=False, require_directional=True,
        )
        self.assertEqual(logged, [])
        self.assertEqual(len(self.ledger.open_positions()), 0)

    def test_single_sided_market_kept(self) -> None:
        client = FakePricingClient({})
        cands = [_cand("0xGAME", "Yes", 0), _cand("0xOTHER", "No", 1)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=cands,
            mark_entry=False, dedup=False, require_directional=True,
        )
        self.assertEqual({c["conditionId"] for c in logged}, {"0xGAME", "0xOTHER"})

    def test_directional_off_keeps_both_sides(self) -> None:
        client = FakePricingClient({})
        cands = [_cand("0xGAME", "Yes", 0), _cand("0xGAME", "No", 1)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=client, candidates=cands,
            mark_entry=False, dedup=False, require_directional=False,
        )
        self.assertEqual(len(logged), 2)


if __name__ == "__main__":
    unittest.main()
