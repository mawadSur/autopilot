"""Tests for the W3 risk-of-ruin gates in whale_follow_runner (SHADOW-ONLY).

Two additive RISK GATES that refuse entries before they reach the ledger
(Constitution: never weaken a risk gate, only add):

  * EXPOSURE CAP (``max_total_exposure``) — refuse a new entry once the total
    open notional + this position's ``size_usd`` would exceed the cap. A single
    scan can't blow the cap because each logged entry bumps the running exposure.
    Marked ``cand['skipped']='exposure_cap'``.
  * DAILY-LOSS KILL SWITCH (``daily_loss_limit``) — once today's (current UTC
    date) realized loss reaches the limit, refuse EVERY candidate this scan.
    Marked ``cand['skipped']='kill_switch'``.

Both default OFF (0.0), so every prior test that doesn't pass them is unchanged.

No network: a minimal fake read-only client returns canned ``get_trades`` prices.
These exercise :func:`_log_candidates` directly (the shared back-end for both
runner modes), seeding the ledger with :class:`TradeRecord` rows via a helper.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import whale_follow_runner as runner
from state.pnl_ledger import PnlLedger, TradeRecord


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


def _open_record(market_id: str, side: str, size: float) -> TradeRecord:
    """An OPEN whale-convergence record carrying ``size`` USD notional."""
    return TradeRecord(
        trade_id=f"whale-{uuid.uuid4().hex[:12]}",
        ts_utc="2026-05-01T00:00:00+00:00",
        venue="polymarket",
        market_id=market_id,
        side=side,
        entry_price=0.50,
        size=float(size),
        fees_usd=0.0,
        slippage_bps=0.0,
        strategy=runner.STRATEGY,
        status="open",
        notes="SHADOW MODE - NO ORDERS; seeded open",
    )


def _seed_open(ledger: PnlLedger, market_id: str, side: str, size: float) -> None:
    """Append a seeded OPEN position to the ledger (counts toward exposure)."""
    ledger.append(_open_record(market_id, side, size))


def _seed_settled(
    ledger: PnlLedger, market_id: str, side: str, realized_pnl: float, exit_ts: str
) -> None:
    """Append + settle a position so it lands in ``ledger.settled()`` with P/L."""
    record = _open_record(market_id, side, size=100.0)
    ledger.append(record)
    ledger.settle(
        record.trade_id,
        exit_price=0.0,
        exit_ts_utc=exit_ts,
        market_outcome=side,
        realized_pnl_usd=float(realized_pnl),
    )


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class ExposureCapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        # Entries mark at $0.50 (in-band), $100 notional each.
        self.client = FakePricingClient({
            ("0xNEW1", 0): 0.50,
            ("0xNEW2", 0): 0.50,
        })

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_next_candidate_skipped_once_over_cap(self) -> None:
        # Seed open exposure near the cap: 950 of a 1000 cap. A $100 entry would
        # push to 1050 > 1000 -> refused.
        _seed_open(self.ledger, "0xOLD", "Yes", size=950.0)
        cands = [_cand("0xNEW1", "New1", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            max_total_exposure=1000.0,
        )
        self.assertEqual(logged, [])
        self.assertEqual(cands[0].get("skipped"), "exposure_cap")
        # No new open record was written (only the seeded one remains).
        self.assertEqual(len(self.ledger.open_positions()), 1)

    def test_candidate_fits_when_under_cap(self) -> None:
        # Seed open exposure of 800; a $100 entry -> 900 <= 1000 -> logged.
        _seed_open(self.ledger, "0xOLD", "Yes", size=800.0)
        cands = [_cand("0xNEW1", "New1", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            max_total_exposure=1000.0,
        )
        self.assertEqual({c["conditionId"] for c in logged}, {"0xNEW1"})
        self.assertEqual(len(self.ledger.open_positions()), 2)

    def test_running_exposure_caps_within_one_scan(self) -> None:
        # Empty ledger, cap 150, $100 entries: the FIRST logs (0+100<=150), the
        # SECOND is refused (100+100=200>150) — a single scan can't blow the cap.
        cands = [_cand("0xNEW1", "New1", 0), _cand("0xNEW2", "New2", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            max_total_exposure=150.0,
        )
        self.assertEqual([c["conditionId"] for c in logged], ["0xNEW1"])
        self.assertEqual(cands[1].get("skipped"), "exposure_cap")
        self.assertEqual(len(self.ledger.open_positions()), 1)


class KillSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        self.client = FakePricingClient({
            ("0xNEW1", 0): 0.50,
            ("0xNEW2", 0): 0.50,
        })
        self.now = datetime.now(timezone.utc)

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_tripped_skips_all_candidates(self) -> None:
        # Today's realized = -250 <= -200 limit -> kill switch trips, ALL refused.
        _seed_settled(
            self.ledger, "0xLOSS", "Yes", realized_pnl=-250.0, exit_ts=_iso(self.now)
        )
        cands = [_cand("0xNEW1", "New1", 0), _cand("0xNEW2", "New2", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            daily_loss_limit=200.0,
        )
        self.assertEqual(logged, [])
        self.assertTrue(all(c.get("skipped") == "kill_switch" for c in cands))
        # Only the seeded settled record exists; no new opens.
        self.assertEqual(len(self.ledger.open_positions()), 0)

    def test_not_tripped_above_limit(self) -> None:
        # Today's realized = -150 > -200 limit -> NOT tripped, candidates log.
        _seed_settled(
            self.ledger, "0xLOSS", "Yes", realized_pnl=-150.0, exit_ts=_iso(self.now)
        )
        cands = [_cand("0xNEW1", "New1", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            daily_loss_limit=200.0,
        )
        self.assertEqual({c["conditionId"] for c in logged}, {"0xNEW1"})
        self.assertIsNone(cands[0].get("skipped"))

    def test_yesterday_loss_does_not_trip_today(self) -> None:
        # A big loss dated YESTERDAY must not trip today's switch (date-scoped).
        yesterday = self.now - timedelta(days=1)
        _seed_settled(
            self.ledger, "0xLOSS", "Yes", realized_pnl=-500.0, exit_ts=_iso(yesterday)
        )
        cands = [_cand("0xNEW1", "New1", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            daily_loss_limit=200.0,
        )
        self.assertEqual({c["conditionId"] for c in logged}, {"0xNEW1"})

    def test_trailing_z_timestamp_counts_today(self) -> None:
        # A 'Z'-suffixed UTC timestamp for today must be parsed and counted.
        z_ts = self.now.strftime("%Y-%m-%dT%H:%M:%SZ")
        _seed_settled(
            self.ledger, "0xLOSS", "Yes", realized_pnl=-300.0, exit_ts=z_ts
        )
        cands = [_cand("0xNEW1", "New1", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
            daily_loss_limit=200.0,
        )
        self.assertEqual(logged, [])
        self.assertEqual(cands[0].get("skipped"), "kill_switch")


class RiskGatesOffByDefaultTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.ledger = PnlLedger(os.path.join(self._dir.name, "l.jsonl"))
        self.client = FakePricingClient({("0xNEW1", 0): 0.50})

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_defaults_do_not_gate(self) -> None:
        # Huge seeded exposure + a huge today loss, but BOTH gates default OFF
        # (no max_total_exposure / daily_loss_limit args) -> candidate still logs.
        _seed_open(self.ledger, "0xOLD", "Yes", size=1_000_000.0)
        _seed_settled(
            self.ledger, "0xLOSS", "No", realized_pnl=-1_000_000.0,
            exit_ts=datetime.now(timezone.utc).isoformat(),
        )
        cands = [_cand("0xNEW1", "New1", 0)]
        logged = runner._log_candidates(
            ledger=self.ledger, client=self.client, candidates=cands,
            size_usd=100.0, mark_entry=True, dedup=False, min_confidence=0.0,
        )
        self.assertEqual({c["conditionId"] for c in logged}, {"0xNEW1"})
        self.assertIsNone(cands[0].get("skipped"))


if __name__ == "__main__":
    unittest.main()
