"""Tests for src/dashboard/state.py (READ-ONLY observability).

``build_state`` is driven against a real tempfile :class:`PnlLedger` carrying a
mix of open + settled records (a stop-loss exit, a take-profit exit, a resolution
win, and a resolution loss). No sockets, no network: we test the pure state
builder directly. We assert:

  * summary counts (n_open / n_settled) match the seeded book;
  * exit_mix categorizes 'exit:stop_loss'/'exit:take_profit' by reason and
    resolution settlements by realized-P/L sign (won/lost);
  * the equity_curve starts at bankroll, is monotonic-in-time (sorted ASC), and
    its final equity == bankroll + total realized P/L;
  * confidence is parsed out of the notes marker for open positions;
  * closed_positions is the exit feed sorted DESC by exit_ts_utc (newest first).

READ-ONLY: the state builder reads the ledger and shapes it for display — there
is no order/settle/write surface anywhere in the module under test.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from dashboard.state import (
    build_state,
    clean_title,
    normalize_exit_reason,
    parse_confidence,
)
from state.pnl_ledger import PnlLedger, TradeRecord

# Entry/exit timestamps chosen so exit_ts >= entry ts (ledger look-ahead guard)
# and so the four settlements have a clear, distinct chronological order.
ENTRY_TS = "2026-06-01T12:00:00+00:00"
EXIT_T1 = "2026-06-02T09:00:00+00:00"  # stop-loss (oldest exit)
EXIT_T2 = "2026-06-02T11:00:00+00:00"  # take-profit
EXIT_T3 = "2026-06-02T13:00:00+00:00"  # resolution win
EXIT_T4 = "2026-06-02T15:00:00+00:00"  # resolution loss (newest exit)

BANKROLL = 1000.0


def _open_record(trade_id, *, side, market_id, notes, entry_price=0.5):
    return TradeRecord(
        trade_id=trade_id,
        ts_utc=ENTRY_TS,
        venue="polymarket",
        market_id=market_id,
        side=side,
        entry_price=entry_price,
        size=100.0,
        fees_usd=0.0,
        slippage_bps=0.0,
        strategy="whale_convergence",
        notes=notes,
    )


class BuildStateTest(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        self.ledger = PnlLedger(self.path)

        # One open position (stays open), with a confidence marker in notes.
        self.ledger.append(
            _open_record(
                "whale-open-1",
                side="Milwaukee Brewers",
                market_id="0x" + "a" * 60,
                notes="SHADOW MODE - NO ORDERS; whale_convergence n=3 "
                "confidence=0.65 (medium); outcomeIndex=0",
            )
        )
        # A second open position with a HIGH confidence label.
        self.ledger.append(
            _open_record(
                "whale-open-2",
                side="Kansas City Royals",
                market_id="0x" + "b" * 60,
                notes="SHADOW MODE; whale_convergence n=4 confidence=0.73 (high)",
            )
        )

        # Four positions that will be settled four different ways.
        for tid, side in [
            ("whale-stop", "Anastasia Potapova"),
            ("whale-tp", "Felix Auger-Aliassime"),
            ("whale-won", "Anna Kalinskaya"),
            ("whale-lost", "Diane Parry"),
        ]:
            self.ledger.append(
                _open_record(
                    tid,
                    side=side,
                    market_id="0x" + tid.encode().hex(),
                    notes="SHADOW MODE; whale_convergence confidence=0.60 (medium)",
                )
            )

        # stop-loss exit: market_outcome 'exit:stop_loss', realized loss.
        self.ledger.settle(
            "whale-stop", exit_price=0.40, exit_ts_utc=EXIT_T1,
            market_outcome="exit:stop_loss", realized_pnl_usd=-18.0,
        )
        # take-profit exit: market_outcome 'exit:take_profit', realized gain.
        self.ledger.settle(
            "whale-tp", exit_price=0.70, exit_ts_utc=EXIT_T2,
            market_outcome="exit:take_profit", realized_pnl_usd=39.0,
        )
        # resolution win: non-exit outcome, positive realized.
        self.ledger.settle(
            "whale-won", exit_price=1.0, exit_ts_utc=EXIT_T3,
            market_outcome="won:Anna Kalinskaya", realized_pnl_usd=100.0,
        )
        # resolution loss: non-exit outcome, non-positive realized.
        self.ledger.settle(
            "whale-lost", exit_price=0.0, exit_ts_utc=EXIT_T4,
            market_outcome="lost:Diane Parry", realized_pnl_usd=-100.0,
        )

        self.state = build_state(self.ledger, bankroll_usd=BANKROLL)

    def tearDown(self):
        try:
            os.remove(self.path)
        except OSError:
            pass

    # ---- summary counts ----
    def test_summary_counts(self):
        s = self.state["summary"]
        self.assertEqual(s["n_open"], 2)
        self.assertEqual(s["n_settled"], 4)
        self.assertEqual(s["bankroll_usd"], BANKROLL)
        # realized = -18 + 39 + 100 - 100 = 21
        self.assertAlmostEqual(s["realized_pnl_usd"], 21.0, places=6)
        # 2 winners (tp, won) out of 4 settled.
        self.assertAlmostEqual(s["win_rate"], 0.5, places=6)

    # ---- exit_mix categorization ----
    def test_exit_mix_categorization(self):
        mix = self.state["exit_mix"]
        self.assertEqual(mix.get("stop_loss"), 1)
        self.assertEqual(mix.get("take_profit"), 1)
        self.assertEqual(mix.get("won"), 1)
        self.assertEqual(mix.get("lost"), 1)
        # exactly four buckets accounted for, summing to n_settled.
        self.assertEqual(sum(mix.values()), 4)

    def test_normalize_exit_reason_helper(self):
        self.assertEqual(normalize_exit_reason("exit:stop_loss", -5.0), "stop_loss")
        self.assertEqual(normalize_exit_reason("exit:take_profit", 5.0), "take_profit")
        self.assertEqual(normalize_exit_reason("won:Someone", 5.0), "won")
        self.assertEqual(normalize_exit_reason("lost:Someone", -5.0), "lost")
        # A resolution with zero realized is a loss (not strictly positive).
        self.assertEqual(normalize_exit_reason("resolved", 0.0), "lost")
        # None outcome falls through to realized-sign logic.
        self.assertEqual(normalize_exit_reason(None, 1.0), "won")

    # ---- equity curve ----
    def test_equity_curve_starts_at_bankroll_and_is_time_ordered(self):
        curve = self.state["equity_curve"]
        # start point + one per settlement.
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve[0]["t"], "start")
        self.assertAlmostEqual(curve[0]["equity"], BANKROLL, places=6)

        # Timestamps (after the start sentinel) are strictly ascending.
        ts = [p["t"] for p in curve[1:]]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(ts, [EXIT_T1, EXIT_T2, EXIT_T3, EXIT_T4])

    def test_equity_curve_ends_at_bankroll_plus_total_realized(self):
        curve = self.state["equity_curve"]
        total_realized = self.state["summary"]["realized_pnl_usd"]
        self.assertAlmostEqual(
            curve[-1]["equity"], BANKROLL + total_realized, places=6
        )
        # Spot-check the running accumulation along the way.
        self.assertAlmostEqual(curve[1]["equity"], BANKROLL - 18.0, places=6)
        self.assertAlmostEqual(curve[2]["equity"], BANKROLL - 18.0 + 39.0, places=6)

    # ---- confidence parsing on open positions ----
    def test_open_positions_parse_confidence(self):
        opens = {p["title"]: p for p in self.state["open_positions"]}
        self.assertEqual(len(self.state["open_positions"]), 2)
        confs = sorted(
            (p["confidence"]["score"], p["confidence"]["label"])
            for p in self.state["open_positions"]
        )
        self.assertEqual(confs, [(0.65, "medium"), (0.73, "high")])
        # Each open carries its side and a numeric entry price.
        for p in self.state["open_positions"]:
            self.assertIn(p["side"], ("Milwaukee Brewers", "Kansas City Royals"))
            self.assertIsInstance(p["entry_price"], float)

    def test_parse_confidence_helper(self):
        self.assertEqual(
            parse_confidence("foo confidence=0.55 (low) bar"),
            {"score": 0.55, "label": "low"},
        )
        self.assertIsNone(parse_confidence("no marker here"))
        self.assertIsNone(parse_confidence(None))

    # ---- closed positions are the exit feed, newest first ----
    def test_closed_positions_sorted_desc_by_exit_ts(self):
        closed = self.state["closed_positions"]
        self.assertEqual(len(closed), 4)
        ts = [c["exit_ts_utc"] for c in closed]
        self.assertEqual(ts, [EXIT_T4, EXIT_T3, EXIT_T2, EXIT_T1])
        # Newest item is the resolution loss; carries its normalized reason.
        self.assertEqual(closed[0]["reason"], "lost")
        self.assertAlmostEqual(closed[0]["realized_pnl_usd"], -100.0, places=6)

    # ---- title cleaning falls back to a short market id ----
    def test_clean_title_shortens_hex_market_id(self):
        title = clean_title(notes="no title here", market_id="0x" + "c" * 60)
        self.assertIn("…", title)
        self.assertTrue(title.startswith("0x"))
        # An explicit title= marker in notes wins.
        self.assertEqual(
            clean_title(notes="x; title=Some Market; y", market_id="0xabc"),
            "Some Market",
        )

    # ---- robustness: empty ledger yields a sane empty state ----
    def test_empty_ledger_is_safe(self):
        fd, empty_path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            state = build_state(PnlLedger(empty_path), bankroll_usd=BANKROLL)
            self.assertEqual(state["summary"]["n_open"], 0)
            self.assertEqual(state["summary"]["n_settled"], 0)
            self.assertEqual(state["open_positions"], [])
            self.assertEqual(state["closed_positions"], [])
            self.assertEqual(state["exit_mix"], {})
            # Only the start point on the curve.
            self.assertEqual(len(state["equity_curve"]), 1)
            self.assertAlmostEqual(
                state["equity_curve"][0]["equity"], BANKROLL, places=6
            )
        finally:
            os.remove(empty_path)


if __name__ == "__main__":
    unittest.main()
