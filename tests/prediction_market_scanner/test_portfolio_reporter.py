import tempfile
import unittest
from pathlib import Path

from state.pnl_ledger import PnlLedger, TradeRecord
import portfolio_reporter as pr


def _rec(trade_id, side, entry_price, size, *, strategy="intramarket_arb",
         fees_usd=0.0, status="open", venue="polymarket", market_id="m1"):
    return TradeRecord(
        trade_id=trade_id,
        ts_utc="2026-05-31T00:00:00+00:00",
        venue=venue,
        market_id=market_id,
        side=side,
        entry_price=entry_price,
        size=size,
        fees_usd=fees_usd,
        slippage_bps=0.0,
        strategy=strategy,
        status=status,
    )


class MarkOpenPositionTest(unittest.TestCase):
    def test_marked_unrealized_pnl_net_of_fee(self):
        # 100 USD notional bought at 0.50 -> 200 units; mark 0.60 -> value 120;
        # unrealized = 120 - 100 - 1 fee = 19.
        rec = _rec("t1", "YES", 0.50, 100.0, fees_usd=1.0)
        row = pr.mark_open_position(rec, 0.60)
        self.assertTrue(row["marked"])
        self.assertAlmostEqual(row["current_value_usd"], 120.0, places=6)
        self.assertAlmostEqual(row["unrealized_pnl_usd"], 19.0, places=6)

    def test_arb_pair_marked_at_one_dollar(self):
        # YES+NO pair, cost basis 0.97, 100 USD -> ~103.09 units; mark 1.0.
        rec = _rec("t2", "YES+NO", 0.97, 100.0, fees_usd=2.0)
        row = pr.mark_open_position(rec, 1.0)
        self.assertAlmostEqual(row["current_value_usd"], 100.0 / 0.97, places=6)
        # ~103.09 - 100 - 2 = ~1.09
        self.assertAlmostEqual(row["unrealized_pnl_usd"], 100.0 / 0.97 - 102.0, places=6)

    def test_none_price_is_pending_not_zero(self):
        rec = _rec("t3", "YES", 0.50, 100.0)
        row = pr.mark_open_position(rec, None)
        self.assertFalse(row["marked"])
        self.assertIsNone(row["unrealized_pnl_usd"])
        self.assertIsNone(row["current_value_usd"])

    def test_zero_entry_price_is_pending(self):
        rec = _rec("t4", "YES", 0.0, 100.0)
        row = pr.mark_open_position(rec, 0.5)
        self.assertFalse(row["marked"])


class BuildReportTest(unittest.TestCase):
    def _ledger(self):
        d = tempfile.mkdtemp()
        led = PnlLedger(str(Path(d) / "ledger.jsonl"))
        led.append(_rec("a", "YES", 0.50, 100.0, fees_usd=1.0))   # markable
        led.append(_rec("b", "YES", 0.40, 100.0, fees_usd=0.0))   # pending
        led.append(_rec("c", "YES", 0.50, 100.0, status="open"))  # to settle
        led.settle("c", exit_price=1.0, exit_ts_utc="2026-06-02T00:00:00+00:00",
                   market_outcome="Yes", realized_pnl_usd=25.0)
        return led

    def _price_fn(self, rec):
        return {"a": 0.60}.get(rec.trade_id)  # only 'a' is markable; 'b' -> None

    def test_report_math(self):
        led = self._ledger()
        rep = pr.build_report(led, price_fn=self._price_fn, bankroll_usd=1000.0)
        # realized from settled 'c' = 25.0
        self.assertAlmostEqual(rep["realized_pnl_usd"], 25.0, places=6)
        # unrealized only from 'a': 200 units * 0.60 - 100 - 1 = 19.0
        self.assertAlmostEqual(rep["unrealized_pnl_usd"], 19.0, places=6)
        self.assertEqual(rep["n_pending_mark"], 1)  # 'b'
        self.assertAlmostEqual(rep["equity_usd"], 1000.0 + 25.0 + 19.0, places=6)
        self.assertEqual(rep["n_settled"], 1)

    def test_pending_does_not_inflate_equity(self):
        led = self._ledger()
        rep = pr.build_report(led, price_fn=self._price_fn, bankroll_usd=1000.0)
        # 'b' (pending) contributes nothing to unrealized.
        self.assertNotIn(None, [r["marked"] for r in rep["open_positions"]])
        marked = [r for r in rep["open_positions"] if r["marked"]]
        self.assertEqual(len(marked), 1)


class _FakeNotifier:
    def __init__(self):
        self.info_calls = []

    def info(self, message, *, fields=None):
        self.info_calls.append((message, fields or {}))
        return True


class ReportToDiscordTest(unittest.TestCase):
    def test_posts_trades_and_portfolio(self):
        d = tempfile.mkdtemp()
        led = PnlLedger(str(Path(d) / "ledger.jsonl"))
        led.append(_rec("a", "YES", 0.50, 100.0, fees_usd=1.0))
        notif = _FakeNotifier()
        rep = pr.report_to_discord(led, notif, price_fn=lambda r: 0.60,
                                   bankroll_usd=1000.0)
        # Two info posts: trades + portfolio.
        self.assertEqual(len(notif.info_calls), 2)
        portfolio_msg, portfolio_fields = notif.info_calls[-1]
        self.assertIn("Portfolio", portfolio_msg)
        self.assertIn("Equity", portfolio_fields)
        self.assertIn("Realized P/L", portfolio_fields)
        self.assertIn("Unrealized P/L", portfolio_fields)
        self.assertAlmostEqual(rep["unrealized_pnl_usd"], 19.0, places=6)

    def test_no_trades_skips_trade_post(self):
        d = tempfile.mkdtemp()
        led = PnlLedger(str(Path(d) / "ledger.jsonl"))  # empty
        notif = _FakeNotifier()
        pr.report_to_discord(led, notif, bankroll_usd=1000.0)
        # Only the portfolio summary, no per-trade post.
        self.assertEqual(len(notif.info_calls), 1)
        self.assertIn("Portfolio", notif.info_calls[0][0])


if __name__ == "__main__":
    unittest.main()
