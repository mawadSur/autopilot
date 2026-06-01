import tempfile
import unittest
from pathlib import Path

from state.pnl_ledger import PnlLedger, TradeRecord
import whale_follow_runner as wf
from whale_follow_runner import _log_candidates
import trade_blocklist as tb


class BlocklistMatchTest(unittest.TestCase):
    def setUp(self):
        self.terms = tb.load_blocklist()

    def test_blocks_alcohol_casino_adult(self):
        self.assertEqual(tb.blocked_term("Will this beer brand IPO?", self.terms), "beer")
        self.assertEqual(tb.blocked_term("New casino in Vegas", self.terms), "casino")
        self.assertIsNotNone(tb.blocked_term("OnlyFans top creator 2026", self.terms))

    def test_no_false_positive_word_boundary(self):
        # 'wine' must NOT match 'winner'; common market words stay tradeable.
        self.assertIsNone(tb.blocked_term("Who will be the winner?", self.terms))
        self.assertIsNone(tb.blocked_term("S&P 500 close above 6000", self.terms))
        self.assertIsNone(tb.blocked_term("US presidential election result", self.terms))

    def test_case_insensitive_and_multiword(self):
        self.assertEqual(tb.blocked_term("ADULT FILM awards night", self.terms), "adult film")

    def test_file_and_extra_terms(self):
        d = tempfile.mkdtemp()
        fp = Path(d) / "bl.txt"
        fp.write_text("# my extras\ncrypto rug\n")
        terms = tb.load_blocklist(str(fp), extra_terms=["tobacco"])
        self.assertIn("tobacco", terms)
        self.assertIn("crypto rug", terms)
        self.assertEqual(tb.blocked_term("Tobacco ban referendum", terms), "tobacco")

    def test_no_blocklist_blocks_nothing(self):
        self.assertIsNone(tb.blocked_term("Casino stock", []))


class _NoMarkClient:
    """mark_entry=False path needs no client calls."""


def _cand(cid, outcome, wallets, *, title=None, oi=0):
    return {
        "conditionId": cid,
        "outcomeIndex": oi,
        "outcome": outcome,
        "title": title,
        "n_target_holders": len(wallets),
        "wallets": wallets,
    }


def _open_record(market_id, side):
    return TradeRecord(
        trade_id=f"x-{market_id}-{side}",
        ts_utc="2026-06-01T00:00:00+00:00",
        venue="polymarket",
        market_id=market_id,
        side=side,
        entry_price=0.5,
        size=100.0,
        fees_usd=0.0,
        slippage_bps=0.0,
        strategy="whale_convergence",
        status="open",
    )


class DedupTest(unittest.TestCase):
    def test_skips_already_open_market_outcome(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        led.append(_open_record("c1", "Yes"))
        logged = _log_candidates(
            ledger=led, client=_NoMarkClient(),
            candidates=[_cand("c1", "Yes", ["a", "b", "c"])],
            min_convergence=3, mark_entry=False, dedup=True,
        )
        self.assertEqual(len(logged), 0)  # already open -> not re-logged

    def test_different_market_still_logs(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        led.append(_open_record("c1", "Yes"))
        logged = _log_candidates(
            ledger=led, client=_NoMarkClient(),
            candidates=[_cand("c2", "No", ["a", "b", "c"])],
            min_convergence=3, mark_entry=False, dedup=True,
        )
        self.assertEqual(len(logged), 1)

    def test_dedup_within_one_batch(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        cands = [_cand("c1", "Yes", ["a", "b", "c"]), _cand("c1", "Yes", ["a", "b", "c"])]
        logged = _log_candidates(
            ledger=led, client=_NoMarkClient(), candidates=cands,
            min_convergence=3, mark_entry=False, dedup=True,
        )
        self.assertEqual(len(logged), 1)

    def test_no_dedup_allows_relog(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        led.append(_open_record("c1", "Yes"))
        logged = _log_candidates(
            ledger=led, client=_NoMarkClient(),
            candidates=[_cand("c1", "Yes", ["a", "b", "c"])],
            min_convergence=3, mark_entry=False, dedup=False,
        )
        self.assertEqual(len(logged), 1)


class BlocklistLogTest(unittest.TestCase):
    def test_blocked_market_not_logged(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        terms = tb.load_blocklist()
        cands = [
            _cand("c1", "Yes", ["a", "b", "c"], title="Best casino stock 2026"),
            _cand("c2", "No", ["a", "b", "c"], title="US election outcome"),
        ]
        logged = _log_candidates(
            ledger=led, client=_NoMarkClient(), candidates=cands,
            min_convergence=3, mark_entry=False, blocklist=terms, dedup=True,
        )
        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0]["conditionId"], "c2")
        self.assertEqual(cands[0].get("blocked"), "casino")


class _PosClient:
    def __init__(self, by_wallet):
        self._by = by_wallet

    def get_positions(self, user, limit=500):
        return self._by.get(user, [])


def _pos(cid, title, *, oi=0, outcome="Yes", size=10.0, cur=0.5, redeemable=False):
    return {
        "conditionId": cid, "outcomeIndex": oi, "title": title, "outcome": outcome,
        "size": size, "curPrice": cur, "redeemable": redeemable,
    }


class ConvergenceTitleTest(unittest.TestCase):
    def test_title_captured_into_candidate(self):
        client = _PosClient({
            "w1": [_pos("c1", "Casino IPO date")],
            "w2": [_pos("c1", "Casino IPO date")],
        })
        cands = wf.convergence_from_positions(client, ["w1", "w2"], min_convergence=2)
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["title"], "Casino IPO date")


if __name__ == "__main__":
    unittest.main()
