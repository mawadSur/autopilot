import tempfile
import unittest
from pathlib import Path

from state.pnl_ledger import PnlLedger
import whale_follow_runner as wf
from portfolio_reporter import _trade_fields


class ComputeConfidenceTest(unittest.TestCase):
    def test_more_holders_raises_score(self):
        s2, _ = wf.compute_confidence(2, [], min_convergence=2)
        s5, _ = wf.compute_confidence(5, [], min_convergence=2)
        self.assertGreater(s5, s2)

    def test_better_wallets_raise_score(self):
        lo, _ = wf.compute_confidence(3, [0.52, 0.55], min_convergence=3)
        hi, _ = wf.compute_confidence(3, [0.85, 0.90], min_convergence=3)
        self.assertGreater(hi, lo)

    def test_labels(self):
        _, high = wf.compute_confidence(6, [0.9, 0.9], min_convergence=3)
        _, low = wf.compute_confidence(2, [0.5], min_convergence=2)
        self.assertEqual(high, "high")
        self.assertEqual(low, "low")

    def test_neutral_quality_when_no_winrates(self):
        # conv_term(3/6=0.5) blended with neutral quality(0.5) -> 0.5.
        score, label = wf.compute_confidence(3, [], min_convergence=3)
        self.assertAlmostEqual(score, 0.5, places=3)
        self.assertEqual(label, "medium")

    def test_score_bounded_0_1(self):
        s, _ = wf.compute_confidence(99, [0.99, 0.99], min_convergence=3)
        self.assertLessEqual(s, 1.0)
        s0, _ = wf.compute_confidence(0, [0.1], min_convergence=3)
        self.assertGreaterEqual(s0, 0.0)


class _StatsObj:
    def __init__(self, win_rate):
        self.win_rate = win_rate


class _HoldersClient:
    """Fake read-only client: two target wallets hold outcomeIndex 0 of c1."""

    def get_holders(self, condition_id, limit=100):
        return [
            {
                "token": "tok",
                "holders": [
                    {"proxyWallet": "wA", "outcomeIndex": 0, "name": "Up"},
                    {"proxyWallet": "wB", "outcomeIndex": 0, "name": "Up"},
                ],
            }
        ]


class RunOnceConfidenceTest(unittest.TestCase):
    def test_confidence_on_candidate_and_record_notes(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        cands = wf.run_once(
            ledger=led,
            client=_HoldersClient(),
            target_wallets=["wA", "wB"],
            markets_condition_ids=["c1"],
            min_convergence=2,
            mark_entry=False,  # confidence is independent of entry pricing
            wallet_stats={"wA": _StatsObj(0.85), "wB": _StatsObj(0.90)},
        )
        self.assertEqual(len(cands), 1)
        self.assertIn("confidence", cands[0])
        self.assertIn(cands[0]["confidence_label"], ("low", "medium", "high"))
        # Only 2 converging wallets caps the count term, so even strong
        # win-rates land 'medium' (2 wallets is not high conviction).
        self.assertEqual(cands[0]["confidence_label"], "medium")
        rec = led.open_positions()[0]
        self.assertRegex(rec.notes, r"confidence=[0-9.]+ \((low|medium|high)\)")

    def test_confidence_computed_without_wallet_stats(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        cands = wf.run_once(
            ledger=led,
            client=_HoldersClient(),
            target_wallets=["wA", "wB"],
            markets_condition_ids=["c1"],
            min_convergence=2,
            mark_entry=False,
        )
        # Neutral quality, still produces a score/label and a notes marker.
        self.assertIn("confidence", cands[0])
        self.assertRegex(led.open_positions()[0].notes, r"confidence=")


class ReporterConfidenceDisplayTest(unittest.TestCase):
    def test_confidence_appended_to_discord_field(self):
        # title carries the record notes (mark_open_position uses notes as title).
        rows = [
            {
                "title": "SHADOW MODE - NO ORDERS; whale_convergence n=2 "
                "confidence=0.85 (high); outcomeIndex=0; wallets=wA,wB",
                "market_id": "c1",
                "side": "Up",
                "entry_price": 0.50,
                "current_price": 0.60,
                "unrealized_pnl_usd": 1.0,
                "trade_id": "whale-abcd1234",
            }
        ]
        fields = _trade_fields(rows, max_shown=10)
        joined = " ".join(fields.values())
        self.assertIn("conf 0.85 (high)", joined)

    def test_no_confidence_marker_is_fine(self):
        rows = [
            {
                "title": "intramarket arb",
                "market_id": "c1",
                "side": "YES+NO",
                "entry_price": 0.97,
                "current_price": 1.0,
                "unrealized_pnl_usd": 1.0,
                "trade_id": "arb-0001",
            }
        ]
        fields = _trade_fields(rows, max_shown=10)
        joined = " ".join(fields.values())
        self.assertNotIn("conf ", joined)


class LeaderboardQualityTest(unittest.TestCase):
    def test_top_rank_is_one(self):
        self.assertEqual(wf.leaderboard_quality(0, 50), 1.0)

    def test_bottom_rank_is_floor(self):
        self.assertAlmostEqual(wf.leaderboard_quality(49, 50), 0.6, places=3)

    def test_single_wallet_is_one(self):
        self.assertEqual(wf.leaderboard_quality(0, 1), 1.0)

    def test_monotonic_decreasing(self):
        self.assertGreater(
            wf.leaderboard_quality(0, 50), wf.leaderboard_quality(25, 50)
        )


class ConfidenceQualityScoresTest(unittest.TestCase):
    def test_quality_scores_override_winrates(self):
        # Win-rate 0.5 alone -> quality 0 -> low; but elite quality_scores win.
        lo, lo_label = wf.compute_confidence(3, [0.5, 0.5], min_convergence=3)
        hi, hi_label = wf.compute_confidence(
            3, [0.5, 0.5], min_convergence=3, quality_scores=[0.99, 0.98, 0.97]
        )
        self.assertGreater(hi, lo)
        self.assertEqual(hi_label, "high")  # 0.5*0.5 + 0.5*~0.98 = ~0.74

    def test_three_elite_wallets_read_high(self):
        # The exact leaderboard recalibration goal: 3 top wallets co-holding
        # should NOT read 'low' anymore.
        q = [wf.leaderboard_quality(i, 50) for i in (0, 1, 2)]
        score, label = wf.compute_confidence(3, [], min_convergence=3, quality_scores=q)
        self.assertEqual(label, "high")


class RunOnceLeaderboardQualityTest(unittest.TestCase):
    def test_wallet_quality_used_for_confidence(self):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        cands = wf.run_once(
            ledger=led,
            client=_HoldersClient(),
            target_wallets=["wA", "wB"],
            markets_condition_ids=["c1"],
            min_convergence=2,
            mark_entry=False,
            wallet_quality={"wA": 0.99, "wB": 0.98},  # elite -> should lift score
        )
        self.assertEqual(len(cands), 1)
        # 2 holders (conv 2/6=0.333) + elite quality ~0.985 -> ~0.66 -> high/medium
        self.assertIn(cands[0]["confidence_label"], ("medium", "high"))
        self.assertGreater(cands[0]["confidence"], 0.4)


class MinConfidenceEntryFilterTest(unittest.TestCase):
    """The ENTRY filter: skip convergences below the confidence floor.

    Drives run_once on the same _HoldersClient (wA + wB converge on idx 0 of c1).
    A low quality forces a low confidence (~0.25); a high quality forces a high
    one (>=0.5). We assert the low one is NOT logged when min_confidence=0.5 and
    the high one IS — proving the filter catches only the strongest signals.
    """

    def _run(self, *, min_confidence, wallet_quality):
        led = PnlLedger(str(Path(tempfile.mkdtemp()) / "l.jsonl"))
        cands = wf.run_once(
            ledger=led,
            client=_HoldersClient(),
            target_wallets=["wA", "wB"],
            markets_condition_ids=["c1"],
            min_convergence=2,
            mark_entry=False,
            wallet_quality=wallet_quality,
            min_confidence=min_confidence,
        )
        return led, cands

    def test_low_confidence_candidate_not_logged(self):
        # 2 holders (conv 2/6=0.333) + weak quality 0.10 -> conf ~0.22 < 0.5.
        led, cands = self._run(
            min_confidence=0.5, wallet_quality={"wA": 0.10, "wB": 0.10}
        )
        self.assertEqual(cands, [])  # filtered out before logging
        self.assertEqual(led.all_records(), [])  # nothing written to the ledger

    def test_high_confidence_candidate_logged(self):
        # 2 holders (conv 2/6=0.333) + elite quality ~0.985 -> conf ~0.66 >= 0.5.
        led, cands = self._run(
            min_confidence=0.5, wallet_quality={"wA": 0.99, "wB": 0.98}
        )
        self.assertEqual(len(cands), 1)
        self.assertGreaterEqual(cands[0]["confidence"], 0.5)
        self.assertEqual(len(led.all_records()), 1)

    def test_default_min_confidence_zero_logs_everything(self):
        # min_confidence defaults to 0.0 (off): even a weak signal is logged.
        led, cands = self._run(
            min_confidence=0.0, wallet_quality={"wA": 0.10, "wB": 0.10}
        )
        self.assertEqual(len(cands), 1)
        self.assertEqual(len(led.all_records()), 1)


if __name__ == "__main__":
    unittest.main()
