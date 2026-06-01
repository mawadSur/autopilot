import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from analytics_dashboard import (
    _extract_trades,
    _flatten_review_entry,
    _is_good_process,
    _is_win,
    _resolve_quadrant,
    summarize_trades,
)


def _review_entry(quadrant: str, *, final_outcome: bool, trade_id: str = "abc") -> dict:
    return {
        "trade_id": trade_id,
        "source_file": f"trade_execution_{trade_id}.json",
        "trade_key": f"trade_execution_{trade_id}.json:{trade_id}",
        "settled_at": "2026-04-25T00:00:00Z",
        "reviewed_at": "2026-04-25T01:00:00Z",
        "outcome_review": {
            "matrix_classification": quadrant,
            "thesis_held": True,
            "unknown_at_entry": False,
            "calibration_reasonable": True,
            "resulting_detected": False,
            "research_module_flaw": False,
            "risk_module_flaw": False,
            "key_takeaways": [],
            "reasoning": "Test entry.",
        },
        "final_outcome": final_outcome,
    }


class FlattenReviewEntryTests(unittest.TestCase):
    def test_lifts_outcome_review_fields_onto_entry(self):
        entry = _review_entry("Deserved Success", final_outcome=True)
        flat = _flatten_review_entry(entry)
        self.assertEqual(flat["matrix_classification"], "Deserved Success")
        self.assertTrue(flat["final_outcome"])
        self.assertEqual(flat["trade_id"], "abc")

    def test_does_not_clobber_existing_top_level_keys(self):
        entry = _review_entry("Dumb Luck", final_outcome=True)
        entry["matrix_classification"] = "Override"  # top-level should win
        flat = _flatten_review_entry(entry)
        self.assertEqual(flat["matrix_classification"], "Override")


class ExtractTradesTests(unittest.TestCase):
    def test_handles_performance_tracker_reviews_shape(self):
        payload = {
            "reviews": [
                _review_entry("Deserved Success", final_outcome=True, trade_id="a"),
                _review_entry("Poetic Justice", final_outcome=False, trade_id="b"),
            ],
            "aggregates": {"review_count": 2},
        }
        trades = _extract_trades(payload)
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0]["matrix_classification"], "Deserved Success")
        self.assertTrue(trades[0]["final_outcome"])

    def test_still_handles_legacy_trades_shape(self):
        payload = {"trades": [{"quadrant": "Good Failure", "is_win": False}]}
        trades = _extract_trades(payload)
        self.assertEqual(trades[0]["quadrant"], "Good Failure")

    def test_still_handles_bare_list(self):
        payload = [{"quadrant": "Dumb Luck", "is_win": True}]
        self.assertEqual(_extract_trades(payload)[0]["quadrant"], "Dumb Luck")

    def test_raises_on_unknown_shape(self):
        with self.assertRaisesRegex(ValueError, "reviews"):
            _extract_trades({"unexpected": "shape"})


class ResolveQuadrantTests(unittest.TestCase):
    def test_recognizes_matrix_classification(self):
        flat = _flatten_review_entry(_review_entry("Good Failure", final_outcome=False))
        self.assertEqual(_resolve_quadrant(flat), "Good Failure")

    def test_returns_none_for_unknown_label(self):
        self.assertIsNone(_resolve_quadrant({"matrix_classification": "Magical Realism"}))

    def test_legacy_quadrant_field_still_works(self):
        self.assertEqual(_resolve_quadrant({"quadrant": "deserved success"}), "Deserved Success")


class IsWinTests(unittest.TestCase):
    def test_reads_final_outcome_from_audit(self):
        flat = _flatten_review_entry(_review_entry("Deserved Success", final_outcome=True))
        self.assertTrue(_is_win(flat))

    def test_final_outcome_false_means_loss(self):
        flat = _flatten_review_entry(_review_entry("Poetic Justice", final_outcome=False))
        self.assertFalse(_is_win(flat))


class IsGoodProcessTests(unittest.TestCase):
    def test_inferred_from_quadrant(self):
        self.assertTrue(_is_good_process({}, "Deserved Success"))
        self.assertTrue(_is_good_process({}, "Good Failure"))
        self.assertFalse(_is_good_process({}, "Dumb Luck"))
        self.assertFalse(_is_good_process({}, "Poetic Justice"))


class SummarizeTradesTests(unittest.TestCase):
    def _summarize(self, payload):
        trades = _extract_trades(payload)
        buf = io.StringIO()
        with redirect_stdout(buf):
            summarize_trades(trades)
        return buf.getvalue()

    def test_end_to_end_against_performance_tracker_shape(self):
        payload = {
            "reviews": [
                _review_entry("Deserved Success", final_outcome=True, trade_id="a"),
                _review_entry("Good Failure", final_outcome=False, trade_id="b"),
                _review_entry("Dumb Luck", final_outcome=True, trade_id="c"),
                _review_entry("Poetic Justice", final_outcome=False, trade_id="d"),
            ],
            "aggregates": {"review_count": 4},
        }
        out = self._summarize(payload)

        self.assertIn("Total Trades: 4", out)
        # Two wins (Deserved Success + Dumb Luck) out of 4
        self.assertIn("Win Rate: 50.00%", out)
        # Two good-process (Deserved Success + Good Failure) out of 4
        self.assertIn("Process Integrity Score: 50.00%", out)
        self.assertIn("Deserved Success: 1", out)
        self.assertIn("Good Failure: 1", out)
        self.assertIn("Dumb Luck: 1", out)
        self.assertIn("Poetic Justice: 1", out)

    def test_emits_luck_warning_when_dumb_luck_dominates_wins(self):
        payload = {
            "reviews": [
                _review_entry("Dumb Luck", final_outcome=True, trade_id="a"),
                _review_entry("Dumb Luck", final_outcome=True, trade_id="b"),
                _review_entry("Deserved Success", final_outcome=True, trade_id="c"),
                _review_entry("Poetic Justice", final_outcome=False, trade_id="d"),
            ],
            "aggregates": {},
        }
        out = self._summarize(payload)
        self.assertIn("WARNING: SYSTEM IS RELYING ON LUCK", out)


if __name__ == "__main__":
    unittest.main()
