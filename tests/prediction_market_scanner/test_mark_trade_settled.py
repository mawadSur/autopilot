import argparse
import json
import tempfile
import unittest
from pathlib import Path

from mark_trade_settled import _parse_outcome, main, mark_settled


def _write_open_trade(path: Path, *, trade_id: str = "mkt-test") -> None:
    payload = {
        "event_id": trade_id,
        "trade_id": trade_id,
        "status": "open",
        "created_at_utc": "2026-04-25T00:00:00+00:00",
        "settled_at": None,
        "final_outcome": None,
        "post_settlement_news": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class ParseOutcomeTests(unittest.TestCase):
    def test_truthy_aliases(self):
        for value in ("win", "WIN", "won", "true", "yes", "1"):
            self.assertTrue(_parse_outcome(value), msg=value)

    def test_falsy_aliases(self):
        for value in ("loss", "lost", "FALSE", "no", "0"):
            self.assertFalse(_parse_outcome(value), msg=value)

    def test_invalid_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_outcome("maybe")


class MarkSettledTests(unittest.TestCase):
    def test_in_place_mutation_sets_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_a.json"
            _write_open_trade(log_path, trade_id="a")

            payload = mark_settled(
                log_path,
                final_outcome=True,
                post_settlement_news="AP called it.",
            )

            self.assertEqual(payload["status"], "settled")
            self.assertTrue(payload["final_outcome"])
            self.assertEqual(payload["post_settlement_news"], "AP called it.")
            self.assertIsNotNone(payload["settled_at"])

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["status"], "settled")
            self.assertTrue(on_disk["final_outcome"])

    def test_settled_at_override_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_b.json"
            _write_open_trade(log_path, trade_id="b")

            payload = mark_settled(
                log_path,
                final_outcome=False,
                settled_at="2026-04-25T12:34:56+00:00",
            )
            self.assertEqual(payload["settled_at"], "2026-04-25T12:34:56+00:00")

    def test_news_omitted_leaves_existing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_c.json"
            _write_open_trade(log_path, trade_id="c")

            payload = mark_settled(log_path, final_outcome=True)
            self.assertIsNone(payload["post_settlement_news"])

    def test_rejects_non_object_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bad.json"
            log_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                mark_settled(log_path, final_outcome=True)


class MainCLITests(unittest.TestCase):
    def test_returns_zero_and_mutates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_d.json"
            _write_open_trade(log_path, trade_id="d")

            exit_code = main(
                [
                    str(log_path),
                    "--outcome",
                    "win",
                    "--news",
                    "Confirmed.",
                ]
            )
            self.assertEqual(exit_code, 0)

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["status"], "settled")
            self.assertTrue(on_disk["final_outcome"])

    def test_returns_two_when_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.json"
            exit_code = main([str(missing), "--outcome", "loss"])
            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
