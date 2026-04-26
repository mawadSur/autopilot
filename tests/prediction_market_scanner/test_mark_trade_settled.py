import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from typing import Optional

from mark_trade_settled import _parse_outcome, main, mark_settled


def _write_open_trade(
    path: Path,
    *,
    trade_id: str = "mkt-test",
    entry_price: Optional[float] = None,
    position_size_usd: Optional[float] = None,
) -> None:
    payload = {
        "event_id": trade_id,
        "trade_id": trade_id,
        "status": "open",
        "created_at_utc": "2026-04-25T00:00:00+00:00",
        "settled_at": None,
        "final_outcome": None,
        "market_outcome": None,
        "post_settlement_news": None,
    }
    if entry_price is not None:
        payload["entry_price"] = entry_price
        payload["exit_price"] = None
        payload["realized_pnl_usd"] = None
    if position_size_usd is not None:
        payload["position_size_usd"] = position_size_usd
        payload["max_loss_usd"] = position_size_usd
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


class PnlTrackingTests(unittest.TestCase):
    def test_settle_with_exit_price_records_pnl(self):
        # Trader closed early at 0.85 (entry 0.50, $200 size). Long-YES PnL:
        #   200 * (0.85 - 0.50) / 0.50 = 200 * 0.70 = 140.00
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_pnl_a.json"
            _write_open_trade(
                log_path,
                trade_id="pnl-a",
                entry_price=0.50,
                position_size_usd=200.0,
            )

            payload = mark_settled(
                log_path,
                final_outcome=True,
                exit_price=0.85,
            )

            self.assertEqual(payload["exit_price"], 0.85)
            self.assertAlmostEqual(payload["realized_pnl_usd"], 140.0, places=6)

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["exit_price"], 0.85)
            self.assertAlmostEqual(on_disk["realized_pnl_usd"], 140.0, places=6)

    def test_settle_winning_trade_without_exit_price_uses_one_dollar_settlement(self):
        # Win, no --exit-price. Binary YES settles at $1.00.
        # PnL = 250 * (1.0 - 0.40) / 0.40 = 250 * 1.5 = 375.00.
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_pnl_b.json"
            _write_open_trade(
                log_path,
                trade_id="pnl-b",
                entry_price=0.40,
                position_size_usd=250.0,
            )

            payload = mark_settled(log_path, final_outcome=True)

            self.assertEqual(payload["exit_price"], 1.0)
            self.assertAlmostEqual(payload["realized_pnl_usd"], 375.0, places=6)

    def test_settle_losing_trade_without_exit_price_records_full_loss(self):
        # Loss, no --exit-price. Binary YES settles at $0.00; full notional gone.
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_pnl_c.json"
            _write_open_trade(
                log_path,
                trade_id="pnl-c",
                entry_price=0.55,
                position_size_usd=180.0,
            )

            payload = mark_settled(log_path, final_outcome=False)

            self.assertEqual(payload["exit_price"], 0.0)
            self.assertAlmostEqual(payload["realized_pnl_usd"], -180.0, places=6)

    def test_settle_legacy_log_without_entry_price_warns_and_skips_pnl(self):
        # Legacy log: no entry_price/position_size_usd recorded at decision time.
        # Settlement must succeed but skip PnL derivation with a stderr warning.
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_legacy.json"
            _write_open_trade(log_path, trade_id="legacy")

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                payload = mark_settled(log_path, final_outcome=True)

            self.assertIsNone(payload.get("realized_pnl_usd"))
            self.assertIsNone(payload.get("exit_price"))
            stderr_value = stderr_buf.getvalue()
            self.assertIn("entry_price", stderr_value)
            self.assertIn("legacy", stderr_value.lower())


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

    def test_explicit_market_outcome_yes(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_e.json"
            _write_open_trade(log_path, trade_id="e")

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                exit_code = main(
                    [
                        str(log_path),
                        "--outcome",
                        "win",
                        "--market-outcome",
                        "yes",
                    ]
                )
            self.assertEqual(exit_code, 0)

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["final_outcome"])
            self.assertTrue(on_disk["market_outcome"])
            # Explicit pass: must NOT emit the implicit-mapping warning.
            self.assertNotIn("market_outcome defaulted", stderr_buf.getvalue())

    def test_explicit_market_outcome_no(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_f.json"
            _write_open_trade(log_path, trade_id="f")

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                exit_code = main(
                    [
                        str(log_path),
                        "--outcome",
                        "loss",
                        "--market-outcome",
                        "no",
                    ]
                )
            self.assertEqual(exit_code, 0)

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertFalse(on_disk["final_outcome"])
            self.assertFalse(on_disk["market_outcome"])
            self.assertNotIn("market_outcome defaulted", stderr_buf.getvalue())

    def test_omitted_market_outcome_defaults_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_g.json"
            _write_open_trade(log_path, trade_id="g")

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                exit_code = main([str(log_path), "--outcome", "win"])
            self.assertEqual(exit_code, 0)

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["final_outcome"])
            # Implicit mapping: market_outcome should mirror final_outcome.
            self.assertTrue(on_disk["market_outcome"])

            stderr_value = stderr_buf.getvalue()
            self.assertIn("market_outcome defaulted", stderr_value)
            self.assertIn("always-long-YES", stderr_value)

    def test_explicit_override_can_diverge_from_final_outcome(self):
        # Trade win on a NO position: --outcome win + --market-outcome no
        # → final_outcome=True, market_outcome=False.
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "trade_execution_h.json"
            _write_open_trade(log_path, trade_id="h")

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                exit_code = main(
                    [
                        str(log_path),
                        "--outcome",
                        "win",
                        "--market-outcome",
                        "no",
                    ]
                )
            self.assertEqual(exit_code, 0)

            on_disk = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["final_outcome"])
            self.assertFalse(on_disk["market_outcome"])
            self.assertNotIn("market_outcome defaulted", stderr_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
