"""Tests for ``calibration_agent.shadow_capture``."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence
from unittest.mock import patch

from calibration_agent.ml_service import FEATURE_COLUMNS
from calibration_agent.shadow_capture import (
    capture_loop,
    capture_once,
)
from models import Market


def _make_market(
    market_id: str,
    *,
    title: str = "Test Market",
    category: str = "Politics",
    implied_prob: float = 0.42,
    volume_24h: float = 12_345.6,
) -> Market:
    return Market(
        market_id=market_id,
        title=title,
        category=category,
        implied_prob=implied_prob,
        bid_price=max(0.0, implied_prob - 0.01),
        ask_price=min(1.0, implied_prob + 0.01),
        volume_24h=volume_24h,
        price_history={"1h": 0.005, "6h": 0.01, "24h": 0.02},
        open_interest=50_000.0,
        resolution_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
        rules_text="Resolves Yes if the test market resolves Yes.",
    )


def _make_fake_fetcher(markets: Sequence[Market]) -> Callable[..., List[Market]]:
    """Return a stub fetcher that ignores kwargs and yields ``markets``."""

    def _fetcher(**_kwargs: Any) -> List[Market]:
        return list(markets)

    return _fetcher


def _capture_stderr(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


class CaptureOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.markets = [
            _make_market("mkt-1", title="Market Alpha", category="Politics"),
            _make_market("mkt-2", title="Market Beta", category="Sports"),
            _make_market("mkt-3", title="Market Gamma", category="Crypto"),
        ]
        self.now = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)

    def _run_capture(self, **overrides: Any) -> Dict[str, Any]:
        kwargs = dict(
            fetcher=_make_fake_fetcher(self.markets),
            now=self.now,
        )
        kwargs.update(overrides)
        result, _ = _capture_stderr(capture_once, self.tmp_path, **kwargs)
        return result

    def _list_logs(self) -> List[Path]:
        return sorted(self.tmp_path.glob("shadow_*.json"))

    def test_capture_once_writes_one_log_per_market(self) -> None:
        self._run_capture()

        logs = self._list_logs()
        self.assertEqual(len(logs), 3)
        # File naming pattern: shadow_<UTC-timestamp>_<market_id>.json.
        pattern = re.compile(r"^shadow_\d{8}T\d{6}Z_mkt-\d+\.json$")
        for path in logs:
            self.assertRegex(path.name, pattern, msg=f"Bad filename: {path.name}")

        market_ids = sorted({"mkt-1", "mkt-2", "mkt-3"})
        ids_from_files = sorted(p.name.split("_")[-1].replace(".json", "") for p in logs)
        self.assertEqual(ids_from_files, market_ids)

        # Schema sanity for the first written log.
        payload = json.loads(logs[0].read_text(encoding="utf-8"))
        for required in (
            "event_id",
            "trade_id",
            "status",
            "created_at_utc",
            "settled_at",
            "final_outcome",
            "market_outcome",
            "post_settlement_news",
            "scanner",
            "features_window",
            "model_meta",
            "research",
            "calibration",
            "risk",
            "source",
            "notes",
        ):
            self.assertIn(required, payload, msg=f"Missing key {required}")
        self.assertEqual(payload["event_id"], payload["trade_id"])
        self.assertIsNone(payload["model_meta"])
        self.assertIsNone(payload["research"])
        self.assertIsNone(payload["calibration"])
        self.assertIsNone(payload["risk"])
        self.assertIsNone(payload["post_settlement_news"])
        self.assertIsNone(payload["settled_at"])
        self.assertIsNone(payload["final_outcome"])
        self.assertIsNone(payload["market_outcome"])
        # Scanner sub-payload covers the documented fields.
        scanner = payload["scanner"]
        for key in ("market_id", "title", "category", "implied_prob", "volume_24h"):
            self.assertIn(key, scanner, msg=f"Missing scanner key {key}")

    def test_capture_once_marks_source_shadow_and_status_open(self) -> None:
        self._run_capture()
        for path in self._list_logs():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "shadow")
            self.assertEqual(payload["status"], "open")
            self.assertIsNone(payload["notes"])

    def test_capture_once_features_window_has_8_market_columns_and_captured_at_utc(self) -> None:
        self._run_capture()
        for path in self._list_logs():
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload["features_window"]
            self.assertIsInstance(features, dict)
            # Exactly the 8 market columns + captured_at_utc; NO research keys.
            expected_keys = set(FEATURE_COLUMNS) | {"captured_at_utc"}
            self.assertEqual(set(features.keys()), expected_keys)
            for col in FEATURE_COLUMNS:
                self.assertIsInstance(features[col], (int, float))
            # captured_at_utc is parseable as an ISO timestamp.
            datetime.fromisoformat(features["captured_at_utc"])

    def test_capture_once_filename_includes_timestamp(self) -> None:
        # Two captures of the same market id at different times must produce
        # two distinct files because the filename embeds the UTC timestamp.
        first = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)
        second = datetime(2026, 4, 25, 12, 0, 30, tzinfo=timezone.utc)
        single_market = [self.markets[0]]
        fetcher = _make_fake_fetcher(single_market)

        _capture_stderr(
            capture_once, self.tmp_path, fetcher=fetcher, now=first
        )
        _capture_stderr(
            capture_once, self.tmp_path, fetcher=fetcher, now=second
        )

        logs = self._list_logs()
        self.assertEqual(len(logs), 2, msg=f"got {[p.name for p in logs]}")
        names = {p.name for p in logs}
        self.assertIn("shadow_20260425T120000Z_mkt-1.json", names)
        self.assertIn("shadow_20260425T120030Z_mkt-1.json", names)

    def test_capture_once_summary_counts_match(self) -> None:
        summary, stderr = _capture_stderr(
            capture_once,
            self.tmp_path,
            fetcher=_make_fake_fetcher(self.markets),
            now=self.now,
        )
        self.assertEqual(summary["markets_captured"], 3)
        self.assertEqual(summary["output_dir"], str(self.tmp_path))
        # captured_at_utc is the iso form of self.now.
        self.assertEqual(summary["captured_at_utc"], self.now.isoformat())
        # Stderr summary echoes the count.
        self.assertIn("Captured 3 markets", stderr)

    def test_capture_once_empty_fetcher_writes_nothing(self) -> None:
        summary, stderr = _capture_stderr(
            capture_once,
            self.tmp_path,
            fetcher=_make_fake_fetcher([]),
            now=self.now,
        )
        self.assertEqual(summary["markets_captured"], 0)
        self.assertEqual(self._list_logs(), [])
        self.assertIn("Captured 0 markets", stderr)

    def test_capture_once_creates_output_dir_if_missing(self) -> None:
        nested = self.tmp_path / "nested" / "shadow"
        self.assertFalse(nested.exists())
        summary, _ = _capture_stderr(
            capture_once,
            nested,
            fetcher=_make_fake_fetcher(self.markets[:1]),
            now=self.now,
        )
        self.assertTrue(nested.is_dir())
        self.assertEqual(summary["markets_captured"], 1)


class CaptureLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def test_capture_loop_handles_keyboard_interrupt_cleanly(self) -> None:
        calls: List[int] = []

        def _fake_capture(_output_dir: Any, **_kwargs: Any) -> Dict[str, Any]:
            calls.append(1)
            if len(calls) == 1:
                return {
                    "markets_captured": 0,
                    "output_dir": str(self.tmp_path),
                    "captured_at_utc": "2026-04-25T12:00:00+00:00",
                }
            raise KeyboardInterrupt()

        sleep_calls: List[float] = []

        def _no_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        summary, stderr = _capture_stderr(
            capture_loop,
            self.tmp_path,
            interval_seconds=0.01,
            sleep_fn=_no_sleep,
            capture_fn=_fake_capture,
        )

        self.assertTrue(summary["interrupted"])
        # We did one successful iteration before the KeyboardInterrupt fired
        # on the second call, so iterations should be 1.
        self.assertEqual(summary["iterations"], 1)
        self.assertEqual(summary["errors"], 0)
        self.assertIsNotNone(summary["last_summary"])
        # No traceback text in stderr; loop emits a polite final line.
        self.assertIn("Interrupted by user", stderr)
        self.assertIn("Loop finished", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_capture_loop_logs_and_continues_on_per_iteration_exception(self) -> None:
        attempts: List[int] = []

        def _flaky_capture(_output_dir: Any, **_kwargs: Any) -> Dict[str, Any]:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("boom")
            return {
                "markets_captured": 2,
                "output_dir": str(self.tmp_path),
                "captured_at_utc": "2026-04-25T12:00:00+00:00",
            }

        sleep_calls: List[float] = []

        def _no_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        # Use assertLogs to verify the ERROR log fires on the failing iteration.
        with self.assertLogs("calibration_agent.shadow_capture", level="ERROR") as captured_logs:
            summary, stderr = _capture_stderr(
                capture_loop,
                self.tmp_path,
                interval_seconds=0.01,
                max_iterations=2,
                sleep_fn=_no_sleep,
                capture_fn=_flaky_capture,
            )

        # Both iterations ran; first errored, second succeeded.
        self.assertEqual(summary["iterations"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertFalse(summary["interrupted"])
        # last_summary is from the successful second iteration.
        self.assertIsNotNone(summary["last_summary"])
        self.assertEqual(summary["last_summary"]["markets_captured"], 2)
        # Error log contains "boom".
        joined_logs = "\n".join(captured_logs.output)
        self.assertIn("boom", joined_logs)
        self.assertIn("ERROR", joined_logs)
        # Loop sleeps once between iteration 1 (failed) and iteration 2.
        self.assertEqual(sleep_calls, [0.01])
        # Final stderr summary reports the error count.
        self.assertIn("errors=1", stderr)

    def test_capture_loop_respects_max_iterations(self) -> None:
        attempts: List[int] = []

        def _capture(_output_dir: Any, **_kwargs: Any) -> Dict[str, Any]:
            attempts.append(1)
            return {
                "markets_captured": 1,
                "output_dir": str(self.tmp_path),
                "captured_at_utc": "2026-04-25T12:00:00+00:00",
            }

        sleep_calls: List[float] = []

        def _no_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        summary, _ = _capture_stderr(
            capture_loop,
            self.tmp_path,
            interval_seconds=5.0,
            max_iterations=3,
            sleep_fn=_no_sleep,
            capture_fn=_capture,
        )
        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(summary["errors"], 0)
        self.assertFalse(summary["interrupted"])
        # We sleep between iterations but NOT after the final one (avoiding a
        # spurious 5s delay before the loop exits).
        self.assertEqual(len(sleep_calls), 2)


class CaptureOnceLazyImportTests(unittest.TestCase):
    """Module-level smoke test: importing must not pull in fetcher at import time."""

    def test_default_fetcher_is_imported_lazily(self) -> None:
        # If we patch fetcher.fetch_active_markets and then call capture_once
        # without passing fetcher=, the patched callable must be used. This
        # confirms the import is lazy (otherwise patching after import would
        # have no effect for already-bound references).
        with tempfile.TemporaryDirectory() as tmp:
            recorded: List[Dict[str, Any]] = []

            def _fake_real_fetch(**kwargs: Any) -> List[Market]:
                recorded.append(kwargs)
                return []

            with patch("fetcher.fetch_active_markets", _fake_real_fetch):
                _capture_stderr(
                    capture_once,
                    tmp,
                    min_volume_24h=1234.0,
                    page_size=7,
                    max_pages=2,
                    now=datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc),
                )

            self.assertEqual(len(recorded), 1)
            self.assertEqual(recorded[0]["min_volume_24h"], 1234.0)
            self.assertEqual(recorded[0]["page_size"], 7)
            self.assertEqual(recorded[0]["max_pages"], 2)


if __name__ == "__main__":
    # Silence noisy ERROR logs from the loop tests during direct invocation.
    logging.getLogger("calibration_agent.shadow_capture").setLevel(logging.CRITICAL)
    unittest.main()
