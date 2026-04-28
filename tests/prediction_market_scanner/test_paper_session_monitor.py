"""Tests for src/paper_session_monitor.py.

Pure-function tests for the parser, percentile helper, aggregator, and
end-to-end run() with a synthetic line stream. No supervisor or exchange
needed.
"""

from __future__ import annotations

import io
import math
import unittest
from datetime import datetime, timezone

from paper_session_monitor import (
    SymbolStats,
    aggregate,
    format_report,
    parse_tick_line,
    percentiles,
    run,
)


# A real supervisor log line, copy-paste from a smoke test.
_REAL_LINE = (
    "2026-04-27 19:31:08,072 INFO live_supervisor: tick #4 | ETH/USD | "
    "action=skipped_low_confidence | confidence=0.500 -- "
    "confidence 0.500 < floor 0.600"
)
_REAL_LINE_NO_NOTES = (
    "2026-04-27 19:31:09,123 INFO live_supervisor: tick #5 | BTC/USD | "
    "action=allowed | confidence=0.722"
)
_REAL_LINE_NA_CONF = (
    "2026-04-27 19:31:10,000 INFO live_supervisor: tick #6 | SOL/USD | "
    "action=errored | confidence=n/a -- get_ticker failed"
)


class ParseTests(unittest.TestCase):
    def test_parses_full_line_with_notes(self) -> None:
        row = parse_tick_line(_REAL_LINE)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.iteration, 4)
        self.assertEqual(row.symbol, "ETH/USD")
        self.assertEqual(row.action, "skipped_low_confidence")
        self.assertAlmostEqual(row.confidence or 0.0, 0.500, places=6)
        self.assertEqual(row.notes, "confidence 0.500 < floor 0.600")
        self.assertEqual(row.timestamp.tzinfo, timezone.utc)

    def test_parses_line_without_notes(self) -> None:
        row = parse_tick_line(_REAL_LINE_NO_NOTES)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.symbol, "BTC/USD")
        self.assertEqual(row.action, "allowed")
        self.assertAlmostEqual(row.confidence or 0.0, 0.722, places=6)
        self.assertIsNone(row.notes)

    def test_parses_na_confidence(self) -> None:
        row = parse_tick_line(_REAL_LINE_NA_CONF)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertIsNone(row.confidence)
        self.assertEqual(row.action, "errored")

    def test_returns_none_for_unrelated_lines(self) -> None:
        for line in (
            "",
            "random log line without tick",
            "2026-04-27 19:31:08,072 INFO live_supervisor: started up",
            "2026-04-27 19:31:08,072 ERROR something broken",
        ):
            self.assertIsNone(parse_tick_line(line), msg=f"line: {line!r}")

    def test_handles_trailing_newline(self) -> None:
        self.assertIsNotNone(parse_tick_line(_REAL_LINE + "\n"))

    def test_returns_none_for_bad_timestamp(self) -> None:
        bad = "BADTIME INFO live_supervisor: tick #1 | ETH/USD | action=allowed | confidence=0.6"
        self.assertIsNone(parse_tick_line(bad))


class PercentilesTests(unittest.TestCase):
    def test_empty_returns_nans(self) -> None:
        out = percentiles([], [50, 90])
        self.assertTrue(math.isnan(out[50]))
        self.assertTrue(math.isnan(out[90]))

    def test_known_distribution(self) -> None:
        # 10 values, nearest-rank: p50 -> idx 4 = 5, p90 -> idx 8 = 9.
        out = percentiles(list(range(1, 11)), [50, 90])
        self.assertEqual(out[50], 5)
        self.assertEqual(out[90], 9)

    def test_single_value(self) -> None:
        out = percentiles([0.42], [25, 50, 75])
        for v in out.values():
            self.assertEqual(v, 0.42)


class SymbolStatsTests(unittest.TestCase):
    def test_update_tracks_actions_and_confidence(self) -> None:
        s = SymbolStats(symbol="ETH/USD")
        rows = [
            parse_tick_line(_REAL_LINE),
            parse_tick_line(_REAL_LINE_NO_NOTES),
            parse_tick_line(_REAL_LINE_NA_CONF),
        ]
        for row in rows:
            assert row is not None
            # SymbolStats only sees its own symbol in real flow.
            if row.symbol == "ETH/USD":
                s.update(row)
        self.assertEqual(s.total_ticks, 1)
        self.assertEqual(s.action_counts["skipped_low_confidence"], 1)
        self.assertEqual(len(s.confidences), 1)
        self.assertAlmostEqual(s.confidences[0], 0.500, places=6)
        self.assertIsNone(s.last_allowed)

    def test_update_records_allowed_timestamp(self) -> None:
        s = SymbolStats(symbol="BTC/USD")
        row = parse_tick_line(_REAL_LINE_NO_NOTES)
        assert row is not None
        s.update(row)
        self.assertEqual(s.last_allowed, row.timestamp)


class AggregateTests(unittest.TestCase):
    def test_groups_by_symbol(self) -> None:
        rows = [
            parse_tick_line(_REAL_LINE),
            parse_tick_line(_REAL_LINE),
            parse_tick_line(_REAL_LINE_NO_NOTES),
            parse_tick_line(_REAL_LINE_NA_CONF),
        ]
        rows = [r for r in rows if r is not None]
        stats = aggregate(rows)
        self.assertEqual(set(stats.keys()), {"ETH/USD", "BTC/USD", "SOL/USD"})
        self.assertEqual(stats["ETH/USD"].total_ticks, 2)
        self.assertEqual(stats["BTC/USD"].total_ticks, 1)
        self.assertEqual(stats["SOL/USD"].total_ticks, 1)
        self.assertEqual(stats["SOL/USD"].action_counts["errored"], 1)


class FormatReportTests(unittest.TestCase):
    def test_empty_report(self) -> None:
        out = format_report({})
        self.assertIn("no tick data", out)

    def test_renders_per_symbol_block(self) -> None:
        rows = [parse_tick_line(_REAL_LINE), parse_tick_line(_REAL_LINE_NO_NOTES)]
        rows = [r for r in rows if r is not None]
        stats = aggregate(rows)
        report = format_report(
            stats,
            now=datetime(2026, 4, 27, 19, 32, 0, tzinfo=timezone.utc),
        )
        self.assertIn("[BTC/USD]", report)
        self.assertIn("[ETH/USD]", report)
        self.assertIn("allowed=1", report)
        self.assertIn("skipped_low_confidence=1", report)
        self.assertIn("p50=", report)


class RunIntegrationTests(unittest.TestCase):
    def test_run_one_shot_emits_report(self) -> None:
        lines = [_REAL_LINE, _REAL_LINE_NO_NOTES, _REAL_LINE_NA_CONF]
        out = io.StringIO()
        stats = run(
            source=iter(lines),
            refresh_seconds=30.0,
            out=out,
            follow=False,
        )
        rendered = out.getvalue()
        self.assertIn("paper-session report", rendered)
        self.assertIn("ETH/USD", rendered)
        self.assertEqual(stats["BTC/USD"].action_counts["allowed"], 1)

    def test_run_silent_when_no_tick_lines(self) -> None:
        out = io.StringIO()
        run(
            source=iter(["junk\n", "more junk\n"]),
            refresh_seconds=30.0,
            out=out,
            follow=False,
        )
        self.assertIn("no recognised tick lines", out.getvalue())


if __name__ == "__main__":
    unittest.main()
