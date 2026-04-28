"""Live read-only monitor for an in-flight supervisor paper session.

Reads `live_supervisor.py`'s per-tick LOGGER lines (the
``tick #N | SYMBOL | action=... | confidence=...`` format) from a file or
stdin and prints rolling per-symbol statistics: action distribution,
confidence percentiles, time-since-last-signal, etc.

This is read-only and parallel-safe -- it never touches the supervisor's
state, position store, or exchange. Run it alongside a session like:

    ./.venv/bin/python src/live_supervisor.py --symbols ETH/USD --mode paper \\
        | tee paper.log

and in a second terminal:

    ./.venv/bin/python src/paper_session_monitor.py paper.log --follow

Or pipe directly:

    tail -f paper.log | ./.venv/bin/python src/paper_session_monitor.py
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# Captures lines like:
#   2026-04-27 19:31:08,072 INFO live_supervisor: tick #4 | ETH/USD |
#   action=skipped_low_confidence | confidence=0.500 -- confidence 0.500 < floor 0.600
_TICK_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<level>INFO|WARNING|ERROR|DEBUG)\s+"
    r"(?P<logger>[\w.]+):\s+"
    r"tick\s+#(?P<iter>\d+)\s+\|\s+"
    r"(?P<symbol>[\w./-]+)\s+\|\s+"
    r"action=(?P<action>[\w_]+)\s+\|\s+"
    r"confidence=(?P<confidence>[\d.]+|n/a)"
    r"(?:\s+--\s+(?P<notes>.+))?$"
)


@dataclass
class TickRow:
    """One parsed supervisor tick log entry."""

    timestamp: datetime
    iteration: int
    symbol: str
    action: str
    confidence: Optional[float]
    notes: Optional[str]


@dataclass
class SymbolStats:
    """Accumulated per-symbol stats across a session."""

    symbol: str
    total_ticks: int = 0
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confidences: List[float] = field(default_factory=list)
    last_seen: Optional[datetime] = None
    last_allowed: Optional[datetime] = None
    last_force_flat: Optional[datetime] = None

    def update(self, tick: TickRow) -> None:
        self.total_ticks += 1
        self.action_counts[tick.action] += 1
        self.last_seen = tick.timestamp
        if tick.confidence is not None:
            self.confidences.append(tick.confidence)
        if tick.action == "allowed":
            self.last_allowed = tick.timestamp
        if tick.action == "force_flatted":
            self.last_force_flat = tick.timestamp


def parse_tick_line(line: str) -> Optional[TickRow]:
    """Parse a single supervisor log line. Returns None if it isn't a tick line."""
    m = _TICK_LINE.match(line.rstrip("\n"))
    if not m:
        return None
    raw_conf = m.group("confidence")
    confidence: Optional[float]
    if raw_conf == "n/a":
        confidence = None
    else:
        try:
            confidence = float(raw_conf)
        except ValueError:
            confidence = None
    try:
        # Python's logging default is "%Y-%m-%d %H:%M:%S,%f" (millis).
        ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None
    return TickRow(
        timestamp=ts.replace(tzinfo=timezone.utc),
        iteration=int(m.group("iter")),
        symbol=m.group("symbol"),
        action=m.group("action"),
        confidence=confidence,
        notes=m.group("notes"),
    )


def percentiles(values: List[float], pcts: Iterable[float]) -> Dict[float, float]:
    """Cheap nearest-rank percentile (avoids numpy dependency)."""
    if not values:
        return {p: float("nan") for p in pcts}
    s = sorted(values)
    n = len(s)
    out: Dict[float, float] = {}
    for p in pcts:
        # nearest-rank: index = ceil(p/100 * n) - 1, clamped.
        idx = max(0, min(n - 1, int(round((p / 100.0) * n)) - 1))
        out[p] = s[idx]
    return out


def aggregate(rows: Iterable[TickRow]) -> Dict[str, SymbolStats]:
    """Roll a stream of ticks into per-symbol stats."""
    stats: Dict[str, SymbolStats] = {}
    for row in rows:
        s = stats.get(row.symbol)
        if s is None:
            s = SymbolStats(symbol=row.symbol)
            stats[row.symbol] = s
        s.update(row)
    return stats


def format_report(
    stats: Dict[str, SymbolStats], *, now: Optional[datetime] = None
) -> str:
    """Render a human-readable rolling report."""
    if now is None:
        now = datetime.now(timezone.utc)
    if not stats:
        return "(no tick data parsed yet)"
    lines: List[str] = []
    lines.append(
        f"=== paper-session report @ {now.strftime('%Y-%m-%d %H:%M:%S UTC')} ==="
    )
    for sym in sorted(stats.keys()):
        s = stats[sym]
        pct = percentiles(s.confidences, [50, 75, 90])
        actions_sorted = sorted(s.action_counts.items(), key=lambda kv: -kv[1])
        actions_str = ", ".join(
            f"{a}={c} ({c / s.total_ticks * 100:.1f}%)" for a, c in actions_sorted
        )
        lines.append(f"\n[{sym}] ticks={s.total_ticks}")
        lines.append(f"  actions: {actions_str}")
        if s.confidences:
            lines.append(
                f"  confidence: n={len(s.confidences)} "
                f"p50={pct[50]:.3f} p75={pct[75]:.3f} p90={pct[90]:.3f} "
                f"max={max(s.confidences):.3f}"
            )
        else:
            lines.append("  confidence: (no numeric values)")
        if s.last_allowed:
            age = (now - s.last_allowed).total_seconds()
            lines.append(
                f"  last allowed: {s.last_allowed.strftime('%H:%M:%S')} "
                f"({_human_age(age)} ago)"
            )
        else:
            lines.append("  last allowed: never (no trade-eligible signal yet)")
        if s.last_force_flat:
            age = (now - s.last_force_flat).total_seconds()
            lines.append(
                f"  last force_flat: {s.last_force_flat.strftime('%H:%M:%S')} "
                f"({_human_age(age)} ago)"
            )
        if s.last_seen:
            age = (now - s.last_seen).total_seconds()
            lines.append(f"  last tick: {_human_age(age)} ago")
    return "\n".join(lines)


def _human_age(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# I/O drivers
# ---------------------------------------------------------------------------


def iter_lines_from_path(
    path: Path, *, follow: bool, poll_interval: float = 0.5
) -> Iterator[str]:
    """Yield lines from a file. If follow=True, tail-f indefinitely."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        # Drain everything currently in the file.
        for line in fh:
            yield line
        if not follow:
            return
        while True:
            line = fh.readline()
            if line:
                yield line
            else:
                time.sleep(poll_interval)


def iter_lines_from_stdin() -> Iterator[str]:
    for line in sys.stdin:
        yield line


def run(
    *,
    source: Iterator[str],
    refresh_seconds: float,
    out=sys.stdout,
    follow: bool,
    now_fn=lambda: datetime.now(timezone.utc),
) -> Dict[str, SymbolStats]:
    """Drive the parse + report loop. Returns the final stats dict."""
    stats: Dict[str, SymbolStats] = {}
    last_report = now_fn()
    parsed_any = False

    for raw in source:
        row = parse_tick_line(raw)
        if row is not None:
            parsed_any = True
            s = stats.get(row.symbol)
            if s is None:
                s = SymbolStats(symbol=row.symbol)
                stats[row.symbol] = s
            s.update(row)

        # Time-driven flush.
        now = now_fn()
        if follow and (now - last_report).total_seconds() >= refresh_seconds:
            out.write(format_report(stats, now=now) + "\n\n")
            out.flush()
            last_report = now

    if not follow and parsed_any:
        out.write(format_report(stats, now=now_fn()) + "\n")
        out.flush()
    elif not parsed_any and not follow:
        out.write("(no recognised tick lines found)\n")
        out.flush()
    return stats


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="paper_session_monitor",
        description=(
            "Read-only live monitor for a running supervisor paper session. "
            "Parses tick log lines and prints rolling per-symbol stats."
        ),
    )
    p.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to supervisor log (e.g. paper.log). Omit to read stdin.",
    )
    p.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Tail the log indefinitely, printing a report every --interval seconds.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Refresh interval in seconds when --follow is set (default: 30).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.log_path is not None:
        if not args.log_path.exists():
            print(f"error: log file not found: {args.log_path}", file=sys.stderr)
            return 2
        source = iter_lines_from_path(args.log_path, follow=args.follow)
    else:
        source = iter_lines_from_stdin()
    try:
        run(source=source, refresh_seconds=args.interval, follow=args.follow)
    except KeyboardInterrupt:
        print("\n(interrupted)", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
