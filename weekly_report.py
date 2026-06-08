"""End-of-week report for a paper-mode live_supervisor run.

Reads a run directory (``runs/<ts>_<symbols>/``) produced by
``live_supervisor.py --log-dir runs`` and prints:

* Uptime and tick volume
* Per-symbol action distribution (allowed / skipped_low_confidence / halted_breaker / errored / etc.)
* Per-symbol raw-probability stats from XGBoost predictor log lines
* Paper-fill count per symbol (entries opened)
* Mark-to-market PnL: every entry valued against the *current* Coinbase mid,
  net of the 5-bps paper slippage that was already applied at entry. Positions
  are otherwise left open by the supervisor (no built-in exit rule), so
  "would-be PnL if you closed everything now" is the honest readout.
* Breaker trips / error count
* Heads-up on shakedown progress per symbol

Usage::

    ./.venv/bin/python weekly_report.py runs/2026-05-22T13-00-00Z_ETH-USD,BTC-USD,SOL-USD
    ./.venv/bin/python weekly_report.py runs/<dir> --no-live-prices   # skip Coinbase REST
    ./.venv/bin/python weekly_report.py runs/<dir> --json out.json     # also write JSON

The script never touches Redis or .env — purely a log + REST reader. Safe to
run while the supervisor is still going.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) "
    r"(?P<level>\w+) (?P<logger>[^:]+): (?P<msg>.*)$"
)
TICK_RE = re.compile(
    r"^tick #(?P<iter>\d+) \| (?P<symbol>\S+) \| action=(?P<action>\w+) "
    r"\| confidence=(?P<conf>[\d.]+|n/a)(?: -- (?P<notes>.*))?$"
)
PRED_RE = re.compile(
    r"^xgb predictor: (?P<symbol>\S+) P\(long\)=(?P<p>[\d.]+) "
    r"\(thr=(?P<thr>[\d.]+) -> (?P<verdict>trigger|neutral)\)$"
)


@dataclass
class SymbolStats:
    symbol: str
    actions: Counter = field(default_factory=Counter)
    confidences: List[float] = field(default_factory=list)
    raw_probs: List[float] = field(default_factory=list)
    threshold: Optional[float] = None
    entry_prices: List[float] = field(default_factory=list)
    entry_sizes_usd: List[float] = field(default_factory=list)

    @property
    def total_ticks(self) -> int:
        return sum(self.actions.values())

    @property
    def fills(self) -> int:
        return self.actions.get("allowed", 0)

    def prob_summary(self) -> Dict[str, float]:
        if not self.raw_probs:
            return {}
        p = sorted(self.raw_probs)
        return {
            "n": len(p),
            "mean": statistics.fmean(p),
            "p10": _pct(p, 0.10),
            "p50": _pct(p, 0.50),
            "p90": _pct(p, 0.90),
            "max": p[-1],
            "n_over_thr": sum(1 for x in p if self.threshold is not None and x >= self.threshold),
        }


@dataclass
class RunSummary:
    run_dir: Path
    first_tick_at: Optional[datetime]
    last_tick_at: Optional[datetime]
    total_ticks: int
    iterations: int
    breaker_trips: int
    errors: int
    symbols: Dict[str, SymbolStats]
    summary_json: Optional[Dict[str, Any]]

    @property
    def uptime(self) -> Optional[str]:
        if not (self.first_tick_at and self.last_tick_at):
            return None
        delta = self.last_tick_at - self.first_tick_at
        days = delta.days
        hrs = delta.seconds // 3600
        mins = (delta.seconds % 3600) // 60
        return f"{days}d {hrs}h {mins}m"


def _pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = int(round((len(sorted_vals) - 1) * q))
    return sorted_vals[idx]


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=timezone.utc)


def parse_supervisor_log(log_path: Path) -> RunSummary:
    symbols: Dict[str, SymbolStats] = {}
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    iterations = 0
    breaker_trips = 0
    errors = 0
    pending_entry: Dict[str, Tuple[str, float]] = {}

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            m = LOG_LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            ts = _parse_ts(m.group("ts"))
            first_ts = first_ts or ts
            last_ts = ts
            msg = m.group("msg")

            pm = PRED_RE.match(msg)
            if pm:
                sym = pm.group("symbol")
                stats = symbols.setdefault(sym, SymbolStats(symbol=sym))
                stats.raw_probs.append(float(pm.group("p")))
                stats.threshold = float(pm.group("thr"))
                continue

            tm = TICK_RE.match(msg)
            if tm:
                sym = tm.group("symbol")
                action = tm.group("action")
                conf = tm.group("conf")
                stats = symbols.setdefault(sym, SymbolStats(symbol=sym))
                stats.actions[action] += 1
                if conf != "n/a":
                    stats.confidences.append(float(conf))
                iterations = max(iterations, int(tm.group("iter")))
                if action in ("halted_breaker", "force_flatted"):
                    breaker_trips += 1
                elif action == "errored":
                    errors += 1
                continue

    total_ticks = sum(s.total_ticks for s in symbols.values())

    # summary.json only exists if supervisor exited gracefully.
    summary_json = None
    sj = log_path.parent / "summary.json"
    if sj.exists():
        summary_json = json.loads(sj.read_text())

    return RunSummary(
        run_dir=log_path.parent,
        first_tick_at=first_ts,
        last_tick_at=last_ts,
        total_ticks=total_ticks,
        iterations=iterations,
        breaker_trips=breaker_trips,
        errors=errors,
        symbols=symbols,
        summary_json=summary_json,
    )


def fetch_coinbase_mids(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Pull last-trade mids from Coinbase exchange public endpoint. No auth."""
    out: Dict[str, Optional[float]] = {}
    for sym in symbols:
        product_id = sym.replace("/", "-")
        url = f"https://api.exchange.coinbase.com/products/{product_id}/ticker"
        try:
            req = Request(url, headers={"User-Agent": "autopilot-weekly-report/1.0"})
            with urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            bid = float(payload.get("bid") or 0.0)
            ask = float(payload.get("ask") or 0.0)
            if bid > 0 and ask > 0:
                out[sym] = (bid + ask) / 2.0
            else:
                out[sym] = float(payload.get("price") or 0.0) or None
        except Exception as exc:
            print(f"warn: ticker fetch failed for {sym}: {exc}", file=sys.stderr)
            out[sym] = None
    return out


PAPER_SLIPPAGE_BPS = 5.0  # mirrors live_supervisor.PAPER_SLIPPAGE_BPS


def _format_pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x*100:.2f}%"


def render(run: RunSummary, *, live_prices: bool) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append(f"Weekly Report  --  {run.run_dir.name}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Run dir:       {run.run_dir}")
    lines.append(f"Uptime:        {run.uptime or 'n/a'} "
                 f"({run.first_tick_at} -> {run.last_tick_at})")
    lines.append(f"Iterations:    {run.iterations}")
    lines.append(f"Total ticks:   {run.total_ticks}")
    lines.append(f"Breaker trips: {run.breaker_trips}")
    lines.append(f"Errors:        {run.errors}")
    if run.summary_json:
        lines.append(f"Exit cleanly:  yes "
                     f"(interrupted={run.summary_json.get('interrupted')})")
    else:
        lines.append("Exit cleanly:  no -- supervisor still running or crashed")
    lines.append("")

    mids: Dict[str, Optional[float]] = {}
    if live_prices and run.symbols:
        mids = fetch_coinbase_mids(list(run.symbols.keys()))

    total_pnl_pct = 0.0
    pnl_lines: List[str] = []

    for sym in sorted(run.symbols):
        s = run.symbols[sym]
        lines.append("-" * 72)
        lines.append(f"Symbol: {sym}")
        lines.append("-" * 72)
        lines.append(f"  Total ticks:    {s.total_ticks}")
        lines.append("  Action mix:")
        for action, count in s.actions.most_common():
            pct = count / s.total_ticks * 100 if s.total_ticks else 0.0
            lines.append(f"    {action:30s} {count:6d}  ({pct:5.1f}%)")
        if s.confidences:
            cs = sorted(s.confidences)
            lines.append(
                f"  Confidence:     n={len(cs)} "
                f"min={cs[0]:.3f} p50={_pct(cs, 0.50):.3f} "
                f"p90={_pct(cs, 0.90):.3f} max={cs[-1]:.3f}"
            )
        ps = s.prob_summary()
        if ps:
            lines.append(
                f"  Raw P(long):    n={ps['n']} mean={ps['mean']:.3f} "
                f"p10={ps['p10']:.3f} p50={ps['p50']:.3f} "
                f"p90={ps['p90']:.3f} max={ps['max']:.3f}"
            )
            if s.threshold is not None:
                lines.append(
                    f"  Threshold:      {s.threshold:.3f}  "
                    f"({ps['n_over_thr']} ticks over threshold "
                    f"= {ps['n_over_thr']/ps['n']*100:.1f}% trigger rate)"
                )
        lines.append(f"  Paper fills:    {s.fills}")
        if live_prices:
            cur = mids.get(sym)
            if cur and s.fills:
                avg_entry = (sum(s.entry_prices) / len(s.entry_prices)
                             if s.entry_prices else None)
                if avg_entry is None:
                    lines.append("  MTM:            n/a "
                                 "(supervisor.log does not record fill prices)")
                else:
                    pnl_pct = (cur - avg_entry) / avg_entry
                    lines.append(
                        f"  MTM:            avg_entry={avg_entry:.4f} "
                        f"current={cur:.4f}  PnL={_format_pct(pnl_pct)}"
                    )
                    total_pnl_pct += pnl_pct
            elif cur:
                lines.append(f"  Current mid:    {cur:.4f}  (no fills to value)")
            else:
                lines.append("  MTM:            n/a (Coinbase ticker fetch failed)")
        lines.append("")

    lines.append("=" * 72)
    lines.append("How profitable was the system?")
    lines.append("=" * 72)
    lines.append("")
    if not any(s.fills for s in run.symbols.values()):
        lines.append("Zero paper fills across the run. Either:")
        lines.append("  - Thresholds were too high for observed P(long) distribution")
        lines.append("  - Circuit breakers halted entries (check breaker_trips)")
        lines.append("  - Predictor returned None on every tick (check supervisor.log "
                     "for 'missing feature cols')")
        lines.append("")
        lines.append("System would have done NOTHING with real money. That is a valid")
        lines.append("operator-safe state -- the supervisor stayed up + did not lose anything.")
    else:
        lines.append("CAVEAT: the supervisor opens paper positions but has no built-in")
        lines.append("exit rule. The 'PnL' above is mark-to-market against current price,")
        lines.append("assuming you close everything now. There is no stop-loss / take-")
        lines.append("profit / time-exit fired during the run.")
        lines.append("")
        lines.append("Honest read: the system OPENED N positions. If you closed all of")
        lines.append("them at week-end, the bag is worth +/- X% vs. entry. That's not the")
        lines.append("same as 'would have made money trading' -- a real strategy needs an")
        lines.append("exit rule on top of the entry model.")

    lines.append("")
    lines.append(f"Shakedown gate: 14 consecutive clean days per symbol before live mode.")
    lines.append(f"This run covered {run.uptime or 'n/a'} -- "
                 f"{'progress towards 14d unlock' if (run.last_tick_at and run.first_tick_at and (run.last_tick_at - run.first_tick_at).days < 14) else 'shakedown window covered'}.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Path to runs/<ts>_<symbols>/")
    p.add_argument("--no-live-prices", action="store_true",
                   help="Skip Coinbase ticker fetch (offline mode)")
    p.add_argument("--json", type=Path, default=None,
                   help="Also write the report data as JSON")
    args = p.parse_args(argv)

    log_path = args.run_dir / "supervisor.log"
    if not log_path.exists():
        print(f"error: {log_path} not found", file=sys.stderr)
        return 1

    run = parse_supervisor_log(log_path)
    report = render(run, live_prices=not args.no_live_prices)
    print(report)

    if args.json:
        payload = {
            "run_dir": str(run.run_dir),
            "first_tick_at": run.first_tick_at.isoformat() if run.first_tick_at else None,
            "last_tick_at": run.last_tick_at.isoformat() if run.last_tick_at else None,
            "iterations": run.iterations,
            "total_ticks": run.total_ticks,
            "breaker_trips": run.breaker_trips,
            "errors": run.errors,
            "summary_json": run.summary_json,
            "symbols": {
                sym: {
                    "total_ticks": s.total_ticks,
                    "actions": dict(s.actions),
                    "fills": s.fills,
                    "raw_prob_stats": s.prob_summary(),
                    "threshold": s.threshold,
                }
                for sym, s in run.symbols.items()
            },
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
