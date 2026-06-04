"""Funding-carry SHADOW accrual runner (SHADOW / NO ORDERS).

Forward-validates the crypto perp funding-carry edge: it picks the best
net-of-cost carries (via :mod:`funding_carry_scanner`), "opens" delta-neutral
shadow positions, and on each scan ACCRUES the funding that position would have
earned since the last scan — building a real, time-stamped track record of
realized carry net of the one-time entry cost. NO orders, no signing, no wallet;
it only READS funding rates and does bookkeeping over an append-only JSONL.

Restart-safe: state is folded from the ledger each scan, so a crash/restart
resumes the same positions and cumulative carry.

Accrual model: sampling the current funding rate ``r`` (per period) for a held
position, the funding P/L over ``periods_elapsed`` periods is
``side_sign * r * notional * periods_elapsed`` — ``side_sign = +1`` when SHORT
the perp (entry funding > 0, receive +r) and ``-1`` when LONG (receive -r). This
samples funding at scan cadence; a venue settles it each funding period, so a
scan interval near the funding period (1h on Hyperliquid) tracks it closely.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

try:  # Flat import under PYTHONPATH=src.
    from funding_carry_scanner import (
        normalize_funding_rates, scan_carry, amortized_round_trip_annual,
        DEFAULT_PERIOD_HOURS, DEFAULT_ROUND_TRIP_BPS, DEFAULT_HOLD_DAYS,
        DEFAULT_BASIS_BUFFER_ANNUAL,
    )
except Exception:  # pragma: no cover
    from src.funding_carry_scanner import (  # type: ignore
        normalize_funding_rates, scan_carry, amortized_round_trip_annual,
        DEFAULT_PERIOD_HOURS, DEFAULT_ROUND_TRIP_BPS, DEFAULT_HOLD_DAYS,
        DEFAULT_BASIS_BUFFER_ANNUAL,
    )

__all__ = ["accrue_delta", "fold_ledger", "DEFAULT_LEDGER"]

DEFAULT_LEDGER = "runs/funding_carry_ledger.jsonl"
_EVENT_OPEN = "open"
_EVENT_ACCRUE = "accrue"


def accrue_delta(side_sign: float, funding_rate: float, notional: float,
                 periods_elapsed: float) -> float:
    """Funding P/L (USD) accrued for a held carry over ``periods_elapsed`` periods.

    ``side_sign`` is +1 (short perp) / -1 (long perp); a short receives ``+r`` per
    period, a long receives ``-r``. Returns ``side_sign * r * notional * periods``.
    """
    return float(side_sign) * float(funding_rate) * float(notional) * float(periods_elapsed)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def fold_ledger(path: str) -> Dict[str, Dict[str, Any]]:
    """Replay the JSONL into per-symbol carry state.

    Returns ``{symbol: {side, side_sign, notional, entry_funding_annual,
    rt_cost_usd, entry_ts, last_ts_ms, realized_usd, n_accruals}}``. ``open``
    sets the position; ``accrue`` adds realized + advances ``last_ts_ms``.
    """
    state: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return state
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = e.get("symbol")
            if not sym:
                continue
            if e.get("_event") == _EVENT_OPEN:
                state[sym] = {
                    "side": e.get("side"),
                    "side_sign": float(e.get("side_sign", 1.0)),
                    "notional": float(e.get("notional", 0.0)),
                    "entry_funding_annual": float(e.get("entry_funding_annual", 0.0)),
                    "rt_cost_usd": float(e.get("rt_cost_usd", 0.0)),
                    "entry_ts": e.get("ts"),
                    "last_ts_ms": float(e.get("ts_ms", 0.0)),
                    "realized_usd": 0.0,
                    "n_accruals": 0,
                }
            elif e.get("_event") == _EVENT_ACCRUE:
                pos = state.get(sym)
                if pos is None:
                    continue
                pos["realized_usd"] += float(e.get("realized_delta", 0.0))
                pos["last_ts_ms"] = float(e.get("ts_ms", pos["last_ts_ms"]))
                pos["n_accruals"] += 1
    return state


def _net_usd(pos: Dict[str, Any]) -> float:
    """Realized carry minus the one-time entry round-trip cost."""
    return pos["realized_usd"] - pos["rt_cost_usd"]


def run_scan(
    path: str,
    funding_rows: List[Dict[str, Any]],
    *,
    now_ms: float,
    period_hours: float,
    notional: float,
    top_k: int,
    min_net_annual: float,
    round_trip_bps: float,
    basis_buffer_annual: float,
    hold_days: float,
) -> Dict[str, Any]:
    """One scan: open the basket on first run, else accrue each held position.

    Pure w.r.t. I/O except the append-only ledger writes (so it is exercised in
    tests with a temp ledger + injected ``funding_rows``). Returns a summary dict.
    """
    state = fold_ledger(path)
    by_symbol = {r["symbol"]: r for r in funding_rows}

    if not state:
        ranked = scan_carry(
            funding_rows, round_trip_bps=round_trip_bps, hold_days=hold_days,
            basis_buffer_annual=basis_buffer_annual, min_net_annual=min_net_annual,
        )[:top_k]
        rt_cost_usd = (round_trip_bps / 10_000.0) * notional
        for c in ranked:
            _write(path, {
                "_event": _EVENT_OPEN, "ts": _iso(now_ms), "ts_ms": now_ms,
                "symbol": c.symbol, "side": c.side,
                "side_sign": 1.0 if c.side == "short_perp" else -1.0,
                "notional": notional, "entry_funding_annual": c.gross_annual,
                "rt_cost_usd": rt_cost_usd,
            })
        return {"action": "opened", "n_positions": len(ranked),
                "symbols": [c.symbol for c in ranked]}

    # Accrue each held position at the current funding rate.
    accrued = 0
    for sym, pos in state.items():
        row = by_symbol.get(sym)
        if row is None:
            continue  # no current quote this scan; skip (no fabricated accrual)
        periods = (now_ms - pos["last_ts_ms"]) / (period_hours * 3_600_000.0)
        if periods <= 0:
            continue
        delta = accrue_delta(pos["side_sign"], float(row["funding_rate"]), pos["notional"], periods)
        _write(path, {
            "_event": _EVENT_ACCRUE, "ts": _iso(now_ms), "ts_ms": now_ms,
            "symbol": sym, "funding_rate": float(row["funding_rate"]),
            "periods_elapsed": periods, "realized_delta": delta,
        })
        accrued += 1
    return {"action": "accrued", "n_positions": accrued}


def _iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def _report(path: str, notional_total: float) -> None:
    state = fold_ledger(path)
    if not state:
        print("  (no positions yet)")
        return
    total_net = 0.0
    print(f"  {'symbol':<20}{'side':>11}{'realized$':>11}{'net$':>9}{'accr':>6}")
    for sym, pos in sorted(state.items(), key=lambda kv: -_net_usd(kv[1])):
        net = _net_usd(pos)
        total_net += net
        print(f"  {sym:<20}{str(pos['side']):>11}{pos['realized_usd']:>11.3f}"
              f"{net:>9.3f}{pos['n_accruals']:>6}")
    deployed = sum(p["notional"] for p in state.values()) or 1.0
    print(f"  TOTAL net realized: ${total_net:.3f} on ${deployed:.0f} deployed "
          f"({total_net/deployed*100:+.3f}% so far, SHADOW)")


def _fetch_rows(exchange_id: str, period_hours: float) -> List[Dict[str, Any]]:
    import ccxt
    ex = getattr(ccxt, exchange_id)({"timeout": 20000, "enableRateLimit": True})
    return normalize_funding_rates(ex.fetch_funding_rates(), period_hours=period_hours)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Funding-carry SHADOW runner (NO ORDERS).")
    p.add_argument("--exchange", default="hyperliquid")
    p.add_argument("--period-hours", type=float, default=DEFAULT_PERIOD_HOURS)
    p.add_argument("--interval", type=float, default=3600.0, help="seconds between scans (default 1h)")
    p.add_argument("--once", action="store_true")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--min-net", type=float, default=0.10)
    p.add_argument("--notional", type=float, default=1000.0, help="USD per position (shadow)")
    p.add_argument("--round-trip-bps", type=float, default=DEFAULT_ROUND_TRIP_BPS)
    p.add_argument("--basis-buffer", type=float, default=DEFAULT_BASIS_BUFFER_ANNUAL)
    p.add_argument("--hold-days", type=float, default=DEFAULT_HOLD_DAYS)
    p.add_argument("--ledger-path", default=DEFAULT_LEDGER)
    args = p.parse_args(argv)

    def scan() -> None:
        rows = _fetch_rows(args.exchange, args.period_hours)
        res = run_scan(
            args.ledger_path, rows, now_ms=time.time() * 1000.0,
            period_hours=args.period_hours, notional=args.notional, top_k=args.top,
            min_net_annual=args.min_net, round_trip_bps=args.round_trip_bps,
            basis_buffer_annual=args.basis_buffer, hold_days=args.hold_days,
        )
        print(f"SHADOW carry scan @ {_utc_now_iso()}: {res['action']} "
              f"{res.get('n_positions', 0)} position(s)")
        _report(args.ledger_path, args.notional * args.top)

    print("=== funding-carry SHADOW runner — NO ORDERS PLACED ===")
    if args.once:
        scan()
        return 0
    try:
        while True:
            try:
                scan()
            except Exception as exc:  # noqa: BLE001 - a transient API error must not kill the run.
                print(f"  scan failed ({type(exc).__name__}: {str(exc)[:80]}); retrying next tick")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped by user. No orders were ever placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
