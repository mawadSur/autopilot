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
        rank_persistent, annualize_funding, DEFAULT_CARRY_UNIVERSE,
        DEFAULT_PERIOD_HOURS, DEFAULT_ROUND_TRIP_BPS, DEFAULT_HOLD_DAYS,
        DEFAULT_BASIS_BUFFER_ANNUAL,
    )
    from funding_carry_backtest import backtest_symbol, normalize_funding_history
except Exception:  # pragma: no cover
    from src.funding_carry_scanner import (  # type: ignore
        normalize_funding_rates, scan_carry, amortized_round_trip_annual,
        rank_persistent, annualize_funding, DEFAULT_CARRY_UNIVERSE,
        DEFAULT_PERIOD_HOURS, DEFAULT_ROUND_TRIP_BPS, DEFAULT_HOLD_DAYS,
        DEFAULT_BASIS_BUFFER_ANNUAL,
    )
    from src.funding_carry_backtest import backtest_symbol, normalize_funding_history  # type: ignore

__all__ = ["accrue_delta", "fold_ledger", "build_persistent_candidates", "DEFAULT_LEDGER"]

DEFAULT_LEDGER = "runs/funding_carry_ledger.jsonl"
_EVENT_OPEN = "open"
_EVENT_ACCRUE = "accrue"
_EVENT_CLOSE = "close"


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
    rt_cost_usd, entry_ts, last_ts_ms, realized_usd, n_accruals, closed}}``.
    ``open`` sets the position (``closed=False``); ``accrue`` adds realized +
    advances ``last_ts_ms``; ``close`` marks ``closed=True`` and records the exit
    reason/ts/funding. Closed positions stay in the returned dict.
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
                    "closed": False,
                }
            elif e.get("_event") == _EVENT_ACCRUE:
                pos = state.get(sym)
                if pos is None:
                    continue
                pos["realized_usd"] += float(e.get("realized_delta", 0.0))
                pos["last_ts_ms"] = float(e.get("ts_ms", pos["last_ts_ms"]))
                pos["n_accruals"] += 1
            elif e.get("_event") == _EVENT_CLOSE:
                pos = state.get(sym)
                if pos is None:
                    continue
                pos["closed"] = True
                pos["exit_reason"] = e.get("exit_reason")
                pos["exit_ts"] = e.get("ts")
                pos["exit_funding_rate"] = e.get("exit_funding_rate")
    return state


def _net_usd(pos: Dict[str, Any]) -> float:
    """Realized carry minus the one-time entry round-trip cost."""
    return pos["realized_usd"] - pos["rt_cost_usd"]


def build_persistent_candidates(
    funding_histories: Dict[str, Sequence[Dict[str, Any]]],
    *,
    period_hours: float,
    round_trip_bps: float,
    basis_buffer_annual: float,
    min_window_days: float = 7.0,
    min_favorable_pct: float = 0.60,
    min_net_annual: float = 0.0,
) -> List[Any]:
    """Rank a universe by REALIZED persistence (not a snapshot spike). Pure.

    ``funding_histories`` is ``{symbol: [{timestamp, fundingRate}, ...]}`` —
    already normalized or raw (we normalize each defensively). Each symbol is run
    through :func:`backtest_symbol` over its real paid funding; ``None`` results
    are dropped, then the survivors are filtered + ranked by
    :func:`funding_carry_scanner.rank_persistent`. No I/O.
    """
    bts: List[Any] = []
    for symbol, history in funding_histories.items():
        hist = normalize_funding_history(history)
        bt = backtest_symbol(
            symbol, hist, period_hours=period_hours,
            round_trip_bps=round_trip_bps, basis_buffer_annual=basis_buffer_annual,
        )
        if bt is not None:
            bts.append(bt)
    return rank_persistent(
        bts, min_window_days=min_window_days,
        min_favorable_pct=min_favorable_pct, min_net_annual=min_net_annual,
    )


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
    funding_histories: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
    min_window_days: float = 7.0,
    min_favorable_pct: float = 0.60,
    exit_below_annual: Optional[float] = None,
) -> Dict[str, Any]:
    """One scan: open the basket on first run, else accrue each held position.

    Pure w.r.t. I/O except the append-only ledger writes (so it is exercised in
    tests with a temp ledger + injected ``funding_rows``). Returns a summary dict.

    Back-compat: with the new keyword-only params absent (``funding_histories``
    None, ``exit_below_annual`` None) the behavior is exactly the snapshot
    ``scan_carry`` open + plain accrual. ``funding_histories`` switches the OPEN
    path to realized-persistence ranking; ``exit_below_annual`` enables
    decay/funding-flip exits during the ACCRUE path.
    """
    state = fold_ledger(path)
    by_symbol = {r["symbol"]: r for r in funding_rows}

    if not state:
        if funding_histories is not None:
            candidates = build_persistent_candidates(
                funding_histories, period_hours=period_hours,
                round_trip_bps=round_trip_bps, basis_buffer_annual=basis_buffer_annual,
                min_window_days=min_window_days, min_favorable_pct=min_favorable_pct,
                min_net_annual=min_net_annual,
            )
            rt_cost_usd = (round_trip_bps / 10_000.0) * notional
            for c in candidates[:top_k]:
                _write(path, {
                    "_event": _EVENT_OPEN, "ts": _iso(now_ms), "ts_ms": now_ms,
                    "symbol": c.symbol, "side": c.side,
                    "side_sign": 1.0 if c.side == "short_perp" else -1.0,
                    "notional": notional, "entry_funding_annual": c.gross_annual,
                    "rt_cost_usd": rt_cost_usd,
                })
            opened = candidates[:top_k]
            return {"action": "opened", "n_positions": len(opened),
                    "symbols": [c.symbol for c in opened]}

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

    # Accrue each held (not-closed) position at the current funding rate.
    accrued = 0
    closed = 0
    for sym, pos in state.items():
        if pos.get("closed"):
            continue
        row = by_symbol.get(sym)
        if row is None:
            continue  # no current quote this scan; skip (no fabricated accrual)
        periods = (now_ms - pos["last_ts_ms"]) / (period_hours * 3_600_000.0)
        if periods <= 0:
            continue
        current_rate = float(row["funding_rate"])
        delta = accrue_delta(pos["side_sign"], current_rate, pos["notional"], periods)
        _write(path, {
            "_event": _EVENT_ACCRUE, "ts": _iso(now_ms), "ts_ms": now_ms,
            "symbol": sym, "funding_rate": current_rate,
            "periods_elapsed": periods, "realized_delta": delta,
        })
        accrued += 1
        if exit_below_annual is not None:
            current_gross_annual = annualize_funding(current_rate, period_hours)
            favorable = pos["side_sign"] * current_gross_annual
            if favorable < exit_below_annual:
                _write(path, {
                    "_event": _EVENT_CLOSE, "ts": _iso(now_ms), "ts_ms": now_ms,
                    "symbol": sym,
                    "exit_reason": "funding_flip" if favorable < 0 else "decay",
                    "exit_funding_rate": current_rate,
                })
                closed += 1
    return {"action": "accrued", "n_positions": accrued, "n_closed": closed}


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


def _fetch_histories(universe: Sequence[str], exchange_id: str, days: float,
                     period_hours: float) -> Dict[str, List[Dict[str, Any]]]:
    """Funding history per symbol via ccxt (CLI only; tests inject histories)."""
    import ccxt  # lazy import: keep the module importable without ccxt
    ex = getattr(ccxt, exchange_id)({"timeout": 20000, "enableRateLimit": True})
    since = int(time.time() * 1000.0 - days * 86_400_000.0)
    limit = int(days * 24.0 / max(period_hours, 1e-9)) + 10
    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym in universe:
        try:
            raw = ex.fetch_funding_rate_history(sym, since=since, limit=min(limit, 5000))
            out[sym] = normalize_funding_history(raw)
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the fetch.
            print(f"  history fetch failed for {sym} "
                  f"({type(exc).__name__}: {str(exc)[:60]})")
    return out


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
    p.add_argument("--universe", nargs="*", default=list(DEFAULT_CARRY_UNIVERSE),
                   help="perp symbols to consider for persistence-based opens (default BTC/ETH)")
    p.add_argument("--persistence", dest="persistence", action="store_true", default=True,
                   help="open from REALIZED persistence over funding history (default ON)")
    p.add_argument("--no-persistence", dest="persistence", action="store_false",
                   help="open from the snapshot scan instead of funding history")
    p.add_argument("--history-days", type=float, default=30.0,
                   help="days of funding history to replay for persistence (default 30)")
    p.add_argument("--min-window-days", type=float, default=7.0,
                   help="min realized window to trust a carry (default 7)")
    p.add_argument("--min-favorable", type=float, default=0.60,
                   help="min share of periods the side earned (default 0.60)")
    p.add_argument("--exit-below-annual", type=float, default=0.0,
                   help="close when favorable annualized funding < this; <0 => funding flip, "
                        "0 => flip exit, >0 => decay exit (default 0.0)")
    p.add_argument("--no-exit", dest="exit_off", action="store_true",
                   help="disable decay/funding-flip exits (back-compat accrual only)")
    args = p.parse_args(argv)

    exit_below = None if args.exit_off else args.exit_below_annual

    def scan() -> None:
        rows = _fetch_rows(args.exchange, args.period_hours)
        histories = None
        if args.persistence and not fold_ledger(args.ledger_path):
            histories = _fetch_histories(args.universe, args.exchange,
                                         args.history_days, args.period_hours)
        res = run_scan(
            args.ledger_path, rows, now_ms=time.time() * 1000.0,
            period_hours=args.period_hours, notional=args.notional, top_k=args.top,
            min_net_annual=args.min_net, round_trip_bps=args.round_trip_bps,
            basis_buffer_annual=args.basis_buffer, hold_days=args.hold_days,
            funding_histories=histories, min_window_days=args.min_window_days,
            min_favorable_pct=args.min_favorable, exit_below_annual=exit_below,
        )
        print(f"SHADOW carry scan @ {_utc_now_iso()}: {res['action']} "
              f"{res.get('n_positions', 0)} position(s)"
              + (f", {res['n_closed']} closed" if res.get("n_closed") else ""))
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
