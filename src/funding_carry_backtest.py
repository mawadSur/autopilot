"""Funding-rate carry BACKTEST over real history (SHADOW / READ-ONLY).

Replays each perp's ACTUAL paid funding (ccxt ``fetch_funding_rate_history``) to
answer: if you had held the delta-neutral carry on this perp over the window,
what carry would you actually have KEPT — net of cost — and did funding stay on
your side or whip against you?

Why this is a clean historical test (unlike the Polymarket /positions dead-end):
funding history is the REAL cashflow the venue paid, unambiguous — there is no
outcome to reconstruct and no sell-the-bounce selection bias. The only honest
caveats are (1) SELECTION: ranking by today's funding and backtesting is
forward-looking, so we also report a fixed LIQUID universe with no selection;
(2) this models the FUNDING leg only — basis P/L of the hedge is assumed to net
to ~0 over the hold (a real hedge has small basis drift) and is covered by the
scanner's basis buffer; (3) a static hold (one side fixed at entry) is
pessimistic vs a strategy that ROTATES out when funding flips.

SCOPE — READ-ONLY, NO ORDERS: reads funding history + arithmetic. No signing,
no wallet, no execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

try:  # Flat import under PYTHONPATH=src.
    from funding_carry_scanner import amortized_round_trip_annual, DEFAULT_PERIOD_HOURS
except Exception:  # pragma: no cover
    from src.funding_carry_scanner import amortized_round_trip_annual, DEFAULT_PERIOD_HOURS  # type: ignore

__all__ = [
    "SymbolBacktest",
    "normalize_funding_history",
    "backtest_symbol",
    "DEFAULT_LIQUID_UNIVERSE",
]

# A fixed, liquid, hedgeable set — backtested WITHOUT selection bias to show the
# robust baseline carry (vs the headline 100%+ thin names).
DEFAULT_LIQUID_UNIVERSE = [
    "BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC",
    "XRP/USDC:USDC", "DOGE/USDC:USDC", "AVAX/USDC:USDC",
]

_MS_PER_DAY = 86_400_000.0


@dataclass
class SymbolBacktest:
    symbol: str
    n_periods: int
    window_days: float
    side: str                 # 'short_perp' | 'long_perp' (the persistent side)
    realized_total: float     # cumulative funding return over window (fraction of notional)
    gross_annual: float       # realized annualized (before cost)
    net_annual: float         # after amortized round-trip + basis buffer
    favorable_pct: float      # share of periods the side earned (not paid)
    worst_period: float       # most adverse single-period funding P/L (fraction)


def normalize_funding_history(raw: Any) -> List[Dict[str, float]]:
    """ccxt ``fetch_funding_rate_history`` -> sorted [{timestamp, fundingRate}].

    Keeps only entries with a numeric ``fundingRate``; sorts ascending by
    ``timestamp`` when present.
    """
    out: List[Dict[str, float]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for e in raw:
        if not isinstance(e, dict):
            continue
        r = e.get("fundingRate")
        try:
            r_f = float(r)
        except (TypeError, ValueError):
            continue
        ts = e.get("timestamp")
        try:
            ts_f = float(ts) if ts is not None else None
        except (TypeError, ValueError):
            ts_f = None
        out.append({"timestamp": ts_f, "fundingRate": r_f})
    out.sort(key=lambda x: (x["timestamp"] is None, x["timestamp"] or 0.0))
    return out


def backtest_symbol(
    symbol: str,
    history: Sequence[Dict[str, float]],
    *,
    period_hours: float = DEFAULT_PERIOD_HOURS,
    round_trip_bps: float = 20.0,
    basis_buffer_annual: float = 0.05,
) -> Optional[SymbolBacktest]:
    """Backtest the static delta-neutral carry on one perp's funding history.

    Side is the PERSISTENT direction (sign of mean funding): short the perp when
    funding is usually positive (receive ``+r`` each period), long when usually
    negative (receive ``-r``). Per-period P/L fraction is ``side_sign * r_t``;
    when funding flips, that period is a LOSS — so ``favorable_pct`` and
    ``worst_period`` expose the whip risk a snapshot can't. ``None`` if no usable
    history.
    """
    rows = [h for h in history if isinstance(h.get("fundingRate"), (int, float))]
    n = len(rows)
    if n == 0:
        return None
    rates = [float(h["fundingRate"]) for h in rows]
    mean_r = sum(rates) / n
    side_sign = 1.0 if mean_r >= 0 else -1.0
    side = "short_perp" if side_sign > 0 else "long_perp"

    per_period = [side_sign * r for r in rates]      # funding P/L each period
    realized_total = sum(per_period)
    favorable_pct = sum(1 for x in per_period if x > 0) / n
    worst_period = min(per_period)

    # Window length: prefer real timestamps; fall back to n * period.
    ts = [h["timestamp"] for h in rows if h.get("timestamp") is not None]
    if len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
        window_days = (ts[-1] - ts[0]) / _MS_PER_DAY
    else:
        window_days = n * period_hours / 24.0
    if window_days <= 0:
        window_days = max(period_hours / 24.0, 1e-9)

    gross_annual = realized_total * (365.0 / window_days)
    cost = amortized_round_trip_annual(round_trip_bps, window_days) + basis_buffer_annual
    net_annual = gross_annual - cost

    return SymbolBacktest(
        symbol=symbol, n_periods=n, window_days=window_days, side=side,
        realized_total=realized_total, gross_annual=gross_annual,
        net_annual=net_annual, favorable_pct=favorable_pct, worst_period=worst_period,
    )


def _fetch_history(symbol: str, exchange_id: str, days: float, period_hours: float):
    """Live funding history via ccxt (CLI only; tests inject)."""
    import time
    import ccxt
    ex = getattr(ccxt, exchange_id)({"timeout": 20000, "enableRateLimit": True})
    since = int(time.time() * 1000 - days * _MS_PER_DAY)
    limit = int(days * 24.0 / max(period_hours, 1e-9)) + 10
    raw = ex.fetch_funding_rate_history(symbol, since=since, limit=min(limit, 5000))
    return normalize_funding_history(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Funding-carry backtest (READ-ONLY).")
    p.add_argument("--exchange", default="hyperliquid")
    p.add_argument("--symbols", nargs="*", default=None,
                   help="perp symbols; default = the fixed liquid universe (no selection bias)")
    p.add_argument("--days", type=float, default=90.0)
    p.add_argument("--period-hours", type=float, default=DEFAULT_PERIOD_HOURS)
    p.add_argument("--round-trip-bps", type=float, default=20.0)
    p.add_argument("--basis-buffer", type=float, default=0.05)
    args = p.parse_args(argv)

    symbols = args.symbols or DEFAULT_LIQUID_UNIVERSE
    print(f"READ-ONLY funding-carry backtest on {args.exchange}, ~{args.days:.0f}d, "
          f"{len(symbols)} symbol(s); net = realized funding - amortized "
          f"{args.round_trip_bps:.0f}bps round-trip - {args.basis_buffer*100:.0f}% buffer")
    print(f"  {'symbol':<20}{'side':>11}{'n':>6}{'days':>7}{'gross/yr':>10}{'net/yr':>9}{'fav%':>7}{'worst':>9}")
    results: List[SymbolBacktest] = []
    for sym in symbols:
        try:
            hist = _fetch_history(sym, args.exchange, args.days, args.period_hours)
            bt = backtest_symbol(sym, hist, period_hours=args.period_hours,
                                 round_trip_bps=args.round_trip_bps,
                                 basis_buffer_annual=args.basis_buffer)
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym:<20}  ERROR {type(exc).__name__}: {str(exc)[:60]}")
            continue
        if bt is None:
            print(f"  {sym:<20}  no history")
            continue
        results.append(bt)
        print(f"  {bt.symbol:<20}{bt.side:>11}{bt.n_periods:>6}{bt.window_days:>7.0f}"
              f"{bt.gross_annual*100:>9.1f}%{bt.net_annual*100:>8.1f}%"
              f"{bt.favorable_pct*100:>6.0f}%{bt.worst_period*100:>8.2f}%")
    if results:
        avg_net = sum(b.net_annual for b in results) / len(results)
        print(f"  {'— AVG net annual —':<20}{avg_net*100:>54.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
