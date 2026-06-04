"""Crypto perpetual funding-rate carry scanner (SHADOW / READ-ONLY).

The market-neutral edge we pivoted to: harvest perpetual-futures FUNDING on a
delta-hedged position — short the perp + hold the equal spot (or long the perp +
short spot when funding is negative). You take NO directional view; you get paid
the funding rate for providing the hedge. The risk is funding flipping, basis
drift, and margin on the perp leg — not "did I guess the price right".

WHY EVERY NUMBER HERE IS NET OF COST: the legacy crypto-1m stack was killed
(docs/CRYPTO_1M_KILL.md) because its edge (+10-20bps) was 6-12x smaller than the
~120bps round-trip cost. The discipline that prevents repeating that mistake is
to subtract the cost UP FRONT. So this scanner ranks perps by carry NET of the
amortized round-trip fee (4 fills: open+close on both legs) and a basis buffer —
never by gross funding.

SCOPE — READ-ONLY, NO MONEY MOVES (Constitution: safety first):
    This module READS funding rates (via ccxt; Hyperliquid by default) and does
    arithmetic. It places no orders, signs nothing, holds no key, touches no
    wallet. Validation runs in SHADOW; live execution is a separate, gated,
    explicitly opted-in step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CarryRow",
    "annualize_funding",
    "amortized_round_trip_annual",
    "net_carry_annual",
    "normalize_funding_rates",
    "scan_carry",
    "DEFAULT_PERIOD_HOURS",
    "DEFAULT_ROUND_TRIP_BPS",
    "DEFAULT_HOLD_DAYS",
    "DEFAULT_BASIS_BUFFER_ANNUAL",
]

# Hyperliquid funds HOURLY. Other venues differ (Binance/Bybit 8h) — pass the
# right period when normalizing those.
DEFAULT_PERIOD_HOURS = 1.0
# Round-trip taker cost for the PAIR: open+close on BOTH legs = 4 fills. ~5 bps
# a fill is a conservative blended taker (Hyperliquid perp ~3.5bps + a spot leg);
# tune per real venue tiers before live. 4 * 5 = 20 bps round trip.
DEFAULT_ROUND_TRIP_BPS = 20.0
# Assumed holding period the one-time round-trip cost is amortized over. Funding
# carry is a HOLD strategy; longer holds shrink the amortized cost. 14 days is a
# deliberately short (pessimistic) default.
DEFAULT_HOLD_DAYS = 14.0
# Annualized haircut for basis drift / slippage / funding compression over the
# hold. Conservative; refine from shadow data.
DEFAULT_BASIS_BUFFER_ANNUAL = 0.05  # 5%/yr


@dataclass
class CarryRow:
    """One perp's carry opportunity, fully costed.

    ``side`` is the leg that HARVESTS the funding: ``short_perp`` when funding is
    positive (longs pay shorts) and ``long_perp`` when negative (shorts pay
    longs). ``gross_annual`` / ``net_annual`` are fractions (0.11 == 11%/yr).
    """

    symbol: str
    funding_rate: float        # per-period rate as reported by the venue
    period_hours: float
    gross_annual: float        # signed annualized funding
    side: str                  # 'short_perp' | 'long_perp'
    harvest_annual: float      # abs gross the chosen side collects
    cost_annual: float         # amortized round-trip + basis buffer
    net_annual: float          # harvest_annual - cost_annual (the honest carry)
    mark_price: Optional[float] = None


def annualize_funding(rate_per_period: float, period_hours: float) -> float:
    """Annualize a per-period funding rate. 1h venue: x 24 x 365."""
    if period_hours <= 0:
        return 0.0
    periods_per_year = (24.0 / period_hours) * 365.0
    return float(rate_per_period) * periods_per_year


def amortized_round_trip_annual(round_trip_bps: float, hold_days: float) -> float:
    """The one-time round-trip cost spread over the hold, expressed annualized.

    A round-trip fee is paid ONCE; funding accrues continuously. So as an annual
    rate the cost is ``(bps/1e4) * (365/hold_days)`` — hold longer, pay less per
    year. A 20bps round trip held 14 days ~= 5.2%/yr.
    """
    if hold_days <= 0:
        return float("inf")
    return (float(round_trip_bps) / 10_000.0) * (365.0 / float(hold_days))


def net_carry_annual(
    gross_annual: float,
    *,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
    hold_days: float = DEFAULT_HOLD_DAYS,
    basis_buffer_annual: float = DEFAULT_BASIS_BUFFER_ANNUAL,
) -> float:
    """Carry you'd actually keep: |gross| funding minus amortized fee + buffer.

    Uses ``abs(gross_annual)`` because either sign is harvestable by the matching
    side; the cost + buffer are always subtracted. Can be negative (don't trade).
    """
    harvest = abs(float(gross_annual))
    cost = amortized_round_trip_annual(round_trip_bps, hold_days) + float(basis_buffer_annual)
    return harvest - cost


def normalize_funding_rates(
    raw: Dict[str, Any],
    *,
    period_hours: float = DEFAULT_PERIOD_HOURS,
) -> List[Dict[str, Any]]:
    """Normalize a ccxt ``fetch_funding_rates()`` mapping into simple dicts.

    ccxt returns ``{symbol: {fundingRate, markPrice, ...}}``. We keep only rows
    with a numeric ``fundingRate`` and carry through the mark when present.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, dict):
        return out
    for symbol, info in raw.items():
        if not isinstance(info, dict):
            continue
        rate = info.get("fundingRate")
        try:
            rate_f = float(rate)
        except (TypeError, ValueError):
            continue
        mark = info.get("markPrice", info.get("indexPrice"))
        try:
            mark_f = float(mark) if mark is not None else None
        except (TypeError, ValueError):
            mark_f = None
        out.append({"symbol": str(symbol), "funding_rate": rate_f,
                    "period_hours": float(period_hours), "mark_price": mark_f})
    return out


def scan_carry(
    funding_rows: Sequence[Dict[str, Any]],
    *,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
    hold_days: float = DEFAULT_HOLD_DAYS,
    basis_buffer_annual: float = DEFAULT_BASIS_BUFFER_ANNUAL,
    min_net_annual: float = 0.0,
) -> List[CarryRow]:
    """Rank perps by NET annualized carry (descending), keeping only net > floor.

    ``funding_rows`` is the output of :func:`normalize_funding_rates`. Each row is
    costed via :func:`net_carry_annual`; only rows whose net clears
    ``min_net_annual`` are returned, best first. This is the honest opportunity
    set — what's left after the fees that killed the 1m model.
    """
    rows: List[CarryRow] = []
    for r in funding_rows:
        try:
            rate = float(r["funding_rate"])
            ph = float(r.get("period_hours", DEFAULT_PERIOD_HOURS))
        except (KeyError, TypeError, ValueError):
            continue
        gross = annualize_funding(rate, ph)
        net = net_carry_annual(
            gross, round_trip_bps=round_trip_bps, hold_days=hold_days,
            basis_buffer_annual=basis_buffer_annual,
        )
        if net <= min_net_annual:
            continue
        side = "short_perp" if gross > 0 else "long_perp"
        cost = amortized_round_trip_annual(round_trip_bps, hold_days) + basis_buffer_annual
        rows.append(CarryRow(
            symbol=str(r.get("symbol", "?")), funding_rate=rate, period_hours=ph,
            gross_annual=gross, side=side, harvest_annual=abs(gross),
            cost_annual=cost, net_annual=net, mark_price=r.get("mark_price"),
        ))
    rows.sort(key=lambda c: c.net_annual, reverse=True)
    return rows


def _fetch_live(exchange_id: str = "hyperliquid", period_hours: float = DEFAULT_PERIOD_HOURS):
    """Live funding via ccxt (only used by the CLI; tests inject rows)."""
    import ccxt  # local import: keep the module importable without ccxt
    ex = getattr(ccxt, exchange_id)({"timeout": 20000, "enableRateLimit": True})
    return normalize_funding_rates(ex.fetch_funding_rates(), period_hours=period_hours)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Crypto funding-carry scanner (READ-ONLY, no orders).")
    p.add_argument("--exchange", default="hyperliquid")
    p.add_argument("--period-hours", type=float, default=DEFAULT_PERIOD_HOURS)
    p.add_argument("--round-trip-bps", type=float, default=DEFAULT_ROUND_TRIP_BPS)
    p.add_argument("--hold-days", type=float, default=DEFAULT_HOLD_DAYS)
    p.add_argument("--basis-buffer", type=float, default=DEFAULT_BASIS_BUFFER_ANNUAL)
    p.add_argument("--min-net", type=float, default=0.10, help="min net annual carry to list (default 0.10 = 10%%/yr)")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)

    rows = _fetch_live(args.exchange, args.period_hours)
    ranked = scan_carry(
        rows, round_trip_bps=args.round_trip_bps, hold_days=args.hold_days,
        basis_buffer_annual=args.basis_buffer, min_net_annual=args.min_net,
    )
    cost = amortized_round_trip_annual(args.round_trip_bps, args.hold_days) + args.basis_buffer
    print(f"READ-ONLY funding carry on {args.exchange}: {len(rows)} perps, "
          f"{len(ranked)} clear net>{args.min_net*100:.0f}%/yr "
          f"(cost model: {args.round_trip_bps:.0f}bps round-trip over {args.hold_days:.0f}d + "
          f"{args.basis_buffer*100:.0f}% buffer = {cost*100:.1f}%/yr hurdle)")
    print(f"  {'symbol':<22}{'side':>11}{'gross/yr':>10}{'net/yr':>9}")
    for c in ranked[: args.top]:
        print(f"  {c.symbol:<22}{c.side:>11}{c.gross_annual*100:>9.1f}%{c.net_annual*100:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
