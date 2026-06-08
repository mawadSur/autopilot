"""Funding-carry HEDGE feasibility (SHADOW / READ-ONLY) — the make-or-break gate.

The scanner/backtest price only the FUNDING leg. But you only collect funding if
you can HOLD BOTH legs of the delta-neutral hedge — and whether that's cheap or
nearly impossible depends on the funding's SIGN:

  * funding POSITIVE  -> harvest by SHORT perp + LONG spot. The hedge is
    "buy and hold the token", trivial IF spot is listed on a venue you can reach.
  * funding NEGATIVE  -> harvest by LONG perp + SHORT spot. The hedge needs you
    to BORROW the token to short it — for niche coins that's expensive or simply
    unavailable to retail, so the carry is usually NOT harvestable.

This module asks, per carry candidate: which hedge does it need, is the spot leg
actually available (Hyperliquid spot / Coinbase / Kraken), and what's the verdict
— so we don't chase a carry we can't build. Same edge-vs-cost discipline that
killed the 1m model, applied to the HEDGE.

SCOPE — READ-ONLY: reads venue market lists + funding; no orders, no signing, no
wallet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

__all__ = [
    "HedgeAssessment",
    "base_token",
    "required_hedge",
    "find_spot_venues",
    "all_in_net",
    "assess",
    "DEFAULT_SPOT_VENUES",
    "DEFAULT_BORROW_ANNUAL",
]

# Venues we'll probe for a holdable spot leg (ccxt ids). Binance is geo-blocked
# (451) from here, so it's intentionally omitted.
DEFAULT_SPOT_VENUES = ["hyperliquid", "coinbase", "kraken"]
# Assumed annual borrow rate for the SHORT-spot leg when it IS borrowable. A
# placeholder hurdle — real hard-to-borrow rates on niche coins are far worse or
# the borrow doesn't exist at all (handled by the 'needs_borrow' verdict).
DEFAULT_BORROW_ANNUAL = 0.20  # 20%/yr


@dataclass
class HedgeAssessment:
    symbol: str
    side: str                  # carry side: 'short_perp' | 'long_perp'
    funding_net_annual: float  # the scanner's net carry (funding leg, after its cost model)
    hedge_kind: str            # 'buy_spot' | 'short_spot'
    spot_venues: List[str]     # venues that list the spot (empty = none found)
    all_in_net_annual: float   # net after the hedge-leg cost (borrow for short_spot)
    verdict: str               # 'harvestable' | 'needs_borrow' | 'no_spot'


def base_token(perp_symbol: str) -> str:
    """'XMR/USDC:USDC' -> 'XMR' (the base coin), upper-cased. Tolerant of plain ids."""
    s = str(perp_symbol)
    if "/" in s:
        s = s.split("/", 1)[0]
    return s.strip().upper()


def required_hedge(side: str) -> str:
    """Hedge kind for a carry side. short_perp (pos funding) -> buy_spot;
    long_perp (neg funding) -> short_spot (needs borrow)."""
    return "buy_spot" if side == "short_perp" else "short_spot"


def find_spot_venues(base: str, spot_by_base: Dict[str, Set[str]]) -> List[str]:
    """Venues listing a spot market for ``base`` (from a {BASE: {venues}} map)."""
    return sorted(spot_by_base.get(base.upper(), set()))


def all_in_net(funding_net_annual: float, hedge_kind: str, *,
               borrow_annual: float = DEFAULT_BORROW_ANNUAL) -> float:
    """Net carry after the hedge-leg cost.

    buy_spot: the scanner's round-trip already covers the spot fee, so all-in ==
    funding net. short_spot: subtract the borrow rate you'd pay to hold the short.
    """
    if hedge_kind == "short_spot":
        return float(funding_net_annual) - float(borrow_annual)
    return float(funding_net_annual)


def assess(
    candidates: Sequence[Any],
    spot_by_base: Dict[str, Set[str]],
    *,
    borrow_annual: float = DEFAULT_BORROW_ANNUAL,
) -> List[HedgeAssessment]:
    """Classify each carry candidate by hedge buildability.

    ``candidates`` are scanner ``CarryRow``-like objects (need ``.symbol``,
    ``.side``, ``.net_annual``). ``spot_by_base`` maps an upper-cased base token
    to the set of venues listing its spot. Verdicts:
      * ``buy_spot`` + spot found -> ``harvestable`` (all-in == funding net).
      * ``buy_spot`` + no spot    -> ``no_spot`` (can't hold the hedge).
      * ``short_spot``            -> ``needs_borrow`` (all-in nets the borrow; flag
        that borrow is often unavailable/expensive for these names).
    Sorted by all-in net, descending.
    """
    out: List[HedgeAssessment] = []
    for c in candidates:
        sym = getattr(c, "symbol")
        side = getattr(c, "side")
        net = float(getattr(c, "net_annual"))
        base = base_token(sym)
        hedge = required_hedge(side)
        venues = find_spot_venues(base, spot_by_base)
        all_in = all_in_net(net, hedge, borrow_annual=borrow_annual)
        if hedge == "buy_spot":
            verdict = "harvestable" if venues else "no_spot"
        else:
            verdict = "needs_borrow"
        out.append(HedgeAssessment(
            symbol=sym, side=side, funding_net_annual=net, hedge_kind=hedge,
            spot_venues=venues, all_in_net_annual=all_in, verdict=verdict,
        ))
    out.sort(key=lambda a: a.all_in_net_annual, reverse=True)
    return out


def _load_spot_by_base(venue_ids: Sequence[str]) -> Dict[str, Set[str]]:
    """Live: build {BASE: {venues}} of spot markets via ccxt (CLI only)."""
    import ccxt
    spot_by_base: Dict[str, Set[str]] = {}
    for vid in venue_ids:
        try:
            ex = getattr(ccxt, vid)({"timeout": 20000, "enableRateLimit": True})
            markets = ex.load_markets()
        except Exception as exc:  # noqa: BLE001
            print(f"  (skip {vid}: {type(exc).__name__})")
            continue
        for m in markets.values():
            if not isinstance(m, dict) or not m.get("spot"):
                continue
            base = m.get("base")
            if base:
                spot_by_base.setdefault(str(base).upper(), set()).add(vid)
    return spot_by_base


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    try:
        from funding_carry_scanner import normalize_funding_rates, scan_carry
    except Exception:  # pragma: no cover
        from src.funding_carry_scanner import normalize_funding_rates, scan_carry  # type: ignore
    import ccxt

    p = argparse.ArgumentParser(description="Funding-carry hedge feasibility (READ-ONLY).")
    p.add_argument("--exchange", default="hyperliquid")
    p.add_argument("--period-hours", type=float, default=1.0)
    p.add_argument("--min-net", type=float, default=0.10)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--borrow-annual", type=float, default=DEFAULT_BORROW_ANNUAL)
    p.add_argument("--spot-venues", nargs="*", default=DEFAULT_SPOT_VENUES)
    args = p.parse_args(argv)

    ex = getattr(ccxt, args.exchange)({"timeout": 20000, "enableRateLimit": True})
    rows = normalize_funding_rates(ex.fetch_funding_rates(), period_hours=args.period_hours)
    cands = scan_carry(rows, min_net_annual=args.min_net)[: args.top]
    print(f"Loading spot markets from {args.spot_venues} ...")
    spot_by_base = _load_spot_by_base(args.spot_venues)
    res = assess(cands, spot_by_base, borrow_annual=args.borrow_annual)

    harv = [a for a in res if a.verdict == "harvestable"]
    print(f"\nHEDGE FEASIBILITY on {len(cands)} top carries (borrow assumed {args.borrow_annual*100:.0f}%/yr for short-spot):")
    print(f"  {'symbol':<20}{'hedge':>11}{'fund net/yr':>12}{'all-in/yr':>11}  verdict / spot venues")
    for a in res:
        venues = ",".join(a.spot_venues) if a.spot_venues else "-"
        print(f"  {a.symbol:<20}{a.hedge_kind:>11}{a.funding_net_annual*100:>11.1f}%"
              f"{a.all_in_net_annual*100:>10.1f}%  {a.verdict:<13} {venues}")
    print(f"\n  HARVESTABLE (buy-spot + spot available): {len(harv)} of {len(cands)}")
    for a in harv[:10]:
        print(f"     {a.symbol:<20} all-in {a.all_in_net_annual*100:>6.1f}%/yr  via {','.join(a.spot_venues)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
