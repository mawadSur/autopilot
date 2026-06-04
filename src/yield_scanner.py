"""Stablecoin YIELD scanner (READ-ONLY) — the "automate a known yield" pivot.

After three edge-seeking strategies all died on execution constraints (crypto-1m
fees, Polymarket -EV, funding-carry hedge), the goal changed: stop trying to BEAT
the market and instead COLLECT a structural yield (supply stablecoins, earn the
lending/LP rate) and manage the risk. No directional view, no counterparty to
out-trade.

The risk just changes shape — for stablecoin yield it is SMART-CONTRACT risk
(the protocol gets exploited), DEPEG risk (the stablecoin breaks $1), and CUSTODY
risk (self-custody wallet / keys) — NOT fees or prediction. So this scanner does
NOT fake a single "risk-adjusted APY"; it surfaces the honest signals that let a
human judge risk: TVL tier (a battle-tested proxy), BASE vs INCENTIVE apy (reward
tokens are less durable), and an established-protocol allowlist for the
conservative view.

Data: DefiLlama's free yields aggregator (``yields.llama.fi/pools``). READ-ONLY:
no orders, no signing, no wallet — just reads the public aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "YieldPool",
    "normalize_pools",
    "filter_pools",
    "rank_pools",
    "tvl_tier",
    "DEFAULT_TRUSTED_PROJECTS",
    "DEFAULT_MIN_TVL",
    "DEFAULT_MAX_APY",
]

# Battle-tested lending/stable protocols for the CONSERVATIVE view. TVL +
# track record, not a guarantee — smart-contract risk is never zero.
DEFAULT_TRUSTED_PROJECTS = {
    "aave-v3", "aave-v2", "compound-v3", "compound-v2", "morpho-blue", "morpho-aave",
    "sky-lending", "makerdao", "spark", "fluid-lending", "fluid", "curve-dex",
    "convex-finance", "ethena-usde", "sdai",
}

DEFAULT_MIN_TVL = 10_000_000.0   # $10M floor: avoid tiny/illiquid/rug-prone pools
DEFAULT_MAX_APY = 40.0           # above this, an organic stable yield is usually a mirage/incentive


@dataclass
class YieldPool:
    project: str
    chain: str
    symbol: str
    apy: float            # total APY (%)
    apy_base: float       # organic (interest/fees) APY — the durable part
    apy_reward: float     # incentive-token APY — less durable
    tvl_usd: float
    stablecoin: bool
    il_risk: str          # 'no' for stablecoin pools
    tier: str             # 'established' | 'mid' | 'small'
    trusted: bool         # in the established-protocol allowlist


def tvl_tier(tvl_usd: float) -> str:
    """TVL as a (rough) battle-tested proxy. >$100M established, $20-100M mid."""
    if tvl_usd >= 100_000_000:
        return "established"
    if tvl_usd >= 20_000_000:
        return "mid"
    return "small"


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def normalize_pools(raw: Any, *, trusted: Optional[Sequence[str]] = None) -> List[YieldPool]:
    """Normalize DefiLlama ``/pools`` rows into :class:`YieldPool`.

    Accepts the ``{"data": [...]}`` envelope or a bare list. Skips rows missing a
    numeric ``apy``.
    """
    trusted_set = set(trusted) if trusted is not None else DEFAULT_TRUSTED_PROJECTS
    if isinstance(raw, dict):
        raw = raw.get("data", [])
    out: List[YieldPool] = []
    if not isinstance(raw, (list, tuple)):
        return out
    for p in raw:
        if not isinstance(p, dict) or p.get("apy") is None:
            continue
        tvl = _num(p.get("tvlUsd"))
        project = str(p.get("project") or "")
        out.append(YieldPool(
            project=project, chain=str(p.get("chain") or ""),
            symbol=str(p.get("symbol") or ""), apy=_num(p.get("apy")),
            apy_base=_num(p.get("apyBase")), apy_reward=_num(p.get("apyReward")),
            tvl_usd=tvl, stablecoin=bool(p.get("stablecoin")),
            il_risk=str(p.get("ilRisk") or "?"), tier=tvl_tier(tvl),
            trusted=project in trusted_set,
        ))
    return out


def filter_pools(
    pools: Sequence[YieldPool],
    *,
    min_tvl: float = DEFAULT_MIN_TVL,
    max_apy: float = DEFAULT_MAX_APY,
    stablecoin_only: bool = True,
    trusted_only: bool = False,
) -> List[YieldPool]:
    """Keep pools clearing the TVL floor + sane APY, stablecoin (no IL), optionally
    only established protocols. ``max_apy`` cuts incentive mirages."""
    out: List[YieldPool] = []
    for p in pools:
        if stablecoin_only and not p.stablecoin:
            continue
        if p.tvl_usd < min_tvl:
            continue
        if not (0.0 < p.apy <= max_apy):
            continue
        if trusted_only and not p.trusted:
            continue
        out.append(p)
    return out


def rank_pools(pools: Sequence[YieldPool]) -> List[YieldPool]:
    """Rank by BASE apy (the durable, organic yield), descending."""
    return sorted(pools, key=lambda p: p.apy_base, reverse=True)


def _fetch_live(timeout: float = 30.0) -> Any:
    import requests
    return requests.get("https://yields.llama.fi/pools", timeout=timeout).json()


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Stablecoin yield scanner (READ-ONLY).")
    p.add_argument("--min-tvl", type=float, default=DEFAULT_MIN_TVL)
    p.add_argument("--max-apy", type=float, default=DEFAULT_MAX_APY)
    p.add_argument("--trusted-only", action="store_true", help="only established/allowlisted protocols")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)

    pools = normalize_pools(_fetch_live())
    kept = rank_pools(filter_pools(
        pools, min_tvl=args.min_tvl, max_apy=args.max_apy, trusted_only=args.trusted_only,
    ))
    scope = "TRUSTED protocols only" if args.trusted_only else "all protocols"
    print(f"READ-ONLY stablecoin yields ({scope}, >= ${args.min_tvl/1e6:.0f}M TVL, <= {args.max_apy:.0f}% APY): "
          f"{len(kept)} pools")
    print(f"  {'project':<18}{'chain':<10}{'symbol':<16}{'base%':>7}{'rwd%':>7}{'TVL$M':>8}{'tier':>12}{'trust':>6}")
    for y in kept[: args.top]:
        print(f"  {y.project[:16]:<18}{y.chain[:8]:<10}{y.symbol[:14]:<16}"
              f"{y.apy_base:>6.2f}%{y.apy_reward:>6.2f}%{y.tvl_usd/1e6:>7.0f}{y.tier:>12}"
              f"{'  Y' if y.trusted else '  -':>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
