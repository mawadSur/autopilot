"""Rank Polymarket wallets by realized profit / win-rate (SHADOW-ONLY input).

This is the discovery half of the "smart-money wallet follower" edge: build a
roster of wallets that have *historically* been profitable on Polymarket, so a
later stage (``whale_follow_runner``) can fire a convergence signal when several
of them pile into the same outcome.

HONEST CAVEAT — past PnL is NOT future edge:
    Ranking on realized PnL is survivorship-prone. We only ever see the wallets
    that *did* win (the API surfaces active, often-large accounts), so a
    high-PnL roster is selected on exactly the variable we then "predict". A
    wallet's settled win-rate also says nothing about position sizing,
    correlation across its bets, or whether its edge persists. This is precisely
    why the ranker's output feeds a SHADOW ledger for forward validation, not
    live capital — we measure whether following these wallets *would have*
    worked before risking a dollar.

NO LOOK-AHEAD:
    Win-rate and realized PnL are computed ONLY from *settled* positions
    (``redeemable=true``). An open/unsettled position has an unknown final
    outcome, so counting it would leak future information into the record.

Win-rate approximation:
    The data-api does not label a position win/loss directly. We approximate a
    "win" as a settled position with ``realizedPnl > 0``. A position closed
    exactly flat (``realizedPnl == 0``) is counted as settled-but-not-a-win.
    Fees already net into ``realizedPnl`` server-side, so a marginal winner that
    paid more in fees than it made shows as a non-win — which is the honest
    accounting we want.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


__all__ = [
    "WalletStats",
    "stats_from_positions",
    "discover_active_wallets",
    "rank_wallets",
]


@dataclass
class WalletStats:
    """Realized-performance summary for a single wallet (settled positions only).

    Attributes:
        wallet:             The proxy-wallet address.
        n_settled:          Count of settled (``redeemable=true``) positions
                            sampled for this wallet.
        n_wins:             Of those, how many had ``realizedPnl > 0``.
        win_rate:           ``n_wins / n_settled`` (0.0 when ``n_settled == 0``).
        realized_pnl_usd:   Sum of ``realizedPnl`` over the settled positions.
        sampled_positions:  Total positions fetched for the wallet (settled +
                            unsettled), i.e. the denominator of what we looked
                            at — useful for spotting thin/unreliable samples.
    """

    wallet: str
    n_settled: int
    n_wins: int
    win_rate: float
    realized_pnl_usd: float
    sampled_positions: int


def _is_settled(position: Dict[str, Any]) -> bool:
    """Return True iff a position is resolved/settled (``redeemable=true``).

    Defensive: a missing or non-bool ``redeemable`` is treated as NOT settled,
    so an ambiguous row never leaks into the no-look-ahead record.
    """
    return position.get("redeemable") is True


def _realized_pnl(position: Dict[str, Any]) -> float:
    """Extract ``realizedPnl`` as a float; 0.0 when missing/non-numeric."""
    value = position.get("realizedPnl")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stats_from_positions(
    wallet: str,
    positions: Sequence[Dict[str, Any]],
) -> WalletStats:
    """Compute a wallet's realized-performance stats from its /positions list.

    Uses ONLY settled positions (``redeemable=true``) for the win-rate and
    realized-PnL record — open positions have an unknown outcome and are
    excluded to avoid look-ahead. A win is a settled position with
    ``realizedPnl > 0``.

    Args:
        wallet:     The proxy-wallet address (stored verbatim on the result).
        positions:  The wallet's positions, exactly as returned by
                    ``PolymarketDataAPIClient.get_positions``.

    Returns:
        A :class:`WalletStats`. With zero settled positions, ``win_rate`` is
        0.0 and ``realized_pnl_usd`` is 0.0.
    """
    settled = [p for p in positions if isinstance(p, dict) and _is_settled(p)]
    n_settled = len(settled)
    n_wins = sum(1 for p in settled if _realized_pnl(p) > 0)
    realized = sum(_realized_pnl(p) for p in settled)
    win_rate = (n_wins / n_settled) if n_settled else 0.0

    return WalletStats(
        wallet=wallet,
        n_settled=n_settled,
        n_wins=n_wins,
        win_rate=win_rate,
        realized_pnl_usd=float(realized),
        sampled_positions=sum(1 for p in positions if isinstance(p, dict)),
    )


def discover_active_wallets(
    client: Any,
    *,
    pages: int = 3,
    page_size: int = 100,
) -> List[str]:
    """Collect distinct proxy-wallets from recent trades (bounded).

    Walks at most ``pages`` pages of the global ``/trades`` feed (``page_size``
    rows each, paged by ``offset``) and returns the distinct ``proxyWallet``
    addresses seen, in first-seen order. The page bound keeps this from being an
    unbounded crawl of the entire trade history.

    A short page (fewer rows than ``page_size``) means the feed is exhausted, so
    we stop early rather than paging into empty responses.
    """
    seen: Dict[str, None] = {}  # insertion-ordered set preserving first-seen.
    for page in range(max(0, pages)):
        offset = page * page_size
        trades = client.get_trades(limit=page_size, offset=offset)
        if not trades:
            break
        for trade in trades:
            wallet = trade.get("proxyWallet") if isinstance(trade, dict) else None
            if wallet and wallet not in seen:
                seen[wallet] = None
        if len(trades) < page_size:
            break  # exhausted the feed; no point requesting an empty page.
    return list(seen.keys())


def rank_wallets(
    client: Any,
    *,
    candidate_wallets: Optional[Sequence[str]] = None,
    min_settled: int = 20,
    min_win_rate: float = 0.60,
    top_n: int = 50,
    positions_per_wallet: int = 500,
) -> List[WalletStats]:
    """Rank wallets by realized PnL, filtered for a credible track record.

    Pipeline:
        1. Resolve the candidate roster — use ``candidate_wallets`` if given,
           else :func:`discover_active_wallets`.
        2. For each candidate, fetch its positions (bounded by
           ``positions_per_wallet`` to keep the per-wallet fetch from being
           unbounded) and compute :func:`stats_from_positions`.
        3. Keep only wallets with ``n_settled >= min_settled`` AND
           ``win_rate >= min_win_rate`` (a thin or low-hit-rate sample is not a
           credible track record).
        4. Sort by ``realized_pnl_usd`` descending and return the top
           ``top_n``.

    This is one ``/positions`` request per candidate wallet (an N+1-style fan-out
    after discovery), so the candidate roster size directly bounds the request
    count; cap it via ``candidate_wallets`` or the discovery page bound when
    scanning live.

    See the module docstring for the honest survivorship caveat: this roster is
    a hypothesis to be validated in the SHADOW ledger, not a live-trading green
    light.
    """
    if candidate_wallets is None:
        candidate_wallets = discover_active_wallets(client)

    ranked: List[WalletStats] = []
    for wallet in candidate_wallets:
        if not wallet:
            continue
        positions = client.get_positions(wallet, limit=positions_per_wallet)
        stats = stats_from_positions(wallet, positions)
        if stats.n_settled >= min_settled and stats.win_rate >= min_win_rate:
            ranked.append(stats)

    ranked.sort(key=lambda s: s.realized_pnl_usd, reverse=True)
    return ranked[: max(0, top_n)]
