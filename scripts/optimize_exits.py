"""Optimize whale-follow stop-loss / take-profit on REAL resolved winner positions.

This is the READ-ONLY live half of the exit-overlay study. It pulls the top profit
winners off the Polymarket leaderboard, reconstructs each winner's REAL resolved
positions and the forward price path they sat through, then sweeps a grid of
stop-loss / take-profit thresholds via :func:`exit_backtest.grid_search` to find the
overlay that would have improved the realized return — and reports it against the
baseline of doing nothing (hold to resolution).

SCOPE — READ-ONLY, NO MONEY MOVES (Constitution: safety first):
    This script issues ONLY data reads through :class:`PolymarketDataAPIClient`
    (``get_profit_leaderboard`` / ``get_positions`` / ``get_trades``). It places no
    orders, signs nothing, redeems nothing, and touches no wallet/web3 path. The
    grid search is a pure offline computation. Nothing here moves money.

No look-ahead:
    For each resolved position the whale's ENTRY timestamp is found from THEIR OWN
    trades on that outcome (the min timestamp on the matching outcomeIndex). The
    forward price path is then built ONLY from market trades on that same outcome
    with ``timestamp > entry_ts`` — strictly post-entry prices. The exit simulation
    consumes that path in time order and never consults the resolved outcome until
    the path is exhausted. So every exit decision uses only information that was
    observable at decision time. Positions with no usable post-entry path are
    skipped (counted), never backfilled with a guessed price.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_SRC_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exchanges.polymarket_data_api import (  # noqa: E402
    PolymarketDataAPIClient,
    PolymarketDataAPIError,
)
from exit_backtest import grid_search  # noqa: E402


# The grid swept by default. ``None`` on each axis tests "no stop" / "no TP", so the
# baseline (all-None hold-to-resolution) is always part of the search.
SL_VALUES: List[Optional[float]] = [None, 0.2, 0.3, 0.4, 0.5, 0.6]
TP_PCT_VALUES: List[Optional[float]] = [None, 0.3, 0.5, 0.75, 1.0]
TP_PRICE_VALUES: List[Optional[float]] = [None, 0.85, 0.90, 0.95]


def _is_resolved(position: Dict[str, Any]) -> bool:
    """A position is RESOLVED iff redeemable, or its curPrice is pinned to 0/1.

    A settled Polymarket outcome marks to exactly $0 (lost) or $1 (won); we treat
    ``curPrice`` within 1e-3 of either bound, OR an explicit ``redeemable: true``,
    as resolved. Anything else is still open and is left out of the dataset.
    """
    if position.get("redeemable") is True:
        return True
    try:
        cur = float(position.get("curPrice"))
    except (TypeError, ValueError):
        return False
    return cur <= 0.001 or cur >= 0.999


def _final_price(position: Dict[str, Any]) -> float:
    """Resolved price of the held outcome: 1.0 (won) or 0.0 (lost).

    Uses ``curPrice`` when it is pinned near a bound; otherwise (a redeemable
    position whose curPrice is mid-range or missing) falls back to ``redeemable``
    meaning a winning, redeemable claim -> 1.0, else 0.0. This is intentionally
    conservative: a redeemable position is one the wallet can claim, i.e. it won.
    """
    try:
        cur = float(position.get("curPrice"))
    except (TypeError, ValueError):
        cur = None
    if cur is not None:
        if cur >= 0.999:
            return 1.0
        if cur <= 0.001:
            return 0.0
    # Mid-range/missing curPrice but redeemable: a claimable position is a win.
    return 1.0 if position.get("redeemable") is True else 0.0


def _entry_ts(
    client: PolymarketDataAPIClient,
    wallet: str,
    condition_id: str,
    outcome_index: int,
    *,
    limit: int = 500,
) -> Optional[int]:
    """The whale's ENTRY timestamp on this outcome: min ts over THEIR own trades.

    Fetches the wallet's trades on this market and returns the earliest timestamp
    among trades matching ``outcome_index``. Returns ``None`` on any data-api error
    or when no matching trade carries a usable timestamp — the caller then skips the
    position rather than guess an entry time.
    """
    try:
        trades = client.get_trades(user=wallet, market=condition_id, limit=limit)
    except PolymarketDataAPIError:
        return None

    earliest: Optional[int] = None
    for trade in trades or []:
        if not isinstance(trade, dict):
            continue
        if trade.get("outcomeIndex") != outcome_index:
            continue
        try:
            ts = int(trade.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if earliest is None or ts < earliest:
            earliest = ts
    return earliest


def _forward_path(
    client: PolymarketDataAPIClient,
    condition_id: str,
    outcome_index: int,
    entry_ts: int,
    *,
    limit: int = 500,
    max_pages: int = 6,
) -> tuple:
    """Build the post-entry price path for one outcome (NO look-ahead).

    Paginates the market's trades (the data-api returns them newest-first) and
    keeps ONLY those on ``outcome_index`` with ``timestamp > entry_ts`` (strictly
    after the whale entered), sorted ascending by timestamp. Walks back page by
    page until a page's oldest trade is at/before ``entry_ts`` (the full
    post-entry window is covered) or ``max_pages`` is hit.

    Returns ``(path, truncated)``. ``truncated`` is True when the page cap was
    reached before getting back to ``entry_ts`` — the path may then miss the
    EARLIEST post-entry prices (a backtest bias that can understate a stop-loss,
    NOT look-ahead). Prices validated to ``[0, 1]``.
    """
    rows: List[tuple] = []
    truncated = False
    offset = 0
    for _page in range(max_pages):
        try:
            trades = client.get_trades(
                market=condition_id, limit=limit, offset=offset
            )
        except PolymarketDataAPIError:
            break
        if not trades:
            break
        oldest_on_page = None
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            try:
                ts = int(trade.get("timestamp"))
            except (TypeError, ValueError):
                continue
            if oldest_on_page is None or ts < oldest_on_page:
                oldest_on_page = ts
            if trade.get("outcomeIndex") != outcome_index:
                continue
            try:
                price = float(trade.get("price"))
            except (TypeError, ValueError):
                continue
            if ts <= entry_ts or not (0.0 <= price <= 1.0):
                continue
            rows.append((ts, price))
        # Newest-first paging: once a page reaches at/before entry_ts the whole
        # post-entry window is covered; a short page means no more trades.
        if oldest_on_page is not None and oldest_on_page <= entry_ts:
            break
        if len(trades) < limit:
            break
        offset += limit
    else:
        truncated = True

    rows.sort(key=lambda r: r[0])
    return [price for _ts, price in rows], truncated


def collect_samples(
    client: PolymarketDataAPIClient,
    *,
    top_wallets: int,
    max_positions_per_wallet: int,
    max_samples: int,
    window: str,
) -> List[Dict[str, Any]]:
    """Assemble the backtest dataset from real resolved winner positions.

    Pulls the top ``top_wallets`` profit-leaderboard wallets, and for each (bounded
    by ``max_positions_per_wallet`` and the global ``max_samples`` cap) reconstructs
    its resolved positions into ``{'entry_price', 'price_path', 'final_price'}``
    samples. Read-only and bounded; prints progress as it goes.
    """
    print(f"[1/2] fetching profit leaderboard (window={window!r}, top={top_wallets}) ...")
    try:
        rows = client.get_profit_leaderboard(window=window, limit=top_wallets)
    except PolymarketDataAPIError as exc:
        print(f"  leaderboard fetch failed: {exc}")
        return []

    wallets: List[str] = []
    for row in rows:
        wallet = row.get("proxyWallet") if isinstance(row, dict) else None
        if wallet and wallet not in wallets:
            wallets.append(wallet)
        if len(wallets) >= top_wallets:
            break
    print(f"  got {len(wallets)} wallet(s)")

    samples: List[Dict[str, Any]] = []
    n_truncated = 0
    print(f"[2/2] reconstructing resolved positions (cap {max_samples} samples) ...")
    for w_idx, wallet in enumerate(wallets):
        if len(samples) >= max_samples:
            break
        try:
            positions = client.get_positions(user=wallet)
        except PolymarketDataAPIError as exc:
            print(f"  wallet {w_idx + 1}/{len(wallets)} {wallet[:10]}.. positions failed: {exc}")
            continue

        kept_this_wallet = 0
        for position in positions:
            if len(samples) >= max_samples or kept_this_wallet >= max_positions_per_wallet:
                break
            if not isinstance(position, dict):
                continue
            if not _is_resolved(position):
                continue
            try:
                entry_price = float(position.get("avgPrice"))
            except (TypeError, ValueError):
                continue
            if entry_price <= 0:
                continue
            condition_id = position.get("conditionId")
            outcome_index = position.get("outcomeIndex")
            if not condition_id or outcome_index is None:
                continue

            entry_ts = _entry_ts(client, wallet, condition_id, outcome_index)
            if entry_ts is None:
                continue
            path, path_truncated = _forward_path(
                client, condition_id, outcome_index, entry_ts
            )
            if not path:
                continue
            if path_truncated:
                n_truncated += 1

            samples.append(
                {
                    "entry_price": entry_price,
                    "price_path": path,
                    "final_price": _final_price(position),
                    # Bookkeeping (not used by grid_search; aids inspection):
                    "wallet": wallet,
                    "conditionId": condition_id,
                    "outcomeIndex": outcome_index,
                    "title": position.get("title"),
                }
            )
            kept_this_wallet += 1

        print(
            f"  wallet {w_idx + 1}/{len(wallets)} {wallet[:10]}.. "
            f"kept {kept_this_wallet} resolved position(s); total samples={len(samples)}"
        )

    if n_truncated:
        print(
            f"  NOTE: {n_truncated}/{len(samples)} sample path(s) hit the page cap "
            f"-- may miss the earliest post-entry prices (backtest bias, can "
            f"understate stop-loss; not look-ahead)."
        )
    return samples


def _fmt_combo(combo: Optional[Dict[str, Any]]) -> str:
    """One-line human summary of a combo aggregate."""
    if combo is None:
        return "(none)"
    return (
        f"sl={combo['stop_loss_pct']} tp%={combo['take_profit_pct']} "
        f"tp$={combo['take_profit_price']} | "
        f"n={combo['n']} mean={combo['mean_return']:+.4f} "
        f"win={combo['win_rate']:.1%} worst={combo['worst_return']:+.4f} "
        f"mix={combo['exit_mix']}"
    )


def run(args: argparse.Namespace) -> int:
    """Collect the dataset, run the grid search, and print the verdict."""
    client = PolymarketDataAPIClient()

    samples = collect_samples(
        client,
        top_wallets=args.top_wallets,
        max_positions_per_wallet=args.max_positions_per_wallet,
        max_samples=args.max_samples,
        window=args.window,
    )

    print()
    print(f"dataset: {len(samples)} resolved-position sample(s)")
    if not samples:
        print("no usable samples (no post-entry price paths); nothing to optimize.")
        return 1

    result = grid_search(
        samples,
        sl_values=SL_VALUES,
        tp_pct_values=TP_PCT_VALUES,
        tp_price_values=TP_PRICE_VALUES,
        fee_bps=args.fee_bps,
    )

    baseline = result["baseline"]
    print()
    print("BASELINE (hold to resolution, no overlay):")
    print(f"  {_fmt_combo(baseline)}")

    print()
    print("TOP 8 combos by mean_return:")
    for combo in result["combos"][:8]:
        print(f"  {_fmt_combo(combo)}")

    print()
    print("RECOMMENDED (best risk_adjusted: max mean_return with worst_return >= -0.6):")
    print(f"  {_fmt_combo(result['risk_adjusted'])}")
    rec = result["risk_adjusted"]
    if rec is not None and baseline["n"]:
        delta = rec["mean_return"] - baseline["mean_return"]
        print(f"  improvement over baseline mean_return: {delta:+.4f}")

    if args.out:
        out_path = Path(args.out)
        payload = {
            "window": args.window,
            "fee_bps": args.fee_bps,
            "n_samples": len(samples),
            "baseline": baseline,
            "best_by_meanret": result["best_by_meanret"],
            "risk_adjusted": result["risk_adjusted"],
            "combos": result["combos"],
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {out_path}")

    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize whale-follow stop-loss / take-profit on REAL resolved "
            "winner positions (READ-ONLY; no orders)."
        )
    )
    parser.add_argument("--top-wallets", type=int, default=25)
    parser.add_argument("--max-positions-per-wallet", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=250)
    parser.add_argument("--window", default="all")
    parser.add_argument("--fee-bps", type=float, default=200.0)
    parser.add_argument("--out", default=None, help="optional JSON path for full results")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
