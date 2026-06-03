"""Historical read of the FILTERED whale-follow strategy on REAL resolved data.

Answers one question fast, without waiting days for the live shadow ledger to
accrue: **does the FILTERED whale-follow signal beat the line at resolution?**
It pulls the top all-time profit-leaderboard wallets, collects their RESOLVED
positions (outcome known), and reports RAW vs FILTERED performance plus the key
win-rate-vs-line table.

SCOPE — READ-ONLY, NO MONEY MOVES (Constitution: safety first):
    This script issues ONLY data reads through :class:`PolymarketDataAPIClient`
    (``get_profit_leaderboard`` / ``get_positions``). It places no orders, signs
    nothing, redeems nothing, and touches no wallet/web3 path. It writes a single
    JSON report under ``runs/``. Nothing here moves money.

The FILTERED set mirrors the LIVE whale-follow runner's CLI defaults
(``src/whale_follow_runner.py``), so the historical read matches the live filter:
  * ENTRY-BAND ``[0.15, 0.85]`` on ``avgPrice`` (the live --min/--max-entry-price);
  * the do-not-trade BLOCKLIST (``src/trade_blocklist.py``); and
  * the DIRECTIONAL / "both-sides" read: drop any ``conditionId`` where the roster
    held MORE THAN ONE outcome (the live --allow-split-markets default-OFF rule).
The live min-convergence default is 3, but that is a multi-wallet co-entry test
that this resolved-holdings snapshot cannot reconstruct in time (see CAVEAT 3); we
record the threshold for the record and do NOT fabricate a convergence count.

Win / loss / entry / fee conventions are taken verbatim from the existing stack so
the numbers are comparable to the shadow ledger:
  * RESOLVED = ``redeemable is True`` (``wallet_ranker._is_settled``). NOTE: the
    ``curPrice``-pinned-to-0/1 test from ``scripts/optimize_exits._is_resolved``
    is NOT used as the WIN signal here — see the next bullet for why.
  * WIN = ``realizedPnl > 0`` (``wallet_ranker.stats_from_positions``). This is
    the ONLY reliable resolved-outcome signal on the live data-api: on a
    live probe (2026-06-02) EVERY resolved position — winners included —
    reported ``curPrice == 0`` and ``percentPnl == -99.99``, so a
    ``curPrice``-based final-price (as in ``optimize_exits._final_price``, which
    only needs it for the price PATH, not the binary outcome) misclassifies ~all
    winners as losses. ``realizedPnl`` (fees already netted server-side) is the
    leakage-free win signal the ranker itself trusts.
  * ENTRY = ``avgPrice`` (dollars per outcome share paid);
  * FEE = 200 bps (~2%) settlement haircut on PROCEEDS, the
    ``exit_backtest._return_on_cost`` convention applied to the BINARY resolved
    outcome: a win nets ``(1/entry)*1.0*(1 - 0.02) - 1`` (held-to-resolution
    payout at $1, net fee); a loss redeems at $0 -> ``-1.0`` (full stake
    forfeit, no fee on a $0 redemption). The win-rate-vs-line comparison is on
    this binary outcome — exactly what the entry price (the "line") is pricing.

CAVEATS (also stamped at the top of the printed output and in the JSON):
    (1) SURVIVORSHIP — the leaderboard is TODAY's winners; their PAST resolved
        positions are win-biased, so every number here is OPTIMISTIC.
    (2) NO HISTORICAL ORDER BOOK — the CLOB book is current-only, so the
        book-aware W1/W2 fills CANNOT be backtested; this tests the SIGNAL +
        entry-band at the resolved OUTCOME only, not fill quality.
    (3) CONVERGENCE TIMING is APPROXIMATED by current/resolved holdings, not true
        multi-wallet co-entry at a point in time.
    => Sufficient to answer "does the filtered signal beat the line at
       resolution?" (the first kill-gate cut) but NOT "are the fills achievable."
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_SRC_DIR, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from exchanges.polymarket_data_api import (  # noqa: E402
    PolymarketDataAPIClient,
    PolymarketDataAPIError,
)
from trade_blocklist import blocked_term, load_blocklist  # noqa: E402


# --- LIVE thresholds, mirrored from src/whale_follow_runner.py CLI defaults ---
MIN_ENTRY_PRICE = 0.15          # --min-entry-price
MAX_ENTRY_PRICE = 0.85          # --max-entry-price
MIN_CONVERGENCE = 3             # --min-convergence (recorded; see CAVEAT 3)
FEE_BPS = 200.0                 # exit_backtest.DEFAULT_FEE_BPS (~2%)

# Entry-price bands for the win-rate-vs-line table. The first/last edges are the
# entry band; the interior splits long-shot / coin-flip / favorite / heavy-favorite.
BANDS: List[Tuple[float, float]] = [
    (0.15, 0.35),
    (0.35, 0.50),
    (0.50, 0.65),
    (0.65, 0.85),
]

CAVEATS = [
    "SURVIVORSHIP: the leaderboard is TODAY's winners; their PAST resolved "
    "positions are win-biased -> every number here is OPTIMISTIC.",
    "NO HISTORICAL ORDER BOOK: the CLOB book is current-only, so book-aware "
    "W1/W2 fills CANNOT be backtested; this tests the SIGNAL + entry-band at the "
    "resolved OUTCOME only, not fill quality.",
    "CONVERGENCE TIMING approximated by current/resolved holdings, not true "
    "multi-wallet co-entry at a point in time.",
    "WIN-RATE IS A FLOOR (data-shape limit): the live /positions snapshot reports "
    "curPrice=0 on ALL resolved rows (winners included), so the ONLY available "
    "binary signal is realizedPnl>0 (wallet_ranker's win def). But a position the "
    "wallet PARTIALLY SOLD before resolution books most of its P/L on the sale and "
    "shows realizedPnl~0 on the leftover redeemable dust -> it is counted NOT-a-win "
    "even if the outcome resolved its way. This UNDERCOUNTS wins, worst in the "
    "mid/high entry bands (where trimming a runner is common). So per-band win-rates "
    "are a conservative FLOOR, not the true hit-rate; treat low mid-band win-rates "
    "with suspicion. (A faithful hit-rate needs each position's full /trades "
    "history, as scripts/optimize_exits.py reconstructs — out of scope here.)",
    "=> Sufficient to answer 'does the filtered signal beat the line at "
    "resolution?' (first kill-gate, directionally) but NOT 'are the fills "
    "achievable' nor an exact hit-rate.",
]


# ----------------------------------------------------------------------------
# Resolved-position helpers. Win/settled semantics mirror src/wallet_ranker.py
# (the ranker's _is_settled + realizedPnl>0 win), NOT optimize_exits' curPrice
# logic — on the live data-api curPrice is 0 even for resolved WINNERS, so it
# cannot encode the binary outcome (see module docstring).
# ----------------------------------------------------------------------------

def _is_resolved(position: Dict[str, Any]) -> bool:
    """RESOLVED iff ``redeemable is True`` (== ``wallet_ranker._is_settled``).

    A missing/non-bool ``redeemable`` is treated as NOT resolved, so an
    ambiguous row never leaks an unknown outcome into the dataset.
    """
    return position.get("redeemable") is True


def _realized_pnl(position: Dict[str, Any]) -> Optional[float]:
    """``realizedPnl`` as a float, or None when missing/unparseable.

    Mirrors ``wallet_ranker._realized_pnl`` but returns None (not 0.0) on a bad
    value so the caller can DROP an unscorable row rather than count it a loss.
    """
    try:
        return float(position.get("realizedPnl"))
    except (TypeError, ValueError):
        return None


def _won(position: Dict[str, Any]) -> bool:
    """WIN == ``realizedPnl > 0`` (== ``wallet_ranker`` win definition).

    The only reliable resolved-outcome signal on the live data-api: curPrice is
    0 even on winners (see module docstring), so realizedPnl is what decides.
    """
    rp = _realized_pnl(position)
    return rp is not None and rp > 0.0


def _entry_price(position: Dict[str, Any]) -> Optional[float]:
    """``avgPrice`` as a float in (0, 1], else None (unsizable / unparseable)."""
    try:
        entry = float(position.get("avgPrice"))
    except (TypeError, ValueError):
        return None
    if entry <= 0.0 or entry > 1.0:
        return None
    return entry


def _return_on_cost(entry_price: float, won: bool, fee_bps: float) -> float:
    """Realized return per $1 of cost held to resolution, net of the fee.

    The ``exit_backtest._return_on_cost`` convention applied to a BINARY outcome:
    a win redeems at $1 -> ``(1/entry)*1.0*(1 - fee) - 1``; a loss redeems at $0
    -> ``-1.0`` (full stake forfeit, the fee never applies to a $0 redemption).
    """
    final_price = 1.0 if won else 0.0
    return (1.0 / entry_price) * final_price * (1.0 - fee_bps / 10_000.0) - 1.0


# ----------------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------------

def collect_resolved_positions(
    client: PolymarketDataAPIClient,
    *,
    top_wallets: int,
    window: str,
    sleep_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pull the top profit wallets and collect their RESOLVED positions.

    Per-wallet try/except: an API error skips that wallet (the scan continues).
    If the leaderboard itself rate-limits or errors, we return what we have and
    record it. Returns ``(rows, meta)`` where each row is a flattened resolved
    position and ``meta`` records sampling counts + any API limits hit.
    """
    meta: Dict[str, Any] = {
        "leaderboard_window": window,
        "wallets_requested": top_wallets,
        "wallets_returned": 0,
        "wallets_sampled": 0,
        "wallets_errored": 0,
        "api_notes": [],
    }

    print(f"[1/2] fetching profit leaderboard (window={window!r}, top={top_wallets}) ...")
    try:
        lb_rows = client.get_profit_leaderboard(window=window, limit=top_wallets)
    except PolymarketDataAPIError as exc:
        note = f"LEADERBOARD FETCH FAILED: {exc}"
        print(f"  {note}")
        meta["api_notes"].append(note)
        return [], meta

    wallets: List[str] = []
    for row in lb_rows:
        wallet = row.get("proxyWallet") if isinstance(row, dict) else None
        if wallet and wallet not in wallets:
            wallets.append(wallet)
        if len(wallets) >= top_wallets:
            break
    meta["wallets_returned"] = len(wallets)
    print(f"  got {len(wallets)} wallet(s)")

    rows: List[Dict[str, Any]] = []
    consecutive_errors = 0
    print(f"[2/2] collecting resolved positions from {len(wallets)} wallet(s) ...")
    for w_idx, wallet in enumerate(wallets):
        try:
            positions = client.get_positions(user=wallet)
        except PolymarketDataAPIError as exc:
            meta["wallets_errored"] += 1
            consecutive_errors += 1
            print(
                f"  wallet {w_idx + 1}/{len(wallets)} {wallet[:10]}.. "
                f"positions FAILED ({exc}); skipping"
            )
            # Back off and stop early if the API appears to be rate-limiting us
            # (several consecutive failures) — sample fewer wallets and say so.
            if consecutive_errors >= 4:
                note = (
                    f"STOPPED EARLY after {consecutive_errors} consecutive "
                    f"/positions failures (likely rate-limited); sampled "
                    f"{meta['wallets_sampled']}/{len(wallets)} wallet(s)."
                )
                print(f"  {note}")
                meta["api_notes"].append(note)
                break
            continue
        except Exception as exc:  # noqa: BLE001 - one bad wallet must not sink the run.
            meta["wallets_errored"] += 1
            consecutive_errors += 1
            print(
                f"  wallet {w_idx + 1}/{len(wallets)} {wallet[:10]}.. "
                f"positions RAISED ({exc!r}); skipping"
            )
            continue

        consecutive_errors = 0
        meta["wallets_sampled"] += 1
        kept = 0
        for position in positions or []:
            if not isinstance(position, dict):
                continue
            if not _is_resolved(position):
                continue
            entry = _entry_price(position)
            if entry is None:
                continue
            condition_id = position.get("conditionId")
            outcome_index = position.get("outcomeIndex")
            if not condition_id or outcome_index is None:
                continue
            # WIN is decided by realizedPnl (the only reliable resolved-outcome
            # signal on the live data-api); drop a row whose realizedPnl is
            # unparseable rather than count it a loss.
            rp = _realized_pnl(position)
            if rp is None:
                continue
            rows.append(
                {
                    "wallet": wallet,
                    "conditionId": condition_id,
                    "outcomeIndex": outcome_index,
                    "outcome": position.get("outcome"),
                    "title": position.get("title"),
                    "entry_price": entry,
                    "won": rp > 0.0,
                    "realizedPnl": rp,
                }
            )
            kept += 1
        print(
            f"  wallet {w_idx + 1}/{len(wallets)} {wallet[:10]}.. "
            f"kept {kept} resolved position(s); total={len(rows)}"
        )
        if sleep_s > 0:
            time.sleep(sleep_s)

    return rows, meta


# ----------------------------------------------------------------------------
# Filtering + stats
# ----------------------------------------------------------------------------

def _split_markets(rows: Sequence[Dict[str, Any]]) -> set:
    """conditionIds where the roster held MORE THAN ONE distinct outcome.

    Mirrors the live directional-read filter (whale_follow_runner: drop a market
    where convergence fired on >1 outcome). Here, the resolved-holdings proxy is:
    the roster collectively held both sides of this conditionId at some point.
    """
    outcomes_by_market: Dict[Any, set] = {}
    for row in rows:
        cid = row.get("conditionId")
        outcomes_by_market.setdefault(cid, set()).add(row.get("outcomeIndex"))
    return {cid for cid, outs in outcomes_by_market.items() if len(outs) > 1}


def apply_filter(
    rows: Sequence[Dict[str, Any]],
    blocklist: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Keep only positions passing the LIVE filter; return (kept, drop_counts).

    Drops, each counted (never silent):
      * entry outside [MIN_ENTRY_PRICE, MAX_ENTRY_PRICE];
      * title/outcome on the do-not-trade blocklist;
      * any position in a SPLIT market (roster held both sides) — directional read.
    """
    split = _split_markets(rows)
    counts = {"out_of_band": 0, "blocked": 0, "split_market": 0}
    kept: List[Dict[str, Any]] = []
    for row in rows:
        entry = row["entry_price"]
        if not (MIN_ENTRY_PRICE <= entry <= MAX_ENTRY_PRICE):
            counts["out_of_band"] += 1
            continue
        if blocklist:
            text = f"{row.get('title') or ''} {row.get('outcome') or ''}"
            if blocked_term(text, blocklist):
                counts["blocked"] += 1
                continue
        if row.get("conditionId") in split:
            counts["split_market"] += 1
            continue
        kept.append(row)
    return kept, counts


def perf(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """n, win-rate, mean realized return on cost NET of the settlement fee."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "win_rate": 0.0, "mean_return_net_fee": 0.0}
    wins = sum(1 for r in rows if r["won"])
    returns = [
        _return_on_cost(r["entry_price"], r["won"], FEE_BPS) for r in rows
    ]
    return {
        "n": n,
        "win_rate": wins / n,
        "mean_return_net_fee": sum(returns) / n,
    }


def win_rate_vs_line(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """THE KEY METRIC: per entry-band, win-rate vs the line's implied probability.

    For each band [lo, hi): count, realized win-rate, implied prob (the band
    midpoint = what the "line" charged), edge_vs_line = win_rate - implied, and
    the band's mean net-fee realized return. A POSITIVE edge_vs_line means the
    signal's outcomes beat what the entry price implied; a POSITIVE mean net-fee
    return is the honest "beats the line AFTER the 2% settlement cost" signal.
    Bands are half-open [lo, hi) except the top band, which includes its ceiling
    so an entry exactly at MAX_ENTRY_PRICE (0.85) is not dropped.
    """
    table: List[Dict[str, Any]] = []
    for lo, hi in BANDS:
        # Half-open [lo, hi) except the top band includes its ceiling so 0.85
        # entries (the band edge) are not dropped.
        in_band = [
            r for r in rows
            if lo <= r["entry_price"] < hi
            or (hi == BANDS[-1][1] and r["entry_price"] == hi)
        ]
        n = len(in_band)
        wins = sum(1 for r in in_band if r["won"])
        win_rate = (wins / n) if n else 0.0
        implied = (lo + hi) / 2.0
        returns = [
            _return_on_cost(r["entry_price"], r["won"], FEE_BPS)
            for r in in_band
        ]
        mean_ret = (sum(returns) / n) if n else 0.0
        table.append(
            {
                "band": f"{lo:.2f}-{hi:.2f}",
                "lo": lo,
                "hi": hi,
                "count": n,
                "wins": wins,
                "win_rate": win_rate,
                "implied_prob": implied,
                "edge_vs_line": win_rate - implied,
                "mean_return_net_fee": mean_ret,
            }
        )
    return table


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def _print_report(report: Dict[str, Any]) -> None:
    raw = report["raw"]
    filt = report["filtered"]
    drops = report["filter_drops"]
    meta = report["meta"]

    bar = "=" * 78
    print("\n" + bar)
    print("FILTERED WHALE-FOLLOW — HISTORICAL READ ON REAL RESOLVED DATA")
    print(bar)
    print("CAVEATS (read these first — the numbers are optimistic by construction):")
    for c in report["caveats"]:
        # wrap long caveats roughly to width
        print(f"  * {c}")
    print(bar)

    print(
        f"Sampling: window={meta['leaderboard_window']!r}  "
        f"wallets_returned={meta['wallets_returned']}  "
        f"wallets_sampled={meta['wallets_sampled']}  "
        f"wallets_errored={meta['wallets_errored']}"
    )
    if meta["api_notes"]:
        for note in meta["api_notes"]:
            print(f"  API NOTE: {note}")
    print(
        f"Live thresholds mirrored: entry-band [{MIN_ENTRY_PRICE}, {MAX_ENTRY_PRICE}]; "
        f"min-convergence={MIN_CONVERGENCE} (recorded, not reconstructable — caveat 3); "
        f"fee={FEE_BPS:.0f}bps; blocklist={'ON' if report['blocklist_active'] else 'OFF'}"
    )
    print(bar)

    print("RAW vs FILTERED (resolved positions held to resolution, net 2% fee on wins):")
    print(f"  {'set':<10}{'n':>8}{'win_rate':>12}{'mean_ret_net_fee':>20}")
    print(
        f"  {'RAW':<10}{raw['n']:>8}{raw['win_rate']*100:>11.1f}%"
        f"{raw['mean_return_net_fee']*100:>19.2f}%"
    )
    print(
        f"  {'FILTERED':<10}{filt['n']:>8}{filt['win_rate']*100:>11.1f}%"
        f"{filt['mean_return_net_fee']*100:>19.2f}%"
    )
    print(
        f"  filter dropped: {drops['out_of_band']} out-of-band, "
        f"{drops['blocked']} blocked, {drops['split_market']} split-market"
    )
    print(bar)

    print("KEY METRIC — WIN-RATE vs LINE (FILTERED set, by entry-price band):")
    print(
        f"  {'band':<12}{'count':>7}{'win_rate':>11}{'implied':>10}"
        f"{'edge(wr-impl)':>16}{'mean_ret_netfee':>18}"
    )
    for b in report["win_rate_vs_line"]:
        edge = b["edge_vs_line"]
        flag = "  <-- beats line" if edge > 0 else ""
        print(
            f"  {b['band']:<12}{b['count']:>7}{b['win_rate']*100:>10.1f}%"
            f"{b['implied_prob']*100:>9.1f}%{edge*100:>15.1f}pp"
            f"{b['mean_return_net_fee']*100:>17.2f}%{flag}"
        )
    print(bar)

    verdict = report["verdict"]
    print(f"READ: {verdict}")
    print(bar + "\n")


def _verdict(report: Dict[str, Any]) -> str:
    """One-line honest read on whether the FILTERED signal shows edge net of fees."""
    filt = report["filtered"]
    if filt["n"] == 0:
        return (
            "NO DATA in the filtered set — cannot judge edge. "
            "(Likely an API/rate-limit issue or too-narrow sample; see API notes.)"
        )
    bands = [b for b in report["win_rate_vs_line"] if b["count"] > 0]
    n_pos_edge = sum(1 for b in bands if b["edge_vs_line"] > 0)
    n_pos_ret = sum(1 for b in bands if b["mean_return_net_fee"] > 0)
    mean_ret = filt["mean_return_net_fee"]
    head = (
        f"FILTERED mean realized return net of fee = {mean_ret*100:.2f}% over "
        f"n={filt['n']} (win-rate {filt['win_rate']*100:.1f}%, a FLOOR per CAVEAT 4). "
        f"{n_pos_edge}/{len(bands)} band(s) beat the line on win-rate; "
        f"{n_pos_ret}/{len(bands)} band(s) positive net-fee return. "
    )
    if mean_ret > 0 and n_pos_ret >= max(1, len(bands) // 2):
        tail = (
            "Apparent edge AT RESOLUTION — but CAVEAT 1 (survivorship inflates it) "
            "and CAVEAT 2 (no fill-quality test) mean this is at most a PASS of the "
            "first kill-gate, not proof of a tradeable edge."
        )
    elif n_pos_edge > 0:
        tail = (
            "MIXED: most bands look like NO edge net of fees, BUT the win-rate is a "
            "downward-biased FLOOR (CAVEAT 4: partially-sold winners miscounted as "
            "losses), so the true per-band hit-rate is higher than shown and the "
            "loss is overstated. Any positive-edge band that SURVIVES that bias is "
            "the only place to keep looking. NOT a clean kill, NOT a clean pass — "
            "the /positions snapshot cannot settle this; a /trades-reconstructed "
            "hit-rate (per band) is the next honest step."
        )
    else:
        tail = (
            "NO band beats the line even before correcting the win-undercount, on "
            "this survivorship-biased, best-case-fills dataset — the filtered signal "
            "does NOT clear the first kill-gate. Real fills + non-winners are worse."
        )
    return head + tail


def build_report(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
    blocklist: Sequence[str],
) -> Dict[str, Any]:
    filtered, drops = apply_filter(rows, blocklist)
    report: Dict[str, Any] = {
        "caveats": CAVEATS,
        "meta": meta,
        "blocklist_active": bool(blocklist),
        "thresholds": {
            "min_entry_price": MIN_ENTRY_PRICE,
            "max_entry_price": MAX_ENTRY_PRICE,
            "min_convergence": MIN_CONVERGENCE,
            "fee_bps": FEE_BPS,
        },
        "raw": perf(rows),
        "filtered": perf(filtered),
        "filter_drops": drops,
        "win_rate_vs_line": win_rate_vs_line(filtered),
    }
    report["verdict"] = _verdict(report)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY historical backtest of the FILTERED whale-follow signal on "
            "real resolved leaderboard-wallet positions. Places NO orders."
        )
    )
    parser.add_argument(
        "--top-wallets", type=int, default=50,
        help="Top profit-leaderboard wallets to sample (default 50).",
    )
    parser.add_argument(
        "--window", type=str, default="all",
        help="Profit-leaderboard window: 'all' (all-time, default) or '1d'.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Seconds to sleep between per-wallet /positions calls (default 0; "
        "raise it if you hit rate limits).",
    )
    parser.add_argument(
        "--blocklist-file", type=str, default=None,
        help="Extra do-not-trade terms file (on top of the built-in defaults).",
    )
    parser.add_argument(
        "--no-blocklist", action="store_true",
        help="Disable the do-not-trade blocklist for the filtered set.",
    )
    parser.add_argument(
        "--out", type=str, default=str(_REPO_ROOT / "runs" / "backtest_filtered.json"),
        help="Where to write the JSON report (default runs/backtest_filtered.json).",
    )
    args = parser.parse_args(argv)

    blocklist = [] if args.no_blocklist else load_blocklist(args.blocklist_file)
    if blocklist:
        print(f"do-not-trade blocklist active: {len(blocklist)} term(s)")

    client = PolymarketDataAPIClient()
    rows, meta = collect_resolved_positions(
        client,
        top_wallets=args.top_wallets,
        window=args.window,
        sleep_s=args.sleep,
    )

    report = build_report(rows, meta, blocklist)
    _print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
