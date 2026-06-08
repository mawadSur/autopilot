"""Historical read of the FILTERED whale-follow strategy on REAL resolved data,
with win/loss labeled from the **CANONICAL market resolution** (not the wallet's
own realized PnL).

WHY THIS EXISTS (the fix over scripts/backtest_filtered.py):
    ``backtest_filtered.py`` labels a resolved position a WIN iff ``realizedPnl
    > 0`` because, on the data-api /positions snapshot, EVERY resolved position
    reports ``curPrice == 0`` (winners included), so curPrice cannot encode the
    binary outcome. But ``realizedPnl`` UNDERCOUNTS wins: a whale who partially
    sold a winning side before resolution books most of its P/L on the SALE and
    shows ``realizedPnl ~ 0`` on the leftover redeemable dust -> it is miscounted
    NOT-a-win even though its outcome resolved its way. So that script's per-band
    win-rates are a downward-biased FLOOR and it cannot settle the keep/kill gate.

THE FIX (this script):
    Label win/loss from the TRUE market resolution via
    ``polymarket_market_data.get_market_resolution(condition_id)``, which returns
    ``{"closed": bool, "tokens": [{"outcome","price","winner"}, ...]}`` with
    ``tokens`` ordered by outcomeIndex and ``winner`` the TRUE binary outcome
    (the same live-verified read shadow settlement trusts). A position
    ``(conditionId, outcomeIndex)`` WON iff the market is closed AND
    ``tokens[outcomeIndex]["winner"]`` is true. This is the canonical outcome, so
    the win-rate-vs-line table below is trustworthy (modulo the caveats stamped
    at the top of the output).

SCOPE — READ-ONLY, NO MONEY MOVES (Constitution: safety first):
    This script issues ONLY public data reads:
      * ``PolymarketDataAPIClient.get_profit_leaderboard`` / ``get_positions``
        (wallet roster + their resolved positions); and
      * ``polymarket_market_data.get_market_resolution`` (the canonical outcome),
        called EXACTLY ONCE per unique conditionId and cached.
    It places no orders, signs nothing, redeems nothing, and touches no
    wallet/web3 path. It writes a single JSON report under ``runs/``. No money
    moves. It does not modify or run any other module.

The FILTERED set mirrors the LIVE whale-follow runner's CLI defaults
(``src/whale_follow_runner.py``), matching ``backtest_filtered.py``:
  * ENTRY-BAND ``[0.15, 0.85]`` on ``avgPrice`` (the live --min/--max-entry-price);
  * the do-not-trade BLOCKLIST (``src/trade_blocklist.py``); and
  * the DIRECTIONAL / "both-sides" read: drop any ``conditionId`` where the roster
    held MORE THAN ONE outcome (the live --allow-split-markets default-OFF rule).
The live min-convergence default is 3 — a multi-wallet co-entry test that a
resolved-holdings snapshot cannot reconstruct; we record it and do NOT fabricate
a convergence count (see CAVEAT 3).

Win / entry / fee conventions (comparable to backtest_filtered.py + the shadow ledger):
  * RESOLVED = ``redeemable is True`` (``wallet_ranker._is_settled``) — the
    collection gate. (We then re-verify the TRUE outcome per market below; a
    redeemable position whose market get_market_resolution cannot settle is
    dropped + counted as UNRESOLVABLE, never guessed.)
  * WIN = canonical: ``get_market_resolution(conditionId)`` is not None AND
    ``closed`` is True AND ``0 <= outcomeIndex < len(tokens)`` AND
    ``tokens[outcomeIndex]["winner"]`` is true. (NOT realizedPnl — that's the bug
    this script fixes.)
  * ENTRY = ``avgPrice`` (dollars per outcome share paid);
  * FEE = 200 bps (~2%) settlement haircut on PROCEEDS: a win nets
    ``(1/entry)*1.0*(1 - 0.02) - 1`` (held-to-resolution payout at $1, net fee);
    a loss redeems at $0 -> ``-1.0`` (full stake forfeit, no fee on a $0 redeem).

CAVEATS the OUTCOME fix does NOT remove (stamped at the top of the output + JSON):
    (1) SURVIVORSHIP — the leaderboard is TODAY's winners; their PAST resolved
        positions are win-biased, so every number here is OPTIMISTIC.
    (2) NO HISTORICAL ORDER BOOK — the CLOB book is current-only, so the
        book-aware W1/W2 fills CANNOT be backtested; this tests the SIGNAL +
        entry-band at the resolved OUTCOME only, not fill quality.
    (3) CONVERGENCE TIMING is APPROXIMATED by resolved holdings, not true
        multi-wallet co-entry at a point in time.
    => With the OUTCOME now canonical, the win-rate-vs-line IS trustworthy modulo
       (1)-(3): it answers "does the filtered signal beat the line at resolution?"
       (the first kill-gate) but NOT "are the fills achievable."
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
from exchanges.polymarket_market_data import get_market_resolution  # noqa: E402
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
    "OUTCOME IS NOW CANONICAL: win/loss is labeled from get_market_resolution "
    "(tokens[outcomeIndex]['winner'] on a CLOSED market), NOT realizedPnl. This "
    "REMOVES the partial-sale win-undercount that made backtest_filtered.py's "
    "per-band win-rates a downward-biased floor. The win-rate-vs-line below is "
    "trustworthy modulo the caveats that follow.",
    "SURVIVORSHIP: the leaderboard is TODAY's winners; their PAST resolved "
    "positions are win-biased -> every number here is OPTIMISTIC.",
    "NO HISTORICAL ORDER BOOK: the CLOB book is current-only, so book-aware "
    "W1/W2 fills CANNOT be backtested; this tests the SIGNAL + entry-band at the "
    "resolved OUTCOME only, not fill quality.",
    "CONVERGENCE TIMING approximated by current/resolved holdings, not true "
    "multi-wallet co-entry at a point in time.",
    "COVERAGE GAP (not a bias): markets get_market_resolution returns None / "
    "not-closed for, or whose outcomeIndex is out of range, are UNRESOLVABLE and "
    "DROPPED + COUNTED (never guessed). Check n_unresolvable vs the resolved n.",
    "=> Sufficient to answer 'does the filtered signal beat the line at "
    "resolution?' (first kill-gate) but NOT 'are the fills achievable.'",
]


# ----------------------------------------------------------------------------
# Resolved-position COLLECTION helpers — reused verbatim in spirit from
# scripts/backtest_filtered.py. The RESOLVED (redeemable) gate is the same; the
# WIN label is the only thing this script changes (canonical resolution, below).
# ----------------------------------------------------------------------------

def _is_resolved(position: Dict[str, Any]) -> bool:
    """RESOLVED iff ``redeemable is True`` (== ``wallet_ranker._is_settled``)."""
    return position.get("redeemable") is True


def _entry_price(position: Dict[str, Any]) -> Optional[float]:
    """``avgPrice`` as a float in (0, 1], else None (unsizable / unparseable)."""
    try:
        entry = float(position.get("avgPrice"))
    except (TypeError, ValueError):
        return None
    if entry <= 0.0 or entry > 1.0:
        return None
    return entry


def _outcome_index(position: Dict[str, Any]) -> Optional[int]:
    """``outcomeIndex`` coerced to int, or None when missing/unparseable."""
    raw = position.get("outcomeIndex")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _return_on_cost(entry_price: float, won: bool, fee_bps: float) -> float:
    """Realized return per $1 of cost held to resolution, net of the fee.

    The ``exit_backtest._return_on_cost`` convention applied to a BINARY outcome:
    a win redeems at $1 -> ``(1/entry)*1.0*(1 - fee) - 1``; a loss redeems at $0
    -> ``-1.0`` (full stake forfeit, the fee never applies to a $0 redemption).
    """
    final_price = 1.0 if won else 0.0
    return (1.0 / entry_price) * final_price * (1.0 - fee_bps / 10_000.0) - 1.0


# ----------------------------------------------------------------------------
# Collection: top profit wallets -> their RESOLVED positions (outcome TBD).
# Mirrors backtest_filtered.collect_resolved_positions, MINUS the realizedPnl
# win label (we keep the raw realizedPnl only for reference; the WIN is decided
# later from the canonical resolution).
# ----------------------------------------------------------------------------

def collect_resolved_positions(
    client: PolymarketDataAPIClient,
    *,
    top_wallets: int,
    window: str,
    sleep_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pull the top profit wallets and collect their RESOLVED positions.

    Per-wallet try/except: an API error skips that wallet (the scan continues),
    and several consecutive failures back off + stop early (rate-limit guard).
    Returns ``(rows, meta)``; each row is a flattened resolved position whose
    WIN is NOT yet decided (that comes from the canonical resolution).
    """
    meta: Dict[str, Any] = {
        "leaderboard_window": window,
        "wallets_requested": top_wallets,
        "wallets_returned": 0,
        "wallets_sampled": 0,
        "wallets_errored": 0,
        "api_notes": [],
    }

    print(f"[1/3] fetching profit leaderboard (window={window!r}, top={top_wallets}) ...")
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
    print(f"[2/3] collecting resolved positions from {len(wallets)} wallet(s) ...")
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
            outcome_index = _outcome_index(position)
            if not condition_id or outcome_index is None:
                continue
            # realizedPnl kept ONLY for reference / cross-check vs the old method;
            # it is NOT the win signal here (that's the canonical resolution).
            try:
                realized = float(position.get("realizedPnl"))
            except (TypeError, ValueError):
                realized = None
            rows.append(
                {
                    "wallet": wallet,
                    "conditionId": condition_id,
                    "outcomeIndex": outcome_index,
                    "outcome": position.get("outcome"),
                    "title": position.get("title"),
                    "entry_price": entry,
                    "size": position.get("size"),
                    "realizedPnl": realized,
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
# TRUE OUTCOME: dedupe unique conditionIds, call get_market_resolution ONCE per
# market, cache, and label each position WON/UNRESOLVABLE canonically.
# ----------------------------------------------------------------------------

def resolve_outcomes(
    rows: Sequence[Dict[str, Any]],
    *,
    sleep_s: float,
    meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], int, Dict[str, Optional[Dict[str, Any]]]]:
    """Label each row's WIN from the canonical market resolution.

    Dedupes the unique conditionIds and calls ``get_market_resolution`` EXACTLY
    ONCE per market (cached in ``resolution_cache``). A position WON iff the
    market resolution is not None AND ``closed`` is True AND
    ``0 <= outcomeIndex < len(tokens)`` AND ``tokens[outcomeIndex]['winner']``.

    Markets that are None / not closed / index-out-of-range are UNRESOLVABLE:
    those positions are DROPPED and COUNTED (never guessed). Returns
    ``(labeled_rows, n_unresolvable, resolution_cache)``.
    """
    unique_cids: List[str] = []
    seen = set()
    for row in rows:
        cid = row["conditionId"]
        if cid not in seen:
            seen.add(cid)
            unique_cids.append(cid)

    print(
        f"[3/3] resolving TRUE outcomes for {len(unique_cids)} unique market(s) "
        f"({len(rows)} positions) via get_market_resolution (cached, 1 call/market) ..."
    )

    resolution_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    n_calls = 0
    n_closed = 0
    n_none = 0
    n_not_closed = 0
    for i, cid in enumerate(unique_cids):
        try:
            res = get_market_resolution(cid)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully; never crash the sweep.
            res = None
            if "RESOLUTION CALL RAISED" not in " ".join(meta["api_notes"]):
                meta["api_notes"].append(
                    f"RESOLUTION CALL RAISED at least once (e.g. {exc!r}); "
                    f"those markets counted UNRESOLVABLE."
                )
        n_calls += 1
        resolution_cache[cid] = res
        if res is None:
            n_none += 1
        elif not res.get("closed"):
            n_not_closed += 1
        else:
            n_closed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
        if (i + 1) % 100 == 0:
            print(
                f"  resolved {i + 1}/{len(unique_cids)} markets "
                f"(closed={n_closed}, not_closed={n_not_closed}, none={n_none}) ..."
            )

    meta["resolution_calls"] = n_calls
    meta["unique_markets"] = len(unique_cids)
    meta["markets_closed"] = n_closed
    meta["markets_not_closed"] = n_not_closed
    meta["markets_none"] = n_none
    print(
        f"  done: {n_calls} resolution call(s) -> closed={n_closed}, "
        f"not_closed={n_not_closed}, none={n_none}"
    )

    labeled: List[Dict[str, Any]] = []
    n_unresolvable = 0
    n_index_oob = 0
    for row in rows:
        res = resolution_cache.get(row["conditionId"])
        if res is None or not res.get("closed"):
            n_unresolvable += 1
            continue
        tokens = res.get("tokens") or []
        idx = row["outcomeIndex"]
        if not (0 <= idx < len(tokens)):
            n_unresolvable += 1
            n_index_oob += 1
            continue
        won = bool(tokens[idx].get("winner"))
        labeled.append({**row, "won": won})

    if n_index_oob:
        meta["api_notes"].append(
            f"{n_index_oob} position(s) had outcomeIndex out of range vs the "
            f"resolution token list -> counted UNRESOLVABLE."
        )
    print(
        f"  labeled {len(labeled)} resolvable position(s); "
        f"{n_unresolvable} UNRESOLVABLE (dropped + counted)."
    )
    return labeled, n_unresolvable, resolution_cache


# ----------------------------------------------------------------------------
# Filtering + stats — reused verbatim from scripts/backtest_filtered.py.
# ----------------------------------------------------------------------------

def _split_markets(rows: Sequence[Dict[str, Any]]) -> set:
    """conditionIds where the roster held MORE THAN ONE distinct outcome.

    Mirrors the live directional-read filter (whale_follow_runner: drop a market
    where convergence fired on >1 outcome). The resolved-holdings proxy: the
    roster collectively held both sides of this conditionId at some point.
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
    the band's mean net-fee realized return. With the outcome now canonical, a
    POSITIVE edge_vs_line means the signal's outcomes genuinely beat what the
    entry price implied; a POSITIVE mean net-fee return is "beats the line AFTER
    the 2% settlement cost." Bands are half-open [lo, hi) except the top band,
    which includes its ceiling so an entry exactly at MAX_ENTRY_PRICE (0.85) is
    not dropped.
    """
    table: List[Dict[str, Any]] = []
    for lo, hi in BANDS:
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
    print("FILTERED WHALE-FOLLOW — RESOLVED-OUTCOME READ (canonical win/loss)")
    print(bar)
    print("CAVEATS (read first — outcome is now canonical; these remain):")
    for c in report["caveats"]:
        print(f"  * {c}")
    print(bar)

    print(
        f"Sampling: window={meta['leaderboard_window']!r}  "
        f"wallets_returned={meta['wallets_returned']}  "
        f"wallets_sampled={meta['wallets_sampled']}  "
        f"wallets_errored={meta['wallets_errored']}"
    )
    print(
        f"Markets: unique={meta.get('unique_markets', 0)}  "
        f"resolution_calls={meta.get('resolution_calls', 0)}  "
        f"closed={meta.get('markets_closed', 0)}  "
        f"not_closed={meta.get('markets_not_closed', 0)}  "
        f"none={meta.get('markets_none', 0)}"
    )
    print(
        f"Positions: resolved_collected={report['n_positions_collected']}  "
        f"true_labeled={report['n_positions_resolved']}  "
        f"UNRESOLVABLE={report['n_unresolvable']} (dropped, coverage gap)"
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

    print("RAW vs FILTERED (true-labeled resolved positions, net 2% fee on wins):")
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

    print(f"VERDICT: {report['verdict_label'].upper()}")
    print(f"READ: {report['verdict']}")
    print(bar + "\n")


def _verdict(report: Dict[str, Any]) -> Tuple[str, str]:
    """Honest read + a one-word label on whether the FILTERED signal beats the
    line net of fees, now that the outcome is canonical.

    Returns ``(verdict_label, prose)`` where verdict_label is one of
    edge / no_edge / mixed / inconclusive.
    """
    filt = report["filtered"]
    if filt["n"] == 0:
        return (
            "inconclusive",
            "NO DATA in the filtered set — cannot judge edge. (Likely an "
            "API/rate-limit issue, too-narrow sample, or a coverage gap; see API "
            "notes + n_unresolvable.)",
        )
    bands = [b for b in report["win_rate_vs_line"] if b["count"] > 0]
    n_pos_edge = sum(1 for b in bands if b["edge_vs_line"] > 0)
    n_pos_ret = sum(1 for b in bands if b["mean_return_net_fee"] > 0)
    mean_ret = filt["mean_return_net_fee"]
    head = (
        f"FILTERED mean realized return net of fee = {mean_ret*100:.2f}% over "
        f"n={filt['n']} (canonical win-rate {filt['win_rate']*100:.1f}%). "
        f"{n_pos_edge}/{len(bands)} band(s) beat the line on win-rate; "
        f"{n_pos_ret}/{len(bands)} band(s) positive net-fee return. "
        f"Outcome is CANONICAL (get_market_resolution), so this is NOT the "
        f"downward-biased floor backtest_filtered.py reported. "
    )
    # Thresholds: an edge call needs the aggregate net-fee return positive AND a
    # majority of populated bands positive net-fee; a clean kill needs no band to
    # beat the line at all; everything between is mixed.
    if mean_ret > 0 and n_pos_ret >= max(1, (len(bands) + 1) // 2):
        label = "edge"
        tail = (
            "APPARENT EDGE AT RESOLUTION (canonical outcome). But CAVEAT 1 "
            "(survivorship inflates it) and CAVEAT 2 (no fill-quality test) mean "
            "this is a PASS of the first kill-gate, not yet proof of a tradeable "
            "edge — the next honest step is testing achievable fills, not more "
            "outcome labeling."
        )
    elif n_pos_edge == 0 and n_pos_ret == 0:
        label = "no_edge"
        tail = (
            "NO band beats the line on the CANONICAL outcome, even on this "
            "survivorship-biased best-case-fills dataset. The win-undercount is "
            "now fixed, so this is no longer a floor — it is a clean read that "
            "the filtered signal does NOT clear the first kill-gate. Real fills + "
            "non-survivor wallets are only worse."
        )
    else:
        label = "mixed"
        tail = (
            "MIXED: some bands beat the line on the canonical outcome and some do "
            "not, and the aggregate net-fee return is not cleanly positive. With "
            "the outcome now canonical this is a genuine split (not a labeling "
            "artifact): the positive-edge band(s) are where any real edge would "
            "live, but the signal does not clear the gate across the board. Test "
            "achievable fills on the surviving band(s) before trusting it."
        )
    return label, head + tail


def build_report(
    rows_collected: List[Dict[str, Any]],
    labeled_rows: List[Dict[str, Any]],
    n_unresolvable: int,
    meta: Dict[str, Any],
    blocklist: Sequence[str],
) -> Dict[str, Any]:
    filtered, drops = apply_filter(labeled_rows, blocklist)
    unique_markets_resolved = len({r["conditionId"] for r in labeled_rows})
    n_wallets = len({r["wallet"] for r in labeled_rows}) if labeled_rows else 0
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
        "n_positions_collected": len(rows_collected),
        "n_positions_resolved": len(labeled_rows),
        "n_unresolvable": n_unresolvable,
        "n_unique_markets": meta.get("unique_markets", 0),
        "n_unique_markets_resolved": unique_markets_resolved,
        "n_wallets": n_wallets,
        "raw": perf(labeled_rows),
        "filtered": perf(filtered),
        "filter_drops": drops,
        "win_rate_vs_line": win_rate_vs_line(filtered),
    }
    label, prose = _verdict(report)
    report["verdict_label"] = label
    report["verdict"] = prose
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY historical backtest of the FILTERED whale-follow signal on "
            "real resolved leaderboard-wallet positions, with win/loss labeled "
            "from the CANONICAL market resolution. Places NO orders."
        )
    )
    parser.add_argument(
        "--top-wallets", type=int, default=30,
        help="Top profit-leaderboard wallets to sample (default 30 — a SAMPLE; "
        "dedupe-by-market keeps total get_market_resolution calls manageable).",
    )
    parser.add_argument(
        "--window", type=str, default="all",
        help="Profit-leaderboard window: 'all' (all-time, default) or '1d'.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.0,
        help="Seconds to sleep between per-wallet /positions calls (default 0).",
    )
    parser.add_argument(
        "--resolution-sleep", type=float, default=0.05,
        help="Seconds to sleep between get_market_resolution calls (default 0.05 "
        "— a small courtesy delay; raise it if you hit CLOB rate limits).",
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
        "--out", type=str, default=str(_REPO_ROOT / "runs" / "backtest_resolved.json"),
        help="Where to write the JSON report (default runs/backtest_resolved.json).",
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
    meta["sample_note"] = (
        f"SAMPLE of the top {args.top_wallets} all-time profit wallets "
        f"(window={args.window!r}); a smaller wallet sample keeps "
        f"get_market_resolution calls to a manageable count via dedupe-by-market."
    )

    labeled_rows, n_unresolvable, _cache = resolve_outcomes(
        rows,
        sleep_s=args.resolution_sleep,
        meta=meta,
    )

    report = build_report(rows, labeled_rows, n_unresolvable, meta, blocklist)
    _print_report(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
