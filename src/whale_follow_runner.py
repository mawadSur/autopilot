"""Whale-convergence shadow runner (SHADOW-ONLY — NO ORDERS).

The acting half of the "smart-money wallet follower" edge. Given a roster of
"smart-money" target wallets (e.g. the output of ``wallet_ranker.rank_wallets``)
and a set of markets to watch, this module detects *convergence*: when several
target wallets all hold the SAME outcome of the same market at once. Each
convergence is logged to the shadow PnL ledger as an open ``whale_convergence``
candidate so the edge can be forward-validated before any capital is risked.

SCOPE — SHADOW-ONLY, NO MONEY MOVES (Constitution: safety first):
    This runner reads the data-api ``/holders`` endpoint and WRITES candidates
    to the PnL ledger. It places NO orders, signs nothing, and never touches a
    wallet/web3 path. The injected client is expected to expose only read
    methods. Do NOT add any order-placement / execution path here.

Entry price:
    The ``/holders`` endpoint reports holder *amounts*, not prices, so there is
    no price on the convergence payload itself. By default we record
    ``entry_price = 0.0`` and flag it in the trade notes; a later settlement
    supplies the real outcome price. With ``mark_entry=True`` (the CLI default)
    we instead fetch the converged outcome's *current* market price from the
    ``/trades`` endpoint — the price you'd pay to follow now, a decision-time
    entry, NOT look-ahead — and record that as the entry. Either way the
    ``outcomeIndex=<n>`` marker in the notes lets :func:`make_whale_price_fn`
    re-price the open position to market for the portfolio report. We never
    fabricate a fill: when no current price is available we fall back to
    ``0.0`` (unmarkable), keeping the ledger honest per the no-look-ahead
    standard.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from state.pnl_ledger import DEFAULT_LEDGER_PATH, PnlLedger, TradeRecord
except Exception:  # pragma: no cover - import shim for alternate layouts.
    from src.state.pnl_ledger import (  # type: ignore
        DEFAULT_LEDGER_PATH,
        PnlLedger,
        TradeRecord,
    )

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from exchanges.polymarket_data_api import PolymarketDataAPIError
except Exception:  # pragma: no cover - import shim for alternate layouts.
    try:
        from src.exchanges.polymarket_data_api import (  # type: ignore
            PolymarketDataAPIError,
        )
    except Exception:  # pragma: no cover - keep module importable in minimal envs.
        class PolymarketDataAPIError(Exception):  # type: ignore
            """Fallback when the data-api client cannot be imported."""


__all__ = [
    "STRATEGY",
    "find_convergence",
    "convergence_from_positions",
    "compute_confidence",
    "run_once",
    "make_whale_price_fn",
    "main",
]

# Pulls the ``outcomeIndex=<n>`` marker the runner writes into a record's notes.
_OUTCOME_INDEX_RE = re.compile(r"outcomeIndex=(\d+)")


STRATEGY = "whale_convergence"


def find_convergence(
    client: Any,
    target_wallets: Sequence[str],
    *,
    markets_condition_ids: Sequence[str],
    min_convergence: int = 3,
) -> List[Dict[str, Any]]:
    """Flag markets where >= ``min_convergence`` target wallets share an outcome.

    For each market in ``markets_condition_ids`` we fetch ``/holders`` and, per
    outcome (``outcomeIndex``), count how many of the ``target_wallets`` appear
    among that outcome's holders. When the count meets ``min_convergence`` the
    market+outcome is a convergence candidate.

    Only wallets in ``target_wallets`` are counted — non-target holders are
    ignored — and a wallet is counted at most once per outcome even if it shows
    up in multiple holder rows.

    Args:
        client:                 A read-only data-api client exposing
                                ``get_holders(market_condition_id, limit=...)``.
        target_wallets:         The "smart-money" roster to watch for.
        markets_condition_ids:  ``conditionId`` strings of markets to scan.
        min_convergence:        Minimum distinct target holders on one outcome
                                to flag (default 3).

    Returns:
        A list of candidate dicts, each::

            {conditionId, outcomeIndex, outcome,
             n_target_holders, wallets:[addr, ...]}
    """
    target_set = {w for w in target_wallets if w}
    candidates: List[Dict[str, Any]] = []

    for condition_id in markets_condition_ids:
        if not condition_id:
            continue
        holder_groups = client.get_holders(condition_id)

        # outcomeIndex -> {"outcome": label, "wallets": ordered distinct set}
        per_outcome: Dict[int, Dict[str, Any]] = {}
        for group in holder_groups:
            if not isinstance(group, dict):
                continue
            for holder in group.get("holders") or []:
                if not isinstance(holder, dict):
                    continue
                wallet = holder.get("proxyWallet")
                if wallet not in target_set:
                    continue
                outcome_index = holder.get("outcomeIndex")
                if outcome_index is None:
                    continue
                bucket = per_outcome.setdefault(
                    outcome_index,
                    {"outcome": holder.get("name") or holder.get("outcome"),
                     "wallets": {}},
                )
                # name/outcome label may only appear on some rows; keep first seen.
                if bucket["outcome"] is None:
                    bucket["outcome"] = holder.get("name") or holder.get("outcome")
                bucket["wallets"][wallet] = None  # ordered distinct set.

        for outcome_index, bucket in per_outcome.items():
            wallets = list(bucket["wallets"].keys())
            if len(wallets) >= min_convergence:
                candidates.append(
                    {
                        "conditionId": condition_id,
                        "outcomeIndex": outcome_index,
                        "outcome": bucket["outcome"],
                        "n_target_holders": len(wallets),
                        "wallets": wallets,
                    }
                )

    return candidates


def _is_current_open_holding(position: Dict[str, Any]) -> bool:
    """Return True iff a position is a CURRENT open holding (not resolved).

    A current open holding has real notional still at risk:
      * ``size > 0`` — the wallet actually holds the outcome token now; and
      * NOT ``redeemable`` — a redeemable position is resolved/settled; and
      * ``0 < curPrice < 1`` — a resolved/settled outcome marks to exactly 0 or
        1, so a strict-interior price is the live, still-uncertain signal.

    Any missing/non-numeric field is treated as NOT a current holding, so an
    ambiguous row never produces a false convergence signal.
    """
    try:
        size = float(position.get("size"))
    except (TypeError, ValueError):
        return False
    if size <= 0:
        return False
    if position.get("redeemable") is True:
        return False
    try:
        cur_price = float(position.get("curPrice"))
    except (TypeError, ValueError):
        return False
    return 0.0 < cur_price < 1.0


def convergence_from_positions(
    client: Any,
    roster_wallets: Sequence[str],
    *,
    min_convergence: int = 3,
    return_stats: bool = False,
) -> Any:
    """Flag markets where >= ``min_convergence`` roster wallets CURRENTLY hold the
    same outcome, derived from each wallet's live ``/positions``.

    This is the leaderboard-mode counterpart to :func:`find_convergence`. Instead
    of scanning a fixed set of "hot" markets via ``/holders``, it asks each
    roster wallet what it is *holding right now* and looks for the same
    (market, outcome) showing up across several winners. That matters because an
    all-time profit legend may be dormant in today's hot markets — the only way
    to follow them is to read their current book.

    For each wallet in ``roster_wallets`` we fetch
    ``client.get_positions(user=wallet)`` and keep only CURRENT OPEN holdings
    (:func:`_is_current_open_holding`: ``size>0``, not ``redeemable``,
    ``0<curPrice<1``). Each surviving holding records the wallet under its
    ``(conditionId, outcomeIndex)`` key. A per-wallet fetch error is caught and
    that wallet is skipped (the scan continues) so one bad wallet can't sink it.

    A (market, outcome) held by ``>= min_convergence`` DISTINCT roster wallets is
    emitted as a candidate dict in the SAME shape :func:`find_convergence`
    returns::

        {conditionId, outcomeIndex, outcome, n_target_holders, wallets:[...]}

    so the downstream confidence + mark-to-market + logging path is unchanged.

    Args:
        client:           Read-only data-api client exposing
                          ``get_positions(user=..., limit=...)``.
        roster_wallets:   The winners' roster (e.g. profit-leaderboard wallets).
        min_convergence:  Min distinct roster wallets holding one outcome to flag.
        return_stats:     When True, ALSO return ``wallet_stats`` —
                          ``{wallet: WalletStats}`` built from the SAME positions
                          already fetched (no extra calls), so confidence can be
                          scored on each wallet's settled win-rate. A wallet that
                          errored or returned nothing is absent from the map.

    Returns:
        ``candidates`` (a list of candidate dicts), or
        ``(candidates, wallet_stats)`` when ``return_stats`` is True.
    """
    # Lazy import keeps this module importable without wallet_ranker in minimal
    # envs; the stats path only runs when return_stats is requested.
    stats_from_positions = None
    if return_stats:
        try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
            from wallet_ranker import stats_from_positions  # type: ignore
        except Exception:  # pragma: no cover - alternate layout shim.
            from src.wallet_ranker import stats_from_positions  # type: ignore

    roster = [w for w in roster_wallets if w]
    # (conditionId, outcomeIndex) -> {"outcome": label, "wallets": ordered set}
    per_market_outcome: Dict[tuple, Dict[str, Any]] = {}
    wallet_stats: Dict[str, Any] = {}

    for wallet in roster:
        try:
            positions = client.get_positions(user=wallet)
        except PolymarketDataAPIError as exc:
            logging.warning(
                "skipping wallet %s: /positions failed (%s)", wallet, exc
            )
            continue
        except Exception as exc:  # noqa: BLE001 - one bad wallet must not kill the scan.
            logging.warning(
                "skipping wallet %s: /positions raised (%s)", wallet, exc
            )
            continue

        positions = positions or []
        if return_stats and stats_from_positions is not None:
            wallet_stats[wallet] = stats_from_positions(wallet, positions)

        # A wallet is counted at most once per (market, outcome).
        seen_keys: set = set()
        for position in positions:
            if not isinstance(position, dict):
                continue
            if not _is_current_open_holding(position):
                continue
            condition_id = position.get("conditionId")
            outcome_index = position.get("outcomeIndex")
            if not condition_id or outcome_index is None:
                continue
            key = (condition_id, outcome_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bucket = per_market_outcome.setdefault(
                key,
                {"outcome": position.get("outcome"), "wallets": {}},
            )
            if bucket["outcome"] is None:
                bucket["outcome"] = position.get("outcome")
            bucket["wallets"][wallet] = None  # ordered distinct set.

    candidates: List[Dict[str, Any]] = []
    for (condition_id, outcome_index), bucket in per_market_outcome.items():
        wallets = list(bucket["wallets"].keys())
        if len(wallets) >= min_convergence:
            candidates.append(
                {
                    "conditionId": condition_id,
                    "outcomeIndex": outcome_index,
                    "outcome": bucket["outcome"],
                    "n_target_holders": len(wallets),
                    "wallets": wallets,
                }
            )

    if return_stats:
        return candidates, wallet_stats
    return candidates


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (decision/entry timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def _latest_price_for_outcome(
    client: Any,
    condition_id: str,
    outcome_index: int,
    *,
    limit: int = 100,
) -> Optional[float]:
    """Return the current market price of one outcome of a market.

    Fetches a page of the market's recent trades via
    ``client.get_trades(market=condition_id, limit=limit)`` and returns the
    ``price`` of the most-recent (max ``timestamp``) trade whose ``outcomeIndex``
    matches ``outcome_index``. This is the price you'd pay to FOLLOW the whales
    *now* — a decision-time entry mark, not a future/look-ahead price.

    The price is validated to lie in ``[0, 1]`` (Polymarket outcome prices are
    probabilities in dollars). Returns ``None`` on any data-api error, any other
    exception, an empty page, or when no trade matches the outcome — the caller
    then falls back to ``entry_price = 0.0`` (unmarkable) rather than fabricate
    a fill.
    """
    try:
        trades = client.get_trades(market=condition_id, limit=limit)
    except PolymarketDataAPIError:
        return None
    except Exception:  # pragma: no cover - defensive: never let pricing crash a scan.
        return None

    best_price: Optional[float] = None
    best_ts: Optional[float] = None
    for trade in trades or []:
        if not isinstance(trade, dict):
            continue
        if trade.get("outcomeIndex") != outcome_index:
            continue
        try:
            price = float(trade.get("price"))
        except (TypeError, ValueError):
            continue
        if not (0.0 <= price <= 1.0):
            continue
        try:
            ts = float(trade.get("timestamp"))
        except (TypeError, ValueError):
            # No usable timestamp: treat as oldest so a timestamped trade wins.
            ts = float("-inf")
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best_price = price

    return best_price


def compute_confidence(
    n_target_holders: int,
    converging_winrates: Sequence[float],
    *,
    min_convergence: int = 3,
) -> tuple:
    """Heuristic 0-1 conviction for a whale-convergence candidate.

    This is a SIGNAL-STRENGTH indicator, NOT a probability of profit. It blends:
      * COUNT — how many smart-money wallets converged on the outcome
        (saturates at ~6 wallets); and
      * QUALITY — the average historical win-rate of those converging wallets
        (mapped 0.50 -> 0.0 .. 0.90 -> 1.0); neutral 0.5 when unknown.

    Returns ``(score, label)`` where label is ``'low'`` (<0.40), ``'medium'``
    (<0.66), or ``'high'`` (>=0.66). More/better wallets agreeing = higher.
    """
    holders = max(int(n_target_holders or 0), int(min_convergence or 0))
    conv_term = max(0.0, min(1.0, holders / 6.0))
    rates = [float(r) for r in converging_winrates if isinstance(r, (int, float))]
    if rates:
        avg_wr = sum(rates) / len(rates)
        quality_term = max(0.0, min(1.0, (avg_wr - 0.5) / 0.4))
    else:
        quality_term = 0.5  # neutral when wallet quality is unknown
    score = round(0.5 * conv_term + 0.5 * quality_term, 3)
    label = "high" if score >= 0.66 else "medium" if score >= 0.40 else "low"
    return score, label


def _log_candidates(
    *,
    ledger: PnlLedger,
    client: Any,
    candidates: List[Dict[str, Any]],
    min_convergence: int = 3,
    size_usd: float = 0.0,
    mark_entry: bool = False,
    wallet_stats: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Score + log a list of convergence candidates to the SHADOW ledger.

    Shared back-end for both runner modes: the live ``/holders`` path
    (:func:`run_once` -> :func:`find_convergence`) and the leaderboard
    ``/positions`` path (:func:`convergence_from_positions`) hand the SAME
    candidate shape to this one function, so the confidence, mark-to-market
    entry pricing, ``TradeRecord`` construction, and notes are written in
    exactly one place — never duplicated.

    For each candidate: compute confidence from the converging wallets' settled
    win-rates (``wallet_stats``), optionally fetch a real decision-time entry
    price (``mark_entry``), and append an OPEN :class:`TradeRecord`. NO orders
    are placed — the client is read-only. Mutates each candidate in place with
    ``confidence`` / ``confidence_label`` and returns the same list.
    """
    for cand in candidates:
        outcome_label = cand.get("outcome")
        side = str(outcome_label) if outcome_label is not None else str(
            cand.get("outcomeIndex")
        )
        wallets = cand.get("wallets") or []
        condition_id = str(cand.get("conditionId"))
        outcome_index = cand.get("outcomeIndex")

        # Confidence = how many smart-money wallets converged x how good they are
        # (avg historical win-rate from the ranking). Signal strength, not a
        # profit probability. Stored on the candidate + in the notes for Discord.
        winrates: List[float] = []
        if wallet_stats:
            for wallet in wallets:
                stat = wallet_stats.get(wallet)
                if stat is None:
                    continue
                rate = getattr(stat, "win_rate", stat)
                try:
                    winrates.append(float(rate))
                except (TypeError, ValueError):
                    continue
        conf_score, conf_label = compute_confidence(
            int(cand.get("n_target_holders") or len(wallets)),
            winrates,
            min_convergence=min_convergence,
        )
        cand["confidence"] = conf_score
        cand["confidence_label"] = conf_label

        entry_price = 0.0
        if mark_entry and outcome_index is not None:
            entry_price = (
                _latest_price_for_outcome(client, condition_id, outcome_index)
                or 0.0
            )

        if entry_price > 0:
            price_note = f"entry_price={entry_price:.4f} (latest /trades mark)"
        else:
            price_note = "entry_price=0.0 (holders endpoint carries no price)"
        notes = (
            "SHADOW MODE - NO ORDERS; "
            f"whale_convergence n={cand.get('n_target_holders')} "
            f"confidence={conf_score:.2f} ({conf_label}); "
            f"outcomeIndex={outcome_index}; "
            f"{price_note}; "
            f"wallets={','.join(wallets)}"
        )
        record = TradeRecord(
            trade_id=f"whale-{uuid.uuid4().hex[:12]}",
            ts_utc=_utc_now_iso(),
            venue="polymarket",
            market_id=condition_id,
            side=side,
            entry_price=float(entry_price),
            size=float(size_usd),
            fees_usd=0.0,
            slippage_bps=0.0,
            strategy=STRATEGY,
            status="open",
            notes=notes,
        )
        ledger.append(record)

    _print_summary(candidates)
    return candidates


def run_once(
    *,
    ledger: PnlLedger,
    client: Optional[Any] = None,
    target_wallets: Sequence[str] = (),
    markets_condition_ids: Sequence[str] = (),
    min_convergence: int = 3,
    size_usd: float = 0.0,
    mark_entry: bool = False,
    wallet_stats: Optional[Dict[str, Any]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run one convergence scan and log each candidate to the SHADOW ledger.

    For every convergence candidate from :func:`find_convergence`, append an
    OPEN :class:`TradeRecord` to ``ledger`` (``venue='polymarket'``,
    ``strategy='whale_convergence'``, ``side`` = the converged outcome label).
    The notes flag SHADOW MODE and list the contributing wallets so the
    candidate is auditable. NO orders are placed — the client is read-only.

    Entry price:
        * ``mark_entry=False`` (default) — record ``entry_price = 0.0`` with the
          legacy note, because the ``/holders`` endpoint carries no price. This
          preserves the original behavior exactly (no ``get_trades`` call).
        * ``mark_entry=True`` — set ``entry_price`` to the current market price of
          the converged outcome via :func:`_latest_price_for_outcome` (the price
          you'd pay to follow now, a decision-time entry — NOT look-ahead). When
          a real price is found the note reflects it; when none is found we fall
          back to ``0.0`` and keep the "holders endpoint carries no price" note.
        Either way the ``outcomeIndex=<n>`` marker is preserved in the notes so
        :func:`make_whale_price_fn` can re-price the open position later.

    Args:
        ledger:                 The shadow :class:`PnlLedger` to append to.
        client:                 Read-only data-api client. Required in practice;
                                kept optional for signature symmetry with other
                                runners. A ``None`` client raises ``ValueError``.
        target_wallets:         The smart-money roster.
        markets_condition_ids:  Markets to scan.
        min_convergence:        Convergence threshold (default 3).
        size_usd:               Notional label for the shadow record (default
                                0.0 — shadow positions carry no real size).
        mark_entry:             When True, fetch a real decision-time entry price
                                per candidate via the data-api ``/trades`` page.
        candidates:             OPTIONAL precomputed candidate list (same shape
                                :func:`find_convergence` returns). When provided
                                (e.g. leaderboard mode passes the output of
                                :func:`convergence_from_positions`), the
                                ``/holders`` scan is SKIPPED and these candidates
                                are logged directly. When ``None`` (the live
                                default), candidates are discovered from
                                ``markets_condition_ids`` via
                                :func:`find_convergence`.

    Returns:
        The list of convergence candidates that were logged.
    """
    if client is None:
        raise ValueError("run_once requires a read-only data-api client")

    if candidates is None:
        candidates = find_convergence(
            client,
            target_wallets,
            markets_condition_ids=markets_condition_ids,
            min_convergence=min_convergence,
        )

    return _log_candidates(
        ledger=ledger,
        client=client,
        candidates=candidates,
        min_convergence=min_convergence,
        size_usd=size_usd,
        mark_entry=mark_entry,
        wallet_stats=wallet_stats,
    )


def make_whale_price_fn(client: Any) -> Callable[[TradeRecord], Optional[float]]:
    """Build a ``price_fn(record) -> Optional[float]`` for the portfolio reporter.

    The returned callable re-prices an open whale-convergence position to the
    current market: it parses the ``outcomeIndex=<n>`` marker out of the record's
    notes (written by :func:`run_once`) and returns the latest ``/trades`` mark
    for that market+outcome via :func:`_latest_price_for_outcome`.

    Returns ``None`` (position left "pending", not marked at $0) when the notes
    carry no ``outcomeIndex`` marker or when the fetch fails — keeping the mark
    honest per the no-look-ahead / no-fabricated-fill standard.
    """

    def price_fn(record: TradeRecord) -> Optional[float]:
        match = _OUTCOME_INDEX_RE.search(record.notes or "")
        if match is None:
            return None
        outcome_index = int(match.group(1))
        return _latest_price_for_outcome(client, record.market_id, outcome_index)

    return price_fn


def _print_summary(candidates: List[Dict[str, Any]]) -> None:
    """Print a human-readable 'SHADOW MODE - NO ORDERS' run summary."""
    print("=== whale_convergence SHADOW MODE - NO ORDERS ===")
    print(f"logged {len(candidates)} convergence candidate(s) to the shadow ledger")
    for cand in candidates:
        print(
            f"  market={cand.get('conditionId')} "
            f"outcome={cand.get('outcome')!r} "
            f"(idx {cand.get('outcomeIndex')}) "
            f"n_target_holders={cand.get('n_target_holders')} "
            f"confidence={cand.get('confidence')} ({cand.get('confidence_label')}) "
            f"wallets={cand.get('wallets')}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: build a winners' roster, detect convergence, log to the SHADOW
    ledger, and optionally post per-trade P/L + portfolio to Discord.

    Two roster sources, READ-ONLY + SHADOW (NO orders are ever placed):

    * ``--roster-source leaderboard`` (default) — the roster is Polymarket's
      HISTORICAL PROFIT LEADERBOARD (``/profit``, real winners). Convergence
      then comes from what those winners CURRENTLY hold (each wallet's
      ``/positions``) via :func:`convergence_from_positions`, NOT from a fixed
      set of "hot" markets — an all-time legend may be dormant in today's hot
      markets, so the only way to follow them is to read their live book.
    * ``--roster-source live`` — the EXACT prior behavior: discover active
      wallets from recent trades, rank them by realized PnL/win-rate, derive
      hot candidate markets, and flag convergence on ``/holders``.

    Both paths feed the same confidence + mark-to-market + ledger logging.
    """
    parser = argparse.ArgumentParser(
        description="SHADOW whale-convergence follower (logs candidates, places NO orders)."
    )
    parser.add_argument("--roster-source", choices=("live", "leaderboard"),
                        default="leaderboard",
                        help="Where the winners' roster comes from: 'leaderboard' "
                        "(default) ranks Polymarket's historical /profit leaderboard "
                        "and detects convergence from those wallets' CURRENT "
                        "/positions; 'live' discovers + ranks active wallets and "
                        "scans hot markets via /holders (the prior behavior).")
    parser.add_argument("--leaderboard-window", type=str, default="all",
                        help="Profit-leaderboard window for --roster-source leaderboard: "
                        "'all' (all-time, default) or '1d' (today). Other values return HTTP 400.")
    parser.add_argument("--leaderboard-limit", type=int, default=100,
                        help="How many top profit-leaderboard wallets to pull as the "
                        "roster (default 100).")
    parser.add_argument("--min-convergence", type=int, default=3,
                        help="Min distinct target wallets on one outcome to flag (default 3).")
    parser.add_argument("--top-wallets", type=int, default=30,
                        help="Keep the top-N ranked smart-money wallets (default 30).")
    parser.add_argument("--min-settled", type=int, default=20,
                        help="Min settled positions for a wallet to be rankable (default 20).")
    parser.add_argument("--min-win-rate", type=float, default=0.60,
                        help="Min settled win-rate for a wallet to qualify (default 0.60).")
    parser.add_argument("--discover-pages", type=int, default=2,
                        help="Pages of recent /trades to harvest candidate wallets from.")
    parser.add_argument("--max-markets", type=int, default=40,
                        help="Max distinct hot markets to scan for convergence.")
    parser.add_argument("--size", type=float, default=100.0,
                        help="Paper USD notional per shadow position (default 100.0). "
                        "Must be >0 for the portfolio report to mark P/L "
                        "(units = size / entry_price); 0 leaves positions 'pending'.")
    parser.add_argument("--ledger-path", type=str, default=DEFAULT_LEDGER_PATH,
                        help=f"Path to the JSONL PnL ledger (default {DEFAULT_LEDGER_PATH}).")
    parser.add_argument("--discord", action="store_true",
                        help="Post per-trade P/L + portfolio value to Discord "
                        "(reads DISCORD_WEBHOOK_URL; no-op if unset).")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Paper bankroll (USD) baseline for the portfolio report.")
    parser.add_argument("--interval", type=float, default=None,
                        help="If set, run continuously, sleeping this many seconds "
                        "between scans (loop mode). Omit for a single pass.")
    parser.add_argument("--rank-refresh-scans", type=int, default=48,
                        help="In loop mode, re-rank smart-money wallets every N scans "
                        "(default 48); wallets are cached between re-ranks to avoid "
                        "hammering the per-wallet /positions calls.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Local imports keep the module importable without the data-api/ranker in
    # minimal/test environments (the unit tests inject fakes instead).
    from exchanges.polymarket_data_api import PolymarketDataAPIClient
    import wallet_ranker

    client = PolymarketDataAPIClient()
    ledger = PnlLedger(args.ledger_path)

    # A current-market price_fn so the Discord portfolio report marks each open
    # whale position to market (entry -> current). It re-prices via /trades using
    # the outcomeIndex marker in each record's notes; SHADOW/observability only.
    price_fn = make_whale_price_fn(client)

    # wallet -> WalletStats for the current roster, so each convergence candidate
    # can be scored on the win-rate of the wallets that actually converged.
    wallet_stats: Dict[str, Any] = {}

    def _rank_targets() -> List[str]:
        """Build the winners' roster for the configured ``--roster-source``.

        * ``leaderboard`` — pull the top profit-leaderboard wallets (real
          historical winners). ``wallet_stats`` is left empty here; it is filled
          per-scan by :func:`convergence_from_positions` from the SAME
          ``/positions`` fetch used to detect convergence (no extra calls).
        * ``live`` — discover + rank active wallets and capture their
          :class:`WalletStats` for confidence scoring (the prior behavior).
        """
        if args.roster_source == "leaderboard":
            rows = client.get_profit_leaderboard(
                window=args.leaderboard_window,
                limit=args.leaderboard_limit,
            )
            targets: List[str] = []
            for row in rows:
                wallet = row.get("proxyWallet") if isinstance(row, dict) else None
                if wallet and wallet not in targets:
                    targets.append(wallet)
            # Stats come from convergence_from_positions per scan, not from here.
            wallet_stats.clear()
            print(
                f"loaded {len(targets)} profit-leaderboard wallet(s) "
                f"(window={args.leaderboard_window!r})"
            )
            return targets

        discovered = wallet_ranker.discover_active_wallets(
            client, pages=args.discover_pages
        )
        ranked = wallet_ranker.rank_wallets(
            client,
            candidate_wallets=discovered,
            min_settled=args.min_settled,
            min_win_rate=args.min_win_rate,
            top_n=args.top_wallets,
        )
        targets = [w.wallet for w in ranked]
        wallet_stats.clear()
        wallet_stats.update({w.wallet: w for w in ranked})
        print(
            f"ranked {len(targets)} smart-money wallet(s) from "
            f"{len(discovered)} active"
        )
        return targets

    def _hot_markets() -> List[str]:
        """Candidate markets = the hottest recent markets (by trade recency)."""
        market_ids: List[str] = []
        for trade in client.get_trades(
            limit=min(500, max(args.max_markets * 10, 100))
        ):
            cid = trade.get("conditionId") if isinstance(trade, dict) else None
            if cid and cid not in market_ids:
                market_ids.append(cid)
            if len(market_ids) >= args.max_markets:
                break
        return market_ids

    def _report() -> None:
        if not args.discord:
            return
        try:
            from portfolio_reporter import load_env_files, report_to_discord
            from alerts.notifier import Notifier
            load_env_files()  # pick up DISCORD_WEBHOOK_URL from .env for CLI runs
            report_to_discord(
                ledger, Notifier(), price_fn=price_fn, bankroll_usd=args.bankroll,
                label="Whale-follow shadow",
            )
        except ImportError:
            logging.warning(
                "Discord reporting unavailable (alerts/portfolio_reporter import failed)."
            )

    def _scan(targets: Sequence[str]) -> None:
        if args.roster_source == "leaderboard":
            # Convergence comes from the roster's CURRENT positions (not hot
            # markets). The same /positions fetch also yields per-wallet stats,
            # so confidence gets real win-rates with no extra calls.
            candidates, stats = convergence_from_positions(
                client,
                targets,
                min_convergence=args.min_convergence,
                return_stats=True,
            )
            wallet_stats.clear()
            wallet_stats.update(stats)
            run_once(
                ledger=ledger,
                client=client,
                min_convergence=args.min_convergence,
                size_usd=args.size,
                mark_entry=True,
                wallet_stats=wallet_stats,
                candidates=candidates,
            )
        else:
            market_ids = _hot_markets()
            run_once(
                ledger=ledger,
                client=client,
                target_wallets=targets,
                markets_condition_ids=market_ids,
                min_convergence=args.min_convergence,
                size_usd=args.size,
                mark_entry=True,
                wallet_stats=wallet_stats,
            )
        _report()

    # Single pass (default): rank once, scan once, report, done.
    if args.interval is None:
        targets = _rank_targets()
        _scan(targets)
        return 0

    # Loop mode: rank on the FIRST iteration, then re-rank every
    # ``rank_refresh_scans`` iterations (wallets cached between re-ranks so we
    # don't hammer the N+1 /positions calls on every scan). Each iteration
    # re-derives hot markets, runs a scan, reports, and sleeps. Ctrl-C exits
    # cleanly with the no-orders affirmation.
    refresh = max(1, int(args.rank_refresh_scans))
    targets: List[str] = []
    iteration = 0
    try:
        while True:
            try:
                # Re-rank on schedule, or whenever a prior rank left no roster
                # (e.g. the first attempt hit a transient API error).
                if iteration % refresh == 0 or not targets:
                    targets = _rank_targets()
                _scan(targets)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - a transient API/network
                # error must NOT kill an unattended shadow loop; log and retry
                # on the next tick. No orders are ever placed regardless.
                logging.warning(
                    "scan iteration %d failed (%s); retrying after interval.",
                    iteration,
                    exc,
                )
            iteration += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped by user. No orders were ever placed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
