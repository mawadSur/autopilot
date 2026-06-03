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
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from state.pnl_ledger import (
        DEFAULT_LEDGER_PATH,
        PnlLedger,
        TradeRecord,
        duplicate_open_records,
    )
except Exception:  # pragma: no cover - import shim for alternate layouts.
    from src.state.pnl_ledger import (  # type: ignore
        DEFAULT_LEDGER_PATH,
        PnlLedger,
        TradeRecord,
        duplicate_open_records,
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

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from trade_blocklist import blocked_term, load_blocklist
except Exception:  # pragma: no cover - import shim for alternate layouts.
    try:
        from src.trade_blocklist import blocked_term, load_blocklist  # type: ignore
    except Exception:  # pragma: no cover - degrade to no blocklist if missing.
        def blocked_term(text, terms):  # type: ignore
            return None

        def load_blocklist(*_a, **_k):  # type: ignore
            return []


__all__ = [
    "STRATEGY",
    "find_convergence",
    "convergence_from_positions",
    "compute_confidence",
    "leaderboard_quality",
    "run_once",
    "make_whale_price_fn",
    "make_whale_fill_fn",
    "main",
]

# Pulls the ``outcomeIndex=<n>`` marker the runner writes into a record's notes.
_OUTCOME_INDEX_RE = re.compile(r"outcomeIndex=(\d+)")
# Pulls the ``asset=<token_id>`` marker (the CLOB outcome token id) for book-aware exits.
_ASSET_RE = re.compile(r"asset=([^;]+)")


STRATEGY = "whale_convergence"


def _heal_duplicate_opens(ledger: PnlLedger) -> int:
    """Cancel duplicate OPEN records so the ledger holds one per (market, outcome).

    Two writers racing on the same ledger can each append an ``open`` for the same
    convergence (each snapshots the open set before the other's append lands),
    leaving duplicate open records that would show twice on the dashboard and
    settle twice (double-counted P/L). This retires the extras with an append-only
    :meth:`PnlLedger.cancel` — the earliest entry per (market, outcome) is kept,
    the later duplicates are cancelled. Idempotent: a clean ledger heals zero.

    Returns the number of duplicate records cancelled. Never raises (a read/cancel
    failure is logged and the scan continues — settlement and the view layer dedup
    too, so an un-healed duplicate is still harmless).
    """
    try:
        extras = duplicate_open_records(ledger.open_positions())
    except Exception as exc:  # noqa: BLE001 - never let a read crash the loop.
        logging.warning("dedup heal: open_positions() failed (%s); skipping.", exc)
        return 0
    cancelled = 0
    for record in extras:
        try:
            ledger.cancel(
                record.trade_id, reason="duplicate_open (concurrent-writer race)"
            )
            cancelled += 1
        except Exception as exc:  # noqa: BLE001 - one bad cancel must not abort.
            logging.warning(
                "dedup heal: cancel(%s) failed (%s); leaving as-is.",
                getattr(record, "trade_id", "?"),
                exc,
            )
    if cancelled:
        print(f"dedup heal: cancelled {cancelled} duplicate open record(s)")
    return cancelled


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` is currently running (POSIX)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists but owned by another user
        return True
    except OSError:
        return False
    return True


def _acquire_ledger_lock(ledger_path: str) -> Optional[str]:
    """Take a single-writer lock for ``ledger_path``; return the lock path or None.

    Prevents the root cause of duplicate opens: two ``whale_follow_runner`` loops
    writing the SAME ledger. The lock is a ``<ledger>.lock`` file holding this
    process's PID. If a live PID already holds it, we refuse (return ``None`` and
    print how to override). A stale lock (PID no longer running) is reclaimed.

    Best-effort and POSIX-oriented; any filesystem error degrades to "no lock"
    (returns the path) rather than blocking a legitimate run.
    """
    lock_path = f"{ledger_path}.lock"
    parent = os.path.dirname(os.path.abspath(lock_path))
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        for _attempt in range(2):  # one retry after reclaiming a stale lock
            try:
                # ATOMIC create-or-fail: only one racer wins O_CREAT|O_EXCL, so
                # two runners starting at once can't both pass a check and both
                # write a PID.
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                return lock_path
            except FileExistsError:
                try:
                    with open(lock_path, "r", encoding="utf-8") as handle:
                        holder = int((handle.read().strip() or "0"))
                except (ValueError, OSError):
                    holder = 0
                if holder == os.getpid():
                    return lock_path  # already ours
                if holder and _pid_alive(holder):
                    print(
                        f"REFUSING TO START: another writer (pid {holder}) already holds "
                        f"{lock_path}. Two loops on one ledger create duplicate opens. "
                        f"Stop it first (kill {holder}), or use a different --ledger-path."
                    )
                    return None
                # Stale lock (holder gone / unreadable): reclaim and retry once.
                try:
                    os.remove(lock_path)
                except OSError:
                    return lock_path  # can't reclaim; degrade to no-lock
        return lock_path
    except OSError as exc:  # pragma: no cover - lock is best-effort, never fatal.
        logging.warning("could not acquire ledger lock (%s); continuing.", exc)
        return lock_path


def _release_ledger_lock(lock_path: Optional[str]) -> None:
    """Release a lock taken by :func:`_acquire_ledger_lock` (only if we own it)."""
    if not lock_path:
        return
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            holder = int((handle.read().strip() or "0"))
        if holder == os.getpid():
            os.remove(lock_path)
    except (ValueError, OSError):
        pass


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
                     "asset": holder.get("asset"),
                     "wallets": {}},
                )
                # name/outcome label may only appear on some rows; keep first seen.
                if bucket["outcome"] is None:
                    bucket["outcome"] = holder.get("name") or holder.get("outcome")
                # asset = the CLOB outcome token id; keep first seen (for book-aware exits).
                if not bucket.get("asset"):
                    bucket["asset"] = holder.get("asset")
                bucket["wallets"][wallet] = None  # ordered distinct set.

        for outcome_index, bucket in per_outcome.items():
            wallets = list(bucket["wallets"].keys())
            if len(wallets) >= min_convergence:
                candidates.append(
                    {
                        "conditionId": condition_id,
                        "outcomeIndex": outcome_index,
                        "outcome": bucket["outcome"],
                        "asset": bucket.get("asset"),
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
                {
                    "outcome": position.get("outcome"),
                    "title": position.get("title"),
                    "asset": position.get("asset"),
                    "wallets": {},
                },
            )
            if bucket["outcome"] is None:
                bucket["outcome"] = position.get("outcome")
            if not bucket.get("title"):
                bucket["title"] = position.get("title")
            # asset = the CLOB outcome token id; keep first seen (book-aware exits).
            if not bucket.get("asset"):
                bucket["asset"] = position.get("asset")
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
                    "title": bucket.get("title"),
                    "asset": bucket.get("asset"),
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


def leaderboard_quality(rank_index: int, total: int) -> float:
    """Map a 0-based profit-leaderboard rank to a [0.6, 1.0] quality score.

    Being on the all-time top-N profit board is itself strong evidence of skill,
    so even the lowest-ranked board member earns an "elite floor" of 0.6, while
    the #1 wallet earns 1.0. Used as the confidence QUALITY term in leaderboard
    mode: these whales win on SIZE, not hit-rate, so their settled win-rate
    (~0.5) badly understates them — profit-rank is the honest quality signal.
    """
    if total <= 1:
        return 1.0
    idx = max(0, min(int(rank_index), int(total) - 1))
    return round(0.6 + 0.4 * (1.0 - idx / (int(total) - 1)), 3)


def compute_confidence(
    n_target_holders: int,
    converging_winrates: Sequence[float],
    *,
    min_convergence: int = 3,
    quality_scores: Optional[Sequence[float]] = None,
) -> tuple:
    """Heuristic 0-1 conviction for a whale-convergence candidate.

    This is a SIGNAL-STRENGTH indicator, NOT a probability of profit. It blends:
      * COUNT — how many smart-money wallets converged on the outcome
        (saturates at ~6 wallets); and
      * QUALITY — how good the converging wallets are. Two sources:
          - ``quality_scores`` (preferred when given): pre-mapped per-wallet
            quality in [0, 1] — e.g. profit-leaderboard rank via
            :func:`leaderboard_quality`. Used as-is (averaged). This is what
            leaderboard mode passes, because top-profit whales have ~0.5
            win-rates that understate them.
          - else ``converging_winrates``: historical settled win-rates mapped
            0.50 -> 0.0 .. 0.90 -> 1.0; neutral 0.5 when unknown.

    Returns ``(score, label)`` where label is ``'low'`` (<0.40), ``'medium'``
    (<0.66), or ``'high'`` (>=0.66). More/better wallets agreeing = higher.
    """
    holders = max(int(n_target_holders or 0), int(min_convergence or 0))
    conv_term = max(0.0, min(1.0, holders / 6.0))
    qualities = (
        [float(q) for q in quality_scores if isinstance(q, (int, float))]
        if quality_scores is not None
        else []
    )
    if qualities:
        quality_term = max(0.0, min(1.0, sum(qualities) / len(qualities)))
    else:
        rates = [
            float(r) for r in converging_winrates if isinstance(r, (int, float))
        ]
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
    wallet_quality: Optional[Dict[str, float]] = None,
    blocklist: Optional[Sequence[str]] = None,
    dedup: bool = True,
    min_confidence: float = 0.0,
    min_entry_price: float = 0.0,
    max_entry_price: float = 1.0,
    require_directional: bool = False,
    book_entry: bool = False,
    max_book_frac: float = 0.0,
    depth_band: float = 0.05,
) -> List[Dict[str, Any]]:
    """Score + log convergence candidates to the SHADOW ledger.

    Shared back-end for both runner modes: the live ``/holders`` path
    (:func:`run_once` -> :func:`find_convergence`) and the leaderboard
    ``/positions`` path (:func:`convergence_from_positions`) hand the SAME
    candidate shape to this one function, so the confidence, mark-to-market
    entry pricing, ``TradeRecord`` construction, and notes are written in
    exactly one place — never duplicated.

    Filters apply BEFORE logging (each counted, never silent):
      * ``blocklist`` — operator do-not-trade terms (alcohol/casino/adult, etc.).
        Marked ``cand['blocked']=<term>``.
      * ``require_directional`` (default off; CLI on) — DIRECTIONAL READ filter.
        Drop every candidate in a market where convergence fired on more than one
        OUTCOME (target wallets split across both sides). A signal that flags both
        sides of a game carries no directional information — it says "whales are
        here", not "this side is mispriced". Marked ``cand['skipped']='split_market'``.
      * ``dedup`` (default True) — one open position per ``(market, outcome)``.
        Marked ``cand['skipped']='duplicate_open'``.
      * ``min_confidence`` (function default 0.0 = off; CLI 0.5) — only enter the
        strongest convergences. Marked ``cand['skipped']='low_confidence'``.
      * ``[min_entry_price, max_entry_price]`` (function default 0..1 = off; CLI
        0.15..0.85) — ENTRY-BAND filter (needs ``mark_entry``). Skip a candidate
        whose decision-time mark is outside the band: near $1 you pay the full 2%
        fee for ~zero upside (near-decided market); near $0 you're buying the
        losing side. Only enter where the outcome is genuinely uncertain and there
        is room to be right. Marked ``cand['skipped']='entry_out_of_band'``.
      * ``max_book_frac`` (function default 0.0 = off; CLI 0.05) — per-market
        DEPTH CAP (needs the candidate's ``asset`` + an ASK book). Compute the
        available ASK depth in USD near mid (ask levels priced within
        ``depth_band`` of the best ask) and SKIP a candidate whose ``size_usd``
        exceeds ``max_book_frac`` of that depth — we could never exit a market too
        thin to take our size. Marked ``cand['skipped']='thin_book'``. Fails OPEN:
        a book read error never drops a candidate.

    Book-aware ENTRY (``book_entry``, function default off; CLI on, needs
    ``mark_entry`` + the candidate's ``asset``): instead of the last ``/trades``
    mark, price the entry off the CURRENT ASK book — estimate the units we'd buy
    (``size_usd / best_ask``) and take the ask VWAP for those units via
    :func:`polymarket_market_data.vwap_buy_price` (the price we'd actually lift,
    decision-time, NOT look-ahead). Falls back to the ``/trades`` mark on any
    failure / missing ``asset`` / empty asks; the ENTRY-BAND still applies to
    whatever entry results.

    For each surviving candidate: compute confidence, optionally fetch a real
    decision-time entry price (``mark_entry``), and append an OPEN
    :class:`TradeRecord`. NO orders are placed — the client is read-only.
    Returns only the candidates actually LOGGED.
    """
    existing_open = set()
    if dedup:
        try:
            existing_open = {
                (r.market_id, r.side) for r in ledger.open_positions()
            }
        except Exception:  # pragma: no cover - never let a read break logging.
            existing_open = set()

    # Directional read: find markets where the signal fired on >1 outcome — the
    # convergence is split across both sides, so there's no directional edge to
    # follow. Drop the whole market (computed once, up front).
    split_markets: set = set()
    if require_directional:
        outcomes_by_market: Dict[str, set] = {}
        for cand in candidates:
            cid = str(cand.get("conditionId"))
            label = cand.get("outcome")
            side_key = str(label) if label is not None else str(cand.get("outcomeIndex"))
            outcomes_by_market.setdefault(cid, set()).add(side_key)
        split_markets = {cid for cid, outs in outcomes_by_market.items() if len(outs) > 1}

    band_active = (min_entry_price > 0.0) or (max_entry_price < 1.0)
    depth_cap_active = max_book_frac > 0.0

    # Lazy import keeps the module importable in minimal/test envs; only loaded
    # when book-aware entry or the depth cap is actually requested (both read the
    # CLOB ASK book — SHADOW/read-only, no orders).
    pmd = None
    if book_entry or depth_cap_active:
        try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
            from exchanges import polymarket_market_data as pmd  # type: ignore
        except Exception:  # pragma: no cover - alternate layout shim.
            try:
                from src.exchanges import polymarket_market_data as pmd  # type: ignore
            except Exception:  # pragma: no cover - degrade to mark-only if missing.
                pmd = None

    logged: List[Dict[str, Any]] = []
    n_blocked = 0
    n_dup = 0
    n_low_conf = 0
    n_split = 0
    n_band = 0
    n_thin = 0
    for cand in candidates:
        outcome_label = cand.get("outcome")
        side = str(outcome_label) if outcome_label is not None else str(
            cand.get("outcomeIndex")
        )
        wallets = cand.get("wallets") or []
        condition_id = str(cand.get("conditionId"))
        outcome_index = cand.get("outcomeIndex")

        # Do-not-trade blocklist (operator policy): skip blocked topics. Matched
        # against the market title + outcome label (title present in leaderboard
        # mode via convergence_from_positions).
        if blocklist:
            match = blocked_term(
                f"{cand.get('title') or ''} {outcome_label or ''}", blocklist
            )
            if match:
                cand["blocked"] = match
                n_blocked += 1
                continue

        # Directional read: if the signal fired on both sides of this market,
        # it carries no directional edge — skip the whole market.
        if require_directional and condition_id in split_markets:
            cand["skipped"] = "split_market"
            n_split += 1
            continue

        # Dedup: one open position per (market, outcome) — no per-scan pile-up.
        key = (condition_id, side)
        if dedup and key in existing_open:
            cand["skipped"] = "duplicate_open"
            n_dup += 1
            continue

        # Confidence = how many smart-money wallets converged x how good they
        # are. Quality is profit-leaderboard rank when available (leaderboard
        # mode), else historical win-rate (live mode). Signal strength, not a
        # profit probability. Stored on the candidate + in the notes for Discord.
        quality_scores: List[float] = []
        if wallet_quality:
            quality_scores = [
                float(wallet_quality[w]) for w in wallets if w in wallet_quality
            ]
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
            quality_scores=quality_scores or None,
        )
        cand["confidence"] = conf_score
        cand["confidence_label"] = conf_label

        # ENTRY filter: only follow the strongest convergences. A candidate below
        # the confidence floor is counted (never silently dropped) and skipped
        # before any record is written — this is the "catch the most profitable"
        # lever that keeps weak signals out of the shadow ledger entirely.
        if conf_score < min_confidence:
            cand["skipped"] = "low_confidence"
            n_low_conf += 1
            continue

        asset_token = str(cand.get("asset") or "").strip()

        entry_price = 0.0
        entry_source = "mark"  # which source set entry_price (for an honest note)
        if mark_entry and outcome_index is not None:
            mark = (
                _latest_price_for_outcome(client, condition_id, outcome_index)
                or 0.0
            )
            entry_price = mark
            # Book-aware ENTRY (W2): price off the CURRENT ask book — the price we'd
            # actually lift — instead of the last /trades print. Estimate the units
            # we'd buy from the best ask (fall back to the /trades mark for the unit
            # estimate when the book has no asks), then take the ask VWAP for those
            # units. Any failure / missing asset / empty asks FALLS BACK to the mark.
            if book_entry and pmd is not None and asset_token:
                try:
                    book = pmd.get_order_book(asset_token)
                    asks = book.get("asks") if isinstance(book, dict) else None
                    best = pmd.best_ask(book) if isinstance(book, dict) else None
                    unit_px = best if (best is not None and best > 0) else mark
                    if unit_px and unit_px > 0 and size_usd > 0:
                        units = float(size_usd) / float(unit_px)
                        result = pmd.vwap_buy_price(asks, units)
                        if result is not None:
                            entry_price = result[0]
                            entry_source = "book"
                except Exception:  # noqa: BLE001 - never let a book read crash the scan.
                    entry_price = mark
                    entry_source = "mark"

        # ENTRY-BAND filter: only enter where the outcome is genuinely uncertain.
        # Outside [min, max] is fee-drag (near $1) or the losing side (near $0); an
        # unmarkable 0.0 is also out-of-band, so the band keeps unpriceable entries
        # out of the filtered loop.
        if band_active and not (min_entry_price <= entry_price <= max_entry_price):
            cand["skipped"] = "entry_out_of_band"
            n_band += 1
            continue

        # Per-market DEPTH CAP (W2): refuse a market too thin to take our size — a
        # convergence in a $300-liquidity market is untradeable at $100 (we could
        # never exit). Sum the ASK depth in USD near mid (levels within depth_band
        # of the best ask); skip when size_usd exceeds max_book_frac of it. Fails
        # OPEN: a missing asset or any book error proceeds rather than dropping a
        # candidate because a read failed.
        if depth_cap_active and pmd is not None and asset_token and size_usd > 0:
            try:
                cap_book = pmd.get_order_book(asset_token)
                cap_asks = cap_book.get("asks") if isinstance(cap_book, dict) else None
                cap_best = pmd.best_ask(cap_book) if isinstance(cap_book, dict) else None
                if cap_best is not None and isinstance(cap_asks, (list, tuple)):
                    cutoff = cap_best + float(depth_band)
                    depth_usd = 0.0
                    for lvl in cap_asks:
                        if isinstance(lvl, dict):
                            px = lvl.get("price")
                            sz = lvl.get("size")
                        elif isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                            px, sz = lvl[0], lvl[1]
                        else:
                            continue
                        try:
                            px_f = float(px)
                            sz_f = float(sz)
                        except (TypeError, ValueError):
                            continue
                        if not (0.0 <= px_f <= 1.0) or sz_f <= 0:
                            continue
                        if px_f <= cutoff:
                            depth_usd += sz_f * px_f
                    if depth_usd > 0 and float(size_usd) > max_book_frac * depth_usd:
                        cand["skipped"] = "thin_book"
                        n_thin += 1
                        continue
            except Exception:  # noqa: BLE001 - fail OPEN: a read error never drops a candidate.
                pass

        if entry_price > 0:
            source_label = "ask-book VWAP" if entry_source == "book" else "latest /trades mark"
            price_note = f"entry_price={entry_price:.4f} ({source_label})"
        else:
            price_note = "entry_price=0.0 (holders endpoint carries no price)"
        # Human market title (leaderboard mode carries it via /positions) so the
        # dashboard shows a real name instead of a hex conditionId slug. Strip
        # ';' so the `title=...;` marker stays parseable by clean_title.
        title_raw = (cand.get("title") or "").replace(";", ",").strip()
        title_note = f"title={title_raw}; " if title_raw else ""
        # asset = the CLOB outcome token id, so the exit sweep can walk THIS
        # outcome's bid book for a realistic fill (W1 book-aware exits).
        asset_raw = str(cand.get("asset") or "").replace(";", "").strip()
        asset_note = f"asset={asset_raw}; " if asset_raw else ""
        notes = (
            "SHADOW MODE - NO ORDERS; "
            f"{title_note}{asset_note}"
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
        existing_open.add(key)
        logged.append(cand)

    _print_summary(logged)
    if n_blocked or n_dup or n_low_conf or n_split or n_band or n_thin:
        print(
            f"  (skipped {n_dup} already-open duplicate(s); "
            f"blocked {n_blocked} do-not-trade market(s); "
            f"skipped {n_low_conf} below-confidence; "
            f"{n_split} split-market (no directional read); "
            f"{n_band} out-of-band entry; "
            f"{n_thin} thin-book (depth cap))"
        )
    return logged


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
    wallet_quality: Optional[Dict[str, float]] = None,
    blocklist: Optional[Sequence[str]] = None,
    dedup: bool = True,
    min_confidence: float = 0.0,
    min_entry_price: float = 0.0,
    max_entry_price: float = 1.0,
    require_directional: bool = False,
    book_entry: bool = False,
    max_book_frac: float = 0.0,
    depth_band: float = 0.05,
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
        book_entry:             When True (with ``mark_entry``), price the entry
                                off the CURRENT ASK book (ask VWAP for the units
                                we'd buy) instead of the last ``/trades`` mark;
                                falls back to the mark on any failure. Default off.
        max_book_frac:          Per-market DEPTH CAP (default 0.0 = off): skip a
                                candidate whose ``size_usd`` exceeds this fraction
                                of the near-mid ASK depth in USD (too thin to take
                                our size / to exit). Fails open on a book error.
        depth_band:             Price window (default 0.05) above the best ask used
                                to sum near-mid ASK depth for ``max_book_frac``.
        min_confidence:         ENTRY filter (default 0.0 = off): only log
                                candidates whose computed ``confidence`` is
                                >= this floor, keeping the strongest convergences.
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
        wallet_quality=wallet_quality,
        blocklist=blocklist,
        dedup=dedup,
        min_confidence=min_confidence,
        min_entry_price=min_entry_price,
        max_entry_price=max_entry_price,
        require_directional=require_directional,
        book_entry=book_entry,
        max_book_frac=max_book_frac,
        depth_band=depth_band,
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


def make_whale_fill_fn(session: Any = None) -> Callable[[TradeRecord], Optional[float]]:
    """Build ``fill_fn(record) -> Optional[float]``: the REALISTIC sell price.

    Where :func:`make_whale_price_fn` returns the top-of-book *mark* (what the
    position is theoretically worth), this returns the price you would actually
    REALIZE liquidating the position now: it parses the ``asset=<token_id>``
    marker from the record's notes, fetches that outcome's CLOB bid book, and
    walks the bids for the position's units (``size / entry_price``) via
    :func:`polymarket_market_data.vwap_sell_price`. On a thin book the fill is
    well below the mark — which is exactly the honest exit number the take-profit
    accounting was missing (W1).

    Returns ``None`` — caller falls back to the mark — when the notes carry no
    ``asset`` marker (e.g. a pre-W1 record), the entry price is unusable, the
    book fetch fails, or the bid side is empty. Read-only; SHADOW.
    """
    try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
        from exchanges import polymarket_market_data as pmd
    except Exception:  # pragma: no cover - alternate layout shim.
        from src.exchanges import polymarket_market_data as pmd  # type: ignore

    def fill_fn(record: TradeRecord) -> Optional[float]:
        match = _ASSET_RE.search(record.notes or "")
        if match is None:
            return None
        token_id = match.group(1).strip()
        entry = getattr(record, "entry_price", None)
        size = getattr(record, "size", None)
        if not token_id or not entry or entry <= 0 or not size or size <= 0:
            return None
        units = float(size) / float(entry)
        try:
            book = pmd.get_order_book(token_id, session=session)
        except Exception:  # noqa: BLE001 - never let a pricing read crash the sweep.
            return None
        result = pmd.vwap_sell_price(book.get("bids") if isinstance(book, dict) else None, units)
        if result is None:
            return None
        return result[0]

    return fill_fn


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
    parser.add_argument("--blocklist-file", type=str, default=None,
                        help="Path to an extra do-not-trade list (one term per line, "
                        "# comments ok). Added on top of the built-in alcohol / "
                        "casino / adult defaults.")
    parser.add_argument("--no-blocklist", action="store_true",
                        help="Disable the do-not-trade blocklist entirely.")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable dedup (allow re-logging a convergence that "
                        "already has an open position). Default keeps one per market.")
    parser.add_argument("--no-settle", action="store_true",
                        help="Disable settlement of resolved positions. Default "
                        "(settlement ON) sweeps open positions at the start of each "
                        "scan, closing any whose Polymarket market has RESOLVED so "
                        "realized P/L shows in the report and dedup frees up.")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="ENTRY filter: only enter convergences whose computed "
                        "confidence is >= this (default 0.5) — catch the most "
                        "profitable, strongest convergences. 0.0 disables.")
    parser.add_argument("--min-entry-price", type=float, default=0.15,
                        help="ENTRY-BAND floor (default 0.15): skip entries priced "
                        "below this — buying near $0 is the losing side. Set 0.0 to "
                        "disable the band.")
    parser.add_argument("--max-entry-price", type=float, default=0.85,
                        help="ENTRY-BAND ceiling (default 0.85): skip entries priced "
                        "above this — buying near $1 pays the full 2%% fee for ~zero "
                        "upside (near-decided market). Set 1.0 to disable.")
    parser.add_argument("--no-book-entry", action="store_true",
                        help="Disable book-aware entry pricing. Default (book entry "
                        "ON) prices each entry off the CURRENT ASK book — the ask "
                        "VWAP for the units we'd buy, the price we'd actually lift — "
                        "instead of the last /trades print, falling back to the mark "
                        "on any book error (SHADOW-ONLY read).")
    parser.add_argument("--max-book-frac", type=float, default=0.05,
                        help="Per-market DEPTH CAP (default 0.05 = 5%%): skip a "
                        "candidate whose --size exceeds this fraction of the near-mid "
                        "ASK depth (USD) — a market too thin to take our size we "
                        "could never exit. Set 0.0 to disable.")
    parser.add_argument("--depth-band", type=float, default=0.05,
                        help="Price window (default 0.05) above the best ask used to "
                        "sum near-mid ASK depth for --max-book-frac.")
    parser.add_argument("--allow-split-markets", action="store_true",
                        help="Disable the directional-read filter. Default (filter "
                        "ON) drops any market where convergence fired on >1 outcome "
                        "(whales split both sides = no directional edge to follow).")
    parser.add_argument("--stop-loss-pct", type=float, default=0.40,
                        help="Early-exit a position once it is down >= this fraction "
                        "of its entry cost (default 0.40 = -40%%) — cap the loss "
                        "instead of riding a loser to $0.")
    parser.add_argument("--take-profit-pct", type=float, default=0.50,
                        help="Early-exit a position once it is up >= this fraction "
                        "of its entry cost (default 0.50 = +50%%) — lock the win.")
    parser.add_argument("--take-profit-price", type=float, default=0.90,
                        help="Early-exit a position once the outcome marks >= this "
                        "absolute price (default 0.90) — a near-certain win, lock "
                        "it in and redeploy.")
    parser.add_argument("--no-exit-rules", action="store_true",
                        help="Disable the stop-loss / take-profit early-exit sweep. "
                        "Default (exit rules ON) sweeps open positions each scan "
                        "and settles those past a stop/take-profit at the current "
                        "mark (SHADOW-ONLY, NO orders).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Do-not-trade blocklist (operator policy): built-in alcohol/casino/adult
    # defaults plus any --blocklist-file terms. Empty when --no-blocklist.
    blocklist = (
        []
        if args.no_blocklist
        else load_blocklist(args.blocklist_file)
    )
    dedup = not args.no_dedup
    if blocklist:
        print(f"do-not-trade blocklist active: {len(blocklist)} term(s)")

    # Local imports keep the module importable without the data-api/ranker in
    # minimal/test environments (the unit tests inject fakes instead).
    from exchanges.polymarket_data_api import PolymarketDataAPIClient
    from exchanges import polymarket_market_data
    import shadow_settlement
    import exit_rules
    import wallet_ranker

    client = PolymarketDataAPIClient()
    ledger = PnlLedger(args.ledger_path)

    # Settlement (ON unless --no-settle): a resolver maps a market conditionId
    # to its CLOB resolution status. SHADOW-ONLY read — closes resolved
    # positions into the ledger, places NO orders.
    settle_enabled = not args.no_settle
    resolver = (
        (lambda cid: polymarket_market_data.get_market_resolution(cid))
        if settle_enabled
        else None
    )

    def _settle() -> None:
        """Sweep + settle any resolved open positions. Never crashes the loop."""
        if not settle_enabled:
            return
        try:
            res = shadow_settlement.settle_resolved_positions(ledger, resolver)
            print(
                f"settled {res['settled']}: {res['won']} won / "
                f"{res['lost']} lost, realized ${res['total_realized_pnl_usd']:.2f}; "
                f"still_open {res['still_open']}"
            )
        except Exception as exc:  # noqa: BLE001 - settlement must never crash the scan.
            logging.warning("settlement sweep failed (%s); continuing.", exc)

    # A current-market price_fn so the Discord portfolio report marks each open
    # whale position to market (entry -> current). It re-prices via /trades using
    # the outcomeIndex marker in each record's notes; SHADOW/observability only.
    price_fn = make_whale_price_fn(client)
    # W1 book-aware fill: prices an exit at the bid-book VWAP for the position's
    # units (the price we'd actually realize), used for the realized P/L while
    # price_fn (the mark) still drives the exit DECISION. SHADOW/read-only.
    fill_fn = make_whale_fill_fn()

    # Early-exit sweep (ON unless --no-exit-rules): re-price each open position to
    # the CURRENT mark and settle those past the stop-loss / take-profit — capping
    # losers (instead of riding to $0) and locking winners, booked at the realistic
    # book-walked fill. SHADOW-ONLY read + ledger bookkeeping; places NO orders.
    exit_rules_enabled = not args.no_exit_rules

    def _apply_exit_rules() -> None:
        """Sweep + early-exit open positions. Never crashes the loop (wrapped so a
        transient pricing error can't kill the scan)."""
        if not exit_rules_enabled:
            return
        try:
            res = exit_rules.apply_exit_rules(
                ledger,
                price_fn,
                stop_loss_pct=args.stop_loss_pct,
                take_profit_pct=args.take_profit_pct,
                take_profit_price=args.take_profit_price,
                fill_fn=fill_fn,
            )
            print(
                f"exits: {res['stop_loss']} stop-loss / "
                f"{res['take_profit']} take-profit, "
                f"realized ${res['realized_pnl_usd']:.2f}"
            )
        except Exception as exc:  # noqa: BLE001 - exit rules must never crash the scan.
            logging.warning("exit-rules sweep failed (%s); continuing.", exc)

    # wallet -> WalletStats for the current roster, so each convergence candidate
    # can be scored on the win-rate of the wallets that actually converged.
    wallet_stats: Dict[str, Any] = {}
    # wallet -> [0,1] quality from profit-leaderboard rank (leaderboard mode);
    # empty in live mode, where confidence falls back to win-rate.
    wallet_quality: Dict[str, float] = {}

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
            # Quality = profit-leaderboard rank (these whales win on size, not
            # win-rate, so rank is the honest quality signal for confidence).
            wallet_quality.clear()
            wallet_quality.update(
                {w: leaderboard_quality(i, len(targets)) for i, w in enumerate(targets)}
            )
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
        wallet_quality.clear()  # live mode scores on win-rate, not rank
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
        # (0) Self-heal duplicate opens: retire any (market, outcome) that ended
        # up with two open records (e.g. a past concurrent-writer race) so the
        # rest of the scan — settle, exit rules, report — sees one per position
        # and never double-counts. Idempotent; a clean ledger heals zero.
        _heal_duplicate_opens(ledger)
        # (1) Settle resolved positions FIRST (before everything else): this
        # closes any market that has resolved since the last scan, freeing dedup
        # and surfacing realized P/L in the report.
        _settle()
        # (2) Early-exit sweep: cap losers / lock winners at the CURRENT mark
        # BEFORE opening new convergences, so the risk overlay runs on the
        # already-open book each scan (SHADOW-ONLY, NO orders).
        _apply_exit_rules()
        # (3) Convergence detection -> log new candidates (entry-filtered).
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
                wallet_quality=wallet_quality,
                blocklist=blocklist,
                dedup=dedup,
                min_confidence=args.min_confidence,
                min_entry_price=args.min_entry_price,
                max_entry_price=args.max_entry_price,
                require_directional=not args.allow_split_markets,
                book_entry=not args.no_book_entry,
                max_book_frac=args.max_book_frac,
                depth_band=args.depth_band,
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
                blocklist=blocklist,
                dedup=dedup,
                min_confidence=args.min_confidence,
                min_entry_price=args.min_entry_price,
                max_entry_price=args.max_entry_price,
                require_directional=not args.allow_split_markets,
                book_entry=not args.no_book_entry,
                max_book_frac=args.max_book_frac,
                depth_band=args.depth_band,
            )
        # (4) Report.
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
    #
    # Single-writer lock: two loops on one ledger are the ROOT CAUSE of duplicate
    # opens (each snapshots the open set before the other's append lands). Refuse
    # to start a second live writer on the same ledger.
    lock_path = _acquire_ledger_lock(args.ledger_path)
    if lock_path is None:
        return 1
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
    finally:
        _release_ledger_lock(lock_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
