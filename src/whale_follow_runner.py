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
    no cheap current price available at convergence time. Rather than fabricate
    a fill price, we record ``entry_price = 0.0`` and flag it in the trade
    notes; a later settlement supplies the real outcome price. This keeps the
    ledger honest (no invented fills) per the no-look-ahead standard.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from state.pnl_ledger import DEFAULT_LEDGER_PATH, PnlLedger, TradeRecord
except Exception:  # pragma: no cover - import shim for alternate layouts.
    from src.state.pnl_ledger import (  # type: ignore
        DEFAULT_LEDGER_PATH,
        PnlLedger,
        TradeRecord,
    )


__all__ = [
    "STRATEGY",
    "find_convergence",
    "run_once",
    "main",
]


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


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (decision/entry timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def run_once(
    *,
    ledger: PnlLedger,
    client: Optional[Any] = None,
    target_wallets: Sequence[str],
    markets_condition_ids: Sequence[str],
    min_convergence: int = 3,
    size_usd: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run one convergence scan and log each candidate to the SHADOW ledger.

    For every convergence candidate from :func:`find_convergence`, append an
    OPEN :class:`TradeRecord` to ``ledger`` (``venue='polymarket'``,
    ``strategy='whale_convergence'``, ``side`` = the converged outcome label).
    The notes flag SHADOW MODE and list the contributing wallets so the
    candidate is auditable. NO orders are placed — the client is read-only.

    Entry price is recorded as ``0.0`` with a note, because ``/holders`` carries
    no price; see the module docstring.

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

    Returns:
        The list of convergence candidates that were logged.
    """
    if client is None:
        raise ValueError("run_once requires a read-only data-api client")

    candidates = find_convergence(
        client,
        target_wallets,
        markets_condition_ids=markets_condition_ids,
        min_convergence=min_convergence,
    )

    for cand in candidates:
        outcome_label = cand.get("outcome")
        side = str(outcome_label) if outcome_label is not None else str(
            cand.get("outcomeIndex")
        )
        wallets = cand.get("wallets") or []
        notes = (
            "SHADOW MODE - NO ORDERS; "
            f"whale_convergence n={cand.get('n_target_holders')} "
            f"outcomeIndex={cand.get('outcomeIndex')}; "
            "entry_price=0.0 (holders endpoint carries no price); "
            f"wallets={','.join(wallets)}"
        )
        record = TradeRecord(
            trade_id=f"whale-{uuid.uuid4().hex[:12]}",
            ts_utc=_utc_now_iso(),
            venue="polymarket",
            market_id=str(cand.get("conditionId")),
            side=side,
            entry_price=0.0,
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
            f"wallets={cand.get('wallets')}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: rank smart-money wallets, scan hot markets for convergence, log to
    the SHADOW ledger, and optionally post per-trade P/L + portfolio to Discord.

    Full pipeline, READ-ONLY + SHADOW: ranks wallets from the public data-api,
    derives candidate markets from recent trades, flags convergence, and logs
    candidates. NO orders are ever placed.
    """
    parser = argparse.ArgumentParser(
        description="SHADOW whale-convergence follower (logs candidates, places NO orders)."
    )
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
    parser.add_argument("--size", type=float, default=0.0,
                        help="USD notional label for shadow records (default 0.0).")
    parser.add_argument("--ledger-path", type=str, default=DEFAULT_LEDGER_PATH,
                        help=f"Path to the JSONL PnL ledger (default {DEFAULT_LEDGER_PATH}).")
    parser.add_argument("--discord", action="store_true",
                        help="Post per-trade P/L + portfolio value to Discord "
                        "(reads DISCORD_WEBHOOK_URL; no-op if unset).")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Paper bankroll (USD) baseline for the portfolio report.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Local imports keep the module importable without the data-api/ranker in
    # minimal/test environments (the unit tests inject fakes instead).
    from exchanges.polymarket_data_api import PolymarketDataAPIClient
    import wallet_ranker

    client = PolymarketDataAPIClient()

    # 1) Rank smart-money wallets (bounded discovery + per-wallet positions).
    discovered = wallet_ranker.discover_active_wallets(client, pages=args.discover_pages)
    ranked = wallet_ranker.rank_wallets(
        client,
        candidate_wallets=discovered,
        min_settled=args.min_settled,
        min_win_rate=args.min_win_rate,
        top_n=args.top_wallets,
    )
    targets = [w.wallet for w in ranked]
    print(f"ranked {len(targets)} smart-money wallet(s) from {len(discovered)} active")

    # 2) Candidate markets = the hottest recent markets (by trade recency).
    market_ids: List[str] = []
    for trade in client.get_trades(limit=min(500, max(args.max_markets * 10, 100))):
        cid = trade.get("conditionId") if isinstance(trade, dict) else None
        if cid and cid not in market_ids:
            market_ids.append(cid)
        if len(market_ids) >= args.max_markets:
            break

    ledger = PnlLedger(args.ledger_path)
    run_once(
        ledger=ledger,
        client=client,
        target_wallets=targets,
        markets_condition_ids=market_ids,
        min_convergence=args.min_convergence,
        size_usd=args.size,
    )

    if args.discord:
        try:
            from portfolio_reporter import load_env_files, report_to_discord
            from alerts.notifier import Notifier
            load_env_files()  # pick up DISCORD_WEBHOOK_URL from .env for CLI runs
            report_to_discord(
                ledger, Notifier(), bankroll_usd=args.bankroll,
                label="Whale-follow shadow",
            )
        except ImportError:
            logging.warning(
                "Discord reporting unavailable (alerts/portfolio_reporter import failed)."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
