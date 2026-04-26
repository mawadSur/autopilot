"""Mark an orchestrator trade execution log as settled.

Used to drive the outcome-review loop manually without waiting for a real
Polymarket settlement. Mutates a single ``trade_execution_*.json`` in place.

Example:
    ./.venv/bin/python src/mark_trade_settled.py path/to/trade_execution_mkt-1.json \\
        --outcome win --news "Final results confirmed by AP."
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _compute_realized_pnl(
    *,
    final_outcome: bool,
    entry_price: float,
    position_size_usd: float,
    exit_price: Optional[float],
) -> tuple[float, float]:
    """Compute (resolved_exit_price, realized_pnl_usd) for an always-long-YES trade.

    YES contracts settle at $1.00 on win and $0.00 on loss. When ``exit_price``
    is provided it overrides the binary settlement (handy for trades that were
    sold prior to resolution); otherwise the binary value is used directly.
    """
    if exit_price is None:
        resolved_exit = 1.0 if final_outcome else 0.0
    else:
        resolved_exit = float(exit_price)

    if final_outcome:
        # Long YES: contract value moves from entry_price toward $1.00.
        # PnL = position_size_usd * (resolved_exit - entry_price) / entry_price.
        # When resolved_exit == 1.0 this collapses to (1 - entry_price)/entry_price.
        if entry_price <= 0.0:
            realized_pnl = 0.0
        else:
            realized_pnl = position_size_usd * (resolved_exit - entry_price) / entry_price
    else:
        if exit_price is None:
            # Binary loss: contract worth $0, so we lose the full notional.
            realized_pnl = -float(position_size_usd)
        elif entry_price <= 0.0:
            realized_pnl = 0.0
        else:
            # Trader-reported exit on a losing trade (e.g. cut early).
            realized_pnl = position_size_usd * (resolved_exit - entry_price) / entry_price

    return resolved_exit, realized_pnl


def mark_settled(
    trade_log_path: Path,
    *,
    final_outcome: bool,
    post_settlement_news: Optional[str] = None,
    settled_at: Optional[str] = None,
    market_outcome: Optional[bool] = None,
    exit_price: Optional[float] = None,
    realized_pnl_usd: Optional[float] = None,
) -> dict:
    """Update a trade execution log in place. Returns the mutated payload.

    ``final_outcome`` records whether the trade we took won.
    ``market_outcome`` records whether the market resolved YES, independent of
    which side we took. The two fields are equal only when we always go long
    YES; when ``market_outcome`` is omitted we leave the slot untouched.

    ``exit_price`` (optional) overrides the binary YES settlement (1.0 / 0.0)
    and is useful when a position was closed before resolution. When omitted,
    the binary settlement price implied by ``final_outcome`` is recorded.

    ``realized_pnl_usd`` (optional) lets callers force a PnL value rather than
    deriving it from entry_price + position_size_usd. Logs missing those two
    fields skip PnL derivation entirely (legacy log compatibility).
    """
    with trade_log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            f"{trade_log_path} does not contain a JSON object (got {type(payload).__name__})"
        )

    payload["status"] = "settled"
    payload["final_outcome"] = bool(final_outcome)
    payload["settled_at"] = settled_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    if post_settlement_news is not None:
        payload["post_settlement_news"] = post_settlement_news
    if market_outcome is not None:
        payload["market_outcome"] = bool(market_outcome)

    entry_price_raw = payload.get("entry_price")
    position_size_raw = payload.get("position_size_usd")

    if realized_pnl_usd is not None:
        payload["realized_pnl_usd"] = float(realized_pnl_usd)
        if exit_price is not None:
            payload["exit_price"] = float(exit_price)
        elif "exit_price" not in payload or payload.get("exit_price") is None:
            payload["exit_price"] = 1.0 if final_outcome else 0.0
    elif entry_price_raw is None or position_size_raw is None:
        # Legacy log: skip derivation rather than fabricate PnL.
        print(
            "warning: trade log missing entry_price/position_size_usd -- "
            "skipping realized PnL computation (legacy log).",
            file=sys.stderr,
        )
        if exit_price is not None:
            payload["exit_price"] = float(exit_price)
        else:
            payload.setdefault("exit_price", None)
        payload.setdefault("realized_pnl_usd", None)
    else:
        try:
            entry_price_value = float(entry_price_raw)
            position_size_value = float(position_size_raw)
        except (TypeError, ValueError):
            print(
                "warning: trade log has non-numeric entry_price/position_size_usd "
                "-- skipping realized PnL computation.",
                file=sys.stderr,
            )
            if exit_price is not None:
                payload["exit_price"] = float(exit_price)
            else:
                payload.setdefault("exit_price", None)
            payload.setdefault("realized_pnl_usd", None)
        else:
            resolved_exit, realized_pnl = _compute_realized_pnl(
                final_outcome=bool(final_outcome),
                entry_price=entry_price_value,
                position_size_usd=position_size_value,
                exit_price=exit_price,
            )
            payload["exit_price"] = resolved_exit
            payload["realized_pnl_usd"] = realized_pnl

    with trade_log_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return payload


def _parse_outcome(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"win", "won", "true", "yes", "1"}:
        return True
    if normalized in {"loss", "lost", "false", "no", "0"}:
        return False
    raise argparse.ArgumentTypeError(
        f"--outcome must be one of: win|loss|true|false|yes|no|1|0 (got {value!r})"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trade_log_path", type=Path, help="Path to trade_execution_<id>.json")
    parser.add_argument(
        "--outcome",
        type=_parse_outcome,
        required=True,
        help="Final outcome: win|loss (also accepts true/false, yes/no, 1/0).",
    )
    parser.add_argument(
        "--market-outcome",
        dest="market_outcome",
        type=_parse_outcome,
        default=None,
        help=(
            "Did the market resolve YES? Accepts yes|no|true|false|1|0. "
            "Defaults to --outcome (assumes always-long-YES); pass explicitly "
            "when trading both sides so the XGBoost label is correct."
        ),
    )
    parser.add_argument(
        "--news",
        dest="post_settlement_news",
        default=None,
        help="Post-settlement news context to feed the OutcomeReviewAgent.",
    )
    parser.add_argument(
        "--settled-at",
        default=None,
        help="ISO-8601 settlement timestamp (defaults to now).",
    )
    parser.add_argument(
        "--exit-price",
        dest="exit_price",
        type=float,
        default=None,
        help=(
            "Optional exit price (0.0-1.0) for trades closed before resolution. "
            "When omitted, the binary YES settlement price (1.0 win / 0.0 loss) "
            "is recorded."
        ),
    )
    args = parser.parse_args(argv)

    if not args.trade_log_path.is_file():
        print(f"error: trade log not found: {args.trade_log_path}", file=sys.stderr)
        return 2

    if args.market_outcome is None:
        market_outcome = bool(args.outcome)
        print(
            "warning: market_outcome defaulted to final_outcome -- assumes "
            "always-long-YES; pass --market-outcome explicitly when trading "
            "both sides.",
            file=sys.stderr,
        )
    else:
        market_outcome = args.market_outcome

    payload = mark_settled(
        args.trade_log_path,
        final_outcome=args.outcome,
        post_settlement_news=args.post_settlement_news,
        settled_at=args.settled_at,
        market_outcome=market_outcome,
        exit_price=args.exit_price,
    )
    summary = (
        f"Marked {args.trade_log_path.name} as settled "
        f"(outcome={'win' if payload['final_outcome'] else 'loss'}, "
        f"market_outcome={'yes' if payload['market_outcome'] else 'no'}, "
        f"settled_at={payload['settled_at']}"
    )
    realized_pnl = payload.get("realized_pnl_usd")
    if realized_pnl is not None:
        summary += f", realized_pnl_usd={float(realized_pnl):+.2f}"
    summary += ")."
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
