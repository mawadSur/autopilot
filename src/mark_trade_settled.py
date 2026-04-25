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


def mark_settled(
    trade_log_path: Path,
    *,
    final_outcome: bool,
    post_settlement_news: Optional[str] = None,
    settled_at: Optional[str] = None,
) -> dict:
    """Update a trade execution log in place. Returns the mutated payload."""
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
    args = parser.parse_args(argv)

    if not args.trade_log_path.is_file():
        print(f"error: trade log not found: {args.trade_log_path}", file=sys.stderr)
        return 2

    payload = mark_settled(
        args.trade_log_path,
        final_outcome=args.outcome,
        post_settlement_news=args.post_settlement_news,
        settled_at=args.settled_at,
    )
    print(
        f"Marked {args.trade_log_path.name} as settled "
        f"(outcome={'win' if payload['final_outcome'] else 'loss'}, "
        f"settled_at={payload['settled_at']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
