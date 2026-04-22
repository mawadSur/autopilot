from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

QUADRANT_LABELS = {
    "deserved success": "Deserved Success",
    "good failure": "Good Failure",
    "dumb luck": "Dumb Luck",
    "poetic justice": "Poetic Justice",
}

DISPLAY_ORDER = (
    "Deserved Success",
    "Good Failure",
    "Dumb Luck",
    "Poetic Justice",
)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _extract_trades(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [trade for trade in payload if isinstance(trade, dict)]

    if isinstance(payload, dict):
        trades = payload.get("trades")
        if isinstance(trades, list):
            return [trade for trade in trades if isinstance(trade, dict)]

    raise ValueError("performance_audit.json must be a list of trades or contain a 'trades' list")


def _resolve_quadrant(trade: dict[str, Any]) -> str | None:
    candidate_fields = (
        trade.get("quadrant"),
        trade.get("audit_quadrant"),
        trade.get("category"),
        trade.get("result_quadrant"),
    )

    for candidate in candidate_fields:
        normalized = _normalize_text(candidate)
        if normalized in QUADRANT_LABELS:
            return QUADRANT_LABELS[normalized]

    return None


def _is_win(trade: dict[str, Any]) -> bool:
    for key in ("is_win", "won", "win"):
        value = trade.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        normalized = _normalize_text(value)
        if normalized in {"true", "yes", "win", "won", "success", "profitable", "profit"}:
            return True
        if normalized in {"false", "no", "loss", "lost", "fail", "failed", "unprofitable"}:
            return False

    pnl = trade.get("pnl")
    if isinstance(pnl, (int, float)):
        return pnl > 0

    outcome = _normalize_text(trade.get("outcome") or trade.get("result") or trade.get("status"))
    if outcome in {"win", "won", "success", "profitable", "profit"}:
        return True
    if outcome in {"loss", "lost", "failure", "failed", "unprofitable"}:
        return False

    return False


def _is_good_process(trade: dict[str, Any], quadrant: str | None) -> bool:
    for key in ("good_process", "is_good_process", "process_good"):
        value = trade.get(key)
        if isinstance(value, bool):
            return value
        normalized = _normalize_text(value)
        if normalized in {"true", "yes", "good", "pass"}:
            return True
        if normalized in {"false", "no", "bad", "fail"}:
            return False

    return quadrant in {"Deserved Success", "Good Failure"}


def _percentage(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100


def summarize_trades(trades: Iterable[dict[str, Any]]) -> None:
    trade_list = list(trades)
    total_trades = len(trade_list)

    quadrant_counts = {label: 0 for label in DISPLAY_ORDER}
    wins = 0
    good_process_total = 0

    for trade in trade_list:
        quadrant = _resolve_quadrant(trade)
        if quadrant:
            quadrant_counts[quadrant] += 1

        if _is_win(trade):
            wins += 1

        if _is_good_process(trade, quadrant):
            good_process_total += 1

    win_rate = _percentage(wins, total_trades)
    process_integrity_score = _percentage(good_process_total, total_trades)

    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(
        f"Process Integrity Score: {process_integrity_score:.2f}% "
        f"(Total Good Process / Total Trades)"
    )
    print("Quadrant Breakdown:")
    print(f"- [Green] Deserved Success: {quadrant_counts['Deserved Success']}")
    print(f"- [Yellow] Good Failure: {quadrant_counts['Good Failure']}")
    print(f"- [Red] Dumb Luck: {quadrant_counts['Dumb Luck']}")
    print(f"- [Dark Red] Poetic Justice: {quadrant_counts['Poetic Justice']}")

    dumb_luck_wins = quadrant_counts["Dumb Luck"]
    if wins > 0 and _percentage(dumb_luck_wins, wins) > 20:
        print("WARNING: SYSTEM IS RELYING ON LUCK. RE-AUDIT RESEARCH AGENTS.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read performance_audit.json and print a CLI analytics summary"
    )
    parser.add_argument(
        "--file",
        default="performance_audit.json",
        help="Path to performance_audit.json (default: performance_audit.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = Path(args.file)

    with audit_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    trades = _extract_trades(payload)
    summarize_trades(trades)


if __name__ == "__main__":
    main()
