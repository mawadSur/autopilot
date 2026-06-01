"""Pure, testable view-state builder for the SHADOW dashboard.

This module turns a :class:`state.pnl_ledger.PnlLedger` into a single JSON-able
``dict`` the web UI can render directly. It is intentionally free of any sockets,
HTTP, or I/O beyond reading the ledger through the already-audited
:func:`portfolio_reporter.build_report` and the ledger's own read methods — so it
can be unit-tested without standing up a server.

OBSERVABILITY ONLY: nothing here places orders or mutates the ledger. We read the
ledger and shape it for display; that is the entire contract.

Key shaping decisions
---------------------
* ``build_report`` is called with ``price_fn=None`` on purpose: we never fetch a
  live mark on the request path (that would make every page-load slow and hit the
  network). Open positions are therefore shown at their entry with their recorded
  confidence; unrealized P/L stays "pending".
* The exit-mix / equity-curve are derived from the *settled* records only, which
  are the honest realized track record (no look-ahead — every settlement already
  happened at an observed mark or a real resolution).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Optional

# Deliberate sys.path bootstrap (matches main.py / src/orchestrator.py): ensure
# the repo's ``src/`` is importable so the flat ``from portfolio_reporter import
# ...`` / ``from state.pnl_ledger import ...`` imports resolve even when this file
# is run via a script path (``python src/dashboard/server.py``), where sys.path[0]
# is ``src/dashboard`` rather than ``src``. Read-only; touches no files.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from portfolio_reporter import DEFAULT_BANKROLL_USD, build_report  # noqa: E402
from state.pnl_ledger import PnlLedger, TradeRecord  # noqa: E402

__all__ = ["build_state", "parse_confidence", "clean_title", "normalize_exit_reason"]

# Recovers a "confidence=0.65 (medium)" marker written into a record's notes by
# the whale-convergence runner. Group 1 = numeric score, group 2 = label.
_CONFIDENCE_RE = re.compile(r"confidence=([0-9.]+) \(([a-z]+)\)")

# An optional "title=..." marker in notes, should the runner ever embed a clean
# human title. Today's notes don't carry one, so we fall back to a short id.
_TITLE_RE = re.compile(r"title=([^;]+)")


def parse_confidence(notes: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse ``confidence=0.XX (label)`` out of a record's notes.

    Returns ``{"score": float, "label": str}`` or ``None`` when no confidence
    marker is present (or notes is missing).
    """
    if not notes:
        return None
    match = _CONFIDENCE_RE.search(notes)
    if not match:
        return None
    try:
        score = float(match.group(1))
    except (TypeError, ValueError):
        return None
    return {"score": score, "label": match.group(2)}


def clean_title(*, notes: Optional[str], market_id: Optional[str]) -> str:
    """Best clean display title for a position.

    Prefers an explicit ``title=...`` marker in notes if present; otherwise falls
    back to a short, readable slice of the (typically hex) ``market_id``.
    """
    if notes:
        match = _TITLE_RE.search(notes)
        if match:
            title = match.group(1).strip()
            if title:
                return title
    mid = (market_id or "").strip()
    if not mid:
        return "market"
    # Hex condition ids are long; show a recognizable head…tail slug.
    if len(mid) > 16:
        return f"{mid[:10]}…{mid[-4:]}"
    return mid


def normalize_exit_reason(
    market_outcome: Optional[Any],
    realized_pnl_usd: Optional[float],
) -> str:
    """Normalize a settled record into one of four exit-mix buckets.

    ``market_outcome`` strings that start with ``"exit:"`` (written by the exit
    rules) map to ``"take_profit"`` / ``"stop_loss"`` (anything else after the
    prefix falls back to the raw reason). All other settlements are *resolutions*:
    a positive realized P/L is ``"won"``, otherwise ``"lost"``.
    """
    outcome = market_outcome if isinstance(market_outcome, str) else ""
    if outcome.startswith("exit:"):
        reason = outcome[len("exit:"):].strip()
        return reason or "exit"
    pnl = realized_pnl_usd if isinstance(realized_pnl_usd, (int, float)) else 0.0
    return "won" if pnl > 0 else "lost"


def _exit_ts_key(value: Optional[str]) -> str:
    """Sort key for ISO-8601 exit timestamps that tolerates ``None``.

    ISO-8601 sorts lexicographically in chronological order, so the raw string is
    a fine key; ``None``/missing timestamps sort first (treated as oldest).
    """
    return value or ""


def build_state(
    ledger: PnlLedger,
    *,
    bankroll_usd: float = DEFAULT_BANKROLL_USD,
) -> Dict[str, Any]:
    """Build the full dashboard view-state from ``ledger``.

    Returns a JSON-able dict with ``summary``, ``open_positions``,
    ``closed_positions`` (exit feed, newest first), ``exit_mix``, and
    ``equity_curve`` (realized track record over time). Robust to missing/None
    fields on any record.
    """
    report = build_report(ledger, price_fn=None, bankroll_usd=bankroll_usd)

    summary = {
        "equity_usd": float(report.get("equity_usd", bankroll_usd)),
        "realized_pnl_usd": float(report.get("realized_pnl_usd", 0.0)),
        "unrealized_pnl_usd": float(report.get("unrealized_pnl_usd", 0.0)),
        "win_rate": float(report.get("win_rate", 0.0)),
        "n_open": int(report.get("n_open", 0)),
        "n_settled": int(report.get("n_settled", 0)),
        "total_fees_usd": float(report.get("total_fees_usd", 0.0)),
        "bankroll_usd": float(bankroll_usd),
    }

    # --- Open positions: clean title + parsed confidence (entry only, no mark) ---
    open_positions: List[Dict[str, Any]] = []
    for record in ledger.open_positions():
        notes = getattr(record, "notes", "") or ""
        market_id = getattr(record, "market_id", "") or ""
        entry_price = getattr(record, "entry_price", None)
        open_positions.append(
            {
                "title": clean_title(notes=notes, market_id=market_id),
                "side": getattr(record, "side", "") or "",
                "entry_price": (
                    float(entry_price) if isinstance(entry_price, (int, float)) else None
                ),
                "confidence": parse_confidence(notes),
                "strategy": getattr(record, "strategy", "") or "",
            }
        )

    # --- Settled records: shared source for the exit feed, mix, and curve. ---
    settled: List[TradeRecord] = list(ledger.settled())

    closed_positions: List[Dict[str, Any]] = []
    exit_mix: Dict[str, int] = {}
    for record in settled:
        notes = getattr(record, "notes", "") or ""
        market_id = getattr(record, "market_id", "") or ""
        realized = getattr(record, "realized_pnl_usd", None)
        realized_f = float(realized) if isinstance(realized, (int, float)) else 0.0
        market_outcome = getattr(record, "market_outcome", None)
        reason = normalize_exit_reason(market_outcome, realized_f)
        exit_mix[reason] = exit_mix.get(reason, 0) + 1
        closed_positions.append(
            {
                "title": clean_title(notes=notes, market_id=market_id),
                "side": getattr(record, "side", "") or "",
                "realized_pnl_usd": realized_f,
                "market_outcome": market_outcome,
                "reason": reason,
                "exit_ts_utc": getattr(record, "exit_ts_utc", None),
            }
        )

    # Exit feed: newest first.
    closed_positions.sort(key=lambda row: _exit_ts_key(row["exit_ts_utc"]), reverse=True)

    # --- Equity curve: realized track record, accumulated oldest -> newest. ---
    settled_asc = sorted(
        settled, key=lambda r: _exit_ts_key(getattr(r, "exit_ts_utc", None))
    )
    equity_curve: List[Dict[str, Any]] = [
        {"t": "start", "equity": float(bankroll_usd)}
    ]
    running = float(bankroll_usd)
    for record in settled_asc:
        realized = getattr(record, "realized_pnl_usd", None)
        running += float(realized) if isinstance(realized, (int, float)) else 0.0
        equity_curve.append(
            {"t": getattr(record, "exit_ts_utc", None), "equity": running}
        )

    return {
        "summary": summary,
        "open_positions": open_positions,
        "closed_positions": closed_positions,
        "exit_mix": exit_mix,
        "equity_curve": equity_curve,
    }
