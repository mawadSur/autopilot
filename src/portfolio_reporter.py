"""Shadow-portfolio reporting: mark the PnL ledger to market and post a
per-trade P/L + portfolio-value summary to Discord (and Telegram) via the
existing :class:`alerts.notifier.Notifier`.

This is OBSERVABILITY ONLY. It reads the append-only PnL ledger
(:mod:`state.pnl_ledger`), values open shadow positions against a caller-
supplied current price, and reports. It places NO orders and never mutates the
ledger.

Honesty notes (Constitution: honest reporting, no look-ahead):
    * In shadow mode a trade's REALIZED win/loss only exists once the underlying
      Polymarket market resolves and ``PnlLedger.settle()`` has recorded it.
      Until then this module reports an *unrealized* mark-to-market, clearly
      labelled, and never invents a realized number.
    * Positions we cannot price right now are reported as "pending" rather than
      marked at cost (which would fake a $0 P/L).

Pricing convention:
    A ``price_fn(record) -> Optional[float]`` returns the current per-unit value
    of the position's payoff in dollars [0, 1]:
        * intra-market arbitrage (``side == 'YES+NO'``): the YES+NO pair redeems
          for $1 at resolution, so the locked mark is ``1.0`` (pass 1.0).
        * a directional/whale position: the current price of the held outcome.
    Returning ``None`` marks the position "pending" (left unvalued).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

try:  # Flat import (PYTHONPATH=src), matching the rest of the codebase.
    from state.pnl_ledger import PnlLedger, TradeRecord
except ImportError:  # pragma: no cover - fallback for package-style import.
    from src.state.pnl_ledger import PnlLedger, TradeRecord  # type: ignore

__all__ = [
    "DEFAULT_BANKROLL_USD",
    "load_env_files",
    "mark_open_position",
    "build_report",
    "report_to_discord",
]


def load_env_files(paths: Sequence[str] = (".env", "src/.env")) -> None:
    """Best-effort: populate ``os.environ`` from ``.env`` files for any key not
    already set, so a CLI run (which doesn't go through pydantic-settings) can
    still pick up ``DISCORD_WEBHOOK_URL`` / ``TELEGRAM_*``. Dependency-free; an
    existing env var always wins over the file.
    """
    for path in paths:
        fp = Path(path)
        if not fp.is_file():
            continue
        try:
            for raw in fp.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except OSError:  # pragma: no cover - unreadable .env is non-fatal.
            continue

DEFAULT_BANKROLL_USD = 1000.0

# A pricing callback: given a ledger record, return its current per-unit mark in
# dollars [0, 1], or None when the position cannot be priced right now.
PriceFn = Callable[[TradeRecord], Optional[float]]


def _units(record: TradeRecord) -> float:
    """Payoff units implied by a USD-notional position bought at entry_price.

    ``size`` is USD notional spent; ``entry_price`` is the per-unit cost, so the
    number of payoff units is ``size / entry_price``. Returns 0.0 when entry
    price is non-positive (cannot derive units).
    """
    if record.entry_price and record.entry_price > 0:
        return record.size / record.entry_price
    return 0.0


def mark_open_position(
    record: TradeRecord,
    current_price: Optional[float],
) -> Dict[str, Any]:
    """Value one OPEN position at ``current_price``.

    Returns a dict with the entry, the mark, and the *unrealized* P/L net of the
    entry fee already recorded on the position. ``marked`` is False (and
    ``unrealized_pnl_usd`` is None) when ``current_price`` is None — i.e. the
    position is pending a price, not worth $0.
    """
    units = _units(record)
    if current_price is None or units <= 0.0:
        return {
            "trade_id": record.trade_id,
            "market_id": record.market_id,
            "title": (record.notes or record.market_id),
            "strategy": record.strategy,
            "side": record.side,
            "entry_price": record.entry_price,
            "size_usd": record.size,
            "current_price": current_price,
            "current_value_usd": None,
            "unrealized_pnl_usd": None,
            "marked": False,
        }
    current_value = units * current_price
    # P/L vs. cost basis (size), net of the entry fee already on the record.
    unrealized = current_value - record.size - float(record.fees_usd or 0.0)
    return {
        "trade_id": record.trade_id,
        "market_id": record.market_id,
        "title": (record.notes or record.market_id),
        "strategy": record.strategy,
        "side": record.side,
        "entry_price": record.entry_price,
        "size_usd": record.size,
        "current_price": current_price,
        "current_value_usd": current_value,
        "unrealized_pnl_usd": unrealized,
        "marked": True,
    }


def build_report(
    ledger: PnlLedger,
    *,
    price_fn: Optional[PriceFn] = None,
    bankroll_usd: float = DEFAULT_BANKROLL_USD,
) -> Dict[str, Any]:
    """Mark the ledger to market and return a structured portfolio report.

    Realized figures come straight from the ledger's settled records (no
    look-ahead). Unrealized figures come from ``price_fn`` over the open
    positions; positions ``price_fn`` cannot value are left "pending".
    """
    summary = ledger.summary()
    realized_pnl = float(summary.get("total_realized_pnl_usd", 0.0))

    open_rows: List[Dict[str, Any]] = []
    unrealized_total = 0.0
    n_pending = 0
    for record in ledger.open_positions():
        price = price_fn(record) if price_fn is not None else None
        row = mark_open_position(record, price)
        if row["marked"]:
            unrealized_total += float(row["unrealized_pnl_usd"])
        else:
            n_pending += 1
        open_rows.append(row)

    settled_rows: List[Dict[str, Any]] = []
    for record in ledger.settled():
        settled_rows.append(
            {
                "trade_id": record.trade_id,
                "market_id": record.market_id,
                "title": (record.notes or record.market_id),
                "strategy": record.strategy,
                "side": record.side,
                "realized_pnl_usd": float(record.realized_pnl_usd or 0.0),
                "market_outcome": record.market_outcome,
            }
        )

    equity = bankroll_usd + realized_pnl + unrealized_total
    return {
        "bankroll_usd": bankroll_usd,
        "realized_pnl_usd": realized_pnl,
        "unrealized_pnl_usd": unrealized_total,
        "total_pnl_usd": realized_pnl + unrealized_total,
        "equity_usd": equity,
        "n_open": int(summary.get("n_open", len(open_rows))),
        "n_settled": int(summary.get("n_settled", len(settled_rows))),
        "n_pending_mark": n_pending,
        "win_rate": float(summary.get("win_rate", 0.0)),
        "total_fees_usd": float(summary.get("total_fees_usd", 0.0)),
        "open_positions": open_rows,
        "settled_positions": settled_rows,
    }


def _fmt_usd(value: Optional[float]) -> str:
    if value is None:
        return "pending"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.4f}"


def _trade_fields(rows: List[Dict[str, Any]], *, max_shown: int) -> Dict[str, str]:
    """Build Discord embed fields, one per trade (capped)."""
    fields: Dict[str, str] = {}
    for row in rows[:max_shown]:
        title = str(row.get("title") or row.get("market_id") or "market")[:80]
        if "unrealized_pnl_usd" in row:  # open
            pl = row["unrealized_pnl_usd"]
            mark = row.get("current_price")
            mark_str = "pending" if mark is None else f"${mark:.3f}"
            value = (
                f"{row['side']} | entry ${row['entry_price']:.3f} -> {mark_str} "
                f"| unrl {_fmt_usd(pl)}"
            )
        else:  # settled
            value = f"{row['side']} | realized {_fmt_usd(row.get('realized_pnl_usd'))}"
        # Discord rejects duplicate-name collisions silently; suffix the id tail.
        key = f"{title} ({str(row.get('trade_id', ''))[-4:]})"
        fields[key[:256]] = value[:1024]
    if len(rows) > max_shown:
        fields[f"(+{len(rows) - max_shown} more)"] = "trimmed for Discord limits"
    return fields


def report_to_discord(
    ledger: PnlLedger,
    notifier: Any,
    *,
    price_fn: Optional[PriceFn] = None,
    bankroll_usd: float = DEFAULT_BANKROLL_USD,
    max_trades_shown: int = 10,
    label: str = "Shadow",
) -> Dict[str, Any]:
    """Post a per-trade P/L breakdown + portfolio summary to Discord/Telegram.

    Reuses :class:`alerts.notifier.Notifier` (``notifier.info`` / ``daily_summary``).
    The notifier degrades to a no-op when no channel is configured, so this is
    safe to call unconditionally. Returns the report dict (also usable headless).
    """
    report = build_report(ledger, price_fn=price_fn, bankroll_usd=bankroll_usd)

    # 1) Per-trade P/L (open first, then settled), capped for Discord field limits.
    trade_rows = report["open_positions"] + report["settled_positions"]
    if trade_rows:
        notifier.info(
            f"{label} trades — {report['n_open']} open / {report['n_settled']} settled"
            + (f" ({report['n_pending_mark']} pending price)" if report["n_pending_mark"] else ""),
            fields=_trade_fields(trade_rows, max_shown=max_trades_shown),
        )

    # 2) Portfolio value summary.
    notifier.info(
        f"{label} Portfolio — equity ${report['equity_usd']:,.2f}",
        fields={
            "Bankroll": f"${report['bankroll_usd']:,.2f}",
            "Realized P/L": _fmt_usd(report["realized_pnl_usd"]),
            "Unrealized P/L": _fmt_usd(report["unrealized_pnl_usd"]),
            "Equity": f"${report['equity_usd']:,.2f}",
            "Open / Settled": f"{report['n_open']} / {report['n_settled']}",
            "Win Rate": f"{report['win_rate'] * 100:.1f}%",
            "Fees Paid": f"${report['total_fees_usd']:,.4f}",
        },
    )
    return report
