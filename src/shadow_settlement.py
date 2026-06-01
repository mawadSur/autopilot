"""Shadow settlement — turn the open whale-follow ledger into a TRACK RECORD.

The whale-convergence shadow runner logs open ``whale_convergence`` candidates
to the PnL ledger but never closes them, so the ledger only ever shows
unrealized marks. This module is the missing other half: it sweeps the open
positions, asks Polymarket whether each position's market has RESOLVED, and —
only for the resolved ones — settles them into the ledger with a real realized
PnL. That converts the running shadow into a forward-validated track record the
edge can be judged on before any capital is ever risked.

SCOPE — SHADOW-ONLY, NO MONEY MOVES (Constitution: safety first):
    Settlement is pure bookkeeping over the append-only ledger. It reads the
    CLOB ``/markets/<conditionId>`` resolution endpoint (via an injected
    ``resolver`` callable) and appends ``settle`` events to the ledger. It
    places NO orders, signs nothing, redeems nothing on-chain, and touches no
    wallet. The realized PnL it records is the PnL a paper position WOULD have
    booked — never a real fill.

No look-ahead / honest reporting:
    A position is settled ONLY when the resolver reports ``closed: true`` for
    its market — an open market is left open (no fabricating an outcome from a
    not-yet-resolved book). The ledger's own settle guard additionally rejects
    any ``exit_ts_utc`` before the entry time. When a position has no usable
    entry price (``entry_price <= 0`` — the ``/holders`` path that records
    ``0.0`` when no current mark is available) we CANNOT compute a realized PnL
    honestly, so we skip it and count it ``unpriced`` rather than fabricate a
    number.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


__all__ = [
    "compute_settlement_pnl",
    "settle_resolved_positions",
]

# Pulls the ``outcomeIndex=<n>`` marker the runner writes into a record's notes.
# Kept in sync with whale_follow_runner._OUTCOME_INDEX_RE.
_OUTCOME_INDEX_RE = re.compile(r"outcomeIndex=(\d+)")

# Polymarket charges no fee to enter/exit a position on the CLOB, but applies a
# settlement fee (~2%, i.e. 200 bps) to the WINNING gross redemption payout. We
# model only that: a losing position simply forfeits its stake (no fee on $0).
DEFAULT_FEE_BPS = 200.0


def compute_settlement_pnl(
    entry_price: float,
    size_usd: float,
    won: bool,
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> float:
    """Realized PnL (USD) of a shadow outcome-share position at resolution.

    A shadow position buys ``size_usd`` of notional at ``entry_price`` dollars
    per outcome share, so it holds ``units = size_usd / entry_price`` shares.
    At resolution each share redeems for $1 if the outcome WON, or $0 if it
    lost.

    Fee assumption (Polymarket): there is no fee to enter or exit a position,
    but a settlement fee of ``fee_bps`` basis points (default 200 bps = ~2%)
    is applied to the WINNING GROSS PAYOUT only. A losing position pays no fee —
    it simply forfeits the stake. Hence::

        won  -> units * 1.0 * (1 - fee_bps / 10_000) - size_usd
        lost -> -size_usd

    Returns ``0.0`` when the position cannot be sized — ``entry_price <= 0``
    (the ``/holders`` unmarkable case) or ``size_usd <= 0`` — so the caller can
    skip it rather than fabricate a realized number. Rounded to 4 decimal
    places (sub-cent) to keep the ledger tidy.

    Args:
        entry_price: Dollars per outcome share paid at entry, in ``(0, 1]``.
        size_usd: USD notional staked on the position.
        won: Whether the held outcome resolved as the winner.
        fee_bps: Settlement fee on the winning gross payout, in basis points.

    Returns:
        Realized PnL in USD (positive for a win net of fee, ``-size_usd`` for a
        loss, ``0.0`` when unsizable).
    """
    if entry_price <= 0 or size_usd <= 0:
        return 0.0

    if not won:
        return round(-float(size_usd), 4)

    units = float(size_usd) / float(entry_price)
    gross_payout = units * 1.0
    net_payout = gross_payout * (1.0 - float(fee_bps) / 10_000.0)
    return round(net_payout - float(size_usd), 4)


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (settlement/exit timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def settle_resolved_positions(
    ledger: Any,
    resolver: Callable[[str], Optional[Dict[str, Any]]],
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
    now_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Settle every open ledger position whose Polymarket market has RESOLVED.

    Sweeps ``ledger.open_positions()`` and, for each, asks ``resolver`` whether
    that position's market (``record.market_id`` = the ``conditionId``) has
    resolved. ``resolver`` is a ``callable(market_id) -> resolution-dict | None``
    (so tests inject a fake; production passes
    :func:`exchanges.polymarket_market_data.get_market_resolution`). The
    resolution dict is shaped ``{"closed": bool, "tokens": [{"winner": bool,
    "price": float, ...}, ...]}`` with ``tokens`` ordered to match
    ``outcomeIndex``.

    Per-position handling (one bad market never aborts the sweep — each is
    wrapped so a resolver/settle error is logged and counted, then skipped):

      * resolver returns ``None`` OR ``closed`` is falsy -> leave OPEN
        (``still_open``); an unresolved market must not be settled (no
        look-ahead).
      * ``outcomeIndex`` out of range for ``tokens`` (or unparseable from the
        notes) -> skip + count ``errors``.
      * ``entry_price <= 0`` -> skip + count ``unpriced``; we never fabricate a
        realized PnL for an unmarkable entry.
      * otherwise -> ``won = bool(token["winner"])`` (falling back to
        ``token["price"] >= 0.5`` if ``winner`` is absent/false-y but the price
        marks the outcome a winner), compute the realized PnL via
        :func:`compute_settlement_pnl`, and append a ``settle`` event with
        ``exit_price = 1.0`` (won) / ``0.0`` (lost), ``market_outcome =
        "won:<side>"`` / ``"lost:<side>"`` and ``realized_pnl_usd``. The
        ledger's own guard rejects an ``exit_ts`` before entry.

    Args:
        ledger: A :class:`state.pnl_ledger.PnlLedger` (or compatible) exposing
            ``open_positions()`` and ``settle(...)``.
        resolver: ``callable(market_id) -> resolution-dict | None``.
        fee_bps: Settlement fee passed through to
            :func:`compute_settlement_pnl`.
        now_iso: Exit timestamp to record (ISO-8601 UTC). Defaults to now. Must
            be at/after each record's entry time or the ledger guard rejects it.

    Returns:
        A counts dict::

            {"settled", "won", "lost", "still_open", "unpriced", "errors",
             "total_realized_pnl_usd"}
    """
    exit_ts = now_iso or _utc_now_iso()
    counts = {
        "settled": 0,
        "won": 0,
        "lost": 0,
        "still_open": 0,
        "unpriced": 0,
        "errors": 0,
        "total_realized_pnl_usd": 0.0,
    }

    try:
        open_positions = ledger.open_positions()
    except Exception as exc:  # pragma: no cover - never let a read crash the sweep.
        logging.warning("shadow settlement: open_positions() failed (%s)", exc)
        return counts

    for record in open_positions:
        try:
            resolution = resolver(record.market_id)
        except Exception as exc:  # noqa: BLE001 - one bad market must not abort the sweep.
            logging.warning(
                "shadow settlement: resolver(%s) raised (%s); leaving open.",
                record.market_id,
                exc,
            )
            counts["errors"] += 1
            continue

        if not resolution or not resolution.get("closed"):
            counts["still_open"] += 1
            continue

        match = _OUTCOME_INDEX_RE.search(record.notes or "")
        if match is None:
            logging.warning(
                "shadow settlement: no outcomeIndex in notes for trade %s; skipping.",
                record.trade_id,
            )
            counts["errors"] += 1
            continue
        outcome_index = int(match.group(1))

        tokens = resolution.get("tokens") or []
        if not (0 <= outcome_index < len(tokens)):
            logging.warning(
                "shadow settlement: outcomeIndex %d out of range (%d tokens) "
                "for trade %s; skipping.",
                outcome_index,
                len(tokens),
                record.trade_id,
            )
            counts["errors"] += 1
            continue
        token = tokens[outcome_index]

        # Prefer the explicit winner flag; fall back to the resolved price ONLY
        # when it marks an unambiguous win (a closed market marks the winning
        # token at exactly 1.0). The >=0.999 bound avoids mislabeling a
        # degenerate closed-but-oddly-priced book as a win.
        won = bool(token.get("winner"))
        if not won:
            price = token.get("price")
            if isinstance(price, (int, float)) and price >= 0.999:
                won = True

        if record.entry_price is None or record.entry_price <= 0:
            # Unmarkable entry (e.g. the /holders path that records 0.0): we
            # cannot compute a realized PnL honestly, so never fabricate one.
            counts["unpriced"] += 1
            continue

        pnl = compute_settlement_pnl(
            float(record.entry_price), float(record.size), won, fee_bps=fee_bps
        )

        try:
            ledger.settle(
                record.trade_id,
                exit_price=(1.0 if won else 0.0),
                exit_ts_utc=exit_ts,
                market_outcome=(f"won:{record.side}" if won else f"lost:{record.side}"),
                realized_pnl_usd=pnl,
            )
        except Exception as exc:  # noqa: BLE001 - a single settle failure must not abort the sweep.
            logging.warning(
                "shadow settlement: settle(%s) failed (%s); leaving open.",
                record.trade_id,
                exc,
            )
            counts["errors"] += 1
            continue

        counts["settled"] += 1
        counts["won" if won else "lost"] += 1
        counts["total_realized_pnl_usd"] += pnl

    counts["total_realized_pnl_usd"] = round(counts["total_realized_pnl_usd"], 4)
    return counts
