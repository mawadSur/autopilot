"""Shadow early-exit rules — cap the loss, lock the win (SHADOW-ONLY, NO ORDERS).

The whale-follow shadow loop's last live settlement batch realized -$917 over 18
trades (9 won / 9 lost): the LOSERS rode all the way to $0 (a full -$100 stake)
while the WINNERS were capped at resolution. That asymmetry — symmetric hit-rate,
asymmetric realized P/L — is exactly what a stop-loss / take-profit overlay fixes.

This module sweeps the OPEN ledger positions, re-prices each to the CURRENT
market mark (via the injected ``price_fn`` — the same decision-time
:func:`whale_follow_runner.make_whale_price_fn` re-pricer used by the portfolio
report), and settles the ones that have hit a stop-loss or take-profit threshold
at that current mark. A losing position is closed early at the mark instead of
forfeiting the whole stake at $0; a winning position is locked in instead of
risking a reversal before resolution.

SCOPE — SHADOW-ONLY, NO MONEY MOVES (Constitution: safety first):
    These rules are pure bookkeeping over the append-only ledger. They read the
    current observable mark through ``price_fn`` and append ``settle`` events.
    They place NO orders, sign nothing, redeem nothing on-chain, and touch no
    wallet. The realized P/L recorded is the P/L a paper position WOULD have
    booked by selling at the current mark — never a real fill.

No look-ahead / honest reporting:
    Exit decisions use ONLY the current observable mark (``price_fn(record)`` is
    a decision-time re-price, NOT a future price). A position with no current
    mark (``price_fn`` returns ``None``) or an out-of-range mark is LEFT OPEN
    rather than settled on a guessed price. The settlement is stamped "now",
    which is at/after the entry time, so the ledger's own no-look-ahead guard
    (``exit_ts >= entry ts``) is satisfied.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

try:  # Flat import under PYTHONPATH=src (matches the rest of the stack).
    from state.pnl_ledger import dedupe_open_positions
except Exception:  # pragma: no cover - alternate layout shim.
    from src.state.pnl_ledger import dedupe_open_positions  # type: ignore


__all__ = [
    "evaluate_exit",
    "compute_exit_pnl",
    "apply_exit_rules",
]

# Polymarket-style proceeds haircut on an early exit: ~2% fee/slippage on the
# gross proceeds of selling the outcome token at the current mark. Conservative
# — it understates the realized exit P/L rather than flattering it.
DEFAULT_FEE_BPS = 200.0


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (exit/settlement timestamp)."""
    return datetime.now(timezone.utc).isoformat()


def evaluate_exit(
    entry_price: float,
    current_price: Any,
    *,
    stop_loss_pct: Optional[float],
    take_profit_pct: Optional[float],
    take_profit_price: Optional[float],
) -> Optional[str]:
    """Decide whether an open position should exit NOW, and why.

    The decision is on the return-on-cost ``r = current_price / entry_price - 1``
    — the percentage move of the outcome token's mark since entry. It is only
    computed when the position is markable: ``entry_price > 0`` and
    ``current_price`` is a number in ``[0, 1]`` (Polymarket outcome prices are
    probabilities in dollars). Anything else (unpriced entry, ``None`` mark,
    out-of-range mark) returns ``None`` (HOLD) — we never act on a guessed price.

    Stop-loss is checked FIRST so a position that is simultaneously past its
    stop AND (somehow) past a take-profit threshold is treated as a loss to cut,
    never as a win to lock — the conservative, capital-preserving choice.

    Args:
        entry_price:        Dollars per outcome share paid at entry, in ``(0, 1]``.
        current_price:      The current market mark of the held outcome.
        stop_loss_pct:      Cut the loss when ``r <= -abs(stop_loss_pct)`` (e.g.
                            ``0.40`` exits a position down >= 40%). ``None``
                            disables the stop.
        take_profit_pct:    Lock the win when ``r >= take_profit_pct`` (e.g.
                            ``0.50`` exits a position up >= 50%). ``None``
                            disables this take-profit leg.
        take_profit_price:  Lock the win when ``current_price >= take_profit_price``
                            (e.g. ``0.90`` exits a near-certain winner outright).
                            Checked BEFORE ``take_profit_pct``. ``None`` disables.

    Returns:
        ``'stop_loss'`` | ``'take_profit'`` | ``None`` (hold).
    """
    if entry_price is None or entry_price <= 0:
        return None
    if not isinstance(current_price, (int, float)) or isinstance(current_price, bool):
        return None
    if not (0.0 <= float(current_price) <= 1.0):
        return None

    r = float(current_price) / float(entry_price) - 1.0

    # Stop-loss FIRST: cap the downside before considering any upside lock.
    if stop_loss_pct is not None and r <= -abs(float(stop_loss_pct)):
        return "stop_loss"

    # Take-profit: an absolute price target (near-certain win) takes precedence
    # over the relative return target.
    if take_profit_price is not None and float(current_price) >= float(take_profit_price):
        return "take_profit"
    if take_profit_pct is not None and r >= float(take_profit_pct):
        return "take_profit"

    return None


def compute_exit_pnl(
    entry_price: float,
    size_usd: float,
    exit_price: float,
    *,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> float:
    """Realized P/L (USD) of selling a shadow outcome-share position at a mark.

    A shadow position buys ``size_usd`` of notional at ``entry_price`` dollars
    per outcome share, so it holds ``units = size_usd / entry_price`` shares.
    Exiting early SELLS those shares at the current ``exit_price`` mark, net of a
    ``fee_bps`` basis-point haircut (default 200 bps = ~2%) on the gross
    proceeds — a conservative fee/slippage allowance::

        units    = size_usd / entry_price
        proceeds = units * exit_price * (1 - fee_bps / 10_000)
        realized = proceeds - size_usd

    Because ``exit_price`` is the CURRENT mark (not $0), a losing exit recovers
    most of the stake instead of forfeiting all of it — that is the whole point
    of the stop-loss lever. Returns ``0.0`` when the position cannot be sized
    (``entry_price <= 0`` or ``size_usd <= 0``). Rounded to 4 decimal places.

    Args:
        entry_price: Dollars per outcome share paid at entry, in ``(0, 1]``.
        size_usd: USD notional staked on the position.
        exit_price: Current market mark the shares are sold at, in ``[0, 1]``.
        fee_bps: Haircut on the gross proceeds, in basis points.

    Returns:
        Realized P/L in USD (negative on a losing exit but > -size_usd, positive
        on a winning exit), or ``0.0`` when unsizable.
    """
    if entry_price <= 0 or size_usd <= 0:
        return 0.0
    units = float(size_usd) / float(entry_price)
    proceeds = units * float(exit_price) * (1.0 - float(fee_bps) / 10_000.0)
    return round(proceeds - float(size_usd), 4)


def apply_exit_rules(
    ledger: Any,
    price_fn: Callable[[Any], Optional[float]],
    *,
    stop_loss_pct: Optional[float],
    take_profit_pct: Optional[float],
    take_profit_price: Optional[float],
    fill_fn: Optional[Callable[[Any], Optional[float]]] = None,
    fee_bps: float = DEFAULT_FEE_BPS,
    now_iso: Optional[str] = None,
) -> Dict[str, Any]:
    """Sweep open positions and settle the ones that hit a stop / take-profit.

    For each ``ledger.open_positions()`` record:
      * re-price it to the CURRENT mark via ``price_fn(record)`` (decision-time,
        NOT a future price);
      * decide the exit via :func:`evaluate_exit` on the record's
        ``entry_price`` and that mark;
      * if an exit fires, compute the realized P/L via :func:`compute_exit_pnl`
        at the FILL price and append a ``settle`` event with
        ``market_outcome = "exit:<reason>"``;
      * otherwise leave it open.

    Mark vs fill (W1): the exit DECISION uses the mark (``price_fn`` — what the
    position is worth), but the realized P/L is booked at the price you'd
    actually GET. When ``fill_fn`` is supplied and returns a price, the P/L and
    recorded ``exit_price`` use that book-walked fill (well below the mark on a
    thin book — the honest number). When ``fill_fn`` is ``None`` or returns
    ``None``, the fill falls back to the mark (the prior behavior).

    A position with no current mark (``price_fn`` returns ``None``) or one that
    does not meet a threshold is LEFT OPEN — settlement only ever happens at a
    real observed mark. Each position is wrapped so a ``price_fn`` / settle error
    is logged and skipped, never aborting the sweep (one bad market must not stop
    the others). SHADOW-ONLY: appends ledger events, places NO orders.

    Args:
        ledger: A :class:`state.pnl_ledger.PnlLedger` (or compatible) exposing
            ``open_positions()`` and ``settle(...)``.
        price_fn: ``callable(record) -> current mark | None`` (decision-time).
        stop_loss_pct / take_profit_pct / take_profit_price: see
            :func:`evaluate_exit`.
        fill_fn: optional ``callable(record) -> realistic sell price | None``. The
            book-walked fill the realized P/L is booked at; falls back to the mark
            when absent/None.
        fee_bps: Proceeds haircut passed to :func:`compute_exit_pnl`.
        now_iso: Exit timestamp to record (ISO-8601 UTC). Defaults to now. Must
            be at/after each record's entry time or the ledger guard rejects it.

    Returns:
        A counts dict::

            {"exited", "stop_loss", "take_profit", "still_open",
             "realized_pnl_usd"}
    """
    exit_ts = now_iso or _utc_now_iso()
    counts = {
        "exited": 0,
        "stop_loss": 0,
        "take_profit": 0,
        "still_open": 0,
        "realized_pnl_usd": 0.0,
    }

    try:
        open_positions = ledger.open_positions()
    except Exception as exc:  # pragma: no cover - never let a read crash the sweep.
        logging.warning("exit rules: open_positions() failed (%s)", exc)
        return counts

    # One early-exit per (market, outcome): if two writers raced and wrote
    # duplicate opens, exiting both would book the realized P/L twice. Settle the
    # earliest and leave any duplicate untouched (hidden from the dashboard and
    # retired by the runner's self-heal). Mirrors shadow_settlement.
    open_positions = dedupe_open_positions(open_positions)

    for record in open_positions:
        try:
            current_price = price_fn(record)
            reason = evaluate_exit(
                record.entry_price,
                current_price,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                take_profit_price=take_profit_price,
            )
            if reason is None:
                counts["still_open"] += 1
                continue

            # Decide on the mark; FILL at the price we'd actually realize. The
            # book-walked fill (when available) is the honest exit number — on a
            # thin book it's well below the mark, so a "take-profit" can book a
            # smaller gain or even a loss. Fall back to the mark if no fill price.
            fill_price = float(current_price)
            if fill_fn is not None:
                fp = fill_fn(record)
                if fp is not None:
                    fill_price = float(fp)
            pnl = compute_exit_pnl(
                float(record.entry_price),
                float(record.size),
                fill_price,
                fee_bps=fee_bps,
            )
            ledger.settle(
                record.trade_id,
                exit_price=fill_price,
                exit_ts_utc=exit_ts,
                market_outcome=f"exit:{reason}",
                realized_pnl_usd=pnl,
            )
        except Exception as exc:  # noqa: BLE001 - one bad position must not abort the sweep.
            logging.warning(
                "exit rules: position %s failed (%s); leaving open.",
                getattr(record, "trade_id", "?"),
                exc,
            )
            counts["still_open"] += 1
            continue

        counts["exited"] += 1
        counts[reason] += 1
        counts["realized_pnl_usd"] += pnl

    counts["realized_pnl_usd"] = round(counts["realized_pnl_usd"], 4)
    return counts
