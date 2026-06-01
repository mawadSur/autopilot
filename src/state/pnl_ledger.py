"""Shadow PnL ledger — the single per-trade truth record.

This module is the one auditable place where *every* trade decision and its
eventual settlement is recorded, regardless of which stack produced it (the
legacy crypto backtest/paper loop or the prediction-market shadow loop). Before
this existed there was no auditable paper/backtest result anywhere in the repo.

JSONL event-log model
---------------------
The ledger is an **append-only JSONL file**: one JSON object per line, never
rewritten in place. Each line is an immutable *event*. There are two event
kinds, both keyed by ``trade_id``:

1. ``"open"`` — emitted by :meth:`PnlLedger.append` when a position is entered.
   Carries the full :class:`TradeRecord` snapshot at decision/entry time.
2. ``"settle"`` — emitted by :meth:`PnlLedger.settle` when the position closes.
   Carries the exit fields (``exit_price``, ``exit_ts_utc``, ``market_outcome``,
   ``realized_pnl_usd``, ``status``) plus the ``trade_id`` they apply to.

Readers *fold* the event stream into current state: for each ``trade_id`` the
latest event wins for the fields it carries. The file is therefore an immutable
audit trail — you can replay it to reconstruct any intermediate state, and a
settlement never destroys the original entry line. This matches the repo's
honest-reporting / no-look-ahead ethos: the entry price and decision timestamp
are written once, at decision time, and can never be silently overwritten.

No look-ahead integrity
-----------------------
:meth:`PnlLedger.settle` refuses (raises :class:`ValueError`) any settlement
whose ``exit_ts_utc`` is strictly before the original entry ``ts_utc``. A trade
cannot close before it opens; allowing it would let backtests fabricate fills
from future information, which the Constitution forbids.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


DEFAULT_LEDGER_PATH = "runs/pnl_ledger.jsonl"

# Event-kind discriminators written into each JSONL line under "_event".
EVENT_OPEN = "open"
EVENT_SETTLE = "settle"
EVENT_CANCEL = "cancel"


@dataclass
class TradeRecord:
    """One trade's full lifecycle state, folded from the event log.

    Conventions
    -----------
    - ``ts_utc`` is the DECISION/entry time, ISO-8601 (e.g.
      ``"2026-05-31T14:30:00+00:00"``). It is written once and never changes.
    - ``size`` is **USD notional** for the position (not unit count). Document
      it as units only at the call site if a venue ever needs that; the summary
      math here treats it purely as a label and never derives PnL from it.
    - ``realized_pnl_usd`` is supplied by the caller at settle time (already net
      of fees/slippage in the caller's accounting); the ledger does not invent
      PnL, it only records what the strategy reports.
    """

    trade_id: str
    ts_utc: str  # ISO-8601 decision/entry time
    venue: str  # e.g. 'polymarket' | 'kalshi' | 'coinbase'
    market_id: str
    side: str  # e.g. 'YES' | 'NO' | 'long' | 'short'
    entry_price: float
    size: float  # USD notional (see class docstring)
    fees_usd: float
    slippage_bps: float
    strategy: str
    status: str = "open"  # 'open' | 'settled' | 'cancelled'
    exit_price: Optional[float] = None
    exit_ts_utc: Optional[str] = None
    market_outcome: Optional[Union[str, float]] = None
    realized_pnl_usd: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeRecord":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def _parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a tz-aware UTC ``datetime``.

    Accepts a trailing ``Z`` (treated as ``+00:00``) and naive timestamps
    (assumed UTC). Raises :class:`ValueError` on anything unparseable so the
    look-ahead guard fails loudly rather than silently passing.
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _open_position_key(record: "TradeRecord") -> tuple:
    """The identity of an open position for dedup: one per ``(market_id, side)``.

    Two open records sharing this key are the SAME position — we shadow one unit
    of a (market, outcome), never two. Such pairs only arise when two writers race
    on the same ledger (each snapshots the open set before the other appends).
    """
    return (getattr(record, "market_id", None), getattr(record, "side", None))


def dedupe_open_positions(records: List["TradeRecord"]) -> List["TradeRecord"]:
    """Collapse open ``records`` to one per ``(market_id, side)``.

    Keeps the EARLIEST-entered record for each key (by ``ts_utc``, which ISO-8601
    sorts chronologically) — the original entry, whose decision-time price is the
    honest one we'd have actually filled at. Input order is otherwise preserved
    for the kept records. Pure and side-effect free; callers decide whether to
    cancel the dropped duplicates in the ledger.
    """
    earliest: Dict[tuple, "TradeRecord"] = {}
    for record in records:
        key = _open_position_key(record)
        kept = earliest.get(key)
        if kept is None or (getattr(record, "ts_utc", "") or "") < (
            getattr(kept, "ts_utc", "") or ""
        ):
            earliest[key] = record
    # Preserve first-seen order of the kept records for stable display.
    seen: set = set()
    ordered: List["TradeRecord"] = []
    for record in records:
        key = _open_position_key(record)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(earliest[key])
    return ordered


def duplicate_open_records(records: List["TradeRecord"]) -> List["TradeRecord"]:
    """Return the DROPPED duplicate open records (everything dedup would discard).

    These are the later-entered records that share a ``(market_id, side)`` with an
    earlier one — the exact set a caller should :meth:`PnlLedger.cancel` to make
    the ledger self-consistent (one open per market/outcome).
    """
    keep = {id(r) for r in dedupe_open_positions(records)}
    return [r for r in records if id(r) not in keep]


class PnlLedger:
    """Append-only JSONL ledger of trade-open and trade-settle events.

    Parameters
    ----------
    path:
        Filesystem path to the JSONL ledger. The parent directory is created
        on construction if it does not already exist. Defaults to
        :data:`DEFAULT_LEDGER_PATH`.
    """

    def __init__(self, path: str = DEFAULT_LEDGER_PATH) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def append(self, record: TradeRecord) -> None:
        """Append an ``open`` event for ``record`` (one JSON line)."""
        payload = record.to_dict()
        payload["_event"] = EVENT_OPEN
        self._write_line(payload)

    def settle(
        self,
        trade_id: str,
        exit_price: float,
        exit_ts_utc: str,
        market_outcome: Optional[Union[str, float]] = None,
        realized_pnl_usd: Optional[float] = None,
    ) -> TradeRecord:
        """Append a ``settle`` event for ``trade_id`` and return the folded record.

        Raises
        ------
        KeyError
            If ``trade_id`` has no prior ``open`` event in the ledger.
        ValueError
            If ``exit_ts_utc`` is strictly before the record's entry ``ts_utc``
            (the no-look-ahead guard), or if either timestamp is unparseable.
        """
        state = self._fold()
        record = state.get(trade_id)
        if record is None:
            raise KeyError(f"cannot settle unknown trade_id: {trade_id!r}")

        entry_dt = _parse_iso8601(record.ts_utc)
        exit_dt = _parse_iso8601(exit_ts_utc)
        if exit_dt < entry_dt:
            raise ValueError(
                "look-ahead guard: exit_ts_utc "
                f"({exit_ts_utc}) is before entry ts_utc ({record.ts_utc}) "
                f"for trade_id {trade_id!r}"
            )

        settle_payload: Dict[str, Any] = {
            "_event": EVENT_SETTLE,
            "trade_id": trade_id,
            "status": "settled",
            "exit_price": float(exit_price),
            "exit_ts_utc": exit_ts_utc,
            "market_outcome": market_outcome,
            "realized_pnl_usd": (
                None if realized_pnl_usd is None else float(realized_pnl_usd)
            ),
        }
        self._write_line(settle_payload)

        # Return the up-to-date folded record so callers don't re-read.
        record.status = "settled"
        record.exit_price = float(exit_price)
        record.exit_ts_utc = exit_ts_utc
        record.market_outcome = market_outcome
        record.realized_pnl_usd = (
            None if realized_pnl_usd is None else float(realized_pnl_usd)
        )
        return record

    def cancel(self, trade_id: str, reason: str = "") -> "TradeRecord":
        """Append a ``cancel`` event for ``trade_id`` and return the folded record.

        A cancel marks an OPEN record as ``cancelled`` so it drops out of both
        :meth:`open_positions` and :meth:`settled` — it never contributes P/L and
        never settles. This is the append-only way to retire a position that
        should not have been opened (e.g. a duplicate written by two concurrent
        writers racing on the same ledger): the original ``open`` line is left
        intact for the audit trail, and a ``cancel`` line records the retraction
        plus its ``reason``.

        Unlike :meth:`settle` there is no exit price and no look-ahead concern (a
        cancel carries no realized P/L), so there is no timestamp guard.

        Raises
        ------
        KeyError
            If ``trade_id`` has no prior ``open`` event in the ledger.
        """
        state = self._fold()
        record = state.get(trade_id)
        if record is None:
            raise KeyError(f"cannot cancel unknown trade_id: {trade_id!r}")

        cancel_payload: Dict[str, Any] = {
            "_event": EVENT_CANCEL,
            "trade_id": trade_id,
            "status": "cancelled",
        }
        if reason:
            cancel_payload["cancel_reason"] = reason
        self._write_line(cancel_payload)

        record.status = "cancelled"
        return record

    def _write_line(self, payload: Dict[str, Any]) -> None:
        """Append a single JSON object as one line, flushing to disk.

        Uses a single ``write()`` of a newline-terminated string under append
        mode so concurrent appenders never interleave partial lines, and
        ``flush`` + ``fsync`` so a crash right after the call still leaves a
        complete, parseable ledger.
        """
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def _fold(self) -> "Dict[str, TradeRecord]":
        """Replay the event log into current per-``trade_id`` state.

        Missing or empty file yields an empty mapping. Lines that are blank or
        unparseable JSON are skipped (the ledger stays readable even if a write
        was truncated by a crash mid-line).
        """
        state: Dict[str, TradeRecord] = {}
        if not os.path.exists(self.path):
            return state
        with open(self.path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("_event")
                trade_id = event.get("trade_id")
                if not trade_id:
                    continue
                if kind == EVENT_OPEN:
                    state[trade_id] = TradeRecord.from_dict(event)
                elif kind == EVENT_CANCEL:
                    record = state.get(trade_id)
                    if record is None:
                        # Cancel with no prior open — ignore (keep the audit
                        # honest; we never fabricate an entry to cancel).
                        continue
                    record.status = event.get("status", "cancelled")
                elif kind == EVENT_SETTLE:
                    record = state.get(trade_id)
                    if record is None:
                        # Settlement with no prior open — keep the audit trail
                        # honest by ignoring it rather than fabricating an entry.
                        continue
                    if "status" in event:
                        record.status = event["status"]
                    if "exit_price" in event:
                        record.exit_price = event["exit_price"]
                    if "exit_ts_utc" in event:
                        record.exit_ts_utc = event["exit_ts_utc"]
                    if "market_outcome" in event:
                        record.market_outcome = event["market_outcome"]
                    if "realized_pnl_usd" in event:
                        record.realized_pnl_usd = event["realized_pnl_usd"]
        return state

    def all_records(self) -> List[TradeRecord]:
        """Return every folded record (open + settled + cancelled)."""
        return list(self._fold().values())

    def open_positions(self) -> List[TradeRecord]:
        """Return records still in ``open`` status."""
        return [r for r in self._fold().values() if r.status == "open"]

    def unique_open_positions(self) -> List[TradeRecord]:
        """Open positions collapsed to one per ``(market_id, side)``.

        Defense-in-depth against duplicate opens written by concurrent writers:
        readers that care about the *positions we actually hold* (the dashboard,
        the settlement sweep) use this so a stray duplicate is never shown twice
        nor settled twice. See :func:`dedupe_open_positions`.
        """
        return dedupe_open_positions(self.open_positions())

    def settled(self) -> List[TradeRecord]:
        """Return records in ``settled`` status."""
        return [r for r in self._fold().values() if r.status == "settled"]

    def by_strategy(self, name: str) -> List[TradeRecord]:
        """Return all records (any status) for a given strategy name."""
        return [r for r in self._fold().values() if r.strategy == name]

    def summary(self) -> Dict[str, Any]:
        """Aggregate the ledger into a headline dict.

        Returns keys: ``total_realized_pnl_usd``, ``n_trades``, ``n_settled``,
        ``n_open``, ``win_rate`` (settled wins / settled, 0.0 if none settled),
        ``total_fees_usd``. A missing/empty ledger returns all-zero values.
        """
        records = list(self._fold().values())
        n_trades = len(records)
        settled = [r for r in records if r.status == "settled"]
        n_settled = len(settled)
        n_open = sum(1 for r in records if r.status == "open")

        total_realized = sum(
            r.realized_pnl_usd
            for r in settled
            if r.realized_pnl_usd is not None
        )
        total_fees = sum(
            r.fees_usd for r in records if r.fees_usd is not None
        )
        wins = sum(
            1
            for r in settled
            if r.realized_pnl_usd is not None and r.realized_pnl_usd > 0
        )
        win_rate = (wins / n_settled) if n_settled else 0.0

        return {
            "total_realized_pnl_usd": float(total_realized),
            "n_trades": n_trades,
            "n_settled": n_settled,
            "n_open": n_open,
            "win_rate": win_rate,
            "total_fees_usd": float(total_fees),
        }


__all__ = [
    "DEFAULT_LEDGER_PATH",
    "EVENT_OPEN",
    "EVENT_SETTLE",
    "EVENT_CANCEL",
    "PnlLedger",
    "TradeRecord",
    "dedupe_open_positions",
    "duplicate_open_records",
]
