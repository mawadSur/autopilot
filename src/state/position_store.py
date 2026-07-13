"""Redis-backed position store — source of truth for the live trader.

This store survives crashes: every mutation is persisted to Redis inside an
atomic ``MULTI/EXEC`` pipeline, and the live trader rehydrates open positions
from here on startup. The JSON ``trade_execution_*.json`` files written by
the orchestrator remain the long-term audit trail; this store is the
short-horizon, low-latency view the supervisor consults on every tick.

Redis key layout
----------------
All keys are namespaced (default ``autopilot``) so multiple bots can share
one Redis instance without colliding:

* ``{ns}:positions:{position_id}`` — JSON blob of the :class:`Position` model
  (Redis string).
* ``{ns}:open_set`` — Redis SET of position_ids currently in ``open`` or
  ``pending`` status. Membership is what the supervisor uses to decide
  whether it has any live exposure.
* ``{ns}:closed:{YYYY-MM-DD}`` — Redis SET of position_ids closed during the
  given UTC date. :meth:`PositionStore.daily_realized_pnl_usd` reads this set,
  fetches each position blob, and sums ``realized_pnl_usd``.

Atomicity
---------
Every multi-key mutation goes through a Redis pipeline executed as one
``MULTI/EXEC`` block. That keeps the open-set / closed-set / position blob
consistent across crashes — readers never see a position that's in the
open-set but missing its blob, or a closed position still listed as open.

PnL formula (matches semantics of ``src/mark_trade_settled.py``):

* long  → ``realized_pnl_usd = base_size * (exit_price - entry_price) - fees_usd``
* short → ``realized_pnl_usd = base_size * (entry_price - exit_price) - fees_usd``

``fees_usd`` on a :class:`Position` is the running total across entry + exit;
:meth:`record_close` adds the close-side ``fees_usd`` to whatever was already
on the position (typically the entry-side fees from :meth:`mark_filled`).

Reconcile
---------
:meth:`reconcile` is the defensive hatch run on supervisor startup (and
periodically thereafter). For every locally ``pending`` position older than
one hour, if the exchange has no record of the corresponding
``entry_order_id``, the position is dropped from the open-set and marked
``closed`` with ``notes="reconciled-orphan"``. This catches the failure mode
where we placed an order, lost connection before getting the order id back,
and never want to count it as exposure.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)

# Lane E E2: postmortem trigger threshold (D5).
# A losing trade triggers the swarm when ``|loss| >= 0.5% * bankroll``
# OR the trade was force-flatted by a circuit breaker / kill-switch.
POSTMORTEM_LOSS_PCT_THRESHOLD = 0.005

# Substrings on Position.notes that indicate a forced-flat close. Match the
# breaker reasons live_supervisor + circuit_breakers emit today; new reasons
# can be added here without touching the trigger logic.
_FORCED_FLAT_NOTE_SIGNALS: tuple[str, ...] = (
    "force_flat",
    "forced_flat",
    "force-flat",
    "force_flatted",
    "kill_switch",
    "daily_loss_limit",
    "halt_new_entries",
    "reconciled-orphan",
)

REDIS_URL_ENV_VAR = "REDIS_URL"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_NAMESPACE = "autopilot"

# Pending positions older than this with no exchange-side record are dropped
# by :meth:`PositionStore.reconcile`.
PENDING_ORPHAN_AGE = timedelta(hours=1)

# Sprint 2.5 — write-side seam for the Lane E swarm's trade-context snapshots.
# The reader (``state.trade_context_store.TradeContextStore``) keys snapshots
# at ``{ns}:trade_ctx:{trade_id}:{phase}`` with a 30-day TTL. ``PositionStore``
# offers the three ``record_*_snapshot`` writers below so callers that already
# hold a ``PositionStore`` handle (e.g. the live supervisor) can persist
# snapshots without taking a second store dependency. Keys + TTL are kept in
# sync with TradeContextStore so either side can read what either side wrote.
_TRADE_CTX_KEY_PREFIX = "trade_ctx"
_TRADE_CTX_TTL_SECONDS = 30 * 24 * 3600  # 30 days, matches TradeContextStore
_VALID_SNAPSHOT_PHASES: tuple[str, ...] = ("signal", "fill", "breaker", "close")


PositionStatus = Literal["pending", "open", "closing", "closed"]


def _utc_now() -> datetime:
    """Return tz-aware UTC ``datetime`` (microseconds zeroed for stable round-trips)."""

    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso_utc(value: str) -> datetime:
    """Parse an ISO-8601 string back to a tz-aware UTC datetime."""

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class Position(BaseModel):
    """A single live-trader position.

    A position transitions ``pending`` → ``open`` → ``closing`` → ``closed``.
    ``pending`` means the entry order is placed but unfilled; ``open`` means
    we have non-zero exposure; ``closing`` is reserved for partial-exit flows;
    ``closed`` is terminal.
    """

    model_config = ConfigDict(extra="forbid")

    position_id: str
    exchange: str
    symbol: str
    side: Literal["long", "short"]
    status: PositionStatus
    entry_price: float
    entry_quote_usd: float
    base_size: float
    exit_price: Optional[float] = None
    exit_quote_usd: Optional[float] = None
    realized_pnl_usd: Optional[float] = None
    fees_usd: float = 0.0
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    opened_at_utc: str
    closed_at_utc: Optional[str] = None
    model_meta: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    # Phase-16 structured fill metadata for A2 ExecutionForensics. Today A2
    # regex-scrapes ``notes`` for "partial" / "rejected" / "stop_price=...".
    # These typed fields are the canonical seam — A2 prefers them and falls
    # back to notes when they're None (legacy positions in Redis).
    # ``partial_fills`` is a list of {"size": ..., "price": ..., "filled_at_utc": ...}
    # dicts so we don't pull a second pydantic class into the schema.
    partial_fills: Optional[List[Dict[str, Any]]] = None
    rejection_reason: Optional[str] = None
    stop_trigger_price: Optional[float] = None
    # Sprint 1 Wave 2 (exit policy) — per-tick runtime state. ``bars_held``
    # is incremented by the supervisor each tick the position survives;
    # ``high_water_mark`` is the running price extreme (max for longs, min
    # for shorts) used by the trailing-stop branch in ``ExitPolicy``. Both
    # default sensibly so legacy positions in Redis that predate this
    # schema (no Wave 2 fields) round-trip cleanly: pydantic v2 fills the
    # defaults on load, and ``ExitPolicy.update_high_water_mark`` seeds
    # ``high_water_mark`` from ``entry_price`` on the first tick when None.
    bars_held: int = 0
    high_water_mark: Optional[float] = None
    # Sprint 2.5 — entry attribution. ``entry_confidence`` is the predictor's
    # confidence on the signal that produced this entry; ``resolved_kelly_pct``
    # is the Kelly fraction the predictor surfaced when Kelly sizing was
    # active (or None when the flat sizing path took it). Both default to
    # None so legacy positions written before this commit round-trip cleanly
    # via the pydantic v2 default. Calibration drift (``diagnose_calibration
    # _drift.py``) reads ``entry_confidence`` first, so populating it here
    # short-circuits the snapshot fallback and works even when the
    # ``TradeContextStore`` is offline.
    entry_confidence: Optional[float] = None
    resolved_kelly_pct: Optional[float] = None
    # Sprint 2.6 — regime label the predictor's regime_lookup resolved at
    # entry time, when ``regime_lookup`` was wired AND the lookup landed at
    # confidence >= 0.5 (otherwise the predictor leaves the cache None and
    # this stays None too). Mirrors the entry_confidence / resolved_kelly_pct
    # pattern above: pydantic v2 default fills in on load so legacy Redis
    # blobs that pre-date this field round-trip cleanly. Lit by the
    # ``OutcomeAdjuster`` resolver (priority 1 reads ``model_meta`` but the
    # signal-snapshot seam reads this field's writer at
    # ``signal_snapshot["regime_label"]`` and
    # ``signal_snapshot["risk_metrics_input"]["regime_label"]``).
    regime_label: Optional[str] = None


class PositionStore:
    """Crash-recoverable Redis-backed store for live-trader positions.

    Construct with either a ``redis_url`` (production) or a ``redis_client``
    (tests — pass a ``fakeredis.FakeRedis()`` to avoid needing a live Redis).
    All keys are prefixed with ``namespace`` so multiple stacks can share one
    Redis instance.
    """

    def __init__(
        self,
        *,
        redis_url: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        redis_client: Any | None = None,
        postmortem_queue: Optional[Any] = None,
        bankroll_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        # Lane E E2 wiring (D5 trigger gate):
        # ``postmortem_queue`` is any object with an ``enqueue(trade_id)``
        # method (typically ``RedisPostmortemQueue`` from
        # ``state.trade_context_store``). When None, the trigger evaluates
        # but only logs — never raises — so existing callers keep working
        # without code changes.
        # ``bankroll_provider`` is a callable returning the current bankroll
        # in USD. Chosen seam: callable, not a value, so a long-running
        # supervisor can grow its bankroll over time and the trigger uses
        # the current value at close time. ``record_close`` also accepts a
        # ``bankroll_usd=`` keyword for explicit per-call override.
        self.namespace = namespace
        self._lock = threading.Lock()
        self._postmortem_queue = postmortem_queue
        self._bankroll_provider = bankroll_provider

        if redis_client is not None:
            self._redis = redis_client
            self._redis_url: Optional[str] = None
        else:
            # Lazy import keeps redis a soft dependency at module import time.
            import redis  # type: ignore[import-not-found]

            self._redis_url = (
                redis_url
                or os.environ.get(REDIS_URL_ENV_VAR)
                or DEFAULT_REDIS_URL
            )
            self._redis = redis.Redis.from_url(
                self._redis_url, decode_responses=True
            )

    # ------------------------------------------------------------------
    # key helpers
    # ------------------------------------------------------------------
    def _position_key(self, position_id: str) -> str:
        return f"{self.namespace}:positions:{position_id}"

    @property
    def _open_set_key(self) -> str:
        return f"{self.namespace}:open_set"

    def _closed_set_key(self, when: datetime) -> str:
        date_part = when.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return f"{self.namespace}:closed:{date_part}"

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _dump(position: Position) -> str:
        return position.model_dump_json()

    @staticmethod
    def _load(blob: Optional[str]) -> Optional[Position]:
        if blob is None:
            return None
        return Position.model_validate_json(blob)

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def record_open(self, position: Position) -> Position:
        """Persist a freshly-opened (filled) position. Adds to ``open_set``."""

        position = position.model_copy(update={"status": "open"})
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(self._position_key(position.position_id), self._dump(position))
            pipe.sadd(self._open_set_key, position.position_id)
            pipe.execute()
        return position

    def record_pending(self, position: Position) -> Position:
        """Persist a pending (placed but unfilled) position. Adds to ``open_set``.

        Pending positions show up in :meth:`list_open` because they represent
        live risk exposure even before the fill confirms.
        """

        position = position.model_copy(update={"status": "pending"})
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(self._position_key(position.position_id), self._dump(position))
            pipe.sadd(self._open_set_key, position.position_id)
            pipe.execute()
        return position

    def mark_filled(
        self,
        position_id: str,
        *,
        fill_price: float,
        fill_size: float,
        fees_usd: float = 0.0,
    ) -> Position:
        """Promote a ``pending`` position to ``open`` once the fill confirms.

        Updates ``entry_price``/``base_size`` to the actual fill (which may
        differ from the requested values due to slippage) and accumulates
        ``fees_usd`` onto the running total.
        """

        existing = self.get(position_id)
        if existing is None:
            raise KeyError(f"unknown position_id: {position_id!r}")

        updated = existing.model_copy(
            update={
                "status": "open",
                "entry_price": float(fill_price),
                "base_size": float(fill_size),
                "entry_quote_usd": float(fill_price) * float(fill_size),
                "fees_usd": float(existing.fees_usd) + float(fees_usd),
            }
        )

        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(self._position_key(position_id), self._dump(updated))
            # Defensive: pending positions are already in open_set, but a
            # crash mid-record_pending could have left this stale. Re-add.
            pipe.sadd(self._open_set_key, position_id)
            pipe.execute()
        return updated

    def update_runtime_fields(
        self,
        position_id: str,
        *,
        bars_held: Optional[int] = None,
        high_water_mark: Optional[float] = None,
    ) -> Optional[Position]:
        """Patch per-tick runtime fields on an OPEN position.

        Sprint 1 Wave 2: ``ExitPolicy`` needs ``bars_held`` and
        ``high_water_mark`` to round-trip across ticks (the supervisor may
        be a short-lived process under multiprocessing, so in-memory state
        isn't enough). This helper rewrites only those two fields atomically;
        the position must already exist and be in the open set, or we return
        ``None`` (caller can decide whether to log — this is a per-tick
        write, not a control-flow event).
        """

        existing = self.get(position_id)
        if existing is None:
            return None
        updates: Dict[str, Any] = {}
        if bars_held is not None:
            updates["bars_held"] = int(bars_held)
        if high_water_mark is not None:
            updates["high_water_mark"] = float(high_water_mark)
        if not updates:
            return existing
        updated = existing.model_copy(update=updates)
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(self._position_key(position_id), self._dump(updated))
            pipe.execute()
        return updated

    def record_entry_attribution(
        self,
        position_id: str,
        *,
        entry_confidence: Optional[float] = None,
        resolved_kelly_pct: Optional[float] = None,
        regime_label: Optional[str] = None,
    ) -> Optional[Position]:
        """Atomically patch the two entry-attribution fields onto a position.

        Sprint 2.5: ``entry_confidence`` is what
        :func:`scripts.diagnose_calibration_drift.resolve_confidence` reads
        first; persisting it here means the calibration check works even
        when the ``TradeContextStore`` snapshot path is offline. The supervisor
        calls this immediately after ``record_open`` / paper drain so the
        position blob in Redis has the confidence the model fired at.

        Returns the updated :class:`Position` on success, ``None`` when the
        position doesn't exist (caller may log — this isn't a control-flow
        event). Best-effort: a Redis hiccup during this write must not
        wreck the trade path, so the supervisor wraps the call in a
        ``redis.exceptions.RedisError`` try/except.
        """
        existing = self.get(position_id)
        if existing is None:
            return None
        updates: Dict[str, Any] = {}
        if entry_confidence is not None:
            try:
                updates["entry_confidence"] = float(entry_confidence)
            except (TypeError, ValueError):
                pass
        if resolved_kelly_pct is not None:
            try:
                updates["resolved_kelly_pct"] = float(resolved_kelly_pct)
            except (TypeError, ValueError):
                pass
        if regime_label is not None:
            # Strings only — coerce non-empty repr defensively. ``None`` keeps
            # the existing field (no overwrite).
            try:
                label_str = str(regime_label).strip()
            except (TypeError, ValueError):
                label_str = ""
            if label_str:
                updates["regime_label"] = label_str
        if not updates:
            return existing
        updated = existing.model_copy(update=updates)
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(self._position_key(position_id), self._dump(updated))
            pipe.execute()
        return updated

    # ------------------------------------------------------------------
    # Sprint 2.5 — trade-context snapshot writers
    # ------------------------------------------------------------------
    def _trade_ctx_key(self, trade_id: str, phase: str) -> str:
        return f"{self.namespace}:{_TRADE_CTX_KEY_PREFIX}:{trade_id}:{phase}"

    def _record_trade_ctx_snapshot(
        self,
        *,
        trade_id: str,
        phase: str,
        snapshot: Mapping[str, Any],
    ) -> None:
        """Shared writer for the three ``record_*_snapshot`` methods.

        Keys + TTL match :class:`state.trade_context_store.TradeContextStore`
        so the Lane E specialists' existing ``get_signal_snapshot`` /
        ``get_fill_snapshot`` / ``get_snapshots`` calls find what's written
        here without any reader-side change. ``snapshot`` is a dict shaped
        like :class:`TradeContextSnapshot.to_dict()` (the canonical shape).
        Writers MUST pre-include ``trade_id``, ``symbol``, ``captured_at_utc``,
        and ``phase`` so the reader can rehydrate via ``from_json``.
        """
        if not trade_id:
            raise ValueError("trade_id must be a non-empty string")
        if phase not in _VALID_SNAPSHOT_PHASES:
            raise ValueError(
                f"phase must be one of {_VALID_SNAPSHOT_PHASES!r}, got {phase!r}"
            )
        blob = json.dumps(dict(snapshot), allow_nan=False, sort_keys=True)
        key = self._trade_ctx_key(trade_id, phase)
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(key, blob)
            pipe.expire(key, _TRADE_CTX_TTL_SECONDS)
            pipe.execute()

    def record_signal_snapshot(
        self, trade_id: str, snapshot: Mapping[str, Any]
    ) -> None:
        """Persist the signal-time snapshot for ``trade_id``.

        ``snapshot`` is a dict matching :class:`TradeContextSnapshot.to_dict`
        with ``phase="signal"``. Caller is responsible for sanitising NaN/Inf
        before passing (the dict is JSON-dumped with ``allow_nan=False``).
        """
        self._record_trade_ctx_snapshot(
            trade_id=trade_id, phase="signal", snapshot=snapshot
        )

    def record_fill_snapshot(
        self, trade_id: str, snapshot: Mapping[str, Any]
    ) -> None:
        """Persist the fill-time snapshot for ``trade_id``."""
        self._record_trade_ctx_snapshot(
            trade_id=trade_id, phase="fill", snapshot=snapshot
        )

    def record_breaker_snapshot(
        self, trade_id: str, snapshot: Mapping[str, Any]
    ) -> None:
        """Persist the breaker-time snapshot for ``trade_id`` (forced-flat path)."""
        self._record_trade_ctx_snapshot(
            trade_id=trade_id, phase="breaker", snapshot=snapshot
        )

    def get_trade_ctx_snapshot(
        self, trade_id: str, phase: str
    ) -> Optional[Dict[str, Any]]:
        """Read back a snapshot blob as a dict; ``None`` when missing.

        Symmetric with the three ``record_*_snapshot`` writers so tests can
        round-trip without taking a second store dependency. Production
        readers continue to use ``TradeContextStore.get_snapshot(...)``.
        """
        if phase not in _VALID_SNAPSHOT_PHASES:
            raise ValueError(
                f"phase must be one of {_VALID_SNAPSHOT_PHASES!r}, got {phase!r}"
            )
        blob = self._redis.get(self._trade_ctx_key(trade_id, phase))
        if blob is None:
            return None
        try:
            return json.loads(blob)
        except (ValueError, TypeError, json.JSONDecodeError):
            return None

    def record_close(
        self,
        position_id: str,
        *,
        exit_price: float,
        exit_quote_usd: float,
        fees_usd: float = 0.0,
        bankroll_usd: Optional[float] = None,
    ) -> Position:
        """Mark a position closed, compute realized PnL, move sets atomically.

        Adds ``fees_usd`` (the close-side fees) to the position's running total
        before computing PnL. Removes from ``open_set`` and adds to the
        UTC-dated ``closed:{date}`` set in the same ``MULTI/EXEC`` block.

        Lane E E2 trigger gate (D5)
        ----------------------------
        After the close lands, evaluates whether the trade should be enqueued
        for postmortem investigation. The condition is::

            realized_pnl_usd < 0
              AND (
                |realized_pnl_usd| >= 0.005 * bankroll_usd
                OR position.notes signals a forced flat
              )

        Bankroll is taken from (in priority order): the explicit
        ``bankroll_usd=`` kwarg, then the constructor ``bankroll_provider``
        callable. If neither is available we still trigger on the
        forced-flat path (size-relative threshold can't be evaluated, but
        the breaker-driven path is unambiguous). The enqueue itself is a
        best-effort call — failures are logged, never raised, so a
        misbehaving queue cannot prevent a close from completing.
        """

        existing = self.get(position_id)
        if existing is None:
            raise KeyError(f"unknown position_id: {position_id!r}")

        total_fees = float(existing.fees_usd) + float(fees_usd)
        if existing.side == "long":
            realized = (
                float(existing.base_size) * (float(exit_price) - float(existing.entry_price))
                - total_fees
            )
        else:  # short
            realized = (
                float(existing.base_size) * (float(existing.entry_price) - float(exit_price))
                - total_fees
            )

        now = _utc_now()
        updated = existing.model_copy(
            update={
                "status": "closed",
                "exit_price": float(exit_price),
                "exit_quote_usd": float(exit_quote_usd),
                "fees_usd": total_fees,
                "realized_pnl_usd": realized,
                "closed_at_utc": now.isoformat(),
            }
        )

        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(self._position_key(position_id), self._dump(updated))
            pipe.srem(self._open_set_key, position_id)
            pipe.sadd(self._closed_set_key(now), position_id)
            pipe.execute()

        # Lane E E2 trigger gate (D5) — evaluate after the close has landed
        # so a queue failure cannot strand the close write half-done.
        self._maybe_enqueue_postmortem(updated, bankroll_usd=bankroll_usd)
        return updated

    # ------------------------------------------------------------------
    # Lane E E2: postmortem trigger gate
    # ------------------------------------------------------------------
    def _resolve_bankroll(
        self, *, bankroll_usd: Optional[float]
    ) -> Optional[float]:
        """Pick the best available bankroll value for trigger evaluation."""
        if bankroll_usd is not None:
            try:
                return float(bankroll_usd)
            except (TypeError, ValueError):
                return None
        if self._bankroll_provider is not None:
            try:
                return float(self._bankroll_provider())
            except Exception as exc:  # noqa: BLE001 - tolerate flaky providers
                LOGGER.warning(
                    "bankroll_provider raised during postmortem trigger eval: %r",
                    exc,
                )
                return None
        return None

    @staticmethod
    def _is_forced_flat(position: Position) -> bool:
        notes = (position.notes or "").lower()
        if not notes:
            return False
        return any(sig.lower() in notes for sig in _FORCED_FLAT_NOTE_SIGNALS)

    def _should_enqueue_postmortem(
        self, position: Position, *, bankroll_usd: Optional[float]
    ) -> bool:
        """Evaluate the D5 threshold-gated trigger. Pure function — no I/O."""
        realized = position.realized_pnl_usd
        if realized is None or realized >= 0:
            return False  # winners and even-money closes never trigger
        forced = self._is_forced_flat(position)
        if forced:
            return True
        # Size-relative threshold path — needs a bankroll number to evaluate.
        if bankroll_usd is None or bankroll_usd <= 0:
            return False
        return abs(float(realized)) >= POSTMORTEM_LOSS_PCT_THRESHOLD * float(
            bankroll_usd
        )

    def _maybe_enqueue_postmortem(
        self, position: Position, *, bankroll_usd: Optional[float] = None
    ) -> bool:
        """Evaluate the trigger and enqueue if it fires. Returns whether enqueued."""
        try:
            br = self._resolve_bankroll(bankroll_usd=bankroll_usd)
            should_fire = self._should_enqueue_postmortem(
                position, bankroll_usd=br
            )
        except Exception as exc:  # noqa: BLE001 - never let evaluation crash a close
            LOGGER.warning(
                "postmortem trigger evaluation crashed for %s: %r",
                position.position_id,
                exc,
            )
            return False

        if not should_fire:
            return False

        if self._postmortem_queue is None:
            LOGGER.debug(
                "postmortem trigger fired for %s (pnl=%.4f forced=%s) "
                "but no queue wired; skipping enqueue",
                position.position_id,
                float(position.realized_pnl_usd or 0.0),
                self._is_forced_flat(position),
            )
            return False

        try:
            self._postmortem_queue.enqueue(position.position_id)
        except Exception as exc:  # noqa: BLE001 - queue is best-effort
            LOGGER.warning(
                "postmortem queue enqueue failed for %s: %r",
                position.position_id,
                exc,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def get(self, position_id: str) -> Optional[Position]:
        blob = self._redis.get(self._position_key(position_id))
        return self._load(blob)

    def list_open(self) -> list[Position]:
        ids = self._redis.smembers(self._open_set_key) or set()
        results: List[Position] = []
        for pid in ids:
            position = self.get(pid)
            if position is None:
                # The blob was lost (manual surgery, eviction); skip silently.
                continue
            if position.status in ("open", "pending", "closing"):
                results.append(position)
        # Stable order helps tests + dashboards.
        results.sort(key=lambda p: p.position_id)
        return results

    def list_closed_today(self, *, now_utc: Optional[datetime] = None) -> list[Position]:
        when = now_utc or _utc_now()
        ids = self._redis.smembers(self._closed_set_key(when)) or set()
        results: List[Position] = []
        for pid in ids:
            position = self.get(pid)
            if position is None:
                continue
            results.append(position)
        results.sort(key=lambda p: p.closed_at_utc or "")
        return results

    def open_notional_usd(self) -> float:
        return float(sum(p.entry_quote_usd for p in self.list_open()))

    def open_notional_for_symbol(self, symbol: str) -> float:
        return float(
            sum(p.entry_quote_usd for p in self.list_open() if p.symbol == symbol)
        )

    def daily_realized_pnl_usd(self, *, now_utc: Optional[datetime] = None) -> float:
        return float(
            sum(
                (p.realized_pnl_usd or 0.0)
                for p in self.list_closed_today(now_utc=now_utc)
            )
        )

    def daily_realized_pnl_usd_for_symbol(
        self, symbol: str, *, now_utc: Optional[datetime] = None
    ) -> float:
        """Per-symbol realised PnL for today (UTC). Used by per-symbol shakedown."""
        return float(
            sum(
                (p.realized_pnl_usd or 0.0)
                for p in self.list_closed_today(now_utc=now_utc)
                if p.symbol == symbol
            )
        )

    # ------------------------------------------------------------------
    # Task 4: orphan-position telemetry
    # ------------------------------------------------------------------
    def orphan_count(self) -> int:
        """Count currently-open positions that look like reconciled orphans.

        ``PositionStore.reconcile`` rewrites stale ``pending`` positions
        with ``status="closed"`` + ``notes="reconciled-orphan"`` — those
        are NOT counted here (they're already closed). This helper counts
        OPEN positions whose ``notes`` reference an orphan-tagged
        condition, plus pending positions older than the orphan age, so
        the operator dashboard can see drift surfaces in real time.

        The reconciliation report (``ops.reconciliation``) is the
        authoritative orphan view (it cross-checks the exchange); this
        helper is a cheap, no-network gauge for Prometheus.
        """
        try:
            opens = self.list_open()
        except Exception:  # noqa: BLE001 - telemetry must never raise
            return 0
        now = _utc_now()
        count = 0
        for position in opens:
            notes = (position.notes or "").lower()
            if "orphan" in notes:
                count += 1
                continue
            if position.status == "pending":
                try:
                    opened_at = _parse_iso_utc(position.opened_at_utc)
                except (TypeError, ValueError):
                    continue
                if now - opened_at > PENDING_ORPHAN_AGE:
                    count += 1
        return count

    # ------------------------------------------------------------------
    # reconcile
    # ------------------------------------------------------------------
    def reconcile(self, exchange: Any, *, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """Drop stale ``pending`` positions the exchange no longer knows about.

        ``exchange`` must expose ``get_open_orders(symbol=None) -> Iterable``.
        Each item should either expose an ``order_id`` attribute or be a dict
        with an ``"order_id"`` key (or ``"id"``) — we match against the
        ``entry_order_id`` stored on each pending position.

        A pending position is dropped when ALL hold:
            1. status is ``pending``,
            2. its ``opened_at_utc`` is older than :data:`PENDING_ORPHAN_AGE`,
            3. its ``entry_order_id`` is not in the exchange's open-orders set.

        Dropped positions are removed from ``open_set`` and rewritten with
        ``status="closed"``, ``realized_pnl_usd=0.0``, and
        ``notes="reconciled-orphan"`` for audit-trail purposes.
        """

        now = now_utc or _utc_now()
        warnings: List[str] = []

        try:
            raw_orders = exchange.get_open_orders(symbol=None)
        except Exception as exc:  # noqa: BLE001 - defensive, log and bail
            warnings.append(f"exchange.get_open_orders failed: {exc!r}")
            return {"reconciled": 0, "dropped": 0, "warnings": warnings}

        known_order_ids = set()
        for order in raw_orders or ():
            order_id: Optional[str] = None
            if hasattr(order, "order_id"):
                order_id = getattr(order, "order_id")
            elif isinstance(order, dict):
                order_id = order.get("order_id") or order.get("id")
            if order_id:
                known_order_ids.add(str(order_id))

        reconciled = 0
        dropped = 0
        for position in self.list_open():
            if position.status != "pending":
                reconciled += 1
                continue

            try:
                opened_at = _parse_iso_utc(position.opened_at_utc)
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"{position.position_id}: bad opened_at_utc {position.opened_at_utc!r} ({exc})"
                )
                reconciled += 1
                continue

            age = now - opened_at
            order_known = (
                position.entry_order_id is not None
                and str(position.entry_order_id) in known_order_ids
            )

            if order_known:
                reconciled += 1
                continue
            if age <= PENDING_ORPHAN_AGE:
                # Too young to declare orphaned — fill confirmation may still arrive.
                reconciled += 1
                continue

            # Orphan: drop from open_set, mark closed for audit.
            orphaned = position.model_copy(
                update={
                    "status": "closed",
                    "realized_pnl_usd": 0.0,
                    "closed_at_utc": now.isoformat(),
                    "notes": "reconciled-orphan",
                }
            )
            with self._lock:
                pipe = self._redis.pipeline()
                pipe.multi()
                pipe.set(self._position_key(orphaned.position_id), self._dump(orphaned))
                pipe.srem(self._open_set_key, orphaned.position_id)
                pipe.sadd(self._closed_set_key(now), orphaned.position_id)
                pipe.execute()
            dropped += 1
            LOGGER.warning(
                "Reconciled orphan pending position %s (order %s unknown to exchange)",
                orphaned.position_id,
                orphaned.entry_order_id,
            )

        return {"reconciled": reconciled, "dropped": dropped, "warnings": warnings}

    # ------------------------------------------------------------------
    # per-symbol error counter (Lane A P0 #3)
    # ------------------------------------------------------------------
    #
    # Multiple symbol-supervisor processes (under the multiprocessing model
    # in D1) need to increment a shared error counter -- the in-memory
    # dict that lived on Supervisor was per-process and lost increments
    # across restarts. Redis HASH ``errors:by_symbol:{date}`` solves both:
    # cross-process visibility and crash-survivability. The TTL keeps the
    # set bounded automatically.
    _ERROR_COUNTER_TTL_SECONDS = 48 * 3600  # 48h, two daily-close windows.

    def _errors_key(self, when: datetime) -> str:
        date_part = when.astimezone(timezone.utc).strftime("%Y-%m-%d")
        return f"{self.namespace}:errors:by_symbol:{date_part}"

    def increment_error(
        self, symbol: str, *, now_utc: Optional[datetime] = None
    ) -> int:
        """Atomically bump today's error count for ``symbol``. Returns new count.

        Uses Redis HINCRBY so two processes racing on the same symbol
        don't lose increments. The first writer also sets a 48h expire on
        the key so stale daily counters age out automatically.
        """
        when = now_utc or _utc_now()
        key = self._errors_key(when)
        with self._lock:
            new_value = int(self._redis.hincrby(key, symbol, 1))
            # Expire only needs to be set once but EXPIRE is idempotent
            # and ~free; calling it on every increment keeps the logic
            # trivially correct across crashes.
            self._redis.expire(key, self._ERROR_COUNTER_TTL_SECONDS)
        return new_value

    def errors_today(
        self, symbol: str, *, now_utc: Optional[datetime] = None
    ) -> int:
        """Read today's error count for ``symbol``. Returns 0 if unset."""
        when = now_utc or _utc_now()
        raw = self._redis.hget(self._errors_key(when), symbol)
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def errors_today_all(
        self, *, now_utc: Optional[datetime] = None
    ) -> Dict[str, int]:
        """Snapshot today's full per-symbol error map. Empty dict if unset."""
        when = now_utc or _utc_now()
        raw = self._redis.hgetall(self._errors_key(when)) or {}
        out: Dict[str, int] = {}
        for sym, val in raw.items():
            try:
                out[str(sym)] = int(val)
            except (TypeError, ValueError):
                continue
        return out

    def reset_errors_for_day(
        self, *, now_utc: Optional[datetime] = None
    ) -> int:
        """Delete today's error counter HASH. Returns 1 if deleted, 0 otherwise.

        Called by the supervisor at daily_close after evidence has been
        rolled into the shakedown record, so the next day starts clean.
        """
        when = now_utc or _utc_now()
        key = self._errors_key(when)
        return int(self._redis.delete(key))

    # ------------------------------------------------------------------
    # admin
    # ------------------------------------------------------------------
    def clear_namespace(self) -> int:
        """Delete every key under this store's namespace. Returns count deleted.

        Intended for tests + manual ops; never call from production code paths.
        """

        pattern = f"{self.namespace}:*"
        deleted = 0
        # ``scan_iter`` is the safe walk; ``keys`` would block Redis under load.
        keys = list(self._redis.scan_iter(match=pattern))
        if keys:
            deleted = int(self._redis.delete(*keys))
        return deleted


# ----------------------------------------------------------------------
# module-level singleton (mirrors src/storage/sqlite_store.py pattern)
# ----------------------------------------------------------------------
_DEFAULT_STORE: Optional[PositionStore] = None
_DEFAULT_STORE_LOCK = threading.Lock()


def get_default_store() -> PositionStore:
    """Process-wide singleton :class:`PositionStore` keyed off ``REDIS_URL``."""

    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = PositionStore()
        return _DEFAULT_STORE


def reset_default_store() -> None:
    """Forget the cached default store. Used by tests for isolation."""

    global _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        _DEFAULT_STORE = None


__all__ = [
    "DEFAULT_NAMESPACE",
    "DEFAULT_REDIS_URL",
    "PENDING_ORPHAN_AGE",
    "POSTMORTEM_LOSS_PCT_THRESHOLD",
    "Position",
    "PositionStatus",
    "PositionStore",
    "REDIS_URL_ENV_VAR",
    "get_default_store",
    "reset_default_store",
    # Sprint 2.5 — trade-context snapshot writer constants. The reader keys
    # are owned by ``state.trade_context_store``; these are exported for
    # callers (the supervisor, tests) that want the same constants without
    # also importing TradeContextStore.
    "_TRADE_CTX_KEY_PREFIX",
    "_TRADE_CTX_TTL_SECONDS",
]
