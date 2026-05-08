"""Live trader supervisor — Phase 5.

Long-running process that owns the per-tick trading loop and wires together:

* **Phase 1**: ``CoinbaseExchange`` (the only execution surface).
* **Phase 2**: ``CircuitBreakerSet`` (kill-switch + daily-loss + drawdown +
  notional caps).
* **Phase 3**: ``PositionStore`` (Redis-backed source of truth for positions).
* **Phase 4**: ``Notifier`` (Discord + Telegram alerts).

Operator-intent gating
----------------------
The supervisor enforces a **mandatory 14-day paper-trade shakedown** before
``mode="live"`` is honoured. The rolling shakedown evidence lives in a JSON
file and is updated on every ``daily_close``. Until ``paper_days_clean`` clears
``shakedown_min_days`` (default 14), live mode silently falls back to
paper-trade simulation and emits a one-time WARNING.

Coinbase sandbox reality
------------------------
``CoinbaseExchange.is_sandbox()`` reflects operator intent, but ccxt 4.x has no
real Coinbase sandbox. Paper mode therefore synthesises fills locally
(applying a fixed 5-bps slippage to the ticker mid) and never touches
``exchange.place_market_order`` — the supervisor must respect this.

Tick flow (per symbol)
----------------------
1. Kill switch check (force-flat all open positions if tripped).
2. Fetch ticker.
3. Build :class:`DecisionContext`, run circuit breakers.
4. Call ``model_predict_fn`` -> ``(side, confidence)``.
5. Skip if ``confidence < min_confidence_to_trade``.
6. Mode gate: if ``live`` but shakedown not unlocked, fall back to paper.
7. Place order (live exchange call OR paper-synthesised fill).
8. Catch ``ExchangeError`` and any other exception; alert + continue.

This module has no new third-party dependencies and is fully hermetic for
testing — every Phase 1-4 collaborator is injected at construction time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import sys
import time
import traceback
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

# Mirror the sys.path shim used by main.py / orchestrator.py /
# calibration_agent/build_dataset.py so this CLI runs without the caller
# setting PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import BaseModel, ConfigDict, Field

from alerts.notifier import Notifier
from exchanges.coinbase import CoinbaseExchange, ExchangeError, OrderResult, Ticker
from risk.auto_pause import AutoPauseGate
from risk.circuit_breakers import (
    CircuitBreakerSet,
    CircuitBreakerVerdict,
    DecisionContext,
)
from state.confidence_history import ConfidenceHistory
from state.position_store import Position, PositionStore
from state.trade_context_store import (
    TradeContextSnapshot,
    TradeContextStore,
    utc_now_iso,
)

LOGGER = logging.getLogger(__name__)


def _capture_exception(exc: BaseException) -> None:
    """Best-effort hand-off to Sentry, never raises.

    The Sentry SDK is an optional install. The live trader must continue
    running even if Sentry is missing or its capture path errors.
    """
    try:
        import sentry_sdk  # type: ignore[import-not-found]

        sentry_sdk.capture_exception(exc)
    except Exception:  # noqa: BLE001 - never let observability kill the trader
        pass


# Fixed paper-mode slippage applied to the ticker mid when synthesising fills.
# 5 bps = 0.05% -- conservative but representative of real Coinbase taker
# slippage on liquid majors. Buys eat ask side, sells eat bid side.
PAPER_SLIPPAGE_BPS = 5.0


# Exponential backoff (seconds) used when an exclusive flock on the
# shakedown JSON is contested. Five attempts spread over ~310ms total --
# generous enough to ride out a peer process's atomic-rename window
# without hanging the loop on a stuck holder.
_SHAKEDOWN_LOCK_BACKOFF_S: Tuple[float, ...] = (0.010, 0.020, 0.040, 0.080, 0.160)


class _NullLock:
    """Best-effort no-op file-lock context manager.

    Returned when the platform doesn't expose ``fcntl`` (Windows) or the
    lock acquisition fails after exhausting the backoff schedule. We
    prefer to keep persisting state under contention rather than crash;
    the supervisor can always re-load and re-write on the next iteration.
    """

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FileLock:
    """``fcntl.flock(LOCK_EX | LOCK_NB)`` with exponential-backoff retries.

    Python's ``fcntl`` works on macOS and Linux; on Windows the supervisor
    falls back to :class:`_NullLock`. The lock is held for the duration
    of the ``with`` block and released by closing the underlying fd.
    """

    def __init__(self, path: Path, *, exclusive: bool = True) -> None:
        self._path = Path(path)
        self._exclusive = exclusive
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        try:
            import fcntl  # type: ignore[import-not-found]
        except ImportError:
            # Windows or otherwise unavailable: act like _NullLock.
            return self
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open the lock file (we lock the SAME file we read/write so that
        # a fresh process opening it also sees the lock; using a sibling
        # ``.lock`` file would be slightly cleaner but means more state
        # to manage on disk).
        flags = os.O_CREAT | os.O_RDWR
        self._fd = os.open(str(self._path), flags, 0o644)
        op = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        op |= fcntl.LOCK_NB
        for delay in _SHAKEDOWN_LOCK_BACKOFF_S:
            try:
                fcntl.flock(self._fd, op)
                return self
            except OSError:
                time.sleep(delay)
        # One last attempt without backoff.
        try:
            fcntl.flock(self._fd, op)
        except OSError as exc:
            LOGGER.warning(
                "shakedown lock contended for %s after %d retries (%s); "
                "proceeding without lock",
                self._path,
                len(_SHAKEDOWN_LOCK_BACKOFF_S),
                exc,
            )
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is None:
            return
        try:
            import fcntl  # type: ignore[import-not-found]
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - best-effort unlock
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


def _acquire_file_lock(path: Path, *, exclusive: bool = True) -> Any:
    """Return a context manager protecting ``path`` from concurrent writers.

    On platforms with ``fcntl`` the lock is process-shared and protects
    against multi-process races (the D1 multiprocessing-per-symbol
    deployment makes this mandatory, not optional). On platforms without
    ``fcntl`` the returned object is a no-op so the call site doesn't
    have to branch.
    """
    try:
        import fcntl  # noqa: F401  - test for availability only
    except ImportError:
        return _NullLock()
    return _FileLock(path, exclusive=exclusive)


# A paper-mode signal is queued at one tick and filled at the next tick's
# ticker price (the supervisor's best approximation of the next-bar open
# relative to the signal). Pending fills older than this many ticks are
# treated as stale and dropped with a warning -- a long-deferred fill is
# almost certainly a sign the supervisor was paused / lost ticks for the
# symbol.
PAPER_PENDING_FILL_MAX_AGE_TICKS = 2


@dataclasses.dataclass
class PendingPaperFill:
    """A paper-trade signal that has been emitted but not yet filled.

    The supervisor consumes one pending entry per symbol per tick. The
    fill price is the ticker observed at the *consuming* tick, not the
    signal tick — this approximates "fill at next bar open" rather than
    "fill at signal-tick mid", removing the magic look-ahead that came
    from immediately filling at the same ticker that produced the signal.
    """

    symbol: str
    side: Literal["buy", "sell"]
    quote_size_usd: float
    base_size: float
    signal_tick_at: str
    slippage_bps: float
    enqueued_tick_index: int = 0
    # Lane E E1: optional trade_id that paper-fill drains will reuse so the
    # signal + fill snapshots share a key. None when no snapshot store is
    # wired (snapshot capture is a no-op anyway).
    trade_id: Optional[str] = None


_ActionTaken = Literal[
    "skipped_low_confidence",
    "allowed",
    "halted_breaker",
    "force_flatted",
    "errored",
]


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------


class SupervisorConfig(BaseModel):
    """Operator-supplied configuration for the supervisor."""

    model_config = ConfigDict(extra="forbid")

    symbols: List[str] = Field(..., min_length=1)
    tick_interval_s: float = 5.0
    bankroll_usd: float = 10_000.0
    mode: Literal["paper", "live"] = "paper"
    shakedown_min_days: int = 14
    shakedown_state_path: Path
    risk_pct_per_trade: float = 0.005
    min_confidence_to_trade: float = 0.6


class SymbolShakedownState(BaseModel):
    """Per-symbol shakedown evidence.

    Each tracked symbol carries its own ``paper_days_clean`` counter and
    daily history so a freshly added symbol starts at 0 even when other
    symbols are already unlocked. Account-level events (kill switch,
    daily-loss breaker) reset every symbol; per-symbol errors only reset
    the offending symbol.

    ``equity_peak_usd`` (P0 #1): per-symbol high-water mark on the
    bankroll-plus-symbol-pnl line. Drawdown is then computed against THIS
    symbol's peak, so one symbol's losses don't halt another symbol's
    trading via a shared global peak.
    """

    model_config = ConfigDict(extra="forbid")

    paper_days_clean: int = 0
    last_evaluation_utc: Optional[str] = None
    daily_history: List[Dict[str, Any]] = Field(default_factory=list)
    live_unlocked_at_utc: Optional[str] = None
    equity_peak_usd: float = 0.0


class ShakedownState(BaseModel):
    """Rolling shakedown evidence persisted between supervisor processes.

    Per-symbol evidence lives in ``per_symbol``; ``equity_peak_usd`` and
    ``started_at_utc`` remain global (they describe the whole bankroll).

    Migration: the previous schema stored ``paper_days_clean`` /
    ``daily_history`` / ``live_unlocked_at_utc`` at the top level for a
    single implicit symbol. ``_load_or_init_shakedown`` detects that
    layout and rewrites it into ``per_symbol`` on first load.
    """

    model_config = ConfigDict(extra="forbid")

    started_at_utc: str
    equity_peak_usd: float = 0.0
    per_symbol: Dict[str, SymbolShakedownState] = Field(default_factory=dict)

    def get_or_init(self, symbol: str) -> SymbolShakedownState:
        if symbol not in self.per_symbol:
            self.per_symbol[symbol] = SymbolShakedownState()
        return self.per_symbol[symbol]

    # ------------------------------------------------------------------
    # Backward-compat read-only helpers used by older callers / dashboards.
    # ------------------------------------------------------------------
    @property
    def paper_days_clean(self) -> int:
        """Most-restrictive (min) clean-day count across tracked symbols.

        Returns 0 if no symbols are tracked yet -- a fresh supervisor
        hasn't accumulated evidence for anything.
        """
        if not self.per_symbol:
            return 0
        return min(s.paper_days_clean for s in self.per_symbol.values())

    @property
    def daily_history(self) -> List[Dict[str, Any]]:
        """Flattened history across all symbols, newest-last per symbol.

        Each entry already carries its ``symbol`` key (added by
        ``evaluate_shakedown``), so consumers can group by it.
        """
        out: List[Dict[str, Any]] = []
        for s in self.per_symbol.values():
            out.extend(s.daily_history)
        return out

    @property
    def live_unlocked_at_utc(self) -> Optional[str]:
        """Earliest unlock timestamp across symbols, or None if none unlocked."""
        timestamps = [
            s.live_unlocked_at_utc
            for s in self.per_symbol.values()
            if s.live_unlocked_at_utc is not None
        ]
        return min(timestamps) if timestamps else None

    @property
    def last_evaluation_utc(self) -> Optional[str]:
        """Latest evaluation timestamp across symbols."""
        timestamps = [
            s.last_evaluation_utc
            for s in self.per_symbol.values()
            if s.last_evaluation_utc is not None
        ]
        return max(timestamps) if timestamps else None


class SupervisorTick(BaseModel):
    """Per-symbol record of one tick for telemetry / tests."""

    model_config = ConfigDict(extra="forbid")

    tick_at_utc: str
    symbol: str
    verdict: CircuitBreakerVerdict
    model_confidence: Optional[float] = None
    action_taken: _ActionTaken
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _today_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).date().isoformat()


def _placeholder_predict(
    symbol: str, ticker: Ticker
) -> Tuple[Literal["buy", "sell"], float]:
    """TODO(phase-6): replace with real ML inference.

    Returns a neutral, non-actionable signal so the loop runs end-to-end
    without trading anything. Confidence 0.5 sits below the default
    ``min_confidence_to_trade`` (0.6) so no order is ever placed.
    """
    return ("buy", 0.5)


class Supervisor:
    """Owns the trading loop and the shakedown lifecycle.

    All Phase 1-4 collaborators are injected so tests can stub them. The
    ``sleep_fn`` and ``now_fn`` injection points keep tests deterministic.
    """

    def __init__(
        self,
        *,
        config: SupervisorConfig,
        exchange: CoinbaseExchange,
        position_store: PositionStore,
        circuit_breakers: CircuitBreakerSet,
        notifier: Notifier,
        model_predict_fn: Callable[
            [str, Ticker], Tuple[Literal["buy", "sell"], float]
        ],
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = _utcnow,
        metrics_pusher: Optional[Any] = None,
        auto_pause_gate: Optional[AutoPauseGate] = None,
        confidence_history: Optional[ConfidenceHistory] = None,
        auto_trip_threshold: int = 3,
        trade_context_store: Optional[TradeContextStore] = None,
    ) -> None:
        self.config = config
        self.exchange = exchange
        self.position_store = position_store
        self.circuit_breakers = circuit_breakers
        self.notifier = notifier
        self.model_predict_fn = model_predict_fn
        self._sleep = sleep_fn
        self._now = now_fn
        self.metrics_pusher = metrics_pusher
        # Lane E E1 wiring: optional snapshot store for the loss-postmortem
        # forensics swarm. When None, the three snapshot capture points
        # (signal / fill / breaker) are no-ops, so legacy callers and
        # existing tests continue working without code changes.
        self.trade_context_store = trade_context_store
        # symbol → pre-allocated trade_id picked at signal time so the
        # fill snapshot binds to the same id as the signal snapshot.
        # Cleared by _consume_pending_trade_id after the fill is recorded.
        self._pending_trade_ids: Dict[str, str] = {}
        # Task 2: optional auto-pause gate. When None, the gate is
        # disabled and ``daily_close`` skips the check entirely (preserves
        # existing test fixtures that don't supply it).
        self.auto_pause_gate = auto_pause_gate
        # Task 2: rolling confidence history for the auto-pause baseline.
        # Defaults to an in-process buffer so tests don't need Redis.
        self.confidence_history = confidence_history or ConfidenceHistory(
            redis_client=None
        )
        # Task 3: N consecutive errors on the same symbol auto-trips the
        # kill switch by writing the configured kill-switch file. The
        # threshold is conservative (3 errors) and bounded; the supervisor
        # keeps an in-process latch so we only emit the alert + metric
        # once per trip.
        self.auto_trip_threshold = int(auto_trip_threshold)
        self._auto_tripped_symbols: set[str] = set()

        # Kill-switch trips are account-level (every symbol resets) and
        # remain in-process: the only writer is the supervisor itself,
        # so cross-process visibility isn't needed. Per-symbol error
        # counts now live in Redis (P0 #3) so multiple symbol-supervisor
        # processes can share state under the D1 multiprocessing model.
        self._kill_switch_trips_today = 0

        # One-time warning latches, keyed by symbol so the live-locked
        # warning fires once per locked symbol per process.
        self._warned_live_locked: Dict[str, bool] = {}

        # Defer-to-next-tick paper fill state machine: one pending fill per
        # symbol. Populated when paper mode produces a signal; drained at
        # the start of the next tick for that symbol against the live
        # ticker. See PendingPaperFill above for the rationale.
        self._pending_paper_fills: Dict[str, PendingPaperFill] = {}
        # Monotonic per-symbol tick counter, used to age out stale pending
        # fills (anything older than PAPER_PENDING_FILL_MAX_AGE_TICKS).
        self._symbol_tick_index: Dict[str, int] = {
            sym: 0 for sym in config.symbols
        }

        # Hydrate (or initialise) the shakedown evidence file.
        self.shakedown_state: ShakedownState = self._load_or_init_shakedown()

        # P0 #8: track the last UTC date we ran ``daily_close`` for so the
        # run loop can fire it automatically when the clock crosses
        # midnight. ``None`` on construction means "no close has fired
        # yet"; the first iteration sets it without firing daily_close
        # (otherwise a freshly booted supervisor would always emit one
        # immediately).
        self._last_close_date: Optional[date] = None

        # One-shot "supervisor started" gauge -- only if a pusher is wired in.
        self._safe_metric_call(
            lambda: self.metrics_pusher.gauge("supervisor_started", 1.0)
            if self._pusher_enabled()
            else None
        )

    # ------------------------------------------------------------------
    # Shakedown persistence
    # ------------------------------------------------------------------
    def _load_or_init_shakedown(self) -> ShakedownState:
        """flock-guarded boot-time read of the shakedown state.

        Boot-time read is part of the same race that ``_persist_shakedown``
        guards on the write side: without an exclusive lock here, two
        processes booting simultaneously can both observe a partial blob
        from each other's atomic-rename window and mutually wipe the
        shakedown counters. The lock is held for the duration of the
        read + (optional) re-persist after migration.
        """
        path = self.config.shakedown_state_path
        with _acquire_file_lock(path, exclusive=True):
            if path.exists():
                try:
                    blob = path.read_text(encoding="utf-8")
                    state, was_migrated = self._parse_shakedown_blob(blob)
                    # Make sure every configured symbol has an entry so
                    # callers can rely on ``per_symbol[symbol]``.
                    missing_symbols = [
                        s for s in self.config.symbols
                        if s not in state.per_symbol
                    ]
                    for sym in missing_symbols:
                        state.get_or_init(sym)
                    # Persist if we migrated or added new symbols. This is
                    # done OUTSIDE the lock context to avoid re-acquiring
                    # the same lock recursively (fcntl flock isn't
                    # reentrant on macOS / Linux).
                    needs_persist = bool(was_migrated or missing_symbols)
                except Exception as exc:  # noqa: BLE001 - corrupt-file recovery
                    LOGGER.warning(
                        "Shakedown state at %s is corrupt (%s); reinitialising.",
                        path,
                        exc,
                    )
                    state = ShakedownState(started_at_utc=self._now().isoformat())
                    for sym in self.config.symbols:
                        state.get_or_init(sym)
                    needs_persist = True
                else:
                    # Successful read.
                    pass
            else:
                state = ShakedownState(started_at_utc=self._now().isoformat())
                for sym in self.config.symbols:
                    state.get_or_init(sym)
                needs_persist = True
        # Out of the lock now; re-persist takes its own lock.
        if needs_persist:
            self._persist_shakedown(state)
        return state

    def _parse_shakedown_blob(self, blob: str) -> Tuple[ShakedownState, bool]:
        """Parse JSON blob, migrating from the legacy global format if needed.

        Returns ``(state, was_migrated)`` so the caller can re-persist after
        a migration to canonicalise the on-disk layout.
        """
        import json

        raw = json.loads(blob)
        was_migrated = False
        # Legacy format: top-level paper_days_clean / daily_history etc., no
        # per_symbol field. Migrate by promoting the legacy values into a
        # per-symbol entry for every currently-configured symbol so they
        # inherit the existing clean-day streak.
        if "per_symbol" not in raw and any(
            k in raw
            for k in (
                "paper_days_clean",
                "daily_history",
                "live_unlocked_at_utc",
                "last_evaluation_utc",
            )
        ):
            # Legacy ``equity_peak_usd`` lived at the top level. We keep
            # the top-level value (so ShakedownState validates) but ALSO
            # seed every per-symbol entry with the same value, since per-
            # symbol drawdown is now tracked separately (P0 #1).
            legacy_equity_peak = float(raw.get("equity_peak_usd", 0.0) or 0.0)
            legacy_per_symbol_payload = {
                "paper_days_clean": int(raw.pop("paper_days_clean", 0) or 0),
                "last_evaluation_utc": raw.pop("last_evaluation_utc", None),
                "daily_history": list(raw.pop("daily_history", []) or []),
                "live_unlocked_at_utc": raw.pop("live_unlocked_at_utc", None),
                "equity_peak_usd": legacy_equity_peak,
            }
            raw["per_symbol"] = {
                sym: dict(legacy_per_symbol_payload)
                for sym in self.config.symbols
            }
            was_migrated = True
            LOGGER.info(
                "Migrated legacy shakedown state to per-symbol layout for %d "
                "symbol(s); preserved paper_days_clean=%d, equity_peak=%.2f.",
                len(self.config.symbols),
                legacy_per_symbol_payload["paper_days_clean"],
                legacy_equity_peak,
            )
        # Defense for the in-between case: an existing per_symbol layout
        # missing the new equity_peak_usd field on individual entries.
        # Pydantic's default takes care of the nominal case, but if a
        # caller persisted entries with explicit other fields the new
        # field needs to inherit the global peak when promoting up.
        elif "per_symbol" in raw:
            global_peak = float(raw.get("equity_peak_usd", 0.0) or 0.0)
            for sym, entry in (raw.get("per_symbol") or {}).items():
                if isinstance(entry, dict) and "equity_peak_usd" not in entry:
                    entry["equity_peak_usd"] = global_peak
                    was_migrated = True
        return ShakedownState.model_validate(raw), was_migrated

    def _increment_symbol_errors(self, symbol: str) -> None:
        """Bump the per-symbol error counter for today's shakedown evaluation.

        Backed by Redis (P0 #3) so that multiple symbol-supervisor
        processes under the D1 multiprocessing model share a single
        canonical counter and survive crashes. Falls back to a no-op
        with a logged WARNING if the store is missing the method or
        the call fails -- the supervisor must NEVER crash because the
        counter is unreachable.
        """
        try:
            self.position_store.increment_error(symbol, now_utc=self._now())
        except AttributeError:
            # Older / stub stores without the new method. Log once per
            # process (the latch keeps the noise floor low).
            if not getattr(self, "_warned_no_error_counter", False):
                LOGGER.warning(
                    "position_store has no increment_error; running without "
                    "shared error counter (test stub or pre-Lane-A store?)"
                )
                self._warned_no_error_counter = True
        except Exception as exc:  # noqa: BLE001 - never crash the trader
            LOGGER.warning(
                "increment_error failed for %s: %s; continuing", symbol, exc
            )

    def _handle_tick_error(self, symbol: str) -> None:
        """Increment the per-symbol error counter and auto-trip kill switch
        if it crosses ``self.auto_trip_threshold`` consecutive errors.

        Task 3: after incrementing, if the count for THIS symbol >=
        threshold AND we haven't already tripped on this symbol in this
        process, write the kill-switch file (if configured), emit the
        ``autopilot_auto_trip_total`` metric, and send a critical alert.
        Each symbol can only trip once per process to avoid alert storms;
        the operator must clear the switch + restart to re-arm.
        """
        self._increment_symbol_errors(symbol)
        # Read the latest count back from Redis so we cross-check against
        # the persisted view (other symbol-supervisor processes may have
        # racing increments).
        count = self._errors_today_for_symbol(symbol)
        if count < int(self.auto_trip_threshold):
            return
        if symbol in self._auto_tripped_symbols:
            return
        self._auto_tripped_symbols.add(symbol)
        reason = (
            f"{count} consecutive errors on {symbol} >= "
            f"threshold {self.auto_trip_threshold}"
        )
        LOGGER.error("AUTO-TRIPPING KILL SWITCH: %s", reason)
        # Write the kill-switch file if configured. Best-effort.
        try:
            ks_path = getattr(self.circuit_breakers, "kill_switch_file", None)
            if ks_path is not None:
                Path(ks_path).parent.mkdir(parents=True, exist_ok=True)
                Path(ks_path).write_text(
                    f"auto-tripped: {reason}\n", encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001 - best-effort
            LOGGER.warning(
                "auto-trip kill-switch file write failed: %s", exc
            )
        # Sentry breadcrumb (capture_exception with synthesized error).
        try:
            _capture_exception(RuntimeError(reason))
        except Exception:  # noqa: BLE001
            pass
        self._safe_metric_call(
            lambda: self._emit_auto_trip_metric(symbol=symbol)
        )
        # Alert via the notifier; route through kill_switch_tripped so it
        # goes to BOTH Discord and Telegram.
        try:
            self.notifier.kill_switch_tripped(reason)
        except Exception as exc:  # noqa: BLE001 - notifier best-effort
            LOGGER.warning("auto-trip notifier raised: %s", exc)

    def _emit_auto_trip_metric(self, *, symbol: str) -> None:
        if not self._pusher_enabled():
            return
        self.metrics_pusher.counter(
            "auto_trip_total",
            1.0,
            labels={"symbol": symbol, "reason": "consecutive_errors"},
        )

    def _errors_today_for_symbol(self, symbol: str) -> int:
        """Read today's error count from the store. Returns 0 on any failure."""
        try:
            return int(
                self.position_store.errors_today(symbol, now_utc=self._now())
            )
        except AttributeError:
            return 0
        except Exception as exc:  # noqa: BLE001 - reads must not crash
            LOGGER.warning(
                "errors_today read failed for %s: %s; treating as 0",
                symbol,
                exc,
            )
            return 0

    def _persist_shakedown(self, state: ShakedownState) -> None:
        """Atomic, flock-guarded write of the shakedown state file.

        Under D1's multiprocessing-per-symbol model, two booting peers can
        race on this file -- without flock + atomic-rename, one peer can
        read a torn JSON blob and reset clean-day counters to 0. The
        sequence here:

          1. Acquire an exclusive flock on the canonical path.
          2. Write the new JSON to a sibling ``.tmp`` file.
          3. fsync the tmp file so the bytes are durable before rename.
          4. ``os.replace`` (atomic on POSIX) the tmp over the canonical.
          5. Release the lock.

        Failures are swallowed (with a WARNING) -- the supervisor must
        keep ticking even if disk IO falters; the in-memory state is the
        live source of truth for the current process.
        """
        path = self.config.shakedown_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = state.model_dump_json(indent=2)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with _acquire_file_lock(path, exclusive=True):
                # Write + fsync into the tmp file.
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(blob)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        # Some filesystems (tmpfs) don't support fsync;
                        # the rename below is still effectively atomic.
                        pass
                # Atomic rename. ``os.replace`` overwrites on POSIX +
                # Windows; ``Path.rename`` would raise on Windows when
                # the target exists.
                os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            LOGGER.warning("Failed to persist shakedown state to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_live_unlocked(self, symbol: Optional[str] = None) -> bool:
        """True only if mode='live' AND shakedown evidence is sufficient.

        With ``symbol`` set: returns the per-symbol gate. With no symbol:
        returns True only if EVERY configured symbol is unlocked (used for
        whole-supervisor summary / metrics).
        """
        if self.config.mode != "live":
            return False
        if symbol is not None:
            sym_state = self.shakedown_state.per_symbol.get(symbol)
            if sym_state is None:
                return False
            return sym_state.paper_days_clean >= self.config.shakedown_min_days
        # No symbol: most-restrictive aggregate over configured symbols.
        if not self.config.symbols:
            return False
        return all(
            (
                self.shakedown_state.per_symbol.get(s)
                and self.shakedown_state.per_symbol[s].paper_days_clean
                >= self.config.shakedown_min_days
            )
            for s in self.config.symbols
        )

    def run_once(self) -> List[SupervisorTick]:
        """Run a single iteration over all configured symbols."""
        ticks: List[SupervisorTick] = []
        loop_start = time.monotonic()
        for symbol in self.config.symbols:
            tick_start = time.monotonic()
            try:
                tick = self._tick_symbol(symbol)
            except Exception as exc:  # noqa: BLE001 - keep loop alive
                # Genuinely uncaught path -- tick_symbol should catch
                # ExchangeError / generic exceptions itself, but if something
                # leaks through (eg pydantic validation), don't kill the loop.
                self._handle_tick_error(symbol)
                LOGGER.exception("Unhandled tick error on %s", symbol)
                _capture_exception(exc)
                self._safe_alert(
                    f"Supervisor tick error: {symbol}: {exc}",
                    severity="alert",
                )
                tick = SupervisorTick(
                    tick_at_utc=self._now().isoformat(),
                    symbol=symbol,
                    verdict=self._allow_verdict(),
                    model_confidence=None,
                    action_taken="errored",
                    notes=f"unhandled: {exc!r}",
                )
            ticks.append(tick)
            self._safe_metric_call(
                lambda dur=time.monotonic() - tick_start, sym=symbol, tk=tick: (
                    self._emit_per_symbol_tick_metrics(sym=sym, tick=tk, duration_s=dur)
                )
            )
        # End-of-tick aggregate gauges + best-effort push.
        self._safe_metric_call(
            lambda total=time.monotonic() - loop_start: self._emit_loop_metrics(
                loop_duration_s=total
            )
        )
        self._safe_metric_call(self._safe_push)
        return ticks

    def run_loop(
        self, *, max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run until ``max_iterations`` is reached or KeyboardInterrupt.

        Returns a small summary dict (iterations completed, total ticks,
        action counts) so callers / tests can introspect results.
        """
        iterations = 0
        total_ticks = 0
        action_counts: Dict[str, int] = {}
        interrupted = False
        daily_closes_fired = 0
        try:
            while True:
                if max_iterations is not None and iterations >= max_iterations:
                    break
                ticks = self.run_once()
                iterations += 1
                total_ticks += len(ticks)
                for t in ticks:
                    action_counts[t.action_taken] = (
                        action_counts.get(t.action_taken, 0) + 1
                    )
                    # Per-tick INFO log so operators see live activity.
                    confidence = (
                        f"{t.model_confidence:.3f}"
                        if t.model_confidence is not None
                        else "n/a"
                    )
                    note_suffix = f" -- {t.notes}" if t.notes else ""
                    LOGGER.info(
                        "tick #%d | %s | action=%s | confidence=%s%s",
                        iterations,
                        t.symbol,
                        t.action_taken,
                        confidence,
                        note_suffix,
                    )

                # P0 #8: auto-fire daily_close when the UTC clock crosses
                # midnight. The first iteration after process boot only
                # records the current date (no close fires) so a fresh
                # supervisor doesn't emit one immediately on startup.
                if self._maybe_fire_daily_close():
                    daily_closes_fired += 1

                # Sleep last so a single-iter call doesn't pay an idle wait.
                if max_iterations is None or iterations < max_iterations:
                    self._sleep(self.config.tick_interval_s)
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.info("Supervisor received KeyboardInterrupt; exiting cleanly.")

        return {
            "iterations": iterations,
            "total_ticks": total_ticks,
            "action_counts": action_counts,
            "interrupted": interrupted,
            "daily_closes_fired": daily_closes_fired,
        }

    def _maybe_fire_daily_close(self) -> bool:
        """If the UTC date has rolled past ``self._last_close_date``, run
        ``daily_close`` exactly once for the new boundary.

        Returns True if a close fired, False otherwise. The first call on
        a freshly constructed Supervisor sets the baseline date but does
        not fire a close -- otherwise every booted process would emit a
        spurious close on its first tick.
        """
        today = self._now().astimezone(timezone.utc).date()
        if self._last_close_date is None:
            self._last_close_date = today
            return False
        if today > self._last_close_date:
            try:
                self.daily_close()
            except Exception as exc:  # noqa: BLE001 - never crash the loop
                LOGGER.warning(
                    "auto daily_close failed: %s; will retry on next crossing",
                    exc,
                )
                # Don't advance _last_close_date so the next iteration
                # tries again -- but we also don't want to retry every
                # tick on the same crossing, so the date IS advanced
                # only on success. (If you want retry-this-tick semantics
                # remove the early return.)
                return False
            self._last_close_date = today
            return True
        return False

    # ------------------------------------------------------------------
    # Daily rollup + shakedown evaluation
    # ------------------------------------------------------------------
    def evaluate_shakedown(self) -> ShakedownState:
        """Roll today's evidence into the shakedown state and persist it.

        Per-symbol reset triggers (reset that symbol's clean streak):
            * uncaught error inside ``_tick_symbol`` for that symbol
        Account-level reset triggers (reset EVERY symbol's clean streak):
            * kill switch tripped during the day (operator intervention)
            * realised daily PnL <= -daily_loss_limit_usd
              (only when the breaker is configured)
        """
        now = self._now()
        today_iso = _today_iso(now)
        account_daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        ks_trips = int(self._kill_switch_trips_today)

        account_loss_breaker_tripped = False
        limit = self.circuit_breakers.daily_loss_limit_usd
        if limit is not None and account_daily_pnl <= -float(limit):
            account_loss_breaker_tripped = True

        # Account-level events affect every symbol; pre-compute the flag.
        account_dirty = ks_trips > 0 or account_loss_breaker_tripped

        # Make sure every configured symbol has an entry -- a symbol added
        # mid-stream still gets evaluated.
        for sym in self.config.symbols:
            self.shakedown_state.get_or_init(sym)

        for symbol, sym_state in self.shakedown_state.per_symbol.items():
            errors_for_symbol = self._errors_today_for_symbol(symbol)
            try:
                pnl_for_symbol = float(
                    self.position_store.daily_realized_pnl_usd_for_symbol(symbol)
                )
            except Exception:  # noqa: BLE001 - state read is best-effort
                pnl_for_symbol = 0.0

            clean = (not account_dirty) and errors_for_symbol == 0

            if clean:
                sym_state.paper_days_clean += 1
            else:
                sym_state.paper_days_clean = 0

            sym_state.last_evaluation_utc = now.isoformat()
            sym_state.daily_history.append(
                {
                    "date": today_iso,
                    "symbol": symbol,
                    "daily_pnl_usd": pnl_for_symbol,
                    "errors_count": errors_for_symbol,
                    "kill_switch_trips": ks_trips,
                    "daily_loss_breaker_tripped": account_loss_breaker_tripped,
                    "clean": clean,
                }
            )
            if (
                sym_state.paper_days_clean >= self.config.shakedown_min_days
                and sym_state.live_unlocked_at_utc is None
            ):
                sym_state.live_unlocked_at_utc = now.isoformat()

            # Per-symbol high-water mark (P0 #1). The peak is bankroll +
            # this symbol's realised PnL, NOT the account aggregate; that
            # way one symbol's drawdown can't be paid for by another
            # symbol's gains, and the breaker context can compute drawdown
            # against THIS symbol only.
            sym_equity_now = self.config.bankroll_usd + pnl_for_symbol
            if sym_equity_now > sym_state.equity_peak_usd:
                sym_state.equity_peak_usd = sym_equity_now

        # Account-level: update the bankroll high-water mark.
        equity_now = self.config.bankroll_usd + account_daily_pnl
        if equity_now > self.shakedown_state.equity_peak_usd:
            self.shakedown_state.equity_peak_usd = equity_now

        self._persist_shakedown(self.shakedown_state)

        # Reset per-day counters once they've been folded into evidence.
        # Redis-backed error counter is reset via store.reset_errors_for_day
        # so other symbol-supervisor processes also see the wipe.
        try:
            self.position_store.reset_errors_for_day(now_utc=now)
        except AttributeError:
            pass  # stub stores without the method
        except Exception as exc:  # noqa: BLE001 - reset is best-effort
            LOGGER.warning("reset_errors_for_day failed: %s; continuing", exc)
        self._kill_switch_trips_today = 0
        return self.shakedown_state

    def daily_close(self) -> Dict[str, Any]:
        """Send daily summary, persist evidence, return rollup dict."""
        now = self._now()
        daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        open_positions = self.position_store.list_open()
        closed_today = self.position_store.list_closed_today(now_utc=now)
        equity_usd = self.config.bankroll_usd + daily_pnl

        # Task 5: Sentry breadcrumb so postmortems see daily-close events.
        try:
            from observability.monitoring import breadcrumb as _breadcrumb

            _breadcrumb(
                category="supervisor.daily_close",
                message=f"daily_close at {now.isoformat()}",
                data={
                    "daily_pnl_usd": daily_pnl,
                    "open_positions": len(open_positions),
                    "closed_today": len(closed_today),
                    "equity_usd": equity_usd,
                },
            )
        except Exception:  # noqa: BLE001 - never let breadcrumbs crash
            pass

        # Send Discord-only summary -- best-effort, never raises.
        try:
            self.notifier.daily_summary(
                equity_usd=equity_usd,
                daily_pnl_usd=daily_pnl,
                open_positions=len(open_positions),
                closed_today=len(closed_today),
            )
        except Exception as exc:  # noqa: BLE001 - notifier is best-effort
            LOGGER.warning("notifier.daily_summary raised: %s", exc)

        # Task 2: auto-pause check. The gate is opt-in -- if no gate was
        # injected, this block is a no-op. Combined daily-loss + confidence-
        # shift trip writes a marker file + alerts + metric. The next tick
        # observes the marker and halts via ``_check_auto_pause_marker``.
        if self.auto_pause_gate is not None:
            self._check_auto_pause(daily_pnl_usd=daily_pnl)

        # Roll today's evidence into shakedown (this also persists).
        state = self.evaluate_shakedown()

        # Daily-close monitoring gauges -- best-effort, never raises.
        self._safe_metric_call(
            lambda: self._emit_daily_close_metrics(
                equity_usd=equity_usd,
                daily_pnl_usd=daily_pnl,
                open_positions=len(open_positions),
                paper_days_clean=state.paper_days_clean,
            )
        )
        self._safe_metric_call(self._safe_push)

        return {
            "date": _today_iso(now),
            "equity_usd": equity_usd,
            "daily_pnl_usd": daily_pnl,
            "open_positions": len(open_positions),
            "closed_today": len(closed_today),
            "paper_days_clean": state.paper_days_clean,
            "live_unlocked": self.is_live_unlocked(),
        }

    # ------------------------------------------------------------------
    # Per-symbol tick
    # ------------------------------------------------------------------
    def _tick_symbol(self, symbol: str) -> SupervisorTick:
        now = self._now()
        # 1. Kill switch first.
        if self.circuit_breakers.is_kill_switch_tripped():
            self._kill_switch_trips_today += 1
            count = self._force_flat_all(reason="kill_switch_file_present")
            self._safe_kill_switch_alert(
                f"kill switch tripped; force-closed {count} position(s)"
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=CircuitBreakerVerdict(
                    allow=False,
                    tripped=["kill_switch"],
                    reason="kill switch active",
                    recommended_action="force_flat",
                    details={},
                ),
                model_confidence=None,
                action_taken="force_flatted",
                notes=f"force_closed={count}",
            )

        # 2. Fetch ticker (this is the first place a network ExchangeError
        # is plausible).
        try:
            ticker = self.exchange.get_ticker(symbol)
        except ExchangeError as exc:
            self._handle_tick_error(symbol)
            self._safe_alert(
                f"get_ticker failed for {symbol}: {exc}", severity="alert"
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=self._allow_verdict(),
                model_confidence=None,
                action_taken="errored",
                notes=f"get_ticker: {exc!r}",
            )

        # 2b. Bump per-symbol tick index, then drain any pending paper fill
        # against the FRESH ticker. This is the heart of the
        # defer-to-next-tick state machine -- the fill price for a paper
        # signal emitted at tick N is the ticker observed at tick N+1.
        self._symbol_tick_index[symbol] = (
            self._symbol_tick_index.get(symbol, 0) + 1
        )
        self._drain_pending_paper_fill(symbol, ticker)

        # 3. Build context + run breakers.
        proposed = float(self.config.bankroll_usd) * float(
            self.config.risk_pct_per_trade
        )
        daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        # Per-symbol equity tracking (P0 #1). Drawdown is computed against
        # THIS symbol's realised PnL only, against THIS symbol's peak. A
        # losing symbol therefore halts only itself; its peers keep
        # trading.
        try:
            symbol_pnl = float(
                self.position_store.daily_realized_pnl_usd_for_symbol(symbol)
            )
        except Exception:  # noqa: BLE001 - state read is best-effort
            symbol_pnl = 0.0
        equity_current = float(self.config.bankroll_usd) + symbol_pnl
        sym_state = self.shakedown_state.per_symbol.get(symbol)
        sym_peak = float(sym_state.equity_peak_usd) if sym_state else 0.0
        equity_peak = max(equity_current, sym_peak)

        # We need a tentative side for the breaker context. Predict first so
        # the breaker sees the actual side, but the canonical confidence
        # gate happens AFTER the breaker decides.
        try:
            side, confidence = self.model_predict_fn(symbol, ticker)
        except Exception as exc:  # noqa: BLE001 - model errors should not crash
            self._handle_tick_error(symbol)
            self._safe_alert(
                f"model_predict_fn failed for {symbol}: {exc}",
                severity="alert",
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=self._allow_verdict(),
                model_confidence=None,
                action_taken="errored",
                notes=f"predict: {exc!r}",
            )

        # Task 2: record confidence into the rolling baseline buffer.
        # Best-effort and silent on failure.
        try:
            if math.isfinite(float(confidence)):
                self.confidence_history.record(symbol, float(confidence))
        except Exception as exc:  # noqa: BLE001 - never let history kill trader
            LOGGER.warning(
                "confidence_history.record failed for %s: %s", symbol, exc
            )

        ctx = DecisionContext(
            symbol=symbol,
            side=side,
            proposed_notional_usd=proposed,
            current_open_notional_usd=float(
                self.position_store.open_notional_usd()
            ),
            current_per_symbol_notional_usd=float(
                self.position_store.open_notional_for_symbol(symbol)
            ),
            daily_realized_pnl_usd=daily_pnl,
            equity_peak_usd=equity_peak,
            equity_current_usd=equity_current,
            as_of_utc=now.isoformat(),
        )
        verdict = self.circuit_breakers.check(ctx)

        if verdict.recommended_action == "force_flat":
            count = self._force_flat_all(reason=verdict.reason or "force_flat")
            self._safe_kill_switch_alert(
                f"force_flat verdict; closed {count} position(s) — {verdict.reason}"
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                action_taken="force_flatted",
                notes=f"force_closed={count}",
            )

        if verdict.recommended_action == "halt_new_entries":
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                action_taken="halted_breaker",
                notes=verdict.reason or None,
            )

        # 5a. Defense-in-depth (P0 #6): predictors should already neutralise
        # NaN/inf, but a misbehaving predict_fn could still return one. The
        # native ``confidence < floor`` comparison silently passes NaN
        # (NaN < anything is False), which would let a bogus signal through.
        # Reject explicitly here as a last line of defence.
        if not math.isfinite(float(confidence)):
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=None,
                action_taken="skipped_low_confidence",
                notes="nan_confidence",
            )

        # 5. Confidence gate.
        if confidence < self.config.min_confidence_to_trade:
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                action_taken="skipped_low_confidence",
                notes=(
                    f"confidence {confidence:.3f} < floor "
                    f"{self.config.min_confidence_to_trade:.3f}"
                ),
            )

        # 5b. Lane E E1: capture the signal snapshot BEFORE the mode gate
        # AND before order placement so a forensics agent can reconstruct
        # decision-time context even if the order itself fails or the
        # tick falls back to paper because shakedown isn't unlocked.
        # Pre-allocate a trade_id so the fill snapshot binds to the same id.
        signal_trade_id = self._allocate_trade_id(symbol)
        self._safe_snapshot_signal(
            symbol=symbol,
            trade_id=signal_trade_id,
            side=side,
            confidence=confidence,
            ticker=ticker,
            ctx=ctx,
        )

        # 6. Mode gate -- live falls back to paper if THIS symbol's shakedown
        # isn't unlocked. Other symbols can still trade live.
        effective_mode: Literal["paper", "live"] = "paper"
        notes_extra: Optional[str] = None
        if self.config.mode == "live":
            if self.is_live_unlocked(symbol):
                effective_mode = "live"
            else:
                if not self._warned_live_locked.get(symbol, False):
                    sym_state = self.shakedown_state.per_symbol.get(symbol)
                    sym_clean = sym_state.paper_days_clean if sym_state else 0
                    LOGGER.warning(
                        "mode='live' but shakedown not unlocked for %s "
                        "(paper_days_clean=%d, required=%d); "
                        "falling back to paper-trade simulation.",
                        symbol,
                        sym_clean,
                        self.config.shakedown_min_days,
                    )
                    self._warned_live_locked[symbol] = True
                # Tick recorded as halted_breaker per spec, with paper sim.
                self._paper_simulate_fill(
                    symbol=symbol,
                    side=side,
                    ticker=ticker,
                    proposed_usd=proposed,
                )
                return SupervisorTick(
                    tick_at_utc=now.isoformat(),
                    symbol=symbol,
                    verdict=verdict,
                    model_confidence=confidence,
                    action_taken="halted_breaker",
                    notes="live_mode_locked",
                )

        # 7. Place order.
        try:
            if effective_mode == "live":
                self._place_live_order(
                    symbol=symbol,
                    side=side,
                    ticker=ticker,
                    proposed_usd=proposed,
                )
                notes_extra = "live"
            else:
                self._paper_simulate_fill(
                    symbol=symbol,
                    side=side,
                    ticker=ticker,
                    proposed_usd=proposed,
                )
                notes_extra = "paper"
        except ExchangeError as exc:
            self._handle_tick_error(symbol)
            self._safe_alert(
                f"place_market_order failed for {symbol}: {exc}",
                severity="alert",
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                action_taken="errored",
                notes=f"order: {exc!r}",
            )
        except Exception as exc:  # noqa: BLE001 - keep loop alive
            self._handle_tick_error(symbol)
            LOGGER.exception("Unexpected order error on %s", symbol)
            self._safe_alert(
                f"unexpected order error for {symbol}: {exc}",
                severity="alert",
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                action_taken="errored",
                notes=f"order: {exc!r}",
            )

        return SupervisorTick(
            tick_at_utc=now.isoformat(),
            symbol=symbol,
            verdict=verdict,
            model_confidence=confidence,
            action_taken="allowed",
            notes=notes_extra,
        )

    # ------------------------------------------------------------------
    # Order placement helpers
    # ------------------------------------------------------------------
    def _place_live_order(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        ticker: Ticker,
        proposed_usd: float,
    ) -> None:
        """Place a real market order. Raises ExchangeError on failure."""
        order_start = time.monotonic()
        try:
            order: OrderResult = self.exchange.place_market_order(
                symbol, side, quote_size_usd=proposed_usd
            )
        finally:
            # Task 5: order placement latency histogram, even on failure.
            self._safe_metric_call(
                lambda d=time.monotonic() - order_start, sym=symbol: (
                    self._emit_order_latency_metric(symbol=sym, duration_s=d)
                )
            )
        # Record the position; promote to open if the response shows fully
        # filled, otherwise keep as pending so reconcile can clean up later.
        position = self._position_from_order(
            order=order, symbol=symbol, side=side, fallback_price=ticker.mid
        )
        # Lane E E1: rebind position_id to the trade_id pre-allocated at
        # signal time so signal + fill snapshots share the same key. If
        # there's no pending trade_id (capture path disabled / no signal
        # snapshot taken) the original UUID stays.
        pending_trade_id = self._consume_pending_trade_id(symbol)
        if pending_trade_id is not None:
            position = position.model_copy(
                update={"position_id": pending_trade_id}
            )
        if order.status == "filled":
            self.position_store.record_open(position)
        else:
            self.position_store.record_pending(position)

        # Lane E E1: fill snapshot — captures observed fill price + slippage
        # vs the signal-time ticker mid. Bound to the same trade_id as the
        # signal snapshot when possible.
        if pending_trade_id is not None:
            try:
                slippage_bps = (
                    (float(position.entry_price) - float(ticker.mid))
                    / float(ticker.mid)
                    * 10_000.0
                ) if float(ticker.mid) > 0 else 0.0
            except (TypeError, ValueError, ZeroDivisionError):
                slippage_bps = 0.0
            self._safe_snapshot_fill(
                symbol=symbol,
                trade_id=pending_trade_id,
                position=position,
                ticker=ticker,
                slippage_bps=slippage_bps,
                notes=f"live order_status={order.status}",
            )

        # Best-effort fill notification.
        try:
            self.notifier.fill_event(
                symbol=symbol,
                side=side,
                fill_price=float(order.avg_fill_price or ticker.mid),
                fill_size=float(order.filled_base or position.base_size),
                fees_usd=float(order.fee_usd),
            )
        except Exception as exc:  # noqa: BLE001 - notifier best-effort
            LOGGER.warning("notifier.fill_event raised: %s", exc)

    def _paper_simulate_fill(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        ticker: Ticker,
        proposed_usd: float,
    ) -> None:
        """Queue a paper-trade signal for next-tick fill (P0 #4).

        The classic mistake here was to fill at the *current* ticker mid,
        which leaks the same bar that produced the signal back into the
        fill price (a kind of look-ahead bias). The defer-to-next-tick
        state machine instead records a :class:`PendingPaperFill` against
        ``symbol``; the *next* tick that runs ``_tick_symbol(symbol)``
        drains the queue and synthesises the actual position using the
        ticker observed at that later tick.

        Slippage at 5 bps is still applied -- but only when the deferred
        fill consumes against the next ticker.

        If a pending fill already exists for ``symbol`` (eg. the previous
        tick produced a signal that hasn't been drained yet), the existing
        one is replaced with the newer signal: the next tick consumes only
        one entry, and a fresher signal is more representative of intent.
        """
        # Compute base size off the *signal-tick* ticker only as a
        # provisional sizing input -- the actual fill price is set when
        # the pending entry is drained.
        provisional_price = float(ticker.mid) or 1.0
        base_size = float(proposed_usd) / provisional_price
        # Lane E E1: forward the signal-time trade_id (if any) so the
        # deferred fill snapshot can rebind. Pop here so the live-mode
        # path can't reuse a stale trade_id later in the same tick.
        pending_trade_id = self._consume_pending_trade_id(symbol)
        self._pending_paper_fills[symbol] = PendingPaperFill(
            symbol=symbol,
            side=side,
            quote_size_usd=float(proposed_usd),
            base_size=base_size,
            signal_tick_at=self._now().isoformat(),
            slippage_bps=PAPER_SLIPPAGE_BPS,
            enqueued_tick_index=self._symbol_tick_index.get(symbol, 0),
            trade_id=pending_trade_id,
        )
        LOGGER.info(
            "paper signal queued for next-tick fill: %s side=%s notional=%.2f USD "
            "(provisional_price=%.4f)",
            symbol,
            side,
            float(proposed_usd),
            provisional_price,
        )

    def _drain_pending_paper_fill(
        self, symbol: str, ticker: Ticker
    ) -> Optional[Position]:
        """Consume a queued :class:`PendingPaperFill` against ``ticker``.

        Returns the recorded :class:`Position` if a fill was synthesised,
        ``None`` otherwise. Stale pending fills (older than
        :data:`PAPER_PENDING_FILL_MAX_AGE_TICKS` ticks) are discarded with
        a WARNING -- a long-deferred fill almost always means the
        supervisor lost ticks for that symbol and re-using the signal would
        compound the error.
        """
        pending = self._pending_paper_fills.pop(symbol, None)
        if pending is None:
            return None

        current_tick_index = self._symbol_tick_index.get(symbol, 0)
        age_ticks = current_tick_index - pending.enqueued_tick_index
        if age_ticks > PAPER_PENDING_FILL_MAX_AGE_TICKS:
            LOGGER.warning(
                "paper fill: dropping stale pending signal for %s "
                "(age=%d ticks > max=%d); signal_at=%s",
                symbol,
                age_ticks,
                PAPER_PENDING_FILL_MAX_AGE_TICKS,
                pending.signal_tick_at,
            )
            return None

        slip = float(pending.slippage_bps) / 10_000.0
        if pending.side == "buy":
            fill_price = float(ticker.mid) * (1.0 + slip)
        else:
            fill_price = float(ticker.mid) * (1.0 - slip)
        if fill_price <= 0:
            fill_price = float(ticker.mid) or 1.0

        base_size = float(pending.quote_size_usd) / fill_price
        # Lane E E1: bind the position_id to the signal-time trade_id when
        # one exists, so the eventual fill snapshot shares a key with the
        # signal snapshot. Otherwise fall back to a fresh UUID.
        position_id = pending.trade_id or str(uuid.uuid4())
        position = Position(
            position_id=position_id,
            exchange="coinbase-paper",
            symbol=symbol,
            side="long" if pending.side == "buy" else "short",
            status="open",
            entry_price=fill_price,
            entry_quote_usd=fill_price * base_size,
            base_size=base_size,
            entry_order_id=f"paper-{uuid.uuid4().hex[:12]}",
            opened_at_utc=self._now().isoformat(),
            fees_usd=0.0,
            notes="paper-deferred-fill",
        )
        self.position_store.record_open(position)

        # Lane E E1: paper-fill snapshot. Only emit when a trade_id was
        # carried forward from the signal — without one the fill snapshot
        # would orphan itself.
        if pending.trade_id is not None:
            self._safe_snapshot_fill(
                symbol=symbol,
                trade_id=pending.trade_id,
                position=position,
                ticker=ticker,
                slippage_bps=float(pending.slippage_bps),
                notes="paper-deferred-fill",
            )

        LOGGER.info(
            "paper fill: drained pending %s side=%s at next-tick price %.4f "
            "(signal_at=%s, fill_at=%s)",
            symbol,
            pending.side,
            fill_price,
            pending.signal_tick_at,
            position.opened_at_utc,
        )
        return position

    def _position_from_order(
        self,
        *,
        order: OrderResult,
        symbol: str,
        side: Literal["buy", "sell"],
        fallback_price: float,
    ) -> Position:
        fill_price = float(order.avg_fill_price or fallback_price or 0.0) or 1.0
        base_size = float(order.filled_base or 0.0)
        if base_size <= 0 and order.quote_size_usd:
            base_size = float(order.quote_size_usd) / fill_price
        if base_size <= 0:
            base_size = 0.0
        return Position(
            position_id=str(uuid.uuid4()),
            exchange="coinbase",
            symbol=symbol,
            side="long" if side == "buy" else "short",
            status="open" if order.status == "filled" else "pending",
            entry_price=fill_price,
            entry_quote_usd=fill_price * base_size,
            base_size=base_size,
            entry_order_id=order.order_id,
            opened_at_utc=self._now().isoformat(),
            fees_usd=float(order.fee_usd),
        )

    # ------------------------------------------------------------------
    # Force-flat
    # ------------------------------------------------------------------
    def _force_flat_all(self, *, reason: str) -> int:
        """Best-effort close of every open position. Returns count closed."""
        closed = 0
        for position in self.position_store.list_open():
            # Lane E E1: capture a breaker snapshot per affected position
            # BEFORE the close attempt, so the snapshot is recorded even
            # if the exchange close call fails (for the postmortem swarm
            # to investigate why we tried to force-flat).
            self._safe_snapshot_breaker(position=position, reason=reason)

            close_side: Literal["buy", "sell"] = (
                "sell" if position.side == "long" else "buy"
            )
            try:
                self.exchange.place_market_order(
                    position.symbol, close_side, base_size=position.base_size
                )
                closed += 1
            except Exception as exc:  # noqa: BLE001 - keep iterating
                LOGGER.warning(
                    "force-flat close failed for %s (%s): %s",
                    position.position_id,
                    position.symbol,
                    exc,
                )
        return closed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _allow_verdict(self) -> CircuitBreakerVerdict:
        return CircuitBreakerVerdict(
            allow=True,
            tripped=[],
            reason="",
            recommended_action="allow",
            details={},
        )

    def _safe_alert(self, message: str, *, severity: str = "alert") -> None:
        try:
            self.notifier.alert(message, severity=severity)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 - notifier is best-effort
            LOGGER.warning("notifier.alert raised: %s", exc)

    def _safe_kill_switch_alert(self, reason: str) -> None:
        try:
            self.notifier.kill_switch_tripped(reason)
        except Exception as exc:  # noqa: BLE001 - notifier is best-effort
            LOGGER.warning("notifier.kill_switch_tripped raised: %s", exc)

    # ------------------------------------------------------------------
    # Lane E E1 snapshot capture helpers
    # ------------------------------------------------------------------
    #
    # Three capture points wired into _tick_symbol + force-flat:
    #   1. signal  — after model_predict_fn returns, BEFORE order placement.
    #   2. fill    — after the fill is recorded in position_store.
    #   3. breaker — when a breaker forces flat, per affected position.
    #
    # All three helpers are no-ops when self.trade_context_store is None.
    # All three swallow exceptions: snapshot writes must NEVER crash the
    # tick loop. Forensics is observability, not the trading critical path.

    def _allocate_trade_id(self, symbol: str) -> str:
        """Pre-allocate the trade_id used for both signal + fill snapshots.

        Stored in ``self._pending_trade_ids[symbol]`` so the fill-side path
        can rebind to the same id without further plumbing.
        """
        trade_id = uuid.uuid4().hex
        self._pending_trade_ids[symbol] = trade_id
        return trade_id

    def _consume_pending_trade_id(self, symbol: str) -> Optional[str]:
        """Pop the trade_id pre-allocated for ``symbol`` at signal time."""
        return self._pending_trade_ids.pop(symbol, None)

    def _safe_snapshot_signal(
        self,
        *,
        symbol: str,
        trade_id: str,
        side: Literal["buy", "sell"],
        confidence: float,
        ticker: Ticker,
        ctx: DecisionContext,
    ) -> None:
        """Capture the signal-time snapshot. No-op if no store is wired."""
        if self.trade_context_store is None:
            return
        try:
            snap = TradeContextSnapshot(
                trade_id=trade_id,
                symbol=symbol,
                captured_at_utc=utc_now_iso(),
                phase="signal",
                feature_buffer={},
                feature_window=None,
                model_probs={},
                model_confidence=float(confidence),
                risk_metrics_input={
                    "side": str(side),
                    "proposed_notional_usd": float(ctx.proposed_notional_usd),
                    "current_open_notional_usd": float(
                        ctx.current_open_notional_usd
                    ),
                    "current_per_symbol_notional_usd": float(
                        ctx.current_per_symbol_notional_usd
                    ),
                    "daily_realized_pnl_usd": float(ctx.daily_realized_pnl_usd),
                    "equity_peak_usd": float(ctx.equity_peak_usd),
                    "equity_current_usd": float(ctx.equity_current_usd),
                },
                risk_metrics_output={},
                breaker_context={},
                ticker_buffer=[
                    {
                        "bid": float(ticker.bid),
                        "ask": float(ticker.ask),
                        "last": float(ticker.last),
                        "mid": float(ticker.mid),
                    }
                ],
                notes=None,
            )
            self.trade_context_store.record_snapshot(snap)
        except Exception as exc:  # noqa: BLE001 - never crash the tick
            LOGGER.warning(
                "trade_context_store signal snapshot failed for %s: %r",
                symbol,
                exc,
            )

    def _safe_snapshot_fill(
        self,
        *,
        symbol: str,
        trade_id: str,
        position: Position,
        ticker: Optional[Ticker] = None,
        slippage_bps: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Capture the fill snapshot. No-op if no store is wired."""
        if self.trade_context_store is None:
            return
        try:
            ticker_buf: List[Dict[str, float]] = []
            if ticker is not None:
                ticker_buf.append(
                    {
                        "bid": float(ticker.bid),
                        "ask": float(ticker.ask),
                        "last": float(ticker.last),
                        "mid": float(ticker.mid),
                    }
                )
            risk_out: Dict[str, Any] = {
                "position_id": position.position_id,
                "side": position.side,
                "fill_price": float(position.entry_price),
                "fill_size": float(position.base_size),
                "fill_quote_usd": float(position.entry_quote_usd),
                "fees_usd": float(position.fees_usd),
                "exchange": position.exchange,
                "status": position.status,
            }
            if slippage_bps is not None:
                risk_out["slippage_bps"] = float(slippage_bps)
            snap = TradeContextSnapshot(
                trade_id=trade_id,
                symbol=symbol,
                captured_at_utc=utc_now_iso(),
                phase="fill",
                feature_buffer={},
                feature_window=None,
                model_probs={},
                model_confidence=0.0,
                risk_metrics_input={},
                risk_metrics_output=risk_out,
                breaker_context={},
                ticker_buffer=ticker_buf,
                notes=notes,
            )
            self.trade_context_store.record_snapshot(snap)
        except Exception as exc:  # noqa: BLE001 - never crash the tick
            LOGGER.warning(
                "trade_context_store fill snapshot failed for %s: %r",
                symbol,
                exc,
            )

    def _safe_snapshot_breaker(
        self,
        *,
        position: Position,
        verdict: Optional[CircuitBreakerVerdict] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Capture the breaker-time snapshot for a single forced-flat position."""
        if self.trade_context_store is None:
            return
        try:
            br_ctx: Dict[str, Any] = {}
            if verdict is not None:
                br_ctx = {
                    "tripped": list(verdict.tripped),
                    "reason": verdict.reason,
                    "recommended_action": verdict.recommended_action,
                    "details": dict(verdict.details or {}),
                }
            elif reason is not None:
                br_ctx = {"reason": reason}

            snap = TradeContextSnapshot(
                trade_id=position.position_id,
                symbol=position.symbol,
                captured_at_utc=utc_now_iso(),
                phase="breaker",
                feature_buffer={},
                feature_window=None,
                model_probs={},
                model_confidence=0.0,
                risk_metrics_input={
                    "side": position.side,
                    "entry_price": float(position.entry_price),
                    "base_size": float(position.base_size),
                    "entry_quote_usd": float(position.entry_quote_usd),
                },
                risk_metrics_output={},
                breaker_context=br_ctx,
                ticker_buffer=[],
                notes=reason,
            )
            self.trade_context_store.record_snapshot(snap)
        except Exception as exc:  # noqa: BLE001 - never crash the tick
            LOGGER.warning(
                "trade_context_store breaker snapshot failed for %s: %r",
                position.position_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Task 2: auto-pause gate
    # ------------------------------------------------------------------
    def _check_auto_pause(self, *, daily_pnl_usd: float) -> bool:
        """If the auto-pause gate trips, write the marker + alert + metric.

        Returns True iff the gate tripped on this call. Best-effort: no
        path raises through the supervisor's daily_close.
        """
        gate = self.auto_pause_gate
        if gate is None:
            return False

        # Aggregate the most recent confidences across all configured
        # symbols. The gate's own ``recent_window`` truncates if needed.
        recent: List[float] = []
        baselines: List[Tuple[float, float, int]] = []
        for sym in self.config.symbols:
            try:
                recent.extend(self.confidence_history.values(sym))
                baselines.append(self.confidence_history.baseline(sym))
            except Exception as exc:  # noqa: BLE001 - read is best-effort
                LOGGER.warning(
                    "confidence_history read failed for %s: %s", sym, exc
                )

        # Aggregate baseline = sample-size-weighted mean of per-symbol means.
        # If only one symbol has a usable baseline, just use it.
        usable = [(m, s, n) for (m, s, n) in baselines if n >= 2]
        if usable:
            total_n = sum(n for (_, _, n) in usable)
            agg_mean = sum(m * n for (m, _, n) in usable) / max(total_n, 1)
            # Pooled std (rough): mean of per-symbol stds, weighted by N.
            agg_std = sum(s * n for (_, s, n) in usable) / max(total_n, 1)
        else:
            agg_mean = 0.0
            agg_std = 0.0

        try:
            decision = gate.evaluate_detailed(
                daily_pnl_usd=float(daily_pnl_usd),
                recent_confidences=recent,
                baseline_confidence_mean=agg_mean,
                baseline_confidence_std=agg_std,
                bankroll_usd=float(self.config.bankroll_usd),
            )
        except Exception as exc:  # noqa: BLE001 - gate must never crash trader
            LOGGER.warning("auto_pause_gate.evaluate failed: %s", exc)
            return False

        if not decision.should_pause:
            return False

        LOGGER.warning("AUTO-PAUSE TRIPPED: %s", decision.reason)
        try:
            gate.write_marker(reason=decision.reason)
        except Exception as exc:  # noqa: BLE001 - marker write is best-effort
            LOGGER.warning("auto_pause write_marker failed: %s", exc)

        self._safe_metric_call(
            lambda: self._emit_auto_pause_metric(reason=decision.reason)
        )
        try:
            self.notifier.alert(
                f"AUTO-PAUSE TRIPPED: {decision.reason}",
                severity="critical",
                fields={
                    "daily_pnl_usd": f"{decision.daily_pnl_usd:.2f}",
                    "loss_threshold_usd": f"{decision.loss_threshold_usd:.2f}",
                    "recent_mean": (
                        f"{decision.recent_mean:.3f}"
                        if decision.recent_mean is not None
                        else "n/a"
                    ),
                    "baseline_mean": (
                        f"{decision.baseline_mean:.3f}"
                        if decision.baseline_mean is not None
                        else "n/a"
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001 - notifier best-effort
            LOGGER.warning("auto_pause notifier.alert raised: %s", exc)
        return True

    def _emit_auto_pause_metric(self, *, reason: str) -> None:
        if not self._pusher_enabled():
            return
        # Coerce reason into a label-safe truncated form (Prometheus
        # label values are unbounded but high-cardinality reasons are
        # bad practice; we surface the structural reason only).
        label_reason = "loss_and_confidence_shift"
        self.metrics_pusher.counter(
            "auto_pause_total", 1.0, labels={"reason": label_reason}
        )

    # ------------------------------------------------------------------
    # Observability helpers (defensive: monitoring failures must never
    # crash the trader)
    # ------------------------------------------------------------------
    def _pusher_enabled(self) -> bool:
        if self.metrics_pusher is None:
            return False
        try:
            return bool(self.metrics_pusher.is_enabled())
        except Exception:  # noqa: BLE001 - tolerate flaky stubs
            return False

    def _safe_metric_call(self, fn: Callable[[], Any]) -> None:
        """Run ``fn``; swallow + log any exception."""
        if self.metrics_pusher is None:
            return
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - never let metrics kill trader
            LOGGER.warning("metrics call raised: %s", exc)

    def _safe_push(self) -> None:
        if not self._pusher_enabled():
            return
        try:
            self.metrics_pusher.push()
        except Exception as exc:  # noqa: BLE001 - push must never raise
            LOGGER.warning("metrics_pusher.push raised: %s", exc)

    def _emit_per_symbol_tick_metrics(
        self,
        *,
        sym: str,
        tick: SupervisorTick,
        duration_s: float,
    ) -> None:
        if not self._pusher_enabled():
            return
        labels = {"symbol": sym, "action": tick.action_taken}
        self.metrics_pusher.counter("ticks_total", 1.0, labels=labels)
        self.metrics_pusher.histogram(
            "tick_duration_seconds", float(duration_s), labels={"symbol": sym}
        )
        # Task 5: explicit per-tick duration histogram on the canonical
        # ``autopilot_tick_duration_s`` name requested for the SLO board.
        self.metrics_pusher.histogram(
            "tick_duration_s", float(duration_s), labels={"symbol": sym}
        )
        # Task 5: model confidence distribution. Only observe when we have
        # a finite value -- skipped/errored ticks legitimately have None.
        if tick.model_confidence is not None and math.isfinite(
            float(tick.model_confidence)
        ):
            self.metrics_pusher.histogram(
                "model_confidence",
                float(tick.model_confidence),
                labels={"symbol": sym},
            )
        if tick.action_taken == "errored":
            self.metrics_pusher.counter("errors_total", 1.0, labels={"symbol": sym})
        if tick.action_taken == "allowed":
            # Successful entry-fill counter (live or paper).
            self.metrics_pusher.counter("fills_total", 1.0, labels={"symbol": sym})
        if tick.action_taken == "force_flatted":
            self.metrics_pusher.counter(
                "kill_switch_trips_total", 1.0, labels={"symbol": sym}
            )

    def _emit_loop_metrics(self, *, loop_duration_s: float) -> None:
        """Aggregate gauges + breaker state at the end of each tick loop."""
        if not self._pusher_enabled():
            return
        try:
            daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        except Exception:  # noqa: BLE001 - state read is best-effort
            daily_pnl = 0.0
        equity_usd = float(self.config.bankroll_usd) + daily_pnl
        try:
            open_positions = len(self.position_store.list_open())
        except Exception:  # noqa: BLE001 - state read is best-effort
            open_positions = 0
        try:
            kill_active = 1.0 if self.circuit_breakers.is_kill_switch_tripped() else 0.0
        except Exception:  # noqa: BLE001 - breaker read is best-effort
            kill_active = 0.0

        self.metrics_pusher.gauge("equity_usd", equity_usd)
        self.metrics_pusher.gauge("daily_pnl_usd", daily_pnl)
        self.metrics_pusher.gauge("open_positions_count", float(open_positions))
        self.metrics_pusher.gauge("kill_switch_active", kill_active)
        self.metrics_pusher.gauge(
            "paper_days_clean", float(self.shakedown_state.paper_days_clean)
        )
        # Task 5: per-symbol shakedown clean days + daily PnL. Surfaces
        # the per-symbol view that's already in the shakedown state but
        # wasn't reaching Prometheus -- needed for the per-symbol SLO
        # board because the global gauge collapses min(per_symbol).
        for sym in self.config.symbols:
            sym_state = self.shakedown_state.per_symbol.get(sym)
            if sym_state is not None:
                self.metrics_pusher.gauge(
                    "shakedown_clean_days",
                    float(sym_state.paper_days_clean),
                    labels={"symbol": sym},
                )
            try:
                pnl_for_symbol = float(
                    self.position_store.daily_realized_pnl_usd_for_symbol(sym)
                )
            except Exception:  # noqa: BLE001 - read is best-effort
                continue
            # Per-symbol PnL goes on a distinct metric name -- the unlabeled
            # ``autopilot_daily_pnl_usd`` collector cache would reject a
            # labeled call (Prometheus collectors can't change label names
            # after registration). Use ``autopilot_daily_pnl_usd_by_symbol``
            # for the per-symbol cut.
            self.metrics_pusher.gauge(
                "daily_pnl_usd_by_symbol",
                pnl_for_symbol,
                labels={"symbol": sym},
            )
        # Task 4 surface: orphan position count gauge (from PositionStore).
        try:
            orphans = float(self.position_store.orphan_count())
            self.metrics_pusher.gauge("orphan_positions", orphans)
        except (AttributeError, Exception):  # noqa: BLE001 - tolerate stub stores
            pass

    def _emit_daily_close_metrics(
        self,
        *,
        equity_usd: float,
        daily_pnl_usd: float,
        open_positions: int,
        paper_days_clean: int,
    ) -> None:
        if not self._pusher_enabled():
            return
        self.metrics_pusher.gauge("equity_usd", float(equity_usd))
        self.metrics_pusher.gauge("daily_pnl_usd", float(daily_pnl_usd))
        self.metrics_pusher.gauge("open_positions_count", float(open_positions))
        self.metrics_pusher.gauge("paper_days_clean", float(paper_days_clean))
        self.metrics_pusher.gauge(
            "shakedown_unlocked", 1.0 if self.is_live_unlocked() else 0.0
        )

    def _emit_order_latency_metric(
        self, *, symbol: str, duration_s: float
    ) -> None:
        """Task 5: ``autopilot_order_latency_s`` histogram. Always called
        from ``_place_live_order``'s finally-block so latency is recorded
        even when the exchange call raises.
        """
        if not self._pusher_enabled():
            return
        self.metrics_pusher.histogram(
            "order_latency_s", float(duration_s), labels={"symbol": symbol}
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live trader supervisor (paper + live) with shakedown gate."
    )
    p.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated symbols, e.g. 'ETH/USDT,BTC/USDT'.",
    )
    p.add_argument(
        "--mode",
        choices=("paper", "live"),
        default="paper",
        help="Operator intent (live requires shakedown to be unlocked).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Tick interval in seconds.",
    )
    p.add_argument(
        "--bankroll",
        type=float,
        default=10_000.0,
        help="Starting equity in USD.",
    )
    p.add_argument(
        "--shakedown-state-path",
        type=str,
        default="./.shakedown.json",
        help="JSON file for rolling shakedown evidence.",
    )
    p.add_argument(
        "--shakedown-min-days",
        type=int,
        default=14,
        help="Minimum clean paper-trade days before live is unlocked.",
    )
    p.add_argument(
        "--risk-pct",
        type=float,
        default=0.005,
        help="Fraction of bankroll per trade (0.005 = 0.5%%).",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Model confidence floor below which trades are skipped.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Run a single iteration and exit (cron-friendly).",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "If set, save this run's outputs to a timestamped subdirectory "
            "under here (supervisor.log + summary.json or ticks.json). "
            "Recommended: --log-dir runs"
        ),
    )
    # Auto-pause + confidence-history wiring (Phase 16 ops polish). The
    # gate is OFF by default — operators must opt in. When --auto-pause-
    # enabled is set, daily_close evaluates the combined daily-loss +
    # confidence-shift gate and, if it trips, writes the marker file the
    # next tick observes. The three other flags expose the gate's tunables
    # and the rolling confidence-baseline window size.
    p.add_argument(
        "--auto-pause-enabled",
        action="store_true",
        help=(
            "Enable the combined daily-loss + confidence-shift auto-pause "
            "gate. When tripped, writes ~/.autopilot_auto_paused marker so "
            "the next tick halts."
        ),
    )
    p.add_argument(
        "--auto-pause-loss-pct",
        type=float,
        default=0.02,
        help=(
            "Auto-pause daily-loss threshold as a fraction of bankroll "
            "(0.02 = 2%%). Only used when --auto-pause-enabled."
        ),
    )
    p.add_argument(
        "--auto-pause-confidence-window",
        type=int,
        default=200,
        help=(
            "Rolling window size for the confidence-history baseline "
            "(per symbol). Default 200 ticks. Only used when "
            "--auto-pause-enabled."
        ),
    )
    p.add_argument(
        "--auto-pause-confidence-sigma",
        type=float,
        default=2.0,
        help=(
            "How many baseline std-deviations recent mean confidence must "
            "fall below for the gate's confidence condition to fire. "
            "Default 2.0 (a roughly 2.3%% tail event)."
        ),
    )
    return p.parse_args(argv)


def _setup_run_dir(
    log_dir: Optional[str],
    *,
    symbols: List[str],
    now_utc: Optional[datetime] = None,
) -> Optional[Path]:
    """If ``log_dir`` is set, create ``<log_dir>/<ts>_<symbols>/`` and attach
    a file handler to the root logger so every LOGGER call lands in
    ``supervisor.log``. Returns the run directory path or ``None``.
    """
    if not log_dir:
        return None
    now_utc = now_utc or datetime.now(timezone.utc)
    safe_symbols = ",".join(sym.replace("/", "-") for sym in symbols) or "run"
    ts = now_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = Path(log_dir).expanduser().resolve() / f"{ts}_{safe_symbols}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "supervisor.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    # Make sure the root level lets INFO through. main() already calls
    # basicConfig with level=INFO, but tests + non-CLI callers might not.
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    LOGGER.info("supervisor: saving run outputs to %s", run_dir)
    return run_dir


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    # Load .env from repo root and src/.env (matches src/config.py convention).
    # Do this BEFORE constructing the exchange / breakers / pusher so they see
    # the env vars the operator put in .env. Silent if python-dotenv is absent.
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        repo_root = Path(__file__).resolve().parent.parent
        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(repo_root / "src" / ".env", override=False)
    except Exception:  # noqa: BLE001 - .env loading is best-effort
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,  # ensure live streaming when output is piped/tee'd
        force=True,
    )
    # Force unbuffered stdout so each log record appears immediately when the
    # process is invoked under a pipe (the default 4KB block buffer otherwise
    # makes the supervisor look frozen for several seconds per tick).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    args = _parse_args(argv)
    raw_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    # Dedupe preserving first-seen order. Duplicates are usually a config typo
    # (eg. ``--symbols ETH/USD,ETH/USD``) and would otherwise spawn two
    # supervisor entries fighting over the same Redis position keys + the
    # same flock'ed shakedown file. Drop them with a warning per dropped
    # entry; if the deduped list is empty (every symbol was a dup of nothing),
    # exit 2.
    symbols: List[str] = []
    seen: set[str] = set()
    for sym in raw_symbols:
        if sym in seen:
            LOGGER.warning(
                "supervisor: dropping duplicate symbol %r from --symbols", sym
            )
            continue
        seen.add(sym)
        symbols.append(sym)
    if not symbols:
        print("error: --symbols must contain at least one entry", file=sys.stderr)
        return 2

    config = SupervisorConfig(
        symbols=symbols,
        tick_interval_s=args.interval,
        bankroll_usd=args.bankroll,
        mode=args.mode,
        shakedown_min_days=args.shakedown_min_days,
        shakedown_state_path=Path(args.shakedown_state_path),
        risk_pct_per_trade=args.risk_pct,
        min_confidence_to_trade=args.min_confidence,
    )

    # Optional Sentry init -- a no-op if SENTRY_DSN is unset.
    try:
        from observability import MetricsPusher, init_sentry

        init_sentry(environment=os.getenv("SENTRY_ENVIRONMENT", "dev"))
        metrics_pusher: Optional[Any] = MetricsPusher()
    except Exception as exc:  # noqa: BLE001 - never let observability crash boot
        LOGGER.warning("observability bootstrap failed: %s", exc)
        metrics_pusher = None

    # Wire the dependencies from env. Each constructor reads its own env vars.
    exchange = CoinbaseExchange()
    position_store = PositionStore()
    circuit_breakers = CircuitBreakerSet()
    notifier = Notifier()

    # Try the legacy transformer; fall back to neutral placeholder if anything
    # goes wrong (missing artifacts, torch import error, etc.). The supervisor
    # must never crash on model boot.
    predict_fn: Callable[[str, Ticker], Tuple[Literal["buy", "sell"], float]] = (
        _placeholder_predict
    )
    try:
        from predictor import build_default_predict_fn

        legacy = build_default_predict_fn(exchange)
        if legacy is not None:
            predict_fn = legacy
            LOGGER.info(
                "supervisor: using %s", type(legacy).__name__
            )
        else:
            LOGGER.warning(
                "supervisor: no legacy predictor available; using placeholder "
                "(every tick will be skipped_low_confidence)"
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "supervisor: predictor bootstrap failed (%s); using placeholder", exc
        )

    # Auto-pause + confidence-history wiring (opt-in via CLI flag). The
    # gate's daily-loss threshold + sigma come straight from argv; the
    # confidence-history window is the per-symbol rolling buffer length.
    auto_pause_gate: Optional[AutoPauseGate] = None
    confidence_history: Optional[ConfidenceHistory] = None
    if getattr(args, "auto_pause_enabled", False):
        try:
            auto_pause_gate = AutoPauseGate(
                loss_threshold_pct=float(args.auto_pause_loss_pct),
                z_threshold=float(args.auto_pause_confidence_sigma),
                recent_window=int(args.auto_pause_confidence_window),
            )
        except Exception as exc:  # noqa: BLE001 - never crash boot on knob errors
            LOGGER.warning(
                "auto-pause gate construction failed (%s); disabling", exc
            )
            auto_pause_gate = None
        # Wire the rolling confidence buffer through the same Redis client
        # the position store already uses (when available) so baseline reads
        # survive process restarts.
        try:
            redis_client = getattr(position_store, "_redis", None)
            confidence_history = ConfidenceHistory(
                redis_client=redis_client,
                window_size=int(args.auto_pause_confidence_window),
            )
        except Exception as exc:  # noqa: BLE001 - fall back to in-process buffer
            LOGGER.warning(
                "confidence-history wiring failed (%s); using in-process buffer",
                exc,
            )
            confidence_history = ConfidenceHistory(
                redis_client=None,
                window_size=int(args.auto_pause_confidence_window),
            )

    supervisor = Supervisor(
        config=config,
        exchange=exchange,
        position_store=position_store,
        circuit_breakers=circuit_breakers,
        notifier=notifier,
        model_predict_fn=predict_fn,
        metrics_pusher=metrics_pusher,
        auto_pause_gate=auto_pause_gate,
        confidence_history=confidence_history,
    )

    run_dir = _setup_run_dir(args.log_dir, symbols=symbols)

    if args.once:
        ticks = supervisor.run_once()
        ticks_payload = [t.model_dump(mode="json") for t in ticks]
        print(json.dumps(ticks_payload, indent=2))
        if run_dir is not None:
            (run_dir / "ticks.json").write_text(
                json.dumps(ticks_payload, indent=2), encoding="utf-8"
            )
            LOGGER.info("supervisor: wrote %s/ticks.json", run_dir)
        return 0

    summary = supervisor.run_loop()
    print(json.dumps(summary, indent=2))
    if run_dir is not None:
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        LOGGER.info("supervisor: wrote %s/summary.json", run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
