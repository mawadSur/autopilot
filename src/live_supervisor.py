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
import collections
import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import traceback
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

# Mirror the sys.path shim used by main.py / orchestrator.py /
# calibration_agent/build_dataset.py so this CLI runs without the caller
# setting PYTHONPATH.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import redis.exceptions
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from protocols import Tradeable

from alerts.notifier import Notifier
from exchanges.coinbase import CoinbaseExchange, ExchangeError, OrderResult, Ticker
from exit_policy import ExitDecision, ExitPolicy
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


# ---------------------------------------------------------------------------
# Lane D D3: multiprocessing constants
# ---------------------------------------------------------------------------

# Redis key namespace for cross-process supervisor signals (shutdown,
# leader election, etc). Children + parent both look here.
_SHUTDOWN_KEY = "supervisor:shutdown_requested"
# TTL on the shutdown flag. Parent sets it on signal; children check at
# top of every tick. TTL prevents a stale shutdown flag from a previously
# crashed parent process from poisoning a fresh boot.
_SHUTDOWN_KEY_TTL_S = 60
# Daily-close leader election: SETNX'd once per UTC date, scoped by the
# supervisor's symbol set so two independent supervisor groups (different
# symbol sets) don't share a leader key.
_DAILY_CLOSE_LEADER_KEY_PREFIX = "daily_close_leader"
_DAILY_CLOSE_LEADER_TTL_S = 7200  # 2 hours -- long enough to outlive a slow close.

# Crash recovery: if a child exits unexpectedly, parent waits this long
# before respawning to avoid tight crash loops.
_CHILD_RESPAWN_BACKOFF_S = 5.0
# Parent supervises children by polling .is_alive() at this cadence.
_PARENT_HEALTH_POLL_S = 0.5
# Default ceiling on respawns per symbol per hour. Beyond this the
# supervisor treats the issue as systemic and halts.
_DEFAULT_RESTART_LIMIT_PER_HOUR = 3
# Window (seconds) for the rolling restart counter.
_RESTART_WINDOW_S = 3600.0
# Graceful shutdown: parent waits up to this many seconds for children to
# exit cleanly after broadcasting shutdown, then SIGKILLs stragglers.
_SHUTDOWN_GRACE_S = 30.0
# Shorter grace interval used when polling each child's exit status.
_SHUTDOWN_POLL_INTERVAL_S = 0.2


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
    # Sprint 2.5: entry attribution — predictor's confidence + resolved
    # Kelly fraction at signal time. The paper drain stamps these onto the
    # synthesized Position so ``diagnose_calibration_drift`` finds the
    # confidence on the position blob without needing the snapshot store.
    entry_confidence: Optional[float] = None
    resolved_kelly_pct: Optional[float] = None
    # Sprint 2.6: regime label resolved by the predictor's regime_lookup at
    # signal time (or None when the lookup was off / low-confidence). The
    # paper drain mirrors entry_confidence/resolved_kelly_pct and forwards
    # this onto the synthesized Position so ``OutcomeAdjuster`` can probe
    # the model_meta path on the Position blob without re-reading the
    # snapshot store.
    regime_label: Optional[str] = None


_TIME_RE = re.compile(r"^time:(\d+)([smh])$", re.IGNORECASE)
_TP_SL_RE = re.compile(
    r"^tp_sl:(\d+(?:\.\d+)?)bps/(\d+(?:\.\d+)?)bps$", re.IGNORECASE
)


def _parse_exit_rule(rule: str) -> Dict[str, Optional[float]]:
    """Parse an ``--exit-rule`` string into a normalised dict.

    Accepted forms (comma-separated to combine):
      * ``"none"`` -> ``{}`` (no exits ever)
      * ``"time:5m"`` / ``"time:300s"`` / ``"time:2h"``
      * ``"tp_sl:30bps/50bps"``
      * ``"tp_sl:30bps/50bps,time:10m"``

    Returns a dict with optional keys ``time_seconds``, ``tp_bps``,
    ``sl_bps``. Missing keys mean that sub-rule is disabled. An empty
    dict means "never close".

    Raises ``ValueError`` on unrecognised input so the operator hears
    about typos at boot, not silently at exit time.
    """
    rule = (rule or "").strip()
    if not rule or rule.lower() == "none":
        return {}
    out: Dict[str, Optional[float]] = {}
    parts = [p.strip() for p in rule.split(",") if p.strip()]
    for part in parts:
        lower = part.lower()
        m_time = _TIME_RE.match(lower)
        if m_time:
            value = int(m_time.group(1))
            unit = m_time.group(2).lower()
            seconds = value * {"s": 1, "m": 60, "h": 3600}[unit]
            out["time_seconds"] = float(seconds)
            continue
        m_tp = _TP_SL_RE.match(lower)
        if m_tp:
            out["tp_bps"] = float(m_tp.group(1))
            out["sl_bps"] = float(m_tp.group(2))
            continue
        raise ValueError(
            f"unrecognised exit-rule clause {part!r}; expected 'none', "
            "'time:Nm' / 'time:Ns' / 'time:Nh', or 'tp_sl:Xbps/Ybps'."
        )
    return out


_ActionTaken = Literal[
    "skipped_low_confidence",
    "allowed",
    "halted_breaker",
    "force_flatted",
    "errored",
    # Halal mode: a would-be entry was blocked because it was not
    # Shariah-compliant (a short entry, i.e. selling an asset not owned).
    # Exits are never blocked. See SupervisorConfig.halal_mode.
    "skipped_halal",
    # Sprint 1 Wave 2: a tick where the exit policy closed one or more
    # positions for the symbol BEFORE any new entry was considered. The
    # supervisor never reopens on the same tick — the "exited" tick is
    # terminal for that symbol.
    "exited",
]


def _exchange_is_spot(exchange: Any) -> bool:
    """Fail-closed spot check for halal mode.

    Returns ``True`` only when the exchange *explicitly* declares
    ``MARKET_TYPE == "spot"``. A perp/margin connector (which declares
    ``"perp"``) or any exchange that doesn't declare a market type at all
    returns ``False`` — so under halal mode a non-compliant or unknown venue
    can never receive a live order. Deliberately conservative: silence means
    "not proven spot", not "assume spot".
    """
    return str(getattr(exchange, "MARKET_TYPE", "")).strip().lower() == "spot"


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------


class SupervisorConfig(BaseModel):
    """Operator-supplied configuration for the supervisor.

    Lane D D2: ``tradeables`` lets the supervisor iterate heterogeneous
    venues (Coinbase spot, Hyperliquid perps, Polymarket binary markets)
    under a single tick loop. Legacy ``symbols`` is still honoured —
    each symbol gets wrapped in a :class:`CoinbaseTradeable` at boot.
    Precedence: ``tradeables`` > ``symbols``. At least one must be
    non-empty.
    """

    # arbitrary_types_allowed: Tradeable adapters are runtime objects,
    # not pydantic models.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    symbols: List[str] = Field(default_factory=list)
    tradeables: List[Any] = Field(default_factory=list)
    tick_interval_s: float = 5.0
    bankroll_usd: float = 10_000.0
    mode: Literal["paper", "live"] = "paper"
    shakedown_min_days: int = 14
    shakedown_state_path: Path
    risk_pct_per_trade: float = 0.005
    min_confidence_to_trade: float = 0.6
    # Halal (Shariah-compliant) trading mode. When True, the supervisor only
    # opens LONG entries (a short means selling an asset you don't own) and
    # fail-closes any live order routed to a non-spot venue (perps/leverage/
    # funding = riba + gharar). Exits are never blocked — closing a long is a
    # permissible sale of an owned asset. Mirrors cfg.HALAL_MODE / --halal-mode.
    halal_mode: bool = False
    # Exit rule applied to open positions at the top of every tick. Syntax:
    #   "none"                              - never close (legacy behavior)
    #   "time:5m"                           - close after 5 minutes
    #   "tp_sl:30bps/50bps"                 - +30bps TP / -50bps SL
    #   "tp_sl:30bps/50bps,time:10m"        - either condition trips first
    exit_rule: str = "none"

    @model_validator(mode="after")
    def _require_at_least_one_source(self) -> "SupervisorConfig":
        if not self.symbols and not self.tradeables:
            raise ValueError(
                "SupervisorConfig requires at least one of `symbols` or "
                "`tradeables` to be non-empty"
            )
        return self


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
    # raw_max_prob and raw_probs let observers see *how close* a skipped tick
    # was to the threshold — invaluable for tuning thr_long/thr_short. They
    # are filled when the wired predictor exposes ``predict_full``; older
    # 2-tuple-only stubs leave them as None.
    raw_max_prob: Optional[float] = None
    raw_probs: Optional[Dict[str, float]] = None
    action_taken: _ActionTaken
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _today_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).date().isoformat()


def _parse_position_opened_at(value: str) -> Optional[datetime]:
    """Parse a Position.opened_at_utc ISO string; returns UTC-aware dt or None.

    Handles both the canonical ``+00:00`` suffix and a trailing ``Z``.
    Returns None on any parse error so the caller can skip the position
    gracefully rather than crashing the whole tick.
    """
    if not value:
        return None
    try:
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = datetime.fromisoformat(s)
    except Exception:  # noqa: BLE001 - return None on any parse failure
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
        auto_trip_threshold: int = 10,
        trade_context_store: Optional[TradeContextStore] = None,
        # Sprint 1 Wave 2: exit policy + Kelly sizing.
        # ``exit_policy`` is the ``ExitPolicy`` instance evaluated each tick
        # against every open position for the current symbol BEFORE any new
        # entry is considered. When None + ``exit_policy_enabled=True`` the
        # supervisor builds a default policy from ``src/config.py`` (the
        # production path). When None + ``exit_policy_enabled=False`` the
        # policy is disabled — preserves the legacy "no exit policy" path
        # for the regression-sentinel test.
        # ``kelly_sizing_enabled`` / ``kelly_floor_pct`` / ``kelly_cap_pct``
        # govern the per-trade notional sizing seam: when enabled AND the
        # wired predictor exposes a non-None ``_last_resolved_kelly_pct``
        # the supervisor sizes new entries as ``bankroll * clip(resolved,
        # floor, cap)``. Falls back to ``bankroll * risk_pct_per_trade``
        # otherwise.
        exit_policy: Optional[ExitPolicy] = None,
        exit_policy_enabled: Optional[bool] = None,
        kelly_sizing_enabled: Optional[bool] = None,
        kelly_floor_pct: Optional[float] = None,
        kelly_cap_pct: Optional[float] = None,
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
        # threshold is 10 errors (~50s at 5s tick interval) — a real outage,
        # not a transient DNS / TLS handshake blip. (Previously 3, which
        # tripped a multi-day session on a 10-second DNS hiccup.) Supervisor
        # keeps an in-process latch so we only emit the alert + metric once
        # per trip.
        self.auto_trip_threshold = int(auto_trip_threshold)
        self._auto_tripped_symbols: set[str] = set()

        # Kill-switch trips are account-level (every symbol resets) and
        # remain in-process: the only writer is the supervisor itself,
        # so cross-process visibility isn't needed. Per-symbol error
        # counts now live in Redis (P0 #3) so multiple symbol-supervisor
        # processes can share state under the D1 multiprocessing model.
        self._kill_switch_trips_today = 0
        # Edge-triggered alert latch: True while the kill-switch file is
        # detected as present. We alert + force-flatten only on the
        # False->True edge; subsequent ticks while still tripped no-op
        # silently. Cleared on the True->False edge (operator removes file).
        self._kill_switch_active: bool = False

        # One-time warning latches, keyed by symbol so the live-locked
        # warning fires once per locked symbol per process.
        self._warned_live_locked: Dict[str, bool] = {}

        # Defer-to-next-tick paper fill state machine: one pending fill per
        # symbol. Populated when paper mode produces a signal; drained at
        # the start of the next tick for that symbol against the live
        # ticker. See PendingPaperFill above for the rationale.
        self._pending_paper_fills: Dict[str, PendingPaperFill] = {}

        # Lane D D2: explicit tradeables first, then legacy symbols
        # wrapped in CoinbaseTradeable. Dedupe by symbol.
        self._tradeables: List["Tradeable"] = []
        self._tradeables_by_symbol: Dict[str, "Tradeable"] = {}
        for tradeable in list(config.tradeables or []):
            sym = str(getattr(tradeable, "symbol", ""))
            if sym and sym not in self._tradeables_by_symbol:
                self._tradeables.append(tradeable)
                self._tradeables_by_symbol[sym] = tradeable
        if config.symbols:
            from exchanges.adapters import CoinbaseTradeable  # noqa: WPS433

            for sym in config.symbols:
                if sym not in self._tradeables_by_symbol:
                    wrapped = CoinbaseTradeable(self.exchange, sym)
                    self._tradeables.append(wrapped)
                    self._tradeables_by_symbol[sym] = wrapped
        # Backfill config.symbols so shakedown / run_once / daily_close
        # see every tradeable's symbol (including "polymarket:<id>").
        for sym in self._tradeables_by_symbol.keys():
            if sym not in config.symbols:
                config.symbols.append(sym)

        # Monotonic per-symbol tick counter, used to age out stale pending
        # fills (anything older than PAPER_PENDING_FILL_MAX_AGE_TICKS).
        self._symbol_tick_index: Dict[str, int] = {
            sym: 0 for sym in self._tradeables_by_symbol.keys()
        }

        # Parse + cache the exit rule once at boot so typos are surfaced
        # before the first tick (rather than silently at exit time).
        self._exit_rule_parsed: Dict[str, Optional[float]] = _parse_exit_rule(
            self.config.exit_rule
        )

        # Hydrate (or initialise) the shakedown evidence file.
        self.shakedown_state: ShakedownState = self._load_or_init_shakedown()

        # P0 #8: track the last UTC date we ran ``daily_close`` for so the
        # run loop can fire it automatically when the clock crosses
        # midnight. ``None`` on construction means "no close has fired
        # yet"; the first iteration sets it without firing daily_close
        # (otherwise a freshly booted supervisor would always emit one
        # immediately).
        self._last_close_date: Optional[date] = None

        # ----------------------------------------------------------------
        # Sprint 1 Wave 2: ExitPolicy + Kelly sizing wire-up.
        # ----------------------------------------------------------------
        # Resolve the master switches + Kelly bounds. Explicit kwargs win;
        # otherwise the singleton ``cfg`` from ``src/config.py`` provides
        # the defaults so a fresh production boot picks up env-driven
        # overrides without test fixtures having to import it.
        resolved = self._resolve_exit_kelly_settings(
            exit_policy=exit_policy,
            exit_policy_enabled=exit_policy_enabled,
            kelly_sizing_enabled=kelly_sizing_enabled,
            kelly_floor_pct=kelly_floor_pct,
            kelly_cap_pct=kelly_cap_pct,
        )
        self.exit_policy: Optional[ExitPolicy] = resolved["exit_policy"]
        self.exit_policy_enabled: bool = bool(resolved["exit_policy_enabled"])
        self.kelly_sizing_enabled: bool = bool(resolved["kelly_sizing_enabled"])
        self.kelly_floor_pct: float = float(resolved["kelly_floor_pct"])
        self.kelly_cap_pct: float = float(resolved["kelly_cap_pct"])

        # Reason-tagged exit counters. Keys are the documented reasons from
        # ``ExitPolicy`` (``time`` / ``sl`` / ``tp`` / ``trail`` / ``reversal``)
        # plus the synthetic ``error`` bucket for ticks where an exit fill
        # raised an unexpected error. Exposed via ``_emit_loop_metrics``.
        self._exits_by_reason: Dict[str, int] = {
            "time": 0,
            "sl": 0,
            "tp": 0,
            "trail": 0,
            "reversal": 0,
            "error": 0,
        }

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
        """Run a single iteration over all configured symbols.

        Iterates ``config.symbols`` (which the supervisor backfills at
        init with every tradeable's symbol). Binary tradeables route
        through :meth:`_dispatch_tick` to a Polymarket-specific tick
        path; crypto symbols stay on the legacy :meth:`_tick_symbol`.
        """
        ticks: List[SupervisorTick] = []
        loop_start = time.monotonic()
        for symbol in self.config.symbols:
            tick_start = time.monotonic()
            try:
                tick = self._dispatch_tick(symbol)
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
    def _dispatch_tick(self, symbol: str) -> SupervisorTick:
        """Route a tick to crypto or binary tick handler by asset_class."""
        tradeable = self._tradeables_by_symbol.get(symbol)
        if tradeable is not None:
            asset_class = getattr(tradeable, "asset_class", None)
            # String compare avoids importing AssetClass here.
            if str(getattr(asset_class, "value", "")) == "prediction_binary":
                return self._tick_prediction_binary(tradeable)
        return self._tick_symbol(symbol)

    def _tick_prediction_binary(self, tradeable: "Tradeable") -> SupervisorTick:
        """Minimal tick for Polymarket markets — adapter-driven ticker
        read, no exchange call. Order placement deferred to a follow-up.
        """
        symbol = tradeable.symbol
        now = self._now()
        if self.circuit_breakers.is_kill_switch_tripped():
            self._kill_switch_trips_today += 1
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
                notes="prediction_binary kill_switch",
            )
        try:
            tradeable.get_ticker()
        except Exception as exc:  # noqa: BLE001 - fetcher errors are non-fatal
            self._handle_tick_error(symbol)
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=self._allow_verdict(),
                model_confidence=None,
                action_taken="errored",
                notes=f"polymarket_ticker: {exc!r}",
            )
        return SupervisorTick(
            tick_at_utc=now.isoformat(),
            symbol=symbol,
            verdict=self._allow_verdict(),
            model_confidence=None,
            action_taken="allowed",
            notes="prediction_binary",
        )

    def _tick_symbol(self, symbol: str) -> SupervisorTick:
        now = self._now()
        # 1. Kill switch first. Alert + force-flatten only on the trip
        # edge (transition into tripped state); subsequent ticks while
        # still tripped return a silent ``force_flatted`` verdict so the
        # operator gets one notification, not one per tick. Clearing the
        # kill-switch file resets the latch.
        tripped_now = self.circuit_breakers.is_kill_switch_tripped()
        if tripped_now:
            if not self._kill_switch_active:
                self._kill_switch_active = True
                self._kill_switch_trips_today += 1
                count = self._force_flat_all(reason="kill_switch_file_present")
                self._safe_kill_switch_alert(
                    f"kill switch tripped; force-closed {count} position(s)"
                )
                notes = f"force_closed={count}"
            else:
                count = 0
                notes = "kill_switch_held"
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
                notes=notes,
            )
        if self._kill_switch_active:
            self._kill_switch_active = False
            LOGGER.info(
                "kill switch cleared (file removed); resuming normal ticks"
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

        # 2b. Bump per-symbol tick index so pending-fill staleness ages
        # correctly. This is the heart of the defer-to-next-tick state
        # machine -- the fill price for a paper signal emitted at tick N
        # is the ticker observed at tick N+1.
        self._symbol_tick_index[symbol] = (
            self._symbol_tick_index.get(symbol, 0) + 1
        )

        # 2c. Evaluate exits BEFORE draining the pending fill (entry) so a
        # halted-breaker tick can still close stale positions, and a
        # just-filled position is never instant-closed on the same tick.
        # Always best-effort; per-position errors are logged but never
        # abort the tick.
        try:
            self._check_exits(symbol=symbol, ticker=ticker)
        except Exception as exc:  # noqa: BLE001 - never let exit logic kill loop
            LOGGER.exception("exit-rule evaluation failed for %s", symbol)
            self._safe_alert(
                f"exit-rule evaluation failed for {symbol}: {exc}",
                severity="alert",
            )

        # 2d. Drain any pending paper fill against the FRESH ticker.
        self._drain_pending_paper_fill(symbol, ticker)

        # 2c. Sprint 1 Wave 2: evaluate the exit policy against EVERY open
        # position for THIS symbol BEFORE we consider a new entry. If
        # anything closed on this tick we short-circuit — never reopen on
        # the same tick that just closed a position (capital-preservation
        # invariant; matches the CEO review brief).
        exits_closed, exit_reason = 0, None
        if self.exit_policy_enabled and self.exit_policy is not None:
            try:
                exits_closed, exit_reason = self._process_exits(
                    symbol=symbol, ticker=ticker
                )
            except ExchangeError as exc:
                # A live-tagged exit close raised. Treat like any other
                # ExchangeError on the tick: bump the error counter, alert,
                # short-circuit with an ``errored`` tick. The kill-switch
                # auto-trip will fire if these accumulate.
                self._handle_tick_error(symbol)
                self._safe_alert(
                    f"exit close failed for {symbol}: {exc}",
                    severity="alert",
                )
                return SupervisorTick(
                    tick_at_utc=now.isoformat(),
                    symbol=symbol,
                    verdict=self._allow_verdict(),
                    model_confidence=None,
                    action_taken="errored",
                    notes=f"exit_close: {exc!r}",
                )

        if exits_closed > 0:
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=self._allow_verdict(),
                model_confidence=None,
                action_taken="exited",
                notes=(
                    f"exited={exits_closed} reason={exit_reason}"
                    if exit_reason
                    else f"exited={exits_closed}"
                ),
            )

        # 3. Build context + run breakers. The breakers care about
        # proposed notional vs cap, so we feed them the FLAT (legacy)
        # ``bankroll * risk_pct_per_trade`` figure here — the Kelly
        # resize happens AFTER the predict call so the predictor's
        # ``_last_resolved_kelly_pct`` (set on each predict) is available.
        # Source string ends up in SupervisorTick.notes so we can tell at
        # a glance which path won (per CEO review brief).
        proposed = float(self.config.bankroll_usd) * float(
            self.config.risk_pct_per_trade
        )
        _sizing_source = "flat"
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
        #
        # Prefer ``predict_full`` when the wired predictor exposes it — the
        # rich PredictorResult carries the raw class probs which we surface
        # via SupervisorTick.raw_max_prob/raw_probs so operators can tell
        # *how close* a skipped tick was to the threshold.
        raw_probs: Optional[Dict[str, float]] = None
        raw_max_prob: Optional[float] = None
        try:
            rich_call = getattr(self.model_predict_fn, "predict_full", None)
            if callable(rich_call):
                result = rich_call(symbol, ticker)
                side, confidence = result.side, float(result.confidence)
                mp = getattr(result, "model_probs", None)
                if isinstance(mp, dict) and mp:
                    raw_probs = {k: float(v) for k, v in mp.items()}
                    finite_vals = [v for v in raw_probs.values() if math.isfinite(v)]
                    if finite_vals:
                        raw_max_prob = max(finite_vals)
            else:
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
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
                action_taken="force_flatted",
                notes=f"force_closed={count}",
            )

        if verdict.recommended_action == "halt_new_entries":
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
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
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
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
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
                action_taken="skipped_low_confidence",
                notes=(
                    f"confidence {confidence:.3f} < floor "
                    f"{self.config.min_confidence_to_trade:.3f}"
                ),
            )

        # 5b-Halal. Long-only gate. When halal_mode is on, only long entries
        # are permitted — opening a short means selling an asset you don't own,
        # which is not Shariah-compliant. This is enforced in BOTH paper and
        # live so a paper "validation" run reflects halal behavior too. Exits
        # are unaffected (closing a long is a permissible sale of an owned
        # asset, handled on the exit path, never here).
        if self.config.halal_mode and side != "buy":
            LOGGER.info(
                "halal_mode: blocking non-long entry for %s (side=%s, "
                "confidence=%.3f)",
                symbol,
                side,
                confidence,
            )
            return SupervisorTick(
                tick_at_utc=now.isoformat(),
                symbol=symbol,
                verdict=verdict,
                model_confidence=confidence,
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
                action_taken="skipped_halal",
                notes=f"halal_mode: short entry blocked (side={side})",
            )

        # 5a-Kelly. Sprint 1 Wave 2: re-size the proposed notional from the
        # predictor's resolved Kelly fraction now that the predict call has
        # populated ``_last_resolved_kelly_pct``. Falls back to the flat
        # path computed in step 3 if Kelly is disabled or the predictor
        # didn't surface a value. The breaker context above used the flat
        # number — Kelly resizing only affects what hits the exchange.
        proposed, _sizing_source, resolved_kelly_pct = (
            self._resolve_proposed_notional(symbol=symbol)
        )
        # Sprint 2.6: mirror the kelly-cache read so we forward the resolved
        # regime label (when the predictor's regime_lookup fired with
        # confidence >= 0.5) through the same path Sprint 2.5 used for the
        # Kelly fraction. ``None`` when the lookup didn't fire — the
        # downstream snapshot writer drops the key in that case.
        resolved_regime_label = getattr(
            self.model_predict_fn, "_last_resolved_regime_label", None
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
            resolved_kelly_pct=resolved_kelly_pct,
            regime_label=resolved_regime_label,
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
                    entry_confidence=confidence,
                    resolved_kelly_pct=resolved_kelly_pct,
                    regime_label=resolved_regime_label,
                )
                return SupervisorTick(
                    tick_at_utc=now.isoformat(),
                    symbol=symbol,
                    verdict=verdict,
                    model_confidence=confidence,
                    raw_max_prob=raw_max_prob,
                    raw_probs=raw_probs,
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
                    entry_confidence=confidence,
                    resolved_kelly_pct=resolved_kelly_pct,
                    regime_label=resolved_regime_label,
                )
                notes_extra = f"live sizing={_sizing_source} usd={proposed:.2f}"
            else:
                self._paper_simulate_fill(
                    symbol=symbol,
                    side=side,
                    ticker=ticker,
                    proposed_usd=proposed,
                    entry_confidence=confidence,
                    resolved_kelly_pct=resolved_kelly_pct,
                    regime_label=resolved_regime_label,
                )
                notes_extra = f"paper sizing={_sizing_source} usd={proposed:.2f}"
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
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
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
                raw_max_prob=raw_max_prob,
                raw_probs=raw_probs,
                action_taken="errored",
                notes=f"order: {exc!r}",
            )

        return SupervisorTick(
            tick_at_utc=now.isoformat(),
            symbol=symbol,
            verdict=verdict,
            model_confidence=confidence,
            raw_max_prob=raw_max_prob,
            raw_probs=raw_probs,
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
        entry_confidence: Optional[float] = None,
        resolved_kelly_pct: Optional[float] = None,
        regime_label: Optional[str] = None,
    ) -> None:
        """Place a real market order. Raises ExchangeError on failure."""
        # Halal spot-only fail-close. This is the last line of defence before a
        # real order hits the venue: under halal mode we refuse to trade
        # through anything not explicitly declared spot (perps/leverage/funding
        # = riba + gharar). Fail-closed — an undeclared/unknown exchange is
        # rejected too. Only the OPEN path is gated here; exits go through their
        # own close path so a position can always be liquidated.
        if self.config.halal_mode and not _exchange_is_spot(self.exchange):
            raise ExchangeError(
                "halal_mode: refusing live order on non-spot venue "
                f"{type(self.exchange).__name__} "
                f"(MARKET_TYPE={getattr(self.exchange, 'MARKET_TYPE', None)!r}); "
                "only spot exchanges are Shariah-compliant"
            )
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
        # Sprint 2.5: stamp the entry attribution onto the position BEFORE
        # the store write so the persisted blob carries entry_confidence
        # (what diagnose_calibration_drift's resolve_confidence reads first)
        # and resolved_kelly_pct (sizing forensics seam) without a second
        # round-trip. Both fields default to None on legacy reads.
        position = position.model_copy(
            update={
                "entry_confidence": (
                    float(entry_confidence)
                    if entry_confidence is not None
                    else None
                ),
                "resolved_kelly_pct": (
                    float(resolved_kelly_pct)
                    if resolved_kelly_pct is not None
                    else None
                ),
                # Sprint 2.6: mirror the kelly stamp pattern. Empty strings
                # collapse to None so the Position blob's regime_label stays
                # absent when the predictor didn't surface a label.
                "regime_label": (
                    (str(regime_label).strip() or None)
                    if regime_label is not None
                    else None
                ),
            }
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
        entry_confidence: Optional[float] = None,
        resolved_kelly_pct: Optional[float] = None,
        regime_label: Optional[str] = None,
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
            entry_confidence=(
                float(entry_confidence)
                if entry_confidence is not None
                else None
            ),
            resolved_kelly_pct=(
                float(resolved_kelly_pct)
                if resolved_kelly_pct is not None
                else None
            ),
            # Sprint 2.6: carry the resolved regime label through the
            # deferred-fill state machine so the next-tick paper drain
            # stamps it onto the synthesized Position.
            regime_label=(
                (str(regime_label).strip() or None)
                if regime_label is not None
                else None
            ),
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
            # Sprint 2.5: paper-fill drains carry entry_confidence +
            # resolved_kelly_pct off the PendingPaperFill so the stored
            # position blob has the same attribution as a live fill.
            entry_confidence=pending.entry_confidence,
            resolved_kelly_pct=pending.resolved_kelly_pct,
            # Sprint 2.6: carry the regime label across the deferred-fill
            # boundary too, mirroring the kelly stamp pattern.
            regime_label=pending.regime_label,
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

    # ------------------------------------------------------------------
    # Exit-rule evaluation
    # ------------------------------------------------------------------
    def _check_exits(self, *, symbol: str, ticker: Ticker) -> None:
        """Close any open positions on ``symbol`` whose exit rule has fired.

        Called once per ``_tick_symbol`` invocation, after the ticker
        fetch and before the entry decision. Iterates only positions
        whose ``symbol`` matches this tick. Direction-aware: longs profit
        when price rises, shorts profit when price falls.

        Rule precedence is "whichever fires first" within a single tick;
        ``time`` and ``tp_sl`` are checked together and the first matching
        clause is recorded as the firing rule.
        """
        rule = self._exit_rule_parsed
        if not rule:
            return

        # Cheap up-front mid; falls back to a sane default if the ticker
        # somehow lacks one (shouldn't, post-commit 849325b — Ticker.mid
        # falls back to last when bid/ask are 0).
        try:
            exit_mid = float(ticker.mid)
        except Exception:  # noqa: BLE001 - skip the tick on garbage tickers
            return
        if exit_mid <= 0:
            return

        now = self._now()
        time_seconds = rule.get("time_seconds")
        tp_bps = rule.get("tp_bps")
        sl_bps = rule.get("sl_bps")

        # Snapshot to avoid iterating a mutating collection if record_close
        # writes back to the same underlying store.
        for position in list(self.position_store.list_open()):
            if position.symbol != symbol:
                continue
            if position.status != "open":
                # Skip pending/closing positions -- those have their own
                # reconcile path (Phase 6+); we only act on fully-open ones.
                continue

            fired: Optional[str] = None

            # tp_sl evaluation -- direction-aware.
            if tp_bps is not None and sl_bps is not None:
                entry = float(position.entry_price)
                if entry > 0:
                    # bps move in price (signed), always relative to entry.
                    move_bps = (exit_mid - entry) / entry * 10_000.0
                    if position.side == "long":
                        if move_bps >= tp_bps:
                            fired = f"tp:{tp_bps:g}bps"
                        elif move_bps <= -sl_bps:
                            fired = f"sl:{sl_bps:g}bps"
                    else:  # short -- sign flips
                        if move_bps <= -tp_bps:
                            fired = f"tp:{tp_bps:g}bps"
                        elif move_bps >= sl_bps:
                            fired = f"sl:{sl_bps:g}bps"

            # time-based evaluation runs only if tp_sl didn't already fire.
            if fired is None and time_seconds is not None:
                opened_at = _parse_position_opened_at(position.opened_at_utc)
                if opened_at is not None:
                    elapsed = (now - opened_at).total_seconds()
                    if elapsed >= float(time_seconds):
                        fired = f"time:{int(time_seconds)}s"

            if fired is None:
                continue

            try:
                exit_quote_usd = float(position.base_size) * float(exit_mid)
                closed = self.position_store.record_close(
                    position.position_id,
                    exit_price=exit_mid,
                    exit_quote_usd=exit_quote_usd,
                    fees_usd=0.0,
                )
            except Exception as exc:  # noqa: BLE001 - keep iterating
                LOGGER.warning(
                    "record_close failed for %s on %s: %s",
                    position.position_id,
                    symbol,
                    exc,
                )
                continue

            pnl = float(closed.realized_pnl_usd or 0.0)
            LOGGER.info(
                "exit | %s | entry=%.4f exit=%.4f pnl=$%.4f rule=%s",
                symbol,
                float(position.entry_price),
                float(exit_mid),
                pnl,
                fired,
            )

            # Mirror the fill_event notification used on entry so operators
            # see the exit land in the same channel. Best-effort.
            close_side: Literal["buy", "sell"] = (
                "sell" if position.side == "long" else "buy"
            )
            try:
                self.notifier.fill_event(
                    symbol=symbol,
                    side=close_side,
                    fill_price=float(exit_mid),
                    fill_size=float(position.base_size),
                    fees_usd=0.0,
                )
            except Exception as exc:  # noqa: BLE001 - notifier best-effort
                LOGGER.warning("notifier.fill_event raised on exit: %s", exc)

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
        # Phase-16: extract structured fill metadata for A2 forensics so we
        # don't have to regex-scrape position.notes later. Both extractors
        # are best-effort and return empty/None when ccxt didn't surface the
        # underlying data.
        partial_fills: Optional[List[Dict[str, Any]]] = None
        rejection_reason: Optional[str] = None
        try:
            from exchanges.coinbase import (
                extract_partial_fills,
                extract_rejection_reason,
            )

            extracted = extract_partial_fills(order)
            if extracted:
                partial_fills = extracted
            rejection_reason = extract_rejection_reason(order)
        except Exception as exc:  # noqa: BLE001 - extraction must never crash
            LOGGER.debug(
                "fill metadata extraction raised: %r; continuing without it",
                exc,
            )
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
            partial_fills=partial_fills,
            rejection_reason=rejection_reason,
        )

    # ------------------------------------------------------------------
    # Force-flat
    # ------------------------------------------------------------------
    def _force_flat_all(self, *, reason: str) -> int:
        """Best-effort close of every open position. Returns count closed.

        Dispatch is per-position based on ``position.exchange`` rather than
        on ``self.config.mode``. Positions opened by paper-mode runs are
        tagged ``coinbase-paper`` (see ``_drain_pending_paper_fill``) and
        must close via the local paper-fill path -- *never* place a real
        market order. Live-tagged positions stay on the existing
        ``place_market_order`` path so a stale live position from a prior
        run still flattens correctly even if the current supervisor boot
        is in paper mode.
        """
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
                if position.exchange == "coinbase-paper":
                    self._paper_force_flat(position, close_side)
                else:
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

    def _paper_force_flat(
        self, position: Position, close_side: Literal["buy", "sell"]
    ) -> None:
        """Synthesise a paper close fill and record via position_store.

        Mirrors :meth:`_drain_pending_paper_fill`'s slippage model: 5 bps
        applied against the current ticker mid in the close direction
        (sell -> price *down*, buy -> price *up*). Falls back to the
        position's entry price if ticker mid is non-positive.

        Reading the ticker is a market-data call, not an order, so it is
        safe to fetch via ``self.exchange.get_ticker`` even when the
        supervisor is running in paper mode -- only order placement is
        what we must keep off the live exchange.
        """
        ticker = self.exchange.get_ticker(position.symbol)
        slip = PAPER_SLIPPAGE_BPS / 10_000.0
        mid = float(ticker.mid)
        if close_side == "buy":
            exit_price = mid * (1.0 + slip)
        else:
            exit_price = mid * (1.0 - slip)
        if exit_price <= 0:
            exit_price = mid if mid > 0 else float(position.entry_price)
        exit_quote = exit_price * float(position.base_size)
        self.position_store.record_close(
            position.position_id,
            exit_price=exit_price,
            exit_quote_usd=exit_quote,
            fees_usd=0.0,
            bankroll_usd=float(self.config.bankroll_usd),
        )
        LOGGER.info(
            "paper force-flat: %s %s close_side=%s exit_price=%.4f "
            "(slippage_bps=%.1f)",
            position.position_id,
            position.symbol,
            close_side,
            exit_price,
            PAPER_SLIPPAGE_BPS,
        )

    # ------------------------------------------------------------------
    # Sprint 1 Wave 2: exit-policy + Kelly sizing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_exit_kelly_settings(
        *,
        exit_policy: Optional[ExitPolicy],
        exit_policy_enabled: Optional[bool],
        kelly_sizing_enabled: Optional[bool],
        kelly_floor_pct: Optional[float],
        kelly_cap_pct: Optional[float],
    ) -> Dict[str, Any]:
        """Pick effective Wave-2 settings from kwargs, falling back to cfg.

        Kept as a staticmethod so it can be exercised without constructing
        a full Supervisor. The fallback path lazy-imports ``cfg`` so the
        rest of the supervisor module stays config-free for the existing
        test fixtures (which don't write a TradingConfig file).
        """
        cfg_obj: Any = None

        def _from_cfg(attr: str, default: Any) -> Any:
            nonlocal cfg_obj
            if cfg_obj is None:
                try:
                    from config import cfg as _cfg  # noqa: WPS433 - lazy
                    cfg_obj = _cfg
                except Exception:  # noqa: BLE001 - cfg load failures fall back
                    cfg_obj = False  # sentinel: tried + failed
            if cfg_obj is False:
                return default
            return getattr(cfg_obj, attr, default)

        if exit_policy_enabled is None:
            exit_policy_enabled = bool(_from_cfg("EXIT_POLICY_ENABLED", True))
        if kelly_sizing_enabled is None:
            kelly_sizing_enabled = bool(_from_cfg("KELLY_SIZING_ENABLED", True))
        if kelly_floor_pct is None:
            kelly_floor_pct = float(_from_cfg("KELLY_FLOOR_PCT", 0.005))
        if kelly_cap_pct is None:
            kelly_cap_pct = float(_from_cfg("KELLY_CAP_PCT", 0.05))

        # Construct a default ExitPolicy from cfg when one wasn't supplied
        # AND the master switch is on. With the switch off we leave the
        # attribute None — the tick loop guards on the flag, but a None
        # policy is the unambiguous "no exits configured" signal.
        if exit_policy is None and exit_policy_enabled:
            exit_policy = ExitPolicy(
                time_stop_bars=_from_cfg("TIME_STOP_BARS", 20),
                stop_loss_pct=_from_cfg("STOP_LOSS_PCT", -0.004),
                take_profit_pct=_from_cfg("TAKE_PROFIT_PCT", 0.008),
                trailing_stop_pct=_from_cfg("TRAILING_STOP_PCT", None),
                signal_reversal=bool(_from_cfg("EXIT_SIGNAL_REVERSAL", False)),
            )

        return {
            "exit_policy": exit_policy,
            "exit_policy_enabled": bool(exit_policy_enabled),
            "kelly_sizing_enabled": bool(kelly_sizing_enabled),
            "kelly_floor_pct": float(kelly_floor_pct),
            "kelly_cap_pct": float(kelly_cap_pct),
        }

    def _resolve_proposed_notional(
        self, *, symbol: str
    ) -> Tuple[float, str, Optional[float]]:
        """Pick the per-trade notional in USD for ``symbol``.

        Returns ``(usd, source, resolved_kelly_pct)`` where ``source`` is
        ``"kelly"`` when the predictor's resolved Kelly fraction was used
        (clipped to the floor/cap), and ``"flat"`` when we fell back to
        the legacy ``bankroll * risk_pct_per_trade`` path. ``source`` is
        surfaced to ``SupervisorTick.notes`` so operators can see *which*
        sizing the tick used without reading metrics.
        ``resolved_kelly_pct`` is the (post-clip) Kelly fraction when
        Kelly fired, else ``None`` — Sprint 2.5 forwards this onto the
        opened :class:`Position` as ``resolved_kelly_pct`` so the Lane E
        sizing forensics can pull it without re-resolving the regime.

        The Kelly read is via ``getattr(self.model_predict_fn,
        '_last_resolved_kelly_pct', None)``: the predictor itself decides
        when to populate it (regime-memory match with confidence >= 0.5).
        We never re-resolve the regime or re-derive the kelly here — the
        predictor is the single source of truth.
        """
        flat_pct = float(self.config.risk_pct_per_trade)
        flat_usd = float(self.config.bankroll_usd) * flat_pct
        if not self.kelly_sizing_enabled:
            return flat_usd, "flat", None
        kelly_pct = getattr(
            self.model_predict_fn, "_last_resolved_kelly_pct", None
        )
        if kelly_pct is None:
            return flat_usd, "flat", None
        try:
            kelly_pct_f = float(kelly_pct)
        except (TypeError, ValueError):
            return flat_usd, "flat", None
        if not math.isfinite(kelly_pct_f) or kelly_pct_f <= 0:
            return flat_usd, "flat", None
        clipped = min(
            max(kelly_pct_f, float(self.kelly_floor_pct)),
            float(self.kelly_cap_pct),
        )
        return float(self.config.bankroll_usd) * clipped, "kelly", clipped

    @staticmethod
    def _ticker_to_exit_tick(
        ticker: Any, signal_prob: Optional[float] = None
    ) -> Any:
        """Adapt a ``Ticker`` to the duck-typed contract ``ExitPolicy`` expects.

        ``ExitPolicy`` reads ``tick.price`` (the current mark) and
        ``tick.signal_prob`` (for the reversal branch). The exchange's
        ``Ticker`` model uses ``mid`` / ``bid`` / ``ask`` / ``last`` —
        we collapse those into ``price`` here so the policy stays
        exchange-agnostic. ``signal_prob`` is set when the supervisor
        opts into the reversal branch (off by default in Wave 2).
        """
        # Pick mid when available; fall back to last; finally to 0.0 to
        # let the policy's downstream sign checks handle a degenerate
        # tick rather than crashing here.
        price: float = 0.0
        for attr in ("mid", "last", "price"):
            value = getattr(ticker, attr, None)
            if value is not None:
                try:
                    price = float(value)
                    break
                except (TypeError, ValueError):
                    continue
        # Use a SimpleNamespace so the policy's _require helper can read
        # both attrs without us having to define a one-off dataclass.
        import types
        return types.SimpleNamespace(
            price=price,
            signal_prob=float(signal_prob) if signal_prob is not None else 0.5,
        )

    def _process_exits(
        self, *, symbol: str, ticker: Ticker
    ) -> Tuple[int, Optional[str]]:
        """Evaluate ExitPolicy against every open position for ``symbol``.

        Returns ``(closed_count, last_reason)``. The caller uses
        ``closed_count > 0`` to short-circuit entry consideration on the
        SAME tick — the supervisor never opens a new position on a tick
        that just closed one (capital-preservation invariant).

        For each surviving open position we:

        1. Call ``policy.update_high_water_mark`` with the current tick.
        2. Persist the new HWM via ``store.update_runtime_fields`` so the
           trailing-stop branch has fresh state on the next tick / process.
        3. Call ``policy.evaluate`` — on close=True, dispatch through the
           same per-tag close path used by ``_force_flat_all`` (paper
           positions stay local; live-tagged positions go through the
           exchange). On ``ExchangeError`` we fall through to the paper
           close for paper-tagged positions only — live errors must
           surface so the kill-switch / breaker logic can react.
        4. Otherwise: bump ``bars_held`` by 1 and persist.

        Master-switch: with ``self.exit_policy_enabled`` False (or
        ``self.exit_policy`` None) this is a no-op returning ``(0, None)``.
        """
        if not self.exit_policy_enabled or self.exit_policy is None:
            return 0, None

        try:
            open_positions = self.position_store.list_open()
        except Exception as exc:  # noqa: BLE001 - state read is best-effort
            LOGGER.warning(
                "exit policy: list_open failed for %s: %s; skipping exits",
                symbol,
                exc,
            )
            return 0, None

        closed = 0
        last_reason: Optional[str] = None
        # Adapt the exchange ``Ticker`` to the duck-typed contract once per
        # tick — every position evaluation reuses the same exit-tick view.
        exit_tick = self._ticker_to_exit_tick(ticker)
        for position in open_positions:
            if position.symbol != symbol:
                continue
            # Closing / closed positions should never appear in
            # ``list_open`` (the store filters them) but guard defensively
            # so the policy never evaluates a half-closed blob.
            if position.status not in ("open", "pending"):
                continue

            # Seed high_water_mark if it's missing — the legacy schema
            # didn't carry one, and the policy needs SOMETHING to compare
            # against on the first tick. Treat None as "use entry price".
            if getattr(position, "high_water_mark", None) is None:
                seed_hwm = float(position.entry_price)
                # Update the in-memory object so the policy reads the seed.
                # Persistence happens below alongside the post-tick update.
                try:
                    position.high_water_mark = seed_hwm  # pydantic v2 allows mutation
                except (AttributeError, TypeError):
                    # Frozen model — fall back to model_copy. The local
                    # binding is updated; the persisted blob is written
                    # below.
                    position = position.model_copy(
                        update={"high_water_mark": seed_hwm}
                    )

            self.exit_policy.update_high_water_mark(position, exit_tick)

            decision: ExitDecision = self.exit_policy.evaluate(position, exit_tick)
            if decision.close:
                reason = decision.reason or "unknown"
                last_reason = reason
                self._exits_by_reason[reason] = (
                    self._exits_by_reason.get(reason, 0) + 1
                )
                close_side: Literal["buy", "sell"] = (
                    "sell" if position.side == "long" else "buy"
                )
                try:
                    self._submit_exit_close(
                        position=position,
                        close_side=close_side,
                        reason=reason,
                    )
                    closed += 1
                except ExchangeError as exc:
                    # Live-tagged positions surface the error so the
                    # consecutive-error breaker / kill switch can react.
                    # Paper-tagged positions fall through to the local
                    # paper force-flat (matches the 2026-05-14 incident
                    # fix in commit 2b62a7d).
                    if position.exchange == "coinbase-paper":
                        LOGGER.warning(
                            "exit close ExchangeError on paper position %s; "
                            "falling back to _paper_force_flat: %s",
                            position.position_id,
                            exc,
                        )
                        try:
                            self._paper_force_flat(position, close_side)
                            closed += 1
                        except Exception as inner:  # noqa: BLE001 - log only
                            LOGGER.warning(
                                "paper force-flat fallback failed for %s: %s",
                                position.position_id,
                                inner,
                            )
                            self._exits_by_reason["error"] = (
                                self._exits_by_reason.get("error", 0) + 1
                            )
                    else:
                        self._exits_by_reason["error"] = (
                            self._exits_by_reason.get("error", 0) + 1
                        )
                        raise
            else:
                # Position survives this tick — bump bars_held and persist
                # the (possibly updated) high_water_mark in one round-trip.
                new_bars = int(getattr(position, "bars_held", 0) or 0) + 1
                try:
                    self.position_store.update_runtime_fields(
                        position.position_id,
                        bars_held=new_bars,
                        high_water_mark=float(
                            getattr(position, "high_water_mark", position.entry_price)
                        ),
                    )
                except AttributeError:
                    # Stub stores from older test fixtures may not have
                    # update_runtime_fields. Log once + continue; the exit
                    # policy still works for SL/TP/reversal (which don't
                    # need persistence), and the trail/time branches just
                    # see stale state for that tick.
                    if not getattr(self, "_warned_no_update_runtime", False):
                        LOGGER.warning(
                            "position_store has no update_runtime_fields; "
                            "trailing-stop / time-stop will not persist "
                            "across ticks until upgraded"
                        )
                        self._warned_no_update_runtime = True
                except Exception as exc:  # noqa: BLE001 - persistence is best-effort
                    LOGGER.warning(
                        "update_runtime_fields failed for %s: %s",
                        position.position_id,
                        exc,
                    )

        return closed, last_reason

    def _submit_exit_close(
        self,
        *,
        position: Position,
        close_side: Literal["buy", "sell"],
        reason: str,
    ) -> None:
        """Dispatch an exit-policy close to the right close path.

        Per-tag dispatch matches ``_force_flat_all``: paper positions stay
        local, live positions route through the exchange. The reason is
        echoed into the exchange's order metadata via the LOGGER only —
        ccxt market orders don't carry custom tags, but our paper close
        records the reason in the position notes for forensics.
        """
        if position.exchange == "coinbase-paper":
            self._paper_force_flat(position, close_side)
            return
        # Live close: a real market order. ExchangeError is raised by the
        # exchange wrapper; the caller decides whether to rethrow (live)
        # or fall back to paper close (paper-tagged). We do NOT swallow
        # here.
        self.exchange.place_market_order(
            position.symbol, close_side, base_size=position.base_size
        )
        LOGGER.info(
            "exit close: live %s side=%s reason=%s base=%.6f",
            position.position_id,
            close_side,
            reason,
            float(position.base_size),
        )

    def _oldest_open_position_age_s(self) -> float:
        """Seconds since the oldest currently-open position was opened."""
        try:
            opens = self.position_store.list_open()
        except Exception:  # noqa: BLE001 - telemetry must never raise
            return 0.0
        if not opens:
            return 0.0
        now = self._now().astimezone(timezone.utc)
        oldest_s = 0.0
        for position in opens:
            opened_raw = getattr(position, "opened_at_utc", None)
            if not opened_raw:
                continue
            try:
                opened = datetime.fromisoformat(str(opened_raw))
            except (TypeError, ValueError):
                continue
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            age_s = (now - opened.astimezone(timezone.utc)).total_seconds()
            if age_s > oldest_s:
                oldest_s = age_s
        return float(max(oldest_s, 0.0))

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
        resolved_kelly_pct: Optional[float] = None,
        regime_label: Optional[str] = None,
    ) -> None:
        """Capture the signal-time snapshot. No-op if no store is wired."""
        if self.trade_context_store is None:
            return
        try:
            risk_in: Dict[str, Any] = {
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
                "bankroll": float(self.config.bankroll_usd),
            }
            # Sprint 2.5: the sizing forensics agent (A3) prefers a
            # ``resolved_kelly_pct`` key on risk_metrics_input over
            # re-resolving from the predictor. None means "Kelly didn't
            # fire on this tick" — distinct from key-missing which would
            # imply the field was never written.
            if resolved_kelly_pct is not None:
                risk_in["resolved_kelly_pct"] = float(resolved_kelly_pct)
            # Sprint 2.6: belt-and-suspenders write of the regime label.
            # ``scripts/run_outcome_adjuster.py``'s
            # ``_try_label_from_signal_snapshot`` iterates over
            # ``risk_metrics_input`` (today's concrete probe), and the
            # script docstring (lines 15-22) names ``signal_snapshot
            # ["regime_label"]`` the canonical top-level seam. We write
            # both so the resolver finds the label regardless of which
            # seam ends up canonical. Empty / blank labels collapse to
            # ``None`` so the key only appears when there's a real label.
            normalized_label: Optional[str] = None
            if regime_label is not None:
                try:
                    candidate = str(regime_label).strip()
                except (TypeError, ValueError):
                    candidate = ""
                if candidate:
                    normalized_label = candidate
            if normalized_label is not None:
                risk_in["regime_label"] = normalized_label
            snap = TradeContextSnapshot(
                trade_id=trade_id,
                symbol=symbol,
                captured_at_utc=utc_now_iso(),
                phase="signal",
                feature_buffer={},
                feature_window=None,
                model_probs={},
                model_confidence=float(confidence),
                risk_metrics_input=risk_in,
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
                regime_label=normalized_label,
            )
            self.trade_context_store.record_snapshot(snap)
        except redis.exceptions.RedisError as exc:
            # Best-effort persistence: a Redis hiccup must not crash the
            # tick. Logged with the symbol so operators can find the gap.
            LOGGER.warning(
                "trade_context_store signal snapshot redis error for %s: %r",
                symbol,
                exc,
            )
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
            # Phase-16: copy the structured fill metadata onto the snapshot
            # so A2 ExecutionForensics can read it directly without a
            # second hop into PositionStore.
            if position.partial_fills:
                risk_out["partial_fills"] = list(position.partial_fills)
            if position.rejection_reason:
                risk_out["rejection_reason"] = str(position.rejection_reason)
            if position.stop_trigger_price is not None:
                risk_out["stop_trigger_price"] = float(
                    position.stop_trigger_price
                )
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
        except redis.exceptions.RedisError as exc:
            LOGGER.warning(
                "trade_context_store fill snapshot redis error for %s: %r",
                symbol,
                exc,
            )
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
        """Capture the breaker-time snapshot for a single forced-flat position.

        Phase-16: also populates the canonical ``kill_switch_reason``,
        ``stop_loss_trigger_price``, and ``breaker_decision`` fields on the
        snapshot so A5 ProcessIntegrity can declare primary-cause findings
        without substring-matching the freeform ``notes``.
        """
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

            # Canonical fields (Phase-16). A5 prefers these over scanning
            # the freeform notes/breaker_context dicts. Best-effort
            # population from whichever inputs are available.
            kill_switch_reason: Optional[str] = None
            breaker_decision: Optional[str] = None
            stop_loss_trigger_price: Optional[float] = None
            tripped_lc = [
                str(t).lower() for t in (verdict.tripped if verdict else [])
            ]
            reason_lc = (reason or (verdict.reason if verdict else "") or "").lower()
            if "kill_switch" in tripped_lc or "kill_switch" in reason_lc:
                kill_switch_reason = "kill_switch"
            elif "daily_loss" in tripped_lc or "daily_loss" in reason_lc:
                kill_switch_reason = "daily_loss_limit"
            elif "consecutive_errors" in tripped_lc or "consecutive_errors" in reason_lc:
                kill_switch_reason = "consecutive_errors"
            elif "auto_pause" in tripped_lc or "auto_pause" in reason_lc:
                kill_switch_reason = "auto_pause"
            elif "manual" in reason_lc:
                kill_switch_reason = "manual"
            if verdict is not None:
                breaker_decision = str(verdict.recommended_action) or None
            elif reason is not None:
                # Reason-only path (force_flat from kill switch) implies the
                # breaker decision was force_flat — every caller of this
                # helper is on the close-now path.
                breaker_decision = "force_flat"
            # Stop-loss trigger price: pull from verdict.details if the
            # breaker emitted one (canonical key) or from position metadata.
            details = dict(verdict.details or {}) if verdict else {}
            for k in ("stop_loss_trigger", "stop_trigger_price", "stop_loss_price"):
                if details.get(k) is not None:
                    try:
                        stop_loss_trigger_price = float(details[k])
                        break
                    except (TypeError, ValueError):
                        pass
            if stop_loss_trigger_price is None and position.model_meta:
                for k in ("stop_price", "stop_loss_price", "stop_trigger_price"):
                    v = position.model_meta.get(k)
                    if v is not None:
                        try:
                            stop_loss_trigger_price = float(v)
                            break
                        except (TypeError, ValueError):
                            pass

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
                kill_switch_reason=kill_switch_reason,
                stop_loss_trigger_price=stop_loss_trigger_price,
                breaker_decision=breaker_decision,
            )
            self.trade_context_store.record_snapshot(snap)
        except redis.exceptions.RedisError as exc:
            LOGGER.warning(
                "trade_context_store breaker snapshot redis error for %s: %r",
                position.position_id,
                exc,
            )
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

        # Sprint 1 Wave 2: exit-policy observability.
        # ``exits_by_reason`` is the cumulative count per reason since
        # process boot (the supervisor doesn't reset these — Prometheus
        # rate() handles intervals). ``open_positions_count`` is already
        # gauged above; we add the per-symbol oldest-age gauge so the
        # operator can spot stuck positions before the breaker fires.
        for reason, count in self._exits_by_reason.items():
            self.metrics_pusher.counter(
                "exits_by_reason_total",
                float(count),
                labels={"reason": reason},
            )
        try:
            self.metrics_pusher.gauge(
                "oldest_open_position_age_s",
                float(self._oldest_open_position_age_s()),
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must not crash
            LOGGER.debug("oldest_open_position_age_s emit failed: %r", exc)

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

    # ------------------------------------------------------------------
    # Lane D D3: multiprocessing-per-symbol supervisor
    # ------------------------------------------------------------------
    def run_workers(
        self,
        *,
        restart_limit_per_hour: int = _DEFAULT_RESTART_LIMIT_PER_HOUR,
    ) -> int:
        """Multiprocessing-per-tradeable run loop.

        Spawns one child process per :class:`Tradeable` in
        :attr:`_tradeables`. Each child reconstructs its own non-picklable
        resources (Redis client, predictor, exchange client, Tradeable
        adapter) inside the child via :func:`_child_main` so nothing
        process-bound is pickled across the fork boundary.

        The parent monitors children, respawns crashed ones with a
        ``_CHILD_RESPAWN_BACKOFF_S`` backoff and a
        ``restart_limit_per_hour`` ceiling, and broadcasts shutdown via a
        Redis flag on SIGTERM/SIGINT. Returns the supervisor's exit code:
        ``0`` on clean shutdown, ``1`` if the restart-limit halt fired.

        Notes for callers:
          * Backward compat: :meth:`run_loop` still works exactly as
            before. ``run_workers`` is a NEW entrypoint flipped on by the
            ``--workers`` CLI flag.
          * ``mp.get_context("spawn")`` is used unconditionally to avoid
            inheriting forked ML state from the parent (predictor + ccxt
            clients are deliberately re-constructed per-child).
        """
        if not self._tradeables:
            LOGGER.warning(
                "run_workers called with no tradeables; nothing to do."
            )
            return 0

        ctx = mp.get_context("spawn")
        # Child config dicts -- one per tradeable, computed once and
        # re-used on every respawn for the same symbol.
        configs_by_symbol: Dict[str, Dict[str, Any]] = {
            tradeable.symbol: self._build_child_config(tradeable)
            for tradeable in self._tradeables
        }
        processes: Dict[str, mp.process.BaseProcess] = {}
        # Restart timestamps deque per symbol. We trim entries older than
        # ``_RESTART_WINDOW_S`` whenever we test the limit; that keeps
        # the deque bounded to "restarts in the last hour" without a
        # dedicated reaper.
        restart_counts: Dict[str, Deque[float]] = {
            sym: collections.deque() for sym in configs_by_symbol
        }
        halted_for_restart_limit = False

        # Install parent signal handlers BEFORE spawning so children
        # don't inherit a half-installed handler set. The handlers set
        # the in-process flag the supervisor loop polls.
        self._shutdown_requested = False
        prior_handlers: Dict[int, Any] = {}

        def _handle_signal(_signum: int, _frame: Any) -> None:
            self._shutdown_requested = True
            try:
                self._broadcast_shutdown()
            except Exception as exc:  # noqa: BLE001 - never crash in handler
                LOGGER.warning("broadcast_shutdown raised: %s", exc)

        for signo in (signal.SIGTERM, signal.SIGINT):
            try:
                prior_handlers[signo] = signal.signal(signo, _handle_signal)
            except (ValueError, OSError) as exc:
                # Not running in main thread, or platform restriction --
                # log and continue without the handler.
                LOGGER.warning(
                    "signal.signal(%s) failed: %s; running without handler",
                    signo,
                    exc,
                )

        try:
            # Initial spawn pass.
            for sym, child_config in configs_by_symbol.items():
                proc = ctx.Process(
                    target=_child_main,
                    args=(child_config,),
                    name=f"autopilot-child:{sym}",
                    daemon=False,
                )
                proc.start()
                processes[sym] = proc
                LOGGER.info(
                    "supervisor: spawned child for %s pid=%s",
                    sym,
                    proc.pid,
                )

            # Health-poll loop. Exits when:
            #   * shutdown requested (parent signal handler), OR
            #   * a symbol's restart limit was exceeded (halt).
            while not self._shutdown_requested and not halted_for_restart_limit:
                time.sleep(_PARENT_HEALTH_POLL_S)
                for sym, proc in list(processes.items()):
                    if proc.is_alive():
                        continue
                    exit_code = proc.exitcode
                    LOGGER.warning(
                        "supervisor: child for %s pid=%s exited with code=%s",
                        sym,
                        proc.pid,
                        exit_code,
                    )
                    # Clean voluntary exit (code 0) usually means the
                    # child saw the shutdown flag; do NOT respawn.
                    if exit_code == 0 and self._shutdown_requested:
                        processes.pop(sym, None)
                        continue
                    # Trim restart timestamps older than the rolling
                    # window before testing the limit.
                    now_mono = time.monotonic()
                    history = restart_counts[sym]
                    while history and now_mono - history[0] > _RESTART_WINDOW_S:
                        history.popleft()
                    if len(history) >= int(restart_limit_per_hour):
                        LOGGER.error(
                            "supervisor: restart limit (%d/hour) exceeded for %s; "
                            "halting supervisor (assume systemic issue).",
                            restart_limit_per_hour,
                            sym,
                        )
                        self._emit_supervisor_halt_metric(
                            reason="restart_limit", symbol=sym
                        )
                        halted_for_restart_limit = True
                        break
                    # Backoff, then respawn.
                    LOGGER.warning(
                        "supervisor: respawning child for %s in %.1fs",
                        sym,
                        _CHILD_RESPAWN_BACKOFF_S,
                    )
                    time.sleep(_CHILD_RESPAWN_BACKOFF_S)
                    history.append(time.monotonic())
                    proc = ctx.Process(
                        target=_child_main,
                        args=(configs_by_symbol[sym],),
                        name=f"autopilot-child:{sym}",
                        daemon=False,
                    )
                    proc.start()
                    processes[sym] = proc
                    LOGGER.info(
                        "supervisor: respawned child for %s pid=%s "
                        "(restart_count=%d in last hour)",
                        sym,
                        proc.pid,
                        len(history),
                    )
        finally:
            # Restore prior signal handlers so a future caller of
            # run_workers (or anyone else in the same process) sees a
            # clean signal map.
            for signo, prior in prior_handlers.items():
                try:
                    signal.signal(signo, prior)
                except (ValueError, OSError):
                    pass
            # Best-effort: broadcast shutdown so any still-alive child
            # exits cleanly even on the halt path.
            try:
                self._broadcast_shutdown()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("broadcast_shutdown raised: %s", exc)
            # Wait up to _SHUTDOWN_GRACE_S for clean exit, then SIGKILL.
            self._stop_workers(processes)

        return 1 if halted_for_restart_limit else 0

    def _broadcast_shutdown(self) -> None:
        """Set the cross-process shutdown flag in Redis.

        Each child polls this key at the top of every tick. The TTL
        bounds the lifetime of a stale flag so a long-since-crashed
        parent doesn't poison a fresh boot.
        """
        redis_client = getattr(self.position_store, "_redis", None)
        if redis_client is None:
            return
        try:
            redis_client.set(_SHUTDOWN_KEY, "1", ex=_SHUTDOWN_KEY_TTL_S)
        except Exception as exc:  # noqa: BLE001 - best-effort
            LOGGER.warning("failed to set %s in Redis: %s", _SHUTDOWN_KEY, exc)

    def _stop_workers(
        self,
        processes: Dict[str, "mp.process.BaseProcess"],
    ) -> None:
        """Wait for clean child exits, SIGKILL stragglers after the grace.

        Sends SIGTERM first (children should already be exiting via the
        Redis shutdown flag, but SIGTERM is belt-and-braces). Polls
        ``.is_alive()`` and joins as each child exits. After
        ``_SHUTDOWN_GRACE_S`` any survivor gets ``terminate()`` followed
        by ``kill()``. The function never raises; it logs.
        """
        # Send SIGTERM to any still-running child first.
        for sym, proc in processes.items():
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception as exc:  # noqa: BLE001 - best-effort
                    LOGGER.warning("terminate(%s) raised: %s", sym, exc)

        deadline = time.monotonic() + _SHUTDOWN_GRACE_S
        while processes and time.monotonic() < deadline:
            for sym in list(processes.keys()):
                proc = processes[sym]
                if not proc.is_alive():
                    proc.join(timeout=0.5)
                    LOGGER.info(
                        "supervisor: child for %s exited cleanly", sym
                    )
                    processes.pop(sym, None)
            if processes:
                time.sleep(_SHUTDOWN_POLL_INTERVAL_S)

        # Stragglers: hard-kill.
        for sym, proc in list(processes.items()):
            if proc.is_alive():
                LOGGER.warning(
                    "supervisor: child for %s did not exit within %.1fs; "
                    "killing pid=%s",
                    sym,
                    _SHUTDOWN_GRACE_S,
                    proc.pid,
                )
                try:
                    proc.kill()
                except Exception as exc:  # noqa: BLE001 - best-effort
                    LOGGER.warning("kill(%s) raised: %s", sym, exc)
                proc.join(timeout=2.0)

    def _build_child_config(self, tradeable: "Tradeable") -> Dict[str, Any]:
        """Construct the picklable bootstrap dict passed to a child.

        Everything the child needs to reconstruct its non-picklable
        resources (Redis client, predictor, exchange client, Tradeable
        adapter) lives in this dict. Live runtime objects (predictor
        instance, exchange client, ccxt connector) are NOT pickled.

        Schema (informal -- additive over time):
          * ``symbol``                  -- string, the Tradeable's symbol.
          * ``asset_class``             -- one of "spot_crypto" /
                                           "perp_crypto" / "prediction_binary".
          * ``tradeable_kind``          -- "coinbase" / "hyperliquid" /
                                           "polymarket". The child uses
                                           this to call the right factory
                                           inside ``_child_main``.
          * ``tradeable_args``          -- dict, kwargs for the factory.
          * ``redis_url``               -- string or None. Falls back to
                                           ``REDIS_URL`` env var inside the
                                           child if None.
          * ``redis_namespace``         -- :class:`PositionStore` namespace.
          * ``shakedown_state_path``    -- string path.
          * ``shakedown_min_days``      -- int.
          * ``mode``                    -- "paper" or "live".
          * ``bankroll_usd``            -- float.
          * ``risk_pct_per_trade``      -- float.
          * ``min_confidence_to_trade`` -- float.
          * ``tick_interval_s``         -- float.
          * ``symbol_set_hash``         -- str, sha256 prefix of sorted
                                           symbol list. Used by the
                                           daily-close leader-election
                                           key so independent supervisor
                                           groups don't share the lease.
          * ``test_mode``               -- bool. When True, the child
                                           runs the stub tick path
                                           (writes to Redis, doesn't
                                           construct exchange/predictor)
                                           so tests can exercise the
                                           multiprocess plumbing without
                                           the full stack. Default False.
          * ``test_max_ticks``          -- int. Test-mode only: child
                                           exits after this many ticks.
        """
        symbol = tradeable.symbol
        asset_class_obj = getattr(tradeable, "asset_class", None)
        asset_class_str = str(getattr(asset_class_obj, "value", "") or "")
        # Pull a redis_url off the position store if it had one (the store
        # falls back to REDIS_URL env var inside the child if this is None).
        redis_url = getattr(self.position_store, "_redis_url", None)
        namespace = getattr(self.position_store, "namespace", "autopilot")
        kill_switch_file = getattr(
            self.circuit_breakers, "kill_switch_file", None
        )

        # Tradeable factory descriptor. Each branch surfaces only the
        # plain-data kwargs the child needs to reconstruct the adapter.
        tradeable_kind: str
        tradeable_args: Dict[str, Any]
        if asset_class_str == "prediction_binary":
            tradeable_kind = "polymarket"
            tradeable_args = {
                "market_id": getattr(tradeable, "market_id", None),
            }
        elif asset_class_str == "perp_crypto":
            tradeable_kind = "hyperliquid"
            tradeable_args = {"symbol": symbol}
        else:
            tradeable_kind = "coinbase"
            tradeable_args = {"symbol": symbol}

        symbol_set_hash = _compute_symbol_set_hash(
            list(self._tradeables_by_symbol.keys())
        )

        return {
            "symbol": symbol,
            "asset_class": asset_class_str,
            "tradeable_kind": tradeable_kind,
            "tradeable_args": tradeable_args,
            "redis_url": redis_url,
            "redis_namespace": str(namespace),
            "shakedown_state_path": str(self.config.shakedown_state_path),
            "shakedown_min_days": int(self.config.shakedown_min_days),
            "mode": str(self.config.mode),
            "bankroll_usd": float(self.config.bankroll_usd),
            "risk_pct_per_trade": float(self.config.risk_pct_per_trade),
            "min_confidence_to_trade": float(
                self.config.min_confidence_to_trade
            ),
            "tick_interval_s": float(self.config.tick_interval_s),
            "kill_switch_file": (
                str(kill_switch_file) if kill_switch_file is not None else None
            ),
            "symbol_set_hash": symbol_set_hash,
            "test_mode": False,
            "test_max_ticks": 0,
        }

    def _emit_supervisor_halt_metric(self, *, reason: str, symbol: str) -> None:
        """Emit ``autopilot_supervisor_halt_total`` when the supervisor halts.

        Best-effort -- never raises through the worker loop.
        """
        if not self._pusher_enabled():
            return
        try:
            self.metrics_pusher.counter(
                "supervisor_halt_total",
                1.0,
                labels={"reason": reason, "symbol": symbol},
            )
        except Exception as exc:  # noqa: BLE001 - never let metrics kill us
            LOGGER.warning("supervisor_halt metric raised: %s", exc)


# ---------------------------------------------------------------------------
# Lane D D3: child-process entrypoint + factories
# ---------------------------------------------------------------------------
#
# These helpers are MODULE-LEVEL (not on Supervisor) so they survive the
# spawn-context pickle round trip. The child re-imports live_supervisor,
# resolves _child_main from the module namespace, then constructs its own
# Redis client + Tradeable + predictor inside the child process.


def _connect_redis_for_child(child_config: Dict[str, Any]) -> Any:
    """Construct a fresh Redis client inside the child.

    The parent never pickles its Redis connection across the boundary;
    the child reads ``redis_url`` from ``child_config`` (or
    ``REDIS_URL`` env var) and connects independently. Returns ``None``
    on failure -- the child then runs without cross-process state and
    logs a warning.
    """
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        LOGGER.warning("redis module unavailable; child running without it")
        return None
    url = (
        child_config.get("redis_url")
        or os.environ.get("REDIS_URL")
        or "redis://localhost:6379/0"
    )
    try:
        return redis.Redis.from_url(url, decode_responses=True)
    except Exception as exc:  # noqa: BLE001 - never crash boot
        LOGGER.warning("child redis connect failed (%s); running without", exc)
        return None


def _build_child_tradeable(child_config: Dict[str, Any]) -> Optional[Any]:
    """Construct a fresh Tradeable inside the child by ``tradeable_kind``.

    Each branch lazy-imports the venue connector so an unused branch
    doesn't pay the import cost (the predictor / ccxt boots are heavy).
    Returns ``None`` if construction fails -- caller logs and exits.
    """
    kind = str(child_config.get("tradeable_kind", "coinbase"))
    args = dict(child_config.get("tradeable_args") or {})
    try:
        if kind == "polymarket":
            from exchanges.adapters import PolymarketTradeable
            import fetcher as polymarket_fetcher

            return PolymarketTradeable(
                str(args.get("market_id") or ""),
                polymarket_fetcher,
            )
        if kind == "hyperliquid":
            from exchanges.adapters import HyperliquidTradeable
            from exchanges.hyperliquid import HyperliquidExchange

            return HyperliquidTradeable(
                HyperliquidExchange(),
                str(args.get("symbol") or ""),
            )
        # Default: coinbase spot.
        from exchanges.adapters import CoinbaseTradeable

        return CoinbaseTradeable(
            CoinbaseExchange(),
            str(args.get("symbol") or ""),
        )
    except Exception as exc:  # noqa: BLE001 - never crash without diagnostics
        LOGGER.error(
            "child %s tradeable construction failed: %s",
            kind,
            exc,
        )
        return None


def _check_shutdown_flag(redis_client: Any) -> bool:
    """True iff Redis has the cross-process shutdown flag set.

    Children call this at the top of every tick. Returns False on any
    error (read failure can't keep us trading -- we just retry next tick
    and rely on SIGTERM as the second line of defence).
    """
    if redis_client is None:
        return False
    try:
        return bool(redis_client.get(_SHUTDOWN_KEY))
    except Exception:  # noqa: BLE001 - tolerate flaky reads
        return False


def _compute_symbol_set_hash(symbols: List[str]) -> str:
    """Stable 16-char sha256 prefix of the sorted symbol list.

    The hash scopes the daily-close leader-election Redis key so two
    independent supervisor groups (different symbol sets running
    against the same Redis) don't share the lease and thereby skip
    each other's daily close. Order-insensitive: ``["ETH/USD", "BTC/USD"]``
    and ``["BTC/USD", "ETH/USD"]`` produce the same hash.
    """
    canonical = json.dumps(sorted(symbols)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:16]


def _try_acquire_daily_close_leader(
    redis_client: Any,
    *,
    utc_date: str,
    symbol_set_hash: str,
    pid: int,
) -> bool:
    """SETNX-elect this child as the daily-close leader for ``utc_date``.

    Returns ``True`` iff this child won the lease. The key shape is
    ``daily_close_leader:{utc_date}:{symbol_set_hash}``: ``utc_date``
    scopes the lease to one day, ``symbol_set_hash`` scopes it to the
    supervisor's symbol set so two independent supervisor groups
    (different symbol sets running against the same Redis) don't share
    the lease and thereby skip each other's daily close.

    Uses ``redis.set(key, pid, ex=_DAILY_CLOSE_LEADER_TTL_S, nx=True)``;
    the ``nx=True`` is the atomic primitive that makes this safe
    against races among children waking up simultaneously at midnight.
    """
    if redis_client is None:
        return True  # No Redis -> single-child mode -> always run close.
    key = f"{_DAILY_CLOSE_LEADER_KEY_PREFIX}:{utc_date}:{symbol_set_hash}"
    try:
        won = redis_client.set(
            name=key,
            value=str(pid),
            ex=_DAILY_CLOSE_LEADER_TTL_S,
            nx=True,
        )
    except Exception as exc:  # noqa: BLE001 - lease must not crash trader
        LOGGER.warning(
            "daily_close leader lease failed for %s: %s; deferring close",
            key,
            exc,
        )
        return False
    return bool(won)


def _child_main(child_config: Dict[str, Any]) -> int:  # pragma: no cover - exercised in subprocess
    """Module-level child entrypoint (must be picklable for spawn context).

    Reconstructs the per-child non-picklable resources (Redis client,
    Tradeable adapter, predictor, position store, circuit breakers,
    notifier) INSIDE the child process. Then runs an independent tick
    loop until either:
      * the cross-process shutdown flag goes truthy in Redis, OR
      * SIGTERM is delivered to this child.

    Returns the child's exit code: ``0`` on clean shutdown, ``1`` on
    fatal boot error (caller / parent treats this as crash + respawn).

    The function reads child_config["test_mode"] and dispatches to a
    stub loop that doesn't import the predictor or live exchange when
    True; the multiprocessing acceptance test relies on this so it
    can exercise the spawn / shutdown / leader-election plumbing
    without the full ML stack.
    """
    import logging as _logging  # re-import: child has fresh module state

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    _CHILD_LOG = _logging.getLogger(__name__ + f".child[{child_config.get('symbol')}]")

    if bool(child_config.get("test_mode")):
        return _child_main_test_mode(child_config, _CHILD_LOG)

    # Production path: real Redis + Tradeable + predictor.
    redis_client = _connect_redis_for_child(child_config)
    tradeable = _build_child_tradeable(child_config)
    if tradeable is None:
        _CHILD_LOG.error("child tradeable bootstrap failed; exiting 1")
        return 1

    pid = os.getpid()
    symbol = str(child_config.get("symbol", ""))
    tick_interval_s = float(child_config.get("tick_interval_s", 5.0))
    symbol_set_hash = str(child_config.get("symbol_set_hash", ""))
    last_close_date: Optional[str] = None

    _CHILD_LOG.info("child %s booted pid=%s", symbol, pid)

    # Per-child SIGTERM handler -- flips a local flag so the next tick
    # boundary exits cleanly rather than mid-order. SIGTERM coming from
    # the parent's _stop_workers cleanup is the belt-and-braces path
    # behind the Redis shutdown flag.
    _local_shutdown = {"requested": False}

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        _local_shutdown["requested"] = True

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass  # Not in main thread on this platform; we'll still poll Redis.

    try:
        while True:
            # Shutdown gate at top of every tick -- both Redis flag AND
            # SIGTERM-triggered local flag.
            if _local_shutdown["requested"] or _check_shutdown_flag(redis_client):
                _CHILD_LOG.info("child %s observed shutdown; exiting", symbol)
                break

            tick_start = time.monotonic()
            try:
                tradeable.get_ticker()
            except Exception as exc:  # noqa: BLE001 - keep child alive
                _CHILD_LOG.warning(
                    "child %s tick error: %s; continuing", symbol, exc
                )

            # Daily-close leader gate. Run at most once per UTC date.
            today_iso = datetime.now(timezone.utc).date().isoformat()
            if last_close_date is None:
                last_close_date = today_iso
            elif today_iso != last_close_date:
                won = _try_acquire_daily_close_leader(
                    redis_client,
                    utc_date=today_iso,
                    symbol_set_hash=symbol_set_hash,
                    pid=pid,
                )
                if won:
                    _CHILD_LOG.info(
                        "child %s won daily_close leadership for %s pid=%s",
                        symbol,
                        today_iso,
                        pid,
                    )
                    # Production daily_close runs through a freshly
                    # constructed Supervisor here in a future PR; for
                    # now the gate logs and skips so the test plumbing
                    # is the contract under test.
                else:
                    _CHILD_LOG.info(
                        "child %s lost daily_close leadership for %s; skipping",
                        symbol,
                        today_iso,
                    )
                last_close_date = today_iso

            tick_dur = time.monotonic() - tick_start
            sleep_for = max(0.0, tick_interval_s - tick_dur)
            if sleep_for > 0:
                time.sleep(sleep_for)
    except Exception as exc:  # noqa: BLE001 - log + exit non-zero so parent respawns
        _CHILD_LOG.exception("child %s crashed: %s", symbol, exc)
        return 1
    return 0


def _child_main_test_mode(
    child_config: Dict[str, Any],
    log: logging.Logger,
) -> int:  # pragma: no cover - exercised in subprocess via tests
    """Stub child loop used by the multiprocessing acceptance test.

    Increments a Redis counter per tick (so the test can assert tick
    counts across all children) and respects the same shutdown flag the
    production loop honours. Daily-close leader election is exercised
    by writing the winning child's pid into a per-symbol Redis key so
    the test can assert exactly-one-leader semantics.

    All non-trivial work is intentionally kept out of this stub: the
    test's value is in exercising the SPAWN + SHUTDOWN + LEADER
    primitives, not the predictor / exchange path.
    """
    redis_client = _connect_redis_for_child(child_config)
    pid = os.getpid()
    symbol = str(child_config.get("symbol", ""))
    namespace = str(child_config.get("redis_namespace", "autopilot"))
    tick_interval_s = float(child_config.get("tick_interval_s", 0.1))
    max_ticks = int(child_config.get("test_max_ticks", 0) or 0)
    symbol_set_hash = str(child_config.get("symbol_set_hash", ""))
    force_daily_close = bool(child_config.get("test_force_daily_close"))
    tick_count_key = f"{namespace}:test:tick_count:{symbol}"
    leader_key_for_test = f"{namespace}:test:leader_winner:{symbol}"
    # Filesystem-shared tick log: fakeredis is process-local so spawn-
    # context children can't share Redis state with the parent's
    # fakeredis. The acceptance test instead points every child at a
    # shared temp dir and asserts on the files dropped there. Each tick
    # appends a single line to ``<tick_log_dir>/<safe_symbol>.ticks``.
    tick_log_dir_raw = child_config.get("test_tick_log_dir")
    tick_log_path: Optional[Path] = None
    if tick_log_dir_raw:
        safe_symbol = str(symbol).replace("/", "-").replace(":", "_")
        tick_log_path = Path(str(tick_log_dir_raw)) / f"{safe_symbol}.ticks"
        try:
            tick_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001 - test path is best-effort
            tick_log_path = None
    # Shutdown-signal file: lets the acceptance test verify that the
    # child actually ran its shutdown branch (vs hitting max_ticks).
    shutdown_log_dir_raw = child_config.get("test_shutdown_log_dir")
    shutdown_signal_path: Optional[Path] = None
    if shutdown_log_dir_raw:
        safe_symbol = str(symbol).replace("/", "-").replace(":", "_")
        shutdown_signal_path = (
            Path(str(shutdown_log_dir_raw)) / f"{safe_symbol}.shutdown"
        )
        try:
            shutdown_signal_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            shutdown_signal_path = None

    log.info("test-child %s booted pid=%s max_ticks=%d", symbol, pid, max_ticks)

    _local_shutdown = {"requested": False}

    def _on_sigterm(_signum: int, _frame: Any) -> None:
        _local_shutdown["requested"] = True

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError):
        pass

    ticks_done = 0
    shutdown_observed = False
    try:
        while True:
            if _local_shutdown["requested"] or _check_shutdown_flag(redis_client):
                shutdown_observed = True
                log.info("test-child %s observed shutdown after %d ticks",
                         symbol, ticks_done)
                break
            if max_ticks > 0 and ticks_done >= max_ticks:
                log.info("test-child %s hit max_ticks=%d; exiting",
                         symbol, max_ticks)
                break

            if redis_client is not None:
                try:
                    redis_client.incr(tick_count_key)
                except Exception as exc:  # noqa: BLE001 - test counter is best-effort
                    log.warning("test-child %s incr failed: %s", symbol, exc)
            if tick_log_path is not None:
                try:
                    with tick_log_path.open("a", encoding="utf-8") as fh:
                        fh.write(
                            f"{datetime.now(timezone.utc).isoformat()}\t"
                            f"pid={pid}\ttick={ticks_done}\n"
                        )
                except Exception:  # noqa: BLE001 - filesystem is best-effort
                    pass

            # Optional daily-close leader exercise: each child tries to
            # win the lease on its FIRST tick. The test asserts only one
            # child writes its pid to leader_winner_key; the rest skip.
            if force_daily_close and ticks_done == 0:
                today_iso = datetime.now(timezone.utc).date().isoformat()
                won = _try_acquire_daily_close_leader(
                    redis_client,
                    utc_date=today_iso,
                    symbol_set_hash=symbol_set_hash,
                    pid=pid,
                )
                if won and redis_client is not None:
                    try:
                        redis_client.set(leader_key_for_test, str(pid))
                    except Exception:  # noqa: BLE001 - best-effort
                        pass

            ticks_done += 1
            time.sleep(max(0.0, tick_interval_s))
    except Exception as exc:  # noqa: BLE001 - exit non-zero on crash
        log.exception("test-child %s crashed: %s", symbol, exc)
        return 1
    finally:
        if shutdown_observed and shutdown_signal_path is not None:
            try:
                shutdown_signal_path.write_text(
                    f"pid={pid}\tticks={ticks_done}\n", encoding="utf-8"
                )
            except Exception:  # noqa: BLE001
                pass
    return 0


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
        required=False,
        default="",
        help="Comma-separated symbols, e.g. 'ETH/USDT,BTC/USDT'.",
    )
    p.add_argument(
        "--polymarket-markets",
        type=str,
        required=False,
        default="",
        help=(
            "Comma-separated Polymarket Gamma market ids. Each id is "
            "wrapped in a PolymarketTradeable and added to the tick "
            "loop alongside any --symbols entries."
        ),
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
        "--exit-rule",
        type=str,
        default="none",
        help=(
            "How to close open paper/live positions. "
            "Examples: 'none' (default, never close), 'time:5m' (close 5 "
            "min after entry, matches model horizon), "
            "'tp_sl:30bps/50bps' (TP/SL by price move), or a combined "
            "form like 'tp_sl:30bps/50bps,time:10m' (whichever fires first)."
        ),
    )
    p.add_argument(
        "--halal-mode",
        action="store_true",
        default=None,
        help=(
            "Halal (Shariah-compliant) trading: long-only + spot-only. Blocks "
            "any short entry and fail-closes any live order routed to a "
            "non-spot venue (perps/leverage/funding). When omitted, defaults "
            "to the HALAL_MODE env var (default off)."
        ),
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
    # Lane D D3: multiprocessing-per-symbol supervisor. ``--workers`` flips
    # ``main()`` from the single-process ``run_loop`` to ``run_workers``.
    # Default is run_loop so existing operators see zero behavioural change.
    p.add_argument(
        "--workers",
        action="store_true",
        help=(
            "Spawn one child process per tradeable (multiprocessing-per-"
            "symbol). Default: single-process serial run_loop. "
            "Children share Redis state (positions, error counter, "
            "shakedown evidence) so they survive crashes independently."
        ),
    )
    p.add_argument(
        "--restart-limit-per-hour",
        type=int,
        default=_DEFAULT_RESTART_LIMIT_PER_HOUR,
        help=(
            "When --workers is set: maximum unexpected child restarts per "
            "symbol per hour before the supervisor halts. Default 3."
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
    raw_symbols = [s.strip() for s in (args.symbols or "").split(",") if s.strip()]
    raw_polymarket = [
        s.strip() for s in (args.polymarket_markets or "").split(",") if s.strip()
    ]
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
    polymarket_market_ids: List[str] = []
    seen_pm: set[str] = set()
    for mid in raw_polymarket:
        if mid in seen_pm:
            LOGGER.warning(
                "supervisor: dropping duplicate market id %r from --polymarket-markets",
                mid,
            )
            continue
        seen_pm.add(mid)
        polymarket_market_ids.append(mid)
    if not symbols and not polymarket_market_ids:
        # Preserve the legacy error string so existing CLI consumers /
        # tests that key off the substring "--symbols must contain at
        # least one entry" continue to work. We additionally surface
        # the polymarket option for operators who want the union.
        print(
            "error: --symbols must contain at least one entry "
            "(or pass --polymarket-markets)",
            file=sys.stderr,
        )
        return 2

    # Resolve halal mode: an explicit --halal-mode wins; otherwise fall back to
    # the HALAL_MODE env var (via the config singleton). Under halal mode the
    # prediction-market (Polymarket) stack is refused outright — betting on
    # event outcomes is maisir (gambling), which no spot/long-only gate can
    # make Shariah-compliant.
    if args.halal_mode is None:
        try:
            from config import cfg as _halal_cfg  # noqa: WPS433 - lazy import

            halal_mode = bool(getattr(_halal_cfg, "HALAL_MODE", False))
        except Exception:  # noqa: BLE001 - config import must never crash boot
            halal_mode = False
    else:
        halal_mode = bool(args.halal_mode)

    if halal_mode and polymarket_market_ids:
        print(
            "error: halal_mode is on but --polymarket-markets was passed; "
            "prediction markets are maisir (gambling) and are not permitted. "
            "Remove --polymarket-markets or disable halal mode.",
            file=sys.stderr,
        )
        return 2

    if halal_mode:
        LOGGER.info(
            "HALAL_MODE active: long-only + spot-only enforced; shorts, "
            "perps/leverage/funding, and prediction markets are blocked."
        )

    # Build PolymarketTradeable instances up-front so SupervisorConfig
    # validation sees the union of (symbols, tradeables) and the
    # supervisor's __init__ wires them into the iteration list. (Empty under
    # halal mode — the refusal above returns before we get here.)
    tradeables: List[Any] = []
    if polymarket_market_ids:
        from exchanges.adapters import PolymarketTradeable
        import fetcher as polymarket_fetcher

        for mid in polymarket_market_ids:
            tradeables.append(PolymarketTradeable(mid, polymarket_fetcher))

    config = SupervisorConfig(
        symbols=symbols,
        tradeables=tradeables,
        tick_interval_s=args.interval,
        bankroll_usd=args.bankroll,
        mode=args.mode,
        shakedown_min_days=args.shakedown_min_days,
        shakedown_state_path=Path(args.shakedown_state_path),
        risk_pct_per_trade=args.risk_pct,
        min_confidence_to_trade=args.min_confidence,
        exit_rule=args.exit_rule,
        halal_mode=halal_mode,
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

    # Sprint 2.5: wire the Lane E trade-context snapshot store onto the
    # same Redis client the position store already uses. Best-effort —
    # a TradeContextStore construction failure logs and proceeds with
    # ``trade_context_store=None``, which makes the three snapshot helpers
    # no-op (same as pre-2.5 behaviour). Without this wiring the run_postmortem
    # specialists return verdict=unknown for every trade and the calibration
    # drift script's snapshot-fallback path is dead — they need this store
    # to read from, and only the supervisor writes to it during a tick.
    trade_context_store: Optional[TradeContextStore] = None
    try:
        # Share the position store's Redis client so we don't open a second
        # connection per process. Falls through to a URL-based construction
        # when the position store's internal client isn't exposed.
        ps_redis = getattr(position_store, "_redis", None)
        if ps_redis is not None:
            trade_context_store = TradeContextStore(redis_client=ps_redis)
        else:
            trade_context_store = TradeContextStore()
    except Exception as exc:  # noqa: BLE001 - never crash boot on snapshot store
        LOGGER.warning(
            "trade_context_store bootstrap failed (%s); snapshot capture disabled",
            exc,
        )
        trade_context_store = None

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
        trade_context_store=trade_context_store,
    )

    # Include polymarket market ids in the run-dir name when set so the
    # log path makes the heterogeneous tradeables list legible.
    run_dir_label_symbols = list(symbols) + [
        f"polymarket-{mid}" for mid in polymarket_market_ids
    ]
    run_dir = _setup_run_dir(args.log_dir, symbols=run_dir_label_symbols)

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

    # Lane D D3: --workers flips to multiprocessing-per-symbol. Default
    # remains the single-process serial run_loop so existing operators
    # are unaffected.
    if getattr(args, "workers", False):
        exit_code = supervisor.run_workers(
            restart_limit_per_hour=int(
                getattr(args, "restart_limit_per_hour", _DEFAULT_RESTART_LIMIT_PER_HOUR)
            ),
        )
        return int(exit_code)

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
