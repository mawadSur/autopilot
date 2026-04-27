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
import json
import logging
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
from risk.circuit_breakers import (
    CircuitBreakerSet,
    CircuitBreakerVerdict,
    DecisionContext,
)
from state.position_store import Position, PositionStore

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


class ShakedownState(BaseModel):
    """Rolling shakedown evidence persisted between supervisor processes."""

    model_config = ConfigDict(extra="forbid")

    started_at_utc: str
    paper_days_clean: int = 0
    last_evaluation_utc: Optional[str] = None
    daily_history: List[Dict[str, Any]] = Field(default_factory=list)
    live_unlocked_at_utc: Optional[str] = None
    equity_peak_usd: float = 0.0


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

        # Per-iteration error counter; rolled into shakedown evidence on
        # daily_close().
        self._errors_today = 0
        self._kill_switch_trips_today = 0

        # One-time warning latches.
        self._warned_live_locked = False

        # Hydrate (or initialise) the shakedown evidence file.
        self.shakedown_state: ShakedownState = self._load_or_init_shakedown()

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
        path = self.config.shakedown_state_path
        if path.exists():
            try:
                blob = path.read_text(encoding="utf-8")
                return ShakedownState.model_validate_json(blob)
            except Exception as exc:  # noqa: BLE001 - corrupt file recovery
                LOGGER.warning(
                    "Shakedown state at %s is corrupt (%s); reinitialising.",
                    path,
                    exc,
                )
        state = ShakedownState(started_at_utc=self._now().isoformat())
        self._persist_shakedown(state)
        return state

    def _persist_shakedown(self, state: ShakedownState) -> None:
        path = self.config.shakedown_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            LOGGER.warning("Failed to persist shakedown state to %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_live_unlocked(self) -> bool:
        """True only if mode='live' AND shakedown evidence is sufficient."""
        if self.config.mode != "live":
            return False
        return (
            self.shakedown_state.paper_days_clean
            >= self.config.shakedown_min_days
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
                self._errors_today += 1
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
        }

    # ------------------------------------------------------------------
    # Daily rollup + shakedown evaluation
    # ------------------------------------------------------------------
    def evaluate_shakedown(self) -> ShakedownState:
        """Roll today's evidence into the shakedown state and persist it.

        Reset triggers (any of these resets ``paper_days_clean`` to 0):
            * uncaught error during a tick
            * kill switch tripped during the day
            * realised daily PnL <= -daily_loss_limit_usd
              (only applies when the breaker is configured)
        """
        now = self._now()
        today_iso = _today_iso(now)
        daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        errors = int(self._errors_today)
        ks_trips = int(self._kill_switch_trips_today)

        daily_loss_breaker_tripped = False
        limit = self.circuit_breakers.daily_loss_limit_usd
        if limit is not None and daily_pnl <= -float(limit):
            daily_loss_breaker_tripped = True

        clean = (
            errors == 0
            and ks_trips == 0
            and not daily_loss_breaker_tripped
        )

        if clean:
            self.shakedown_state.paper_days_clean += 1
        else:
            self.shakedown_state.paper_days_clean = 0

        self.shakedown_state.last_evaluation_utc = now.isoformat()
        self.shakedown_state.daily_history.append(
            {
                "date": today_iso,
                "daily_pnl_usd": daily_pnl,
                "errors_count": errors,
                "kill_switch_trips": ks_trips,
                "daily_loss_breaker_tripped": daily_loss_breaker_tripped,
                "clean": clean,
            }
        )
        if (
            self.shakedown_state.paper_days_clean
            >= self.config.shakedown_min_days
            and self.shakedown_state.live_unlocked_at_utc is None
        ):
            self.shakedown_state.live_unlocked_at_utc = now.isoformat()

        # Update the equity high-water mark tracked across processes.
        equity_now = self.config.bankroll_usd + daily_pnl
        if equity_now > self.shakedown_state.equity_peak_usd:
            self.shakedown_state.equity_peak_usd = equity_now

        self._persist_shakedown(self.shakedown_state)

        # Reset per-day counters once they've been folded into the evidence.
        self._errors_today = 0
        self._kill_switch_trips_today = 0
        return self.shakedown_state

    def daily_close(self) -> Dict[str, Any]:
        """Send daily summary, persist evidence, return rollup dict."""
        now = self._now()
        daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        open_positions = self.position_store.list_open()
        closed_today = self.position_store.list_closed_today(now_utc=now)
        equity_usd = self.config.bankroll_usd + daily_pnl

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
            self._errors_today += 1
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

        # 3. Build context + run breakers.
        proposed = float(self.config.bankroll_usd) * float(
            self.config.risk_pct_per_trade
        )
        daily_pnl = float(self.position_store.daily_realized_pnl_usd())
        equity_current = float(self.config.bankroll_usd) + daily_pnl
        equity_peak = max(
            equity_current,
            float(self.shakedown_state.equity_peak_usd),
        )

        # We need a tentative side for the breaker context. Predict first so
        # the breaker sees the actual side, but the canonical confidence
        # gate happens AFTER the breaker decides.
        try:
            side, confidence = self.model_predict_fn(symbol, ticker)
        except Exception as exc:  # noqa: BLE001 - model errors should not crash
            self._errors_today += 1
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

        # 6. Mode gate -- live falls back to paper if shakedown not unlocked.
        effective_mode: Literal["paper", "live"] = "paper"
        notes_extra: Optional[str] = None
        if self.config.mode == "live":
            if self.is_live_unlocked():
                effective_mode = "live"
            else:
                if not self._warned_live_locked:
                    LOGGER.warning(
                        "mode='live' but shakedown not unlocked "
                        "(paper_days_clean=%d, required=%d); "
                        "falling back to paper-trade simulation.",
                        self.shakedown_state.paper_days_clean,
                        self.config.shakedown_min_days,
                    )
                    self._warned_live_locked = True
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
            self._errors_today += 1
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
            self._errors_today += 1
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
        order: OrderResult = self.exchange.place_market_order(
            symbol, side, quote_size_usd=proposed_usd
        )
        # Record the position; promote to open if the response shows fully
        # filled, otherwise keep as pending so reconcile can clean up later.
        position = self._position_from_order(
            order=order, symbol=symbol, side=side, fallback_price=ticker.mid
        )
        if order.status == "filled":
            self.position_store.record_open(position)
        else:
            self.position_store.record_pending(position)

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
        """Synthesise a paper-trade fill and record it directly.

        Slippage model: 5 bps off the ticker mid, signed against the order
        (buys pay above mid, sells receive below mid). Fees are zeroed.
        """
        slip = PAPER_SLIPPAGE_BPS / 10_000.0
        if side == "buy":
            fill_price = float(ticker.mid) * (1.0 + slip)
        else:
            fill_price = float(ticker.mid) * (1.0 - slip)
        if fill_price <= 0:
            fill_price = float(ticker.mid) or 1.0

        base_size = float(proposed_usd) / fill_price
        position = Position(
            position_id=str(uuid.uuid4()),
            exchange="coinbase-paper",
            symbol=symbol,
            side="long" if side == "buy" else "short",
            status="open",
            entry_price=fill_price,
            entry_quote_usd=fill_price * base_size,
            base_size=base_size,
            entry_order_id=f"paper-{uuid.uuid4().hex[:12]}",
            opened_at_utc=self._now().isoformat(),
            fees_usd=0.0,
            notes="paper-simulated",
        )
        self.position_store.record_open(position)

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
    return p.parse_args(argv)


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
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
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
            LOGGER.info("supervisor: using LegacyTransformerPredictor")
        else:
            LOGGER.warning(
                "supervisor: no legacy predictor available; using placeholder "
                "(every tick will be skipped_low_confidence)"
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "supervisor: predictor bootstrap failed (%s); using placeholder", exc
        )

    supervisor = Supervisor(
        config=config,
        exchange=exchange,
        position_store=position_store,
        circuit_breakers=circuit_breakers,
        notifier=notifier,
        model_predict_fn=predict_fn,
        metrics_pusher=metrics_pusher,
    )

    if args.once:
        ticks = supervisor.run_once()
        print(json.dumps([t.model_dump(mode="json") for t in ticks], indent=2))
        return 0

    summary = supervisor.run_loop()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
