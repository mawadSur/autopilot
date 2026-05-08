"""Unit tests for ``src/live_supervisor.py`` (Phase 5).

Every Phase 1-4 collaborator is replaced with an in-memory stub so the suite
runs without Redis, ccxt, requests, or any other external dependency. The
``sleep_fn`` and ``now_fn`` injection points keep timing deterministic.
"""

from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from exchanges.coinbase import ExchangeError, OrderResult, Ticker
from risk.circuit_breakers import CircuitBreakerVerdict, DecisionContext
from state.position_store import Position

import live_supervisor
from live_supervisor import (
    PAPER_SLIPPAGE_BPS,
    ShakedownState,
    Supervisor,
    SupervisorConfig,
    SupervisorTick,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubTicker:
    """Behaves like ``exchanges.coinbase.Ticker`` for tests."""

    def __init__(self, *, symbol: str, mid: float, spread_bps: float = 4.0) -> None:
        bid = mid * (1.0 - spread_bps / 20_000.0)
        ask = mid * (1.0 + spread_bps / 20_000.0)
        self._inner = Ticker(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=mid,
            volume_24h_base=1234.5,
            as_of_utc="2026-04-26T00:00:00+00:00",
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class StubExchange:
    """Records every call so tests can introspect."""

    def __init__(
        self,
        *,
        ticker_mid: float = 2_000.0,
        raise_on_ticker: Optional[Exception] = None,
        raise_on_order: Optional[Exception] = None,
        order_status: str = "filled",
    ) -> None:
        self.ticker_mid = ticker_mid
        self._raise_on_ticker = raise_on_ticker
        self._raise_on_order = raise_on_order
        self._order_status = order_status
        self.ticker_calls: List[str] = []
        self.market_orders: List[Dict[str, Any]] = []

    def get_ticker(self, symbol: str) -> Ticker:
        self.ticker_calls.append(symbol)
        if self._raise_on_ticker is not None:
            raise self._raise_on_ticker
        return StubTicker(symbol=symbol, mid=self.ticker_mid)._inner

    def place_market_order(
        self,
        symbol: str,
        side: str,
        *,
        quote_size_usd: Optional[float] = None,
        base_size: Optional[float] = None,
    ) -> OrderResult:
        self.market_orders.append(
            {
                "symbol": symbol,
                "side": side,
                "quote_size_usd": quote_size_usd,
                "base_size": base_size,
            }
        )
        if self._raise_on_order is not None:
            raise self._raise_on_order
        # Build a plausible filled order.
        size = base_size or (
            (quote_size_usd or 0.0) / self.ticker_mid if self.ticker_mid else 0.0
        )
        return OrderResult(
            order_id=f"ord-{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            type="market",
            quote_size_usd=quote_size_usd,
            base_size=base_size,
            limit_price=None,
            status=self._order_status,  # type: ignore[arg-type]
            filled_base=size if self._order_status == "filled" else 0.0,
            filled_quote_usd=(size * self.ticker_mid)
            if self._order_status == "filled"
            else 0.0,
            avg_fill_price=self.ticker_mid if self._order_status == "filled" else None,
            fee_usd=0.0,
            created_at_utc="2026-04-26T00:00:00+00:00",
            raw_payload={},
        )


class StubPositionStore:
    """In-memory replacement for ``state.position_store.PositionStore``."""

    def __init__(
        self,
        *,
        open_positions: Optional[List[Position]] = None,
        daily_pnl: float = 0.0,
        closed_today: Optional[List[Position]] = None,
    ) -> None:
        self._open: List[Position] = list(open_positions or [])
        self._daily_pnl = daily_pnl
        self._closed_today: List[Position] = list(closed_today or [])
        self.recorded_open: List[Position] = []
        self.recorded_pending: List[Position] = []
        # Lane A P0 #3: error counter shim. The supervisor now calls
        # store.increment_error / errors_today / reset_errors_for_day
        # instead of mutating an in-memory dict on Supervisor itself.
        self._errors_by_symbol: Dict[str, int] = {}

    # -- reads
    def list_open(self) -> List[Position]:
        return list(self._open)

    def list_closed_today(self, *, now_utc: Optional[datetime] = None) -> List[Position]:
        return list(self._closed_today)

    def open_notional_usd(self) -> float:
        return float(sum(p.entry_quote_usd for p in self._open))

    def open_notional_for_symbol(self, symbol: str) -> float:
        return float(
            sum(p.entry_quote_usd for p in self._open if p.symbol == symbol)
        )

    def daily_realized_pnl_usd(self, *, now_utc: Optional[datetime] = None) -> float:
        return float(self._daily_pnl)

    def daily_realized_pnl_usd_for_symbol(
        self, symbol: str, *, now_utc: Optional[datetime] = None
    ) -> float:
        return float(
            sum(
                (p.realized_pnl_usd or 0.0)
                for p in self._closed_today
                if p.symbol == symbol
            )
        )

    # -- writes
    def record_open(self, position: Position) -> Position:
        self.recorded_open.append(position)
        self._open.append(position)
        return position

    def record_pending(self, position: Position) -> Position:
        self.recorded_pending.append(position)
        self._open.append(position)
        return position

    # -- error counter shim (Lane A P0 #3)
    def increment_error(
        self, symbol: str, *, now_utc: Optional[datetime] = None
    ) -> int:
        self._errors_by_symbol[symbol] = (
            self._errors_by_symbol.get(symbol, 0) + 1
        )
        return self._errors_by_symbol[symbol]

    def errors_today(
        self, symbol: str, *, now_utc: Optional[datetime] = None
    ) -> int:
        return int(self._errors_by_symbol.get(symbol, 0))

    def reset_errors_for_day(
        self, *, now_utc: Optional[datetime] = None
    ) -> int:
        had = bool(self._errors_by_symbol)
        self._errors_by_symbol = {}
        return 1 if had else 0


class StubCircuitBreakers:
    """Behaves like ``CircuitBreakerSet`` but driven by canned answers."""

    def __init__(
        self,
        *,
        kill_switch: bool = False,
        verdict_action: Literal[
            "allow", "halt_new_entries", "force_flat"
        ] = "allow",
        daily_loss_limit_usd: Optional[float] = None,
    ) -> None:
        self._kill_switch = kill_switch
        self._verdict_action = verdict_action
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.check_calls: List[DecisionContext] = []

    def is_kill_switch_tripped(self) -> bool:
        return self._kill_switch

    def check(self, ctx: DecisionContext) -> CircuitBreakerVerdict:
        self.check_calls.append(ctx)
        if self._verdict_action == "allow":
            return CircuitBreakerVerdict(
                allow=True,
                tripped=[],
                reason="",
                recommended_action="allow",
                details={},
            )
        return CircuitBreakerVerdict(
            allow=False,
            tripped=[self._verdict_action],
            reason=f"stub {self._verdict_action}",
            recommended_action=self._verdict_action,
            details={},
        )


class StubNotifier:
    def __init__(self) -> None:
        self.info_calls: List[Tuple[str, Optional[Dict[str, str]]]] = []
        self.alert_calls: List[Dict[str, Any]] = []
        self.daily_summary_calls: List[Dict[str, Any]] = []
        self.fill_event_calls: List[Dict[str, Any]] = []
        self.kill_switch_calls: List[str] = []

    def info(self, message: str, *, fields: Optional[Dict[str, str]] = None) -> bool:
        self.info_calls.append((message, fields))
        return True

    def alert(
        self,
        message: str,
        *,
        severity: str = "alert",
        fields: Optional[Dict[str, str]] = None,
    ) -> bool:
        self.alert_calls.append(
            {"message": message, "severity": severity, "fields": fields}
        )
        return True

    def daily_summary(
        self,
        *,
        equity_usd: float,
        daily_pnl_usd: float,
        open_positions: int,
        closed_today: int,
        win_rate_pct: Optional[float] = None,
    ) -> bool:
        self.daily_summary_calls.append(
            {
                "equity_usd": equity_usd,
                "daily_pnl_usd": daily_pnl_usd,
                "open_positions": open_positions,
                "closed_today": closed_today,
                "win_rate_pct": win_rate_pct,
            }
        )
        return True

    def fill_event(
        self,
        *,
        symbol: str,
        side: str,
        fill_price: float,
        fill_size: float,
        fees_usd: float,
    ) -> bool:
        self.fill_event_calls.append(
            {
                "symbol": symbol,
                "side": side,
                "fill_price": fill_price,
                "fill_size": fill_size,
                "fees_usd": fees_usd,
            }
        )
        return True

    def kill_switch_tripped(self, reason: str) -> bool:
        self.kill_switch_calls.append(reason)
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(
    *,
    side: str = "long",
    symbol: str = "ETH/USDT",
    entry_price: float = 2_000.0,
    base_size: float = 0.05,
) -> Position:
    return Position(
        position_id=str(uuid.uuid4()),
        exchange="coinbase",
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status="open",
        entry_price=entry_price,
        entry_quote_usd=entry_price * base_size,
        base_size=base_size,
        entry_order_id=f"order-{uuid.uuid4().hex[:6]}",
        opened_at_utc="2026-04-26T00:00:00+00:00",
    )


# Module-level keep-alive for TemporaryDirectory objects auto-allocated by
# _build_supervisor when callers don't supply a shakedown_path. Holding refs
# here suppresses the noisy implicit-cleanup ResourceWarning at GC time.
_TMPDIRS: List[tempfile.TemporaryDirectory] = []


def _build_supervisor(
    *,
    mode: Literal["paper", "live"] = "paper",
    shakedown_path: Optional[Path] = None,
    paper_days_clean: int = 0,
    shakedown_min_days: int = 14,
    min_confidence: float = 0.6,
    bankroll_usd: float = 10_000.0,
    risk_pct: float = 0.005,
    exchange: Optional[StubExchange] = None,
    position_store: Optional[StubPositionStore] = None,
    circuit_breakers: Optional[StubCircuitBreakers] = None,
    notifier: Optional[StubNotifier] = None,
    predict_fn=None,
    symbols: Optional[List[str]] = None,
) -> Tuple[Supervisor, Dict[str, Any]]:
    """Construct a supervisor with stubbed dependencies, returning both."""
    if shakedown_path is None:
        # Caller didn't supply -- write into a temp dir.
        tmp = tempfile.TemporaryDirectory()
        _TMPDIRS.append(tmp)
        shakedown_path = Path(tmp.name) / "shakedown.json"
    config = SupervisorConfig(
        symbols=symbols or ["ETH/USDT"],
        tick_interval_s=0.0,
        bankroll_usd=bankroll_usd,
        mode=mode,
        shakedown_min_days=shakedown_min_days,
        shakedown_state_path=shakedown_path,
        risk_pct_per_trade=risk_pct,
        min_confidence_to_trade=min_confidence,
    )
    exchange = exchange or StubExchange()
    position_store = position_store or StubPositionStore()
    circuit_breakers = circuit_breakers or StubCircuitBreakers()
    notifier = notifier or StubNotifier()

    sleeps: List[float] = []

    def _fake_sleep(s: float) -> None:
        sleeps.append(s)

    fixed_now = datetime(2026, 4, 26, 12, 0, 0, tzinfo=timezone.utc)

    def _fake_now() -> datetime:
        return fixed_now

    sup = Supervisor(
        config=config,
        exchange=exchange,
        position_store=position_store,
        circuit_breakers=circuit_breakers,
        notifier=notifier,
        model_predict_fn=predict_fn or (lambda s, t: ("buy", 0.9)),
        sleep_fn=_fake_sleep,
        now_fn=_fake_now,
    )
    if paper_days_clean:
        # Per-symbol: seed every configured symbol's clean-day counter so
        # the most-restrictive aggregate matches the legacy single-symbol
        # semantics tests were written against.
        for sym in sup.config.symbols:
            sup.shakedown_state.get_or_init(sym).paper_days_clean = (
                paper_days_clean
            )

    return sup, {
        "exchange": exchange,
        "position_store": position_store,
        "circuit_breakers": circuit_breakers,
        "notifier": notifier,
        "sleeps": sleeps,
        "shakedown_path": shakedown_path,
        "now": fixed_now,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSupervisorInit(unittest.TestCase):
    def test_init_loads_shakedown_state_or_initializes_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "missing.json"
            self.assertFalse(path.exists())
            sup, _ = _build_supervisor(shakedown_path=path)
            self.assertTrue(path.exists(), "supervisor should write a fresh state file")
            self.assertEqual(sup.shakedown_state.paper_days_clean, 0)
            self.assertIsNone(sup.shakedown_state.live_unlocked_at_utc)

            # Now load a second supervisor pointing at the same file: should
            # reuse the persisted state, not overwrite it.
            for sym in sup.config.symbols:
                sup.shakedown_state.get_or_init(sym).paper_days_clean = 7
            sup._persist_shakedown(sup.shakedown_state)
            sup2, _ = _build_supervisor(shakedown_path=path)
            self.assertEqual(sup2.shakedown_state.paper_days_clean, 7)


class TestRunOnceForceFlat(unittest.TestCase):
    def test_run_once_force_flats_when_kill_switch_tripped(self) -> None:
        open_positions = [
            _make_position(side="long", symbol="ETH/USDT"),
            _make_position(side="short", symbol="BTC/USDT"),
        ]
        store = StubPositionStore(open_positions=open_positions)
        breakers = StubCircuitBreakers(kill_switch=True)
        sup, refs = _build_supervisor(
            position_store=store, circuit_breakers=breakers
        )
        ticks = sup.run_once()
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0].action_taken, "force_flatted")
        # Two close orders should have been attempted (one per open position).
        self.assertEqual(len(refs["exchange"].market_orders), 2)
        # And the kill-switch alert went out.
        self.assertEqual(len(refs["notifier"].kill_switch_calls), 1)


class TestRunOnceBreakerHaltsEntries(unittest.TestCase):
    def test_run_once_skips_when_circuit_breaker_halts_new_entries(self) -> None:
        breakers = StubCircuitBreakers(verdict_action="halt_new_entries")
        store = StubPositionStore()
        sup, refs = _build_supervisor(
            position_store=store, circuit_breakers=breakers
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "halted_breaker")
        self.assertEqual(len(refs["exchange"].market_orders), 0)
        self.assertEqual(len(store.recorded_open), 0)


class TestConfidenceFloor(unittest.TestCase):
    def test_run_once_skips_when_model_confidence_below_floor(self) -> None:
        sup, refs = _build_supervisor(
            min_confidence=0.8, predict_fn=lambda s, t: ("buy", 0.5)
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "skipped_low_confidence")
        self.assertEqual(len(refs["exchange"].market_orders), 0)

    def test_run_once_skips_when_predict_returns_nan(self) -> None:
        """Defense-in-depth (P0 #6): even if a predict_fn slips a NaN past
        its own validator, the supervisor must not trade. NaN < floor is
        silently False in Python, which is exactly the bug we're patching.
        """
        store = StubPositionStore()
        exch = StubExchange()
        sup, refs = _build_supervisor(
            min_confidence=0.6,
            predict_fn=lambda s, t: ("buy", float("nan")),
            position_store=store,
            exchange=exch,
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "skipped_low_confidence")
        self.assertEqual(ticks[0].notes, "nan_confidence")
        # Critical: no live order placed and no paper queue populated.
        self.assertEqual(len(refs["exchange"].market_orders), 0)
        self.assertNotIn("ETH/USDT", sup._pending_paper_fills)
        self.assertEqual(len(store.recorded_open), 0)

    def test_run_once_skips_when_predict_returns_pos_inf(self) -> None:
        store = StubPositionStore()
        sup, refs = _build_supervisor(
            min_confidence=0.6,
            predict_fn=lambda s, t: ("buy", float("inf")),
            position_store=store,
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "skipped_low_confidence")
        self.assertEqual(ticks[0].notes, "nan_confidence")
        self.assertEqual(len(store.recorded_open), 0)


class TestPaperModeSynthesisesFill(unittest.TestCase):
    def test_run_once_paper_synthesizes_fill_when_mode_paper(self) -> None:
        """Paper fill is deferred-to-next-tick (P0 #4 state machine).

        First ``run_once`` emits a signal and queues a PendingPaperFill;
        no position is recorded yet. The next ``run_once`` drains the queue
        against the *next* tick's ticker — that is the fill price.
        """
        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0)
        sup, refs = _build_supervisor(
            mode="paper",
            position_store=store,
            exchange=exch,
            predict_fn=lambda s, t: ("buy", 0.9),
        )
        # Signal tick: queue, no live order, no recorded fill yet.
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "allowed")
        self.assertEqual(len(exch.market_orders), 0)
        self.assertEqual(len(store.recorded_open), 0)
        # Now flip the ticker so we can prove the fill price comes from
        # the SECOND tick, not the first.
        exch.ticker_mid = 2_050.0
        sup.run_once()
        self.assertEqual(len(exch.market_orders), 0)
        self.assertEqual(len(store.recorded_open), 1)
        synth = store.recorded_open[0]
        expected_price = 2_050.0 * (1.0 + PAPER_SLIPPAGE_BPS / 10_000.0)
        self.assertAlmostEqual(synth.entry_price, expected_price, places=6)
        self.assertEqual(synth.exchange, "coinbase-paper")
        self.assertEqual(synth.side, "long")


class TestLiveLockedFallsBackToPaper(unittest.TestCase):
    def test_run_once_live_blocked_when_shakedown_not_unlocked(self) -> None:
        """Locked-live -> paper fallback uses the same defer-to-next-tick
        state machine as direct paper mode (P0 #4)."""
        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0)
        sup, refs = _build_supervisor(
            mode="live",
            paper_days_clean=3,
            shakedown_min_days=14,
            position_store=store,
            exchange=exch,
        )
        # Signal tick: queues a pending paper fill but does NOT record it.
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "halted_breaker")
        self.assertEqual(ticks[0].notes, "live_mode_locked")
        self.assertEqual(len(exch.market_orders), 0)
        self.assertEqual(len(store.recorded_open), 0)
        # Next tick drains the queue and synthesises the fill.
        sup.run_once()
        self.assertEqual(len(store.recorded_open), 1)
        self.assertEqual(store.recorded_open[0].exchange, "coinbase-paper")


class TestPaperFillDeferredToNextTick(unittest.TestCase):
    """P0 #4 regression: paper fill price MUST come from the next tick's
    ticker, not the same ticker that produced the signal.

    Without this guard, a live-trader paper backtest secretly leaks the
    signal-tick mid into the fill price -- a subtle but very real
    look-ahead bias that inflates apparent edge.
    """

    def test_paper_fill_uses_next_tick_price_not_signal_tick(self) -> None:
        from live_supervisor import PendingPaperFill

        store = StubPositionStore()
        exch = StubExchange(ticker_mid=1_000.0)
        sup, _ = _build_supervisor(
            mode="paper",
            position_store=store,
            exchange=exch,
            predict_fn=lambda s, t: ("buy", 0.95),
        )
        # First tick at 1000 -- queue, no fill recorded.
        sup.run_once()
        self.assertEqual(len(store.recorded_open), 0)
        self.assertIn("ETH/USDT", sup._pending_paper_fills)
        self.assertIsInstance(
            sup._pending_paper_fills["ETH/USDT"], PendingPaperFill
        )

        # Second tick at 3000 -- the fill is synthesised here against the
        # NEW ticker. The first-tick ticker (1000) must not appear in the
        # fill price; the second-tick ticker (3000) plus 5 bps slippage is
        # the only valid price.
        exch.ticker_mid = 3_000.0
        # Predictor returns < floor so no NEW signal is queued.
        sup.model_predict_fn = lambda s, t: ("buy", 0.1)
        sup.run_once()
        self.assertEqual(len(store.recorded_open), 1)
        recorded = store.recorded_open[0]
        slip = PAPER_SLIPPAGE_BPS / 10_000.0
        self.assertAlmostEqual(
            recorded.entry_price, 3_000.0 * (1.0 + slip), places=4
        )
        # Pending queue is now empty.
        self.assertNotIn("ETH/USDT", sup._pending_paper_fills)

    def test_stale_pending_fill_dropped_after_max_age_ticks(self) -> None:
        """A pending fill older than 2 ticks is dropped, not filled."""
        from live_supervisor import (
            PAPER_PENDING_FILL_MAX_AGE_TICKS,
            PendingPaperFill,
        )

        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0)
        sup, _ = _build_supervisor(
            mode="paper",
            position_store=store,
            exchange=exch,
            predict_fn=lambda s, t: ("buy", 0.1),  # always below floor
        )
        # Manually inject a stale pending fill (enqueued at tick 0, but
        # we'll claim the current tick index is way ahead).
        sup._pending_paper_fills["ETH/USDT"] = PendingPaperFill(
            symbol="ETH/USDT",
            side="buy",
            quote_size_usd=50.0,
            base_size=0.025,
            signal_tick_at="2026-04-26T11:00:00+00:00",
            slippage_bps=PAPER_SLIPPAGE_BPS,
            enqueued_tick_index=0,
        )
        # Pretend we already advanced N ticks beyond the max age.
        sup._symbol_tick_index["ETH/USDT"] = (
            PAPER_PENDING_FILL_MAX_AGE_TICKS + 5
        )
        sup.run_once()
        # Fill was DROPPED, not synthesised.
        self.assertEqual(len(store.recorded_open), 0)
        self.assertNotIn("ETH/USDT", sup._pending_paper_fills)


class TestLiveAllowedAfterShakedown(unittest.TestCase):
    def test_run_once_live_allowed_after_shakedown_unlocked(self) -> None:
        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0, order_status="filled")
        sup, refs = _build_supervisor(
            mode="live",
            paper_days_clean=14,
            shakedown_min_days=14,
            position_store=store,
            exchange=exch,
        )
        self.assertTrue(sup.is_live_unlocked())
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "allowed")
        self.assertEqual(ticks[0].notes, "live")
        # Live order WAS placed.
        self.assertEqual(len(exch.market_orders), 1)
        order_call = exch.market_orders[0]
        self.assertEqual(order_call["side"], "buy")
        # Recorded as 'open' since order_status='filled'.
        self.assertEqual(len(store.recorded_open), 1)
        self.assertEqual(store.recorded_open[0].exchange, "coinbase")
        # Fill event sent.
        self.assertEqual(len(refs["notifier"].fill_event_calls), 1)


class TestExchangeErrorHandling(unittest.TestCase):
    def test_run_once_handles_exchange_error_alerts_and_continues(self) -> None:
        # Get past kill-switch, then have place_market_order raise. We need
        # to put the error on the order path, which means the ticker must
        # succeed and shakedown must allow live so the error fires there --
        # easier to test the path on the live-unlocked branch.
        store = StubPositionStore()
        exch = StubExchange(
            ticker_mid=2_000.0,
            raise_on_order=ExchangeError("boom"),
        )
        sup, refs = _build_supervisor(
            mode="live",
            paper_days_clean=14,
            shakedown_min_days=14,
            position_store=store,
            exchange=exch,
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "errored")
        # Alert sent.
        self.assertEqual(len(refs["notifier"].alert_calls), 1)
        # The loop survived -- a second call should still execute cleanly.
        ticks2 = sup.run_once()
        self.assertEqual(len(ticks2), 1)


class TestRunLoopMaxIterations(unittest.TestCase):
    def test_run_loop_max_iterations_terminates_cleanly(self) -> None:
        sup, refs = _build_supervisor()
        summary = sup.run_loop(max_iterations=3)
        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(summary["total_ticks"], 3)
        self.assertFalse(summary["interrupted"])
        # Sleep is skipped after the final iteration -- so we expect
        # max_iterations - 1 sleep calls.
        self.assertEqual(len(refs["sleeps"]), 2)


class TestRunLoopKeyboardInterrupt(unittest.TestCase):
    def test_run_loop_handles_keyboard_interrupt_with_summary(self) -> None:
        # Predict function raises KeyboardInterrupt mid-tick. The supervisor
        # catches it at the top-level loop layer and exits with a summary.
        called = {"n": 0}

        def kb_pred(s: str, t: Ticker) -> Tuple[Literal["buy", "sell"], float]:
            called["n"] += 1
            if called["n"] >= 2:
                raise KeyboardInterrupt
            return ("buy", 0.9)

        sup, refs = _build_supervisor(predict_fn=kb_pred)
        summary = sup.run_loop(max_iterations=10)
        self.assertTrue(summary["interrupted"])
        self.assertGreaterEqual(summary["iterations"], 1)


class TestEvaluateShakedownIncrementsCleanDay(unittest.TestCase):
    def test_evaluate_shakedown_increments_clean_day(self) -> None:
        store = StubPositionStore(daily_pnl=10.0)
        breakers = StubCircuitBreakers(daily_loss_limit_usd=200.0)
        sup, _ = _build_supervisor(
            position_store=store, circuit_breakers=breakers, paper_days_clean=5
        )
        state = sup.evaluate_shakedown()
        self.assertEqual(state.paper_days_clean, 6)
        self.assertEqual(len(state.daily_history), 1)
        self.assertTrue(state.daily_history[0]["clean"])


class TestEvaluateShakedownResetsOnDailyLossBreaker(unittest.TestCase):
    def test_evaluate_shakedown_resets_on_daily_loss_breaker_trip(self) -> None:
        store = StubPositionStore(daily_pnl=-300.0)
        breakers = StubCircuitBreakers(daily_loss_limit_usd=200.0)
        sup, _ = _build_supervisor(
            position_store=store, circuit_breakers=breakers, paper_days_clean=10
        )
        state = sup.evaluate_shakedown()
        self.assertEqual(state.paper_days_clean, 0)
        self.assertTrue(state.daily_history[0]["daily_loss_breaker_tripped"])
        self.assertFalse(state.daily_history[0]["clean"])


class TestEvaluateShakedownResetsOnKillSwitch(unittest.TestCase):
    def test_evaluate_shakedown_resets_on_kill_switch_trip(self) -> None:
        store = StubPositionStore(open_positions=[_make_position()])
        breakers = StubCircuitBreakers(kill_switch=True)
        sup, _ = _build_supervisor(
            position_store=store,
            circuit_breakers=breakers,
            paper_days_clean=10,
        )
        # Trip the kill switch via a tick.
        sup.run_once()
        state = sup.evaluate_shakedown()
        self.assertEqual(state.paper_days_clean, 0)
        self.assertGreaterEqual(state.daily_history[0]["kill_switch_trips"], 1)


class TestEvaluateShakedownResetsOnUncaughtError(unittest.TestCase):
    def test_evaluate_shakedown_resets_on_uncaught_error(self) -> None:
        # Force an exchange error -> increments the per-day error counter.
        exch = StubExchange(raise_on_ticker=ExchangeError("ticker fail"))
        sup, _ = _build_supervisor(exchange=exch, paper_days_clean=9)
        sup.run_once()  # records 1 error
        state = sup.evaluate_shakedown()
        self.assertEqual(state.paper_days_clean, 0)
        self.assertGreaterEqual(state.daily_history[0]["errors_count"], 1)


class TestIsLiveUnlocked(unittest.TestCase):
    def test_is_live_unlocked_returns_true_only_in_live_mode_with_clean_days(
        self,
    ) -> None:
        # Paper mode -- never unlocked.
        sup_p, _ = _build_supervisor(mode="paper", paper_days_clean=20)
        self.assertFalse(sup_p.is_live_unlocked())

        # Live mode but not enough clean days -- still locked.
        sup_l, _ = _build_supervisor(
            mode="live", paper_days_clean=5, shakedown_min_days=14
        )
        self.assertFalse(sup_l.is_live_unlocked())

        # Live mode with enough clean days -- unlocked.
        sup_ok, _ = _build_supervisor(
            mode="live", paper_days_clean=14, shakedown_min_days=14
        )
        self.assertTrue(sup_ok.is_live_unlocked())


class TestDailyClose(unittest.TestCase):
    def test_daily_close_persists_history_and_sends_summary(self) -> None:
        store = StubPositionStore(daily_pnl=42.5)
        breakers = StubCircuitBreakers(daily_loss_limit_usd=200.0)
        sup, refs = _build_supervisor(
            position_store=store,
            circuit_breakers=breakers,
            paper_days_clean=2,
        )
        rollup = sup.daily_close()
        # Notifier got a daily summary call.
        self.assertEqual(len(refs["notifier"].daily_summary_calls), 1)
        call = refs["notifier"].daily_summary_calls[0]
        self.assertAlmostEqual(call["daily_pnl_usd"], 42.5)
        self.assertAlmostEqual(call["equity_usd"], 10_000.0 + 42.5)
        # State persisted with new history entry.
        self.assertEqual(len(sup.shakedown_state.daily_history), 1)
        self.assertEqual(rollup["paper_days_clean"], 3)
        # File on disk reflects it.
        path = refs["shakedown_path"]
        self.assertTrue(path.exists())
        loaded = ShakedownState.model_validate_json(path.read_text())
        self.assertEqual(loaded.paper_days_clean, 3)


class TestPerSymbolShakedown(unittest.TestCase):
    """Phase 10 multi-symbol orchestration tests.

    Per-symbol semantics:
      * Each configured symbol has its own ``paper_days_clean`` counter.
      * Account-level events (kill switch, daily-loss breaker) reset every
        symbol; per-symbol errors only reset the offending symbol.
      * ``is_live_unlocked(symbol)`` is per-symbol; ``is_live_unlocked()`` is
        the most-restrictive aggregate.
    """

    def test_fresh_supervisor_seeds_entries_for_every_configured_symbol(
        self,
    ) -> None:
        sup, _ = _build_supervisor(symbols=["ETH/USD", "BTC/USD", "SOL/USD"])
        self.assertEqual(
            set(sup.shakedown_state.per_symbol.keys()),
            {"ETH/USD", "BTC/USD", "SOL/USD"},
        )
        for s in sup.shakedown_state.per_symbol.values():
            self.assertEqual(s.paper_days_clean, 0)

    def test_per_symbol_error_only_resets_that_symbol(self) -> None:
        # ETH errors; BTC stays clean -> BTC streak survives.
        sup, _ = _build_supervisor(symbols=["ETH/USD", "BTC/USD"])
        for sym in ("ETH/USD", "BTC/USD"):
            sup.shakedown_state.get_or_init(sym).paper_days_clean = 10
        sup._increment_symbol_errors("ETH/USD")
        sup.evaluate_shakedown()
        self.assertEqual(
            sup.shakedown_state.per_symbol["ETH/USD"].paper_days_clean, 0
        )
        self.assertEqual(
            sup.shakedown_state.per_symbol["BTC/USD"].paper_days_clean, 11
        )

    def test_kill_switch_trip_resets_every_symbol(self) -> None:
        sup, _ = _build_supervisor(symbols=["ETH/USD", "BTC/USD"])
        for sym in ("ETH/USD", "BTC/USD"):
            sup.shakedown_state.get_or_init(sym).paper_days_clean = 7
        sup._kill_switch_trips_today = 1
        sup.evaluate_shakedown()
        self.assertEqual(
            sup.shakedown_state.per_symbol["ETH/USD"].paper_days_clean, 0
        )
        self.assertEqual(
            sup.shakedown_state.per_symbol["BTC/USD"].paper_days_clean, 0
        )

    def test_account_loss_breaker_resets_every_symbol(self) -> None:
        store = StubPositionStore(daily_pnl=-500.0)
        breakers = StubCircuitBreakers(daily_loss_limit_usd=200.0)
        sup, _ = _build_supervisor(
            symbols=["ETH/USD", "BTC/USD"],
            position_store=store,
            circuit_breakers=breakers,
        )
        for sym in ("ETH/USD", "BTC/USD"):
            sup.shakedown_state.get_or_init(sym).paper_days_clean = 12
        sup.evaluate_shakedown()
        for sym in ("ETH/USD", "BTC/USD"):
            self.assertEqual(
                sup.shakedown_state.per_symbol[sym].paper_days_clean, 0
            )

    def test_is_live_unlocked_per_symbol_independent(self) -> None:
        sup, _ = _build_supervisor(
            mode="live",
            symbols=["ETH/USD", "BTC/USD"],
            shakedown_min_days=14,
        )
        sup.shakedown_state.get_or_init("ETH/USD").paper_days_clean = 14
        sup.shakedown_state.get_or_init("BTC/USD").paper_days_clean = 5
        self.assertTrue(sup.is_live_unlocked("ETH/USD"))
        self.assertFalse(sup.is_live_unlocked("BTC/USD"))
        # Aggregate (no-args) = most-restrictive = False.
        self.assertFalse(sup.is_live_unlocked())

    def test_is_live_unlocked_aggregate_true_only_when_all_unlocked(self) -> None:
        sup, _ = _build_supervisor(
            mode="live",
            symbols=["ETH/USD", "BTC/USD"],
            shakedown_min_days=14,
        )
        sup.shakedown_state.get_or_init("ETH/USD").paper_days_clean = 14
        sup.shakedown_state.get_or_init("BTC/USD").paper_days_clean = 14
        self.assertTrue(sup.is_live_unlocked())

    def test_legacy_state_file_migrates_to_per_symbol(self) -> None:
        """A pre-Phase-10 shakedown JSON should load + migrate seamlessly."""
        legacy_blob = (
            '{"started_at_utc": "2026-04-01T00:00:00+00:00", '
            '"paper_days_clean": 9, '
            '"last_evaluation_utc": "2026-04-25T00:00:00+00:00", '
            '"daily_history": [{"date": "2026-04-25", "clean": true}], '
            '"live_unlocked_at_utc": null, '
            '"equity_peak_usd": 12345.67}'
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.json"
            path.write_text(legacy_blob, encoding="utf-8")
            sup, _ = _build_supervisor(
                symbols=["ETH/USD", "BTC/USD"],
                shakedown_path=path,
            )
        # Both symbols inherit the legacy single-counter value.
        self.assertEqual(
            sup.shakedown_state.per_symbol["ETH/USD"].paper_days_clean, 9
        )
        self.assertEqual(
            sup.shakedown_state.per_symbol["BTC/USD"].paper_days_clean, 9
        )
        # Equity peak preserved.
        self.assertAlmostEqual(sup.shakedown_state.equity_peak_usd, 12345.67)
        # Top-level paper_days_clean property = min across symbols.
        self.assertEqual(sup.shakedown_state.paper_days_clean, 9)

    def test_per_symbol_unlock_flips_only_target_symbol(self) -> None:
        sup, _ = _build_supervisor(
            symbols=["ETH/USD", "BTC/USD"], shakedown_min_days=3
        )
        # ETH at 2 (will hit 3 after this evaluation -> unlock); BTC at 0.
        sup.shakedown_state.get_or_init("ETH/USD").paper_days_clean = 2
        sup.evaluate_shakedown()
        self.assertEqual(
            sup.shakedown_state.per_symbol["ETH/USD"].paper_days_clean, 3
        )
        self.assertIsNotNone(
            sup.shakedown_state.per_symbol["ETH/USD"].live_unlocked_at_utc
        )
        # BTC stays at 1 (clean today) and not unlocked.
        self.assertEqual(
            sup.shakedown_state.per_symbol["BTC/USD"].paper_days_clean, 1
        )
        self.assertIsNone(
            sup.shakedown_state.per_symbol["BTC/USD"].live_unlocked_at_utc
        )


class TestPerSymbolEquityPeak(unittest.TestCase):
    """P0 #1: per-symbol equity peak so one symbol's drawdown does not halt
    other symbols' trading.

    Before this fix the supervisor used a single global high-water mark,
    so ETH crashing -50% would have ALSO halted BTC trades via the
    drawdown breaker. After: each symbol's drawdown is computed against
    its own peak.
    """

    def test_per_symbol_pnl_does_not_pollute_other_symbol_drawdown(self) -> None:
        """ETH's losing PnL is invisible from BTC's DecisionContext."""
        from state.position_store import Position

        # Construct closed positions: ETH lost $5000, BTC made $0.
        eth_loss = Position(
            position_id="eth-1",
            exchange="coinbase",
            symbol="ETH/USD",
            side="long",
            status="closed",
            entry_price=2000.0,
            entry_quote_usd=20000.0,
            base_size=10.0,
            opened_at_utc="2026-04-26T00:00:00+00:00",
            realized_pnl_usd=-5000.0,
        )
        store = StubPositionStore(
            daily_pnl=-5000.0,
            closed_today=[eth_loss],
        )
        sup, _ = _build_supervisor(
            symbols=["ETH/USD", "BTC/USD"],
            position_store=store,
            predict_fn=lambda s, t: ("buy", 0.9),
        )
        # Pre-seed BTC's symbol equity peak above the bankroll so a
        # drawdown query for BTC would NOT trip on ETH's losses. ETH's
        # peak stays at default 0; only its own PnL contributes.
        sup.shakedown_state.per_symbol["BTC/USD"].equity_peak_usd = 11_000.0

        ticks = sup.run_once()
        # One tick per symbol -> two contexts captured by StubCircuitBreakers.
        self.assertEqual(len(ticks), 2)
        # Find the BTC context and assert its equity_current is bankroll
        # PLUS BTC's own pnl (0), NOT bankroll PLUS account-aggregate pnl
        # (-5000). The "btc drawdown" check would otherwise see:
        #   peak = 11000, current = 5000 -> 54.5% drawdown.
        # With per-symbol semantics it sees:
        #   peak = 11000, current = 10000 -> 9.1% drawdown only.
        btc_ctx = next(
            c for c in sup.circuit_breakers.check_calls if c.symbol == "BTC/USD"
        )
        self.assertAlmostEqual(btc_ctx.equity_current_usd, 10_000.0, places=2)
        self.assertAlmostEqual(btc_ctx.equity_peak_usd, 11_000.0, places=2)

        eth_ctx = next(
            c for c in sup.circuit_breakers.check_calls if c.symbol == "ETH/USD"
        )
        # ETH's equity_current includes ITS losses: 10000 + (-5000) = 5000.
        self.assertAlmostEqual(eth_ctx.equity_current_usd, 5_000.0, places=2)

    def test_legacy_global_equity_peak_migrates_to_every_symbol(self) -> None:
        """A pre-Lane-A shakedown JSON's global equity_peak_usd should be
        copied into every per-symbol entry on first load (P0 #1)."""
        legacy_blob = (
            '{"started_at_utc": "2026-04-01T00:00:00+00:00", '
            '"paper_days_clean": 5, '
            '"daily_history": [], '
            '"equity_peak_usd": 12345.67}'
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.json"
            path.write_text(legacy_blob, encoding="utf-8")
            sup, _ = _build_supervisor(
                symbols=["ETH/USD", "BTC/USD"],
                shakedown_path=path,
            )
        for sym in ("ETH/USD", "BTC/USD"):
            self.assertAlmostEqual(
                sup.shakedown_state.per_symbol[sym].equity_peak_usd,
                12345.67,
                places=2,
            )

    def test_per_symbol_peak_advances_with_per_symbol_pnl(self) -> None:
        """``evaluate_shakedown`` updates THIS symbol's peak from THIS
        symbol's PnL, not the account aggregate."""
        from state.position_store import Position

        winning = Position(
            position_id="btc-1",
            exchange="coinbase",
            symbol="BTC/USD",
            side="long",
            status="closed",
            entry_price=30000.0,
            entry_quote_usd=30000.0,
            base_size=1.0,
            opened_at_utc="2026-04-26T00:00:00+00:00",
            realized_pnl_usd=2_000.0,
        )
        # Account-level pnl is +500 (-1500 ETH + 2000 BTC), but BTC's own
        # pnl is +2000. BTC's peak should be 12000, not 10500.
        losing = Position(
            position_id="eth-1",
            exchange="coinbase",
            symbol="ETH/USD",
            side="long",
            status="closed",
            entry_price=2000.0,
            entry_quote_usd=2000.0,
            base_size=1.0,
            opened_at_utc="2026-04-26T00:00:00+00:00",
            realized_pnl_usd=-1_500.0,
        )
        store = StubPositionStore(
            daily_pnl=500.0,
            closed_today=[winning, losing],
        )
        sup, _ = _build_supervisor(
            symbols=["ETH/USD", "BTC/USD"],
            position_store=store,
        )
        sup.evaluate_shakedown()
        # BTC's peak: 10000 (bankroll) + 2000 (its pnl).
        self.assertAlmostEqual(
            sup.shakedown_state.per_symbol["BTC/USD"].equity_peak_usd,
            12_000.0,
            places=2,
        )
        # ETH's peak: bankroll-relative -1500 means equity is BELOW
        # bankroll. Peak stays at 0 (its initial value) because the new
        # equity_now (8500) is not greater than 0... wait, is greater.
        # Actually 8500 > 0 so it advances to 8500.
        self.assertAlmostEqual(
            sup.shakedown_state.per_symbol["ETH/USD"].equity_peak_usd,
            8_500.0,
            places=2,
        )


class TestRedisBackedErrorCounter(unittest.TestCase):
    """P0 #3: error counter lives in Redis so the D1 multi-process
    supervisor model preserves increments across processes + restarts.
    """

    def test_two_supervisors_sharing_store_increment_same_counter(self) -> None:
        """Two Supervisor instances backed by ONE PositionStore (one
        FakeRedis) increment the same counter without losing writes."""
        import fakeredis
        from state.position_store import PositionStore

        fake = fakeredis.FakeRedis(decode_responses=True)
        # Use distinct namespaces to prove the counter key still
        # converges to one place when both processes share the same
        # namespace. Namespace 'shared' here is used for both.
        store_a = PositionStore(redis_client=fake, namespace="shared")
        store_b = PositionStore(redis_client=fake, namespace="shared")

        with tempfile.TemporaryDirectory() as td:
            path_a = Path(td) / "shake-a.json"
            path_b = Path(td) / "shake-b.json"
            sup_a, refs_a = _build_supervisor(
                shakedown_path=path_a,
                position_store=store_a,
                exchange=StubExchange(
                    raise_on_ticker=ExchangeError("net glitch")
                ),
            )
            sup_b, refs_b = _build_supervisor(
                shakedown_path=path_b,
                position_store=store_b,
                exchange=StubExchange(
                    raise_on_ticker=ExchangeError("net glitch")
                ),
            )
            # Both supervisors are pinned to the SAME fake clock so they
            # write to the same daily error key.
            sup_a.run_once()
            sup_a.run_once()
            sup_b.run_once()
            sup_b.run_once()
            shared_now = refs_a["now"]
        # Both processes contributed to the SAME shared counter (read
        # against the supervisor's pinned ``now``, not real wallclock).
        self.assertEqual(
            store_a.errors_today("ETH/USDT", now_utc=shared_now), 4
        )
        self.assertEqual(
            store_b.errors_today("ETH/USDT", now_utc=shared_now), 4
        )

    def test_reset_errors_for_day_clears_redis_hash(self) -> None:
        import fakeredis
        from state.position_store import PositionStore

        fake = fakeredis.FakeRedis(decode_responses=True)
        store = PositionStore(redis_client=fake, namespace="reset")
        store.increment_error("ETH/USD")
        store.increment_error("BTC/USD")
        store.increment_error("BTC/USD")
        self.assertEqual(store.errors_today("BTC/USD"), 2)
        # Reset wipes both symbols.
        store.reset_errors_for_day()
        self.assertEqual(store.errors_today("ETH/USD"), 0)
        self.assertEqual(store.errors_today("BTC/USD"), 0)

    def test_supervisor_evaluate_shakedown_reads_redis_counter(self) -> None:
        """End-to-end: supervisor's shakedown evaluation pulls per-symbol
        error counts from the Redis-backed store."""
        import fakeredis
        from state.position_store import PositionStore

        fake = fakeredis.FakeRedis(decode_responses=True)
        store = PositionStore(redis_client=fake, namespace="ev")
        sup, refs = _build_supervisor(
            position_store=store,
            paper_days_clean=10,
        )
        # Pre-load the counter against the SUPERVISOR's pinned ``now``
        # so the per-day key matches what evaluate_shakedown queries.
        pinned_now = refs["now"]
        store.increment_error("ETH/USDT", now_utc=pinned_now)
        store.increment_error("ETH/USDT", now_utc=pinned_now)
        state = sup.evaluate_shakedown()
        # ETH/USDT had 2 pre-existing errors -> NOT clean -> reset to 0.
        self.assertEqual(
            state.per_symbol["ETH/USDT"].paper_days_clean, 0
        )
        # The counter is now reset for the next day.
        self.assertEqual(
            store.errors_today("ETH/USDT", now_utc=pinned_now), 0
        )


class TestUtcMidnightAutoDailyClose(unittest.TestCase):
    """P0 #8: ``run_loop`` automatically fires ``daily_close`` when the UTC
    clock crosses midnight. Exactly once per crossing.
    """

    def _build_clock_supervisor(
        self, current: List[datetime]
    ) -> Supervisor:
        """Construct a supervisor whose ``now_fn`` returns ``current[0]``
        on every call. The test mutates ``current[0]`` to advance time."""

        def fake_now() -> datetime:
            return current[0]

        td = tempfile.TemporaryDirectory()
        _TMPDIRS.append(td)
        path = Path(td.name) / "midnight.json"
        store = StubPositionStore(daily_pnl=10.0)
        breakers = StubCircuitBreakers(daily_loss_limit_usd=1000.0)
        notifier = StubNotifier()
        config = SupervisorConfig(
            symbols=["ETH/USDT"],
            tick_interval_s=0.0,
            bankroll_usd=10_000.0,
            mode="paper",
            shakedown_min_days=14,
            shakedown_state_path=path,
            risk_pct_per_trade=0.005,
            min_confidence_to_trade=0.6,
        )
        sup = Supervisor(
            config=config,
            exchange=StubExchange(),
            position_store=store,
            circuit_breakers=breakers,
            notifier=notifier,
            model_predict_fn=lambda s, t: ("buy", 0.1),
            sleep_fn=lambda s: None,
            now_fn=fake_now,
        )
        return sup

    def test_auto_close_fires_once_per_midnight_crossing(self) -> None:
        """Two iterations on Day-1, two on Day-2 -> exactly one close."""
        day_1 = datetime(2026, 4, 26, 23, 50, tzinfo=timezone.utc)
        day_2 = datetime(2026, 4, 27, 0, 5, tzinfo=timezone.utc)
        clock = [day_1]
        sup = self._build_clock_supervisor(clock)
        # Run two iterations on Day-1 (no close should fire; baseline gets set).
        sup.run_loop(max_iterations=2)
        self.assertEqual(sup._last_close_date, day_1.date())
        # Now cross midnight.
        clock[0] = day_2
        summary = sup.run_loop(max_iterations=2)
        self.assertEqual(summary["daily_closes_fired"], 1)
        self.assertEqual(sup._last_close_date, day_2.date())

    def test_no_close_fires_when_clock_does_not_cross_midnight(self) -> None:
        within_day = datetime(2026, 4, 26, 12, 0, tzinfo=timezone.utc)
        sup = self._build_clock_supervisor([within_day])
        summary = sup.run_loop(max_iterations=5)
        self.assertEqual(summary["daily_closes_fired"], 0)

    def test_close_fires_only_once_per_crossing(self) -> None:
        """Crossing midnight then doing many iterations same day = 1 close."""
        day_1 = datetime(2026, 4, 26, 23, 59, tzinfo=timezone.utc)
        day_2 = datetime(2026, 4, 27, 0, 1, tzinfo=timezone.utc)
        clock = [day_1]
        sup = self._build_clock_supervisor(clock)
        sup.run_loop(max_iterations=1)  # baseline = day_1
        clock[0] = day_2
        summary = sup.run_loop(max_iterations=10)  # fires once on first iter
        self.assertEqual(summary["daily_closes_fired"], 1)


class TestShakedownFileLock(unittest.TestCase):
    """P0 #2: persisted shakedown state is flock-guarded + atomic-renamed.

    Two booting peers can no longer mutually wipe each other's clean-day
    counters by reading a partial JSON blob mid-write.
    """

    def test_persist_uses_atomic_rename(self) -> None:
        """A partial write to the .tmp sibling must be invisible from the
        canonical path's perspective (the rename is atomic)."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shake.json"
            sup, _ = _build_supervisor(shakedown_path=path)
            # First persist creates the canonical file.
            self.assertTrue(path.exists())
            # The .tmp file must NOT linger after a clean rename.
            self.assertFalse(
                (path.with_suffix(path.suffix + ".tmp")).exists(),
                "tmp file should have been renamed away",
            )
            # Second persist on a slightly different state still leaves
            # only the canonical file behind.
            sup.shakedown_state.get_or_init("ETH/USDT").paper_days_clean = 5
            sup._persist_shakedown(sup.shakedown_state)
            self.assertFalse((path.with_suffix(path.suffix + ".tmp")).exists())
            # Re-read confirms write committed.
            sup2, _ = _build_supervisor(shakedown_path=path)
            self.assertEqual(
                sup2.shakedown_state.per_symbol["ETH/USDT"].paper_days_clean, 5
            )

    def test_concurrent_writers_serialize_via_flock(self) -> None:
        """Two threads writing simultaneously each leave a valid JSON blob
        on disk -- no torn reads, no half-written files. We can't easily
        prove "this is a flock that serialised them" without intercepting
        kernel calls, but we CAN prove the invariant: the final file is
        always parseable."""
        import threading

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "concurrent.json"
            sup, _ = _build_supervisor(shakedown_path=path)
            errors: List[BaseException] = []

            def writer(days: int) -> None:
                try:
                    s = sup.shakedown_state.model_copy(deep=True)
                    s.get_or_init("ETH/USDT").paper_days_clean = days
                    sup._persist_shakedown(s)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer, args=(i,)) for i in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertFalse(errors, f"writer thread errored: {errors!r}")
            # Final file must be parseable.
            blob = path.read_text(encoding="utf-8")
            loaded = ShakedownState.model_validate_json(blob)
            # The exact paper_days_clean value depends on which writer
            # last won the lock -- the contract is just "no torn reads".
            self.assertIn("ETH/USDT", loaded.per_symbol)

    def test_load_path_flock_protects_from_simulated_partial_blob(self) -> None:
        """Even if the file currently contains an unparseable blob (a
        partial write from a peer that crashed), the supervisor must
        recover by reinitialising rather than crashing."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.json"
            # Simulate a peer that crashed mid-write.
            path.write_text("{not json", encoding="utf-8")
            sup, _ = _build_supervisor(shakedown_path=path)
            # Recovered cleanly with fresh state, NOT a crash.
            self.assertEqual(sup.shakedown_state.paper_days_clean, 0)


class TestRunDirOutputSaving(unittest.TestCase):
    """``--log-dir`` flag creates a timestamped subdir + FileHandler."""

    def test_setup_run_dir_returns_none_when_unset(self) -> None:
        from live_supervisor import _setup_run_dir

        self.assertIsNone(_setup_run_dir(None, symbols=["ETH/USD"]))
        self.assertIsNone(_setup_run_dir("", symbols=["ETH/USD"]))

    def test_setup_run_dir_creates_timestamped_subdir(self) -> None:
        from live_supervisor import _setup_run_dir

        with tempfile.TemporaryDirectory() as td:
            now = datetime(2026, 4, 29, 15, 30, 0, tzinfo=timezone.utc)
            run_dir = _setup_run_dir(
                td, symbols=["ETH/USD", "BTC/USD"], now_utc=now
            )
            self.assertIsNotNone(run_dir)
            assert run_dir is not None
            # Subdir name embeds the timestamp + sanitized symbols.
            self.assertIn("2026-04-29T15-30-00Z", run_dir.name)
            self.assertIn("ETH-USD", run_dir.name)
            self.assertIn("BTC-USD", run_dir.name)
            self.assertTrue(run_dir.exists())
            # FileHandler attached -- write a log line, confirm it lands.
            import logging as _logging

            _logging.getLogger("live_supervisor").info("hello-from-test")
            log_path = run_dir / "supervisor.log"
            self.assertTrue(log_path.exists())
            self.assertIn("hello-from-test", log_path.read_text(encoding="utf-8"))
            # Cleanup: remove the FileHandler we added so subsequent tests
            # don't accumulate handlers.
            root = _logging.getLogger()
            for h in list(root.handlers):
                if isinstance(h, _logging.FileHandler) and str(log_path) in str(
                    h.baseFilename
                ):
                    root.removeHandler(h)
                    h.close()


class TestTradeContextSnapshotCapture(unittest.TestCase):
    """Lane E E1 snapshot capture wiring.

    Asserts the supervisor records snapshots at the three lifecycle points
    (signal, fill, breaker) when a :class:`TradeContextStore` is injected,
    and remains a no-op when it isn't.
    """

    def _make_store(self) -> Any:
        # Local import keeps the rest of the module unaffected if the
        # snapshot store gets relocated.
        import fakeredis

        from state.trade_context_store import TradeContextStore

        return TradeContextStore(
            redis_client=fakeredis.FakeRedis(decode_responses=True),
            namespace="test-snap",
        )

    def _build(
        self,
        *,
        with_store: bool = True,
        breakers: Optional[StubCircuitBreakers] = None,
        position_store: Optional[StubPositionStore] = None,
    ) -> Tuple[Supervisor, Dict[str, Any], Any]:
        store = self._make_store() if with_store else None
        sup, refs = _build_supervisor(
            min_confidence=0.5,
            predict_fn=lambda s, t: ("buy", 0.9),
            position_store=position_store,
            circuit_breakers=breakers,
        )
        sup.trade_context_store = store
        return sup, refs, store

    def test_signal_and_paper_fill_snapshots_captured(self) -> None:
        sup, refs, store = self._build()
        # First tick queues a paper fill (no snapshot for fill yet).
        sup.run_once()
        # Second tick drains the paper fill → records a position.
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "allowed")
        self.assertGreaterEqual(len(refs["position_store"].recorded_open), 1)

        # Pull the position_id of the recorded open and verify both
        # signal + fill snapshots exist under the same trade_id.
        pos = refs["position_store"].recorded_open[0]
        snaps = store.get_snapshots(pos.position_id)
        self.assertIn("signal", snaps)
        self.assertIn("fill", snaps)
        self.assertEqual(snaps["signal"].trade_id, pos.position_id)
        self.assertEqual(snaps["fill"].trade_id, pos.position_id)
        # Signal snapshot carries the model confidence.
        self.assertAlmostEqual(snaps["signal"].model_confidence, 0.9)
        # Fill snapshot carries the recorded fill_price.
        self.assertEqual(
            snaps["fill"].risk_metrics_output["fill_price"],
            pos.entry_price,
        )

    def test_breaker_snapshot_captured_on_force_flat(self) -> None:
        # Pre-populate the store with one open position that will be
        # force-flatted by a kill-switch trip.
        existing_pos = _make_position(side="long", symbol="ETH/USDT")
        ps = StubPositionStore(open_positions=[existing_pos])
        breakers = StubCircuitBreakers(kill_switch=True)
        sup, refs, store = self._build(
            with_store=True, breakers=breakers, position_store=ps
        )
        sup.run_once()

        snaps = store.get_snapshots(existing_pos.position_id)
        self.assertIn("breaker", snaps)
        breaker_snap = snaps["breaker"]
        self.assertEqual(breaker_snap.symbol, existing_pos.symbol)
        self.assertEqual(breaker_snap.phase, "breaker")
        # The reason is propagated into the snapshot's notes/breaker_context.
        self.assertIsNotNone(breaker_snap.notes)

    def test_no_store_wired_skips_snapshot_capture(self) -> None:
        """Sanity: without a store, the supervisor must still tick cleanly
        and not raise from the snapshot helpers."""
        sup, refs, _ = self._build(with_store=False)
        # Two ticks (queue + drain) must succeed without any snapshot store.
        sup.run_once()
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "allowed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
