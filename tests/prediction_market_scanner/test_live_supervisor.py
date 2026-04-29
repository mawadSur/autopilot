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


class TestPaperModeSynthesisesFill(unittest.TestCase):
    def test_run_once_paper_synthesizes_fill_when_mode_paper(self) -> None:
        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0)
        sup, refs = _build_supervisor(
            mode="paper",
            position_store=store,
            exchange=exch,
            predict_fn=lambda s, t: ("buy", 0.9),
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "allowed")
        # No live order placed.
        self.assertEqual(len(exch.market_orders), 0)
        # Position recorded synthetically.
        self.assertEqual(len(store.recorded_open), 1)
        synth = store.recorded_open[0]
        expected_price = 2_000.0 * (1.0 + PAPER_SLIPPAGE_BPS / 10_000.0)
        self.assertAlmostEqual(synth.entry_price, expected_price, places=6)
        self.assertEqual(synth.exchange, "coinbase-paper")
        self.assertEqual(synth.side, "long")


class TestLiveLockedFallsBackToPaper(unittest.TestCase):
    def test_run_once_live_blocked_when_shakedown_not_unlocked(self) -> None:
        store = StubPositionStore()
        exch = StubExchange(ticker_mid=2_000.0)
        sup, refs = _build_supervisor(
            mode="live",
            paper_days_clean=3,
            shakedown_min_days=14,
            position_store=store,
            exchange=exch,
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "halted_breaker")
        self.assertEqual(ticks[0].notes, "live_mode_locked")
        # No live order placed.
        self.assertEqual(len(exch.market_orders), 0)
        # But paper-trade fallback recorded a synthetic position.
        self.assertEqual(len(store.recorded_open), 1)
        self.assertEqual(store.recorded_open[0].exchange, "coinbase-paper")


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
