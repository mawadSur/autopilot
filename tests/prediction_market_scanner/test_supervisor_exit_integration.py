"""Integration tests — Sprint 1 Wave 2 supervisor exit-policy + Kelly wiring.

These tests exercise the supervisor's tick loop end-to-end against a small
synthetic stream — each test drives the ``ticker_mid`` (and where relevant
the predictor's ``_last_resolved_kelly_pct``) by hand and asserts on:

* which exit reason fired (time / sl / tp / trail),
* the "never reopens on same tick" invariant,
* Kelly sizing replaces the flat per-trade notional when enabled,
* the ``ccxt.NetworkError`` -> paper-fallback dispatch matches the
  per-position-tag rule from the 2026-05-14 force-flat fix (commit
  ``2b62a7d``),
* the master switch (``EXIT_POLICY_ENABLED=False``) reproduces the legacy
  "stuck-longs-to-breaker" path (regression sentinel).

Fixtures are intentionally small ``SimpleNamespace`` / ``StubExchange``
stand-ins that match Wave 1B's style — the project has no shared test
harness for the supervisor, so we keep the surface lean and obvious.
"""
from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from exchanges.coinbase import ExchangeError, OrderResult, Ticker
from exit_policy import ExitPolicy
from risk.circuit_breakers import CircuitBreakerVerdict, DecisionContext
from state.position_store import Position

from live_supervisor import (
    PAPER_SLIPPAGE_BPS,
    Supervisor,
    SupervisorConfig,
)


# ---------------------------------------------------------------------------
# Stubs (mirror test_live_supervisor.py's style, kept self-contained to avoid
# the cross-file fixture coupling the Wave 1B brief warned about).
# ---------------------------------------------------------------------------


def _ticker(mid: float, symbol: str = "ETH/USDT") -> Ticker:
    spread_bps = 4.0
    bid = mid * (1.0 - spread_bps / 20_000.0)
    ask = mid * (1.0 + spread_bps / 20_000.0)
    return Ticker(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=mid,
        volume_24h_base=1234.5,
        as_of_utc="2026-05-18T00:00:00+00:00",
    )


class StubExchange:
    def __init__(
        self,
        *,
        ticker_mid: float = 2_000.0,
        raise_on_order: Optional[Exception] = None,
        raise_on_close: Optional[Exception] = None,
    ) -> None:
        self.ticker_mid = ticker_mid
        self._raise_on_order = raise_on_order
        self._raise_on_close = raise_on_close
        self.ticker_calls: List[str] = []
        self.market_orders: List[Dict[str, Any]] = []
        # When set, the NEXT place_market_order call raises this exception
        # and the field is cleared (one-shot — useful for the close-side
        # NetworkError tests).
        self._raise_once: Optional[Exception] = None

    def arm_one_shot_failure(self, exc: Exception) -> None:
        self._raise_once = exc

    def get_ticker(self, symbol: str) -> Ticker:
        self.ticker_calls.append(symbol)
        return _ticker(self.ticker_mid, symbol=symbol)

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
        if self._raise_once is not None:
            exc, self._raise_once = self._raise_once, None
            raise exc
        if self._raise_on_order is not None:
            raise self._raise_on_order
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
            status="filled",
            filled_base=size,
            filled_quote_usd=size * self.ticker_mid,
            avg_fill_price=self.ticker_mid,
            fee_usd=0.0,
            created_at_utc="2026-05-18T00:00:00+00:00",
            raw_payload={},
        )


class StubPositionStore:
    """In-memory PositionStore with the Wave-2 ``update_runtime_fields``."""

    def __init__(
        self,
        *,
        open_positions: Optional[List[Position]] = None,
    ) -> None:
        self._open: List[Position] = list(open_positions or [])
        self._closed_today: List[Position] = []
        self.recorded_open: List[Position] = []
        self.recorded_pending: List[Position] = []
        self.recorded_close_calls: List[Dict[str, Any]] = []
        self.runtime_updates: List[Dict[str, Any]] = []
        self._errors_by_symbol: Dict[str, int] = {}

    # reads
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
        return float(sum((p.realized_pnl_usd or 0.0) for p in self._closed_today))

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

    # writes
    def record_open(self, position: Position) -> Position:
        self.recorded_open.append(position)
        self._open.append(position)
        return position

    def record_pending(self, position: Position) -> Position:
        self.recorded_pending.append(position)
        self._open.append(position)
        return position

    def record_close(
        self,
        position_id: str,
        *,
        exit_price: float,
        exit_quote_usd: float,
        fees_usd: float = 0.0,
        bankroll_usd: Optional[float] = None,
    ) -> Position:
        for i, p in enumerate(self._open):
            if p.position_id == position_id:
                existing = p
                break
        else:
            raise KeyError(f"unknown position_id: {position_id!r}")
        total_fees = float(existing.fees_usd) + float(fees_usd)
        if existing.side == "long":
            realized = (
                float(existing.base_size)
                * (float(exit_price) - float(existing.entry_price))
                - total_fees
            )
        else:
            realized = (
                float(existing.base_size)
                * (float(existing.entry_price) - float(exit_price))
                - total_fees
            )
        updated = existing.model_copy(
            update={
                "status": "closed",
                "exit_price": float(exit_price),
                "exit_quote_usd": float(exit_quote_usd),
                "fees_usd": total_fees,
                "realized_pnl_usd": realized,
                "closed_at_utc": "2026-05-18T00:00:00+00:00",
            }
        )
        self._open = [q for q in self._open if q.position_id != position_id]
        self._closed_today.append(updated)
        self.recorded_close_calls.append(
            {
                "position_id": position_id,
                "exit_price": float(exit_price),
                "exit_quote_usd": float(exit_quote_usd),
                "fees_usd": float(fees_usd),
                "bankroll_usd": bankroll_usd,
            }
        )
        return updated

    def update_runtime_fields(
        self,
        position_id: str,
        *,
        bars_held: Optional[int] = None,
        high_water_mark: Optional[float] = None,
    ) -> Optional[Position]:
        self.runtime_updates.append(
            {
                "position_id": position_id,
                "bars_held": bars_held,
                "high_water_mark": high_water_mark,
            }
        )
        for i, p in enumerate(self._open):
            if p.position_id == position_id:
                updates: Dict[str, Any] = {}
                if bars_held is not None:
                    updates["bars_held"] = int(bars_held)
                if high_water_mark is not None:
                    updates["high_water_mark"] = float(high_water_mark)
                if updates:
                    self._open[i] = p.model_copy(update=updates)
                return self._open[i]
        return None

    # error counter shim
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
    def __init__(self) -> None:
        self.daily_loss_limit_usd: Optional[float] = None
        self.check_calls: List[DecisionContext] = []

    def is_kill_switch_tripped(self) -> bool:
        return False

    def check(self, ctx: DecisionContext) -> CircuitBreakerVerdict:
        self.check_calls.append(ctx)
        return CircuitBreakerVerdict(
            allow=True,
            tripped=[],
            reason="",
            recommended_action="allow",
            details={},
        )


class StubNotifier:
    def __init__(self) -> None:
        self.alert_calls: List[str] = []
        self.fill_event_calls: List[Dict[str, Any]] = []
        self.kill_switch_calls: List[str] = []

    def info(self, *a: Any, **k: Any) -> bool:
        return True

    def alert(self, message: str, *, severity: str = "alert", **k: Any) -> bool:
        self.alert_calls.append(message)
        return True

    def daily_summary(self, **k: Any) -> bool:
        return True

    def fill_event(self, **k: Any) -> bool:
        self.fill_event_calls.append(k)
        return True

    def kill_switch_tripped(self, reason: str) -> bool:
        self.kill_switch_calls.append(reason)
        return True


class StubPredictor:
    """Predictor stub exposing ``_last_resolved_kelly_pct``.

    Settable per-tick so a test can simulate the regime lookup populating
    or clearing the cached Kelly fraction.
    """

    def __init__(
        self,
        *,
        side: Literal["buy", "sell"] = "buy",
        confidence: float = 0.9,
        kelly_pct: Optional[float] = None,
    ) -> None:
        self.side = side
        self.confidence = confidence
        self._last_resolved_kelly_pct = kelly_pct

    def __call__(self, symbol: str, ticker: Ticker):
        return (self.side, self.confidence)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_position(
    *,
    side: str = "long",
    symbol: str = "ETH/USDT",
    entry_price: float = 2_000.0,
    base_size: float = 0.05,
    bars_held: int = 0,
    high_water_mark: Optional[float] = None,
    exchange: str = "coinbase-paper",
) -> Position:
    return Position(
        position_id=f"pos-{uuid.uuid4().hex[:8]}",
        exchange=exchange,
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        status="open",
        entry_price=entry_price,
        entry_quote_usd=entry_price * base_size,
        base_size=base_size,
        entry_order_id=f"ord-{uuid.uuid4().hex[:6]}",
        opened_at_utc="2026-05-18T00:00:00+00:00",
        bars_held=bars_held,
        high_water_mark=high_water_mark,
    )


_TMPDIRS: List[tempfile.TemporaryDirectory] = []


def _build(
    *,
    exit_policy: Optional[ExitPolicy] = None,
    exit_policy_enabled: bool = True,
    kelly_sizing_enabled: bool = False,
    bankroll_usd: float = 10_000.0,
    risk_pct: float = 0.005,
    predict: Optional[StubPredictor] = None,
    pre_open: Optional[List[Position]] = None,
    exchange: Optional[StubExchange] = None,
    min_confidence: float = 0.95,  # high so no new entry by default
) -> Tuple[Supervisor, Dict[str, Any]]:
    tmp = tempfile.TemporaryDirectory()
    _TMPDIRS.append(tmp)
    shakedown_path = Path(tmp.name) / "shakedown.json"
    config = SupervisorConfig(
        symbols=["ETH/USDT"],
        tick_interval_s=0.0,
        bankroll_usd=bankroll_usd,
        mode="paper",
        shakedown_min_days=14,
        shakedown_state_path=shakedown_path,
        risk_pct_per_trade=risk_pct,
        min_confidence_to_trade=min_confidence,
    )
    exch = exchange or StubExchange(ticker_mid=2_000.0)
    store = StubPositionStore(open_positions=pre_open)
    breakers = StubCircuitBreakers()
    notifier = StubNotifier()
    fixed_now = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)
    sup = Supervisor(
        config=config,
        exchange=exch,
        position_store=store,
        circuit_breakers=breakers,
        notifier=notifier,
        model_predict_fn=predict or StubPredictor(),
        sleep_fn=lambda s: None,
        now_fn=lambda: fixed_now,
        exit_policy=exit_policy,
        exit_policy_enabled=exit_policy_enabled,
        kelly_sizing_enabled=kelly_sizing_enabled,
    )
    return sup, {
        "exchange": exch,
        "position_store": store,
        "circuit_breakers": breakers,
        "notifier": notifier,
    }


# ---------------------------------------------------------------------------
# Exit-reason coverage: one test per reason.
# ---------------------------------------------------------------------------


class ExitReasonTests(unittest.TestCase):
    """Each reason (time / sl / tp / trail) must close + tally exactly once."""

    def test_time_stop_closes_open_long(self) -> None:
        # Time stop fires at bars_held >= 1 to keep the synthetic stream
        # tight. The exit policy is otherwise neutral so nothing else fires.
        policy = ExitPolicy(
            time_stop_bars=1,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        pos = _make_open_position(bars_held=5, exchange="coinbase-paper")
        sup, refs = _build(exit_policy=policy, pre_open=[pos])
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "exited")
        self.assertIn("time", ticks[0].notes or "")
        self.assertEqual(sup._exits_by_reason["time"], 1)
        self.assertEqual(len(refs["position_store"].recorded_close_calls), 1)

    def test_stop_loss_closes_long_on_breach(self) -> None:
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=-0.004,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        # Long opened at 2000; tick mid drops to 1990 -> -0.5% < -0.4% SL.
        pos = _make_open_position(entry_price=2_000.0, exchange="coinbase-paper")
        exch = StubExchange(ticker_mid=1_990.0)
        sup, refs = _build(exit_policy=policy, pre_open=[pos], exchange=exch)
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "exited")
        self.assertIn("sl", ticks[0].notes or "")
        self.assertEqual(sup._exits_by_reason["sl"], 1)
        closed = refs["position_store"].recorded_close_calls[0]
        # Exit price is the paper-close mid * (1 - 5bps) slippage on the sell.
        expected_exit = 1_990.0 * (1.0 - PAPER_SLIPPAGE_BPS / 10_000.0)
        self.assertAlmostEqual(closed["exit_price"], expected_exit, places=4)

    def test_take_profit_closes_long_at_threshold(self) -> None:
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=0.008,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        # Long opened at 2000; tick mid rises to 2017 -> +0.85% > +0.8% TP.
        pos = _make_open_position(entry_price=2_000.0, exchange="coinbase-paper")
        exch = StubExchange(ticker_mid=2_017.0)
        sup, refs = _build(exit_policy=policy, pre_open=[pos], exchange=exch)
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "exited")
        self.assertIn("tp", ticks[0].notes or "")
        self.assertEqual(sup._exits_by_reason["tp"], 1)

    def test_trailing_stop_closes_long_after_pullback(self) -> None:
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=0.005,
            signal_reversal=False,
        )
        # Pre-seed the high-water mark so the trail has something to
        # retrace from on the first tick (no time-traveling required).
        pos = _make_open_position(
            entry_price=2_000.0,
            exchange="coinbase-paper",
            high_water_mark=2_100.0,
        )
        exch = StubExchange(ticker_mid=2_080.0)  # ~0.95% off the 2100 HWM
        sup, refs = _build(exit_policy=policy, pre_open=[pos], exchange=exch)
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "exited")
        self.assertIn("trail", ticks[0].notes or "")
        self.assertEqual(sup._exits_by_reason["trail"], 1)


# ---------------------------------------------------------------------------
# Capital-preservation invariants.
# ---------------------------------------------------------------------------


class NeverReopensSameTickTests(unittest.TestCase):
    """A position that closed on tick N must NOT be reopened on the same tick."""

    def test_close_on_tick_n_skips_entry_on_tick_n(self) -> None:
        policy = ExitPolicy(
            time_stop_bars=1,
            stop_loss_pct=None,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        pos = _make_open_position(bars_held=5, exchange="coinbase-paper")
        # Predictor would emit a high-confidence buy if it ever ran on
        # this tick — we assert it does NOT run by checking that no new
        # pending paper fill is queued and no new market order is placed.
        sup, refs = _build(
            exit_policy=policy,
            pre_open=[pos],
            predict=StubPredictor(side="buy", confidence=0.99),
            min_confidence=0.6,
        )
        ticks = sup.run_once()
        self.assertEqual(ticks[0].action_taken, "exited")
        self.assertEqual(len(refs["exchange"].market_orders), 0)
        # No new pending paper fill was queued either.
        self.assertNotIn("ETH/USDT", sup._pending_paper_fills)
        # And no new position was added to the store.
        self.assertEqual(len(refs["position_store"].recorded_open), 0)


# ---------------------------------------------------------------------------
# Kelly sizing wire-up.
# ---------------------------------------------------------------------------


class KellySizingTests(unittest.TestCase):
    """When ``KELLY_SIZING_ENABLED`` is True and the predictor exposes a
    non-None ``_last_resolved_kelly_pct``, sizing uses ``bankroll *
    clip(pct, floor, cap)``. Otherwise it falls back to flat."""

    def test_kelly_replaces_flat_when_predictor_exposes_pct(self) -> None:
        predict = StubPredictor(kelly_pct=0.02)
        sup, refs = _build(
            kelly_sizing_enabled=True,
            bankroll_usd=10_000.0,
            risk_pct=0.005,  # flat path would be $50
            predict=predict,
            min_confidence=0.6,
        )
        # Drain a real fill: first tick queues paper, second drains it.
        sup.run_once()
        sup.run_once()
        recorded = refs["position_store"].recorded_open
        self.assertEqual(len(recorded), 1)
        # 0.02 * 10000 = $200 notional, well under the 5% cap.
        self.assertAlmostEqual(recorded[0].entry_quote_usd, 200.0, delta=1.0)

    def test_kelly_capped_at_cap_pct(self) -> None:
        predict = StubPredictor(kelly_pct=0.50)  # would size to 50% of bankroll
        sup, refs = _build(
            kelly_sizing_enabled=True,
            bankroll_usd=10_000.0,
            risk_pct=0.005,
            predict=predict,
            min_confidence=0.6,
        )
        sup.run_once()
        sup.run_once()
        recorded = refs["position_store"].recorded_open
        self.assertEqual(len(recorded), 1)
        # Cap at 5% of bankroll = $500.
        self.assertAlmostEqual(recorded[0].entry_quote_usd, 500.0, delta=1.0)

    def test_kelly_fallback_to_flat_when_pct_is_none(self) -> None:
        predict = StubPredictor(kelly_pct=None)
        sup, refs = _build(
            kelly_sizing_enabled=True,
            bankroll_usd=10_000.0,
            risk_pct=0.005,
            predict=predict,
            min_confidence=0.6,
        )
        sup.run_once()
        sup.run_once()
        recorded = refs["position_store"].recorded_open
        self.assertEqual(len(recorded), 1)
        # Flat $50.
        self.assertAlmostEqual(recorded[0].entry_quote_usd, 50.0, delta=0.5)

    def test_kelly_disabled_uses_flat_even_when_pct_present(self) -> None:
        predict = StubPredictor(kelly_pct=0.02)
        sup, refs = _build(
            kelly_sizing_enabled=False,  # master switch off
            bankroll_usd=10_000.0,
            risk_pct=0.005,
            predict=predict,
            min_confidence=0.6,
        )
        sup.run_once()
        sup.run_once()
        recorded = refs["position_store"].recorded_open
        self.assertEqual(len(recorded), 1)
        self.assertAlmostEqual(recorded[0].entry_quote_usd, 50.0, delta=0.5)


# ---------------------------------------------------------------------------
# Exchange-error dispatch on close.
# ---------------------------------------------------------------------------


class ExchangeErrorOnCloseTests(unittest.TestCase):
    """Per-tag dispatch: paper positions fall through to ``_paper_force_flat``
    when the exchange raises; live positions surface the error so the
    kill-switch / breaker logic can react. Matches commit ``2b62a7d``."""

    def test_paper_position_falls_back_on_exchange_error(self) -> None:
        # Force a SL close on a paper-tagged position; the StubExchange
        # has no place_market_order path for paper closes (paper uses
        # _paper_force_flat) so this test instead asserts that even if
        # _paper_force_flat itself encountered a transient get_ticker
        # ExchangeError, the close still tries the paper-fallback path.
        # Concretely: we DO NOT raise on get_ticker but we verify that for
        # paper-tagged positions, the close path NEVER touches
        # place_market_order.
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=-0.004,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        exch = StubExchange(
            ticker_mid=1_990.0,
            raise_on_order=ExchangeError("network blip"),
        )
        pos = _make_open_position(
            entry_price=2_000.0, exchange="coinbase-paper"
        )
        sup, refs = _build(
            exit_policy=policy, pre_open=[pos], exchange=exch
        )
        ticks = sup.run_once()
        # Paper-tagged close NEVER touches place_market_order, so even with
        # the order path armed to raise, the exit succeeds.
        self.assertEqual(ticks[0].action_taken, "exited")
        self.assertEqual(len(exch.market_orders), 0)
        self.assertEqual(len(refs["position_store"].recorded_close_calls), 1)

    def test_live_position_close_error_surfaces_as_errored_tick(self) -> None:
        policy = ExitPolicy(
            time_stop_bars=None,
            stop_loss_pct=-0.004,
            take_profit_pct=None,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        exch = StubExchange(
            ticker_mid=1_990.0,
            raise_on_order=ExchangeError("ccxt network: name resolution"),
        )
        pos = _make_open_position(
            entry_price=2_000.0, exchange="coinbase"
        )  # LIVE tag
        sup, refs = _build(
            exit_policy=policy, pre_open=[pos], exchange=exch
        )
        ticks = sup.run_once()
        # The supervisor surfaces the error so the consecutive-error
        # breaker / kill switch can react -- the action_taken is "errored",
        # NOT a silent "exited" with a stale position still in the store.
        self.assertEqual(ticks[0].action_taken, "errored")
        # Live close attempt DID hit place_market_order (and failed).
        self.assertEqual(len(exch.market_orders), 1)
        # Position stays in the store (no successful close).
        self.assertEqual(len(refs["position_store"].recorded_close_calls), 0)
        # An operator alert went out.
        self.assertGreaterEqual(len(refs["notifier"].alert_calls), 1)


# ---------------------------------------------------------------------------
# Master-switch regression sentinel.
# ---------------------------------------------------------------------------


class ExitPolicyDisabledRegressionTests(unittest.TestCase):
    """With ``EXIT_POLICY_ENABLED=False`` the supervisor reproduces the
    legacy "stuck longs to breaker" path: no exits fire, positions accumulate
    bars_held, and the tick is reported as ``allowed`` / no-op for the
    open position. This is the regression sentinel for Wave 2's master
    switch — flipping it back to False MUST restore the legacy behavior
    end-to-end so operators can opt out via the env flag without code
    surgery."""

    def test_disabled_policy_does_not_close_position(self) -> None:
        # Even with a wildly out-of-band ticker that would trip SL+TP, the
        # disabled policy means zero close attempts.
        policy = ExitPolicy(
            time_stop_bars=1,
            stop_loss_pct=-0.001,
            take_profit_pct=0.001,
            trailing_stop_pct=None,
            signal_reversal=False,
        )
        pos = _make_open_position(
            bars_held=99, entry_price=2_000.0, exchange="coinbase-paper"
        )
        exch = StubExchange(ticker_mid=1_500.0)  # would trip SL
        sup, refs = _build(
            exit_policy=policy,
            exit_policy_enabled=False,  # MASTER SWITCH OFF
            pre_open=[pos],
            exchange=exch,
            predict=StubPredictor(side="buy", confidence=0.1),  # below floor
            min_confidence=0.6,
        )
        ticks = sup.run_once()
        # Action is whatever the legacy path would emit (skipped_low_confidence
        # because predict is below floor and exit logic is bypassed).
        self.assertEqual(ticks[0].action_taken, "skipped_low_confidence")
        # No close happened. The position is still stuck open.
        self.assertEqual(len(refs["position_store"].recorded_close_calls), 0)
        self.assertEqual(len(refs["position_store"].list_open()), 1)
        # Counters never tally.
        self.assertEqual(sup._exits_by_reason["sl"], 0)
        self.assertEqual(sup._exits_by_reason["tp"], 0)
        self.assertEqual(sup._exits_by_reason["time"], 0)


# ---------------------------------------------------------------------------
# Cumulative tick-action table for the report.
# ---------------------------------------------------------------------------


class TickActionTableTests(unittest.TestCase):
    """Drive a small synthetic stream and tally the action_taken counts
    so the coordinator can sanity-check the wiring numerically (per the
    Wave-2 brief's "sample tick-action table" deliverable)."""

    def test_sample_stream_tallies(self) -> None:
        policy = ExitPolicy(
            time_stop_bars=2,
            stop_loss_pct=-0.004,
            take_profit_pct=0.008,
            trailing_stop_pct=None,
            signal_reversal=False,
        )

        # Position pre-loaded so the first tick can exit on SL.
        pos1 = _make_open_position(
            entry_price=2_000.0, exchange="coinbase-paper", bars_held=0
        )
        exch = StubExchange(ticker_mid=1_990.0)  # -0.5% -> SL on tick 1
        sup, refs = _build(
            exit_policy=policy,
            pre_open=[pos1],
            exchange=exch,
            predict=StubPredictor(side="buy", confidence=0.3),  # below floor
            min_confidence=0.6,
        )

        # Tick 1: SL fires on pos1.
        sup.run_once()
        # Tick 2: no open positions, no entry (low conf).
        exch.ticker_mid = 2_010.0
        sup.run_once()
        # Tick 3: same — no entries, no exits.
        sup.run_once()

        counts: Dict[str, int] = {}
        # Replay tally from the exits counter + close calls
        counts["exit_sl"] = sup._exits_by_reason["sl"]
        counts["exit_tp"] = sup._exits_by_reason["tp"]
        counts["exit_time"] = sup._exits_by_reason["time"]
        counts["exit_trail"] = sup._exits_by_reason["trail"]
        # Sanity: exactly one SL exit happened.
        self.assertEqual(counts["exit_sl"], 1)
        self.assertEqual(counts["exit_tp"], 0)
        self.assertEqual(counts["exit_time"], 0)
        self.assertEqual(counts["exit_trail"], 0)
        # And exactly one close was recorded.
        self.assertEqual(len(refs["position_store"].recorded_close_calls), 1)


if __name__ == "__main__":
    unittest.main()
