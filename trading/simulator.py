from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from utils import fmt_money, fmt_pct


# ----------------------------
# Data containers
# ----------------------------


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    timestamp: Optional[Any] = None
    atr: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    regime: Optional[int] = None  # +1 up, -1 down, 0 flat


@dataclass
class SimulationConfig:
    start_capital: float = 10_000.0
    fee_pct: float = 0.0008
    tp_pct: float = 0.005
    sl_pct: float = 0.0025
    slippage_pct: float = 0.0
    cooldown: int = 0
    use_atr_stops: bool = False
    atr_tp_mult: float = 1.8
    atr_sl_mult: float = 1.0
    dynamic_sizing: bool = False
    max_risk_per_trade: float = 0.02
    leverage: float = 1.0
    use_regime_filter: bool = False
    min_atr_pct: float = 0.0
    allow_shorts: bool = True
    trail_stop_long: Optional[float] = None
    trail_stop_short: Optional[float] = None
    breakeven_trigger_long: Optional[float] = None
    breakeven_trigger_short: Optional[float] = None
    record_trades: bool = True
    keep_equity_curve: bool = True


# ----------------------------
# Reporting helper
# ----------------------------


def print_portfolio_report(report: dict, currency: str = "$") -> None:
    m = (report or {}).get("metrics", {}) or {}
    p = (report or {}).get("portfolio", {}) or {}
    n = int(m.get("n", 0))
    start = p.get("start_capital", 0.0)
    end = p.get("end_equity", None)
    trades = int(p.get("trades", 0))
    wins = int(p.get("wins", 0))
    losses = int(p.get("losses", 0))
    mdd = p.get("max_drawdown", None)
    avg_hold_bars = p.get("avg_hold_bars", None)
    trades_per_day = p.get("trades_per_day", None)
    profit_factor = p.get("profit_factor", None)
    avg_win_pct = p.get("avg_win_pct", None)
    avg_loss_pct = p.get("avg_loss_pct", None)
    exposure = p.get("exposure", None)
    multiple = float(end) / float(start) if start not in (None, 0) and end is not None else None
    print("\n=== PORTFOLIO MODE — SUMMARY ===")
    print(f"Bars processed : {n:,}")
    print(f"Trades         : {trades:,}  (wins {wins}, losses {losses}, win rate {wins/max(1,trades):.2%})")
    print(f"Start capital  : {fmt_money(start, currency)}")
    print(f"End equity     : {fmt_money(end, currency)}")
    if multiple is not None and np.isfinite(multiple):
        print(f"Return         : {fmt_pct(multiple-1.0)}  (×{multiple:.2f})")
    else:
        print("Return         : —")
    if mdd is not None:
        print(f"Max drawdown   : {fmt_pct(mdd)}")
    if avg_hold_bars is not None:
        print(f"Avg hold time  : {float(avg_hold_bars):.2f} bars")
    if trades_per_day is not None:
        print(f"Trades / day   : {float(trades_per_day):.2f}")
    if profit_factor is not None:
        if isinstance(profit_factor, float) and not np.isfinite(profit_factor):
            print("Profit factor : —")
        else:
            print(f"Profit factor : {float(profit_factor):.2f}")
    if avg_win_pct is not None:
        print(f"Avg win %      : {fmt_pct(avg_win_pct)}")
    if avg_loss_pct is not None:
        print(f"Avg loss %     : {fmt_pct(avg_loss_pct)}")
    if exposure is not None:
        print(f"Exposure       : {fmt_pct(exposure)}")
    print("")


# ----------------------------
# Core simulator
# ----------------------------


class PortfolioSimulator:
    """Single source of truth for trade simulation.

    - Works in streaming (step-by-step) and batch modes.
    - Accepts raw signals (-1/0/+1) or class-style (0/1/2).
    - Supports TP/SL, ATR stops, cooldown, dynamic sizing, trailing stops,
      regime filter, min-ATR filter, equity curve and trade blotter.
    """

    def __init__(
        self,
        config: SimulationConfig,
        signal_generator: Optional[Callable[..., Union[int, Sequence[int]]]] = None,
    ) -> None:
        self.config = config
        self.signal_generator = signal_generator
        self.reset()

    # Public API -----------------------------------------------------
    def reset(self) -> None:
        self.cash = float(self.config.start_capital)
        self.pos = 0  # -1 short, 0 flat, +1 long
        self.entry_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.position_size: float = 0.0  # notional exposure (pre-leverage)
        self.trail_price: Optional[float] = None
        self.entry_index: Optional[int] = None

        self.cooldown_remaining = 0
        self.pending_signal: Optional[int] = None

        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.hold_bars_sum = 0.0
        self.hold_bars_count = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.win_sum = 0.0
        self.win_count = 0
        self.loss_sum = 0.0
        self.loss_count = 0

        self.peak_equity = float(self.config.start_capital)
        self.max_drawdown = 0.0

        self.n_bars = 0
        self.pos_bars = 0
        self.last_equity = float(self.config.start_capital)

        self._equity_curve: List[float] = [] if self.config.keep_equity_curve else []
        self._equity_times: List[Any] = [] if self.config.keep_equity_curve else []
        self.trade_log: List[dict] = [] if self.config.record_trades else []

    def step(self, bar: Bar, signal: Optional[int] = None, *, regime: Optional[int] = None) -> None:
        """Consume one bar in streaming mode.

        The *previous* pending signal is executed on this bar's open; the
        provided ``signal`` becomes the pending signal for the *next* bar.
        """
        self.n_bars += 1
        self.bar_index = self.n_bars - 1

        # If a regime value is passed separately, override bar.regime
        if regime is not None:
            bar.regime = regime

        # Track time-in-position for exposure
        if self.pos != 0:
            self.pos_bars += 1

        current_signal = self._normalize_signal(signal)
        exec_signal = self._normalize_signal(self.pending_signal)
        exec_signal = self._apply_filters(exec_signal, bar)

        if self.pos == 0:
            self._handle_flat(exec_signal, bar)
        else:
            self._handle_open(exec_signal, bar)

        # Update equity curve after processing the bar
        self._mark_equity(bar.close, bar.timestamp)

        # Store pending for next bar
        self.pending_signal = current_signal

    def finalize(self, last_close: Optional[float] = None, timestamp: Optional[Any] = None) -> None:
        """Close open position at provided price (typically last close)."""
        if self.pos != 0 and last_close is not None:
            bar = Bar(open=last_close, high=last_close, low=last_close, close=last_close, timestamp=timestamp)
            self._force_close(bar)
            self._mark_equity(last_close, timestamp)

    def run_batch(
        self,
        bars: Sequence[Bar],
        signals: Optional[Sequence[int]] = None,
    ) -> Tuple[dict, pd.DataFrame]:
        """Batch mode: iterate through bars once using the shared streaming logic."""
        if signals is None:
            if self.signal_generator is None:
                raise ValueError("signals not provided and no signal_generator attached")
            signals = list(self.signal_generator(bars))  # type: ignore[arg-type]
        if len(bars) == 0:
            raise ValueError("run_batch received no bars")
        if signals is not None and len(signals) != len(bars):
            raise ValueError("signals and bars must have identical length")

        for bar, sig in zip(bars, signals):
            self.step(bar, sig)

        self.finalize(bars[-1].close, bars[-1].timestamp)
        curve = self.equity_curve()
        return self.report(), curve

    def report(self) -> dict:
        avg_hold = self.hold_bars_sum / max(1, self.hold_bars_count)
        days = self.n_bars / 1440.0 if self.n_bars else 0.0
        trades_per_day = self.trades / max(1e-12, days) if days else 0.0
        profit_factor = (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else (
            float("inf") if self.gross_profit > 0 else 0.0
        )
        avg_win = self.win_sum / max(1, self.win_count)
        avg_loss = self.loss_sum / max(1, self.loss_count)
        exposure = self.pos_bars / max(1, self.n_bars) if self.n_bars else 0.0
        return {
            "metrics": {"n": int(self.n_bars)},
            "portfolio": {
                "start_capital": float(self.config.start_capital),
                "end_equity": float(self.last_equity),
                "return": float(self.last_equity / max(1e-12, self.config.start_capital) - 1.0),
                "max_drawdown": float(self.max_drawdown),
                "trades": int(self.trades),
                "wins": int(self.wins),
                "losses": int(self.losses),
                "avg_hold_bars": float(avg_hold),
                "trades_per_day": float(trades_per_day),
                "profit_factor": float(profit_factor),
                "avg_win_pct": float(avg_win),
                "avg_loss_pct": float(avg_loss),
                "exposure": float(exposure),
            },
        }

    def equity_curve(self) -> pd.DataFrame:
        if not self.config.keep_equity_curve:
            return pd.DataFrame(columns=["equity", "timestamp"])
        return pd.DataFrame({
            "equity": np.asarray(self._equity_curve, dtype=float),
            "timestamp": self._equity_times,
        })

    # Internal helpers ------------------------------------------------
    def _normalize_signal(self, sig: Optional[int]) -> int:
        if sig is None:
            return 0
        if sig in (-1, 0, 1):
            return int(sig)
        if sig == 2:
            return 1
        if sig == 0:
            return -1
        if sig == 1:
            return 0
        raise ValueError(f"Unsupported signal value: {sig}")

    def _apply_filters(self, sig: int, bar: Bar) -> int:
        sig_out = sig
        if not self.config.allow_shorts and sig_out < 0:
            sig_out = 0

        # Regime filter (priority over ATR filter)
        if self.config.use_regime_filter:
            regime = bar.regime
            if regime is None and bar.ema_fast is not None and bar.ema_slow is not None:
                if bar.ema_fast > bar.ema_slow * 1.0005:
                    regime = 1
                elif bar.ema_fast < bar.ema_slow * 0.9995:
                    regime = -1
                else:
                    regime = 0
            if regime is not None:
                if sig_out > 0 and regime < 0:
                    sig_out = 0
                if sig_out < 0 and regime > 0:
                    sig_out = 0

        # Min-ATR filter
        if self.config.min_atr_pct and bar.atr is not None and bar.close:
            if (float(bar.atr) / float(bar.close)) < float(self.config.min_atr_pct):
                sig_out = 0

        return sig_out

    def _handle_flat(self, sig: int, bar: Bar) -> None:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return

        if sig == 0:
            return

        if sig > 0:
            self._enter_position(+1, bar)
        elif sig < 0:
            self._enter_position(-1, bar)

    def _handle_open(self, sig: int, bar: Bar) -> None:
        exit_price: Optional[float] = None
        reason = ""

        if self.pos == +1:
            # Trailing & breakeven checks (optional)
            if self.config.trail_stop_long is not None:
                self.trail_price = max(self.trail_price or -np.inf, bar.high, bar.close)
                if bar.close < self.trail_price * (1 - self.config.trail_stop_long):
                    exit_price = bar.close
                    reason = "trail_stop"
            if exit_price is None and self.config.breakeven_trigger_long is not None:
                if (self.trail_price or bar.high) > self.entry_price * self.config.breakeven_trigger_long and bar.close <= self.entry_price:
                    exit_price = bar.close
                    reason = "breakeven"

            if exit_price is None:
                if bar.low <= self.sl_price <= bar.high:
                    exit_price = self.sl_price
                    reason = "stop"
                elif bar.high >= self.tp_price:
                    exit_price = self.tp_price
                    reason = "target"
                elif sig < 0:
                    exit_price = bar.open
                    reason = "reverse"
        else:  # short
            if self.config.trail_stop_short is not None:
                self.trail_price = min(self.trail_price or np.inf, bar.low, bar.close)
                if bar.close > self.trail_price * (1 + self.config.trail_stop_short):
                    exit_price = bar.close
                    reason = "trail_stop"
            if exit_price is None and self.config.breakeven_trigger_short is not None:
                if (self.trail_price or bar.low) < self.entry_price * self.config.breakeven_trigger_short and bar.close >= self.entry_price:
                    exit_price = bar.close
                    reason = "breakeven"

            if exit_price is None:
                if bar.high >= self.sl_price:
                    exit_price = self.sl_price
                    reason = "stop"
                elif bar.low <= self.tp_price:
                    exit_price = self.tp_price
                    reason = "target"
                elif sig > 0:
                    exit_price = bar.open
                    reason = "reverse"

        if exit_price is not None:
            self._exit_position(exit_price, bar, reason)
            return

        # Still in trade — update trailing reference and continue
        if self.pos == +1:
            self.trail_price = max(self.trail_price or -np.inf, bar.high, bar.close)
        else:
            self.trail_price = min(self.trail_price or np.inf, bar.low, bar.close)

    def _enter_position(self, side: int, bar: Bar) -> None:
        price = float(bar.open)
        if side > 0:
            entry = price * (1.0 + self.config.slippage_pct)
        else:
            entry = price * (1.0 - self.config.slippage_pct)

        tp, sl = self._compute_targets(entry, bar.atr, side)
        position_notional = self._compute_position_notional(entry, sl)
        fee_in = self.config.fee_pct * position_notional * self.config.leverage
        self.cash -= fee_in

        self.pos = side
        self.entry_price = entry
        self.tp_price = tp
        self.sl_price = sl
        self.position_size = position_notional
        self.trail_price = entry
        self.entry_index = self.bar_index
        self.trades += 1

        if self.config.record_trades:
            self.trade_log.append({
                "timestamp": bar.timestamp,
                "side": "long" if side > 0 else "short",
                "action": "enter",
                "price": entry,
                "tp": tp,
                "sl": sl,
            })

    def _exit_position(self, exit_price: float, bar: Bar, reason: str) -> None:
        if self.entry_price is None:
            return

        if self.pos > 0:
            exit_exec = exit_price * (1.0 - self.config.slippage_pct)
            ret = (exit_exec / self.entry_price) - 1.0
        else:
            exit_exec = exit_price * (1.0 + self.config.slippage_pct)
            ret = (self.entry_price / exit_exec) - 1.0

        pnl = ret * self.position_size * self.config.leverage
        cash_before = self.cash
        self.cash += pnl
        fee_out = self.config.fee_pct * self.position_size * self.config.leverage
        self.cash -= fee_out

        hold_bars = (self.bar_index - (self.entry_index or self.bar_index) + 1)
        self.hold_bars_sum += hold_bars
        self.hold_bars_count += 1

        win = pnl >= 0
        self.wins += int(win)
        self.losses += int(not win)
        self.gross_profit += max(0.0, pnl)
        self.gross_loss += max(0.0, -pnl)
        self.win_sum += ret if win else 0.0
        self.loss_sum += ret if not win else 0.0
        self.win_count += int(win)
        self.loss_count += int(not win)

        if self.config.record_trades:
            self.trade_log.append({
                "timestamp": bar.timestamp,
                "side": "long" if self.pos > 0 else "short",
                "action": "exit",
                "price": exit_exec,
                "pnl": pnl,
                "ret": ret,
                "win": bool(win),
                "reason": reason,
                "bars_held": hold_bars,
                "equity": self.cash,
            })

        # Reset position
        self.pos = 0
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
        self.position_size = 0.0
        self.trail_price = None
        self.entry_index = None
        self.cooldown_remaining = max(0, int(self.config.cooldown))

    def _force_close(self, bar: Bar) -> None:
        self._exit_position(bar.close, bar, reason="close_end")

    def _compute_targets(self, entry: float, atr: Optional[float], side: int) -> Tuple[float, float]:
        if self.config.use_atr_stops and atr is not None and np.isfinite(atr):
            a = float(max(1e-12, atr))
            if side > 0:
                tp = entry + self.config.atr_tp_mult * a
                sl = entry - self.config.atr_sl_mult * a
            else:
                tp = entry - self.config.atr_tp_mult * a
                sl = entry + self.config.atr_sl_mult * a
        else:
            if side > 0:
                tp = entry * (1.0 + self.config.tp_pct)
                sl = entry * (1.0 - self.config.sl_pct)
            else:
                tp = entry * (1.0 - self.config.tp_pct)
                sl = entry * (1.0 + self.config.sl_pct)
        return tp, sl

    def _compute_position_notional(self, entry: float, stop_price: float) -> float:
        if not self.config.dynamic_sizing:
            return self.cash * self.config.leverage

        stop_dist = abs(entry - stop_price)
        risk_amount = self.cash * max(0.0, self.config.max_risk_per_trade)
        if stop_dist <= 0:
            return self.cash * self.config.leverage
        qty = risk_amount / stop_dist
        notional = qty * entry
        return max(0.0, notional)

    def _mark_equity(self, close_price: float, timestamp: Optional[Any]) -> float:
        if self.pos == 0 or self.entry_price is None:
            equity = self.cash
        else:
            if self.pos > 0:
                ret = (close_price / self.entry_price) - 1.0
            else:
                ret = (self.entry_price / close_price) - 1.0
            equity = self.cash + ret * self.position_size * self.config.leverage

        self.last_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / max(1e-12, self.peak_equity)
        self.max_drawdown = max(self.max_drawdown, dd)

        if self.config.keep_equity_curve:
            self._equity_curve.append(equity)
            self._equity_times.append(timestamp)
        return equity


# ----------------------------
# Compatibility wrappers
# ----------------------------


def _bars_from_arrays(opens, highs, lows, closes, atr=None, regime=None) -> List[Bar]:
    n = len(closes)
    bars: List[Bar] = []
    atr_arr = atr if atr is not None else [None] * n
    reg_arr = regime if regime is not None else [None] * n
    for i in range(n):
        bars.append(
            Bar(
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                atr=None if atr_arr is None else (None if len(atr_arr) <= i else (None if atr_arr[i] is None else float(atr_arr[i]))),
                regime=None if reg_arr is None else (None if len(reg_arr) <= i else reg_arr[i]),
            )
        )
    return bars


def simulate_trades_with_tp_sl(
    opens,
    highs,
    lows,
    closes,
    classes,
    *,
    start_capital,
    fee_pct=0.0008,
    tp_pct=0.005,
    sl_pct=0.005,
    atr: Optional[np.ndarray] = None,
    atr_tp_mult: Optional[float] = None,
    atr_sl_mult: Optional[float] = None,
    cooldown: int = 0,
    slippage_pct: float = 0.0,
    dynamic_sizing: bool = False,
    max_risk_per_trade: float = 0.02,
    leverage: float = 1.0,
    min_atr_pct: float = 0.0,
    allow_shorts: bool = True,
    use_regime_filter: bool = False,
    regime: Optional[Sequence[int]] = None,
) -> Tuple[dict, pd.DataFrame]:
    cfg = SimulationConfig(
        start_capital=float(start_capital),
        fee_pct=float(fee_pct),
        tp_pct=float(tp_pct),
        sl_pct=float(sl_pct),
        slippage_pct=float(slippage_pct),
        cooldown=int(cooldown),
        use_atr_stops=atr_tp_mult is not None and atr_sl_mult is not None,
        atr_tp_mult=float(atr_tp_mult or 0.0) if atr_tp_mult is not None else 0.0,
        atr_sl_mult=float(atr_sl_mult or 0.0) if atr_sl_mult is not None else 0.0,
        dynamic_sizing=bool(dynamic_sizing),
        max_risk_per_trade=float(max_risk_per_trade),
        leverage=float(leverage),
        min_atr_pct=float(min_atr_pct),
        allow_shorts=bool(allow_shorts),
        use_regime_filter=bool(use_regime_filter),
    )
    bars = _bars_from_arrays(opens, highs, lows, closes, atr=atr, regime=regime)
    sim = PortfolioSimulator(cfg)
    report, curve = sim.run_batch(bars, classes)
    return report, curve


def simulate_trades_with_tp_sl_more_aggressive(
    opens,
    highs,
    lows,
    closes,
    classes,
    *,
    start_capital,
    fee_pct=0.0008,
    tp_pct=0.005,
    sl_pct=0.005,
    atr: Optional[np.ndarray] = None,
    atr_tp_mult: Optional[float] = None,
    atr_sl_mult: Optional[float] = None,
    cooldown: int = 0,
    slippage_pct: float = 0.0,
    dynamic_sizing: bool = False,
    max_risk_per_trade: float = 0.02,
    leverage: float = 1.0,
    trail_stop_long: float = 0.0002,
    trail_stop_short: float = 0.0002,
    breakeven_trigger_long: float = 1.0005,
    breakeven_trigger_short: float = 0.9995,
    min_atr_pct: float = 0.0,
    allow_shorts: bool = True,
    use_regime_filter: bool = False,
    regime: Optional[Sequence[int]] = None,
) -> Tuple[dict, pd.DataFrame]:
    cfg = SimulationConfig(
        start_capital=float(start_capital),
        fee_pct=float(fee_pct),
        tp_pct=float(tp_pct),
        sl_pct=float(sl_pct),
        slippage_pct=float(slippage_pct),
        cooldown=int(cooldown),
        use_atr_stops=atr_tp_mult is not None and atr_sl_mult is not None,
        atr_tp_mult=float(atr_tp_mult or 0.0) if atr_tp_mult is not None else 0.0,
        atr_sl_mult=float(atr_sl_mult or 0.0) if atr_sl_mult is not None else 0.0,
        dynamic_sizing=bool(dynamic_sizing),
        max_risk_per_trade=float(max_risk_per_trade),
        leverage=float(leverage),
        trail_stop_long=float(trail_stop_long),
        trail_stop_short=float(trail_stop_short),
        breakeven_trigger_long=float(breakeven_trigger_long),
        breakeven_trigger_short=float(breakeven_trigger_short),
        min_atr_pct=float(min_atr_pct),
        allow_shorts=bool(allow_shorts),
        use_regime_filter=bool(use_regime_filter),
    )
    bars = _bars_from_arrays(opens, highs, lows, closes, atr=atr, regime=regime)
    sim = PortfolioSimulator(cfg)
    report, curve = sim.run_batch(bars, classes)
    return report, curve
