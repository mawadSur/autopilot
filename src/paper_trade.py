# paper_trade.py
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dotenv import load_dotenv
from binance.exceptions import BinanceAPIException

from utils import (
    SignalGenerator,
    get_binance_client,
    prefill_history,
    fetch_latest_bar,
    _get_bool,
)

# -----------------------------
# Config (env-driven)
# -----------------------------
load_dotenv()

SYMBOL = os.getenv("TRADE_SYMBOL", "ETHUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
POLL_MS = int(os.getenv("POLL_MS", "1000"))  # chart refresh interval (ms)

# risk/fees
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.5")) / 100.0
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.75")) / 100.0
FEE_PCT = float(os.getenv("FEE_PCT", "0.075")) / 100.0  # per side

TESTNET = _get_bool("TESTNET", True)

if not ENDPOINT_NAME:
    raise RuntimeError("ENDPOINT_NAME is not set in the environment.")

# -----------------------------
# Small helpers
# -----------------------------
def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)

# -----------------------------
# State containers (mutable)
# -----------------------------
class TradeState:
    def __init__(self):
        self.position_open: bool = False
        self.entry_price: Optional[float] = None
        self.tp_price: Optional[float] = None
        self.sl_price: Optional[float] = None
        self.trade_count: int = 0
        self.pnl_realized: float = 0.0  # fraction (e.g., 0.012 = +1.2%)
        self.last_open_time: int = -1

        # plotting data
        self.times: List[datetime] = []
        self.prices: List[float] = []
        self.buy_times: List[datetime] = []
        self.buy_prices: List[float] = []
        self.sell_times: List[datetime] = []
        self.sell_prices: List[float] = []

# -----------------------------
# Init core services
# -----------------------------
client = get_binance_client(testnet=TESTNET)
sig = SignalGenerator(endpoint_name=ENDPOINT_NAME)
prefill_history(sig, client, SYMBOL, INTERVAL)
state = TradeState()

# -----------------------------
# Matplotlib setup
# -----------------------------
plt.style.use("default")
fig, ax = plt.subplots(figsize=(12, 6))
line_price, = ax.plot([], [], lw=1.5)  # price line
scat_buys = ax.scatter([], [], marker="^", s=60)   # will be recolored each frame
scat_sells = ax.scatter([], [], marker="v", s=60)

tp_line = ax.axhline(y=0, color="gray", linestyle="--", lw=1, alpha=0.5)  # reused/hidden when flat
sl_line = ax.axhline(y=0, color="gray", linestyle="--", lw=1, alpha=0.5)
tp_line.set_visible(False)
sl_line.set_visible(False)

title = ax.set_title(f"{SYMBOL} paper trading (testnet={TESTNET})")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Price (USDT)")

fig.tight_layout()

# -----------------------------
# Core trading step
# -----------------------------
def trading_step() -> Tuple[float, Optional[float], Optional[int], int]:
    """
    Returns: price, confidence (or None), signal (or None), open_time
    """
    bar, open_time = fetch_latest_bar(client, SYMBOL, INTERVAL)
    # Update plotting series once per candle (for a clean OHLC step view)
    if not state.times or open_time != state.last_open_time:
        state.times.append(ms_to_dt(open_time))
        state.prices.append(bar["close"])
        state.last_open_time = open_time
    else:
        # same candle: update last price so plot reflects latest tick
        state.prices[-1] = bar["close"]

    res = sig.get_signal(bar)  # appends/updates internal history
    price = bar["close"]
    conf = res.get("confidence")
    signal = res.get("signal")

    # Entry/exit logic ONLY when a new candle closes/opens (debounce intra-bar)
    if open_time == state.last_open_time and len(state.times) > 1:
        # Still same candle; we only update the chart, not trading decisions.
        pass

    # On NEW candle, make decisions using the *previous* close (just updated by SignalGenerator)
    if len(state.times) >= 1 and open_time == state.last_open_time:
        # decisions below will still run once per new bar because of SignalGenerator debouncing
        if not state.position_open:
            if signal == 1 and conf is not None and conf >= sig.threshold:
                # open long
                state.position_open = True
                state.entry_price = price
                state.tp_price = state.entry_price * (1.0 + TAKE_PROFIT_PCT)
                state.sl_price = state.entry_price * (1.0 - STOP_LOSS_PCT)
                state.trade_count += 1
                state.buy_times.append(state.times[-1])
                state.buy_prices.append(price)
        else:
            # manage exits with TP/SL
            assert state.entry_price is not None
            assert state.tp_price is not None
            assert state.sl_price is not None
            hit_tp = price >= state.tp_price
            hit_sl = price <= state.sl_price
            if hit_tp or hit_sl:
                gross_ret = (price / state.entry_price) - 1.0
                net_ret = gross_ret - (2.0 * FEE_PCT)
                state.pnl_realized += net_ret

                # close position
                state.position_open = False
                state.sell_times.append(state.times[-1])
                state.sell_prices.append(price)

                # clear targets
                state.entry_price = None
                state.tp_price = None
                state.sl_price = None

    return price, (None if conf is None else float(conf)), (None if signal is None else int(signal)), open_time

# -----------------------------
# Animation callback
# -----------------------------
def update(_frame):
    try:
        price, conf, signal, _ot = trading_step()
    except BinanceAPIException as e:
        print(f"BinanceAPIException {e.status_code}: {e.message}")
        time.sleep(2)
        return []
    except Exception as e:
        print(f"update() error: {e}")
        time.sleep(2)
        return []

    # update price line
    line_price.set_data(state.times, state.prices)
    ax.relim()
    ax.autoscale_view()

    # update trade markers
    scat_buys.set_offsets(list(zip(state.buy_times, state.buy_prices)))
    scat_buys.set_color("tab:green")
    scat_sells.set_offsets(list(zip(state.sell_times, state.sell_prices)))
    scat_sells.set_color("tab:red")

    # show TP/SL while in position
    if state.position_open and state.tp_price and state.sl_price:
        tp_line.set_ydata([state.tp_price])
        sl_line.set_ydata([state.sl_price])
        tp_line.set_visible(True); sl_line.set_visible(True)
    else:
        tp_line.set_visible(False); sl_line.set_visible(False)

    # dynamic title with latest stats
    pos_txt = "OPEN" if state.position_open else "FLAT"
    conf_txt = "—" if conf is None else f"{conf:.3f}"
    sig_txt = "—" if signal is None else str(signal)
    title.set_text(
        f"{SYMBOL}  price={price:.2f}  conf={conf_txt}  signal={sig_txt}  "
        f"pnl={state.pnl_realized*100:.2f}%  trades={state.trade_count}  ({pos_txt})"
    )

    return [line_price, scat_buys, scat_sells, tp_line, sl_line, title]

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print(f"Paper trading with live plot on {SYMBOL} (interval={INTERVAL}) | TESTNET={TESTNET}")
    ani = animation.FuncAnimation(fig, update, interval=POLL_MS, blit=False)
    plt.show()
