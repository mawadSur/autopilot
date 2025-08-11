import os
from collections import deque
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

from utils import SignalGenerator

# ---------------------------------------
load_dotenv()
SYMBOL = os.getenv("TRADE_SYMBOL", "ETHUSDT")
INTERVAL = os.getenv("INTERVAL", "1m")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
POLL_MS = int(os.getenv("LIVE_POLL_MS", "1000"))

API_KEY = os.getenv("BINANCE_KEY") or os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE_TESTNET_SECRET")
TESTNET = bool(int(os.getenv("TESTNET", "0")))

def get_client() -> Client:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing BINANCE credentials in env")
    return Client(API_KEY, API_SECRET, testnet=TESTNET)

def to_row(k) -> Dict[str, Any]:
    return {"date": k[0], "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]), "volume": float(k[5])}

def prefill(sig: SignalGenerator, client: Client) -> None:
    kl = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=min(sig.history_size, 1000))
    for k in kl: sig.history.append(to_row(k))

def latest(client: Client) -> Tuple[Dict[str, Any], int]:
    kl = client.get_klines(symbol=SYMBOL, interval=INTERVAL, limit=2)
    last = to_row(kl[-1])
    return last, last["date"]

# ---------------------------------------
signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
client = get_client()
prefill(signal_gen, client)

prices = deque(maxlen=300)
confidences = deque(maxlen=300)
signals = deque(maxlen=300)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.set_title(f"{SYMBOL} live price & model signal")
ax1.set_ylabel("Price")
ax2.set_ylabel("Confidence")
ax2.set_xlabel("Ticks")

last_open_time = -1

def update(_):
    global last_open_time
    try:
        bar, open_time = latest(client)
        if open_time == last_open_time:
            return []
        last_open_time = open_time

        res = signal_gen.get_signal(bar)
        price = bar["close"]
        conf = float(res["confidence"]) if res["confidence"] is not None else 0.0
        sig = int(res["signal"])

        prices.append(price); confidences.append(conf); signals.append(sig)

        ax1.clear(); ax2.clear()
        ax1.plot(list(prices)); ax1.legend(["Price"], loc="upper left")
        ax2.plot(list(confidences)); ax2.plot(list(signals))
        ax2.legend(["Confidence", "Signal"], loc="upper left")
        ax2.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("Price"); ax2.set_ylabel("Conf / Signal")
        ax1.set_title(f"{SYMBOL} | price={price:.2f} conf={conf:.3f} signal={sig}")

    except BinanceAPIException as e:
        print(f"BinanceAPIException: {e.status_code} {e.message}")
    except Exception as e:
        print(f"Error in update: {e}")

    return []

ani = animation.FuncAnimation(fig, update, interval=POLL_MS, blit=False)
plt.tight_layout()
plt.show()
