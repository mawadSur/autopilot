import os
import threading
import queue
import time
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from binance.client import Client
from dotenv import load_dotenv
from utils import SignalGenerator

# --- Setup ---
load_dotenv()
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
client = Client(os.getenv('BINANCE_KEY'), os.getenv('BINANCE_SECRET'))

# Thread-safe queue for background results
result_queue: "queue.Queue[dict]" = queue.Queue(maxsize=512)

# --- Live data buffers ---
price_data = deque(maxlen=300)
time_data = deque(maxlen=300)
confidence_data = deque(maxlen=300)
signal_data = deque(maxlen=300)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))


def data_fetcher(q: "queue.Queue[dict]") -> None:
    """Background loop to fetch klines and signals, then enqueue results."""
    while True:
        try:
            latest = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            kline_data = {
                'date': pd.to_datetime(latest[0], unit='ms'),
                'open': float(latest[1]),
                'high': float(latest[2]),
                'low': float(latest[3]),
                'close': float(latest[4]),
                'volume': float(latest[5])
            }
            res = signal_gen.get_signal(kline_data)
            out = {
                'date': kline_data['date'],
                'close': kline_data['close'],
                'confidence': res.get('confidence', 0.0),
                'signal': res.get('signal', 0),
            }
            try:
                q.put_nowait(out)
            except queue.Full:
                # Drop oldest to keep UI responsive
                try:
                    _ = q.get_nowait()
                except Exception:
                    pass
                q.put_nowait(out)
        except Exception as e:
            print(f"\n[fetcher] Error: {e}")
        time.sleep(5)


def update(_):
    # Non-blocking update: consume latest result if available
    try:
        item = result_queue.get_nowait()
    except queue.Empty:
        return ax1.lines + ax2.lines

    close_price = item['close']
    current_time = item['date']
    confidence = item.get('confidence', 0.0)
    signal = item.get('signal', 0)

    price_data.append(close_price)
    time_data.append(current_time)
    confidence_data.append(confidence)
    signal_data.append(signal)

    ax1.clear()
    ax2.clear()

    # Price
    ax1.plot(time_data, price_data, label='ETH Price', color='deepskyblue')
    ax1.set_ylabel("Price (USDT)")
    ax1.set_title("Live ETH/USDT with Signals and Confidence")
    ax1.grid(True)

    # Signal markers
    for i, sig in enumerate(signal_data):
        if sig == 1:
            ax1.scatter(time_data[i], price_data[i], color='lime', marker='^', s=100, label='BUY' if i == 0 else "")
        elif sig == -1:
            ax1.scatter(time_data[i], price_data[i], color='red', marker='v', s=100, label='SELL' if i == 0 else "")
        else:
            ax1.scatter(time_data[i], price_data[i], color='gray', marker='o', s=30, label='HOLD' if i == 0 else "")

    # Confidence
    ax2.plot(time_data, confidence_data, label='Confidence')
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time")
    ax2.grid(True)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Console print
    try:
        print(f"\rTime: {current_time} | Price: ${close_price:.2f} | Conf: {confidence if confidence else 0:.3f} | Signal: {signal}", end='')
    except Exception:
        pass

    return ax1.lines + ax2.lines


ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)

# Start background fetcher thread (daemon)
t = threading.Thread(target=data_fetcher, args=(result_queue,), daemon=True)
t.start()
plt.tight_layout()
plt.show()

