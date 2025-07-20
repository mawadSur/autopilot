import os
import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from binance.client import Client
from dotenv import load_dotenv
from utils import SignalGenerator, get_client_binance

# --- Setup ---
load_dotenv()
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
client = get_client_binance()

# --- Pre-fill history buffer ---
print("Pre-filling history buffer...")
klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
for k in klines:
    signal_gen.history.append({
        'date': pd.to_datetime(k[0], unit='ms'),
        'open': float(k[1]),
        'high': float(k[2]),
        'low': float(k[3]),
        'close': float(k[4]),
        'volume': float(k[5])
    })

# --- Setup plot ---
fig, ax1 = plt.subplots(figsize=(14, 7))
ax2 = ax1.twinx()
price_data, time_data, confidence_data, signal_data = deque(maxlen=150), deque(maxlen=150), deque(maxlen=150), deque(maxlen=150)

def update(frame):
    try:
        latest_kline = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
        close_price = float(latest_kline[4])
        current_time = pd.to_datetime(latest_kline[0], unit='ms')
        kline_data = {
            'date': current_time,
            'open': float(latest_kline[1]),
            'high': float(latest_kline[2]),
            'low': float(latest_kline[3]),
            'close': close_price,
            'volume': float(latest_kline[5])
        }

        result = signal_gen.get_signal(kline_data)
        confidence = result.get('confidence', 0)
        signal = result.get('signal', 0)

        price_data.append(close_price)
        time_data.append(current_time)
        confidence_data.append(confidence)
        signal_data.append(signal)

        ax1.clear()
        ax2.clear()

        # Plot price
        ax1.plot(time_data, price_data, label='ETH Price', color='deepskyblue')
        ax1.set_ylabel("Price (USDT)")
        ax1.set_title("Live ETH/USDT with Signals and Confidence")
        ax1.grid(True)

        # Plot signal markers
        for i in range(len(signal_data)):
            sig = signal_data[i]
            if sig == 1:
                ax1.scatter(time_data[i], price_data[i], color='lime', marker='^', s=100, label='BUY' if i == 0 else "")
            elif sig == -1:
                ax1.scatter(time_data[i], price_data[i], color='red', marker='v', s=100, label='SELL' if i == 0 else "")
            else:
                ax1.scatter(time_data[i], price_data[i], color='gray', marker='o', s=30, label='HOLD' if i == 0 else "")

        # Plot confidence
        ax2.plot(time_data, confidence_data, color='coral', label='Confidence')
        ax2.axhline(y=signal_gen.threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({signal_gen.threshold:.2f})')
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        ax2.grid(True)

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.2)

        # Console print
        print(f"\r⏱ {current_time} | 💰 Price: ${close_price:.2f} | 🔮 Confidence: {confidence:.3f if confidence else 0:.3f} | Signal: {signal}", end='')

    except Exception as e:
        print(f"\n❌ Error in update loop: {e}")

    return ax1.lines + ax2.lines

# --- Run animation ---
ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)
plt.tight_layout()
plt.show()
