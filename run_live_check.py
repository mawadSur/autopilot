import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from binance.client import Client
from run_live_check import SignalGenerator

# Initialize Binance client and SignalGenerator
client = Client()
ENDPOINT_NAME = "pytorch-training-2025-07-08-06-10-58-197"
signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)

# Pre-fill history buffer
klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
for k in klines:
    signal_gen.history.append({'date': pd.to_datetime(k[0], unit='ms'), 'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])})

# Initialize plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
price_data, time_data, confidence_data, signal_data = deque(maxlen=150), deque(maxlen=150), deque(maxlen=150), deque(maxlen=150)
sma10_data, sma50_data, rsi_data = deque(maxlen=150), deque(maxlen=150), deque(maxlen=150)

# Update function for animation
def update(frame):
    try:
        latest_kline = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
        close_price = float(latest_kline[4])
        current_time = pd.to_datetime(latest_kline[0], unit='ms')
        kline_data = {
            'date': current_time, 'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
            'low': float(latest_kline[3]), 'close': close_price, 'volume': float(latest_kline[5])
        }

        result = signal_gen.get_signal(kline_data)
        confidence = result.get('confidence')
        signal = result.get('signal')

        # Update data stores
        signal_gen.history.append(kline_data)
        df = pd.DataFrame(signal_gen.history)
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        price_data.append(close_price)
        time_data.append(current_time)
        confidence_data.append(confidence)
        signal_data.append(signal)
        sma10_data.append(df['sma_10'].iloc[-1])
        sma50_data.append(df['sma_50'].iloc[-1])
        rsi_data.append(rsi.iloc[-1])

        # --- Price Plot ---
        ax1.clear()
        ax1.plot(time_data, price_data, label='Price', color='deepskyblue')
        ax1.plot(time_data, sma10_data, label='SMA-10', linestyle='--', color='orange')
        ax1.plot(time_data, sma50_data, label='SMA-50', linestyle='--', color='green')
        for i, sig in enumerate(signal_data):
            if sig == 1:
                ax1.scatter(time_data[i], price_data[i], color='green', s=50, label='BUY' if i == len(signal_data) - 1 else "")
            else:
                ax1.scatter(time_data[i], price_data[i], color='gray', s=20)
        ax1.set_ylabel("Price (USDT)")
        ax1.set_title("Live ETH/USDT with SMA & Buy/Hold Signals")
        ax1.grid(True)
        ax1.legend(loc='upper left')

        # --- Confidence Plot ---
        ax2.clear()
        ax2.plot(time_data, confidence_data, color='coral', label='Confidence')
        ax2.axhline(y=signal_gen.threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({signal_gen.threshold:.2f})')
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time")
        ax2.grid(True)
        ax2.legend(loc='lower left')

        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.2, hspace=0.3)

        print(f"\rPrice: ${close_price:.2f} | Confidence: {confidence:.3f if confidence else 'N/A'} | Signal: {'BUY' if signal == 1 else 'HOLD'}", end="")

    except Exception as e:
        print(f"\nError: {e}")

# Launch animation
ani = animation.FuncAnimation(fig, update, interval=1000)  # update every second
plt.show()
