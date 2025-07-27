import os
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from binance.client import Client
from dotenv import load_dotenv
from utils import SignalGenerator, get_client_binance

# --- Setup ---
load_dotenv()
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
if not ENDPOINT_NAME:
    raise ValueError("ENDPOINT_NAME not set in .env file.")

signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
client = get_client_binance()

# --- Pre-fill history buffer ---
print("Pre-filling history buffer...")
klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
for k in klines:
    signal_gen.history.append({
        'date': k[0], 'open': float(k[1]), 'high': float(k[2]),
        'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
    })
print("✅ History buffer pre-filled.")

# --- Setup plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 8))
fig.suptitle("Live ETH/USDT Price and Model Confidence", fontsize=16)

# --- Data storage ---
price_data = deque(maxlen=150)
high_data = deque(maxlen=150)
low_data = deque(maxlen=150)
time_data = deque(maxlen=150)
confidence_data = deque(maxlen=150)

def update(frame):
    """
    This function is called periodically by FuncAnimation to update the plot.
    """
    try:
        latest_kline = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
        kline_data = {
            'date': latest_kline[0], 'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
            'low': float(latest_kline[3]), 'close': float(latest_kline[4]), 'volume': float(latest_kline[5])
        }
        current_time = pd.to_datetime(kline_data['date'], unit='ms')
        
        close_price = kline_data['close']
        high_price = kline_data['high']
        low_price = kline_data['low']

        result = signal_gen.get_signal(kline_data)
        confidence = result.get('confidence')
        signal = result.get('signal', 0)
        threshold = result.get('threshold')

        price_data.append(close_price)
        high_data.append(high_price)
        low_data.append(low_price)
        time_data.append(current_time)
        confidence_data.append(confidence if confidence is not None else 0)

        ax1.clear()
        ax2.clear()

        # Plot 1: Price Chart
        ax1.plot(time_data, price_data, label='Close Price', color='deepskyblue', zorder=10)
        ax1.fill_between(time_data, low_data, high_data, color='deepskyblue', alpha=0.2, label='High-Low Range')
        ax1.set_ylabel("Price (USDT)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        if signal == 1:
            ax1.scatter(current_time, close_price, color='lime', marker='^', s=120, zorder=15, label='Buy Signal')
        ax1.legend(loc='upper left')

        # Plot 2: Confidence Chart
        ax2.plot(time_data, confidence_data, color='coral', label='Confidence')
        if threshold:
            ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({threshold:.2f})')
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='upper left')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # --- MODIFIED: More verbose console logging ---
        # Determine the action based on the signal
        if signal == 1:
            action_str = f"✅ Initiate BUY (Confidence {confidence:.3f} > Threshold {threshold:.2f})"
        else:
            conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
            action_str = f"❌ HOLD (Confidence {conf_str} <= Threshold {threshold:.2f})"

        # Create a detailed, multi-line log for each update that prints on a new line
        log_output = (
            f"--- [ {current_time} ] ---\n"
            f"  📈 Price:         ${close_price:<8.2f} "
            f"  💡 Intention:     {action_str}\n"
        )
        print(log_output, end='') # Use end='' to avoid extra newlines

    except Exception as e:
        print(f"\n❌ Error in update loop: {e}")

# --- Run animation ---
ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)
plt.tight_layout(rect=[0.04, 0, 1, 0.90])
plt.show()