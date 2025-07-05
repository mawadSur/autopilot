import numpy as np
import pandas as pd
import joblib
from collections import deque
from tensorflow.keras.models import load_model
import json
from binance.client import Client

# --- New Imports for Plotting ---
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Assuming these are in your local files ---
from train_model import focal_loss, compute_atr
from utils import compute_rsi

# The SignalGenerator class remains the same
class SignalGenerator:
    def __init__(self, model_path='eth_lstm_model.h5', scaler_path='scaler.pkl', meta_path='model_meta.json'):
        print("⚙️ Initializing Signal Generator...")
        self.window_size = 150
        self.fixed_batch_size = 16
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'body', 'range',
            'upper_wick', 'lower_wick', 'return', 'sma_ratio',
            'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr',
            'price_vs_hourly_trend', 'bb_width'
        ]
        self.model = load_model(model_path, custom_objects={'loss': focal_loss()}, compile=False)
        self.scaler = joblib.load(scaler_path)
        try:
            with open(meta_path, "r") as f:
                self.threshold = json.load(f).get("threshold", 0.5)
        except Exception:
            self.threshold = 0.5
        self.history_size = self.window_size + 100
        self.history = deque(maxlen=self.history_size)
        print(f"✅ Signal Generator ready. Confidence threshold: {self.threshold:.4f}")

    # In run_live_check.py, inside the SignalGenerator class
    def _engineer_features(self, df):
        """
        Calculates features on a given dataframe. This MUST mirror the training process.
        """
        df_out = df.copy()
        df_out.set_index(pd.to_datetime(df_out.index), inplace=True) # Ensure index is datetime

        # --- Standard Features ---
        df_out['body'] = df_out['close'] - df_out['open']
        df_out['range'] = df_out['high'] - df_out['low']
        df_out['upper_wick'] = df_out['high'] - df_out[['close', 'open']].max(axis=1)
        df_out['lower_wick'] = df_out[['close', 'open']].min(axis=1) - df_out['low']
        df_out['return'] = df_out['close'].pct_change()
        df_out['sma_10'] = df_out['close'].rolling(10).mean()
        df_out['sma_50'] = df_out['close'].rolling(50).mean()
        df_out['sma_ratio'] = df_out['sma_10'] / (df_out['sma_50'] + 1e-9) - 1
        df_out['ema_20'] = df_out['close'].ewm(span=20).mean()
        df_out['macd'] = df_out['close'].ewm(span=12).mean() - df_out['close'].ewm(span=26).mean()
        df_out['rsi_14'] = compute_rsi(df_out['close'], 14)
        df_out['vol_change'] = df_out['volume'].pct_change()
        df_out['atr'] = compute_atr(df_out)

        # --- New Multi-Timeframe and Volatility Features ---
        # NOTE: The resample logic is slightly different for live data since we don't have the full history.
        # We approximate the hourly EMA based on the available history.
        if len(df_out) >= 60:
            df_hourly = df_out['close'].resample('1H').mean()
            hourly_ema = df_hourly.ewm(span=20).mean()
            df_out['hourly_ema_20'] = hourly_ema.reindex(df_out.index, method='ffill')
            df_out['price_vs_hourly_trend'] = (df_out['close'] - df_out['hourly_ema_20']) / df_out['hourly_ema_20']
        else: # Not enough data for hourly, fill with 0
            df_out['price_vs_hourly_trend'] = 0

        df_out['bb_std'] = df_out['close'].rolling(20).std()
        df_out['bb_mid'] = df_out['close'].rolling(20).mean()
        df_out['bb_width'] = ((df_out['bb_mid'] + 2 * df_out['bb_std']) - (df_out['bb_mid'] - 2 * df_out['bb_std'])) / df_out['bb_mid']
        
        # --- Data Cleaning ---
        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_out.fillna(0, inplace=True)
        
        return df_out

    def get_signal(self, new_kline_data):
        self.history.append(new_kline_data)
        if len(self.history) < self.window_size + 50:
            return {"confidence": None, "signal": 0, "reason": "History buffer is not full."}
        df_history = pd.DataFrame(list(self.history))
        df_features = self._engineer_features(df_history)
        model_input_df = df_features.tail(self.window_size)
        if model_input_df.isnull().values.any():
            return {"confidence": None, "signal": 0, "reason": "NaNs found in final model input."}
        model_input_scaled = self.scaler.transform(model_input_df[self.feature_cols].values)
        X = np.array([model_input_scaled] * self.fixed_batch_size)
        confidence = self.model.predict(X, batch_size=self.fixed_batch_size, verbose=0).flatten()[0]
        for layer in self.model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        signal = 1 if confidence > self.threshold else 0
        return {
            "confidence": round(float(confidence), 5),
            "signal": signal,
            "threshold": round(float(self.threshold), 3)
        }

# --- Main execution block for live plotting ---
if __name__ == '__main__':
    client = Client()
    signal_gen = SignalGenerator()

    print("Pre-filling history buffer with the last ~250 seconds of data...")
    klines = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1SECOND, limit=signal_gen.history_size)
    for k in klines:
        signal_gen.history.append({'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])})

    # --- Plotting Setup ---
    # Create two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Data storage for plotting
    price_data = deque(maxlen=60)
    time_data = deque(maxlen=60)
    confidence_data = deque(maxlen=60) # New: Store confidence history

    def update(frame):
        try:
            latest_kline = client.get_klines(symbol='ETHUSDT', interval=Client.KLINE_INTERVAL_1SECOND, limit=1)[0]
            close_price = float(latest_kline[4])
            current_time = pd.to_datetime(latest_kline[0], unit='ms')
            kline_data = {'open': float(latest_kline[1]), 'high': float(latest_kline[2]), 'low': float(latest_kline[3]), 'close': close_price, 'volume': float(latest_kline[5])}
            
            result = signal_gen.get_signal(kline_data)
            confidence = result.get('confidence')

            # Update data for the plots
            price_data.append(close_price)
            time_data.append(current_time)
            confidence_data.append(confidence) # Add confidence to its history

            # --- Update Price Plot (ax1) ---
            ax1.clear()
            ax1.plot(time_data, price_data, color='deepskyblue')
            ax1.set_title("Live ETH/USDT Price and Model Confidence")
            ax1.set_ylabel("Price (USDT)")
            ax1.grid(True)

            # --- Update Confidence Plot (ax2) ---
            ax2.clear()
            ax2.plot(time_data, confidence_data, color='coral')
            # Add a horizontal line for the buy threshold
            ax2.axhline(y=signal_gen.threshold, color='r', linestyle='--', linewidth=1, label=f'Threshold ({signal_gen.threshold:.2f})')
            ax2.set_ylabel("Confidence")
            ax2.set_xlabel("Time")
            ax2.set_ylim(0, 1) # Confidence is between 0 and 1
            ax2.grid(True)
            ax2.legend(loc='lower left')

            plt.xticks(rotation=45, ha='right')
            plt.subplots_adjust(bottom=0.20, hspace=0)

            # --- Update Console ---
            signal = "BUY" if result.get('signal') == 1 else "HOLD"
            conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
            print(f"\rPrice: ${close_price:<8.2f} | Confidence: {conf_str:<7} | Signal: {signal:<5}", end="")

        except Exception as e:
            print(f"\nAn error occurred: {e}")

    ani = animation.FuncAnimation(fig, update, interval=1000)
    plt.show()