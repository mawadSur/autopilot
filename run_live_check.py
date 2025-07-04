import numpy as np
import pandas as pd
import joblib
from collections import deque
from tensorflow.keras.models import load_model
import json
from train_model import focal_loss, compute_atr # Re-use the functions from your training script
from utils import compute_rsi

# --- This script is now a reusable module for generating trade signals ---

class SignalGenerator:
    def __init__(self, model_path='eth_lstm_model.h5', scaler_path='scaler.pkl', meta_path='model_meta.json'):
        """
        Initializes the signal generator with the trained model, scaler, and metadata.
        """
        print("⚙️ Initializing Signal Generator...")
        # --- CONFIGURATION ---
        self.window_size = 150
        self.fixed_batch_size = 16
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'body', 'range',
            'upper_wick', 'lower_wick', 'return', 'sma_ratio',
            'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr'
        ]
        
        # --- LOAD ASSETS ---
        self.model = load_model(model_path, custom_objects={'loss': focal_loss()}, compile=False)
        self.scaler = joblib.load(scaler_path)
        
        try:
            with open(meta_path, "r") as f:
                self.threshold = json.load(f).get("threshold", 0.5)
        except Exception:
            self.threshold = 0.5 # Fallback
        
        # --- DATA HISTORY ---
        # We need to keep enough history to calculate all rolling features (e.g., 50 for SMA50)
        # plus the window size for the model input. 150 + 50 = 200. Add buffer.
        self.history_size = self.window_size + 100 
        self.history = deque(maxlen=self.history_size)
        
        print(f"✅ Signal Generator ready. Confidence threshold: {self.threshold:.4f}")

    def _engineer_features(self, df):
        """
        Calculates features on a given dataframe. This MUST mirror the training process.
        """
        df_out = df.copy()
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
        return df_out

    def get_signal(self, new_kline_data):
        """
        Processes a new kline, updates history, and returns a trade signal if ready.
        
        Args:
            new_kline_data (dict): A dictionary with keys {'open', 'high', 'low', 'close', 'volume'}
        
        Returns:
            dict: A dictionary containing confidence and signal.
        """
        self.history.append(new_kline_data)

        if len(self.history) < self.window_size + 50: # Need enough data for SMA50 + model window
            return {"confidence": None, "signal": 0, "reason": "History buffer is not full."}

        # Create a DataFrame from the history
        df_history = pd.DataFrame(list(self.history))
        
        # Calculate features on the entire history to ensure accuracy
        df_features = self._engineer_features(df_history)

        # Get the last `window_size` rows which are now correctly calculated
        model_input_df = df_features.tail(self.window_size)
        
        # Check if there are any NaNs in the final input (can happen at the start)
        if model_input_df.isnull().values.any():
            return {"confidence": None, "signal": 0, "reason": "Not enough data for rolling features."}
            
        # Scale the data
        model_input_scaled = self.scaler.transform(model_input_df[self.feature_cols])
        
        # The stateful model expects a batch. We create a batch of size FIXED_BATCH_SIZE
        # by repeating the input window.
        X = np.array([model_input_scaled] * self.fixed_batch_size)
        
        # Get prediction and reset model states
        confidence = self.model.predict(X, batch_size=self.fixed_batch_size).flatten()[0]
        self.model.reset_states()

        signal = 1 if confidence > self.threshold else 0
        
        return {
            "confidence": round(float(confidence), 5),
            "signal": signal,
            "threshold": round(float(self.threshold), 3)
        }

# Example of how you would use this class in your trading script:
if __name__ == '__main__':
    # This is for demonstration. You would import and use this class in paper_trade.py.
    
    # 1. Initialize the generator once at the start of your script
    signal_gen = SignalGenerator()
    
    # 2. In your loop, you would feed it new klines from Binance
    # (Here we simulate with dummy data)
    print("\n--- DEMO ---")
    print("Simulating feeding new klines to the generator...")
    
    # Pre-fill history with some dummy data
    for i in range(250):
        dummy_kline = {
            'open': 1600 + i*0.1, 'high': 1601 + i*0.1, 'low': 1599 + i*0.1, 
            'close': 1600.5 + i*0.1, 'volume': 100 + i
        }
        signal_gen.history.append(dummy_kline)

    # Now, get a signal for a new incoming kline
    new_kline = {'open': 1625, 'high': 1628, 'low': 1624, 'close': 1627.5, 'volume': 250}
    
    result = signal_gen.get_signal(new_kline)
    
    print(f"\nReceived new kline: {new_kline}")
    print(f"Signal Result: {result}")
    
    if result['signal'] == 1:
        print("💡 Recommendation: BUY")
    else:
        print("💡 Recommendation: HOLD")