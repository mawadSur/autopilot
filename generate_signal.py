import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from utils import load_ohlc_chunks, compute_rsi

# ==== Constants ====
WINDOW_SIZE = 150
FIXED_BATCH_SIZE = 16
BATCH_SIZE = 10000
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'body', 'range',
    'upper_wick', 'lower_wick', 'return', 'sma_ratio',
    'ema_20', 'macd', 'rsi_14', 'vol_change'
]
OUTPUT_FILE = 'eth_signals.csv'

# ==== Feature Engineering ====
def compute_features(df):
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# ==== Streaming Generator ====
def sliding_window_generator(data, window=150, batch_size=10000):
    for start in range(window, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = [data[i - window:i] for i in range(start, end)]
        yield np.array(batch, dtype=np.float32), start

# ==== Main ====
def main():
    print("📥 Loading historical OHLC data...")
    df = load_ohlc_chunks()
    df = compute_features(df)

    print("🧠 Preparing model inputs...")
    X = df[FEATURE_COLS].values
    scaler = joblib.load("scaler.pkl")
    X_scaled = scaler.transform(X)

    print("📈 Loading model and threshold...")
    model = load_model("eth_lstm_model.h5", compile=False)
    try:
        with open("model_meta.json", "r") as f:
            threshold = float(json.load(f).get("threshold", 0.75))
    except Exception:
        threshold = 0.75

    print("🔍 Running inference and writing to CSV...")
    first_write = True
    for X_batch, start_idx in sliding_window_generator(X_scaled, window=WINDOW_SIZE, batch_size=BATCH_SIZE):
        # Trim to match batch size for stateful LSTM
        excess = len(X_batch) % FIXED_BATCH_SIZE
        if excess > 0:
            X_batch = X_batch[:-excess]
        if len(X_batch) == 0:
            continue

        preds = model.predict(X_batch, batch_size=FIXED_BATCH_SIZE).flatten()
        model.reset_states()

        df_slice = df.iloc[start_idx:start_idx + len(preds)].copy()
        df_slice['confidence'] = preds
        df_slice['true_future_return'] = (df_slice['close'].shift(-1) - df_slice['close']) / df_slice['close']
        df_slice['signal'] = np.where(
            (df_slice['confidence'] > threshold) &
            (df_slice['rsi_14'] < 70) &
            (df_slice['macd'] > 0),
            1, 0
        )

        output_cols = ['open', 'high', 'low', 'close', 'volume', 'signal', 'confidence', 'true_future_return']
        df_slice[output_cols].to_csv(OUTPUT_FILE, mode='w' if first_write else 'a', index=False, header=first_write)
        first_write = False

        print(f"🧪 Batch saved: {len(preds)} rows from index {start_idx}")

    print(f"\n✅ All predictions saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
