import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from utils import load_ohlc_chunks, compute_rsi

WINDOW_SIZE = 150
FIXED_BATCH_SIZE = 16
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'body', 'range',
    'upper_wick', 'lower_wick', 'return', 'sma_ratio',
    'ema_20', 'macd', 'rsi_14', 'vol_change'
]

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
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def main():
    print("📥 Loading historical OHLC data...")
    df = load_ohlc_chunks()
    df = compute_features(df)

    print("🧠 Preparing model inputs...")
    X = df[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("📐 Creating input windows...")
    X_windows = []
    for i in range(WINDOW_SIZE, len(X_scaled)):
        X_windows.append(X_scaled[i - WINDOW_SIZE:i])
    X_windows = np.array(X_windows, dtype=np.float32)

    # Trim to ensure batch divisibility
    excess = len(X_windows) % FIXED_BATCH_SIZE
    if excess > 0:
        X_windows = X_windows[:-excess]
        df = df.iloc[:-(excess)]  # align with trimmed X

    print("📈 Loading stateful model...")
    model = load_model('eth_lstm_model.h5', compile=False)

    print(f"🔍 Running inference on {len(X_windows)} samples in batches of {FIXED_BATCH_SIZE}")
    preds = model.predict(X_windows, batch_size=FIXED_BATCH_SIZE).flatten()
    model.reset_states()

    print("📊 Building output DataFrame...")
    result_df = df.iloc[WINDOW_SIZE:WINDOW_SIZE + len(preds)].copy()
    result_df['confidence'] = preds
    result_df['true_future_return'] = (result_df['close'].shift(-1) - result_df['close']) / result_df['close']

    result_df['signal'] = np.where(
        (result_df['confidence'] > 0.75) &
        (result_df['rsi_14'] < 70) &
        (result_df['macd'] > 0),
        1, 0
    )

    result_df[['open', 'high', 'low', 'close', 'volume', 'signal', 'confidence', 'true_future_return']]\
        .to_csv('eth_signals.csv', index=False)

    print(f"✅ Signals saved to eth_signals.csv | Positives: {result_df['signal'].sum()}")

if __name__ == "__main__":
    main()
