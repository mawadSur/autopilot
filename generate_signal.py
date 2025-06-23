import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from utils import load_ohlc_chunks, compute_rsi

def compute_features(df):
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1
    df['vol_change'] = df['volume'].pct_change()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)

    df.dropna(inplace=True)

    # 🚨 New step: remove any remaining bad values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    return df


def sliding_window_generator(data, window=100, batch_size=100000):
    for start in range(window, len(data), batch_size):
        end = min(start + batch_size, len(data))
        windows = np.array([data[i - window:i] for i in range(start, end)])
        yield windows, start, end

def main():
    print("📥 Loading historical OHLC data...")
    df = load_ohlc_chunks()
    df = compute_features(df)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]

    print("🧠 Preparing model inputs...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].values)

    print("📈 Running inference in batches...")
    model = load_model('eth_lstm_model.h5', compile=False)
    all_preds = []

    for X_batch, start_idx, end_idx in sliding_window_generator(X_scaled, window=100, batch_size=100000):
        preds = model.predict(X_batch).flatten()
        all_preds.extend(preds)
        print(f"🧪 Inferred {len(preds)} records ({start_idx} to {end_idx})")

    result_df = df.iloc[100:100+len(all_preds)].copy()
    result_df['confidence'] = all_preds
    result_df['true_future_return'] = (result_df['close'].shift(-1) - result_df['close']) / result_df['close']

    # Apply rule-based signal filtering
    result_df['signal'] = np.where(
        (result_df['confidence'] > 0.75) &
        (result_df['rsi_14'] < 70) &
        (result_df['macd'] > 0),
        1, 0
    )

    result_df[['open', 'high', 'low', 'close', 'volume', 'signal', 'confidence', 'true_future_return']]\
        .to_csv('eth_signals.csv')

    print(f"✅ Signals saved to eth_signals.csv | Positives: {result_df['signal'].sum()}")

if __name__ == "__main__":
    main()
