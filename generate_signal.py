import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

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
    return df

def sliding_window(data, window=100):
    return np.array([data[i - window:i] for i in range(window, len(data))])

def load_ohlc_chunks(folder='eth_1s_data'):
    files = sorted(glob(os.path.join(folder, '*.csv')))
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    print(f"📁 Found {len(files)} CSV files in {folder}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['date'], index_col='date')
            if not df.empty:
                dfs.append(df)
            else:
                print(f"⚠️ Skipping empty file: {f}")
        except Exception as e:
            print(f"⚠️ Skipping corrupted file: {f} | Error: {e}")

    if not dfs:
        raise ValueError("No valid dataframes to concatenate. Please check your CSV files.")

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated()]
    return df

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
    X_scaled = scaler.fit_transform(df[feature_cols])
    X = sliding_window(X_scaled, window=100)

    print("📈 Running inference...")
    model = load_model('eth_lstm_model.h5', compile=False)
    preds = model.predict(X).flatten()

    result_df = df.iloc[100:].copy()
    result_df['confidence'] = preds
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
