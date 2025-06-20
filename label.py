import pandas as pd
import numpy as np
import os
from glob import glob

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def preprocess_and_label(df, window_size=10, threshold=0.02):
    df = df.copy()

    # === Feature Engineering ===
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
    df.dropna(inplace=True)

    # === Labeling ===
    df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['label'] = (df['target'] > threshold).astype(int)
    df.dropna(inplace=True)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]
    features = df[feature_cols].values
    labels = df['label'].values

    # === Sliding Windows ===
    X = []
    y = []

    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(labels[i])

    return np.array(X), np.array(y), df  # Also return DataFrame for saving

def load_all_chunks(folder='data/ohlc_chunks'):
    files = sorted(glob(os.path.join(folder, '*.csv')))
    dfs = [pd.read_csv(f, parse_dates=['date'], index_col='date') for f in files]
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated()].sort_index()
    return combined

if __name__ == "__main__":
    df = load_all_chunks()
    X, y, labeled_df = preprocess_and_label(df, window_size=150, threshold=0.02)
    labeled_df.to_csv('labeled_data.csv')
    print(f"✅ Labeled dataset saved to labeled_data.csv with shape: {X.shape}, Labels: {np.bincount(y)}")
