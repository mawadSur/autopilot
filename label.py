import pandas as pd
import numpy as np

def preprocess_and_label(df, window_size=10, threshold=0.02):
    """
    Automates feature generation and labeling for ML.
    
    Params:
    - df: DataFrame with columns ['open', 'high', 'low', 'close']
    - window_size: number of days to look back for each sample
    - threshold: percent change to trigger a "buy" label (e.g., +2%)

    Returns:
    - X: numpy array of shape [samples, window_size, features]
    - y: binary labels (1 if price goes up > threshold next day)
    """
    df = df.copy()

    # === Feature Engineering ===
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
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

    return np.array(X), np.array(y)
