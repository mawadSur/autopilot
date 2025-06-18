import pandas as pd
import numpy as np
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
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['rsi_14'] = compute_rsi(df['close'])
    df.dropna(inplace=True)
    return df

def sliding_window(data, window=100):
    X = []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
    return np.array(X)

def main():
    df = pd.read_csv('eth_ohlc.csv', parse_dates=['date'], index_col='date')
    df = compute_features(df)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    X = sliding_window(X_scaled, window=100)

    model = load_model('eth_lstm_model.h5', compile=False)
    preds = model.predict(X).flatten()

    signal_df = df.iloc[100:].copy()
    signal_df['model_conf'] = preds
    signal_df['true_future_return'] = (signal_df['close'].shift(-1) - signal_df['close']) / signal_df['close']

    # Smart filter: Only buy if confidence high, RSI < 70, MACD positive
    signal_df['signal'] = np.where(
        (signal_df['model_conf'] > 0.75) & 
        (signal_df['rsi_14'] < 70) & 
        (signal_df['macd'] > 0), 
        1, 0
    )

    signal_df[['open', 'high', 'low', 'close', 'volume', 'signal', 'model_conf', 'true_future_return']].to_csv('eth_signals.csv')
    print("✅ Saved smart signals to eth_signals.csv")

if __name__ == "__main__":
    main()
