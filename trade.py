import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from binance.client import Client
from binance.enums import *
import os

# === Binance Credentials (use env variables or secrets) ===
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
client = Client(API_KEY, API_SECRET)

# === Load Trained Model ===
model = load_model('eth_lstm_model.h5')

# === Feature Engineering (match Step 3) ===
def engineer_features(df):
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df.dropna(inplace=True)
    return df

# === Format Data for Prediction ===
def prepare_input(df, window_size=10):
    df = engineer_features(df)
    feature_cols = [
            'open', 'high', 'low', 'close', 'body', 'range',
            'upper_wick', 'lower_wick', 'return', 'sma_ratio',
            'ema_20', 'macd', 'rsi_14', 'vol_change'
        ]
    recent = df[feature_cols].values[-window_size:]
    return np.expand_dims(recent, axis=0)

# === Fetch Live ETH Data ===
def fetch_eth_ohlcv(days=15):
    klines = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1DAY, limit=days)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# === Trade Logic ===
def trade_if_signal():
    df = fetch_eth_ohlcv()
    X = prepare_input(df)
    prediction = model.predict(X)[0][0]

    print(f"Prediction Confidence: {prediction:.4f}")

    if prediction > 0.9:
        print("💹 Signal: BUY ETH")
        quantity = 0.05  # define your trade size
        try:
            order = client.order_market_buy(
                symbol='ETHUSDT',
                quantity=quantity
            )
            print(f"Order executed: {order}")
        except Exception as e:
            print("Trade failed:", str(e))
    else:
        print("🚫 No action — prediction below threshold.")

# === Schedule Daily Run (manually or with cron)
if __name__ == "__main__":
    trade_if_signal()
