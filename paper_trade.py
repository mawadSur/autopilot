import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from binance.client import Client
from binance.enums import *
from train_model import focal_loss
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(filename='paper_trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

API_KEY = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

model = load_model('eth_lstm_model.h5', custom_objects={'loss': focal_loss()})

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def engineer_features(df):
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
    return df

def prepare_input(df, window_size=150):
    df = engineer_features(df)
    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]
    recent = df[feature_cols].values[-window_size:]
    return np.expand_dims(recent, axis=0), df['close'].iloc[-1]

def fetch_eth_ohlcv(hours=200):
    klines = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1HOUR, limit=hours)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def run_multiple_trades(n=10, interval=5):
    total_profit = 0.0
    trade_count = 0
    for i in range(n):
        try:
            df = fetch_eth_ohlcv()
            X, current_price = prepare_input(df)
            prediction = model.predict(X)[0][0]

            print(f"[{i+1}] Prediction: {prediction:.4f} | Price: {current_price:.2f}")

            if prediction > 0.7:
                entry = current_price
                simulated_exit = entry * 1.01
                pnl = simulated_exit - entry
                total_profit += pnl
                trade_count += 1
                print(f"✅ Trade {i+1}: Buy {entry:.2f} → Sell {simulated_exit:.2f} | PnL: {pnl:.2f}")
            else:
                print(f"🚫 Trade {i+1}: Skipped (Confidence too low)")

        except Exception as e:
            print(f"❌ Error on trade {i+1}: {e}")

        time.sleep(interval)

    print(f"\n📊 Trades completed: {trade_count}/{n}")
    print(f"💰 Total simulated PnL: {total_profit:.2f} USD")

if __name__ == "__main__":
    run_multiple_trades(n=10, interval=5)
