import os
import time
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from train_model import focal_loss
from utils import compute_rsi

# ==== CONFIG ====
WINDOW_SIZE = 150
FIXED_BATCH_SIZE = 16
CONF_THRESHOLD = 0.75  # fallback
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'body', 'range',
    'upper_wick', 'lower_wick', 'return', 'sma_ratio',
    'ema_20', 'macd', 'rsi_14', 'vol_change'
]

# ==== ENV & Logging ====
load_dotenv()
API_KEY = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'
logging.basicConfig(filename='paper_trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# ==== Load Model and Scaler ====
model = load_model('eth_lstm_model.h5', custom_objects={'loss': focal_loss()}, compile=False)
scaler = joblib.load("scaler.pkl")

try:
    with open("model_meta.json", "r") as f:
        CONF_THRESHOLD = json.load(f).get("threshold", CONF_THRESHOLD)
except Exception:
    pass

# ==== Feature Engineering ====
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

def prepare_input(df):
    df = engineer_features(df)
    recent = df[FEATURE_COLS].values[-WINDOW_SIZE:]
    recent_scaled = scaler.transform(recent)
    X = np.array([recent_scaled] * FIXED_BATCH_SIZE)  # stateful batch
    current_price = df['close'].iloc[-1]
    return X, current_price, df[['close', 'macd', 'rsi_14', 'sma_ratio']].tail(3)

# ==== Fetch Binance OHLCV ====
def fetch_eth_ohlcv(minutes=180):
    klines = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=minutes)
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('date', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# ==== Paper Trading ====
def run_multiple_trades(n=10, interval=60):
    total_profit = 0.0
    trade_count = 0
    last_close = None

    for i in range(n):
        try:
            df = fetch_eth_ohlcv()
            X, current_price, debug_tail = prepare_input(df)
            prediction = model.predict(X, batch_size=FIXED_BATCH_SIZE).flatten()[0]
            model.reset_states()

            new_close = df['close'].iloc[-1]
            if last_close == new_close:
                print(f"[{i+1}] Waiting for fresh data...")
                time.sleep(interval)
                continue
            last_close = new_close

            print(debug_tail)
            print(f"[{i+1}] Confidence: {prediction:.4f} | Threshold: {CONF_THRESHOLD:.2f} | Price: {current_price:.2f}")

            if prediction > CONF_THRESHOLD:
                entry = current_price
                simulated_exit = entry * 1.01
                pnl = simulated_exit - entry
                total_profit += pnl
                trade_count += 1
                print(f"✅ Trade {i+1}: Buy {entry:.2f} → Sell {simulated_exit:.2f} | PnL: {pnl:.2f}")
                logging.info(f"Trade {i+1} | Entry: {entry:.2f}, Exit: {simulated_exit:.2f}, PnL: {pnl:.2f}")
            else:
                print(f"🚫 Trade {i+1}: Skipped (Low confidence)")

        except Exception as e:
            print(f"❌ Error on trade {i+1}: {e}")
            logging.error(f"Error on trade {i+1}: {e}")

        time.sleep(interval)

    print(f"\n📊 Trades completed: {trade_count}/{n}")
    print(f"💰 Total simulated PnL: {total_profit:.2f} USD")

# ==== Entry ====
if __name__ == "__main__":
    run_multiple_trades(n=10, interval=60)
