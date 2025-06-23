import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from binance.client import Client
from binance.enums import *
from utils import compute_rsi

# === Binance Credentials ===
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
client = Client(API_KEY, API_SECRET)

# === Load Model ===
model = load_model('eth_lstm_model.h5')

# === Feature Engineering ===


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

def prepare_input(df, window_size=60):
    df = engineer_features(df)
    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]
    recent = df[feature_cols].values[-window_size:]
    return np.expand_dims(recent, axis=0), df['close'].iloc[-1]

# === Fetch ETH OHLCV ===
def fetch_eth_ohlcv(minutes=80):
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

# === Trade Handler ===
def trade_live(log_path="trade_log.csv", portfolio_pct=0.1, iterations=20, interval=60):
    account_info = client.get_account()
    balances = {asset['asset']: float(asset['free']) for asset in account_info['balances']}
    usdt_balance = balances.get('USDT', 0.0)

    if usdt_balance == 0.0:
        print("❌ No USDT available in your Binance account.")
        return

    df_log = []

    for i in range(iterations):
        try:
            df = fetch_eth_ohlcv()
            X, current_price = prepare_input(df)
            prediction = model.predict(X)[0][0]

            print(f"[{i+1}] Prediction: {prediction:.4f} | Price: {current_price:.2f} | USDT Balance: ${usdt_balance:.2f}")

            if prediction > 0.7:
                usd_position = usdt_balance * portfolio_pct
                eth_bought = usd_position / current_price
                sell_price = current_price * 1.01  # simulate 1% gain
                pnl = eth_bought * (sell_price - current_price)
                usdt_balance += pnl

                print(f"✅ BUY → SELL | Entry: {current_price:.2f} → Exit: {sell_price:.2f} | PnL: ${pnl:.2f}")
                df_log.append({
                    'iteration': i + 1,
                    'entry_price': current_price,
                    'exit_price': sell_price,
                    'eth_traded': eth_bought,
                    'prediction': prediction,
                    'pnl': pnl,
                    'balance': usdt_balance,
                    'timestamp': datetime.utcnow()
                })
            else:
                print(f"🚫 Skipped trade — confidence too low.")

        except Exception as e:
            print(f"❌ Error: {e}")

        time.sleep(interval)

    # Save log
    log_df = pd.DataFrame(df_log)
    log_df.to_csv(log_path, index=False)
    print(f"📦 Saved trade log to {log_path}")


# === Main ===
trade_live()
