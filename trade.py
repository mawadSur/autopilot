import os
import time
import logging
import pandas as pd
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from utils import SignalGenerator

TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "ETHUSDT")
TRADE_QUANTITY_USDT = float(os.getenv("TRADE_QUANTITY_USDT", "15"))

load_dotenv()
API_KEY = os.getenv("BINANCE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing BINANCE_KEY/BINANCE_SECRET in env/.env")
if not ENDPOINT_NAME:
    raise ValueError("Missing ENDPOINT_NAME in env/.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def trade_live():
    client = Client(API_KEY, API_SECRET)
    signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)

    # Prefill history
    print(f"Pre-filling history with last {signal_gen.history_size} minutes...")
    klines = client.get_klines(symbol=TRADE_SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
    for k in klines:
        kline_data = {
            'date': pd.to_datetime(k[0], unit='ms'),
            'open': float(k[1]), 'high': float(k[2]),
            'low': float(k[3]), 'close': float(k[4]),
            'volume': float(k[5])
        }
        signal_gen.history.append(kline_data)

    print("✅ History buffer filled. Starting live trading loop...")
    position_open = False
    interval = int(os.getenv("LOOP_INTERVAL_SEC", "60"))

    while True:
        try:
            k = client.get_klines(symbol=TRADE_SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            new_kline_data = {
                'date': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]),
                'volume': float(k[5])
            }

            if not position_open:
                res = signal_gen.get_signal(new_kline_data)
                current_price = new_kline_data['close']
                print(f"Price: {current_price:.2f} | Confidence: {res.get('confidence')} | Signal: {res.get('signal')}")
                if res.get("signal") == 1:
                    print(f"🚀 BUY {TRADE_SYMBOL} for ${TRADE_QUANTITY_USDT} (market)")
                    # TODO: place real order via client.create_order(...)
                    position_open = True
            else:
                print("Position is open. Waiting for exit condition...")
                # TODO: Implement TP/SL logic and sell

        except Exception as e:
            logging.error(f"Error in trade loop: {e}")

        time.sleep(interval)

if __name__ == "__main__":
    trade_live()