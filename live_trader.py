import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv
from binance.client import Client
from utils import SignalGenerator

def run_live_trader():
    """Runs a LIVE trading bot that can execute real orders (scaffold)."""
    print("🚀 Starting LIVE Trading Bot...")
    load_dotenv()

    logging.basicConfig(
        filename=os.getenv("LIVE_TRADER_LOG", "live_trader.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    api_key = os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Missing BINANCE_KEY/BINANCE_SECRET in env.")
    client = Client(api_key, api_secret)

    endpoint_name = os.getenv("ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("ENDPOINT_NAME not set in env.")
    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    SYMBOL = os.getenv("SYMBOL", "ETHUSDT")
    QUOTE_USD = float(os.getenv("TRADE_QUANTITY_USDT", "20"))
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "1.5"))
    STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT", "0.75"))

    in_position = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0

    print(f"Trading {SYMBOL} | Qty=${QUOTE_USD} | TP={TAKE_PROFIT_PCT}% | SL={STOP_LOSS_PCT}%")
    logging.info("Session started")

    while True:
        try:
            k = client.get_klines(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            kline = {
                'date': pd.to_datetime(k[0], unit='ms'),
                'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]),
                'volume': float(k[5]),
            }
            current_price = kline['close']

            if in_position:
                if current_price >= take_profit_price:
                    logging.info(f"TP hit at {current_price:.2f}")
                    # TODO: place real SELL order
                    in_position = False
                elif current_price <= stop_loss_price:
                    logging.info(f"SL hit at {current_price:.2f}")
                    # TODO: place real SELL order
                    in_position = False
            else:
                res = signal_gen.get_signal(kline)
                sig = res.get('signal', 0)
                if sig == 1:
                    logging.info(f"BUY signal at {current_price:.2f}")
                    # TODO: place real BUY order for QUOTE_USD
                    in_position = True
                    entry_price = current_price
                    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT/100.0)
                    stop_loss_price   = entry_price * (1 - STOP_LOSS_PCT/100.0)

            time.sleep(60)
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_live_trader()
