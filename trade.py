import os
import time
import logging
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from run_live_check import SignalGenerator

# ==== CONFIG ====
TRADE_SYMBOL = 'ETHUSDT'
TRADE_QUANTITY_USDT = 15  # Example: trade with $15 worth of ETH

# ==== ENV & Logging ====
load_dotenv()
API_KEY = os.getenv("BINANCE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
client = Client(API_KEY, API_SECRET)
logging.basicConfig(filename='live_trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')


def trade_live(interval=60):
    """
    Runs the live trading bot using the SignalGenerator.
    """
    position_open = False
    
    # 1. Initialize the Signal Generator
    print("Initializing Signal Generator for LIVE trading...")
    signal_gen = SignalGenerator(
        model_path='eth_lstm_model.h5', 
        scaler_path='scaler.pkl', 
        meta_path='model_meta.json'
    )
    
    # 2. Pre-fill history buffer
    print(f"Pre-filling history buffer with the last {signal_gen.history_size} minutes of data...")
    klines = client.get_klines(symbol=TRADE_SYMBOL, interval=KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
    for kline in klines:
        kline_data = {
            'open': float(kline[1]), 'high': float(kline[2]), 
            'low': float(kline[3]), 'close': float(kline[4]), 
            'volume': float(kline[5])
        }
        signal_gen.history.append(kline_data)

    print("✅ History buffer filled. Starting live trading loop...")

    while True:
        try:
            # 3. Get the latest kline
            latest_kline = client.get_klines(symbol=TRADE_SYMBOL, interval=KLINE_INTERVAL_1MINUTE, limit=1)[0]
            new_kline_data = {
                'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
                'low': float(latest_kline[3]), 'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }

            if not position_open:
                # 4. Get signal if we are looking to enter a position
                result = signal_gen.get_signal(new_kline_data)
                
                current_price = new_kline_data['close']
                print(f"Price: {current_price:.2f} | Confidence: {result['confidence']} | Signal: {result['signal']}")

                if result.get("signal") == 1:
                    # --- EXECUTE BUY ORDER ---
                    print(f"🚀 BUY signal received! Executing market buy for {TRADE_QUANTITY_USDT} USDT...")
                    # To place a real order, uncomment the following lines:
                    # order = client.create_order(
                    #     symbol=TRADE_SYMBOL,
                    #     side=SIDE_BUY,
                    #     type=ORDER_TYPE_MARKET,
                    #     quoteOrderQty=TRADE_QUANTITY_USDT
                    # )
                    # logging.info(f"BUY ORDER PLACED: {order}")
                    print("--- (Simulated Buy for Safety) ---")
                    position_open = True
            else:
                # --- LOGIC TO CHECK FOR EXIT ---
                # (You would add your take-profit/stop-loss logic here)
                print("Position is open. Waiting for exit condition...")
                # For this example, we'll just wait for the next interval.
                # In a real system, you'd check if the price hit your TP/SL.

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)

        time.sleep(interval)

# ==== Entry ====
if __name__ == "__main__":
    trade_live()