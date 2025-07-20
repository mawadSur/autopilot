import os
import time
import logging
import pandas as pd
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from utils import SignalGenerator


# ==== ENV & Logging ====
load_dotenv()
API_KEY = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME") 
client = Client(API_KEY, API_SECRET, testnet=True) # Use testnet=True for clarity
logging.basicConfig(filename='paper_trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# ==== Paper Trading ====
def run_paper_trading(n=10, interval_seconds=60):
    """
    Runs a paper trading simulation using the SageMaker endpoint via SignalGenerator.
    """
    total_profit = 0.0
    trade_count = 0
    
    # 1. Initialize the Signal Generator correctly
    print("Initializing Signal Generator for paper trading...")
    if not ENDPOINT_NAME:
        raise ValueError("ENDPOINT_NAME environment variable not set. Please check your .env file.")
    
    signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME) # <-- CORRECTED INITIALIZATION
    
    # 2. Pre-fill history buffer for feature calculation
    print(f"Pre-filling history buffer with the last {signal_gen.history_size} minutes of data...")
    klines = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
    
    # CORRECTED: Loop to include the 'date' field
    for kline in klines:
        kline_data = {
            'date': pd.to_datetime(kline[0], unit='ms'),
            'open': float(kline[1]), 'high': float(kline[2]), 
            'low': float(kline[3]), 'close': float(kline[4]), 
            'volume': float(kline[5])
        }
        signal_gen.history.append(kline_data)

    print("✅ History buffer filled. Starting trade simulation...")
    
    for i in range(n):
        try:
            # 3. Get the latest kline
            latest_kline = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=1)[0]
            
            # CORRECTED: new_kline_data dictionary to include 'date'
            new_kline_data = {
                'date': pd.to_datetime(latest_kline[0], unit='ms'),
                'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
                'low': float(latest_kline[3]), 'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }

            # 4. Get the signal from the generator
            result = signal_gen.get_signal(new_kline_data)
            
            current_price = new_kline_data['close']
            confidence = result.get('confidence', 0)
            signal = result.get('signal', 0)
            
            print(f"[{i+1}/{n}] | Price: {current_price:.2f} | Confidence: {confidence:.3f} | Signal: {signal}")

            if signal == 1:
                entry = current_price
                simulated_exit = entry * 1.01  # Simulate a 1% take-profit
                pnl = simulated_exit - entry
                total_profit += pnl
                trade_count += 1
                
                log_msg = f"Trade {trade_count}: BUY at {entry:.2f} -> Simulated Sell at {simulated_exit:.2f} | PnL: {pnl:.2f}"
                print(f"✅ {log_msg}")
                logging.info(log_msg)
            else:
                print("🚫 No trade signal. Holding.")

        except Exception as e:
            error_msg = f"Error on trade iteration {i+1}: {e}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)

        time.sleep(interval_seconds)

    print(f"\n📊 Simulation Complete. Trades executed: {trade_count}/{n}")
    print(f"💰 Total simulated PnL: {total_profit:.2f} USD")

# ==== Entry ====
if __name__ == "__main__":
    run_paper_trading(n=20, interval_seconds=60)