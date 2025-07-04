import os
import time
import logging
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from run_live_check import SignalGenerator # <-- Import the new SignalGenerator

# ==== ENV & Logging ====
load_dotenv()
API_KEY = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
client = Client(API_KEY, API_SECRET, testnet=True) # Use testnet=True for clarity
logging.basicConfig(filename='paper_trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# ==== Paper Trading ====
def run_multiple_trades(n=10, interval=60):
    """
    Runs a paper trading simulation using the SignalGenerator.
    """
    total_profit = 0.0
    trade_count = 0
    
    # 1. Initialize the Signal Generator
    print("Initializing Signal Generator for paper trading...")
    signal_gen = SignalGenerator(
        model_path='eth_lstm_model.h5', 
        scaler_path='scaler.pkl', 
        meta_path='model_meta.json'
    )
    
    # 2. Pre-fill history buffer for feature calculation
    print("Pre-filling history buffer with the last ~250 minutes of data...")
    # Fetch a bit more than history_size to be safe
    klines = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
    for kline in klines[:-1]: # Add all but the last one to the history
        kline_data = {
            'open': float(kline[1]), 'high': float(kline[2]), 
            'low': float(kline[3]), 'close': float(kline[4]), 
            'volume': float(kline[5])
        }
        signal_gen.history.append(kline_data)

    print("History buffer filled. Starting trade simulation...")
    
    for i in range(n):
        try:
            # 3. Get the latest kline
            latest_kline = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=1)[0]
            new_kline_data = {
                'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
                'low': float(latest_kline[3]), 'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }

            # 4. Get the signal from the generator
            result = signal_gen.get_signal(new_kline_data)
            
            current_price = new_kline_data['close']
            print(f"[{i+1}/{n}] | Price: {current_price:.2f} | Confidence: {result['confidence']} | Signal: {result['signal']}")

            if result.get("signal") == 1:
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

        time.sleep(interval)

    print(f"\n📊 Simulation Complete. Trades executed: {trade_count}/{n}")
    print(f"💰 Total simulated PnL: {total_profit:.2f} USD")

# ==== Entry ====
if __name__ == "__main__":
    run_multiple_trades(n=10, interval=60)