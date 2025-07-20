import os
import time
import logging
import pandas as pd
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv
from utils import SignalGenerator

# ==== Configuration ====
load_dotenv()
API_KEY = os.getenv("BINANCE_TESTNET_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")

# --- Trading Parameters ---
TAKE_PROFIT_PCT = 1.0  # Take profit at 1.0% gain
STOP_LOSS_PCT = 0.5    # Stop loss at 0.5% loss
TRADING_FEE_PCT = 0.075 # Fee for both entry and exit

# ==== Setup ====
client = Client(API_KEY, API_SECRET, testnet=True)
logging.basicConfig(filename='paper_trade_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# ==== Paper Trading Logic ====
def run_paper_trading(interval_seconds=60):
    """
    Runs a paper trading simulation using the SageMaker endpoint.
    """
    if not ENDPOINT_NAME:
        raise ValueError("ENDPOINT_NAME environment variable not set. Please check your .env file.")

    signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
    
    # Pre-fill history buffer
    print(f"Pre-filling history buffer with the last {signal_gen.history_size} minutes of data...")
    klines = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=signal_gen.history_size)
    for kline in klines:
        kline_data = {
            'date': pd.to_datetime(kline[0], unit='ms'), 'open': float(kline[1]), 
            'high': float(kline[2]), 'low': float(kline[3]), 
            'close': float(kline[4]), 'volume': float(kline[5])
        }
        signal_gen.history.append(kline_data)
    print("✅ History buffer filled. Starting trade simulation...")

    # --- State Variables for Trading ---
    position_open = False
    entry_price = 0.0
    trade_count = 0
    total_pnl_pct = 0.0

    while True:
        try:
            # 1. Get latest market data
            latest_kline = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_1MINUTE, limit=1)[0]
            new_kline_data = {
                'date': pd.to_datetime(latest_kline[0], unit='ms'), 'open': float(latest_kline[1]),
                'high': float(latest_kline[2]), 'low': float(latest_kline[3]),
                'close': float(latest_kline[4]), 'volume': float(latest_kline[5])
            }
            current_price = new_kline_data['close']

            # 2. Check current position status
            if position_open:
                take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                
                # Check for Take Profit
                if current_price >= take_profit_price:
                    pnl = TAKE_PROFIT_PCT - (2 * TRADING_FEE_PCT)
                    total_pnl_pct += pnl
                    log_msg = f"✅ SELL (TP): at {current_price:.2f} | PnL: {pnl:.3f}%"
                    print(log_msg); logging.info(log_msg)
                    position_open = False
                
                # Check for Stop Loss
                elif current_price <= stop_loss_price:
                    pnl = -STOP_LOSS_PCT - TRADING_FEE_PCT # Only one fee on entry
                    total_pnl_pct += pnl
                    log_msg = f"❌ SELL (SL): at {current_price:.2f} | PnL: {pnl:.3f}%"
                    print(log_msg); logging.info(log_msg)
                    position_open = False
                
                else:
                    print(f"HOLDING POSITION | Price: {current_price:.2f} | Entry: {entry_price:.2f}")

            # 3. If no position is open, check for a buy signal
            else:
                result = signal_gen.get_signal(new_kline_data)
                signal = result.get('signal')
                confidence = result.get('confidence', 0)
                
                print(f"Price: {current_price:.2f} | Confidence: {confidence:.3f} | Signal: {signal}")

                if signal == 1:
                    trade_count += 1
                    entry_price = current_price
                    position_open = True
                    log_msg = f"🚀 BUY #{trade_count}: Signal received. Entering at {entry_price:.2f}"
                    print(log_msg); logging.info(log_msg)
                else:
                    print("🚫 No trade signal. Holding.")

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(f"⚠️ {error_msg}"); logging.error(error_msg)

        time.sleep(interval_seconds)

# ==== Script Entry Point ====
if __name__ == "__main__":
    run_paper_trading()