import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv
from utils import SignalGenerator, get_client_binance

def run_live_trader():
    """
    Runs a LIVE trading bot that can execute real orders.
    """
    print("🚀 Starting LIVE Trading Bot...")
    load_dotenv()

    # --- Setup Logging ---
    logging.basicConfig(
        filename='live_trade.log', 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Get Environment Variables & Initialize Clients ---
    endpoint_name = os.getenv("ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("ENDPOINT_NAME not set in .env file.")
    
    # Ensure TESTNET is NOT set or is false in your .env file
    try:
        client = get_client_binance()
        print("✅ Successfully connected to LIVE Binance API.")
    except Exception as e:
        print(f"❌ Error connecting to Binance: {e}")
        logging.error(f"Error connecting to Binance: {e}")
        return

    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    # --- Trading Parameters ---
    SYMBOL = 'ETHUSDT'
    TRADE_QUANTITY_USDT = 20 # Amount in USDT to trade each time
    TAKE_PROFIT_PCT = 1.5
    STOP_LOSS_PCT = 0.75
    
    # --- Trading State ---
    in_position = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0
    
    print(f"Trading Parameters: Quantity=${TRADE_QUANTITY_USDT}, TP={TAKE_PROFIT_PCT}%, SL={STOP_LOSS_PCT}%")
    logging.info(f"--- Starting New LIVE Trading Session ---")
    logging.info(f"Parameters: Quantity=${TRADE_QUANTITY_USDT}, TP={TAKE_PROFIT_PCT}%, SL={STOP_LOSS_PCT}%")

    # --- Main Trading Loop ---
    while True:
        try:
            latest_kline = client.get_klines(symbol=SYMBOL, interval=client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            kline_data = {
                'date': pd.to_datetime(latest_kline[0], unit='ms'),
                'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
                'low': float(latest_kline[3]), 'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }
            current_price = kline_data['close']
            current_high = kline_data['high']
            current_low = kline_data['low']

            print(f"\rChecking {kline_data['date']} | Price: ${current_price:.2f} | In Position: {in_position}", end="")

            # --- EXIT LOGIC ---
            if in_position:
                exit_reason = ""
                if current_high >= take_profit_price:
                    exit_reason = f"✅ EXECUTING SELL (TAKE PROFIT) at ~${take_profit_price:.2f}"
                elif current_low <= stop_loss_price:
                    exit_reason = f"❌ EXECUTING SELL (STOP LOSS) at ~${stop_loss_price:.2f}"

                if exit_reason:
                    print(f"\n{exit_reason}")
                    logging.info(exit_reason)
                    # =================================================================
                    # !!! IMPORTANT: REAL SELL ORDER EXECUTION LOGIC GOES HERE !!!
                    # Example: 
                    # quantity_to_sell = TRADE_QUANTITY_USDT / entry_price 
                    # client.create_order(symbol=SYMBOL, side='SELL', type='MARKET', quantity=round(quantity_to_sell, 5))
                    # =================================================================
                    in_position = False

            # --- ENTRY LOGIC ---
            if not in_position:
                result = signal_gen.get_signal(kline_data)
                signal = result.get('signal')

                if signal == 1:
                    log_msg = f"📈 BUY SIGNAL at ~${current_price:.2f}"
                    print(f"\n{log_msg}")
                    logging.info(log_msg)
                    # =================================================================
                    # !!! IMPORTANT: REAL BUY ORDER EXECUTION LOGIC GOES HERE !!!
                    # Example: 
                    # client.create_order(symbol=SYMBOL, side='BUY', type='MARKET', quoteOrderQty=TRADE_QUANTITY_USDT)
                    # =================================================================
                    in_position = True
                    entry_price = current_price
                    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                    logging.info(f"   -> New Position: TP={take_profit_price:.2f}, SL={stop_loss_price:.2f}")

            time.sleep(60)

        except Exception as e:
            error_msg = f"An error occurred in the main loop: {e}"
            print(f"\n{error_msg}")
            logging.error(error_msg)
            time.sleep(60)

if __name__ == "__main__":
    run_live_trader()
