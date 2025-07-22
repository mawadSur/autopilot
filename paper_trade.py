import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv
from utils import SignalGenerator, get_client_binance

def run_paper_trader():
    """
    Runs a paper trading bot on the Binance Testnet to simulate the strategy.
    """
    print("🚀 Starting Paper Trading Bot (Testnet)...")
    load_dotenv()

    # --- Setup Logging ---
    logging.basicConfig(
        filename='paper_trade.log', 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # --- Get Environment Variables & Initialize Clients ---
    endpoint_name = os.getenv("ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("ENDPOINT_NAME not set in .env file.")
    
    # Ensure TESTNET=true in your .env file to use the testnet client
    try:
        client = get_client_binance() 
        print("✅ Successfully connected to Binance Testnet API.")
    except Exception as e:
        print(f"❌ Error connecting to Binance: {e}")
        logging.error(f"Error connecting to Binance: {e}")
        return

    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    # --- Trading Parameters ---
    SYMBOL = 'ETHUSDT'
    TAKE_PROFIT_PCT = 1.5
    STOP_LOSS_PCT = 0.75
    TRADING_FEE_PCT = 0.075 # Standard fee for entry and exit
    
    # --- Trading State & Performance Tracking ---
    in_position = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0
    
    trade_count = 0
    winning_trades = 0
    total_pnl_pct = 0.0
    
    print(f"Trading Parameters: TP={TAKE_PROFIT_PCT}%, SL={STOP_LOSS_PCT}%")
    logging.info(f"--- Starting New Paper Trading Session ---")
    logging.info(f"Parameters: TP={TAKE_PROFIT_PCT}%, SL={STOP_LOSS_PCT}%")

    # --- Main Trading Loop ---
    while True:
        try:
            # Fetch the latest completed kline (candle)
            latest_kline = client.get_klines(symbol=SYMBOL, interval=client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            kline_data = {
                'date': pd.to_datetime(latest_kline[0], unit='ms'),
                'open': float(latest_kline[1]),
                'high': float(latest_kline[2]),
                'low': float(latest_kline[3]),
                'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }
            current_price = kline_data['close']
            current_high = kline_data['high']
            current_low = kline_data['low']

            print(f"\rChecking {kline_data['date']} | Price: ${current_price:.2f} | In Position: {in_position}", end="")

            # --- EXIT LOGIC ---
            if in_position:
                pnl = 0.0
                exit_reason = ""

                # Check for Take-Profit
                if current_high >= take_profit_price:
                    pnl = TAKE_PROFIT_PCT - (2 * TRADING_FEE_PCT)
                    winning_trades += 1
                    exit_reason = f"✅ SELL (TAKE PROFIT) at ~${take_profit_price:.2f}"
                
                # Check for Stop-Loss
                elif current_low <= stop_loss_price:
                    pnl = -STOP_LOSS_PCT - (2 * TRADING_FEE_PCT)
                    exit_reason = f"❌ SELL (STOP LOSS) at ~${stop_loss_price:.2f}"

                # If an exit condition was met, log the trade
                if exit_reason:
                    total_pnl_pct += pnl
                    log_msg = f"{exit_reason} | PnL: {pnl:.3f}%"
                    print(f"\n{log_msg}")
                    logging.info(log_msg)
                    in_position = False

                    win_rate = (winning_trades / trade_count) * 100 if trade_count > 0 else 0
                    summary_msg = (f"SUMMARY: Trades={trade_count}, Win Rate={win_rate:.2f}%, "
                                   f"Total PnL={total_pnl_pct:.3f}%")
                    print(summary_msg)
                    logging.info(summary_msg)

            # --- ENTRY LOGIC ---
            if not in_position:
                result = signal_gen.get_signal(kline_data)
                signal = result.get('signal')

                if signal == 1:
                    trade_count += 1
                    entry_price = current_price
                    take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                    stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                    in_position = True
                    
                    log_msg = f"📈 BUY #{trade_count} at ~${current_price:.2f} | TP: {take_profit_price:.2f}, SL: {stop_loss_price:.2f}"
                    print(f"\n{log_msg}")
                    logging.info(log_msg)

            time.sleep(60)

        except Exception as e:
            error_msg = f"An error occurred in the main loop: {e}"
            print(f"\n{error_msg}")
            logging.error(error_msg)
            time.sleep(60)

if __name__ == "__main__":
    run_paper_trader()
