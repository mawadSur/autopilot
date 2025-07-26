import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv
from utils import SignalGenerator, get_client_binance

def run_paper_trader():
    """
    Runs an enhanced paper trading bot that uses dynamic exits and simulates fees and slippage.
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
    
    try:
        client = get_client_binance() 
        print("✅ Successfully connected to Binance Testnet API.")
    except Exception as e:
        print(f"❌ Error connecting to Binance: {e}")
        logging.error(f"Error connecting to Binance: {e}")
        return

    # Note: Assumes SignalGenerator is updated to return ATR
    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    # --- Trading Parameters ---
    SYMBOL = 'ETHUSDT'
    TRADING_FEE_PCT = 0.075     # Fee for a single trade (buy or sell)
    SLIPPAGE_PCT = 0.02         # Estimated price slippage per trade
    
    # --- NEW: ATR-based Exit Parameters ---
    # Instead of fixed percentages, we use multiples of the Average True Range (ATR)
    ATR_MULTIPLIER_TP = 2.5     # Take-Profit is 2.5x ATR above entry
    ATR_MULTIPLIER_SL = 1.5     # Stop-Loss is 1.5x ATR below entry
    
    # --- Trading State & Performance Tracking ---
    in_position = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0
    
    trade_count = 0
    winning_trades = 0
    total_pnl_pct = 0.0
    
    print(f"Trading Parameters: ATR Multipliers TP={ATR_MULTIPLIER_TP}x, SL={ATR_MULTIPLIER_SL}x")
    logging.info(f"--- Starting New Paper Trading Session ---")
    logging.info(f"Parameters: ATR TP={ATR_MULTIPLIER_TP}x, SL={ATR_MULTIPLIER_SL}x, Fee={TRADING_FEE_PCT}%, Slippage={SLIPPAGE_PCT}%")

    # --- Main Trading Loop ---
    while True:
        try:
            latest_kline = client.get_klines(symbol=SYMBOL, interval=client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            kline_data = {
                'date': latest_kline[0], # Keep as timestamp for SignalGenerator
                'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
                'low': float(latest_kline[3]), 'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }
            current_price = kline_data['close']
            current_high = kline_data['high']
            current_low = kline_data['low']
            
            # For display purposes
            current_time_display = pd.to_datetime(kline_data['date'], unit='ms')

            print(f"\rChecking {current_time_display} | Price: ${current_price:.2f} | In Position: {in_position}", end="")

            # --- EXIT LOGIC ---
            if in_position:
                exit_price = 0.0
                exit_reason = ""

                # Check for Stop-Loss first
                if current_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = f"❌ SELL (STOP LOSS) at ~${exit_price:.2f}"
                
                # Check for Take-Profit
                elif current_high >= take_profit_price:
                    exit_price = take_profit_price
                    winning_trades += 1
                    exit_reason = f"✅ SELL (TAKE PROFIT) at ~${exit_price:.2f}"

                if exit_reason:
                    # Simulate slippage on the exit
                    actual_exit_price = exit_price * (1 - SLIPPAGE_PCT / 100)
                    # Calculate PnL based on actual entry and exit prices
                    pnl_pct = ((actual_exit_price / entry_price) - 1) * 100
                    # Subtract fees for both entry and exit trades
                    net_pnl_pct = pnl_pct - (2 * TRADING_FEE_PCT)
                    
                    total_pnl_pct += net_pnl_pct
                    
                    log_msg = f"{exit_reason} | Net PnL: {net_pnl_pct:.3f}%"
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
                
                # To use ATR, we need to get it from the signal generator
                # This assumes `get_signal` is modified to return 'atr'
                latest_atr = result.get('atr')

                if signal == 1 and latest_atr is not None:
                    trade_count += 1
                    # Simulate slippage on entry price (we buy slightly higher)
                    entry_price = current_price * (1 + SLIPPAGE_PCT / 100)
                    
                    # Set dynamic TP and SL based on ATR
                    take_profit_price = entry_price + (ATR_MULTIPLIER_TP * latest_atr)
                    stop_loss_price = entry_price - (ATR_MULTIPLIER_SL * latest_atr)
                    
                    in_position = True
                    
                    log_msg = (f"📈 BUY #{trade_count} at ~${entry_price:.2f} (ATR: {latest_atr:.4f}) | "
                               f"TP: {take_profit_price:.2f}, SL: {stop_loss_price:.2f}")
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