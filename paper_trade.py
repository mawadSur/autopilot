import os
import time
import pandas as pd
import logging
from dotenv import load_dotenv
from utils import SignalGenerator, get_client_binance

def run_paper_trader():
    """
    Runs a paper trading bot that manages a simulated account balance.
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

    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    # --- Trading Parameters ---
    SYMBOL = 'ETHUSDT'
    TRADING_FEE_PCT = 0.075
    SLIPPAGE_PCT = 0.02
    ATR_MULTIPLIER_TP = 2.5
    ATR_MULTIPLIER_SL = 1.5

    # --- Account and Position Sizing ---
    STARTING_BALANCE_USDT = 10000.0
    current_balance_usdt = STARTING_BALANCE_USDT
    asset_quantity = 0.0

    # --- Trading State & Performance Tracking ---
    in_position = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0

    trade_count = 0
    winning_trades = 0

    print(f"Trading Parameters: Start Balance=${STARTING_BALANCE_USDT}, ATR TP={ATR_MULTIPLIER_TP}x, SL={ATR_MULTIPLIER_SL}x")
    logging.info(f"--- Starting New Paper Trading Session ---")
    logging.info(f"Parameters: Start Balance=${STARTING_BALANCE_USDT}, ATR TP={ATR_MULTIPLIER_TP}x, SL={ATR_MULTIPLIER_SL}x, Fee={TRADING_FEE_PCT}%, Slippage={SLIPPAGE_PCT}%")

    # --- Main Trading Loop ---
    while True:
        try:
            latest_kline = client.get_klines(symbol=SYMBOL, interval=client.KLINE_INTERVAL_1MINUTE, limit=1)[0]
            kline_data = {
                'date': latest_kline[0],
                'open': float(latest_kline[1]), 'high': float(latest_kline[2]),
                'low': float(latest_kline[3]), 'close': float(latest_kline[4]),
                'volume': float(latest_kline[5])
            }
            current_price = kline_data['close']
            current_high = kline_data['high']
            current_low = kline_data['low']
            current_time_display = pd.to_datetime(kline_data['date'], unit='ms')

            # Get signal data before printing status
            result = signal_gen.get_signal(kline_data)

            # --- MODIFIED: Verbose Console Logging ---
            # This block creates a detailed status update for the console only.
            status_log = [f"\n--- [ {current_time_display} ] ---"]
            status_log.append(f"  📈 Price:         ${current_price:<8.2f}")
            status_log.append(f"  💰 Balance:       ${current_balance_usdt:.2f}")
            status_log.append(f"  🧠 Model Output:    {result}")

            if in_position:
                status_log.append(f"  🎯 Position Active: TP @ ${take_profit_price:.2f}, SL @ ${stop_loss_price:.2f}")
            else:
                status_log.append("  ⏳ Status:          Awaiting signal...")
            print("\n".join(status_log))


            # --- EXIT LOGIC ---
            if in_position:
                exit_price = 0.0
                exit_reason = ""

                if current_low <= stop_loss_price:
                    exit_price = stop_loss_price
                    exit_reason = f"❌ SELL (STOP LOSS)"
                elif current_high >= take_profit_price:
                    exit_price = take_profit_price
                    winning_trades += 1
                    exit_reason = f"✅ SELL (TAKE PROFIT)"

                if exit_reason:
                    actual_exit_price = exit_price * (1 - SLIPPAGE_PCT / 100)
                    position_value = asset_quantity * actual_exit_price
                    entry_cost = asset_quantity * entry_price
                    entry_fee = entry_cost * (TRADING_FEE_PCT / 100)
                    exit_fee = position_value * (TRADING_FEE_PCT / 100)
                    pnl_usdt = position_value - entry_cost - entry_fee - exit_fee
                    current_balance_usdt += pnl_usdt

                    # Trade action logs are printed to console AND saved to the .log file
                    log_msg = f"{exit_reason} at ~${exit_price:.2f} | Profit: ${pnl_usdt:+.2f}"
                    print(f"\n{log_msg}")
                    logging.info(log_msg)

                    in_position = False
                    asset_quantity = 0.0

                    win_rate = (winning_trades / trade_count) * 100 if trade_count > 0 else 0
                    total_pnl_pct = ((current_balance_usdt / STARTING_BALANCE_USDT) - 1) * 100
                    summary_msg = (f"SUMMARY: Trades={trade_count}, Win Rate={win_rate:.2f}%, "
                                   f"Current Balance=${current_balance_usdt:.2f} (Total PnL: {total_pnl_pct:.2f}%)")
                    print(summary_msg)
                    logging.info(summary_msg)

            # --- ENTRY LOGIC ---
            if not in_position:
                signal = result.get('signal')
                latest_atr = result.get('atr')

                if signal == 1 and latest_atr is not None:
                    trade_count += 1
                    entry_price = current_price * (1 + SLIPPAGE_PCT / 100)
                    trade_size_usdt = current_balance_usdt * 0.995
                    asset_quantity = trade_size_usdt / entry_price
                    take_profit_price = entry_price + (ATR_MULTIPLIER_TP * latest_atr)
                    stop_loss_price = entry_price - (ATR_MULTIPLIER_SL * latest_atr)
                    in_position = True

                    # Trade action logs are printed to console AND saved to the .log file
                    log_msg = (f"📈 BUY #{trade_count} ({asset_quantity:.5f} ETH) at ~${entry_price:.2f} | "
                               f"TP: {take_profit_price:.2f}, SL: {stop_loss_price:.2f}")
                    print(f"\n{log_msg}")
                    logging.info(log_msg)

            # --- MODIFIED: Set update frequency to 1 minute ---
            time.sleep(60)

        except Exception as e:
            error_msg = f"An error occurred in the main loop: {e}"
            print(f"\n{error_msg}")
            logging.error(error_msg)
            time.sleep(60)

if __name__ == "__main__":
    run_paper_trader()