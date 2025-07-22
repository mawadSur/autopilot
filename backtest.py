import os
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from utils import SignalGenerator, load_ohlc_chunks

def run_backtest(df, signal_gen):
    """
    Runs a backtest using a Take-Profit/Stop-Loss strategy.
    """
    print("\n📈 Starting backtest with Take-Profit/Stop-Loss strategy...")

    # --- Realistic Trading Parameters ---
    TRADING_FEE_PCT = 0.075  # Standard fee for Binance (0.075%)
    TAKE_PROFIT_PCT = 1.5    # Target a 1.5% profit on each trade
    STOP_LOSS_PCT = 0.75     # A 0.75% stop-loss for risk management

    # --- Backtest State & Logging ---
    in_position = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0
    
    trades = []
    total_trades = 0
    winning_trades = 0

    # Pre-fill the history buffer to ensure the model has enough data from the start
    initial_history_size = signal_gen.history_size
    initial_history_df = df.head(initial_history_size)
    for _, row in initial_history_df.iterrows():
        signal_gen.history.append(row.to_dict())
    print(f"Pre-filled history buffer with {len(signal_gen.history)} records.")

    # Iterate through the rest of the data, one minute at a time
    for i in tqdm(range(initial_history_size, len(df)), desc="Backtesting"):
        current_row = df.iloc[i]
        current_price = current_row['close']
        current_high = current_row['high']
        current_low = current_row['low']
        
        # ✅ CORRECTED EXIT LOGIC
        if in_position:
            # 1. Check for Stop-Loss first (conservative, "worst-first" approach)
            if current_low <= stop_loss_price:
                exit_price = stop_loss_price
                pnl = ((exit_price - entry_price) / entry_price) * 100 - (2 * TRADING_FEE_PCT)
                trades.append(pnl)
                print(f"\n❌ STOP LOSS at {exit_price:.2f}. PnL: {pnl:.3f}%")
                in_position = False 

            # 2. Only if not stopped out, check for Take-Profit
            elif current_high >= take_profit_price:
                exit_price = take_profit_price
                pnl = ((exit_price - entry_price) / entry_price) * 100 - (2 * TRADING_FEE_PCT)
                trades.append(pnl)
                winning_trades += 1
                print(f"\n✅ TAKE PROFIT at {exit_price:.2f}. PnL: {pnl:.3f}%")
                in_position = False

        # --- ENTRY LOGIC ---
        if not in_position:
            result = signal_gen.get_signal(current_row.to_dict())
            signal = result.get('signal')
            
            if signal == 1:
                total_trades += 1
                in_position = True
                entry_price = current_price
                take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
                stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
                print(f"\n📈 BUY SIGNAL #{total_trades} at {entry_price:.2f} (Time: {current_row['date']})")
                print(f"   -> TP: {take_profit_price:.2f}, SL: {stop_loss_price:.2f}")

    # --- SUMMARY ---
    print("\n--- Backtest Summary ---")
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_pnl = sum(trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        print(f"Total Trades Executed: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Net PnL: {total_pnl:.2f}%")
        print(f"Average PnL per Trade: {avg_pnl:.3f}%")
    else:
        print("No trades were executed during the backtest.")
    print("------------------------")

def main():
    """Main function to run the backtest."""
    load_dotenv()
    
    endpoint_name = os.getenv("ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("ENDPOINT_NAME environment variable not set. Please create a .env file.")

    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    print("Loading all historical data for backtest...")
    all_data = pd.concat(load_ohlc_chunks(folder='eth_1m_data', chunk_mode=True), ignore_index=True)
    all_data['date'] = pd.to_datetime(all_data['date'], unit='ms')
    
    if all_data.empty:
        print("No data found. Exiting backtest.")
        return
        
    run_backtest(all_data, signal_gen)

if __name__ == "__main__":
    main()