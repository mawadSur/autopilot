import os
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from utils import SignalGenerator, load_ohlc_chunks

def run_backtest(df, signal_gen, lookahead_steps=10):
    """
    Runs a backtest on historical data using signals from the SignalGenerator.
    """
    print("\n📈 Starting backtest...")
    
    # --- Realistic Trading Parameters ---
    TRADING_FEE_PCT = 0.075  # Standard fee for Binance (0.075%)
    STOP_LOSS_PCT = 0.5      # A 0.5% stop-loss for risk management
    
    trade_count = 0
    total_pnl = 0.0
    
    # Pre-fill history buffer from the start of the dataframe
    initial_history_size = signal_gen.history_size
    initial_history_df = df.head(initial_history_size)
    for _, row in initial_history_df.iterrows():
        signal_gen.history.append(row.to_dict())
    print(f"Pre-filled history buffer with {len(signal_gen.history)} records.")

    # Iterate through the rest of the data
    for i in tqdm(range(initial_history_size, len(df) - lookahead_steps), desc="Backtesting"):
        current_row = df.iloc[i]
        
        # Get signal from the generator
        result = signal_gen.get_signal(current_row.to_dict())
        signal = result.get('signal')
        
        if signal == 1:
            trade_count += 1
            entry_price = current_row['close']
            
            log_msg_entry = f"Trade #{trade_count}: BUY signal at {entry_price:.2f} (Time: {current_row['date']})"
            print(f"\n{log_msg_entry}")
            
            # --- Check for stop-loss or take-profit over the lookahead window ---
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
            
            # Find the lowest low in the lookahead period
            lookahead_window = df.iloc[i + 1 : i + 1 + lookahead_steps]
            low_in_period = lookahead_window['low'].min()
            
            pnl = 0.0
            
            # Did the price hit our stop-loss?
            if low_in_period <= stop_loss_price:
                pnl = -STOP_LOSS_PCT # PnL is the stop-loss percentage
                pnl -= TRADING_FEE_PCT # Only pay entry fee if stopped out immediately
                log_msg_outcome = f"  -> ❌ Trade STOPPED OUT. PnL: {pnl:.3f}%"
            else:
                # If not stopped out, sell at the price at the end of the lookahead period
                exit_price = df['close'].iloc[i + lookahead_steps]
                pnl = ((exit_price - entry_price) / entry_price) * 100
                pnl -= (2 * TRADING_FEE_PCT) # Subtract fees for both buy and sell orders
                log_msg_outcome = f"  -> ✅ Trade closed at {exit_price:.2f}. PnL: {pnl:.3f}%"
            
            print(log_msg_outcome)
            total_pnl += pnl

    print("\n--- Backtest Summary ---")
    print(f"Total Trades Executed: {trade_count}")
    print(f"Total Net PnL: {total_pnl:.2f}%")
    if trade_count > 0:
        print(f"Average PnL per Trade: {total_pnl / trade_count:.3f}%")
    print("------------------------")

def main():
    """Main function to run the backtest."""
    load_dotenv()
    
    # Ensure the environment variable for the endpoint is set
    endpoint_name = os.getenv("ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("ENDPOINT_NAME environment variable not set. Please create a .env file.")

    # Initialize the SignalGenerator
    signal_gen = SignalGenerator(endpoint_name=endpoint_name)

    # Load all historical data for the backtest
    print("Loading all historical data for backtest...")
    # Use the same data folder your training job used
    all_data = pd.concat(load_ohlc_chunks(folder='eth_1m_data', chunk_mode=True), ignore_index=True)
    all_data['date'] = pd.to_datetime(all_data['date'], unit='ms')
    
    if all_data.empty:
        print("No data found. Exiting backtest.")
        return
        
    run_backtest(all_data, signal_gen)

if __name__ == "__main__":
    main()