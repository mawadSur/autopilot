# backtest.py
import os
import pandas as pd
from tqdm import tqdm

# --- CORRECTED IMPORTS ---
from utils import load_ohlc_chunks, SignalGenerator

def run_signal_generation(output_path='eth_signals.csv'):
    """
    Loads historical data, generates signals for each timestep using the SageMaker endpoint,
    and saves the combined data to a CSV for backtesting.
    """
    print("📈 Starting signal generation process...")
    
    # 1. Load all historical data
    local_data_path = 'eth_1m_data' 
    data_generator = load_ohlc_chunks(folder=local_data_path, chunk_mode=True)
    df_full = pd.concat(data_generator, ignore_index=True)
    
    # --- CORRECTED INITIALIZATION ---
    # 2. Initialize the Signal Generator to use the SageMaker Endpoint
    # !!! IMPORTANT: Replace this with your actual endpoint name !!!
    ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
    signal_gen = SignalGenerator(endpoint_name=ENDPOINT_NAME)
    
    results = []
    
    print(f"Generating signals for {len(df_full)} data points. This may take a while...")
    
    # 3. Iterate through the historical data and generate signals
    for index, row in tqdm(df_full.iterrows(), total=df_full.shape[0]):
        # The SignalGenerator needs a dictionary with these keys
        kline_data = {
            'date': row['date'], # Pass the date for feature engineering
            'open': row['open'], 
            'high': row['high'],
            'low': row['low'], 
            'close': row['close'],
            'volume': row['volume']
        }
        
        signal_result = signal_gen.get_signal(kline_data)
        
        results.append({
            'date': row['date'],
            'close': row['close'],
            'signal': signal_result.get('signal', 0),
            'confidence': signal_result.get('confidence')
        })
        
    # 4. Create a DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    df_results.set_index('date', inplace=True)
    
    print(f"\nSaving signals to {output_path}...")
    df_results.to_csv(output_path)
    
    print(f"✅ Signal generation complete. File saved to {output_path}.")
    print("\n--- Signal Distribution ---")
    print(df_results['signal'].value_counts(normalize=True))
    print("--------------------------")

if __name__ == '__main__':
    run_signal_generation()