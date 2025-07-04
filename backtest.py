import pandas as pd
from tqdm import tqdm
from utils import load_ohlc_chunks
from run_live_check import SignalGenerator

def run_signal_generation(output_path='eth_signals.csv'):
    """
    Loads historical data, generates signals for each timestep, 
    and saves the combined data to a CSV for backtesting.
    """
    print("📈 Starting signal generation process...")
    
    # 1. Load all historical data
    df_full = load_ohlc_chunks()
    
    # 2. Initialize the Signal Generator
    signal_gen = SignalGenerator(
        model_path='eth_lstm_model.h5', 
        scaler_path='scaler.pkl', 
        meta_path='model_meta.json'
    )
    
    results = []
    
    # 3. Iterate through the historical data and generate signals
    print(f"Generating signals for {len(df_full)} data points. This may take a while...")
    
    # Use tqdm for a progress bar
    for index, row in tqdm(df_full.iterrows(), total=df_full.shape[0]):
        kline_data = {
            'open': row['open'], 'high': row['high'],
            'low': row['low'], 'close': row['close'],
            'volume': row['volume']
        }
        
        # Get signal for the current kline
        signal_result = signal_gen.get_signal(kline_data)
        
        # Append results
        results.append({
            'date': index,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
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
    print(df_results['signal'].value_counts())
    print("--------------------------")

if __name__ == '__main__':
    run_signal_generation()