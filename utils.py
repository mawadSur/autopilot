from glob import glob
import os
import pandas as pd
import numpy as np

def load_ohlc_chunks(folder='eth_1m_data'):
    # --- CORRECTED: Updated glob pattern to find new 1-minute data files ---
    files = sorted(glob(os.path.join(folder, '*.csv')))
    
    if not files:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")

    print(f"📁 Found {len(files)} CSV files in {folder}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['date'], index_col='date')
            if not df.empty:
                dfs.append(df)
            else:
                print(f"⚠️ Skipping empty file: {f}")
        except Exception as e:
            print(f"⚠️ Skipping corrupted file: {f} | Error: {e}")

    if not dfs:
        raise ValueError("No valid dataframes to concatenate. Please check your CSV files.")

    # Concatenate, sort, and remove any potential duplicates from overlapping files
    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df

def compute_rsi(series, period=14):
    """Computes a more stable Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    
    # --- CORRECTED: Add epsilon to prevent division by zero ---
    rs = gain / (loss + 1e-9) 
    
    rsi = 100 - (100 / (1 + rs))
    return rsi