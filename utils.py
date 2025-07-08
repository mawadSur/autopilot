# utils.py

import os
import pandas as pd

def load_ohlc_chunks(folder, chunk_mode=False):
    """
    Loads OHLC data from a folder of CSV files, searching recursively,
    and assigning column names. This version is robust against bad data.
    """
    print(f"[DEBUG] Loading data from: {folder}")
    files = []
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith('.csv'):
                files.append(os.path.join(dirpath, f))

    if not files:
        raise FileNotFoundError(f"No .csv files found recursively in folder: {folder}")

    column_names = ['date', 'open', 'high', 'low', 'close', 'volume']
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    all_dfs = []
    for f in sorted(files):
        try:
            # Read data without forcing types initially
            df = pd.read_csv(f, header=None, names=column_names)
            
            # Attempt to convert numeric columns, turning errors into NaN
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows that now contain NaN due to conversion errors
            df.dropna(inplace=True)
            
            if not df.empty:
                 # Yield or append the cleaned DataFrame
                if chunk_mode:
                    yield df
                else:
                    all_dfs.append(df)
            else:
                print(f"[WARN] File {f} was empty after cleaning and was skipped.")

        except Exception as e:
            print(f"[ERROR] Could not process file {f}: {e}")

    if not chunk_mode:
        if not all_dfs:
            # This will only happen if all files were empty or invalid
            return pd.DataFrame() 
        return pd.concat(all_dfs)


def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, period=14):
    """Computes the Average True Range (ATR)."""
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = df['tr'].rolling(period).mean()
    return atr