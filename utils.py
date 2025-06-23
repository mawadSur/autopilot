from glob import glob
import os
import pandas as pd

def load_ohlc_chunks(folder='eth_1s_data'):
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

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated()]
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))