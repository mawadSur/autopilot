import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from glob import glob
from utils import compute_rsi

def preprocess_and_save_batches(df, window_size=150, threshold=0.02, batch_size=500_000, out_dir="labeled_chunks"):
    os.makedirs(out_dir, exist_ok=True)

    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change()
    df.dropna(inplace=True)

    df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['label'] = (df['target'] > threshold).astype(int)
    df.dropna(inplace=True)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]
    features = df[feature_cols].values.astype(np.float32)
    labels = df['label'].values.astype(np.uint8)

    total = len(features)
    batch_idx = 0
    X_batch, y_batch = [], []

    print(f"📈 Generating labeled windows with batch size: {batch_size}")
    for i in tqdm(range(window_size, total), desc="Labeling windows"):
        X_batch.append(features[i - window_size:i])
        y_batch.append(labels[i])

        if len(X_batch) >= batch_size:
            X_arr = np.array(X_batch, dtype=np.float32)
            y_arr = np.array(y_batch, dtype=np.uint8)
            np.savez_compressed(f"{out_dir}/batch_{batch_idx}.npz", X=X_arr, y=y_arr)
            print(f"💾 Saved batch {batch_idx} at index {i} ({df.index[i]})")
            X_batch, y_batch = [], []
            batch_idx += 1

    if X_batch:
        X_arr = np.array(X_batch, dtype=np.float32)
        y_arr = np.array(y_batch, dtype=np.uint8)
        np.savez_compressed(f"{out_dir}/batch_{batch_idx}.npz", X=X_arr, y=y_arr)
        print(f"💾 Saved final batch {batch_idx} at index {total-1} ({df.index[-1]})")

    print(f"✅ Completed batching. Total batches: {batch_idx + 1}")

def load_all_chunks(folder='eth_1s_data'):
    files = sorted(glob(os.path.join(folder, '*.csv')))
    if not files:
        raise ValueError(f"No CSV files found in directory: {folder}")

    dfs = []
    for f in tqdm(files, desc="Loading CSV chunks"):
        try:
            df = pd.read_csv(f, parse_dates=['date'], index_col='date')
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping file {f}: {e}")

    if not dfs:
        raise ValueError("No valid dataframes loaded — cannot concatenate.")

    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated()].sort_index()
    return combined

if __name__ == "__main__":
    df = load_all_chunks()
    preprocess_and_save_batches(df, window_size=150, threshold=0.02)
