import os
import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================

def load_ohlc_chunks(folder, chunk_mode=False):
    """Loads and concatenates CSV files from a directory."""
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
            df = pd.read_csv(f, header=None, names=column_names)
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            
            if not df.empty:
                if chunk_mode:
                    yield df
                else:
                    all_dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Could not process file {f}: {e}")

    if not chunk_mode:
        if not all_dfs:
            return pd.DataFrame() 
        return pd.concat(all_dfs, ignore_index=True)

def debug_date_parsing(df):
    """Helper to debug date parsing issues."""
    df['date'] = pd.to_datetime(df['date'], unit='ms', errors='coerce')
    if df['date'].isnull().any():
        print("Warning: Null dates found after parsing.")
        df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    if not df.index.is_monotonic_increasing:
        print("Warning: Index is not monotonically increasing.")
        df = df[~df.index.duplicated(keep='first')]
    return df.reset_index()

def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    """Compute ATR."""
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df['tr'].rolling(period).mean()

def preprocess_data(df, feature_cols, window_size, lookahead_period):
    """
    Engineers features, creates labels based on risk/reward, scales data,
    and creates windowed sequences for the LSTM.
    """
    print("Preprocessing data...")
    # --- Feature Engineering ---
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / (df['sma_50'] + 1e-9) - 1
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change()
    df['atr'] = compute_atr(df, period=14)
    df_hourly = df['close'].resample('h').mean()
    hourly_ema = df_hourly.ewm(span=20).mean()
    df['hourly_ema_20'] = hourly_ema.reindex(df.index, method='ffill')
    df['price_vs_hourly_trend'] = (df['close'] - df['hourly_ema_20']) / (df['hourly_ema_20'] + 1e-9)
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_width'] = ((df['bb_mid'] + 2 * df['bb_std']) - (df['bb_mid'] - 2 * df['bb_std'])) / (df['bb_mid'] + 1e-9)
    
    # --- OPTIMIZED LABELING STRATEGY (RISK/REWARD) ---
    future_highs = df['high'].rolling(window=lookahead_period).max().shift(-lookahead_period)
    future_lows = df['low'].rolling(window=lookahead_period).min().shift(-lookahead_period)
    potential_profit = future_highs - df['close']
    potential_loss = df['close'] - future_lows
    profit_threshold = 0.001
    risk_reward_ratio = 1.5
    
    # --- THIS IS THE CORRECTED LINE ---
    df['label'] = ((potential_profit > profit_threshold) & (potential_profit > risk_reward_ratio * potential_loss)).astype(int)
    
    # --- Data Cleaning & Scaling ---
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- Create Windowed Data ---
    X_list, y_list = [], []
    features = df[feature_cols].values
    labels = df['label'].values
    for i in range(len(features) - window_size):
        X_list.append(features[i:i+window_size])
        y_list.append(labels[i+window_size])
        
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1), scaler

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size) # *2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    full_df = pd.concat(load_ohlc_chunks(folder=args.train, chunk_mode=True), ignore_index=True)
    full_df = debug_date_parsing(full_df)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
        'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
    ]
    X, y, scaler = preprocess_data(full_df, feature_cols, args.window_size, args.lookahead_period)
    
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    pos_weight_value = neg_count / pos_count if pos_count > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    print(f"Dataset balanced. Negative (0): {neg_count}, Positive (1): {pos_count}. Weight: {pos_weight_value:.2f}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    model = LSTMModel(len(feature_cols), 128, 3, 1, args.dropout_rate).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved to {best_model_path}")
            
    print("Training complete.")

# ==============================================================================
# 4. SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--window-size', type=int, default=150)
    parser.add_argument('--lookahead-period', type=int, default=10)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    
    args = parser.parse_args()
    main(args)