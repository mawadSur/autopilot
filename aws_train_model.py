import os
import argparse
import json
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================

def load_ohlc_data(folder):
    """Loads and concatenates all CSV files from a directory into a single DataFrame."""
    print(f"--- Loading data from: {folder} ---")
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(folder) for f in fn if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"No .csv files found in folder: {folder}")

    df_list = []
    for f in sorted(files):
        try:
            # Assuming files have no header
            df = pd.read_csv(f, header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])
            df_list.append(df)
        except Exception as e:
            print(f"[Warning] Could not process file {f}: {e}")
            
    if not df_list:
        raise ValueError("No data could be loaded from the provided files.")

    full_df = pd.concat(df_list, ignore_index=True)
    
    # --- Data Cleaning and Date Handling ---
    full_df['date'] = pd.to_datetime(full_df['date'], unit='ms', errors='coerce')
    full_df.dropna(subset=['date'], inplace=True)
    full_df.set_index('date', inplace=True)
    full_df.drop_duplicates(inplace=True)
    full_df.sort_index(inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    full_df[numeric_cols] = full_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    full_df.dropna(inplace=True)

    print(f"--- Data loaded successfully. Shape: {full_df.shape} ---")
    return full_df

def compute_technical_indicators(df):
    """Computes technical indicators for the given DataFrame."""
    
    # Basic candle features
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()

    # Moving Averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / (df['sma_50'] + 1e-9) - 1
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Momentum and Volatility
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['vol_change'] = df['volume'].pct_change()
    
    tr = pd.DataFrame(index=df.index)
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['atr'] = tr.max(axis=1).rolling(14).mean()

    # Trend Analysis
    hourly_index = df.index.floor('h')
    df_hourly = df['close'].groupby(hourly_index).mean()
    hourly_ema = df_hourly.ewm(span=20, adjust=False).mean()
    df['hourly_ema_20'] = hourly_ema.reindex(hourly_index, method='ffill').values
    df['price_vs_hourly_trend'] = (df['close'] - df['hourly_ema_20']) / (df['hourly_ema_20'] + 1e-9)
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_width'] = ((df['bb_mid'] + 2 * bb_std) - (df['bb_mid'] - 2 * bb_std)) / (df['bb_mid'] + 1e-9)
    
    return df

def create_labels(df, lookahead_period, risk_reward_ratio, profit_threshold_pct):
    """Creates binary labels based on a forward-looking profit/loss strategy."""
    future_highs = df['high'].rolling(window=lookahead_period).max().shift(-lookahead_period)
    future_lows = df['low'].rolling(window=lookahead_period).min().shift(-lookahead_period)
    
    potential_profit = future_highs - df['close']
    potential_loss = df['close'] - future_lows
    
    # Dynamic profit threshold based on the current price
    profit_target = df['close'] * (profit_threshold_pct / 100.0)

    # Label is 1 (buy) if the potential profit meets the threshold and R/R ratio
    df['label'] = (
        (potential_profit >= profit_target) &
        (potential_loss > 0) & # Avoid division by zero
        (potential_profit / (potential_loss + 1e-9) >= risk_reward_ratio)
    ).astype(int)
    
    return df

# ==============================================================================
# 2. MODEL AND DATASET DEFINITION
# ==============================================================================

class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time-series windowing."""
    def __init__(self, features, labels, window_size):
        self.features = features
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.features) - self.window_size

    def __getitem__(self, idx):
        # A window of features and the label corresponding to the end of that window
        feature_window = self.features[idx : idx + self.window_size]
        label = self.labels[idx + self.window_size]
        return torch.tensor(feature_window, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).view(1)

class LSTMModel(nn.Module):
    """Bidirectional LSTM model for classification."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        # The output feature size is hidden_size * 2 because it's bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # We only care about the output of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# ==============================================================================
# 3. TRAINING SCRIPT
# ==============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")
    
    # 1. Load and prepare data
    full_df = load_ohlc_data(folder=args.train)
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
        'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
    ]
    
    # 2. Feature engineering and labeling on the whole dataset
    df_with_features = compute_technical_indicators(full_df)
    df_labeled = create_labels(
        df_with_features, 
        args.lookahead_period,
        args.risk_reward_ratio,
        args.profit_threshold_pct
    )
    
    # Drop NaNs created by rolling indicators and labeling lookahead
    df_final = df_labeled.dropna().copy()
    df_final.replace([np.inf, -np.inf], 0, inplace=True)

    # 3. **CRITICAL FIX**: Split data chronologically BEFORE scaling
    train_size = int(len(df_final) * 0.8)
    val_size = int(len(df_final) * 0.1)
    
    train_df = df_final.iloc[:train_size]
    val_df = df_final.iloc[train_size:train_size + val_size]
    # test_df is the remainder, can be used for final evaluation if needed
    
    print(f"--- Data Split (Chronological) ---")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # 4. **CRITICAL FIX**: Fit scaler ONLY on training data
    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    # Transform validation data with the scaler fitted on training data
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"--- Scaler fitted on training data and saved to {scaler_path} ---")

    # 5. Create Datasets and DataLoaders
    train_features = train_df[feature_cols].values
    train_labels = train_df['label'].values
    val_features = val_df[feature_cols].values
    val_labels = val_df['label'].values
    
    train_dataset = TimeSeriesDataset(train_features, train_labels, args.window_size)
    val_dataset = TimeSeriesDataset(val_features, val_labels, args.window_size)
    
    # 6. Handle Class Imbalance
    pos_count = np.sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight_value = neg_count / pos_count if pos_count > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_value], device=device)
    print(f"--- Class Imbalance Handling ---")
    print(f"Positive (1): {pos_count}, Negative (0): {neg_count}. BCE Loss Weight: {pos_weight_value:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 7. Initialize Model, Criterion, and Optimizer
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=1,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # Save model configuration for robust inference
    model_config = {
        "input_size": len(feature_cols), "hidden_size": args.hidden_size, 
        "num_layers": args.num_layers, "output_size": 1, "dropout_rate": args.dropout_rate
    }
    config_path = os.path.join(args.model_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(model_config, f)
    print(f"--- Model config saved to {config_path} ---")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 8. Training Loop with Early Stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n--- Starting Model Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        all_labels, all_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Validation loss improved. Saving best model to {best_model_path}")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Early stopping patience: {epochs_no_improve}/{args.patience}")
            
        if epochs_no_improve >= args.patience:
            print(f"--- Early stopping triggered after {args.patience} epochs with no improvement. ---")
            break
            
    print("--- Training complete. ---")

# ==============================================================================
# 4. SCRIPT ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Model Hyperparameters
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=150)
    
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    
    # Strategy Hyperparameters
    parser.add_argument('--lookahead-period', type=int, default=10)    
    parser.add_argument('--risk-reward-ratio', type=float, default=2.0)
    parser.add_argument('--profit-threshold-pct', type=float, default=0.5)

    args = parser.parse_args()
    main(args)