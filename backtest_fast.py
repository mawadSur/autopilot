import os
import pandas as pd
from tqdm.auto import tqdm
import joblib
import torch
import torch.nn as nn
import numpy as np
import json
from utils import load_ohlc_chunks # Assuming utils.py has this function

# --- 1. Load Model and Feature Logic Locally ---

# The LSTMModel class must be defined here, identical to the one in your training script.
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def compute_technical_indicators(df):
    """
    This function is copied from your training script to ensure the logic is identical.
    It calculates all features for the entire DataFrame at once (vectorized).
    """
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / (df['sma_50'] + 1e-9) - 1
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['vol_change'] = df['volume'].pct_change()
    tr = pd.DataFrame(index=df.index); tr['h-l'] = df['high'] - df['low']; tr['h-pc'] = abs(df['high'] - df['close'].shift(1)); tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['atr'] = tr.max(axis=1).rolling(14).mean()

    hourly_index = df.index.floor('h')
    df_hourly = df['close'].groupby(hourly_index).mean()
    hourly_ema = df_hourly.ewm(span=20, adjust=False).mean()
    df['hourly_ema_20'] = hourly_ema.reindex(hourly_index, method='ffill').values
    df['price_vs_hourly_trend'] = (df['close'] - df['hourly_ema_20']) / (df['hourly_ema_20'] + 1e-9)
    
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_width'] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / (bb_mid + 1e-9)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    return df

def run_fast_backtest(df, signals):
    """
    Runs the backtest using the pre-computed signals. This loop is very fast.
    """
    print("\n📈 Starting fast backtest...")
    TRADING_FEE_PCT = 0.01
    TAKE_PROFIT_PCT = 0.1
    STOP_LOSS_PCT = 5000.0

    in_position = False
    entry_price = 0.0
    trades = []
    winning_trades = 0

    for i in tqdm(range(len(df)), desc="Backtesting"):
        current_row = df.iloc[i]
        current_price = current_row['close']
        
        if in_position:
            if current_row['low'] <= stop_loss_price:
                pnl = -STOP_LOSS_PCT - (2 * TRADING_FEE_PCT)
                trades.append(pnl)
                in_position = False 
            elif current_row['high'] >= take_profit_price:
                pnl = TAKE_PROFIT_PCT - (2 * TRADING_FEE_PCT)
                trades.append(pnl)
                winning_trades += 1
                in_position = False

        if not in_position and signals[i] == 1:
            in_position = True
            entry_price = current_price
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)

    # --- SUMMARY ---
    total_trades = len(trades)
    print("\n--- Backtest Summary ---")
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = sum(trades)
        print(f"Total Trades Executed: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Net PnL: {total_pnl:.2f}%")
    else:
        print("No trades were executed.")
    print("------------------------")

def main():
    """Main function to load data, run batch predictions, and start the backtest."""
    # Define paths to your model artifacts
    MODEL_DIR = ".\\output" # Assumes artifacts are in an 'output' folder
    print("Loading historical data...")
    df = pd.concat(load_ohlc_chunks(folder='eth_1m_data', chunk_mode=True), ignore_index=True)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    if df.empty:
        print("No data found.")
        return

    print("1. Computing features for the entire dataset...")
    df_features = compute_technical_indicators(df.copy())
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
        'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
    ]
    
    print("2. Loading scaler and model...")
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "model_config.json"), 'r') as f:
        model_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(**model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=device))
    model.eval()

    print("3. Getting batch predictions for all data points...")
    # Scale features
    scaled_features = scaler.transform(df_features[feature_cols])
    
    # Create sliding windows for model input
    window_size = model_config['input_size']
    windows = []
    for i in range(len(scaled_features) - window_size + 1):
        windows.append(scaled_features[i:i + window_size])
    
    input_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    # Get all predictions at once
    with torch.no_grad():
        predictions = model(input_tensor)
    
    probabilities = torch.sigmoid(predictions).cpu().numpy().flatten()
        # --- ADD THIS DIAGNOSTIC CODE ---
    print("\n--- Prediction Diagnostics ---")
    if len(probabilities) > 0:
        print(f"Max probability predicted: {np.max(probabilities):.4f}")
        print(f"Average probability predicted: {np.mean(probabilities):.4f}")
        print(f"Number of predictions > 0.6: {np.sum(probabilities > 0.6)}")
    else:
        print("No probabilities were generated.")
    print("----------------------------\n")
    # --------------------------------
    # Generate signals based on a threshold (e.g., 0.8)
    signals_raw = (probabilities > 0.6).astype(int)
    
    # Align signals with the main dataframe (signals correspond to the *end* of each window)
    signals = np.zeros(len(df_features))
    signals[window_size-1:] = signals_raw
    df_features['signal'] = signals

    # Run the fast backtest
    run_fast_backtest(df_features, df_features['signal'].values)

if __name__ == "__main__":
    main()