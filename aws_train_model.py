import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json

from utils import load_ohlc_chunks, compute_rsi

# --- Model and Preprocessing ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x); out = self.dropout(out[:, -1, :]); out = self.fc(out)
        return out

def compute_atr(df, period=14):
    df['h-l'] = df['high'] - df['low']; df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1)); df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df['tr'].rolling(period).mean()

def preprocess_data(df, feature_cols, window_size, lookahead_period):
    df = df.copy()
    timestamp_column = 'date'
    
    # Coerce errors: Turn any un-parseable dates into 'NaT' (Not a Time).
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
    
    # Drop rows with 'NaT' in the date column.
    original_rows = len(df)
    df.dropna(subset=[timestamp_column], inplace=True)
    if len(df) < original_rows:
        print(f"Dropped {original_rows - len(df)} rows with invalid date formats.")
    
    # --- FIX: Check if data remains after cleaning ---
    if df.empty:
        raise ValueError("All rows were dropped due to invalid date formats. Halting training. Please check your source CSV files in 'eth_1m_data'.")

    df.set_index(timestamp_column, inplace=True)
    df['body'] = df['close'] - df['open']; df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean(); df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1; df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14); df['vol_change'] = df['volume'].pct_change()
    df['atr'] = compute_atr(df); df_hourly = df['close'].resample('1h').mean()
    hourly_ema = df_hourly.ewm(span=20).mean()
    df['hourly_ema_20'] = hourly_ema.reindex(df.index, method='ffill')
    df['price_vs_hourly_trend'] = (df['close'] - df['hourly_ema_20']) / df['hourly_ema_20']
    df['bb_std'] = df['close'].rolling(20).std(); df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_width'] = ((df['bb_mid'] + 2 * df['bb_std']) - (df['bb_mid'] - 2 * df['bb_std'])) / df['bb_mid']
    atr_multiplier = 3.0; dynamic_threshold = df['atr'] * atr_multiplier / df['close']
    future_highs = df['high'].rolling(window=lookahead_period).max().shift(-lookahead_period)
    df['target'] = (future_highs - df['close']) / df['close']
    df['label'] = (df['target'] > dynamic_threshold).astype(int)
    df.replace([np.inf, -np.inf], 0, inplace=True); df.dropna(inplace=True)

    # --- FIX: Check if data remains after feature engineering ---
    if df.empty:
        raise ValueError("All rows were dropped after feature engineering (due to NaNs in rolling windows). Halting training. You may need more initial data.")
        
    scaler = StandardScaler(); df[feature_cols] = scaler.fit_transform(df[feature_cols])
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df.iloc[i:i+window_size][feature_cols].values)
        y.append(df.iloc[i+window_size-1]['label'])

    # --- FIX: Check if any sequences were created ---
    if not X:
        raise ValueError(f"Not enough data to create sequences of window size {window_size}. Data length after cleaning is {len(df)}. Halting training.")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

def debug_date_parsing(df):
    """
    A helper function to find rows with un-parseable dates.
    """
    print("--- Running Date Parsing Debug ---")
    bad_rows = []
    for index, value in df['date'].items():
        try:
            pd.to_datetime(value)
        except (ValueError, TypeError):
            bad_rows.append((index, value))
    
    if bad_rows:
        print(f"🚫 Found {len(bad_rows)} rows with bad date formats.")
        # Print the first 5 bad rows for inspection
        for i, (idx, val) in enumerate(bad_rows[:5]):
            print(f"  - Bad Value: '{val}' in row index: {idx}")
    else:
        print("✅ All dates appear to be in a valid format.")
    print("--- Finished Date Parsing Debug ---")

# --- Main Training Function ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    full_df = pd.concat(load_ohlc_chunks(folder=args.train, chunk_mode=True), ignore_index=True)
    
    debug_date_parsing(full_df) 

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
        'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
    ]
    X, y, scaler = preprocess_data(full_df, feature_cols, args.window_size, args.lookahead_period)
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Create train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_data = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = LSTMModel(len(feature_cols), 128, 3, 1, 0.5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(args.model_dir, "best_model.pth")

    print("Starting training with early stopping...")
    for epoch in range(args.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                predicted = (torch.sigmoid(outputs) > 0.5)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (val_correct / val_total) * 100
        print(f"Epoch [{epoch+1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saving best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    print("\nTraining finished. Evaluating best model on the test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            outputs = model(batch_X)
            predicted = (torch.sigmoid(outputs) > 0.5)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
    
    test_accuracy = (test_correct / test_total) * 100
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--window-size', type=int, default=150)
    parser.add_argument('--lookahead-period', type=int, default=10)
    args = parser.parse_args()
    main(args)


def model_fn(model_dir):
    """
    Loads the saved model and scaler from the model directory.
    This function now returns a dictionary containing both artifacts.
    """
    print("--- Loading model and scaler ---")
    
    # --- CRITICAL DEBUG STEP ---
    # This will print the files available in the model directory to the CloudWatch logs.
    # Check the logs to ensure 'best_model.pth' and 'scaler.pkl' are listed.
    print(f"Files in model directory: {os.listdir(model_dir)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the Model
    # Ensure these parameters EXACTLY match your training script
    model = LSTMModel(input_size=17, hidden_size=128, num_layers=3, output_size=1, dropout_rate=0.5)
    model_path = os.path.join(model_dir, "best_model.pth")
    with open(model_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.to(device).eval()
    
    # 2. Load the Scaler
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    print("--- Model and scaler loaded successfully ---")
    return {'model': model, 'scaler': scaler}


def input_fn(request_body, request_content_type):
    """
    Deserializes the input data from the request.
    This function now passes the raw array to predict_fn for scaling.
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        sequence = np.array(data['inputs'], dtype=np.float32)
        return sequence # Return the raw sequence, scaling will happen in predict_fn
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifacts):
    """
    Makes a prediction using the loaded model and scaler.
    """
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_data)
    
    # Convert to tensor and add batch dimension
    input_tensor = torch.from_numpy(scaled_input).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
    
    signal = 1 if probability > 0.5 else 0
    
    return {"probability": probability, "signal": signal}


def output_fn(prediction_output, accept):
    """
    Serializes the prediction output to JSON.
    (No changes needed here)
    """
    if accept == "application/json":
        return json.dumps(prediction_output), accept
    raise ValueError(f"Unsupported accept type: {accept}")