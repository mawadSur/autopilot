import os
import pandas as pd
from tqdm.auto import tqdm
import joblib
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Assuming utils.py has this function.
try:
    from utils import load_ohlc_chunks
except ImportError:
    print("Warning: 'utils.load_ohlc_chunks' not found. Real data loading will fail.")
    def load_ohlc_chunks(folder, chunk_mode=False):
        return []

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
    This function is now identical to the training script's version,
    ensuring all 17 features are calculated consistently.
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

    # Added missing hourly trend features for consistency with training
    try:
        hourly_index = df.index.floor('h')
        df_hourly = df['close'].groupby(hourly_index).mean()
        hourly_ema = df_hourly.ewm(span=20, adjust=False).mean()
        df['hourly_ema_20'] = hourly_ema.reindex(hourly_index, method='ffill').values
        df['price_vs_hourly_trend'] = (df['close'] - df['hourly_ema_20']) / (df['hourly_ema_20'] + 1e-9)
    except Exception:
        df['price_vs_hourly_trend'] = 0

    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_width'] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / (bb_mid + 1e-9)

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.dropna(inplace=True)
    return df

def run_backtest_with_capital(df, signals, initial_capital=10000):
    print("\n📈 Starting backtest with initial capital...")
    TRADING_FEE_PCT = 0.1
    TAKE_PROFIT_PCT = 2.0
    STOP_LOSS_PCT = 0.50
    TRADE_AMOUNT_PCT = 0.50

    capital = initial_capital
    in_position = False
    entry_price = 0.0
    position_size = 0.0
    trades, winning_trades, trade_entries = [], 0, []
    equity_curve = [initial_capital] * len(df)
    last_capital_update_index = 0

    for i in tqdm(range(len(df)), desc="Backtesting"):
        current_row = df.iloc[i]
        if i > last_capital_update_index:
            equity_curve[i] = equity_curve[i-1]

        if in_position:
            exit_price = None
            if current_row['low'] <= stop_loss_price:
                exit_price = stop_loss_price
            elif current_row['high'] >= take_profit_price:
                exit_price = take_profit_price
                winning_trades += 1

            if exit_price is not None:
                gross_pnl = position_size * ((exit_price / entry_price) - 1)
                entry_fee = position_size * (TRADING_FEE_PCT / 100)
                exit_fee = (position_size + gross_pnl) * (TRADING_FEE_PCT / 100)
                net_pnl = gross_pnl - entry_fee - exit_fee
                capital += net_pnl
                trades.append(net_pnl)
                in_position = False
                for j in range(last_capital_update_index, i + 1):
                    equity_curve[j] = capital
                last_capital_update_index = i

        if not in_position and signals[i] == 1:
            in_position = True
            entry_price = current_row['close']
            position_size = capital * TRADE_AMOUNT_PCT
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
            stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
            trade_entries.append(i)
            last_capital_update_index = i

    total_trades = len(trades)
    print("\n--- Backtest Summary ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    if total_trades > 0:
        win_rate = (winning_trades / total_trades) * 100
        total_pnl = sum(trades)
        print(f"Final Equity: ${capital:,.2f}")
        print(f"Total Net PnL: ${total_pnl:,.2f} ({(total_pnl/initial_capital)*100:.2f}%)")
        print(f"Total Trades Executed: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average PnL per trade: ${np.mean(trades):.2f}")
    else:
        print("No trades were executed.")
    print("------------------------")
    return equity_curve, trade_entries

def plot_backtest_results(df, equity_curve, trade_entries):
    print("🎨 Generating plot...")
    fig, ax1 = plt.subplots(figsize=(17, 9))
    plt.style.use('seaborn-v0_8-darkgrid')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price (USD)', color='deepskyblue', fontsize=12)
    ax1.plot(df.index, df['close'], color='deepskyblue', label='ETH Price', alpha=0.8, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    if trade_entries:
        entry_dates = df.index[trade_entries]
        entry_prices = df['close'].iloc[trade_entries]
        ax1.scatter(entry_dates, entry_prices, marker='^', color='lime', s=120, label='Trade Entry', zorder=5, edgecolors='black')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Equity (USD)', color='orange', fontsize=12)
    ax2.plot(df.index, equity_curve, color='orange', label='Equity Curve', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='orange')
    fig.tight_layout(pad=3.0)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Backtest Results: Price vs. Equity', fontsize=16, weight='bold')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.savefig("backtest_results.png", dpi=300, bbox_inches='tight')
    print("\n✅ Plot saved as backtest_results.png")

def main():
    MODEL_DIR = "./output"
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    model_config_path = os.path.join(MODEL_DIR, "model_config.json")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    
    # Ensure dummy config matches the required input size
    model_config = {"input_size": 17, "hidden_size": 50, "num_layers": 2, "output_size": 1, "dropout_rate": 0.2, "window_size": 60}
    if not os.path.exists(model_config_path):
        with open(model_config_path, 'w') as f: json.dump(model_config, f)
    if not os.path.exists(scaler_path):
        from sklearn.preprocessing import StandardScaler
        dummy_scaler = StandardScaler(); dummy_scaler.fit(np.random.rand(100, model_config['input_size'])); joblib.dump(dummy_scaler, scaler_path)
    if not os.path.exists(model_path):
         dummy_model = LSTMModel(**{k: v for k, v in model_config.items() if k != 'window_size'}); torch.save(dummy_model.state_dict(), model_path)

    print("Loading historical data...")
    try:
        chunks = list(load_ohlc_chunks(folder='eth_1m_data', chunk_mode=True))
        df_full = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    except Exception as e:
         print(f"An error occurred while loading data: {e}"); df_full = pd.DataFrame()
    if df_full.empty: print("No data loaded."); return

    df_full['date'] = pd.to_datetime(pd.to_numeric(df_full['date'], errors='coerce'), unit='ms')
    df_full.dropna(subset=['date'], inplace=True)
    df_full.set_index('date', inplace=True)
    if df_full.index.has_duplicates:
        print(f"Found {df_full.index.duplicated().sum()} duplicate timestamps. Aggregating...")
        df_full = df_full.groupby(df_full.index).agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
    df_full.sort_index(inplace=True)

    print("Resampling data to 1-minute frequency and forward-filling gaps...")
    full_range_index = pd.date_range(start=df_full.index.min(), end=df_full.index.max(), freq='min')
    df_resampled = df_full.reindex(full_range_index); df_resampled.ffill(inplace=True)
    df = df_resampled.iloc[-50000:].copy()
    if df.empty: print("No data for backtesting after processing."); return
    
    LONGEST_INDICATOR_WINDOW = 50
    if len(df) < LONGEST_INDICATOR_WINDOW:
        print(f"Error: DataFrame has {len(df)} rows. Need at least {LONGEST_INDICATOR_WINDOW}."); return

    print("1. Computing features for the entire dataset...")
    df_features = compute_technical_indicators(df.copy())
    if df_features.empty: print("Error: DataFrame empty after feature calculation."); return

    # Define the exact 17 features to match the training script
    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
        'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr',
        'price_vs_hourly_trend', 'bb_width'
    ]
    df_features = df_features.reindex(columns=df_features.columns.union(feature_cols), fill_value=0)[feature_cols]

    print("2. Loading scaler and model...")
    scaler = joblib.load(scaler_path)
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure loaded config and model definition match
    model_config['input_size'] = len(feature_cols) 
    model = LSTMModel(**{k: v for k, v in model_config.items() if k != 'window_size'}).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("3. Getting batch predictions...")
    window_size = model_config.get('window_size', 60)
    if len(df_features) < window_size:
        print(f"Error: Not enough data for prediction. Have {len(df_features)}, need {window_size}."); return

    scaled_features = scaler.transform(df_features[feature_cols].values)
    
    # --- REQUIRED CHANGE: Correct the typo from 'str_ides' to 'strides' ---
    windows = np.lib.stride_tricks.as_strided(
        scaled_features,
        shape=(len(scaled_features) - window_size + 1, window_size, scaled_features.shape[1]),
        strides=(scaled_features.strides[0], scaled_features.strides[0], scaled_features.strides[1])
    )
    input_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model(input_tensor)
    probabilities = torch.sigmoid(predictions).cpu().numpy().flatten()
    signals_raw = (probabilities > 0.6).astype(int)
    signals = np.zeros(len(df_features)); signals[window_size-1:] = signals_raw
    df_features['signal'] = signals

    equity_curve, trade_entries = run_backtest_with_capital(df_features, df_features['signal'].values, initial_capital=10000)
    plot_backtest_results(df_features, equity_curve, trade_entries)

if __name__ == "__main__":
    main()