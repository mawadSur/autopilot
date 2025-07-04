import os
import time
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("BINANCE_KEY")
api_secret = os.getenv("BINANCE_SECRET")
client = Client(api_key, api_secret)

# --- CORRECTED: Changed to 1-minute data for consistency and noise reduction ---
OUTPUT_DIR = "eth_1m_data"
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
LIMIT = 1000  # Max rows per Binance call

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_output_filename(dt_object):
    # --- CORRECTED: Changed filename format for clarity with monthly files ---
    return os.path.join(OUTPUT_DIR, f"eth_1m_{dt_object.strftime('%Y-%m')}.csv")

def fetch_historical_data(start_days_ago=730): # Default to ~2 years
    """
    Fetches historical 1-minute data and saves it, appending to existing files.
    """
    ensure_output_dir()
    now = datetime.utcnow()
    start_date = now - timedelta(days=start_days_ago)
    
    # Use the get_historical_klines generator for efficient fetching
    klines = client.get_historical_klines_generator(SYMBOL, INTERVAL, start_date.strftime("%d %b, %Y"))

    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    if df.empty:
        print("⚠️ No data fetched. Please check your start date and symbol.")
        return

    # --- Data Cleaning and Type Conversion ---
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.drop(columns=['open_time', 'close_time', 'ignore'], inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'num_trades', 
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    df.set_index('date', inplace=True)
    
    # --- Save data into monthly chunks ---
    df['year_month'] = df.index.strftime('%Y-%m')
    
    for month in df['year_month'].unique():
        month_df = df[df['year_month'] == month].drop(columns=['year_month'])
        out_path = os.path.join(OUTPUT_DIR, f"eth_1m_{month}.csv")
        
        # Append to existing file or create new one
        if os.path.exists(out_path):
            existing_df = pd.read_csv(out_path, index_col='date', parse_dates=True)
            combined_df = pd.concat([existing_df, month_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            combined_df.to_csv(out_path)
            print(f"🔄 Appended {len(month_df)} rows to {out_path}")
        else:
            month_df.to_csv(out_path)
            print(f"✅ Saved {len(month_df)} rows to {out_path}")

if __name__ == "__main__":
    # Fetch historical data, which is more efficient for backfilling
    fetch_historical_data(start_days_ago=730)