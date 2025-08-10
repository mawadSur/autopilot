import os
import time
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("BINANCE_KEY")
api_secret = os.getenv("BINANCE_SECRET")

# Set a longer timeout (e.g., 30 seconds) to handle slow API responses
client = Client(api_key, api_secret, {"timeout": 30})

OUTPUT_DIR = "eth_1m_data"
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def process_and_save_chunk(klines_chunk):
    """
    Processes a chunk of data and appends/saves into monthly CSV files.
    Uses atomic renames to avoid partial writes.
    """
    if not klines_chunk:
        return

    df = pd.DataFrame(klines_chunk, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df.rename(columns={'open_time': 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], unit='ms', utc=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('date', inplace=True)
    df.dropna(inplace=True)  # Drop rows with any parsing errors

    # --- Save data into monthly chunks ---
    df['year_month'] = df.index.strftime('%Y-%m')

    for month in df['year_month'].unique():
        month_df = df[df['year_month'] == month].drop(columns=['year_month'])
        out_path = os.path.join(OUTPUT_DIR, f"eth_1m_{month}.csv")

        if os.path.exists(out_path):
            existing_df = pd.read_csv(out_path, index_col='date', parse_dates=True)
            combined_df = pd.concat([existing_df, month_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            # atomic write
            combined_df.to_csv(out_path + '.tmp')
            os.replace(out_path + '.tmp', out_path)
        else:
            month_df.to_csv(out_path + '.tmp')
            os.replace(out_path + '.tmp', out_path)

def fetch_historical_data(start_days_ago=730):
    """
    Fetches historical 1-minute data month-by-month.
    Checks if a file for a specific month already exists and skips it to avoid re-downloading.
    """
    ensure_output_dir()
    now = datetime.utcnow()
    start_date = now - timedelta(days=start_days_ago)

    current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = now

    while current <= end:
        month_start = current
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        year_month_str = month_start.strftime('%Y-%m')

        tqdm.write(f"Fetching data for {year_month_str}...")

        start_str = month_start.strftime("%d %b, %Y")
        
        try:
            klines_generator = client.get_historical_klines_generator(SYMBOL, INTERVAL, start_str)
            chunk = list(klines_generator)
            if chunk:
                tqdm.write(f"Processing {len(chunk)} records for {year_month_str}...")
                process_and_save_chunk(chunk)
            
            time.sleep(1)  # polite rate limiting

        except BinanceAPIException as e:
            tqdm.write(f"Binance API Error for {year_month_str}: {e}")
            time.sleep(10)
        except Exception as e:
            tqdm.write(f"An unexpected error occurred for {year_month_str}: {e}")

        current = next_month

    print("\n✅ Historical data fetch complete.")

if __name__ == "__main__":
    try:
        fetch_historical_data(start_days_ago=1000)
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")