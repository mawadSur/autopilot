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
CHUNK_SIZE = 50000 # Process data in chunks of this size

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def process_and_save_chunk(chunk):
    """
    Processes a list of kline data and saves it to the correct monthly CSV file.
    """
    if not chunk:
        return

    df = pd.DataFrame(chunk, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # --- Data Cleaning and Type Conversion ---
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')
    df.drop(columns=['open_time', 'close_time', 'ignore'], inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'num_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.set_index('date', inplace=True)
    df.dropna(inplace=True) # Drop rows with any parsing errors

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
            combined_df.to_csv(out_path)
        else:
            month_df.to_csv(out_path)

def fetch_historical_data(start_days_ago=730):
    """
    Fetches historical 1-minute data in chunks to be memory-efficient and respectful of API limits,
    and saves it, appending to existing files.
    """
    ensure_output_dir()
    now = datetime.utcnow()
    start_date = now - timedelta(days=start_days_ago)

    print(f"Fetching historical data for {SYMBOL} from {start_date.strftime('%Y-%m-%d')}...")
    
    klines_generator = client.get_historical_klines_generator(SYMBOL, INTERVAL, start_date.strftime("%d %b, %Y"))

    chunk = []
    for kline in tqdm(klines_generator, desc="Downloading historical data"):
        chunk.append(kline)
        if len(chunk) >= CHUNK_SIZE:
            print(f"\nProcessing chunk of {len(chunk)} records...")
            process_and_save_chunk(chunk)
            chunk = []  # Reset the chunk
            time.sleep(0.5) # Be polite to the API

    # Process the final remaining chunk
    if chunk:
        print(f"\nProcessing final chunk of {len(chunk)} records...")
        process_and_save_chunk(chunk)

    print("\n✅ Historical data fetch complete.")

if __name__ == "__main__":
    try:
        fetch_historical_data(start_days_ago=360)
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
