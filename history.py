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
CHUNK_SIZE = 50000 

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
    Fetches historical 1-minute data month-by-month.
    Checks if a file for a specific month already exists and skips it to avoid re-downloading.
    """
    ensure_output_dir()
    now = datetime.utcnow()
    start_date = now - timedelta(days=start_days_ago)
    
    # Create a date range for the first day of each month to be processed
    months_to_process = pd.date_range(start_date, now, freq='MS').to_pydatetime().tolist()

    print(f"Starting data check for {SYMBOL} from {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")

    for month_start in tqdm(months_to_process, desc="Processing months"):
        year_month_str = month_start.strftime('%Y-%m')
        file_path = os.path.join(OUTPUT_DIR, f"eth_1m_{year_month_str}.csv")

        # --- MODIFICATION ---
        # Check if the file for the month already exists. If so, skip it.
        if os.path.exists(file_path):
            tqdm.write(f"Skipping {year_month_str}: File already exists.")
            continue

        tqdm.write(f"Fetching data for {year_month_str}...")

        # Define the start and end strings for the API call
        start_str = month_start.strftime("%d %b, %Y")
        
        try:
            # Use the generator to fetch all klines for the current month
            klines_generator = client.get_historical_klines_generator(SYMBOL, INTERVAL, start_str)
            
            # Process the downloaded data in chunks (usually one chunk per month)
            chunk = list(klines_generator)
            if chunk:
                tqdm.write(f"Processing {len(chunk)} records for {year_month_str}...")
                process_and_save_chunk(chunk)
            
            time.sleep(1) # Be polite to the API

        except BinanceAPIException as e:
            tqdm.write(f"Binance API Error for {year_month_str}: {e}")
            time.sleep(10) # Wait longer after an API error
        except Exception as e:
            tqdm.write(f"An unexpected error occurred for {year_month_str}: {e}")

    print("\n✅ Historical data fetch complete.")

if __name__ == "__main__":
    try:
        # Fetches data for the last year, skipping months that are already saved.
        fetch_historical_data(start_days_ago=1000)
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")