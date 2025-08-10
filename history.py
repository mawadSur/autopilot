import os
import time
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables and initialize Binance client
load_dotenv()
api_key = os.getenv("BINANCE_KEY")
api_secret = os.getenv("BINANCE_SECRET")
client = Client(api_key, api_secret, {"timeout": 30})

# --- Configuration ---
OUTPUT_DIR = "eth_1m_data"
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE

def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def process_and_save_chunk(chunk):
    """
    Processes a list of raw kline data and saves it to the correct monthly CSV file
    in the format expected by the training and inference scripts.
    """
    if not chunk:
        return

    # Select only the required columns: open_time, open, high, low, close, volume
    required_data = [[k[0], k[1], k[2], k[3], k[4], k[5]] for k in chunk]
    
    df = pd.DataFrame(required_data, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    # --- Data Type Conversion ---
    df['date'] = pd.to_numeric(df['date'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True) # Drop rows with any parsing errors

    # --- Save data into monthly chunks ---
    # Temporarily convert timestamp to datetime to determine the file name
    df['year_month'] = pd.to_datetime(df['date'], unit='ms').dt.strftime('%Y-%m')

    for month_str in df['year_month'].unique():
        month_df = df[df['year_month'] == month_str].drop(columns=['year_month'])
        out_path = os.path.join(OUTPUT_DIR, f"eth_1m_{month_str}.csv")

        # Combine with existing data if the file already exists
        if os.path.exists(out_path):
            # Read existing data, ensuring no header is assumed
            existing_df = pd.read_csv(out_path, header=None, names=month_df.columns)
            combined_df = pd.concat([existing_df, month_df])
            # Drop duplicates based on the timestamp and keep the latest entry
            combined_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            combined_df.sort_values(by='date', inplace=True)
            # Save without index or header to match downstream requirements
            combined_df.to_csv(out_path, index=False, header=False)
        else:
            month_df.sort_values(by='date', inplace=True)
            month_df.to_csv(out_path, index=False, header=False)

def fetch_historical_data(start_days_ago=730):
    """
    Fetches historical 1-minute data, saving it month-by-month.
    Skips months where the data file already exists to avoid re-downloading.
    """
    ensure_output_dir()
    now = datetime.utcnow()
    start_date = now - timedelta(days=start_days_ago)
    
    # Create a date range for the first day of each month to process
    months_to_process = pd.date_range(start_date, now, freq='MS').to_pydatetime().tolist()

    print(f"Starting data check for {SYMBOL} from {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")

    for month_start in tqdm(months_to_process, desc="Processing months"):
        year_month_str = month_start.strftime('%Y-%m')
        file_path = os.path.join(OUTPUT_DIR, f"eth_1m_{year_month_str}.csv")

        if os.path.exists(file_path):
            tqdm.write(f"Skipping {year_month_str}: File already exists.")
            continue

        tqdm.write(f"Fetching data for {year_month_str}...")
        start_str = month_start.strftime("%d %b, %Y")
        
        try:
            # Fetch all klines for the current time range
            klines_generator = client.get_historical_klines_generator(SYMBOL, INTERVAL, start_str)
            chunk = list(klines_generator) # Convert generator to list

            if chunk:
                # Filter chunk to only include data for the current month to avoid overlap
                current_month_chunk = [k for k in chunk if pd.to_datetime(k[0], unit='ms').strftime('%Y-%m') == year_month_str]
                tqdm.write(f"Processing {len(current_month_chunk)} records for {year_month_str}...")
                process_and_save_chunk(current_month_chunk)
            
            time.sleep(1) # Be polite to the API

        except BinanceAPIException as e:
            tqdm.write(f"Binance API Error for {year_month_str}: {e}")
            time.sleep(10) # Wait longer after an API error
        except Exception as e:
            tqdm.write(f"An unexpected error occurred for {year_month_str}: {e}")

    print("\n✅ Historical data fetch complete.")

if __name__ == "__main__":
    try:
        # Fetches data for the last 1000 days
        fetch_historical_data(start_days_ago=1000)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")