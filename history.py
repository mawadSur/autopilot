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

OUTPUT_DIR = "eth_1s_data"
SYMBOL = "ETHUSDT"
INTERVAL = Client.KLINE_INTERVAL_1SECOND
CHUNK_SECONDS = 60 * 60 * 24  # 1 day = 86400 seconds
LIMIT = 1000  # Max rows per Binance call

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_output_filename(start_ts):
    dt = datetime.utcfromtimestamp(start_ts / 1000).strftime('%Y-%m-%d')
    return os.path.join(OUTPUT_DIR, f"eth_1s_{dt}.csv")

def file_already_exists(start_ts):
    return os.path.exists(get_output_filename(start_ts))

def fetch_chunk(start_ms, end_ms):
    """Fetch one full day's worth of 1-second data."""
    all_rows = []
    current_ms = start_ms

    while current_ms < end_ms:
        try:
            klines = client.get_klines(
                symbol=SYMBOL,
                interval=INTERVAL,
                startTime=current_ms,
                limit=LIMIT
            )
            if not klines:
                break

            for k in klines:
                all_rows.append({
                    "timestamp": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })

            last_ts = klines[-1][0]
            current_ms = last_ts + 1000
            time.sleep(0.25)

        except Exception as e:
            print(f"❌ Error during fetch: {e}")
            time.sleep(2)

    return all_rows

def run_backfill(start_days_ago=365):
    ensure_output_dir()
    now = datetime.utcnow().replace(microsecond=0, second=0, minute=0)
    start_date = now - timedelta(days=start_days_ago)

    current = start_date

    while current < now:
        start_ts = int(current.timestamp() * 1000)
        end_ts = int((current + timedelta(days=1)).timestamp() * 1000)
        out_path = get_output_filename(start_ts)

        if file_already_exists(start_ts):
            print(f"✅ Skipping existing: {out_path}")
        else:
            print(f"📥 Fetching: {current.date()}...")
            data = fetch_chunk(start_ts, end_ts)
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                df.to_csv(out_path)
                print(f"✅ Saved {len(df)} rows → {out_path}")
            else:
                print(f"⚠️ No data for {current.date()}")

        current += timedelta(days=1)

if __name__ == "__main__":
    run_backfill(start_days_ago=365)
