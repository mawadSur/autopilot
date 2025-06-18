import requests
import pandas as pd
import time
from datetime import datetime, timedelta

def fetch_hourly_eth_chunk(start_timestamp, end_timestamp, vs_currency='usd'):
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range"
    params = {
        'vs_currency': vs_currency,
        'from': int(start_timestamp),
        'to': int(end_timestamp)
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'prices' not in data or 'total_volumes' not in data:
        raise ValueError("❌ CoinGecko response missing 'prices'. Likely invalid date range.")

    df_price = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df_volume = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])

    df = pd.merge(df_price, df_volume, on='timestamp')
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)

    df['open'] = df['price']
    df['high'] = df['price']
    df['low'] = df['price']
    df['close'] = df['price']
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def fetch_eth_hourly_365d():
    print("📥 Fetching ETH hourly price & volume (past 365 days)...")
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    end = now
    chunk_days = 90
    all_chunks = []

    # Work backwards in exact chunks
    for _ in range(0, 365, chunk_days):
        start = end - timedelta(days=chunk_days)
        print(f"⏳ Fetching {start.date()} to {end.date()}...")

        try:
            df = fetch_hourly_eth_chunk(start.timestamp(), end.timestamp())
            all_chunks.append(df)
            time.sleep(1.3)
        except Exception as e:
            print(f"⚠️ Failed to fetch chunk {start.date()} to {end.date()}: {e}")

        end = start  # Move window back

    df_all = pd.concat(all_chunks)
    df_all = df_all[~df_all.index.duplicated()]
    df_all = df_all.sort_index()

    # Validate gaps
    expected_range = pd.date_range(
        start=df_all.index.min(), end=df_all.index.max(), freq='h'
    )
    missing = expected_range.difference(df_all.index)

    print(f"📊 Expected rows: {len(expected_range)}")
    print(f"✅ Saved rows:    {len(df_all)}")
    print(f"❌ Missing hours: {len(missing)}")

    if not missing.empty:
        print("⚠️ Missing timestamps:")
        for ts in missing[:10]:
            print("  -", ts)

    return df_all

if __name__ == "__main__":
    df = fetch_eth_hourly_365d()
    df.to_csv("eth_ohlc.csv")
    print(f"✅ Saved {len(df)} hourly rows to eth_ohlc.csv")
