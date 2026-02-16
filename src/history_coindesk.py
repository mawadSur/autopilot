#!/usr/bin/env python3
"""
history_coindesk.py — CoinDesk Data API version of your Binance history puller.

Outputs the same shape as history.py:
  timestamp,open,high,low,close,volume

Defaults:
  • market     : coinbase
  • instrument : ETH-USDT
  • range      : last 3 years (UTC)
  • out-dir    : eth_1m_data

Notes:
- This endpoint is capped at 2000 minute points per call; we page backwards using `to_ts`. :contentReference[oaicite:1]{index=1}
- If you want to confirm the exact market/instrument mappings CoinDesk recognizes, see:
  /spot/v1/markets/instruments (Markets + Instruments Mapped). :contentReference[oaicite:2]{index=2}


Uses reusable coindesk_client.py to fetch minute OHLCV and normalize responses.
"""

import os
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from dotenv import load_dotenv

import pandas as pd

from coindesk_client import CoinDeskConfig, CoinDeskClient, CoinDeskEndpoints


# -------:contentReference[oaicite:4]{index=4}pers --------------------

def parse_date(s: str) -> datetime:
    """Parse YYYY-MM-DD as UTC midnight."""
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def month_floor(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(dt: datetime) -> datetime:
    year = dt.year + (dt.month // 12)
    month = (dt.month % 12) + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def to_unix_s(dt: datetime) -> int:
    return int(dt.timestamp())


# -------------------- File helpers --------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: str) -> None:
    """Atomic write to avoid partial files."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def build_month_path(out_dir: str, market: str, instrument: str, month_dt: datetime) -> str:
    y_m = month_dt.strftime("%Y-%m")
    m = market.lower().replace(" ", "_")
    inst = instrument.lower().replace("/", "-").replace("_", "-")
    # Similar spirit to your Binance file naming (one file per month)
    fname = f"{inst}_{m}_1m_{y_m}.csv"
    return os.path.join(out_dir, fname)


def _normalize_minutes_payload(payload: dict) -> pd.DataFrame:
    """Normalize OHLCV minutes from coindesk_client into history.py-compatible format."""
    df = CoinDeskClient.normalize_ohlcv_minutes(payload)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    # convert unix seconds to UTC datetime for compatibility with history.py
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    # volume_base -> volume
    if "volume_base" in df.columns:
        df["volume"] = df["volume_base"].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# -------------------- Paging logic --------------------

def fetch_range_minutes(
    client: CoinDeskClient,
    *,
    market: str,
    instrument: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    aggregate: int,
    fill: bool,
    groups: Optional[str],
    sleep_s: float,
) -> pd.DataFrame:
    """
    Pull minute candles for [start_dt, end_dt) by paging backwards with to_ts.
    """
    start_s = to_unix_s(start_dt)
    end_s = to_unix_s(end_dt)

    # Work backwards: anchor at the end boundary - 1 second
    to_ts = end_s - 1

    chunks: List[pd.DataFrame] = []
    last_earliest_s: Optional[int] = None

    while to_ts >= start_s:
        payload = client.get_ohlcv_minutes(
            market=market,
            instrument=instrument,
            to_ts=to_ts,
            limit=limit,
            aggregate=aggregate,
            fill=fill,
            groups=groups,
            response_format="JSON",
        )
        df = _normalize_minutes_payload(payload)
        if df.empty:
            break

        # Filter to the requested window
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        ts_i64 = ts.astype("int64")

        # If it's ms-resolution (13 digits), convert ms->seconds; otherwise ns->seconds
        # ms since epoch ~ 1e12–1e13, ns since epoch ~ 1e18–1e19
        if ts_i64.max() < 10_000_000_000_000:   # < 1e13-ish => milliseconds
            df_s = (ts_i64 // 1_000).astype("int64")
        else:                                   # nanoseconds
            df_s = (ts_i64 // 1_000_000_000).astype("int64")

        df = df[(df_s >= start_s) & (df_s < end_s)]
        if df.empty:
            break

        earliest_s = int(df["timestamp"].min().timestamp())
        if last_earliest_s is not None and earliest_s >= last_earliest_s:
            # Not making progress; stop to avoid loop
            break
        last_earliest_s = earliest_s

        chunks.append(df)

        # Page backwards: next request ends just before earliest candle we got
        to_ts = earliest_s - 1

        if sleep_s > 0:
            time.sleep(sleep_s)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


# -------------------- Main / CLI --------------------

def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Download CoinDesk spot minute candles into monthly CSVs (Binance history.py-compatible output).")

    # Defaults you requested:
    ap.add_argument("--market", type=str, default="coinbase", help="Default: coinbase")
    ap.add_argument("--instrument", type=str, default="ETH-USDT", help="Default: ETH-USDT")
    ap.add_argument("--out-dir", type=str, default="eth_1m_data", help="Default: eth_1m_data")

    # Range controls
    ap.add_argument("--days", type=int, default=None, help="Backfill N days up to now (UTC).")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD UTC (inclusive start).")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD UTC (exclusive end; default now).")

    # API paging controls
    ap.add_argument("--limit", type=int, default=2000, help="Rows per request (endpoint max ~2000).")
    ap.add_argument("--aggregate", type=int, default=1, help="Aggregate N minutes per candle. :contentReference[oaicite:6]{index=6}")
    ap.add_argument("--no-fill", action="store_true", help="Disable fill=true (gap filling).")
    ap.add_argument("--groups", type=str, default=None, help="Optional groups parameter.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between API calls (seconds).")

    # Overrides
    ap.add_argument("--base-url", type=str, default=os.getenv("COINDESK_BASE_URL", "https://data-api.coindesk.com"), help="Override base URL.")
    ap.add_argument("--endpoint", type=str, default=os.getenv("COINDESK_OHLCV_ENDPOINT", "/spot/v1/historical/minutes"), help="Override endpoint path.")
    args = ap.parse_args()

    api_key = os.getenv("COINDESK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing COINDESK_API_KEY in environment.")

    utc_now = datetime.now(timezone.utc)

    # Your requested default: last 3 years if user doesn't specify range
    if args.days is not None:
        end_dt = utc_now
        start_dt = end_dt - timedelta(days=int(args.days))
    elif args.start is not None:
        start_dt = parse_date(args.start)
        end_dt = parse_date(args.end) if args.end else utc_now
    else:
        end_dt = utc_now
        start_dt = end_dt - timedelta(days=365 * 3)

    if end_dt <= start_dt:
        raise SystemExit("End must be after start.")

    ensure_dir(args.out_dir)

    endpoints = CoinDeskEndpoints(ohlcv_minutes=args.endpoint)
    cfg = CoinDeskConfig(api_key=api_key, base_url=args.base_url, endpoints=endpoints)
    client = CoinDeskClient(cfg)

    print(f"[config] market={args.market} instrument={args.instrument} range={start_dt:%Y-%m-%d} -> {end_dt:%Y-%m-%d} out={args.out_dir}")

    cur = month_floor(start_dt)
    while cur < end_dt:
        m_end = min(next_month(cur), end_dt)
        w_start = max(cur, start_dt)
        w_end = m_end

        out_path = build_month_path(args.out_dir, args.market, args.instrument, cur)
        print(f"[fetch] {args.market} {args.instrument} {cur:%Y-%m} ...")

        df = fetch_range_minutes(
            client,
            market=args.market,
            instrument=args.instrument,
            start_dt=w_start,
            end_dt=w_end,
            limit=int(args.limit),
            aggregate=int(args.aggregate),
            fill=(not args.no_fill),
            groups=args.groups,
            sleep_s=float(args.sleep),
        )

        if df.empty:
            print(f"[warn] No data returned for {cur:%Y-%m}")
        else:
            # Ensure column order matches your Binance script exactly
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            safe_write_csv(df, out_path)
            print(f"[save] {out_path} rows={len(df):,}")

        cur = next_month(cur)

    print("✅ Done.")


if __name__ == "__main__":
    main()
