#!/usr/bin/env python3
"""
history_coindesk.py — CoinDesk Data API version of your Binance history puller.

Outputs the same shape as history.py:
  timestamp,open,high,low,close,volume,volume_quote,trade_count

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
from typing import List, Optional, Sequence, Tuple
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from coindesk_client import CoinDeskConfig, CoinDeskClient, CoinDeskEndpoints


L2_FEATURE_COLUMNS: List[str] = [
    "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
    "bid_depth_5", "ask_depth_5", "vwap_bid_5", "vwap_ask_5",
    "bid_depth_10", "ask_depth_10", "vwap_bid_10", "vwap_ask_10",
    "bid_depth_20", "ask_depth_20", "vwap_bid_20", "vwap_ask_20",
]

TRADE_AGG_COLUMNS: List[str] = [
    "trade_count",
    "buy_count",
    "sell_count",
    "taker_buy_volume_base",
    "taker_sell_volume_base",
    "taker_buy_volume_quote",
    "taker_sell_volume_quote",
]


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


def normalize_instrument(sym: str) -> str:
    """Normalize symbols into CoinDesk dash format (e.g., ETHUSDT -> ETH-USDT)."""
    raw = (sym or "").strip().replace(" ", "")
    if not raw:
        return ""

    if "-" in raw or "/" in raw:
        return raw.replace("/", "-").upper()

    upper = raw.upper()
    for quote in ("USDT", "USDC", "USD", "DAI"):
        if upper.endswith(quote) and len(upper) > len(quote):
            base = upper[: -len(quote)]
            return f"{base}-{quote}"
    return upper


# -------------------- File helpers --------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_write_csv(df: pd.DataFrame, path: str) -> None:
    """Atomic write to avoid partial files."""
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def month_file_has_data(path: str, start_dt: datetime, end_dt: datetime) -> bool:
    """Return True when an existing month CSV already has rows for [start_dt, end_dt)."""
    if not os.path.exists(path):
        return False
    try:
        existing = pd.read_csv(path, usecols=["timestamp"])
    except Exception:
        return False
    if existing.empty:
        return False
    ts = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
    if ts.isna().all():
        return False
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    return bool(((ts >= start_ts) & (ts < end_ts)).any())


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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "volume_quote", "trade_count"])

    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    # volume_base -> volume
    if "volume_base" in df.columns:
        df["volume"] = df["volume_base"].astype(float)
    else:
        df["volume"] = 0.0
    if "volume_quote" in df.columns:
        df["volume_quote"] = pd.to_numeric(df["volume_quote"], errors="coerce")
    else:
        df["volume_quote"] = pd.NA
    df["volume_quote"] = df["volume_quote"].fillna(df["close"].astype(float) * df["volume"].astype(float)).astype(float)

    if "trade_count" in df.columns:
        df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce").fillna(0).astype(int)
    else:
        df["trade_count"] = 0

    return df[["timestamp", "open", "high", "low", "close", "volume", "volume_quote", "trade_count"]]


def fetch_range_l2_snapshots(
    client: CoinDeskClient,
    *,
    market: str,
    instrument: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    sleep_s: float,
) -> pd.DataFrame:
    """Pull L2 snapshots for [start_dt, end_dt) by paging backwards with to_ts."""
    start_s = to_unix_s(start_dt)
    end_s = to_unix_s(end_dt)
    to_ts = end_s - 1

    chunks: List[pd.DataFrame] = []
    last_earliest_s: Optional[int] = None
    iterations = 0
    max_iterations = 5000

    while to_ts >= start_s and iterations < max_iterations:
        iterations += 1
        payload = client.get_l2_snapshots(
            market=market,
            instrument=instrument,
            to_ts=to_ts,
            limit=limit,
        )
        df = CoinDeskClient.normalize_l2_snapshots(payload)
        if df.empty:
            break

        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            break
        df["timestamp"] = df["timestamp"].astype("int64")

        earliest_s = int(df["timestamp"].min())
        if last_earliest_s is not None and earliest_s >= last_earliest_s:
            break
        last_earliest_s = earliest_s

        df = df[(df["timestamp"] >= start_s) & (df["timestamp"] < end_s)]
        if not df.empty:
            chunks.append(df)

        to_ts = earliest_s - 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "bids", "asks"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out[["timestamp", "bids", "asks"]]


def fetch_range_trades(
    client: CoinDeskClient,
    *,
    market: str,
    instrument: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    sleep_s: float,
) -> pd.DataFrame:
    """Pull trades for [start_dt, end_dt) by paging backwards with to_ts."""
    start_s = to_unix_s(start_dt)
    end_s = to_unix_s(end_dt)
    to_ts = end_s - 1

    chunks: List[pd.DataFrame] = []
    last_earliest_s: Optional[int] = None
    iterations = 0
    max_iterations = 5000

    while to_ts >= start_s and iterations < max_iterations:
        iterations += 1
        payload = client.get_trades(
            market=market,
            instrument=instrument,
            to_ts=to_ts,
            limit=limit,
        )
        df = CoinDeskClient.normalize_trades(payload)
        if df.empty:
            break

        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            break
        df["timestamp"] = df["timestamp"].astype("int64")

        earliest_s = int(df["timestamp"].min())
        if last_earliest_s is not None and earliest_s >= last_earliest_s:
            break
        last_earliest_s = earliest_s

        df = df[(df["timestamp"] >= start_s) & (df["timestamp"] < end_s)]
        if not df.empty:
            chunks.append(df)

        to_ts = earliest_s - 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "price", "qty_base", "qty_quote", "side"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates().sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out[["timestamp", "price", "qty_base", "qty_quote", "side"]]


def _to_float(v) -> Optional[float]:
    try:
        out = float(v)
        if np.isfinite(out):
            return out
        return None
    except Exception:
        return None


def _parse_book_level(level) -> Optional[Tuple[float, float]]:
    if isinstance(level, (list, tuple)) and len(level) >= 2:
        p = _to_float(level[0])
        s = _to_float(level[1])
    elif isinstance(level, dict):
        p = _to_float(
            level.get("price", level.get("p", level.get("px", level.get("bid_price", level.get("ask_price")))))
        )
        s = _to_float(
            level.get("size", level.get("s", level.get("qty", level.get("q", level.get("amount", level.get("volume"))))))
        )
    else:
        return None
    if p is None or s is None:
        return None
    return p, s


def _levels(levels, side: str) -> List[Tuple[float, float]]:
    parsed: List[Tuple[float, float]] = []
    if isinstance(levels, Sequence) and not isinstance(levels, (str, bytes, bytearray)):
        for lvl in levels:
            v = _parse_book_level(lvl)
            if v is not None:
                parsed.append(v)
    reverse = side == "bid"
    parsed.sort(key=lambda x: x[0], reverse=reverse)
    return parsed


def l2_snapshots_to_minute_features(df_snap: pd.DataFrame) -> pd.DataFrame:
    """Convert raw L2 snapshots into minute-level top-of-book/depth/vwap features."""
    if df_snap.empty:
        return pd.DataFrame(columns=["timestamp", *L2_FEATURE_COLUMNS])

    rows = []
    for _, row in df_snap.iterrows():
        bids = _levels(row.get("bids", []), "bid")
        asks = _levels(row.get("asks", []), "ask")

        rec = {
            "timestamp": row["timestamp"],
            "best_bid": bids[0][0] if bids else np.nan,
            "best_ask": asks[0][0] if asks else np.nan,
            "bid_size_l1": bids[0][1] if bids else 0.0,
            "ask_size_l1": asks[0][1] if asks else 0.0,
        }

        for n in (5, 10, 20):
            b = bids[:n]
            a = asks[:n]
            b_depth = float(sum(x[1] for x in b))
            a_depth = float(sum(x[1] for x in a))
            rec[f"bid_depth_{n}"] = b_depth
            rec[f"ask_depth_{n}"] = a_depth
            rec[f"vwap_bid_{n}"] = float(sum(p * s for p, s in b) / b_depth) if b_depth > 0 else np.nan
            rec[f"vwap_ask_{n}"] = float(sum(p * s for p, s in a) / a_depth) if a_depth > 0 else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", *L2_FEATURE_COLUMNS])
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["minute_ts"] = out["timestamp"].dt.floor("min")
    out = out.drop_duplicates(subset=["minute_ts"], keep="last")
    out = out.drop(columns=["timestamp"]).rename(columns={"minute_ts": "timestamp"}).reset_index(drop=True)
    return out[["timestamp", *L2_FEATURE_COLUMNS]]


def trades_to_minute_features(df_trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tick trades into 1-minute trade flow features."""
    if df_trades.empty:
        return pd.DataFrame(columns=["timestamp", *TRADE_AGG_COLUMNS])

    df = df_trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", *TRADE_AGG_COLUMNS])

    df["side"] = df["side"].astype(str).str.lower()
    df["qty_base"] = pd.to_numeric(df["qty_base"], errors="coerce").fillna(0.0)
    df["qty_quote"] = pd.to_numeric(df["qty_quote"], errors="coerce").fillna(0.0)
    df["minute_ts"] = df["timestamp"].dt.floor("min")
    df["is_buy"] = (df["side"] == "buy").astype(float)
    df["is_sell"] = (df["side"] == "sell").astype(float)
    df["qty_base_buy"] = df["qty_base"].where(df["side"] == "buy", 0.0)
    df["qty_base_sell"] = df["qty_base"].where(df["side"] == "sell", 0.0)
    df["qty_quote_buy"] = df["qty_quote"].where(df["side"] == "buy", 0.0)
    df["qty_quote_sell"] = df["qty_quote"].where(df["side"] == "sell", 0.0)

    out = (
        df.groupby("minute_ts", sort=True)
        .agg(
            trade_count=("side", "size"),
            buy_count=("is_buy", "sum"),
            sell_count=("is_sell", "sum"),
            taker_buy_volume_base=("qty_base_buy", "sum"),
            taker_sell_volume_base=("qty_base_sell", "sum"),
            taker_buy_volume_quote=("qty_quote_buy", "sum"),
            taker_sell_volume_quote=("qty_quote_sell", "sum"),
        )
        .reset_index()
        .rename(columns={"minute_ts": "timestamp"})
    )
    for col in TRADE_AGG_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    return out[["timestamp", *TRADE_AGG_COLUMNS]]


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
        )
        df = _normalize_minutes_payload(payload)
        if df.empty:
            break

        # Keep unix seconds internally for stable paging/progress.
        earliest_s = int(df["timestamp"].min())
        if last_earliest_s is not None and earliest_s >= last_earliest_s:
            # Not making progress; stop to avoid loop
            break
        last_earliest_s = earliest_s

        df = df[(df["timestamp"] >= start_s) & (df["timestamp"] < end_s)]
        if not df.empty:
            chunks.append(df)

        # Page backwards: next request ends just before earliest candle we got
        to_ts = earliest_s - 1

        if sleep_s > 0:
            time.sleep(sleep_s)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "volume_quote", "trade_count"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    out = out[["timestamp", "open", "high", "low", "close", "volume", "volume_quote", "trade_count"]]
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
    ap.add_argument("--with-l2", action="store_true", help="Fetch and merge L2 snapshot features.")
    ap.add_argument("--with-trades", action="store_true", help="Fetch and merge trade-flow features.")
    ap.add_argument("--trades-days", type=int, default=14, help="Only enrich trades for the last N days.")

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

    raw_instrument = args.instrument
    instrument = normalize_instrument(raw_instrument)
    print(
        f"[config] market={args.market} raw_instrument={raw_instrument} "
        f"normalized_instrument={instrument} range={start_dt:%Y-%m-%d} -> {end_dt:%Y-%m-%d} out={args.out_dir}"
    )
    trades_cutoff_dt = end_dt - timedelta(days=int(args.trades_days)) if args.with_trades else None

    cur = month_floor(start_dt)
    while cur < end_dt:
        m_end = min(next_month(cur), end_dt)
        w_start = max(cur, start_dt)
        w_end = m_end

        out_path = build_month_path(args.out_dir, args.market, instrument, cur)
        if month_file_has_data(out_path, w_start, w_end):
            if args.with_l2 or args.with_trades:
                try:
                    existing_cols = set(pd.read_csv(out_path, nrows=0).columns.tolist())
                except Exception:
                    existing_cols = set()
                missing_reasons = []
                if args.with_l2 and not all(col in existing_cols for col in L2_FEATURE_COLUMNS):
                    missing_reasons.append("L2")
                if args.with_trades and (trades_cutoff_dt is not None and w_end > trades_cutoff_dt):
                    needed_trade_cols = [c for c in TRADE_AGG_COLUMNS if c != "trade_count"]
                    if not all(col in existing_cols for col in needed_trade_cols):
                        missing_reasons.append("trades")
                if not missing_reasons:
                    print(f"[skip] Existing data found for {cur:%Y-%m}: {out_path}")
                    cur = next_month(cur)
                    continue
                print(
                    f"[skip] Existing file lacks {','.join(missing_reasons)} columns "
                    f"for {cur:%Y-%m}; refetching for enrichment."
                )
            else:
                print(f"[skip] Existing data found for {cur:%Y-%m}: {out_path}")
                cur = next_month(cur)
                continue

        print(f"[fetch] {args.market} {instrument} {cur:%Y-%m} ...")

        df = fetch_range_minutes(
            client,
            market=args.market,
            instrument=instrument,
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
            out = df[["timestamp", "open", "high", "low", "close", "volume", "volume_quote", "trade_count"]].copy()

            if args.with_l2:
                l2_snap = fetch_range_l2_snapshots(
                    client,
                    market=args.market,
                    instrument=instrument,
                    start_dt=w_start,
                    end_dt=w_end,
                    limit=int(args.limit),
                    sleep_s=float(args.sleep),
                )
                l2_min = l2_snapshots_to_minute_features(l2_snap)
                out = out.merge(l2_min, on="timestamp", how="left")

                price_like = ["best_bid", "best_ask", "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20"]
                size_like = [c for c in L2_FEATURE_COLUMNS if c not in price_like]
                present_price_like = [c for c in price_like if c in out.columns]
                present_size_like = [c for c in size_like if c in out.columns]
                if present_price_like:
                    out["_day"] = out["timestamp"].dt.floor("D")
                    out[present_price_like] = out.groupby("_day")[present_price_like].ffill()
                    out = out.drop(columns=["_day"])
                if present_size_like:
                    out[present_size_like] = out[present_size_like].fillna(0.0)

            if args.with_trades:
                out = out.drop(columns=["trade_count"], errors="ignore")
                if trades_cutoff_dt is not None and w_end > trades_cutoff_dt:
                    t_start = max(w_start, trades_cutoff_dt)
                    trades_df = fetch_range_trades(
                        client,
                        market=args.market,
                        instrument=instrument,
                        start_dt=t_start,
                        end_dt=w_end,
                        limit=int(args.limit),
                        sleep_s=float(args.sleep),
                    )
                    trades_min = trades_to_minute_features(trades_df)
                    out = out.merge(trades_min, on="timestamp", how="left")
                for col in TRADE_AGG_COLUMNS:
                    if col not in out.columns:
                        out[col] = 0.0
                out[TRADE_AGG_COLUMNS] = out[TRADE_AGG_COLUMNS].fillna(0.0).astype(float)

            # Ensure column order matches your output contract
            base_cols = ["timestamp", "open", "high", "low", "close", "volume", "volume_quote", "trade_count"]
            out_cols = list(base_cols)
            if args.with_l2:
                out_cols += [c for c in L2_FEATURE_COLUMNS if c in out.columns]
            if args.with_trades:
                out_cols += [c for c in TRADE_AGG_COLUMNS if c in out.columns and c not in out_cols]
            out = out[out_cols].copy()
            safe_write_csv(out, out_path)
            print(f"[save] {out_path} rows={len(out):,}")

        cur = next_month(cur)

    print("✅ Done.")


if __name__ == "__main__":
    main()
