#!/usr/bin/env python3
"""
build_coindesk_dataset.py — unified CoinDesk minute-level dataset builder.

Outputs one merged CSV per month at 1-minute frequency with required columns.

Example:
  COINDESK_API_KEY=... python build_coindesk_dataset.py --market coinbase --instrument ETH-USDT
"""

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from coindesk_client import build_client_from_env


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def month_floor(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def next_month(dt: datetime) -> datetime:
    year = dt.year + (dt.month // 12)
    month = (dt.month % 12) + 1
    return dt.replace(year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0)


def to_unix_s(dt: datetime) -> int:
    return int(dt.timestamp())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def minute_index(start_s: int, end_s: int) -> pd.DataFrame:
    ts = np.arange(start_s, end_s, 60, dtype=np.int64)
    return pd.DataFrame({"timestamp": ts})


def _parse_levels(levels) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if not levels:
        return out
    for lvl in levels:
        if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
            try:
                out.append((float(lvl[0]), float(lvl[1])))
            except Exception:
                continue
        elif isinstance(lvl, dict):
            price = lvl.get("price") or lvl.get("px") or lvl.get("p")
            size = lvl.get("size") or lvl.get("qty") or lvl.get("q")
            if price is None or size is None:
                continue
            try:
                out.append((float(price), float(size)))
            except Exception:
                continue
    return out


def _depth_and_vwap(levels: List[Tuple[float, float]], top_n: int) -> Tuple[float, float]:
    if not levels:
        return 0.0, math.nan
    sl = levels[:top_n]
    total_size = sum(sz for _, sz in sl)
    if total_size <= 0:
        return 0.0, math.nan
    vwap = sum(px * sz for px, sz in sl) / total_size
    return total_size, vwap


def aggregate_l2_snapshots(df_snap: pd.DataFrame) -> pd.DataFrame:
    if df_snap.empty:
        return pd.DataFrame()
    # Bucket to minute
    df = df_snap.copy()
    df["minute_ts"] = (df["timestamp"] // 60) * 60
    # Use last snapshot per minute
    df = df.sort_values("timestamp").groupby("minute_ts").tail(1)

    rows = []
    for _, row in df.iterrows():
        bids = _parse_levels(row.get("bids", []))
        asks = _parse_levels(row.get("asks", []))
        best_bid = bids[0][0] if bids else math.nan
        best_ask = asks[0][0] if asks else math.nan
        bid_size_l1 = bids[0][1] if bids else math.nan
        ask_size_l1 = asks[0][1] if asks else math.nan

        bid_depth_5, vwap_bid_5 = _depth_and_vwap(bids, 5)
        ask_depth_5, vwap_ask_5 = _depth_and_vwap(asks, 5)
        bid_depth_10, vwap_bid_10 = _depth_and_vwap(bids, 10)
        ask_depth_10, vwap_ask_10 = _depth_and_vwap(asks, 10)
        bid_depth_20, vwap_bid_20 = _depth_and_vwap(bids, 20)
        ask_depth_20, vwap_ask_20 = _depth_and_vwap(asks, 20)

        def imbalance(bd, ad):
            denom = bd + ad
            return (bd - ad) / denom if denom > 0 else math.nan

        rows.append({
            "timestamp": int(row["minute_ts"]),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size_l1": bid_size_l1,
            "ask_size_l1": ask_size_l1,
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "bid_depth_10": bid_depth_10,
            "ask_depth_10": ask_depth_10,
            "bid_depth_20": bid_depth_20,
            "ask_depth_20": ask_depth_20,
            "l2_imbalance_5": imbalance(bid_depth_5, ask_depth_5),
            "l2_imbalance_10": imbalance(bid_depth_10, ask_depth_10),
            "l2_imbalance_20": imbalance(bid_depth_20, ask_depth_20),
            "vwap_bid_5": vwap_bid_5,
            "vwap_ask_5": vwap_ask_5,
            "vwap_bid_10": vwap_bid_10,
            "vwap_ask_10": vwap_ask_10,
            "vwap_bid_20": vwap_bid_20,
            "vwap_ask_20": vwap_ask_20,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def aggregate_trades(df_trades: pd.DataFrame) -> pd.DataFrame:
    if df_trades.empty:
        return pd.DataFrame()
    df = df_trades.copy()
    df["minute_ts"] = (df["timestamp"] // 60) * 60

    def _agg(group: pd.DataFrame) -> Dict[str, float]:
        total_qty = group["qty_base"].sum()
        total_quote = group["qty_quote"].sum()
        trade_count = len(group)

        buys = group[group["side"] == "buy"]
        sells = group[group["side"] == "sell"]

        buy_qty = buys["qty_base"].sum()
        sell_qty = sells["qty_base"].sum()
        buy_quote = buys["qty_quote"].sum()
        sell_quote = sells["qty_quote"].sum()

        buy_vwap = (buys["price"] * buys["qty_base"]).sum() / buy_qty if buy_qty > 0 else math.nan
        sell_vwap = (sells["price"] * sells["qty_base"]).sum() / sell_qty if sell_qty > 0 else math.nan

        avg_trade_size_base = total_qty / trade_count if trade_count > 0 else 0.0
        avg_trade_size_quote = total_quote / trade_count if trade_count > 0 else 0.0

        return {
            "trade_count": trade_count,
            "taker_buy_volume_base": buy_qty,
            "taker_sell_volume_base": sell_qty,
            "taker_buy_volume_quote": buy_quote,
            "taker_sell_volume_quote": sell_quote,
            "buy_count": len(buys),
            "sell_count": len(sells),
            "buy_vwap": buy_vwap,
            "sell_vwap": sell_vwap,
            "avg_trade_size_base": avg_trade_size_base,
            "avg_trade_size_quote": avg_trade_size_quote,
            "ofi": buy_qty - sell_qty,
        }

    grouped = df.groupby("minute_ts").apply(_agg)
    out = pd.DataFrame(list(grouped.values), index=grouped.index).reset_index().rename(columns={"minute_ts": "timestamp"})
    return out.sort_values("timestamp").reset_index(drop=True)


def derive_orderbook_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bb = df["best_bid"]
    ba = df["best_ask"]
    bq = df["bid_size_l1"]
    aq = df["ask_size_l1"]
    df["mid_price"] = (bb + ba) / 2.0
    df["spread_abs"] = ba - bb
    df["spread_pct"] = df["spread_abs"] / df["mid_price"].replace(0, np.nan)
    denom = (bq + aq).replace(0, np.nan)
    df["microprice"] = (ba * bq + bb * aq) / denom
    df["l1_imbalance"] = (bq - aq) / denom
    return df


def fetch_range_paged(fetch_fn, normalize_fn, *, start_s: int, end_s: int, limit: int, sleep_s: float, **kwargs) -> pd.DataFrame:
    to_ts = end_s - 1
    chunks = []
    last_earliest = None
    while to_ts >= start_s:
        payload = fetch_fn(to_ts=to_ts, limit=limit, **kwargs)
        df = normalize_fn(payload)
        if df.empty:
            break
        df = df[(df["timestamp"] >= start_s) & (df["timestamp"] < end_s)]
        if df.empty:
            break
        earliest = int(df["timestamp"].min())
        if last_earliest is not None and earliest >= last_earliest:
            break
        last_earliest = earliest
        chunks.append(df)
        to_ts = earliest - 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def build_month(
    client,
    *,
    market: str,
    instrument: str,
    start_dt: datetime,
    end_dt: datetime,
    limit: int,
    sleep_s: float,
    ffill_l1_limit: int = 2,
) -> pd.DataFrame:
    start_s = to_unix_s(start_dt)
    end_s = to_unix_s(end_dt)

    base = minute_index(start_s, end_s)

    ohlcv = fetch_range_paged(
        client.get_ohlcv_minutes,
        client.normalize_ohlcv_minutes,
        start_s=start_s,
        end_s=end_s,
        limit=limit,
        sleep_s=sleep_s,
        market=market,
        instrument=instrument,
        aggregate=1,
        fill=True,
    )

    trades = fetch_range_paged(
        client.get_trades,
        client.normalize_trades,
        start_s=start_s,
        end_s=end_s,
        limit=limit,
        sleep_s=sleep_s,
        market=market,
        instrument=instrument,
    )

    l2_metrics = fetch_range_paged(
        client.get_l2_metrics,
        client.normalize_l2_metrics,
        start_s=start_s,
        end_s=end_s,
        limit=limit,
        sleep_s=sleep_s,
        market=market,
        instrument=instrument,
    )

    l2_snapshots = fetch_range_paged(
        client.get_l2_snapshots,
        client.normalize_l2_snapshots,
        start_s=start_s,
        end_s=end_s,
        limit=limit,
        sleep_s=sleep_s,
        market=market,
        instrument=instrument,
    )

    trade_agg = aggregate_trades(trades) if not trades.empty else pd.DataFrame()
    l2_agg = aggregate_l2_snapshots(l2_snapshots) if not l2_snapshots.empty else pd.DataFrame()

    df = base.merge(ohlcv, on="timestamp", how="left")
    df = df.merge(trade_agg, on="timestamp", how="left", suffixes=("_ohlcv", "_trades"))
    df = df.merge(l2_agg, on="timestamp", how="left")

    # Merge L2 metrics with prefix to preserve unknown fields
    if not l2_metrics.empty:
        lm = l2_metrics.copy()
        # Map known fields to canonical names when possible
        mapping = {
            "best_bid": ["best_bid", "bid", "bid_price", "bid_px"],
            "best_ask": ["best_ask", "ask", "ask_price", "ask_px"],
            "bid_size_l1": ["bid_size", "bid_qty", "bid_size_l1"],
            "ask_size_l1": ["ask_size", "ask_qty", "ask_size_l1"],
            "mid_price": ["mid_price", "mid"],
            "spread_abs": ["spread", "spread_abs"],
            "spread_pct": ["spread_pct", "spread_percentage"],
        }
        rename_map: Dict[str, str] = {}
        for target, candidates in mapping.items():
            for cand in candidates:
                if cand in lm.columns:
                    rename_map[cand] = target
                    break
        lm = lm.rename(columns=rename_map)
        cols = [c for c in lm.columns if c != "timestamp"]
        # Prefix any remaining unknown columns
        lm = lm.rename(columns={c: f"l2m_{c}" for c in cols if c not in mapping.keys()})
        df = df.merge(lm, on="timestamp", how="left")

    # Volume quote approximation if missing
    if "volume_quote" not in df.columns:
        df["volume_quote"] = np.nan
    df["volume_quote_is_approx"] = False
    missing_vq = df["volume_quote"].isna()
    df.loc[missing_vq, "volume_quote"] = df.loc[missing_vq, "close"] * df.loc[missing_vq, "volume_base"]
    df.loc[missing_vq, "volume_quote_is_approx"] = True

    # If trade_count missing from ohlcv, use trade_agg
    if "trade_count" not in df.columns:
        df["trade_count"] = np.nan
    if "trade_count_ohlcv" in df.columns:
        df["trade_count"] = df["trade_count"].fillna(df["trade_count_ohlcv"])
    if "trade_count_trades" in df.columns:
        df["trade_count"] = df["trade_count"].fillna(df["trade_count_trades"])
    df["trade_count"] = df["trade_count"].fillna(0)

    # Fill trade-derived metrics with 0 where no trades
    trade_cols = [
        "taker_buy_volume_base", "taker_sell_volume_base",
        "taker_buy_volume_quote", "taker_sell_volume_quote",
        "buy_count", "sell_count",
        "avg_trade_size_base", "avg_trade_size_quote", "ofi",
    ]
    for c in trade_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Forward-fill L1 order book fields within a short window
    l1_cols = ["best_bid", "best_ask", "bid_size_l1", "ask_size_l1"]
    for c in l1_cols:
        if c in df.columns:
            df[c] = df[c].ffill(limit=ffill_l1_limit)

    # Derive order book metrics from L1
    if all(c in df.columns for c in ["best_bid", "best_ask", "bid_size_l1", "ask_size_l1"]):
        df = derive_orderbook_metrics(df)

    # Ensure required columns exist
    required_cols = [
        "timestamp", "open", "high", "low", "close",
        "volume_base", "volume_quote", "trade_count",
        "taker_buy_volume_base", "taker_sell_volume_base",
        "taker_buy_volume_quote", "taker_sell_volume_quote",
        "best_bid", "best_ask", "bid_size_l1", "ask_size_l1",
        "bid_depth_5", "ask_depth_5", "bid_depth_10", "ask_depth_10", "bid_depth_20", "ask_depth_20",
        "l2_imbalance_5", "l2_imbalance_10", "l2_imbalance_20",
        "vwap_bid_5", "vwap_ask_5", "vwap_bid_10", "vwap_ask_10", "vwap_bid_20", "vwap_ask_20",
        "mid_price", "spread_abs", "spread_pct", "microprice", "l1_imbalance",
        "buy_count", "sell_count", "buy_vwap", "sell_vwap",
        "avg_trade_size_base", "avg_trade_size_quote", "ofi",
        "volume_quote_is_approx",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Acceptance checks
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) != len(base):
        raise ValueError(f"Row count mismatch: expected {len(base)} minutes, got {len(df)}")

    # OHLC sanity
    ohlc_bad = ((df["high"] < df[["open", "close"]].max(axis=1)) |
                (df["low"] > df[["open", "close"]].min(axis=1)))
    if ohlc_bad.any():
        print(f"[warn] OHLC sanity violations: {ohlc_bad.sum()} rows")

    # Top-of-book coverage
    top_ok = df["best_bid"].replace([np.inf, -np.inf], np.nan).notna() & df["best_ask"].replace([np.inf, -np.inf], np.nan).notna()
    coverage = top_ok.mean() if len(df) else 0.0
    print(f"[coverage] top-of-book finite: {coverage*100:.2f}%")

    # Volume reconciliation when trades exist
    if "volume_base" in df.columns:
        tb = df["taker_buy_volume_base"]
        ts = df["taker_sell_volume_base"]
        vol = df["volume_base"]
        mask = (tb + ts) > 0
        if mask.any():
            diff = (tb + ts - vol).abs() / (vol.replace(0, np.nan))
            print(f"[check] taker vs volume_base median rel diff: {diff[mask].median():.4f}")

    return df[required_cols]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified CoinDesk minute-level dataset.")
    ap.add_argument("--market", default="coinbase")
    ap.add_argument("--instrument", default="ETH-USDT")
    ap.add_argument("--out-root", default="data/coindesk")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--ffill-l1-limit", type=int, default=2)
    args = ap.parse_args()

    client = build_client_from_env()

    utc_now = datetime.now(timezone.utc)
    if args.days is not None:
        end_dt = utc_now
        start_dt = end_dt - timedelta(days=int(args.days))
    elif args.start is not None:
        start_dt = parse_date(args.start)
        end_dt = parse_date(args.end) if args.end else utc_now
    else:
        end_dt = utc_now
        start_dt = end_dt - timedelta(days=365 * 3)

    out_dir = Path(args.out_root) / args.instrument / "1m"
    ensure_dir(out_dir)

    cur = month_floor(start_dt)
    while cur < end_dt:
        m_end = min(next_month(cur), end_dt)
        w_start = max(cur, start_dt)
        w_end = m_end

        print(f"[month] {cur:%Y-%m} {w_start:%Y-%m-%d} -> {w_end:%Y-%m-%d}")
        df = build_month(
            client,
            market=args.market,
            instrument=args.instrument,
            start_dt=w_start,
            end_dt=w_end,
            limit=int(args.limit),
            sleep_s=float(args.sleep),
            ffill_l1_limit=int(args.ffill_l1_limit),
        )

        out_path = out_dir / f"{cur:%Y-%m}.csv"
        df.to_csv(out_path, index=False)
        print(f"[save] {out_path} rows={len(df):,}")

        cur = next_month(cur)

    print("✅ Done.")


if __name__ == "__main__":
    main()
