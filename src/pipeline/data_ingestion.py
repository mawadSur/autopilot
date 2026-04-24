#!/usr/bin/env python3
"""
Data ingestion pipeline with efficient batching and incremental updates.

Provides:
- Fast incremental data fetching from Binance
- Parallel data loading with chunking
- Data validation and normalization
- SQLite local cache for offline access
- Streaming data to consumers
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Generator, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from binance.client import Client
from history import (
    backfill_months,
    fetch_month,
    normalize_klines,
    to_ms,
)

load_dotenv()


DATA_DB_PATH = os.getenv("DATA_DB_PATH", "data_cache.db")


class DataCache:
    """
    Local SQLite cache for OHLCV data with incremental updates.
    """
    
    def __init__(self, db_path: str = DATA_DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_time 
            ON ohlcv(symbol, interval, open_time)
        """)
        conn.commit()
        conn.close()
    
    def get_range(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int,
    ) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("""
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            AND open_time >= ? AND open_time < ?
            ORDER BY open_time
        """, conn, params=(symbol, interval, start_time, end_time))
        conn.close()
        return df
    
    def put_klines(
        self,
        symbol: str,
        interval: str,
        klines: List[List],
    ) -> int:
        if not klines:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for k in klines:
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv 
                (symbol, interval, open_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, interval, int(k[0]), float(k[1]), float(k[2]),
                 float(k[3]), float(k[4]), float(k[5])))
        
        conn.commit()
        count = cursor.rowcount
        conn.close()
        return count
    
    def get_latest_time(self, symbol: str, interval: str) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT MAX(open_time) FROM ohlcv
            WHERE symbol = ? AND interval = ?
        """, (symbol, interval))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    
    def get_count(self, symbol: str, interval: str) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM ohlcv
            WHERE symbol = ? AND interval = ?
        """, (symbol, interval))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else 0


class IncrementalFetcher:
    """
    Fast incremental data fetcher with batched parallel requests.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        cache: Optional[DataCache] = None,
    ):
        self.api_key = api_key or os.getenv("BINANCE_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET")
        self.testnet = testnet
        self.cache = cache or DataCache()
        
        self.base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com"
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List]:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        session = await self._get_session()
        
        async with session.get(
            f"{self.base_url}/v3/klines",
            params=params,
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"API error: {response.status}")
            return await response.json()
    
    async def fetch_incremental(
        self,
        symbol: str,
        interval: str,
        lookback_hours: int = 24,
    ) -> pd.DataFrame:
        latest_cached = self.cache.get_latest_time(symbol, interval)
        
        now = datetime.now(timezone.utc)
        start_dt = now - timedelta(hours=lookback_hours)
        
        if latest_cached:
            start_dt = max(
                start_dt,
                datetime.fromtimestamp(latest_cached / 1000, tz=timezone.utc)
            )
        
        end_dt = now
        
        klines = await self.fetch_klines(
            symbol,
            interval,
            to_ms(start_dt),
            to_ms(end_dt),
        )
        
        if klines:
            self.cache.put_klines(symbol, interval, klines)
        
        df = normalize_klines(klines)
        
        if latest_cached:
            existing = self.cache.get_range(
                symbol,
                interval,
                latest_cached,
                to_ms(end_dt),
            )
            if not existing.empty:
                # get_range returns open_time (int ms); normalize to match normalize_klines output
                existing = existing.rename(columns={"open_time": "timestamp"})
                existing["timestamp"] = pd.to_datetime(existing["timestamp"], unit="ms", utc=True)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"])
                df = df.sort_values("timestamp")
        
        return df
    
    async def fetch_recent(
        self,
        symbol: str,
        interval: str,
        count: int = 200,
    ) -> pd.DataFrame:
        klines = await self.fetch_klines(
            symbol,
            interval,
            limit=count,
        )
        
        self.cache.put_klines(symbol, interval, klines)
        
        return normalize_klines(klines)


class DataStreamer:
    """
    Streaming data provider with batch processing and backpressure.
    """
    
    def __init__(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        window_size: int = 192,
        batch_size: int = 100,
    ):
        self.symbol = symbol
        self.interval = interval
        self.window_size = window_size
        self.batch_size = batch_size
        
        self.fetcher = IncrementalFetcher()
        self.cache = DataCache()
        
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._running = False
        self._buffer: Deque[pd.DataFrame] = deque(maxlen=200)
    
    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._fetch_loop())
    
    async def stop(self) -> None:
        self._running = False
        await self.fetcher.close()
    
    async def _fetch_loop(self) -> None:
        while self._running:
            try:
                df = await self.fetcher.fetch_recent(
                    self.symbol,
                    self.interval,
                    self.batch_size,
                )
                
                if not df.empty:
                    self._buffer.append(df)
                    await self._queue.put(df)
                
                await asyncio.sleep(1)
            except Exception as e:
                print(f"[data-streamer] Fetch error: {e}")
                await asyncio.sleep(5)
    
    async def get_window(self) -> Optional[pd.DataFrame]:
        if self._queue.empty():
            return None
        
        try:
            return await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            return None
    
    @property
    def recent_windows(self) -> List[pd.DataFrame]:
        return list(self._buffer)


def batch_fetch_historical(
    symbols: List[str],
    interval: str = "1m",
    lookback_days: int = 30,
    db_path: str = DATA_DB_PATH,
) -> Dict[str, pd.DataFrame]:
    """
    Batch fetch historical data for multiple symbols.
    
    Args:
        symbols: List of trading symbols (e.g., ["ETHUSDT", "BTCUSDT"])
        interval: Kline interval
        lookback_days: Days of historical data to fetch
        db_path: Path to SQLite cache
        
    Returns:
        Dictionary of symbol -> DataFrame
    """
    import concurrent.futures

    cache = DataCache(db_path)
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        
        for symbol in symbols:
            futures[executor.submit(
                _fetch_symbol_data,
                symbol,
                interval,
                lookback_days,
                db_path,
            )] = symbol
        
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                results[symbol] = future.result()
            except Exception as e:
                print(f"[batch] Error fetching {symbol}: {e}")
    
    return results


def _fetch_symbol_data(
    symbol: str,
    interval: str,
    lookback_days: int,
    db_path: str,
) -> pd.DataFrame:
    cache = DataCache(db_path)

    api_key = os.getenv("BINANCE_KEY")
    api_secret = os.getenv("BINANCE_SECRET")
    if not api_key or not api_secret:
        raise ValueError("Missing BINANCE_KEY / BINANCE_SECRET environment variables")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_days)

        backfill_months(
            client=Client(api_key, api_secret),
            symbol=symbol,
            interval=interval,
            start_dt=start,
            end_dt=now,
            out_dir=tmpdir,
            skip_complete=False,
        )

        from utils import read_csv_concat_sorted
        return read_csv_concat_sorted(tmpdir)


class DataAPIFast:
    """
    Fast data API with caching and precomputed windows.
    """
    
    def __init__(
        self,
        model_dir: str = "../model",
        cache: Optional[DataCache] = None,
    ):
        self.model_dir = model_dir
        self.cache = cache or DataCache()
        
        from utils import load_meta
        self.meta = load_meta(model_dir)
        
        self.feature_cols = list(self.meta.get("feature_cols", []))
        self.window_size = int(self.meta.get("window_size", 192))
        self._window_cache: Dict[str, np.ndarray] = {}
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        features = df[self.feature_cols].values.astype(np.float32)
        
        from utils import build_windows
        windows = build_windows(features, self.window_size)
        
        return windows
    
    def get_cached_windows(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[np.ndarray]:
        cache_key = f"{symbol}:{interval}"
        return self._window_cache.get(cache_key)
    
    def cache_windows(
        self,
        symbol: str,
        interval: str,
        windows: np.ndarray,
    ) -> None:
        cache_key = f"{symbol}:{interval}"
        
        if len(self._window_cache) > 10:
            oldest = next(iter(self._window_cache))
            del self._window_cache[oldest]
        
        self._window_cache[cache_key] = windows


__all__ = [
    "DataCache",
    "IncrementalFetcher",
    "DataStreamer",
    "batch_fetch_historical",
    "DataAPIFast",
    "DATA_DB_PATH",
]