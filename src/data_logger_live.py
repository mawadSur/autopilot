#!/usr/bin/env python3
"""
Live data logger for persisting real-time market data.

Consumes candle data from Redis streams and appends to HDF5 store.
Maintains time-series database for backtesting and analysis.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv

load_dotenv()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# HDF5 settings
HDF5_STORE_PATH = os.getenv("HDF5_STORE_PATH", "market_data_store.h5")
BATCH_SIZE = int(os.getenv("LOG_BATCH_SIZE", "100"))  # Write every N candles


class LiveDataLogger:
    """Logger for persisting live market data to HDF5."""

    def __init__(self, symbol: str = "ETHUSDT", store_path: str = HDF5_STORE_PATH):
        self.symbol = symbol.upper()
        self.store_path = store_path

        # Redis (using lists instead of streams for compatibility)
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.candle_stream = f"stream:{self.symbol}:1m"
        self.signal_log_list = f"list:{self.symbol}:signals:log"
        self.trade_list = f"list:{self.symbol}:trades"
        self.last_candle_id = "0"
        self.last_signal_id = "0"
        self.last_trade_id = "0"

        # Buffers for batch writing
        self.candle_buffer: List[Dict] = []
        self.signal_buffer: List[Dict] = []
        self.trade_buffer: List[Dict] = []

        # HDF5 groups
        self.candle_group = f"/{self.symbol}/1m/candles"
        self.signal_group = f"/{self.symbol}/1m/signals"
        self.trade_group = f"/{self.symbol}/1m/trades"

    def start(self):
        """Start logging data from Redis streams."""
        print(f"📝 Starting data logger for {self.symbol}...")

        # Ensure HDF5 structure exists
        self._init_hdf5_structure()

        while True:
            try:
                # Read from candle list — matches LPUSH writer in live_data_stream.py
                candle_res = self.redis.brpop(self.candle_stream, timeout=0.1)
                if candle_res:
                    _, candle_json = candle_res
                    candle = {k: float(v) for k, v in json.loads(candle_json).items()}
                    self._process_candle(candle)

                # Poll lists for signals and trades
                signal_data = self.redis.rpop(self.signal_log_list)
                trade_data = self.redis.rpop(self.trade_list)

                if signal_data:
                    self._process_signal(json.loads(signal_data))

                if trade_data:
                    self._process_trade(json.loads(trade_data))

                # Batch write when buffers are full
                self._check_batch_write()

            except KeyboardInterrupt:
                print("🛑 Data logger stopped")
                self._flush_buffers()  # Write remaining data
                break
            except Exception as e:
                print(f"✗ Data logger error: {e}")
                time.sleep(1)

    def _init_hdf5_structure(self):
        """Initialize HDF5 groups and datasets."""
        try:
            with h5py.File(self.store_path, 'a') as f:
                # Create groups
                if self.candle_group not in f:
                    f.create_group(self.candle_group)
                if self.signal_group not in f:
                    f.create_group(self.signal_group)
                if self.trade_group not in f:
                    f.create_group(self.trade_group)

                print(f"✓ HDF5 structure initialized for {self.symbol}")

        except Exception as e:
            print(f"✗ Failed to initialize HDF5: {e}")

    def _process_candle(self, candle: Dict):
        """Process and buffer candle data."""
        # Convert to standard format
        candle_record = {
            "timestamp": candle["time"],
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
            "close_time": candle["close_time"],
            "trades": candle.get("trades", 0),
            "logged_at": datetime.now().timestamp()
        }

        self.candle_buffer.append(candle_record)

    def _process_signal(self, signal: Dict):
        """Process and buffer signal data."""
        signal_record = {
            "timestamp": datetime.fromisoformat(signal["timestamp"]).timestamp(),
            "candle_time": int(signal["candle_time"]),
            "action": 1 if signal["action"] == "BUY" else 0,  # 1=BUY, 0=SELL
            "confidence": signal["confidence"],
            "threshold": signal["threshold"],
            "logged_at": datetime.now().timestamp()
        }

        self.signal_buffer.append(signal_record)

    def _process_trade(self, trade: Dict):
        """Process and buffer trade data."""
        trade_record = {
            "timestamp": datetime.fromisoformat(trade["timestamp"]).timestamp(),
            "action": trade["action"],  # "OPEN" or "CLOSE"
            "side": 1 if trade["side"] == "BUY" else 0,  # 1=BUY, 0=SELL
            "price": trade["price"],
            "pnl_pct": trade.get("pnl_pct", 0.0),
            "signal_confidence": trade.get("signal_confidence", 0.0),
            "dry_run": trade.get("dry_run", True),
            "logged_at": datetime.now().timestamp()
        }

        self.trade_buffer.append(trade_record)

    def _check_batch_write(self):
        """Write buffers to HDF5 when they reach batch size."""
        if len(self.candle_buffer) >= BATCH_SIZE:
            self._write_candles()
        if len(self.signal_buffer) >= BATCH_SIZE:
            self._write_signals()
        if len(self.trade_buffer) >= BATCH_SIZE:
            self._write_trades()

    def _flush_buffers(self):
        """Write all remaining buffered data."""
        if self.candle_buffer:
            self._write_candles()
        if self.signal_buffer:
            self._write_signals()
        if self.trade_buffer:
            self._write_trades()

    def _write_candles(self):
        """Write candle buffer to HDF5."""
        try:
            df = pd.DataFrame(self.candle_buffer)
            self._append_to_hdf5(self.candle_group, df, "candles")
            print(f"💾 Wrote {len(self.candle_buffer)} candles to HDF5")
            self.candle_buffer.clear()

        except Exception as e:
            print(f"✗ Failed to write candles: {e}")

    def _write_signals(self):
        """Write signal buffer to HDF5."""
        try:
            df = pd.DataFrame(self.signal_buffer)
            self._append_to_hdf5(self.signal_group, df, "signals")
            print(f"💾 Wrote {len(self.signal_buffer)} signals to HDF5")
            self.signal_buffer.clear()

        except Exception as e:
            print(f"✗ Failed to write signals: {e}")

    def _write_trades(self):
        """Write trade buffer to HDF5."""
        try:
            df = pd.DataFrame(self.trade_buffer)
            self._append_to_hdf5(self.trade_group, df, "trades")
            print(f"💾 Wrote {len(self.trade_buffer)} trades to HDF5")
            self.trade_buffer.clear()

        except Exception as e:
            print(f"✗ Failed to write trades: {e}")

    def _append_to_hdf5(self, group_path: str, df: pd.DataFrame, name: str):
        """Append DataFrame to HDF5 dataset."""
        try:
            with h5py.File(self.store_path, 'a') as f:
                group = f[group_path]

                # Create or append to dataset
                if name in group:
                    # Extend existing dataset
                    dset = group[name]
                    current_size = dset.shape[0]
                    new_size = current_size + len(df)

                    # Resize dataset
                    dset.resize((new_size,))

                    # Convert df to numpy and append
                    data_array = df.to_records(index=False)
                    dset[current_size:] = data_array
                else:
                    # Create new dataset
                    data_array = df.to_records(index=False)
                    dset = group.create_dataset(
                        name,
                        data=data_array,
                        maxshape=(None,),  # Allow unlimited extension
                        dtype=data_array.dtype,
                        compression="gzip",
                        chunks=True
                    )

                    # Store column names as attributes
                    dset.attrs["columns"] = list(df.columns)

        except Exception as e:
            print(f"✗ HDF5 append error for {name}: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Live data logger")
    parser.add_argument("--symbol", default=os.getenv("TRADE_SYMBOL", "ETHUSDT"), help="Trading symbol")
    parser.add_argument("--store-path", default=HDF5_STORE_PATH, help="HDF5 store path")

    args = parser.parse_args()

    logger = LiveDataLogger(symbol=args.symbol, store_path=args.store_path)
    logger.start()


if __name__ == "__main__":
    main()
