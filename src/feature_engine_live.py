#!/usr/bin/env python3
"""
Real-time feature engineering service.

Consumes raw candle data from Redis streams and computes technical indicators.
Maintains sliding window for feature computation and caches results for inference.
"""

import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd
import redis
from dotenv import load_dotenv

load_dotenv()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Feature computation constants (from backtest.py)
ROLL_WINDOW = 20
ATR_ALPHA = 1 / 14
RSI_ALPHA = 1 / 14

# Feature columns (from utils.py)
FEATURE_COLS = [
    "open", "high", "low", "close",
    "body", "range", "upper_wick", "lower_wick",
    "return", "sma_ratio", "ema_20", "macd", "rsi_14",
    "vol_change", "atr", "price_vs_hourly_trend", "bb_width",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators (same as backtest.py)."""
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    df = df.copy()
    df["body"] = df["close"] - df["open"]
    rng = (df["high"] - df["low"])
    df["range"] = rng.replace(0, 1e-12)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1))
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"])
    df["return"] = df["close"].pct_change().fillna(0.0)

    sma = df["close"].rolling(ROLL_WINDOW).mean()
    df["sma_ratio"] = (df["close"] / (sma + 1e-12)).fillna(1.0)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()

    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=RSI_ALPHA, adjust=False).mean()
    roll_down = down.ewm(alpha=RSI_ALPHA, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    if "volume" in df.columns:
        df["vol_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    else:
        df["vol_change"] = 0.0

    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=ATR_ALPHA, adjust=False).mean().fillna(tr.mean())

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = (macd - signal).fillna(0.0)

    hourly = df["close"].ewm(span=60, adjust=False).mean()
    df["price_vs_hourly_trend"] = (df["close"] / (hourly + 1e-12)).fillna(1.0)

    std_20 = df["close"].rolling(ROLL_WINDOW).std()
    upper = sma + 2 * std_20
    lower = sma - 2 * std_20
    df["bb_width"] = ((upper - lower) / (sma + 1e-12)).fillna(0.0)

    return df


class RealtimeFeatureEngine:
    """Real-time feature computation from Redis candle streams."""

    def __init__(self, symbol: str = "ETHUSDT", window_size: int = 192):
        self.symbol = symbol.upper()
        self.window_size = window_size

        # Redis keys (using lists instead of streams for compatibility)
        self.input_stream = f"stream:{self.symbol}:1m"
        self.feature_stream = f"stream:{self.symbol}:features"  # For inference engine
        self.cache_key = f"cache:{self.symbol}:window"

        # Redis connection
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # State
        self.candle_window: Deque[Dict] = deque(maxlen=self.window_size * 2)
        self.feature_window: Optional[np.ndarray] = None
        self.last_id = "0"

    def start(self):
        """Start consuming candle data and computing features."""
        print(f"🚀 Starting feature engine for {self.symbol}...")

        while True:
            try:
                # Read from list (blocking)
                res = self.redis.brpop(self.input_stream, timeout=1)

                if res:
                    _, message_data_json = res
                    message_data = json.loads(message_data_json)
                    
                    # Parse candle data — all fields from live_data_stream are numeric strings
                    candle = {k: float(v) for k, v in message_data.items()}
                    self._process_candle(candle)
                else:
                    # Timeout, continue
                    continue

            except KeyboardInterrupt:
                print("🛑 Feature engine stopped")
                break
            except Exception as e:
                print(f"✗ Error in feature engine: {e}")
                time.sleep(1)

    def _process_candle(self, candle: Dict):
        """Process new candle and update features."""
        # Add to window (deque handles maxlen automatically)
        self.candle_window.append(candle)

        # Compute features if we have enough data
        if len(self.candle_window) >= ROLL_WINDOW:
            self._compute_features()

            # Emit feature update
            if self.feature_window is not None:
                self._emit_features()

    def _compute_features(self):
        """Compute features from current candle window."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.candle_window[-self.window_size:])

            # Ensure required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                return

            # Compute features
            df_features = compute_features(df)

            # Extract feature matrix
            feature_data = df_features[FEATURE_COLS].values
            self.feature_window = feature_data

            # Cache in Redis
            self.redis.set(self.cache_key, json.dumps({
                "features": feature_data.tolist(),
                "timestamp": datetime.fromtimestamp(self.candle_window[-1]["time"] / 1000).isoformat(),
                "candle_count": len(self.candle_window)
            }))

        except Exception as e:
            print(f"✗ Feature computation error: {e}")

    def _emit_features(self):
        """Emit feature update to Redis list."""
        try:
            if self.feature_window is None or len(self.feature_window) < self.window_size:
                return  # Still warming up — need a full window for valid LSTM input

            # Emit full [window_size, feature_count] matrix so inference gets correct input shape
            feature_data = {
                "features": json.dumps(self.feature_window.tolist()),
                "timestamp": datetime.fromtimestamp(self.candle_window[-1]["time"] / 1000).isoformat(),
                "candle_time": str(self.candle_window[-1]["close_time"])
            }
            self.redis.lpush(self.feature_stream, json.dumps(feature_data))
            self.redis.ltrim(self.feature_stream, 0, 100)

            print(f"📊 Features updated | {self.symbol} | {self.feature_window[-1][:4]}...")

        except Exception as e:
            print(f"✗ Feature emission error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time feature engineering service")
    parser.add_argument("--symbol", default=os.getenv("TRADE_SYMBOL", "ETHUSDT"), help="Trading symbol")
    parser.add_argument("--window", type=int, default=192, help="Feature window size")

    args = parser.parse_args()

    engine = RealtimeFeatureEngine(symbol=args.symbol, window_size=args.window)
    engine.start()


if __name__ == "__main__":
    main()