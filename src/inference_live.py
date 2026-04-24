#!/usr/bin/env python3
"""
Live inference engine for real-time trading signals.

Consumes feature updates from Redis and runs model predictions.
Emits trading signals with confidence scores.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import redis
import torch
import torch.nn as nn
from dotenv import load_dotenv

load_dotenv()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Fallback constants used only if model_meta.json is missing
_DEFAULT_WINDOW_SIZE = 192
_DEFAULT_FEATURE_COUNT = 17


class LSTMClassifier(nn.Module):
    """LSTM model for price prediction (same as training)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int = 2, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        self.save_hyperparameters = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
            "num_classes": num_classes,
        }

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        direction_factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * direction_factor),
            nn.Linear(hidden_size * direction_factor, 256),  # Match saved model
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),  # Match saved model
        )

    def forward(self, x):
        # x: [B, T, F]
        _, (hn, cn) = self.lstm(x)
        # For bidirectional: hn[-2] is final forward state, hn[-1] is final backward state.
        # output[:,-1,:] is wrong — its backward half saw only 1 token, not the full sequence.
        last = torch.cat([hn[-2], hn[-1]], dim=1) if self.lstm.bidirectional else hn[-1]
        logits = self.head(last)         # [B, C]
        return logits


class LiveInferenceEngine:
    """Real-time model inference from feature streams."""

    def __init__(self, symbol: str = "ETHUSDT", model_dir: str = "./model",
                 buy_threshold: float = 0.6, sell_threshold: float = 0.6):
        self.symbol = symbol.upper()
        self.model_dir = model_dir
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # Redis keys (using lists instead of streams for compatibility)
        self.feature_stream = f"stream:{self.symbol}:features"  # Changed from stream to list
        self.signal_list = f"list:{self.symbol}:signals"
        self.signal_log_list = f"list:{self.symbol}:signals:log"
        self.cache_key = f"cache:{self.symbol}:window"

        # Redis connection
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

        # Model components
        self.model: Optional[LSTMClassifier] = None
        self.scaler: Optional[object] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_model_time = 0
        self.window_size: int = _DEFAULT_WINDOW_SIZE
        self.feature_count: int = _DEFAULT_FEATURE_COUNT

        # State
        self.last_id = "0"
        self.prediction_count = 0   # emitted signals — used as signal id
        self.messages_processed = 0  # every received message — used for periodic model reload

        # Load model
        self._load_model()

    def _load_model(self):
        """Load trained model and scaler."""
        try:
            model_path = os.path.join(self.model_dir, "model.pt")
            if not os.path.exists(model_path):
                print(f"✗ Model file not found: {model_path}")
                return

            mtime = os.path.getmtime(model_path)
            if mtime <= self.last_model_time:
                return

            # Load metadata
            meta_path = os.path.join(self.model_dir, "model_meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            # Update shape constants from meta so _predict stays in sync with the model.
            self.window_size = int(meta.get("window_size", _DEFAULT_WINDOW_SIZE))
            self.feature_count = int(meta["input_size"])

            # Load model
            self.model = LSTMClassifier(
                input_size=meta["input_size"],
                hidden_size=meta["hidden_size"],
                num_layers=meta["num_layers"],
                num_classes=meta["num_classes"],
                dropout=meta.get("dropout", 0.2),
                bidirectional=meta.get("bidirectional", True)
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            self.scaler = joblib.load(scaler_path)

            self.last_model_time = mtime
            print(f"✓ Model loaded: {meta['input_size']} features, {meta['hidden_size']} hidden")

        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            if self.model is None:
                raise

    def start(self):
        """Start consuming features and generating signals."""
        print(f"🧠 Starting inference engine for {self.symbol}...")

        while True:
            try:
                # Read from list (blocking)
                res = self.redis.brpop(self.feature_stream, timeout=1)

                if res:
                    self.messages_processed += 1
                    if self.messages_processed % 100 == 0:
                        self._load_model()

                    _, message_data_json = res
                    message_data = json.loads(message_data_json)

                    # Parse features
                    features = json.loads(message_data["features"])
                    timestamp = message_data["timestamp"]
                    candle_time = message_data["candle_time"]

                    # Generate prediction
                    signal = self._predict(features, timestamp, candle_time)

                    if signal:
                        self._emit_signal(signal)
                else:
                    continue

            except KeyboardInterrupt:
                print("🛑 Inference engine stopped")
                break
            except Exception as e:
                print(f"✗ Error in inference engine: {e}")
                time.sleep(1)

    def _predict(self, feature_vector: List[float], timestamp: str, candle_time: str) -> Optional[Dict]:
        """Run model prediction on feature vector."""
        try:
            if self.model is None or self.scaler is None:
                return None

            # Reconstruct full window: [window_size, feature_count]
            features = np.array(feature_vector, dtype=np.float32)
            if features.shape != (self.window_size, self.feature_count):
                print(f"✗ Unexpected feature shape {features.shape}, expected ({self.window_size}, {self.feature_count})")
                return None

            # Scale all rows, then add batch dimension: [1, window_size, feature_count]
            features_scaled = self.scaler.transform(features)
            X = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(X)
                probabilities = torch.softmax(outputs, dim=1)

            # Model is trained as binary: class 1 = next bar up, class 0 = not up.
            # Always threshold on buy_prob (P(class=1)) so a neutral 0.5 never emits a sell.
            buy_prob = float(probabilities[0][1])

            if buy_prob >= self.buy_threshold:
                action = "BUY"
                confidence = buy_prob
                threshold = self.buy_threshold
            elif buy_prob <= (1.0 - self.sell_threshold):
                action = "SELL"
                confidence = 1.0 - buy_prob  # how confident we are the bar will NOT go up
                threshold = self.sell_threshold
            else:
                return None  # hold — no signal emitted

            self.prediction_count += 1

            return {
                "id": self.prediction_count,
                "symbol": self.symbol,
                "action": action,
                "confidence": confidence,
                "threshold": threshold,
                "timestamp": timestamp,
                "candle_time": candle_time,
                "features": feature_vector
            }

        except Exception as e:
            print(f"✗ Prediction error: {e}")

        return None

    def _emit_signal(self, signal: Dict):
        """Emit trading signal to Redis list."""
        try:
            self.redis.lpush(self.signal_list, json.dumps(signal))
            self.redis.lpush(self.signal_log_list, json.dumps(signal))

            print(f"🎯 Signal: {signal['action']} | Conf: {signal['confidence']:.3f} | {self.symbol}")

        except Exception as e:
            print(f"✗ Signal emission error: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Live inference engine")
    parser.add_argument("--symbol", default=os.getenv("TRADE_SYMBOL", "ETHUSDT"), help="Trading symbol")
    parser.add_argument("--model-dir", default="./model", help="Model directory")
    parser.add_argument("--buy-threshold", type=float, default=0.6, help="Buy signal threshold")
    parser.add_argument("--sell-threshold", type=float, default=0.6, help="Sell signal threshold")

    args = parser.parse_args()

    engine = LiveInferenceEngine(
        symbol=args.symbol,
        model_dir=args.model_dir,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold
    )
    engine.start()


if __name__ == "__main__":
    main()
