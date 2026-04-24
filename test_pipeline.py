#!/usr/bin/env python3
"""
Test script for the live trading pipeline.

Tests individual components and their integration via Redis.
"""

import json
import time
import subprocess
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import redis
from dotenv import load_dotenv

load_dotenv()

def test_redis_connection():
    """Test Redis connectivity."""
    print("🔍 Testing Redis connection...")
    try:
        r = redis.Redis()
        r.ping()
        print("✅ Redis connection: OK")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_feature_computation():
    """Test feature computation."""
    print("🔍 Testing feature computation...")
    try:
        from feature_engine_live import compute_features
        import pandas as pd
        import numpy as np

        # Create test data
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [103, 104, 105],
            'low': [99, 98, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })

        result = compute_features(df)
        expected_features = [
            "open", "high", "low", "close", "body", "range", "upper_wick", "lower_wick",
            "return", "sma_ratio", "ema_20", "macd", "rsi_14", "vol_change", "atr",
            "price_vs_hourly_trend", "bb_width"
        ]

        if all(col in result.columns for col in expected_features):
            print("✅ Feature computation: OK")
            return True
        else:
            print("❌ Feature computation: Missing expected columns")
            return False

    except Exception as e:
        print(f"❌ Feature computation failed: {e}")
        return False

def test_model_loading():
    """Test model and scaler loading."""
    print("🔍 Testing model loading...")
    try:
        from inference_live import LiveInferenceEngine

        engine = LiveInferenceEngine(model_dir='./model')
        print("✅ Model loading: OK")
        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_redis_data_flow():
    """Test data flow through Redis."""
    print("🔍 Testing Redis data flow...")
    try:
        r = redis.Redis(decode_responses=True)

        # Clear any existing data
        r.delete("test:candle:data")

        # Simulate candle data
        test_candle = {
            "symbol": "ETHUSDT",
            "time": 1640995200000,  # 2022-01-01 00:00:00
            "close_time": 1640995260000,
            "open": 3680.00,
            "high": 3685.00,
            "low": 3675.00,
            "close": 3682.00,
            "volume": 123.45,
            "timestamp": "2022-01-01T00:00:00"
        }

        # Publish to Redis (using simple key for testing)
        r.set("test:candle:data", json.dumps(test_candle))

        # Check if data was stored
        data = r.get("test:candle:data")
        if data:
            retrieved = json.loads(data)
            if retrieved["symbol"] == "ETHUSDT":
                print("✅ Redis data publishing: OK")
                return True

        print("❌ Redis data publishing: Failed")
        return False

    except Exception as e:
        print(f"❌ Redis data flow test failed: {e}")
        return False

def test_pipeline_integration():
    """Test end-to-end inference: feature vector → signal with correct threshold logic."""
    print("🔍 Testing pipeline integration (inference + signal thresholds)...")
    try:
        import numpy as np
        from inference_live import LiveInferenceEngine

        engine = LiveInferenceEngine(model_dir='./model')

        # Build a realistic zero-mean feature window (scaler will normalise it).
        rng = np.random.default_rng(42)
        window = rng.standard_normal((engine.window_size, engine.feature_count)).astype(np.float32)
        feature_vector = window.tolist()

        result = engine._predict(feature_vector, "2024-01-01T00:00:00", "1704067200000")

        # _predict returns None (hold) or a dict — both are valid, neither should raise.
        if result is not None:
            assert result["action"] in ("BUY", "SELL"), f"Unexpected action: {result['action']}"
            assert 0.0 <= result["confidence"] <= 1.0, f"Confidence out of range: {result['confidence']}"
            if result["action"] == "BUY":
                assert result["confidence"] >= engine.buy_threshold, "BUY confidence below threshold"
            else:
                assert result["confidence"] >= engine.sell_threshold, "SELL confidence below threshold"

        # Verify threshold symmetry: prob=0.5 must never emit a SELL signal.
        import torch
        fake_probs = torch.tensor([[0.5, 0.5]])
        buy_prob = float(fake_probs[0][1])
        assert not (buy_prob <= (1.0 - engine.sell_threshold)), \
            "prob=0.5 would emit a SELL — threshold logic is broken"

        print(f"✅ Pipeline integration: OK (result={'HOLD' if result is None else result['action']})")
        return True

    except Exception as e:
        print(f"❌ Pipeline integration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 LIVE TRADING PIPELINE TEST SUITE")
    print("=" * 50)

    tests = [
        test_redis_connection,
        test_feature_computation,
        test_model_loading,
        test_redis_data_flow,
        test_pipeline_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())