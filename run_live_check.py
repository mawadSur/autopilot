import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from utils import compute_rsi
from collections import deque

WINDOW_SIZE = 150
FIXED_BATCH_SIZE = 16
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'body', 'range',
    'upper_wick', 'lower_wick', 'return', 'sma_ratio',
    'ema_20', 'macd', 'rsi_14', 'vol_change'
]

# ==== Load trained components ====
scaler = joblib.load("scaler.pkl")
model = load_model("eth_lstm_model.h5", compile=False)
window = deque(maxlen=WINDOW_SIZE)

# ==== Threshold from model_meta.json ====
import json
try:
    with open("model_meta.json", "r") as f:
        threshold = json.load(f).get("threshold", 0.5)
except Exception:
    threshold = 0.5

def prepare_row_features(row, prev_close=None, prev_volume=None):
    row['body'] = row['close'] - row['open']
    row['range'] = row['high'] - row['low']
    row['upper_wick'] = row['high'] - max(row['close'], row['open'])
    row['lower_wick'] = min(row['close'], row['open']) - row['low']
    row['return'] = (row['close'] - prev_close) / prev_close if prev_close else 0
    row['sma_ratio'] = 0  # cannot compute live
    row['ema_20'] = row['close']  # fallback to raw
    row['macd'] = 0
    row['rsi_14'] = 50  # neutral assumption
    row['vol_change'] = (row['volume'] - prev_volume) / prev_volume if prev_volume else 0
    return row

def predict_live(new_row, prev_close=None, prev_volume=None):
    row = prepare_row_features(new_row, prev_close, prev_volume)
    row_vec = np.array([[row[col] for col in FEATURE_COLS]])

    # === Use pre-fitted scaler from training ===
    row_vec = scaler.transform(row_vec)
    window.append(row_vec.flatten())

    if len(window) < WINDOW_SIZE:
        return {
            "confidence": None,
            "signal": 0,
            "reason": "Not enough data to make prediction"
        }

    # === Build batch of repeated input to satisfy stateful model ===
    X_live = np.array([list(window)] * FIXED_BATCH_SIZE)
    confidence = model.predict(X_live, batch_size=FIXED_BATCH_SIZE).flatten()[0]
    model.reset_states()

    signal = int(confidence > threshold)

    return {
        "confidence": round(float(confidence), 5),
        "signal": signal,
        "threshold": round(float(threshold), 3)
    }
