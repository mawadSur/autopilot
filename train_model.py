
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf
from utils import load_ohlc_chunks, compute_rsi
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# ==== GPU Memory Growth Setup ====
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Enabled GPU memory growth.")
    except RuntimeError as e:
        print("❌ GPU memory setup failed:", e)
else:
    print("⚠️ No GPU found.")

# ==== GPU Debug Info ====
from tensorflow.python.client import device_lib
print("🖥️ Available Devices:")
print(device_lib.list_local_devices())

# ==== Constants ====
WINDOW_SIZE = 150
BATCH_SIZE = 50000
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'body', 'range',
    'upper_wick', 'lower_wick', 'return', 'sma_ratio',
    'ema_20', 'macd', 'rsi_14', 'vol_change'
]

def preprocess_chunk(df, window_size=150, threshold=0.005, scaler=None):
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['label'] = (df['target'] > threshold).astype(int)
    df.dropna(inplace=True)

    features = df[FEATURE_COLS].values.astype(np.float64)
    labels = df['label'].values

    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(labels[i])

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    # Sanitize input
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        X_shape = X.shape
        X = scaler.transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    # Assert checks
    assert not np.isnan(X).any(), "❌ X contains NaNs"
    assert not np.isinf(X).any(), "❌ X contains Infs"

    return X, y

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -K.mean(alpha * K.pow(1. - p_t, gamma) * K.log(p_t))
    return loss

def build_lstm_model(input_shape, batch_size):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, input_shape[0], input_shape[1])))
    model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32, stateful=True)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', 'AUC'])
    return model

def train_model():
    df = load_ohlc_chunks()
    chunk_size = BATCH_SIZE + WINDOW_SIZE
    num_chunks = (len(df) - WINDOW_SIZE) // BATCH_SIZE

    checkpoint_file = "training_checkpoint.json"
    model_file = "eth_lstm_model.h5"

    # --- Load progress ---
    start_chunk = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            start_chunk = checkpoint.get("last_completed_chunk", 0) + 1
        print(f"🔁 Resuming from chunk {start_chunk}")

    # --- Fit scaler on a sample ---
    sample_df = df.sample(min(100000, len(df)))
    sample_X, _ = preprocess_chunk(sample_df)
    scaler = StandardScaler()
    scaler.fit(sample_X.reshape(-1, sample_X.shape[-1]))

    model = None
    if os.path.exists(model_file):
        print("📦 Loading saved model...")
        model = tf.keras.models.load_model(model_file, custom_objects={"loss": focal_loss()})

    all_probs = []
    all_true = []

    model = None
    fixed_batch_size = 16  # required for stateful LSTM

    for i in range(start_chunk, num_chunks):
        chunk = df.iloc[i * BATCH_SIZE: i * BATCH_SIZE + chunk_size].copy()
        X, y = preprocess_chunk(chunk, scaler=scaler)

        # Ensure fixed batch size
        excess = len(X) % fixed_batch_size
        if excess > 0:
            X = X[:-excess]
            y = y[:-excess]

        if model is None:
            print("📐 Building model (stateful)...")
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), batch_size=fixed_batch_size)

        print(f"🧪 Training on chunk {i+1}/{num_chunks} with {len(X)} samples")

        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), weights))

        try:
            model.fit(
                X, y,
                epochs=1,
                batch_size=fixed_batch_size,
                class_weight=class_weight_dict,
                shuffle=False,
                verbose=1
            )
            model.reset_states()

            model.save(model_file)
            with open(checkpoint_file, "w") as f:
                json.dump({"last_completed_chunk": i}, f)

            print(f"✅ Saved checkpoint after chunk {i}")

            # Predict confidence for sample
            sample_probs = model.predict(X[:fixed_batch_size], batch_size=fixed_batch_size).flatten()
            all_probs.extend(sample_probs)
            all_true.extend(y[:fixed_batch_size])

        except tf.errors.ResourceExhaustedError as e:
            print("❌ GPU OOM error:", e)
            break

    if all_probs:
        probs = np.array(all_probs)
        y_test = np.array(all_true)

        fpr, tpr, thresholds = roc_curve(y_test, probs)
        best_threshold = float(thresholds[np.argmax(tpr - fpr)])
        y_pred = (probs > best_threshold).astype(int)

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("📦 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        with open("model_meta.json", "w") as f:
            json.dump({"threshold": best_threshold}, f)
        print("✅ Threshold saved.")

    else:
        print("⚠️ No predictions made yet.")

if __name__ == "__main__":
    train_model()
