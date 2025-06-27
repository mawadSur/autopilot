import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.metrics import AUC, Precision, Recall
import tensorflow as tf
from utils import load_ohlc_chunks, compute_rsi
from tensorflow.python.client import device_lib

# ==== GPU Setup ====
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✅ Enabled GPU memory growth.")
else:
    print("⚠️ No GPU found.")
print("🖥️ Available Devices:")
print(device_lib.list_local_devices())

# ==== Constants ====
WINDOW_SIZE = 150
BATCH_SIZE = 50000
FIXED_BATCH_SIZE = 16
EPOCHS_PER_CHUNK = 3  # ⬅️ adjust for shorter or longer training per chunk
THRESHOLD = 0.005
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'body', 'range',
    'upper_wick', 'lower_wick', 'return', 'sma_ratio',
    'ema_20', 'macd', 'rsi_14', 'vol_change'
]

# ==== Preprocessing ====
def preprocess_chunk(df, window_size=150, threshold=THRESHOLD, scaler=None):
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
    print("🔍 Label distribution:", np.bincount(df['label']))
    print(f"✅ Positive ratio: {df['label'].mean():.4f}")
    df.dropna(inplace=True)

    features = df[FEATURE_COLS].values.astype(np.float64)
    labels = df['label'].values

    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(labels[i])

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if scaler is not None:
        X_shape = X.shape
        X = scaler.transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    return X, y

# ==== Loss ====
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -K.mean(alpha * K.pow(1. - p_t, gamma) * K.log(p_t))
    return loss

# ==== Model ====
def build_lstm_model(input_shape, batch_size):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, input_shape[0], input_shape[1])))
    model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32, stateful=True)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss=focal_loss(),
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
    )
    return model

# ==== Training ====
def train_model():
    df = load_ohlc_chunks()
    chunk_size = BATCH_SIZE + WINDOW_SIZE
    num_chunks = (len(df) - WINDOW_SIZE) // BATCH_SIZE

    checkpoint_file = "training_checkpoint.json"
    model_file = "eth_lstm_model.h5"

    # Load checkpoint
    start_chunk = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            start_chunk = checkpoint.get("last_completed_chunk", 0) + 1
        print(f"🔁 Resuming from chunk {start_chunk}")

    # Fit and save scaler
    sample_df = df.sample(min(100000, len(df)))
    sample_X, _ = preprocess_chunk(sample_df)
    scaler = StandardScaler()
    scaler.fit(sample_X.reshape(-1, sample_X.shape[-1]))
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Scaler saved to scaler.pkl")

    model = None
    all_probs = []
    all_true = []

    for i in range(start_chunk, num_chunks):
        chunk = df.iloc[i * BATCH_SIZE: i * BATCH_SIZE + chunk_size].copy()
        X, y = preprocess_chunk(chunk, scaler=scaler)

        # Fix batch size
        excess = len(X) % FIXED_BATCH_SIZE
        if excess > 0:
            X = X[:-excess]
            y = y[:-excess]

        # Train/Val Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
        # Trim to fixed batch size
        def trim_batch(X, y, batch_size):
            excess = len(X) % batch_size
            if excess > 0:
                return X[:-excess], y[:-excess]
            return X, y

        X_train, y_train = trim_batch(X_train, y_train, FIXED_BATCH_SIZE)
        X_val, y_val = trim_batch(X_val, y_val, FIXED_BATCH_SIZE)
        if model is None:
            print("📐 Building model...")
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), batch_size=FIXED_BATCH_SIZE)

        print(f"🧪 Training on chunk {i+1}/{num_chunks} | Samples: {len(X_train)} train / {len(X_val)} val")

        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), weights))

        try:
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS_PER_CHUNK,
                batch_size=FIXED_BATCH_SIZE,
                class_weight=class_weight_dict,
                shuffle=False,
                verbose=1
            )
            model.reset_states()

            model.save(model_file)
            with open(checkpoint_file, "w") as f:
                json.dump({"last_completed_chunk": i}, f)
            print(f"✅ Checkpoint saved after chunk {i}")

            # Confidence samples
            sample_probs = model.predict(X_val[:FIXED_BATCH_SIZE], batch_size=FIXED_BATCH_SIZE).flatten()
            all_probs.extend(sample_probs)
            all_true.extend(y_val[:FIXED_BATCH_SIZE])

        except tf.errors.ResourceExhaustedError as e:
            print("❌ GPU OOM error:", e)
            break

    # === Evaluation ===
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
        print("⚠️ No predictions collected for evaluation.")

if __name__ == "__main__":
    train_model()
