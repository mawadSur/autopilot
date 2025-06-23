import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import json
from utils import load_ohlc_chunks, compute_rsi

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

    if scaler is not None:
        X_shape = X.shape
        X = scaler.transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    return X, y

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -K.mean(alpha * K.pow(1. - p_t, gamma) * K.log(p_t))
    return loss

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', 'AUC'])
    return model

def train_model():
    df = load_ohlc_chunks()
    chunk_size = BATCH_SIZE + WINDOW_SIZE
    num_chunks = (len(df) - WINDOW_SIZE) // BATCH_SIZE

    print("⚖️ Fitting StandardScaler on sample...")
    sample_df = df.sample(min(100000, len(df)))
    sample_X, _ = preprocess_chunk(sample_df)
    scaler = StandardScaler()
    scaler.fit(sample_X.reshape(-1, sample_X.shape[-1]))

    model = None
    all_probs = []
    all_true = []

    for i in range(num_chunks):
        chunk = df.iloc[i * BATCH_SIZE : i * BATCH_SIZE + chunk_size].copy()
        X, y = preprocess_chunk(chunk, scaler=scaler)

        if model is None:
            print("📐 Building model...")
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

        print(f"🧪 Training on chunk {i+1}/{num_chunks} with {len(X)} samples")
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), weights))

        model.fit(
            X, y,
            epochs=1,  # Train one epoch per chunk
            batch_size=32,
            class_weight=class_weight_dict,
            verbose=1
        )

        probs = model.predict(X).flatten()
        all_probs.extend(probs)
        all_true.extend(y)

    probs = np.array(all_probs)
    y_test = np.array(all_true)

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_threshold = float(thresholds[np.argmax(tpr - fpr)])
    y_pred = (probs > best_threshold).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("📦 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save("eth_lstm_model.h5")
    with open("model_meta.json", "w") as f:
        json.dump({"threshold": best_threshold}, f)
    print("✅ Model saved with optimal threshold")

if __name__ == "__main__":
    train_model()
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import json
from utils import load_ohlc_chunks, compute_rsi

WINDOW_SIZE = 150
BATCH_SIZE = 500000

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

    if scaler is not None:
        X_shape = X.shape
        X = scaler.transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    return X, y

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, dtype='float32')
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -K.mean(alpha * K.pow(1. - p_t, gamma) * K.log(p_t))
    return loss

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', 'AUC'])
    return model

def train_model():
    df = load_ohlc_chunks()
    chunk_size = BATCH_SIZE + WINDOW_SIZE
    num_chunks = (len(df) - WINDOW_SIZE) // BATCH_SIZE

    print("⚖️ Fitting StandardScaler on sample...")
    sample_df = df.sample(min(100000, len(df)))
    sample_X, _ = preprocess_chunk(sample_df)
    scaler = StandardScaler()
    scaler.fit(sample_X.reshape(-1, sample_X.shape[-1]))

    model = None
    all_probs = []
    all_true = []

    for i in range(num_chunks):
        chunk = df.iloc[i * BATCH_SIZE : i * BATCH_SIZE + chunk_size].copy()
        X, y = preprocess_chunk(chunk, scaler=scaler)

        if model is None:
            print("📐 Building model...")
            model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

        print(f"🧪 Training on chunk {i+1}/{num_chunks} with {len(X)} samples")
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), weights))

        model.fit(
            X, y,
            epochs=1,  # Train one epoch per chunk
            batch_size=32,
            class_weight=class_weight_dict,
            verbose=1
        )

        probs = model.predict(X).flatten()
        all_probs.extend(probs)
        all_true.extend(y)

    probs = np.array(all_probs)
    y_test = np.array(all_true)

    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_threshold = float(thresholds[np.argmax(tpr - fpr)])
    y_pred = (probs > best_threshold).astype(int)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("📦 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    model.save("eth_lstm_model.h5")
    with open("model_meta.json", "w") as f:
        json.dump({"threshold": best_threshold}, f)
    print("✅ Model saved with optimal threshold")

if __name__ == "__main__":
    train_model()
