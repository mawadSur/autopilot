import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def preprocess_and_label(df, window_size=150, threshold=0.005):
    df = df.copy()

    # Feature engineering
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()

    # Technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change()

    df.dropna(inplace=True)

    # Label creation
    df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
    df['label'] = (df['target'] > threshold).astype(int)
    df.dropna(inplace=True)

    feature_cols = [
        'open', 'high', 'low', 'close', 'body', 'range',
        'upper_wick', 'lower_wick', 'return', 'sma_ratio',
        'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]

    features = df[feature_cols].values
    labels = df['label'].values

    # Create LSTM windows
    X, y = [], []
    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(labels[i])

    X = np.array(X)
    y = np.array(y)

    # Normalize
    scaler = StandardScaler()
    X_shape = X.shape
    X = scaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)

    return X, y

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return -K.mean(alpha * K.pow(1. - p_t, gamma) * K.log(p_t))
    return loss

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', 'AUC'])
    return model

def train_model():
    print("📥 Loading data...")
    df = pd.read_csv('eth_ohlc.csv', parse_dates=['date'], index_col='date')
    X, y = preprocess_and_label(df)

    print("✅ Class balance:", np.bincount(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Auto class weight
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), weights))
    print(f"📊 Class weights: {class_weight_dict}")

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        # callbacks=[early_stop],
        verbose=1
    )

    probs = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    print("\n📈 Threshold candidates (from ROC):")
    for i in range(0, len(thresholds), max(1, len(thresholds) // 5)):
        print(f"  Threshold: {thresholds[i]:.3f}, TPR: {tpr[i]:.3f}, FPR: {fpr[i]:.3f}")

    print(f"\n⚙️ Using best threshold: {best_threshold:.3f}")
    y_pred = (probs > best_threshold).astype(int)

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("📦 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("📤 Prediction Distribution:", np.bincount(y_pred.flatten()))
    print("🔍 Probabilities range:", probs.min(), "→", probs.max())

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    model.save('eth_lstm_model.h5')
    print("✅ Model saved as eth_lstm_model.h5")

if __name__ == "__main__":
    train_model()
