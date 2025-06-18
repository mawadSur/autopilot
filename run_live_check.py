import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------- Data & Model Setup ----------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_latest_data(symbol="ETHUSDT", interval="1m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if isinstance(data, dict) and "code" in data:
            print("⚠️ Binance API error:", data)
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error fetching Binance data: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["date"] = pd.to_datetime(df["close_time"], unit="ms")
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def build_features(df, window_size=150):
    df = df.copy()
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['return'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_ratio'] = df['sma_10'] / df['sma_50'] - 1
    df['vol_change'] = df['volume'].pct_change()

    # ✅ Add missing indicators
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['rsi_14'] = compute_rsi(df['close'], period=14)

    df.dropna(inplace=True)

    # Match training feature order
    features = df[[
        'open', 'high', 'low', 'close',
        'body', 'range', 'upper_wick', 'lower_wick',
        'return', 'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]].values[-window_size:]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return np.expand_dims(features, axis=0), df['close'].iloc[-1]

# ---------- Visualization Setup ----------

model = load_model("eth_lstm_model.h5", compile=False)
prediction_history = []
price_history = []

fig, (ax_price, ax_conf) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("Live ETH Model Forecast", fontsize=16)

line_price, = ax_price.plot([], [], label="ETH Price ($)")
bar = ax_conf.bar(["Confidence"], [0.0], color='blue')
text_pred = ax_conf.text(0.5, 0.85, '', ha='center', va='center', transform=ax_conf.transAxes, fontsize=14)

ax_price.set_ylabel("Price")
ax_conf.set_ylim(0, 1)
ax_conf.set_ylabel("Confidence")
ax_conf.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

def update(frame):
    df = fetch_latest_data()
    if df.empty:
        return line_price, bar, text_pred 
    X_input, current_price = build_features(df)
    prob = model.predict(X_input)[0][0]

    # Store history
    timestamp = datetime.utcnow()
    price_history.append((timestamp, current_price))
    prediction_history.append((timestamp, prob))

    # Trim to last 48 hours for visual clarity
    if len(price_history) > 48:
        price_history.pop(0)
        prediction_history.pop(0)

    # Update price plot
    if len(price_history) > 1:
            times, prices = zip(*price_history)
            line_price.set_data(times, prices)
            ax_price.set_xlim(min(times), max(times))
            ax_price.set_ylim(min(prices) * 0.99, max(prices) * 1.01)
    else:
        line_price.set_data([], [])
    ax_price.legend()

    # Update confidence bar
    bar[0].set_height(prob)
    bar[0].set_color("green" if prob > 0.5 else "red")

    direction = "Uptrend" if prob > 0.6 else "Downtrend" if prob < 0.4 else "Uncertain"
    text_pred.set_text(f"{direction} ({prob:.2%})")
    ax_price.set_title(f"Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    return line_price, bar, text_pred

ani = animation.FuncAnimation(fig, update, interval=3 * 1000, blit=False)  # every 1 minute
plt.tight_layout()
plt.show()
