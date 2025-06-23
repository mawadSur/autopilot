
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import compute_rsi


def fetch_latest_data(symbol="ETHUSDT", interval="1m", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
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
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df.dropna(inplace=True)

    features = df[[
        'open', 'high', 'low', 'close',
        'body', 'range', 'upper_wick', 'lower_wick',
        'return', 'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change'
    ]].values[-window_size:]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return np.expand_dims(features, axis=0), df['close'].iloc[-1]

model = load_model("eth_lstm_model.h5", compile=False)
prediction_history = []
price_history = []

fig, (ax_price, ax_conf) = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle("Live ETH Model Forecast", fontsize=16)

line_price, = ax_price.plot([], [], label="ETH Price ($)")
bar = ax_conf.bar(["Confidence"], [0.0], color='blue')
text_pred = ax_conf.text(0.5, 0.85, '', ha='center', va='center', transform=ax_conf.transAxes, fontsize=14)
text_sell = ax_conf.text(0.5, 0.15, '', ha='center', va='center', transform=ax_conf.transAxes, fontsize=12, color='gray')

ax_price.set_ylabel("Price")
ax_conf.set_ylim(0, 1)
ax_conf.set_ylabel("Confidence")
ax_conf.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

def update(frame):
    df = fetch_latest_data()
    if df.empty:
        return line_price, bar, text_pred, text_sell
    X_input, current_price = build_features(df)
    prob = model.predict(X_input)[0][0]

    timestamp = datetime.utcnow()
    price_history.append((timestamp, current_price))
    prediction_history.append((timestamp, prob))
    if len(price_history) > 300:
        price_history.pop(0)
        prediction_history.pop(0)

    if len(price_history) > 1:
        times, prices = zip(*price_history)
        line_price.set_data(times, prices)
        ax_price.set_xlim(min(times), max(times))
        ax_price.set_ylim(min(prices) * 0.99, max(prices) * 1.01)
    else:
        line_price.set_data([], [])
    ax_price.legend()

    bar[0].set_height(prob)
    bar[0].set_color("green" if prob > 0.5 else "red")

    direction = "Uptrend" if prob > 0.6 else "Downtrend" if prob < 0.4 else "Uncertain"
    text_pred.set_text(f"{direction} ({prob:.2%})")

    # Estimate time to sell
    if prob > 0.7:
        # Confidence strongly up → estimate 30 mins hold
        eta = timestamp + timedelta(minutes=30)
        text_sell.set_text(f"💡 Est. Sell at: {eta.strftime('%H:%M:%S')}")
    elif prob < 0.3:
        eta = timestamp + timedelta(minutes=10)
        text_sell.set_text(f"⚠️ Sell likely before: {eta.strftime('%H:%M:%S')}")
    else:
        text_sell.set_text("")

    ax_price.set_title(f"Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    return line_price, bar, text_pred, text_sell

ani = animation.FuncAnimation(fig, update, interval=1000, blit=False)  # every second
plt.tight_layout()
plt.show()
