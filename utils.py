import os
import time
import pandas as pd
import numpy as np
from collections import deque
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

def get_client_binance():
    """Initializes and returns a Binance client based on environment variables."""
    if os.getenv("TESTNET", "false").lower() == "true":
        print("--- Using Binance Testnet ---")
        api_key = os.getenv("BINANCE_TESTNET_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_SECRET")
        testnet = True
    else:
        print("--- Using Live Binance API ---")
        api_key = os.getenv("BINANCE_KEY")
        api_secret = os.getenv("BINANCE_SECRET")
        testnet = False
        
    if not api_key or not api_secret:
        raise ValueError("Binance API key and secret must be set in the .env file.")
        
    return Client(api_key, api_secret, testnet=testnet)

def compute_rsi(series, period=14):
    """Computes the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    """Computes the Average True Range (ATR)."""
    tr_df = pd.DataFrame(index=df.index)
    tr_df['h-l'] = df['high'] - df['low']
    tr_df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr_df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    return tr_df.max(axis=1).rolling(period).mean()

class SignalGenerator:
    """
    Manages live data, generates features, and gets predictions from a SageMaker endpoint.
    """
    def __init__(self, endpoint_name, window_size=150, threshold=0.8):
        print("⚙️  Initializing Signal Generator...")
        self.endpoint_name = endpoint_name
        self.window_size = window_size
        self.threshold = threshold
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
            'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
        ]
        
        # Configure a longer timeout for the SageMaker client
        config = Config(read_timeout=90, connect_timeout=90, retries={"max_attempts": 0})
        sm_runtime_client = boto3.client("sagemaker-runtime", config=config)
        sagemaker_session = sagemaker.Session(sagemaker_runtime_client=sm_runtime_client)
        
        self.predictor = Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # Use a deque for an efficient rolling data history
        self.history_size = self.window_size + 100 # Keep extra data for rolling calculations
        self.history = deque(maxlen=self.history_size)
        
        self._warmup_endpoint()
        print(f"✅  Signal Generator ready. Connected to endpoint: {self.endpoint_name}")

    def _warmup_endpoint(self, retries=3, delay_seconds=10):
        """Sends a dummy payload to the endpoint to prevent cold start delays."""
        print("🔥  Warming up the SageMaker endpoint...")
        # The endpoint expects unscaled features
        fake_input = np.random.rand(self.window_size, len(self.feature_cols)).tolist()
        payload = {'inputs': fake_input}
        
        for i in range(retries):
            try:
                self.predictor.predict(payload)
                print("✅  Endpoint is warm and responding.")
                return
            except ClientError as e:
                if "ModelError" in str(e) or "invocation timed out" in str(e).lower():
                    print(f"Attempt {i+1}/{retries}: Endpoint is still warming up. Retrying in {delay_seconds}s...")
                    time.sleep(delay_seconds)
                else:
                    raise e
        raise RuntimeError(f"Endpoint did not become ready after {retries} attempts.")

    def _engineer_features(self, df):
        """
        Generates features for a given DataFrame of historical data.
        This logic is an exact mirror of the feature engineering in the training script.
        """
        df_out = df.copy()
        df_out['date'] = pd.to_datetime(df_out['date'], unit='ms')
        df_out.set_index('date', inplace=True)

        # Basic candle features
        df_out['body'] = df_out['close'] - df_out['open']
        df_out['range'] = df_out['high'] - df_out['low']
        df_out['upper_wick'] = df_out['high'] - df_out[['close', 'open']].max(axis=1)
        df_out['lower_wick'] = df_out[['close', 'open']].min(axis=1) - df_out['low']
        df_out['return'] = df_out['close'].pct_change()
        
        # Moving Averages
        df_out['sma_10'] = df_out['close'].rolling(10).mean()
        df_out['sma_50'] = df_out['close'].rolling(50).mean()
        df_out['sma_ratio'] = df_out['sma_10'] / (df_out['sma_50'] + 1e-9) - 1
        
        # --- ✅ CRITICAL FIX: Added adjust=False to all .ewm() calls ---
        df_out['ema_20'] = df_out['close'].ewm(span=20, adjust=False).mean()
        df_out['macd'] = df_out['close'].ewm(span=12, adjust=False).mean() - df_out['close'].ewm(span=26, adjust=False).mean()

        # Momentum and Volatility
        df_out['rsi_14'] = compute_rsi(df_out['close'], 14)
        df_out['vol_change'] = df_out['volume'].pct_change()
        df_out['atr'] = compute_atr(df_out, period=14)

        # Trend Analysis
        if len(df_out) >= 60: # Need enough data for hourly grouping
            hourly_index = df_out.index.floor('h')
            df_hourly = df_out['close'].groupby(hourly_index).mean()
            hourly_ema = df_hourly.ewm(span=20, adjust=False).mean() # ✅ Fixed
            df_out['hourly_ema_20'] = hourly_ema.reindex(hourly_index, method='ffill').values
            df_out['price_vs_hourly_trend'] = (df_out['close'] - df_out['hourly_ema_20']) / (df_out['hourly_ema_20'] + 1e-9)
        else:
            df_out['price_vs_hourly_trend'] = 0

        # Bollinger Bands
        bb_mid = df_out['close'].rolling(20).mean()
        bb_std = df_out['close'].rolling(20).std()
        df_out['bb_width'] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / (bb_mid + 1e-9)
        
        df_out.replace([np.inf, -np.inf], 0, inplace=True)
        df_out.dropna(inplace=True)

        return df_out

    def get_signal(self, new_kline_data):
        """
        Takes new kline data, updates history, generates features, and returns a trading signal.
        """
        self.history.append(new_kline_data)
        if len(self.history) < self.history_size:
            return {"confidence": None, "signal": 0, "reason": "History buffer is not full."}

        # Create a DataFrame from the entire history for feature calculation
        df_history = pd.DataFrame(list(self.history))
        df_features = self._engineer_features(df_history)
        model_input_df = df_features.tail(self.window_size)

        if len(model_input_df) < self.window_size:
            return {"confidence": None, "signal": 0, "reason": "Not enough processed data for a full window."}

        payload = {'inputs': model_input_df[self.feature_cols].values.tolist()}
        result = self.predictor.predict(payload)

        confidence = result.get('probability')
        signal = 1 if confidence is not None and confidence > self.threshold else 0

        return {
            "confidence": round(float(confidence), 5) if confidence is not None else None,
            "signal": signal,
            "threshold": self.threshold
        }