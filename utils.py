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
    print("[DEBUG] Getting Binance client...")
    if os.getenv("TESTNET"):
        print("[DEBUG] Using testnet")
        return Client(os.getenv("BINANCE_TESTNET_KEY"), os.getenv("BINANCE_TESTNET_SECRET"), testnet=True)
    else:
        print("[DEBUG] Using live")
        return Client(os.getenv("BINANCE_KEY"), os.getenv("BINANCE_SECRET"))

def load_ohlc_chunks(folder, chunk_mode=False):
    print(f"[DEBUG] Loading data from: {folder}")
    files = []
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith('.csv'):
                files.append(os.path.join(dirpath, f))

    if not files:
        raise FileNotFoundError(f"No .csv files found recursively in folder: {folder}")

    column_names = ['date', 'open', 'high', 'low', 'close', 'volume']
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    all_dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, header=None, names=column_names)
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            if not df.empty:
                if chunk_mode:
                    yield df
                else:
                    all_dfs.append(df)
        except Exception as e:
            print(f"[ERROR] Could not process file {f}: {e}")

    if not chunk_mode:
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs)

def compute_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    return df['tr'].rolling(period).mean()

class SignalGenerator:
    def __init__(self, endpoint_name, window_size=150, threshold=0.8):
        print("⚙️ Initializing Signal Generator for SageMaker Endpoint...")
        self.endpoint_name = endpoint_name
        self.window_size = window_size
        self.feature_cols = [
            'open', 'high', 'low', 'close', 'body', 'range', 'upper_wick', 'lower_wick', 'return',
            'sma_ratio', 'ema_20', 'macd', 'rsi_14', 'vol_change', 'atr', 'price_vs_hourly_trend', 'bb_width'
        ]
        config = Config(read_timeout=180, connect_timeout=180, retries={"max_attempts": 0})
        sm_runtime_client = boto3.client("sagemaker-runtime", config=config)
        sagemaker_session = sagemaker.Session(sagemaker_runtime_client=sm_runtime_client)
        self.predictor = Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        self.history_size = self.window_size + 100
        self.history = deque(maxlen=self.history_size)
        self.threshold = threshold # Use a configurable, validated threshold
        self._warmup_endpoint()
        print(f"✅ Signal Generator ready. Connected to endpoint: {self.endpoint_name}")

    def _warmup_endpoint(self, retries=3, delay_seconds=10):
        print("🔥 Warming up the SageMaker endpoint...")
        fake_input = np.zeros((self.window_size, len(self.feature_cols))).tolist()
        payload = {'inputs': fake_input}
        for i in range(retries):
            try:
                self.predictor.predict(payload)
                print("✅ Endpoint is warm and responding.")
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
        This function's logic is now synchronized with the preprocess_data
        function in aws_train_model.py.
        """
        df_out = df.copy()
        df_out['date'] = pd.to_datetime(df_out['date'], unit='ms') # Ensure datetime conversion if needed
        df_out.set_index('date', inplace=True)

        df_out['body'] = df_out['close'] - df_out['open']
        df_out['range'] = df_out['high'] - df_out['low']
        df_out['upper_wick'] = df_out['high'] - df_out[['close', 'open']].max(axis=1)
        df_out['lower_wick'] = df_out[['close', 'open']].min(axis=1) - df_out['low']
        df_out['return'] = df_out['close'].pct_change()
        df_out['sma_10'] = df_out['close'].rolling(10).mean()
        df_out['sma_50'] = df_out['close'].rolling(50).mean()
        df_out['sma_ratio'] = df_out['sma_10'] / (df_out['sma_50'] + 1e-9) - 1
        df_out['ema_20'] = df_out['close'].ewm(span=20).mean()
        df_out['macd'] = df_out['close'].ewm(span=12).mean() - df_out['close'].ewm(span=26).mean()
        df_out['rsi_14'] = compute_rsi(df_out['close'], 14)
        df_out['vol_change'] = df_out['volume'].pct_change()
        df_out['atr'] = compute_atr(df_out, period=14)

        # ✅ CORRECTED: Non-leaking hourly trend calculation
        if len(df_out) >= 60:
            hourly_index = df_out.index.floor('h')
            df_hourly = df_out['close'].groupby(hourly_index).mean()
            hourly_ema = df_hourly.ewm(span=20).mean()
            df_out['hourly_ema_20'] = hourly_ema.reindex(hourly_index, method='ffill').values
            df_out['price_vs_hourly_trend'] = (df_out['close'] - df_out['hourly_ema_20']) / (df_out['hourly_ema_20'] + 1e-9)
        else:
            df_out['price_vs_hourly_trend'] = 0

        df_out['bb_std'] = df_out['close'].rolling(20).std()
        df_out['bb_mid'] = df_out['close'].rolling(20).mean()
        df_out['bb_width'] = ((df_out['bb_mid'] + 2 * df_out['bb_std']) - (df_out['bb_mid'] - 2 * df_out['bb_std'])) / (df_out['bb_mid'] + 1e-9)
        
        df_out.replace([np.inf, -np.inf], 0, inplace=True)
        df_out.dropna(inplace=True)

        return df_out

    def get_signal(self, new_kline_data):
        self.history.append(new_kline_data)
        if len(self.history) < self.history_size:
            return {"confidence": None, "signal": 0, "reason": "History buffer is not full."}

        df_history = pd.DataFrame(list(self.history))
        df_features = self._engineer_features(df_history)
        model_input_df = df_features.tail(self.window_size)

        if len(model_input_df) < self.window_size:
            return {"confidence": None, "signal": 0, "reason": "Not enough data for a full window."}

        payload = {'inputs': model_input_df[self.feature_cols].values.tolist()}
        result = self.predictor.predict(payload)

        confidence = result.get('probability')
        # Use the class's threshold for a more robust signal
        signal = 1 if confidence is not None and confidence > self.threshold else 0

        return {
            "confidence": round(float(confidence), 5) if confidence is not None else None,
            "signal": signal,
            "threshold": self.threshold
        }