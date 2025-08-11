# utils.py
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------
# Small env helpers
# ---------------------------------------------------------------------
def _get_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# ---------------------------------------------------------------------
# Model meta loader (optional)
# ---------------------------------------------------------------------
@dataclass
class ModelMeta:
    window_size: int = 150
    threshold: float = 0.60
    features: List[str] = None  # e.g., ["open","high","low","close","volume"]

def _load_model_meta(meta_path: Path = Path("model_meta.json")) -> ModelMeta:
    meta = ModelMeta(
        window_size=_get_int("WINDOW_SIZE", _get_int("HISTORY_SIZE", 150)),
        threshold=_get_float("CONF_THRESHOLD", 0.60),
        features=None,
    )
    if meta_path.exists():
        try:
            with meta_path.open("r") as f:
                j = json.load(f)
            if isinstance(j, dict):
                meta.window_size = int(j.get("window_size", meta.window_size))
                meta.threshold = float(j.get("threshold", meta.threshold))
                feats = j.get("features")
                if isinstance(feats, list) and feats:
                    meta.features = [str(x) for x in feats]
        except Exception:
            # ignore malformed meta; keep defaults
            pass

    # env override for features (comma-separated)
    env_feats = os.getenv("FEATURES")
    if env_feats:
        meta.features = [s.strip() for s in env_feats.split(",") if s.strip()]

    # final default
    if not meta.features:
        meta.features = ["open", "high", "low", "close", "volume"]

    return meta

# ---------------------------------------------------------------------
# Binance helpers (used by paper_trade.py, run_live_check.py, trade.py)
# ---------------------------------------------------------------------
def get_binance_client(testnet: Optional[bool] = None):
    """
    Lazy-imports binance and returns a configured Client.
    Picks keys based on TESTNET flag (env or param).
    """
    from binance.client import Client  # lazy import

    if testnet is None:
        testnet = _get_bool("TESTNET", False)

    if testnet:
        key = os.getenv("BINANCE_TESTNET_KEY")
        sec = os.getenv("BINANCE_TESTNET_SECRET")
    else:
        key = os.getenv("BINANCE_KEY") or os.getenv("BINANCE_TESTNET_KEY")
        sec = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE_TESTNET_SECRET")

    if not key or not sec:
        raise RuntimeError("Missing Binance API credentials in env")

    # Add a small timeout to avoid hanging calls
    return Client(api_key=key, api_secret=sec, testnet=testnet, requests_params={"timeout": 30})

def kline_to_row(k: Iterable[Any]) -> Dict[str, Any]:
    """
    Convert a kline array to the row format expected by SignalGenerator.
    k: [openTime, open, high, low, close, volume, ...]
    """
    k = list(k)
    return {
        "date": int(k[0]),
        "open": float(k[1]),
        "high": float(k[2]),
        "low": float(k[3]),
        "close": float(k[4]),
        "volume": float(k[5]),
    }

def prefill_history(signal_gen: "SignalGenerator", client, symbol: str, interval: str) -> None:
    """
    Warm up the history buffer with the last N bars (N = window_size).
    """
    limit = min(signal_gen.history_size, 1000)
    kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    for k in kl:
        signal_gen.history.append(kline_to_row(k))

def fetch_latest_bar(client, symbol: str, interval: str) -> Tuple[Dict[str, Any], int]:
    """
    Fetch only the most recent bar and its open time.
    """
    kl = client.get_klines(symbol=symbol, interval=interval, limit=2)
    last = kline_to_row(kl[-1])
    return last, last["date"]

# ---------------------------------------------------------------------
# SageMaker Predictor (lazy)
# ---------------------------------------------------------------------
class _SMEndpoint:
    def __init__(self, endpoint_name: str):
        self.endpoint_name = endpoint_name
        self._predictor = None  # built on first use

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        try:
            import boto3  # lazy
            from botocore.config import Config
            import sagemaker
            from sagemaker.predictor import Predictor
            from sagemaker.serializers import JSONSerializer
            from sagemaker.deserializers import JSONDeserializer
        except Exception as e:
            raise RuntimeError(f"Import error for boto3/sagemaker: {e}")

        cfg = Config(read_timeout=180, connect_timeout=180, retries={"max_attempts": 0})
        smrt = boto3.client("sagemaker-runtime", config=cfg)
        sess = sagemaker.Session(sagemaker_runtime_client=smrt)

        self._predictor = Predictor(
            endpoint_name=self.endpoint_name,
            sagemaker_session=sess,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer(),
        )
        return self._predictor

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        pred = self._get_predictor()
        out = pred.predict(payload)
        # Normalize common shapes
        if isinstance(out, dict):
            return out
        # Some containers return {"body": "...json..."}
        try:
            text = out.get("body") if isinstance(out, dict) else out
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8")
            return json.loads(text)
        except Exception:
            return {"raw": out}

# ---------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------
class SignalGenerator:
    """
    Buffers OHLCV rows and queries a SageMaker endpoint for a signal.
    - Reads window_size/threshold/features from model_meta.json or env.
    - Exposes .history (deque of rows) for warm-up.
    - get_signal(row) -> {'signal':0/1, 'confidence':float|None, 'probability':float|None, 'meta':{...}}
    """

    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        window_size: Optional[int] = None,
        threshold: Optional[float] = None,
        features: Optional[List[str]] = None,
    ):
        meta = _load_model_meta()
        self.history_size: int = int(window_size or meta.window_size)
        self.threshold: float = float(threshold if threshold is not None else meta.threshold)
        self.features: List[str] = features or meta.features

        self.history: Deque[Dict[str, Any]] = deque(maxlen=self.history_size)
        self._endpoint_name = endpoint_name or os.getenv("ENDPOINT_NAME")
        self._sagemaker: Optional[_SMEndpoint] = None
        self._last_open_time: Optional[int] = None  # for de-duping if you want to feed live ticks

        if not self._endpoint_name:
            raise RuntimeError("ENDPOINT_NAME is not set (and no endpoint_name provided)")

    # ----------------- public API -----------------
    def get_signal(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append a new OHLCV row and, when we have enough history, call the endpoint.
        Row must contain keys in ['date','open','high','low','close','volume'].
        """
        self._append(row)

        if len(self.history) < self.history_size:
            return {"signal": 0, "confidence": None, "probability": None, "meta": {"reason": "warming_up"}}

        inputs = self._build_inputs()
        result = self._infer({"inputs": inputs})

        # normalize result shapes
        prob = None
        conf = None
        signal = 0

        if isinstance(result, dict):
            # common variants
            prob = result.get("probability") or result.get("prob") or result.get("proba")
            conf = result.get("confidence", prob)
            sig = result.get("signal")
            if sig is not None:
                try:
                    signal = int(sig)
                except Exception:
                    signal = 1 if float(sig) >= self.threshold else 0
            elif conf is not None:
                signal = 1 if float(conf) >= self.threshold else 0

        return {"signal": signal, "confidence": conf, "probability": prob, "meta": {"features": self.features}}

    # ----------------- internals ------------------
    def _append(self, row: Dict[str, Any]) -> None:
        # keep strictly increasing by open time
        ts = int(row["date"])
        if self._last_open_time is None or ts > self._last_open_time:
            self.history.append(row)
            self._last_open_time = ts
        else:
            # same candle update: replace tail (latest) to keep close/volume fresh
            if self.history:
                self.history.pop()
            self.history.append(row)
            self._last_open_time = ts

    def _build_inputs(self) -> List[List[float]]:
        """
        Turn history -> 2D feature matrix [window_size x feature_count].
        Defaults to raw OHLCV (order taken from self.features).
        If you later store a richer feature list in model_meta.json, this will honor it automatically.
        """
        # Quick map to speed up repeated lookups
        feat_keys = self.features
        out: List[List[float]] = []
        for r in list(self.history)[-self.history_size:]:
            out.append([float(r[k]) for k in feat_keys])
        return out

    def _infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._sagemaker is None:
            self._sagemaker = _SMEndpoint(self._endpoint_name)
        return self._sagemaker.predict(payload)
