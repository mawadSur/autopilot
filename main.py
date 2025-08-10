# main.py
from __future__ import annotations
import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import load

from utils import (
    load_meta, FeatureSpec, build_features, make_model_window, proba_to_signal
)


# ------------------------- Model (must match training) -------------------------
class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        last = hn[-1]
        return self.head(last).squeeze(-1)  # logits


# ------------------------- Loaders -------------------------
def load_model_scaler_meta(
    model_path: str = None,
    scaler_path: str = None,
    meta_path: str = None,
):
    model_path = model_path or os.getenv("MODEL_PATH", "model.pt")
    scaler_path = scaler_path or os.getenv("SCALER_PATH", "scaler.joblib")
    meta_path   = meta_path or os.getenv("MODEL_META_PATH", "model_meta.json")

    meta = load_meta(meta_path)
    spec = FeatureSpec(feature_cols=meta["feature_cols"], window_size=int(meta["window_size"]))

    model = LSTMModel(
        input_size=len(spec.feature_cols),
        hidden_size=int(meta.get("hidden_size", 64)),
        num_layers=int(meta.get("num_layers", 2)),
        dropout=float(meta.get("dropout", 0.1)),
        bidirectional=bool(meta.get("bidirectional", False)),
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scaler = load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler, meta, spec


# ------------------------- Inference from OHLCV -------------------------
def predict_from_df(df: pd.DataFrame, model, scaler, meta: Dict[str, Any], spec: FeatureSpec) -> Dict[str, Any]:
    """
    df must contain columns: open, high, low, close, volume and a datetime index or 'timestamp' column.
    """
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    feat_df = build_features(df)

    X, current_price = make_model_window(feat_df, spec=spec, scaler=scaler)
    if X is None:
        return {"signal": "HOLD", "confidence": 0.0, "current_price": current_price}

    with torch.no_grad():
        logits = model(torch.from_numpy(X)).squeeze().float()
        prob = torch.sigmoid(logits).item()

    buy_th = float(meta.get("buy_threshold", 0.5))
    sell_th = float(meta.get("sell_threshold", 0.5))
    signal, confidence = proba_to_signal(prob, buy_th, sell_th)

    return {"signal": signal, "confidence": float(confidence), "current_price": float(current_price), "prob": float(prob)}


if __name__ == "__main__":
    # Example quickcheck:
    # df = pd.read_csv("ohlcv.csv")
    # model, scaler, meta, spec = load_model_scaler_meta()
    # print(predict_from_df(df, model, scaler, meta, spec))
    pass
