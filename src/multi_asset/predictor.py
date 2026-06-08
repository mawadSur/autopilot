"""Inference wrapper for the pooled multi-asset LSTM.

Loads a model dir written by ``train_pooled_lstm`` (``model.pt`` + ``scaler.joblib``
+ ``meta.json``) and turns recent OHLCV for any *trained* asset into a
``(side, confidence)`` decision. Mirrors the role of ``predictor.XGBoostPredictor``
for the existing crypto stack.

SHADOW only — returns a signal, never places an order.

Usage::

    pred = PooledLSTMPredictor("model_multi/pooled_1d_v1/")
    res = pred.predict("crypto:BTC-USD", recent_ohlcv_df)   # >= window+warmup bars
    # res.side in {"long", "flat"}; res.confidence == P(profitable)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch

from models import load_scaler
from multi_asset.model import PooledLSTMClassifier
from multi_asset.sources import OHLCV_COLUMNS
from utils import compute_features


@dataclass
class Prediction:
    asset_id: str
    side: str          # "long" | "flat"
    confidence: float  # P(class==1) == P(forward move clears k*ATR%)
    threshold: Optional[float]
    used_bars: int


class PooledLSTMPredictor:
    def __init__(self, model_dir: str, *, device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("model_type") != "pooled_lstm":
            raise ValueError(f"{model_dir} is not a pooled_lstm model (got {meta.get('model_type')!r})")
        self.meta = meta
        self.feature_cols: List[str] = list(meta["feature_cols"])
        self.asset_vocab = dict(meta["asset_vocab"])
        self.window = int(meta["window"])
        self.threshold: Optional[float] = meta.get("optimal_threshold")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = PooledLSTMClassifier.from_config(meta["config"]).to(self.device)
        state = torch.load(self.model_dir / "model.pt", map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()

        self.scaler = load_scaler(str(self.model_dir / meta.get("scaler_path", "scaler.joblib")))

    def known_assets(self) -> List[str]:
        return sorted(self.asset_vocab)

    def _features_window(self, ohlcv) -> np.ndarray:
        clean = ohlcv[OHLCV_COLUMNS].copy().reset_index(drop=True)
        feats = compute_features(clean)[self.feature_cols]
        if len(feats) < self.window:
            raise ValueError(
                f"Need >= window ({self.window}) feature rows after compute_features, "
                f"got {len(feats)}. Pass more history."
            )
        return feats.iloc[-self.window:].to_numpy(dtype=np.float32)

    @torch.no_grad()
    def predict(self, asset_id: str, ohlcv) -> Prediction:
        if asset_id not in self.asset_vocab:
            raise KeyError(
                f"asset_id {asset_id!r} is not in the trained vocab {self.known_assets()}. "
                "A pooled model can only score assets it was trained on."
            )
        window = self._features_window(ohlcv)                 # [W, F]
        if self.scaler is not None:
            window = self.scaler.transform(window).astype(np.float32)
        x = torch.from_numpy(window).unsqueeze(0).to(self.device)  # [1, W, F]
        a = torch.tensor([self.asset_vocab[asset_id]], dtype=torch.long, device=self.device)
        prob1 = float(torch.softmax(self.model(x, a), dim=-1)[0, 1].item())
        thr = self.threshold if self.threshold is not None else 0.5
        side = "long" if prob1 >= thr else "flat"
        return Prediction(
            asset_id=asset_id, side=side, confidence=prob1,
            threshold=self.threshold, used_bars=self.window,
        )
