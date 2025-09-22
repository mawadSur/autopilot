"""
Unified LSTMClassifier definition and helpers used across training, backtesting, and inference.

Import this everywhere instead of redefining the model:
    from models import LSTMClassifier, build_model_from_meta, load_scaler

The design is intentionally minimal and framework-agnostic, but robust enough for
your training/backtest/inference use cases.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union, List, Tuple


import torch
import torch.nn as nn

try:
    # Prefer joblib (faster, more common for sklearn objects)
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # Fallback handled in load_scaler()


# ----------------------------
# Model definition
# ----------------------------

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over an input sequence.

    Given H of shape [B, T, E], computes attention weights:
        e_t = v^T tanh(W_h h_t)
        a_t = softmax(e_t)
    and returns the context vector c = sum_t a_t * h_t with shape [B, E].
    """

    def __init__(self, input_dim: int, attn_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.attn_dim = attn_dim
        self.W_h = nn.Linear(input_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H: torch.Tensor):
        # H: [B, T, E]
        scores = self.v(torch.tanh(self.W_h(H)))  # [B, T, 1]
        weights = torch.softmax(scores.squeeze(-1), dim=-1)  # [B, T]
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # [B, E]
        return context, weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len]
        x = x + pe_slice.to(x.dtype)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier that consumes [B, T, F] windows."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_classes: int,
        *,
        feedforward_dim: Optional[int] = None,
        activation: str = "gelu",
        max_len: int = 10000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "num_classes": num_classes,
            "feedforward_dim": feedforward_dim,
            "activation": activation,
            "max_len": max_len,
        }

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout, max_len=max_len)
        ff_dim = feedforward_dim if feedforward_dim is not None else hidden_size * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        last_step = x[:, -1, :]
        last_step = self.norm(last_step)
        return self.head(last_step)


class Attention(nn.Module):

    """Simple additive attention over temporal features."""



    def __init__(self, input_dim: int, attn_dim: Optional[int] = None):

        super().__init__()

        attn_dim = attn_dim or input_dim

        self.energy = nn.Linear(input_dim, attn_dim, bias=True)

        self.score = nn.Linear(attn_dim, 1, bias=False)



    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        """Return context vector and attention weights for inputs shaped [B, T, E]."""

        energy = torch.tanh(self.energy(H))

        scores = self.score(energy).squeeze(-1)  # [B, T]

        weights = torch.softmax(scores, dim=-1)

        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)  # [B, E]

        return context, weights





class LSTMAttentionClassifier(nn.Module):

    """LSTM classifier that applies additive attention over sequence outputs."""



    def __init__(

        self,

        input_size: int,

        hidden_size: int = 128,

        num_layers: int = 2,

        dropout: float = 0.1,

        bidirectional: bool = False,

        num_classes: int = 3,

    ):

        super().__init__()

        self.save_hyperparameters = {

            "input_size": input_size,

            "hidden_size": hidden_size,

            "num_layers": num_layers,

            "dropout": dropout,

            "bidirectional": bidirectional,

            "num_classes": num_classes,

        }



        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            dropout=dropout if num_layers > 1 else 0.0,

            bidirectional=bidirectional,

            batch_first=True,

        )



        direction_factor = 2 if bidirectional else 1

        embed_dim = hidden_size * direction_factor

        self.attn = Attention(input_dim=embed_dim, attn_dim=hidden_size)



        self.head = nn.Sequential(

            nn.LayerNorm(embed_dim),

            nn.Linear(embed_dim, hidden_size),

            nn.ReLU(),

            nn.Dropout(p=dropout),

            nn.Linear(hidden_size, num_classes),

        )

        self._attn_weights: Optional[torch.Tensor] = None



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: [B, T, F]

        output, _ = self.lstm(x)         # [B, T, H*D]

        context, weights = self.attn(output)  # [B, H*D], [B, T]

        self._attn_weights = weights

        logits = self.head(context)      # [B, C]

        return logits





class LSTMClassifier(LSTMAttentionClassifier):

    """Backward-compatible alias for the attention-based LSTM classifier."""



    def __init__(

        self,

        input_size: int,

        hidden_size: int = 128,

        num_layers: int = 2,

        dropout: float = 0.1,

        bidirectional: bool = False,

        num_classes: int = 3,

    ):

        super().__init__(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            dropout=dropout,

            bidirectional=bidirectional,

            num_classes=num_classes,

        )



# ----------------------------
# Metadata utilities
# ----------------------------

@dataclass
class ModelMeta:
    # Architecture
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    bidirectional: bool = False
    num_classes: int = 3
    task: str = "classification"

    # Artifacts (relative to model dir unless absolute)
    model_state_path: str = "model.pt"
    scaler_path: Optional[str] = "scaler.joblib"

    # Optional flags
    feature_scaling: bool = True  # If false, scaler_path may be None
    framework: str = "pytorch"
    model_type: str = "lstm_classifier"

    # Data/Features (optional but commonly present in meta JSONs)
    feature_cols: List[str] = None  # type: ignore[assignment]
    price_col: str = "close"
    window_size: int = 150
    buy_threshold: float = 0.60
    tx_cost: float = 0.0008

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelMeta":
        # Accept flexible keys and sensible fallbacks
        return ModelMeta(
            input_size=int(d.get("input_size") or d.get("n_features") or d["features"]),
            hidden_size=int(d.get("hidden_size", 128)),
            num_layers=int(d.get("num_layers", 2)),
            num_heads=int(d.get("num_heads", d.get("transformer_heads", 4))),
            dropout=float(d.get("dropout", 0.1)),
            bidirectional=bool(d.get("bidirectional", False)),
            num_classes=int(d.get("num_classes", d.get("classes", 3))),
            task=str(d.get("task", "classification")),
            model_state_path=str(d.get("model_state_path", d.get("weights", "model.pt"))),
            scaler_path=d.get("scaler_path", d.get("scaler", "scaler.joblib")),
            feature_scaling=bool(d.get("feature_scaling", d.get("scale_features", True))),
            framework=str(d.get("framework", "pytorch")),
            model_type=str(d.get("model_type", "lstm_classifier")),
            feature_cols=list(d.get("feature_cols", []) or []),
            price_col=str(d.get("price_col", "close")),
            window_size=int(d.get("window_size", 150)),
            buy_threshold=float(d.get("buy_threshold", 0.60)),
            tx_cost=float(d.get("tx_cost", 0.0008)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_model_from_meta(meta: Union[ModelMeta, Dict[str, Any]]) -> nn.Module:
    """
    Build a model instance from metadata dict or ModelMeta object.

    Supports LSTM and transformer classifiers.
    """
    if not isinstance(meta, ModelMeta):
        meta = ModelMeta.from_dict(meta)

    model_type = str(getattr(meta, "model_type", "lstm_classifier")).lower()
    if model_type in {"transformer", "transformer_classifier"}:
        num_heads = getattr(meta, "num_heads", 4) or 4
        return TransformerClassifier(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            num_heads=int(num_heads),
            dropout=meta.dropout,
            num_classes=meta.num_classes,
        )
    if model_type in {"lstm_attention", "lstm_attention_classifier"}:
        return LSTMAttentionClassifier(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            dropout=meta.dropout,
            bidirectional=meta.bidirectional,
            num_classes=meta.num_classes,
        )

    return LSTMClassifier(
        input_size=meta.input_size,
        hidden_size=meta.hidden_size,
        num_layers=meta.num_layers,
        dropout=meta.dropout,
        bidirectional=meta.bidirectional,
        num_classes=meta.num_classes,
    )


# ----------------------------
# Artifact loading helpers
# ----------------------------

def resolve_path(base_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    import os
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def load_meta(meta_path: str) -> ModelMeta:
    with open(meta_path, "r") as f:
        data = json.load(f)
    return ModelMeta.from_dict(data)


def load_model_state(model: nn.Module, state_path: str, strict: bool = False) -> None:
    state = torch.load(state_path, map_location="cpu")
    # Allow non-strict to tolerate minor key mismatches or buffer names
    model.load_state_dict(state, strict=strict)


def load_scaler(scaler_path: Optional[str]):
    """
    Loads an sklearn-like scaler (StandardScaler/MinMax/etc) saved via joblib or pickle.
    Returns None if path is None. Users should check for None before applying.
    """
    if scaler_path is None:
        return None

    import os
    import pickle

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

    # Try joblib first
    if joblib is not None:
        try:
            return joblib.load(scaler_path)
        except Exception:
            pass

    # Fallback to pickle
    with open(scaler_path, "rb") as f:
        return pickle.load(f)
