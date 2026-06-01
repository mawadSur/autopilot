"""
Unified LSTMClassifier definition and helpers used across training, backtesting, and inference.

Import this everywhere instead of redefining the model:
    from models import LSTMClassifier, build_model_from_meta, load_scaler

The design is intentionally minimal and framework-agnostic, but robust enough for
your training/backtest/inference use cases.

Note on legacy architectures (Feb 2026):
- Removed PerformerBiLSTM and the regression LSTM variants after sustained underperformance and added maintenance cost.
- Maintained winners: TransformerClassifier and LSTMAttentionClassifier (and its alias LSTMClassifier for back-compat).
  build_model_from_meta will now reject the legacy model_type values and prompt retraining.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Union, List, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Prefer joblib (faster, more common for sklearn objects)
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # Fallback handled in load_scaler()
import warnings
warnings.filterwarnings("ignore")

PROFIT_MODEL_VERSION = "profit_v3"


def _coerce_optional_market_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Market:
    market_id: str
    title: str
    category: str
    implied_prob: float
    bid_price: float
    ask_price: float
    volume_24h: float
    price_history: Dict[str, float]
    open_interest: float
    resolution_date: Union[datetime, str]
    rules_text: str
    avg_volume_7d: Optional[float] = None
    volume_change_1h: Optional[float] = None
    category_avg_spread: Optional[float] = None
    # Raw Gamma ``clobTokenIds`` ([YES_token, NO_token] as a JSON-string or
    # list), retained so the read-only CLOB order-book reader can resolve the
    # two outcome tokens for intra-market arbitrage. ``None`` for non-binary or
    # not-yet-CLOB-listed markets. Consumed by
    # ``exchanges.polymarket_market_data.get_yes_no_best_asks``.
    clob_token_ids: Optional[Any] = None
    spread: float = field(init=False)
    days_to_resolution: float = field(init=False)

    def __post_init__(self) -> None:
        self.implied_prob = float(self.implied_prob)
        self.bid_price = float(self.bid_price)
        self.ask_price = float(self.ask_price)
        self.volume_24h = float(self.volume_24h)
        self.open_interest = float(self.open_interest)
        self.avg_volume_7d = _coerce_optional_market_float(self.avg_volume_7d)
        self.volume_change_1h = _coerce_optional_market_float(self.volume_change_1h)
        self.category_avg_spread = _coerce_optional_market_float(self.category_avg_spread)
        self.price_history = {
            window: float(self.price_history.get(window, 0.0))
            for window in ("1h", "6h", "24h")
        }
        if isinstance(self.resolution_date, str):
            normalized = self.resolution_date.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            self.resolution_date = datetime.fromisoformat(normalized)
        if self.resolution_date.tzinfo is None:
            self.resolution_date = self.resolution_date.replace(tzinfo=timezone.utc)
        self.refresh_derived_fields()

    def refresh_derived_fields(self, now: Optional[datetime] = None) -> None:
        self.spread = max(0.0, self.ask_price - self.bid_price)
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)
        resolution_time = self.resolution_date.astimezone(timezone.utc)
        delta = resolution_time - current_time
        self.days_to_resolution = max(0.0, delta.total_seconds() / 86400.0)


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


def _normalize_transformer_pooling(pooling: Optional[str], *, default: str = "cls") -> str:
    pooling = str(pooling or default).strip().lower()
    aliases = {
        "last_attention": "weighted_last",
        "last_step_attention": "weighted_last",
        "weighted_last_step": "weighted_last",
    }
    pooling = aliases.get(pooling, pooling)
    supported = {"cls", "weighted_last"}
    if pooling not in supported:
        raise ValueError(
            f"Unsupported transformer pooling '{pooling}'. Expected one of {sorted(supported)}."
        )
    return pooling


class WeightedLastStepAttention(nn.Module):
    """Recency-biased last-step attention over encoded transformer states."""

    def __init__(self, input_dim: int, *, init_decay: float = 0.35) -> None:
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.gate_proj = nn.Linear(input_dim * 2, input_dim, bias=True)
        self.recency_decay = nn.Parameter(torch.tensor(float(init_decay), dtype=torch.float32))
        self.last_attention_weights: Optional[torch.Tensor] = None

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if H.ndim != 3:
            raise ValueError(f"Expected [B, T, E] hidden states, got shape {tuple(H.shape)}")
        if H.size(1) == 0:
            raise ValueError("WeightedLastStepAttention requires at least one time step")

        last_hidden = H[:, -1:, :]
        query = self.query_proj(last_hidden)
        keys = self.key_proj(H)
        scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(H.size(-1))

        distances = torch.arange(H.size(1) - 1, -1, -1, device=H.device, dtype=H.dtype)
        decay = F.softplus(self.recency_decay).to(device=H.device, dtype=H.dtype)
        scores = scores - decay * distances.unsqueeze(0)

        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), H).squeeze(1)
        last_step = last_hidden.squeeze(1)
        gate = torch.sigmoid(self.gate_proj(torch.cat([last_step, context], dim=-1)))
        pooled = gate * last_step + (1.0 - gate) * context
        self.last_attention_weights = weights.detach()
        return pooled, weights


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
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        pooling = _normalize_transformer_pooling(pooling, default="cls")
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
            "pooling": pooling,
        }

        self.input_proj = nn.Linear(input_size, hidden_size)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        else:
            self.register_parameter("cls_token", None)
        self.last_step_pool = WeightedLastStepAttention(hidden_size) if pooling == "weighted_last" else None
        pos_max_len = max_len + 1 if pooling == "cls" else max_len
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout, max_len=pos_max_len)
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
        self.pooling = pooling
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.input_proj(x)
        if self.pooling == "cls":
            cls = self.cls_token.expand(x.size(0), -1, -1).to(dtype=x.dtype)
            x = torch.cat([cls, x], dim=1)
            x = self.pos_encoder(x)
            x = self.encoder(x)
            pooled = x[:, 0, :]  # learned [CLS] summary
        else:
            x = self.pos_encoder(x)
            x = self.encoder(x)
            pooled, _weights = self.last_step_pool(x)
        pooled = self.norm(pooled)
        return self.head(pooled)


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

        input_size: Optional[int] = None,

        hidden_size: int = 128,

        num_layers: int = 2,

        dropout: float = 0.1,

        bidirectional: bool = False,

        num_classes: int = 3,

    ):

        super().__init__()
        if input_size is None:
            raise ValueError("input_size is required for LSTMAttentionClassifier")
        input_size = int(input_size)

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
        self.dropout = nn.Dropout(p=0.2)
        direction_factor = 2 if bidirectional else 1
        embed_dim = hidden_size * direction_factor
        self.attn = Attention(input_dim=embed_dim, attn_dim=hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        self._attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        output, _ = self.lstm(x)  # [B, T, H*D]
        output = self.dropout(output)
        context, weights = self.attn(output)  # [B, H*D], [B, T]
        self._attn_weights = weights
        output = self.head(context)  # [B, 1]
        return output



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
    transformer_pooling: str = "weighted_last"

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
    profitability_score: float = 0.0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelMeta":
        # Accept flexible keys and sensible fallbacks
        pooling = d.get("transformer_pooling", d.get("pooling", "cls"))
        return ModelMeta(
            input_size=int(d.get("input_size") or d.get("n_features") or d["features"]),
            hidden_size=int(d.get("hidden_size", 128)),
            num_layers=int(d.get("num_layers", 2)),
            num_heads=int(d.get("num_heads", d.get("transformer_heads", 4))),
            dropout=float(d.get("dropout", 0.1)),
            bidirectional=bool(d.get("bidirectional", False)),
            num_classes=int(d.get("num_classes", d.get("classes", 3))),
            task=str(d.get("task", "classification")),
            transformer_pooling=_normalize_transformer_pooling(pooling, default="cls"),
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
            profitability_score=float(d.get("profitability_score", 0.0)),
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
            pooling=str(getattr(meta, "transformer_pooling", "cls")),
        )
    if model_type in {"lstm_attention", "lstm_attention_classifier", "lstm_classifier"}:
        return LSTMAttentionClassifier(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            dropout=meta.dropout,
            bidirectional=meta.bidirectional,
            num_classes=meta.num_classes,
        )

    legacy = {"lstm_regressor", "lstm_attention_regressor", "performer_bilstm", "performer"}
    if model_type in legacy:
        raise ValueError(
            f"Model type '{model_type}' was deprecated after underperforming and maintenance burden. "
            "Please retrain using 'transformer' or 'lstm_attention' (the two maintained winners)."
        )

    # Fallback to maintained LSTM attention classifier for unknown types
    return LSTMAttentionClassifier(
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
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    # Allow non-strict to tolerate minor key mismatches or buffer names
    model.load_state_dict(state, strict=strict)


def load_checkpoint_state(state_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a checkpoint saved via torch.save({"state_dict": ..., "metadata": ...}).
    Enforces presence of profitability metadata and version.
    """
    raw = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "state_dict" not in raw:
        raise ValueError(
            "Checkpoint missing metadata. Retrain required — delete old checkpoint and retrain."
        )
    state_dict = raw.get("state_dict")
    metadata = raw.get("metadata")
    if not isinstance(state_dict, dict) or not isinstance(metadata, dict):
        raise ValueError(
            "Checkpoint metadata missing. Retrain required — delete old checkpoint and retrain."
        )
    version = str(metadata.get("version", "")).strip()
    if version != PROFIT_MODEL_VERSION:
        raise ValueError(
            f"Model version mismatch (checkpoint={version or 'unknown'}, expected={PROFIT_MODEL_VERSION}). "
            "Retrain required — delete old checkpoint and retrain."
        )
    return state_dict, metadata


def _copy_input_weights(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Copy overlapping input columns from src into dst (for input-size migrations)."""
    out = dst.clone()
    if src.ndim != 2 or dst.ndim != 2:
        return out
    cols = min(src.shape[1], dst.shape[1])
    out[:, :cols] = src[:, :cols]
    return out


def load_state_dict_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    Load state_dict with safe remapping for input-size changes.
    Only adapts the first-layer input projection (LSTM or Transformer).
    """
    target_state = model.state_dict()
    remapped: Dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        if key not in target_state:
            continue
        if tensor.shape == target_state[key].shape:
            remapped[key] = tensor
            continue
        if key.startswith("lstm.weight_ih_l"):
            remapped[key] = _copy_input_weights(tensor, target_state[key])
            continue
        if key == "input_proj.weight":
            remapped[key] = _copy_input_weights(tensor, target_state[key])
            continue

    return model.load_state_dict(remapped, strict=False)


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
