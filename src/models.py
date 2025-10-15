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
import warnings
warnings.filterwarnings("ignore")

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
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        #self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        pooled = torch.mean(x, dim=1)  # [B, hidden_size]
        pooled = self.norm(pooled)
        return self.head(pooled)  # [B]

        # last_step = x[:, -1, :]
        # last_step = self.norm(last_step)
        # return self.head(last_step)


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


class LSTMAttentionRegressor(nn.Module):
    """LSTM regressor with additive attention for time-series regression."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 1,  # Fixed to 1 for regression
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

class LSTMRegressor(nn.Module):
    """Simple LSTM regressor without attention for time-series regression."""
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        num_classes: int = 1,  # Fixed to 1 for regression
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
        self.dropout = nn.Dropout(p=0.2)
        direction_factor = 2 if bidirectional else 1
        embed_dim = hidden_size * direction_factor
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        output, _ = self.lstm(x)  # [B, T, H*D]
        output = self.dropout(output)
        pooled = torch.mean(output, dim=1)  # Mean pooling over timesteps [B, H*D]
        pooled = self.norm(pooled)
        output = self.head(pooled)  # [B, 1]
        return output


# --------------------------
# Performer-style linear attention block (approx FAVOR+)
# reference: https://arxiv.org/abs/2009.14794
# --------------------------
class RandomFeatureMap(nn.Module):
    """
    Random feature map for FAVOR+/Performer-style linear attention.
    We implement a stable, commonly-used map: phi(x) = elu(x) + 1.
    This is the practical variant used in many linear-attention implementations.
    (Exact FAVOR+ uses orthogonal random features and exponentials; to keep code
    stable and simple we use elu+1.)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x: (..., dim)
        return torch.nn.functional.elu(x) + 1.0 + self.eps


class LinearAttention(nn.Module):
    """
    Linear attention using feature maps: attention(Q, K, V) ≈ phi(Q) (phi(K)^T V) / normalization
    Implementation works with batch-first tensors.
    """

    def __init__(self, dim, feature_map=None, causal: bool = False):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map or RandomFeatureMap()
        self.causal = causal

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None):
        # q, k, v: (B, T, D)
        # Compute phi(q), phi(k)
        phi_q = self.feature_map(q)        # (B, T, D)
        phi_k = self.feature_map(k)        # (B, T, D)

        # If a mask is provided, zero out masked positions in phi_k and v
        if mask is not None:
            # mask: (B, T) or (B, 1, T)
            mask_ = mask.unsqueeze(-1).to(q.dtype)  # (B, T, 1)
            phi_k = phi_k * mask_
            v = v * mask_

        # Compute K^T V: (B, D, D_v)  where D_v = v.size(-1)
        kv = torch.einsum('btd, bte -> bde', phi_k, v)  # (B, D, Vdim)

        # compute normalization (fixed)
        z = torch.einsum('btd, bd -> bt', phi_q, phi_k.sum(dim=1))
        z = z.unsqueeze(-1).clamp(min=1e-6)

        out = torch.einsum('btd, bde -> bte', phi_q, kv)
        out = out / z
        return out


class PerformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, head_dim=32, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.total_head_dim = n_heads * head_dim
        assert self.total_head_dim == dim, "dim must equal n_heads * head_dim"

        # projectors
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn = LinearAttention(dim=head_dim)  # will be applied per-head
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def _split_heads(self, x):
        # x: (B, T, dim) -> (B, n_heads, T, head_dim)
        B, T, D = x.shape
        x = x.view(B, T, self.n_heads, self.head_dim).permute(0,2,1,3)
        return x

    def _merge_heads(self, x):
        # x: (B, n_heads, T, head_dim) -> (B, T, dim)
        x = x.permute(0,2,1,3).contiguous()
        B, T, _, _ = x.shape
        return x.view(B, T, self.total_head_dim)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (B, T, dim)
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = self._split_heads(q)  # (B, H, T, head_dim)
        kh = self._split_heads(k)
        vh = self._split_heads(v)

        # compute per-head linear attention
        heads_out = []
        for hi in range(self.n_heads):
            qi = qh[:, hi, :, :]  # (B, T, head_dim)
            ki = kh[:, hi, :, :]
            vi = vh[:, hi, :, :]
            out_hi = self.attn(qi, ki, vi, mask=mask)  # (B, T, head_dim)
            heads_out.append(out_hi)

        heads_out = torch.stack(heads_out, dim=1)  # (B, H, T, head_dim)
        out = self._merge_heads(heads_out)  # (B, T, dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Residual + FFN
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x

# --------------------------
# Full model: Performer -> BiLSTM x2 -> FC
# --------------------------
class PerformerBiLSTM(nn.Module):
    def __init__(self, input_dim, model_dim=128, n_heads=8, head_dim=16,
                 lstm_hidden=128, lstm_layers=1, dropout=0.0, out_dim=1):
        super().__init__()
        print(n_heads, head_dim, model_dim)
        assert n_heads * head_dim == model_dim, "n_heads * head_dim must equal model_dim"

        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_dropout = nn.Dropout(dropout)

        self.performer = PerformerBlock(dim=model_dim, n_heads=n_heads, head_dim=head_dim, dropout=dropout)

        # two BiLSTM layers
        self.bilstm1 = nn.LSTM(input_size=model_dim, hidden_size=lstm_hidden,
                               num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=2 * lstm_hidden, hidden_size=lstm_hidden,
                               num_layers=1, batch_first=True, bidirectional=True)

        # FC head
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (B, T, input_dim)
        x = self.input_linear(x)  # (B, T, model_dim)
        x = self.pos_dropout(x)

        x = self.performer(x, mask=mask)  # (B, T, model_dim)

        # BiLSTM layers: return sequences
        out1, _ = self.bilstm1(x)  # (B, T, 2*hidden)
        out2, _ = self.bilstm2(out1)  # (B, T, 2*hidden)

        # Predict one-step-ahead return / price for last timestep of sequence:
        # select last time-step representation
        last = out2[:, -1, :]  # (B, 2*hidden)
        y = self.fc(last)  # (B, out_dim)
        return y


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
    if model_type in {"lstm_regressor", "lstm_attention_regressor"}:
        return LSTMRegressor(
            input_size=meta.input_size,
            hidden_size=meta.hidden_size,
            num_layers=meta.num_layers,
            dropout=meta.dropout,
            bidirectional=meta.bidirectional,
            num_classes=meta.num_classes,
        )
    if model_type in {"performer_bilstm", "performer"}:
        return PerformerBiLSTM(
            input_dim=meta.input_size,
            model_dim=meta.hidden_size,
            n_heads=8,
            head_dim=getattr(meta, "head_dim", 16) or 16,
            lstm_hidden=meta.hidden_size,
            lstm_layers=meta.num_layers,
            dropout=meta.dropout,
            out_dim=meta.num_classes,
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
    state = torch.load(state_path, map_location="cpu", weights_only=False)
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
