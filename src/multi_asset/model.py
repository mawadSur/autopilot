"""Pooled multi-asset LSTM.

Wraps the project's existing :class:`models.LSTMClassifier` (the attention-LSTM —
the "same LSTM model" used elsewhere) with a learned **asset embedding**. The
embedding is concatenated to every timestep's feature vector, so asset identity
reaches the network through a dedicated, learnable channel instead of leaking
through feature magnitude. One network serves the whole universe.

    x: [B, T, n_features]   asset_idx: [B]
    emb = Embedding(asset_idx)            -> [B, embed_dim]
    x'  = concat(x, emb broadcast over T) -> [B, T, n_features + embed_dim]
    logits = LSTMClassifier(x')           -> [B, num_classes]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn as nn

from models import LSTMClassifier


class PooledLSTMClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_assets: int,
        *,
        embed_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.config: Dict[str, Any] = {
            "n_features": int(n_features),
            "n_assets": int(n_assets),
            "embed_dim": int(embed_dim),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "bidirectional": bool(bidirectional),
            "num_classes": int(num_classes),
        }
        self.asset_embedding = nn.Embedding(n_assets, embed_dim)
        self.backbone = LSTMClassifier(
            input_size=n_features + embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor, asset_idx: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected x of shape [B, T, F], got {tuple(x.shape)}")
        b, t, _ = x.shape
        emb = self.asset_embedding(asset_idx)        # [B, embed_dim]
        emb = emb.unsqueeze(1).expand(b, t, emb.shape[-1])  # [B, T, embed_dim]
        x = torch.cat([x, emb], dim=-1)              # [B, T, F + embed_dim]
        return self.backbone(x)                      # [B, num_classes]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PooledLSTMClassifier":
        return cls(
            n_features=int(config["n_features"]),
            n_assets=int(config["n_assets"]),
            embed_dim=int(config.get("embed_dim", 8)),
            hidden_size=int(config.get("hidden_size", 128)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            bidirectional=bool(config.get("bidirectional", False)),
            num_classes=int(config.get("num_classes", 2)),
        )
