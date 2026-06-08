"""Multi-asset (crypto + equities) data + pooled-LSTM training pipeline.

A single LSTM is trained across many symbols and asset classes at once
("pooled"). The design problem with pooling BTC (~$60k) next to a $40 stock is
**scale leakage**: if the model can read raw price/volume magnitude it learns to
identify the asset instead of its direction. The pipeline defends against that
on three fronts:

1. ``build_pooled_dataset`` keeps only ``SCALE_INVARIANT_FEATURES`` (returns,
   ratios, z-scores, normalized vol) and labels each asset with a
   *vol-normalized* forward-return gate (k * ATR%), so "profitable" means the
   same thing on every asset regardless of price or volatility level.
2. Asset identity is supplied *deliberately* through a learned embedding in
   :class:`multi_asset.model.PooledLSTMClassifier` rather than leaking through
   feature magnitude.
3. Splits are by **global wall-clock time** (no look-ahead) and sequences never
   span an asset boundary or a data gap.

Everything here is research / SHADOW — no orders are ever placed.
"""

from __future__ import annotations

# Flat-import shim so the package's CLIs run without PYTHONPATH=src, matching
# the convention in src/crypto_training/*.
import sys as _sys
from pathlib import Path as _Path

_SRC_DIR = _Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))

__all__ = [
    "sources",
    "universe",
    "build_pooled_dataset",
    "sequences",
    "model",
    "train_pooled_lstm",
    "predictor",
]
