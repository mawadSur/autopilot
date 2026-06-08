"""Turn the pooled table into LSTM sequence tensors — leakage-safe.

A sequence is ``window`` consecutive bars of one asset; its label is the label
of the *last* bar in the window. Two leakage guards:

1. **No cross-asset windows** — sequences are built per ``asset_id`` group, so a
   window never mixes BTC bars with AAPL bars.
2. **No data-hole windows** — a window is rejected if any step inside it is a
   jump far larger than that asset's own typical bar spacing (``gap_multiplier``
   × the per-asset median gap). This catches missing-data holes (e.g. an outage
   leaving a chunk of 1m bars absent) WITHOUT shattering daily equity series at
   every weekend/holiday — a Fri→Mon gap is ~3× the median, well under the 5×
   default, whereas a multi-day hole in a minute series is thousands× the median.

To keep train/val/test strictly non-overlapping, the trainer calls this once per
split *subframe* — a window therefore lives entirely inside one split and can't
peek across a boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class SequenceBatch:
    X: np.ndarray          # [N, window, n_features] float32
    y: np.ndarray          # [N] int
    asset_idx: np.ndarray  # [N] int  (index into the asset vocab)
    end_ts: np.ndarray     # [N] object (ISO timestamp of the window's last bar)

    def __len__(self) -> int:
        return int(self.X.shape[0])


def _to_epoch_seconds(ts: pd.Series) -> np.ndarray:
    # .asi8 -> int64 nanoseconds since epoch; stable for tz-aware across pandas 1.x/2.x.
    return pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).asi8 / 1e9


def build_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: List[str],
    asset_to_idx: Dict[str, int],
    window: int,
    gap_multiplier: float = 5.0,
    dtype=np.float32,
) -> SequenceBatch:
    """Build sliding windows per asset. See module docstring for the guards."""
    n_features = len(feature_cols)
    if window < 1:
        raise ValueError("window must be >= 1")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    aidx_list: List[int] = []
    ts_list: List[str] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        if asset_id not in asset_to_idx:
            continue
        g = g.assign(_k=pd.to_datetime(g["timestamp"], utc=True)).sort_values("_k")
        n = len(g)
        if n < window:
            continue

        feats = g[feature_cols].to_numpy(dtype=dtype)
        labels = g["label"].to_numpy()
        ts_str = g["timestamp"].astype(str).to_numpy()
        secs = _to_epoch_seconds(g["timestamp"])
        diffs = np.diff(secs)  # length n-1, gap between bar i and i+1

        positive = diffs[diffs > 0]
        med = float(np.median(positive)) if positive.size else 0.0
        max_gap = med * gap_multiplier if med > 0 else np.inf
        idx = asset_to_idx[asset_id]

        for end in range(window - 1, n):
            start = end - window + 1
            if window > 1 and med > 0:
                # diffs covering the W-1 transitions inside this window.
                if np.any(diffs[start:end] > max_gap):
                    continue
            X_list.append(feats[start : end + 1])
            y_list.append(int(labels[end]))
            aidx_list.append(idx)
            ts_list.append(str(ts_str[end]))

    if not X_list:
        return SequenceBatch(
            X=np.empty((0, window, n_features), dtype=dtype),
            y=np.empty((0,), dtype=int),
            asset_idx=np.empty((0,), dtype=int),
            end_ts=np.empty((0,), dtype=object),
        )

    return SequenceBatch(
        X=np.stack(X_list).astype(dtype),
        y=np.asarray(y_list, dtype=int),
        asset_idx=np.asarray(aidx_list, dtype=int),
        end_ts=np.asarray(ts_list, dtype=object),
    )


def build_asset_vocab(df: pd.DataFrame) -> Dict[str, int]:
    """Stable, sorted ``{asset_id: index}`` map for the embedding table."""
    return {aid: i for i, aid in enumerate(sorted(df["asset_id"].unique()))}
