"""Adapter that turns the legacy ETH transformer into a supervisor predict_fn.

The supervisor calls ``model_predict_fn(symbol, ticker) -> (side, confidence)``
once per tick. The legacy transformer needs:

  * a rolling buffer of 1m OHLCV bars per symbol,
  * the 36 derived features computed by ``utils.compute_features``,
  * a ``window_size``-bar window scaled by the saved StandardScaler,
  * a 3-class softmax: ``[short, hold, long]``.

This adapter hides all of that. It seeds a per-symbol buffer from the Coinbase
candles endpoint on first call, refreshes it from the same endpoint on each
tick (cheap REST GET), and returns ``("buy"|"sell", probability_of_chosen_side)``.
The ``confidence`` is ``max(p_long, p_short)`` so the supervisor's
``min_confidence_to_trade`` gate behaves intuitively.

Anything fishy (load failure, insufficient warmup, NaN features) returns
``("buy", 0.5)`` -- below the default 0.6 trade threshold, so the loop is
safe-by-default rather than firing bad orders.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

_NEUTRAL_RESULT: Tuple[Literal["buy", "sell"], float] = ("buy", 0.5)


class LegacyTransformerPredictor:
    """Wraps ``model_sanity/`` artifacts as a supervisor-shaped callable."""

    def __init__(
        self,
        *,
        model_dir: str,
        exchange: Any,
        warmup_bars: int = 350,
        max_buffer_bars: int = 5000,
    ) -> None:
        # Lazy heavy imports so unit tests can stub them and so module import
        # in the supervisor doesn't pay torch's startup cost when the env flag
        # is off.
        import torch  # noqa: F401  (used in __call__)
        from utils import load_model_bundle, align_feature_columns

        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self.exchange = exchange
        self.warmup_bars = int(warmup_bars)
        self.max_buffer_bars = int(max_buffer_bars)

        model, scaler, meta = load_model_bundle(self.model_dir)
        self.model = model
        self.scaler = scaler
        self.meta = meta
        self.window_size = int(meta.get("window_size", 90))
        self.num_classes = int(meta.get("num_classes", 3))
        self.thr_long = float(meta.get("thr_long", 0.55))
        self.thr_short = float(meta.get("thr_short", 0.60))
        self.margin = float(meta.get("margin", meta.get("p_margin", 0.0)))
        feature_cols_raw = meta.get("feature_cols") or []
        if not feature_cols_raw:
            raise ValueError("model_meta.json missing feature_cols")
        expected_size = int(meta.get("input_size") or len(feature_cols_raw))
        self.feature_cols: List[str] = align_feature_columns(
            feature_cols_raw, expected_size=expected_size
        )

        self._buffers: Dict[str, Any] = {}
        self._last_seeded_minute: Dict[str, int] = {}
        self._lock = threading.Lock()

        LOGGER.info(
            "LegacyTransformerPredictor ready: dir=%s window=%d features=%d "
            "thr_long=%.3f thr_short=%.3f",
            self.model_dir,
            self.window_size,
            len(self.feature_cols),
            self.thr_long,
            self.thr_short,
        )

    # ------------------------------------------------------------------
    # Public callable -- this is what the supervisor invokes.
    # ------------------------------------------------------------------
    def __call__(
        self, symbol: str, ticker: Any
    ) -> Tuple[Literal["buy", "sell"], float]:
        try:
            self._refresh_buffer(symbol)
        except Exception as exc:  # noqa: BLE001 -- never crash the loop on data issues
            LOGGER.warning("predictor: candles refresh failed for %s: %s", symbol, exc)
            return _NEUTRAL_RESULT

        try:
            probs = self._predict(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("predictor: inference failed for %s: %s", symbol, exc)
            return _NEUTRAL_RESULT

        if probs is None:
            return _NEUTRAL_RESULT
        return self._probs_to_decision(probs)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------
    def _refresh_buffer(self, symbol: str) -> None:
        """Refresh per-symbol buffer with Coinbase 1m candles.

        On first call: seeds with ``warmup_bars`` candles. On subsequent calls
        within the same minute boundary, no-ops (the latest closed bar can't
        change inside its minute). Crossing a minute boundary triggers a
        small re-fetch and append.
        """
        import pandas as pd

        now_utc = datetime.now(timezone.utc)
        current_minute = int(now_utc.timestamp() // 60)
        last_minute = self._last_seeded_minute.get(symbol)
        if last_minute == current_minute and symbol in self._buffers:
            return  # cached; the latest closed bar hasn't changed yet

        candles = self.exchange.fetch_recent_candles(
            symbol, granularity="ONE_MINUTE", limit=self.warmup_bars
        )
        if not candles:
            return

        with self._lock:
            df = pd.DataFrame(candles)
            existing = self._buffers.get(symbol)
            if existing is not None:
                merged = pd.concat([existing, df], ignore_index=True)
                merged = merged.drop_duplicates(subset="timestamp", keep="last")
                merged = merged.sort_values("timestamp").reset_index(drop=True)
                df = merged.tail(self.max_buffer_bars).reset_index(drop=True)
            self._buffers[symbol] = df
            self._last_seeded_minute[symbol] = current_minute

    def _predict(self, symbol: str) -> Optional[np.ndarray]:
        """Compute features, build a window, return softmax probs or ``None``."""
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from utils import compute_features

        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < max(self.window_size, 240):
            LOGGER.info(
                "predictor: %s buffer warming up (%d bars, need %d)",
                symbol,
                0 if buf is None else len(buf),
                max(self.window_size, 240),
            )
            return None

        feats = compute_features(buf.copy())
        missing = [c for c in self.feature_cols if c not in feats.columns]
        if missing:
            LOGGER.warning(
                "predictor: %s missing feature cols, skipping: %s",
                symbol,
                missing[:5],
            )
            return None

        window_frame = feats[self.feature_cols].tail(self.window_size).astype(np.float32)
        if len(window_frame) < self.window_size:
            return None

        arr = window_frame.to_numpy()
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            arr = self.scaler.transform(arr)
        arr = arr.reshape(1, self.window_size, -1).astype(np.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)
        with torch.no_grad():
            tensor = torch.from_numpy(arr).to(device)
            logits = model(tensor)
            if self.num_classes <= 1 or logits.shape[-1] == 1:
                probs_t = torch.sigmoid(logits)
            else:
                probs_t = F.softmax(logits, dim=-1)
        probs = probs_t.squeeze(0).detach().cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([float(probs)])
        return probs

    # ------------------------------------------------------------------
    # Decision policy
    # ------------------------------------------------------------------
    def _probs_to_decision(
        self, probs: np.ndarray
    ) -> Tuple[Literal["buy", "sell"], float]:
        """Map softmax probabilities to ``(side, confidence)``.

        For 3-class output ordering ``[short, hold, long]``:
          * Long if ``p_long >= thr_long`` and ``p_long >= p_short``.
          * Short if ``p_short >= thr_short`` and ``p_short > p_long``.
          * Otherwise neutral (returns ``("buy", 0.5)``).

        Confidence is the chosen side's probability so the supervisor's
        ``min_confidence_to_trade`` gate is meaningful.
        """
        flat = np.asarray(probs).flatten()
        if flat.size >= 3:
            p_short, _p_hold, p_long = float(flat[0]), float(flat[1]), float(flat[2])
        elif flat.size == 2:
            p_short, p_long = float(flat[0]), float(flat[1])
        elif flat.size == 1:
            # Single-class binary head treated as P(long).
            p_long = float(flat[0])
            p_short = 1.0 - p_long
        else:
            return _NEUTRAL_RESULT

        # Apply thresholds + dominance check.
        if p_long >= p_short and p_long >= self.thr_long:
            return ("buy", p_long)
        if p_short > p_long and p_short >= self.thr_short:
            return ("sell", p_short)
        return _NEUTRAL_RESULT


def build_default_predict_fn(exchange: Any) -> Optional[Any]:
    """Build the predictor from ``LEGACY_MODEL_DIR`` env. Returns ``None`` on failure.

    The supervisor will fall back to its placeholder if this returns ``None``.
    """
    model_dir = os.getenv("LEGACY_MODEL_DIR", "model_sanity").strip()
    if not model_dir:
        return None
    if not Path(model_dir).expanduser().exists():
        LOGGER.info("predictor: %s does not exist, skipping legacy load", model_dir)
        return None
    try:
        return LegacyTransformerPredictor(model_dir=model_dir, exchange=exchange)
    except Exception as exc:  # noqa: BLE001 -- supervisor must never crash on model issues
        LOGGER.warning("predictor: legacy bundle load failed (%s); using placeholder", exc)
        return None
