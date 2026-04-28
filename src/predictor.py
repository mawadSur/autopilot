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

import json
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


class XGBoostPredictor:
    """Wraps an XGBoost calibration bundle (``model_crypto/<v>/``) for the supervisor.

    The model trained by ``crypto_training.train_xgboost`` is a binary
    classifier: ``predict_proba(X)[:, 1]`` is ``P(forward return >
    threshold_bps over horizon_bars)``. We treat that as ``P(long is
    profitable)`` and:

      * map ``proba >= thr_long`` -> ``("buy", proba)``
      * everything else -> neutral ``("buy", 0.5)`` (below the supervisor's
        confidence floor, so no order is placed).

    There is no ``short`` side -- the binary head doesn't predict it. If you
    want shorts, train a 3-class model.

    The buffer + feature computation mirrors ``LegacyTransformerPredictor``:
    Coinbase 1m candles refreshed on minute boundary, ``utils.compute_features``
    over the buffer, take the LAST row's selected feature columns.
    """

    def __init__(
        self,
        *,
        model_dir: str,
        exchange: Any,
        thr_long: float = 0.65,
        warmup_bars: int = 350,
        max_buffer_bars: int = 5000,
    ) -> None:
        # Lazy heavy imports.
        import joblib

        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self.exchange = exchange
        self.warmup_bars = int(warmup_bars)
        self.max_buffer_bars = int(max_buffer_bars)
        self.thr_long = float(thr_long)

        meta_path = Path(self.model_dir) / "meta.json"
        model_path = Path(self.model_dir) / "model.joblib"
        if not meta_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost bundle missing at {self.model_dir}: "
                f"need model.joblib + meta.json"
            )
        with meta_path.open() as fh:
            self.meta: Dict[str, Any] = json.load(fh)
        feature_cols_raw = self.meta.get("feature_cols") or []
        if not feature_cols_raw:
            raise ValueError("meta.json missing feature_cols")
        self.feature_cols: List[str] = list(feature_cols_raw)
        self.model = joblib.load(model_path)

        self._buffers: Dict[str, Any] = {}
        self._last_seeded_minute: Dict[str, int] = {}
        self._lock = threading.Lock()

        LOGGER.info(
            "XGBoostPredictor ready: dir=%s features=%d thr_long=%.3f",
            self.model_dir,
            len(self.feature_cols),
            self.thr_long,
        )

    def __call__(
        self, symbol: str, ticker: Any
    ) -> Tuple[Literal["buy", "sell"], float]:
        try:
            self._refresh_buffer(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("xgb predictor: candles refresh failed for %s: %s", symbol, exc)
            return _NEUTRAL_RESULT
        try:
            proba = self._predict(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("xgb predictor: inference failed for %s: %s", symbol, exc)
            return _NEUTRAL_RESULT
        if proba is None:
            return _NEUTRAL_RESULT
        if proba >= self.thr_long:
            return ("buy", float(proba))
        return _NEUTRAL_RESULT

    def _refresh_buffer(self, symbol: str) -> None:
        """Same minute-boundary refresh pattern as the transformer predictor."""
        import pandas as pd

        now_utc = datetime.now(timezone.utc)
        current_minute = int(now_utc.timestamp() // 60)
        last_minute = self._last_seeded_minute.get(symbol)
        if last_minute == current_minute and symbol in self._buffers:
            return
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

    def _predict(self, symbol: str) -> Optional[float]:
        """Compute features + return P(long-profitable) for the latest bar."""
        import numpy as np
        from utils import compute_features

        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < 240:
            LOGGER.info(
                "xgb predictor: %s buffer warming up (%d bars, need 240)",
                symbol,
                0 if buf is None else len(buf),
            )
            return None
        # compute_features drops the timestamp column; we only need the
        # final row's feature values for the XGBoost path.
        feats = compute_features(buf.copy())
        missing = [c for c in self.feature_cols if c not in feats.columns]
        if missing:
            LOGGER.warning(
                "xgb predictor: %s missing feature cols, skipping: %s",
                symbol,
                missing[:5],
            )
            return None
        latest = feats[self.feature_cols].iloc[-1:].astype("float32")
        if latest.isna().any().any():
            latest = latest.fillna(0.0)
        proba = self.model.predict_proba(latest.to_numpy())[0, 1]
        return float(proba)


def build_default_predict_fn(exchange: Any) -> Optional[Any]:
    """Try XGBoost first (``CRYPTO_MODEL_DIR``), fall back to the legacy
    transformer (``LEGACY_MODEL_DIR``), fall back to None (placeholder).

    The supervisor wires this in ``main()`` and falls back to its
    placeholder predictor if this returns ``None``.
    """
    # 1. XGBoost (preferred when present -- USD-native model).
    crypto_dir = os.getenv("CRYPTO_MODEL_DIR", "").strip()
    if crypto_dir and Path(crypto_dir).expanduser().exists():
        try:
            thr_env = os.getenv("CRYPTO_MODEL_THR_LONG", "").strip()
            thr_long = float(thr_env) if thr_env else 0.65
            LOGGER.info(
                "predictor: trying XGBoost bundle at %s (thr_long=%.3f)",
                crypto_dir,
                thr_long,
            )
            return XGBoostPredictor(
                model_dir=crypto_dir, exchange=exchange, thr_long=thr_long
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "predictor: XGBoost bundle load failed (%s); falling back to legacy",
                exc,
            )

    # 2. Legacy transformer (model_sanity/) as fallback.
    legacy_dir = os.getenv("LEGACY_MODEL_DIR", "model_sanity").strip()
    if not legacy_dir:
        return None
    if not Path(legacy_dir).expanduser().exists():
        LOGGER.info("predictor: %s does not exist, skipping legacy load", legacy_dir)
        return None
    try:
        return LegacyTransformerPredictor(model_dir=legacy_dir, exchange=exchange)
    except Exception as exc:  # noqa: BLE001 -- supervisor must never crash on model issues
        LOGGER.warning("predictor: legacy bundle load failed (%s); using placeholder", exc)
        return None
