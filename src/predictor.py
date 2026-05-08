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


def _is_valid_confidence(conf: float) -> bool:
    """Confidence must be a finite float in ``[0.0, 1.0]``.

    NaN, +/-inf, negative, or >1 values fail. Used by both predictor
    families as a last-line defence before returning to the supervisor —
    the supervisor's confidence floor would skip NaN anyway via < check
    semantics, but we prefer not to pass garbage downstream.
    """
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(c)) and 0.0 <= c <= 1.0


def _validated_decision(
    side: Literal["buy", "sell"], conf: float
) -> Tuple[Literal["buy", "sell"], float]:
    """Return ``(side, conf)`` if ``conf`` is finite + in [0, 1] else neutral."""
    if not _is_valid_confidence(conf):
        LOGGER.warning(
            "predictor: rejecting invalid confidence %r; returning neutral",
            conf,
        )
        return _NEUTRAL_RESULT
    return (side, float(conf))


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
        # Operator-visible: surface the raw class probabilities so threshold
        # tuning is data-driven. Logged once per minute (buffer refresh
        # cadence).
        if probs.size >= 3:
            LOGGER.info(
                "transformer predictor: %s P(short)=%.3f P(hold)=%.3f "
                "P(long)=%.3f (thr_long=%.2f thr_short=%.2f)",
                symbol,
                float(probs[0]),
                float(probs[1]),
                float(probs[2]),
                self.thr_long,
                self.thr_short,
            )
        elif probs.size == 2:
            LOGGER.info(
                "transformer predictor: %s P(short)=%.3f P(long)=%.3f",
                symbol,
                float(probs[0]),
                float(probs[1]),
            )
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

        Defensive: any non-finite or out-of-range confidence (NaN, +/-inf,
        negative, > 1.0) routes to ``_NEUTRAL_RESULT``. The supervisor's
        confidence floor would already skip such values, but we'd rather
        not pass garbage downstream.
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
            return _validated_decision("buy", p_long)
        if p_short > p_long and p_short >= self.thr_short:
            return _validated_decision("sell", p_short)
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
        thr_long: Optional[float] = None,
        warmup_bars: int = 350,
        max_buffer_bars: int = 5000,
    ) -> None:
        # Lazy heavy imports.
        import joblib

        self.model_dir = str(Path(model_dir).expanduser().resolve())
        if not Path(self.model_dir).exists():
            raise FileNotFoundError(
                f"XGBoost model_dir does not exist: {self.model_dir}"
            )
        self.exchange = exchange
        self.warmup_bars = int(warmup_bars)
        self.max_buffer_bars = int(max_buffer_bars)

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

        # Threshold precedence: explicit kwarg > meta.json optimal_threshold
        # > 0.5 default. Document so operators know what overrides what.
        meta_optimal = self.meta.get("optimal_threshold")
        if thr_long is not None:
            self.thr_long = float(thr_long)
            self._thr_source = "explicit"
        elif meta_optimal is not None:
            self.thr_long = float(meta_optimal)
            self._thr_source = "meta.optimal_threshold"
        else:
            self.thr_long = 0.5
            self._thr_source = "default"

        # Optional StandardScaler bundle. If present, its column order MUST
        # match meta.feature_cols exactly; silently scaling features in the
        # wrong order is the kind of footgun that produces seemingly-fine
        # probabilities that are wrong by every measure.
        scaler_path = Path(self.model_dir) / "scaler.joblib"
        self.scaler: Optional[Any] = None
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        if self.scaler is not None:
            if not hasattr(self.scaler, "feature_names_in_"):
                raise ValueError(
                    f"XGBoost bundle at {self.model_dir}: scaler is missing "
                    f"feature_names_in_ attribute; cannot verify feature "
                    f"column order. Refusing to load."
                )
            scaler_cols = list(self.scaler.feature_names_in_)
            if scaler_cols != self.feature_cols:
                raise ValueError(
                    f"XGBoost bundle at {self.model_dir}: scaler column "
                    f"order does not match meta.feature_cols. "
                    f"scaler_first5={scaler_cols[:5]} "
                    f"meta_first5={self.feature_cols[:5]}"
                )

        self._buffers: Dict[str, Any] = {}
        self._last_seeded_minute: Dict[str, int] = {}
        self._lock = threading.Lock()

        LOGGER.info(
            "XGBoostPredictor ready: dir=%s features=%d thr_long=%.3f (src=%s)",
            self.model_dir,
            len(self.feature_cols),
            self.thr_long,
            self._thr_source,
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
            return _validated_decision("buy", float(proba))
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
        proba = float(self.model.predict_proba(latest.to_numpy())[0, 1])
        # Defensive: a corrupted booster or numerically unstable feature row
        # could yield NaN / inf. Returning ``None`` makes ``__call__`` map
        # back to ``_NEUTRAL_RESULT`` rather than passing junk downstream.
        if not _is_valid_confidence(proba):
            LOGGER.warning(
                "xgb predictor: %s produced invalid proba %r; returning neutral",
                symbol,
                proba,
            )
            return None
        # Operator-visible: surface the raw probability so threshold tuning
        # is data-driven, not guesswork. One line per minute (buffer only
        # refreshes on minute boundary).
        verdict = "trigger" if proba >= self.thr_long else "neutral"
        LOGGER.info(
            "xgb predictor: %s P(long)=%.3f (thr=%.2f -> %s)",
            symbol,
            proba,
            self.thr_long,
            verdict,
        )
        return proba


class MultiSymbolXGBoostPredictor:
    """Per-symbol XGBoost models behind one supervisor predict_fn.

    Wraps multiple ``XGBoostPredictor`` instances keyed by symbol so each
    symbol can use its own trained model + per-symbol threshold. Required
    when the booster's probability range differs by symbol (BTC tops at
    0.34, ETH at 0.67 even on the same timestamp), making one global
    threshold useless.

    Map format (env CRYPTO_MODEL_MAP):
        "ETH/USD=model_crypto/eth_usd_v1:0.50,BTC/USD=model_crypto/btc_usd_v1:0.30"
    The ``:thr`` suffix on each entry is optional; if omitted, the
    predictor's default ``thr_long`` is used.

    Symbols with no entry in the map raise ``KeyError`` at predict time
    (rather than silently using the wrong model).
    """

    def __init__(self, *, model_map: Dict[str, "XGBoostPredictor"]) -> None:
        if not model_map:
            raise ValueError("model_map cannot be empty")
        self.model_map = model_map
        LOGGER.info(
            "MultiSymbolXGBoostPredictor ready for %d symbol(s): %s",
            len(model_map),
            ", ".join(sorted(model_map.keys())),
        )

    def __call__(
        self, symbol: str, ticker: Any
    ) -> Tuple[Literal["buy", "sell"], float]:
        predictor = self.model_map.get(symbol)
        if predictor is None:
            LOGGER.warning(
                "multi-symbol predictor: no model wired for %s; returning neutral",
                symbol,
            )
            return _NEUTRAL_RESULT
        return predictor(symbol, ticker)


def _parse_crypto_model_map(raw: str) -> Dict[str, Tuple[str, Optional[float]]]:
    """Parse ``"ETH/USD=path:0.5,BTC/USD=path"`` -> ``{symbol: (path, thr_or_None)}``.

    Empty / blank input returns ``{}``. Malformed entries are dropped with
    a warning so a single bad config line doesn't kill the whole map.
    """
    out: Dict[str, Tuple[str, Optional[float]]] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            LOGGER.warning(
                "CRYPTO_MODEL_MAP: skipping malformed entry %r (no '=')", entry
            )
            continue
        sym, _, rhs = entry.partition("=")
        sym = sym.strip()
        rhs = rhs.strip()
        thr: Optional[float] = None
        if ":" in rhs:
            path, _, thr_str = rhs.rpartition(":")
            try:
                thr = float(thr_str)
            except ValueError:
                LOGGER.warning(
                    "CRYPTO_MODEL_MAP: %s has invalid threshold %r; using default",
                    sym,
                    thr_str,
                )
                path = rhs  # treat the whole rhs as path, no threshold
        else:
            path = rhs
        if not sym or not path:
            LOGGER.warning("CRYPTO_MODEL_MAP: skipping incomplete entry %r", entry)
            continue
        out[sym] = (path, thr)
    return out


def build_default_predict_fn(exchange: Any) -> Optional[Any]:
    """Predictor selection priority:
    1. ``CRYPTO_MODEL_MAP`` -- multi-symbol XGBoost (one model per symbol)
    2. ``CRYPTO_MODEL_DIR`` -- single XGBoost model used for every symbol
    3. ``LEGACY_MODEL_DIR`` -- transformer in ``model_sanity/``
    4. ``None`` -- supervisor falls back to its neutral placeholder.

    The supervisor wires this in ``main()`` and never crashes on load
    failures; it falls back to the placeholder if every option fails.
    """
    # 1. Multi-symbol XGBoost map.
    raw_map = os.getenv("CRYPTO_MODEL_MAP", "").strip()
    if raw_map:
        parsed = _parse_crypto_model_map(raw_map)
        loaded: Dict[str, XGBoostPredictor] = {}
        for sym, (path, thr) in parsed.items():
            if not Path(path).expanduser().exists():
                LOGGER.warning(
                    "predictor: %s model dir %s missing; symbol will be neutral",
                    sym,
                    path,
                )
                continue
            try:
                eff_thr = thr if thr is not None else 0.5
                loaded[sym] = XGBoostPredictor(
                    model_dir=path, exchange=exchange, thr_long=eff_thr
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "predictor: %s model load from %s failed (%s); skipping",
                    sym,
                    path,
                    exc,
                )
        if loaded:
            return MultiSymbolXGBoostPredictor(model_map=loaded)
        LOGGER.warning(
            "predictor: CRYPTO_MODEL_MAP set but no models loaded; falling back"
        )

    # 2. Single XGBoost (preferred when present -- USD-native model).
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
