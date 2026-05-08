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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

_NEUTRAL_RESULT: Tuple[Literal["buy", "sell"], float] = ("buy", 0.5)


# ---------------------------------------------------------------------------
# Rich predictor return type (Lane B / A1 SignalForensics gap closure).
#
# The supervisor's loop only needs ``(side, confidence)``, so ``__call__``
# stays a 2-tuple — backward-compatible with every existing caller and the
# 700+ tests that destructure ``side, conf = predictor(...)``. But A1's
# Mahalanobis / per-bin reliability / feature-quality checks need the
# *latest-bar features* and the *raw class probabilities* at signal time,
# not just the chosen side.
#
# ``predict_full`` returns ``PredictorResult`` exposing both. The supervisor
# can adopt it later (separate change); A1 / context-snapshot writers can
# adopt it now.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictorResult:
    """Rich predictor output for downstream forensics + snapshot capture.

    ``side`` / ``confidence`` mirror the legacy 2-tuple. ``feature_buffer``
    is the per-feature snapshot from the *latest* bar (used by A1's
    Mahalanobis + NaN/inf checks). ``model_probs`` is the raw class-prob
    dict (used by A1's reliability bin lookup).

    Both rich fields are ``None`` for neutral / warmup / error returns, so
    callers must handle ``None`` (A1 already does).
    """

    side: Literal["buy", "sell"]
    confidence: float
    feature_buffer: Optional[Dict[str, float]] = None
    model_probs: Optional[Dict[str, float]] = None
    # Free-form metadata for the future (e.g. raw logits, feature window).
    # Keep it ``Mapping[str, Any]`` so callers can add fields without
    # bumping the dataclass schema.
    extras: Dict[str, Any] = field(default_factory=dict)


_NEUTRAL_RICH_RESULT = PredictorResult(side="buy", confidence=0.5)


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
        # Backward-compatible 2-tuple surface. ``predict_full`` does the real
        # work and we project down to ``(side, confidence)`` so the
        # supervisor + every existing destructure-style caller is unchanged.
        result = self.predict_full(symbol, ticker)
        return (result.side, result.confidence)

    # ------------------------------------------------------------------
    # Rich predictor surface (A1 forensics + future snapshot writers).
    # ------------------------------------------------------------------
    def predict_full(
        self, symbol: str, ticker: Any
    ) -> PredictorResult:
        """Return the full ``(side, conf, feature_buffer, model_probs)``.

        On any neutral / warmup / error path the rich fields are ``None``
        (matches A1's tolerant ``signal.feature_buffer or {}`` handling).
        """
        try:
            self._refresh_buffer(symbol)
        except Exception as exc:  # noqa: BLE001 -- never crash the loop on data issues
            LOGGER.warning("predictor: candles refresh failed for %s: %s", symbol, exc)
            return _NEUTRAL_RICH_RESULT

        try:
            probs, feature_buffer = self._predict_with_features(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("predictor: inference failed for %s: %s", symbol, exc)
            return _NEUTRAL_RICH_RESULT

        if probs is None:
            return _NEUTRAL_RICH_RESULT
        side, conf = self._probs_to_decision(probs)
        # Build the model_probs dict from the raw softmax. For 3-class
        # ordering ``[short, hold, long]`` we expose all three keys; for
        # 2-class binary heads we expose ``short`` + ``long``.
        flat = np.asarray(probs).flatten()
        model_probs: Dict[str, float] = {}
        if flat.size >= 3:
            model_probs = {
                "short": float(flat[0]),
                "hold": float(flat[1]),
                "long": float(flat[2]),
            }
        elif flat.size == 2:
            model_probs = {
                "short": float(flat[0]),
                "long": float(flat[1]),
            }
        elif flat.size == 1:
            p_long = float(flat[0])
            model_probs = {"long": p_long, "short": 1.0 - p_long}
        return PredictorResult(
            side=side,
            confidence=conf,
            feature_buffer=feature_buffer if feature_buffer else None,
            model_probs=model_probs if model_probs else None,
        )

    @property
    def model_meta(self) -> Dict[str, Any]:
        """Return the loaded ``model_meta.json`` blob.

        A1 reads ``feature_means``, ``feature_stds``, ``optimal_threshold``,
        ``threshold_metrics`` etc. via this handle so the agent doesn't have
        to re-parse the file (and so tests can stub a meta dict cleanly).
        """
        # ``self.meta`` is set in ``__init__`` from ``load_model_bundle``.
        return dict(self.meta) if isinstance(self.meta, dict) else {}

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
        """Compute features, build a window, return softmax probs or ``None``.

        Thin wrapper around ``_predict_with_features`` for back-compat;
        any caller that wants the latest-bar feature snapshot too should
        use ``_predict_with_features`` directly.
        """
        probs, _features = self._predict_with_features(symbol)
        return probs

    def _predict_with_features(
        self, symbol: str
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
        """Return ``(probs, feature_buffer)`` -- both ``None`` on warmup.

        ``feature_buffer`` is the *latest* bar's per-feature dict (raw,
        un-scaled, NaN-dropped) so A1's Mahalanobis check sees the same
        feature values the trader saw at signal time.
        """
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
            return None, None

        feats = compute_features(buf.copy())
        missing = [c for c in self.feature_cols if c not in feats.columns]
        if missing:
            LOGGER.warning(
                "predictor: %s missing feature cols, skipping: %s",
                symbol,
                missing[:5],
            )
            return None, None

        window_frame = feats[self.feature_cols].tail(self.window_size).astype(np.float32)
        if len(window_frame) < self.window_size:
            return None, None

        # Snapshot the *latest* bar's raw feature values BEFORE scaling so
        # A1 can reason about them in the same units they had on disk.
        latest_row = window_frame.iloc[-1]
        feature_buffer: Dict[str, float] = {}
        for feat in self.feature_cols:
            try:
                v = float(latest_row[feat])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(v):
                # Capture NaN/inf as ``None`` so A1's NaN/inf check fires.
                feature_buffer[feat] = None  # type: ignore[assignment]
            else:
                feature_buffer[feat] = v

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
        return probs, feature_buffer

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

        # Optional regime-memory lookup. ``REGIME_STORE_PATH`` points at an
        # .npz produced by ``regime_memory.backfill``. When set + valid, we
        # construct a RegimeLookup and consult it per-predict; on resolve
        # confidence >= 0.5 we override ``thr_long`` for that prediction
        # only (and cache the resolved kelly fraction on
        # ``self._last_resolved_kelly_pct`` for the risk engine). When
        # unset (or load fails) the static threshold path runs unchanged —
        # the documented backward-compat contract in
        # ``src/regime_memory/INTEGRATION.md``.
        self.regime_lookup: Optional[Any] = None
        self._last_resolved_kelly_pct: Optional[float] = None
        self._maybe_init_regime_lookup()

        LOGGER.info(
            "XGBoostPredictor ready: dir=%s features=%d thr_long=%.3f (src=%s) "
            "regime_lookup=%s",
            self.model_dir,
            len(self.feature_cols),
            self.thr_long,
            self._thr_source,
            "on" if self.regime_lookup is not None else "off",
        )

    def _maybe_init_regime_lookup(self) -> None:
        """Best-effort load of a RegimeLookup from ``REGIME_STORE_PATH``.

        Sets ``self.regime_lookup`` to a constructed
        :class:`regime_memory.lookup.RegimeLookup` on success; leaves it as
        ``None`` and logs a warning on any failure. Never raises out — a
        broken regime store must not stop a predictor from constructing.
        """

        store_path = os.getenv("REGIME_STORE_PATH", "").strip()
        if not store_path:
            return
        path_obj = Path(store_path).expanduser()
        if not path_obj.exists():
            LOGGER.warning(
                "REGIME_STORE_PATH=%s does not exist; regime lookup disabled",
                store_path,
            )
            return
        try:
            from regime_memory.encoder import RegimeEncoder
            from regime_memory.lookup import RegimeLookup
            from regime_memory.store import NaiveRegimeStore

            store = NaiveRegimeStore.load(path_obj)
            encoder = RegimeEncoder(feature_cols=self.feature_cols)
            defaults = {
                "optimal_threshold": float(self.thr_long),
                "kelly_size_pct": 0.0,
            }
            self.regime_lookup = RegimeLookup(
                store=store, encoder=encoder, defaults=defaults
            )
            LOGGER.info(
                "regime_lookup initialised from %s (size=%d)",
                store_path,
                len(store),
            )
        except Exception as exc:  # noqa: BLE001 - never crash on regime store issues
            LOGGER.warning(
                "regime_lookup load failed from %s: %r; falling back to static",
                store_path,
                exc,
            )
            self.regime_lookup = None

    def __call__(
        self, symbol: str, ticker: Any
    ) -> Tuple[Literal["buy", "sell"], float]:
        # Backward-compatible 2-tuple surface. ``predict_full`` does the
        # real inference; we project down here so the supervisor + every
        # destructure-style caller stays unchanged.
        result = self.predict_full(symbol, ticker)
        return (result.side, result.confidence)

    def predict_full(
        self, symbol: str, ticker: Any
    ) -> PredictorResult:
        """Return the full ``(side, conf, feature_buffer, model_probs)``.

        ``feature_buffer`` is the latest-bar raw feature dict (per
        ``self.feature_cols``); ``model_probs`` is the binary head's
        ``{"long": p1, "short": p0}`` (since the booster is binary today).
        """
        try:
            self._refresh_buffer(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("xgb predictor: candles refresh failed for %s: %s", symbol, exc)
            return _NEUTRAL_RICH_RESULT
        try:
            proba, feature_buffer, feature_window = self._predict_with_features(symbol)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("xgb predictor: inference failed for %s: %s", symbol, exc)
            return _NEUTRAL_RICH_RESULT
        if proba is None:
            return _NEUTRAL_RICH_RESULT

        # Binary booster: P(long-profitable) is class 1; P(flat/short) is
        # the complement. ``label_classes`` could be [-1, 1] or [0, 1] —
        # we expose it under ``short`` (the negative class) so A1 can pick
        # max() and compare against threshold bins consistently.
        model_probs: Dict[str, float] = {
            "long": float(proba),
            "short": float(1.0 - proba),
        }

        # Resolve threshold via regime lookup if available — falls back to
        # static ``self.thr_long`` when lookup is None, returns low
        # confidence, or raises mid-prediction.
        effective_thr_long = self._resolve_threshold(feature_window)
        if proba >= effective_thr_long:
            side, conf = _validated_decision("buy", float(proba))
        else:
            side, conf = _NEUTRAL_RESULT
        return PredictorResult(
            side=side,
            confidence=conf,
            feature_buffer=feature_buffer if feature_buffer else None,
            model_probs=model_probs,
        )

    def _resolve_threshold(self, feature_window: Any) -> float:
        """Return the threshold to use for *this* prediction.

        Consults :attr:`regime_lookup` when set. On
        ``_regime_confidence >= 0.5`` we use the resolved
        ``optimal_threshold`` and cache the resolved ``kelly_size_pct``
        on ``self._last_resolved_kelly_pct`` for the risk engine to read.
        On low confidence, lookup failure, or no lookup configured, the
        static ``self.thr_long`` is returned and the cached kelly is
        cleared (so a stale value can't leak into the next tick).

        ``feature_window`` should be a pandas DataFrame of trailing
        feature rows (whatever the predictor used for inference). Passing
        ``None`` or any non-DataFrame falls back to static.
        """

        lookup = self.regime_lookup
        if lookup is None or feature_window is None:
            self._last_resolved_kelly_pct = None
            return self.thr_long
        try:
            resolved = lookup.resolve_params(feature_window, k=10)
        except Exception as exc:  # noqa: BLE001 - graceful degradation
            LOGGER.warning(
                "xgb predictor: regime_lookup raised mid-predict: %r; "
                "falling back to static threshold",
                exc,
            )
            self._last_resolved_kelly_pct = None
            return self.thr_long
        try:
            confidence = float(resolved.get("_regime_confidence", 0.0))
        except (TypeError, ValueError):
            self._last_resolved_kelly_pct = None
            return self.thr_long
        if confidence < 0.5:
            # Low-match → soft prior at most. Static path stays in charge.
            self._last_resolved_kelly_pct = None
            return self.thr_long
        try:
            new_thr = float(resolved.get("optimal_threshold", self.thr_long))
        except (TypeError, ValueError):
            new_thr = self.thr_long
        # Cache the resolved kelly fraction so the risk engine can pull it
        # without re-running the lookup. None when the lookup didn't
        # surface one (defensive).
        try:
            self._last_resolved_kelly_pct = float(
                resolved.get("kelly_size_pct", 0.0)
            )
        except (TypeError, ValueError):
            self._last_resolved_kelly_pct = None
        LOGGER.info(
            "regime_lookup: confidence=%.3f -> thr=%.4f kelly=%s",
            confidence,
            new_thr,
            f"{self._last_resolved_kelly_pct:.4f}"
            if self._last_resolved_kelly_pct is not None
            else "n/a",
        )
        return new_thr

    @property
    def model_meta(self) -> Dict[str, Any]:
        """Return the loaded ``meta.json`` blob.

        A1 uses this to reach ``feature_means`` / ``feature_stds`` /
        ``optimal_threshold`` / ``threshold_metrics`` without re-parsing
        from disk.
        """
        return dict(self.meta) if isinstance(self.meta, dict) else {}

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
        """Compute features + return P(long-profitable) for the latest bar.

        Thin wrapper around ``_predict_with_features`` for back-compat.
        """
        proba, _features, _window = self._predict_with_features(symbol)
        return proba

    def _predict_with_features(
        self, symbol: str
    ) -> Tuple[Optional[float], Optional[Dict[str, float]], Optional[Any]]:
        """Return ``(proba, feature_buffer, feature_window)`` -- all ``None`` on warmup.

        ``feature_buffer`` is the latest-bar's raw feature dict (NaN
        captured as ``None`` so A1's NaN/inf check fires), keyed on
        ``self.feature_cols``.

        ``feature_window`` is the trailing 60-bar DataFrame of feature
        values (post-fill, ready for the regime encoder). Returned
        alongside ``feature_buffer`` so ``predict_full`` can pass it to
        the optional :class:`RegimeLookup` without recomputing features.
        """
        import numpy as np
        from utils import compute_features

        buf = self._buffers.get(symbol)
        if buf is None or len(buf) < 240:
            LOGGER.info(
                "xgb predictor: %s buffer warming up (%d bars, need 240)",
                symbol,
                0 if buf is None else len(buf),
            )
            return None, None, None
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
            return None, None, None
        latest = feats[self.feature_cols].iloc[-1:].astype("float32")

        # Trailing 60-bar window for the regime encoder. We keep this
        # separate from ``latest`` because the encoder operates on a
        # window of features, not just the most recent row.
        feature_window = feats[self.feature_cols].tail(60)

        # Snapshot the latest-bar feature values BEFORE NaN-fill so A1's
        # NaN/inf detector still sees them (we capture NaN/inf as ``None``
        # in the buffer dict; it's the inference path that fills with 0).
        feature_buffer: Dict[str, float] = {}
        latest_row = latest.iloc[0]
        for feat in self.feature_cols:
            try:
                v = float(latest_row[feat])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(v):
                feature_buffer[feat] = None  # type: ignore[assignment]
            else:
                feature_buffer[feat] = v

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
            return None, None, None
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
        return proba, feature_buffer, feature_window


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
        # Validate every symbol's predictor at construction time -- a bad
        # bundle should fail loudly here, not on first inference call.
        for sym, predictor in model_map.items():
            if predictor is None:
                raise ValueError(
                    f"MultiSymbolXGBoostPredictor: model for {sym!r} is None"
                )
            # XGBoostPredictor instances must already have their feature_cols
            # + model attributes populated (set in __init__). This catches
            # half-constructed stubs that would NPE later.
            for attr in ("feature_cols", "model", "thr_long"):
                if isinstance(predictor, XGBoostPredictor) and not hasattr(
                    predictor, attr
                ):
                    raise ValueError(
                        f"MultiSymbolXGBoostPredictor: predictor for {sym!r} "
                        f"is missing required attribute {attr!r}"
                    )
        self.model_map = model_map
        # Optional per-symbol regime-store override. The global
        # ``REGIME_STORE_PATH`` was applied during each per-symbol predictor's
        # __init__; here we honour the per-symbol form
        # ``REGIME_STORE_PATH_<SAFE_SYMBOL>=...`` (e.g. for ``ETH/USD`` →
        # ``REGIME_STORE_PATH_ETH_USD``) so a multi-symbol bot can wire one
        # store per symbol cleanly. When the per-symbol var is set we
        # re-run regime-init on that predictor; when unset we leave its
        # regime_lookup as set by the global env var (or None).
        self._maybe_apply_per_symbol_regime_paths()
        LOGGER.info(
            "MultiSymbolXGBoostPredictor ready for %d symbol(s): %s",
            len(model_map),
            ", ".join(sorted(model_map.keys())),
        )

    @staticmethod
    def _symbol_env_token(symbol: str) -> str:
        """Map ``"ETH/USD"`` → ``"ETH_USD"`` for env-var key construction."""

        out = []
        for ch in symbol:
            if ch.isalnum():
                out.append(ch.upper())
            else:
                out.append("_")
        return "".join(out)

    def _maybe_apply_per_symbol_regime_paths(self) -> None:
        """Honour ``REGIME_STORE_PATH_<SAFE_SYMBOL>`` overrides per predictor."""

        for symbol, predictor in self.model_map.items():
            if not isinstance(predictor, XGBoostPredictor):
                continue
            token = self._symbol_env_token(symbol)
            per_sym_var = f"REGIME_STORE_PATH_{token}"
            override = os.getenv(per_sym_var, "").strip()
            if not override:
                continue
            # Temporarily flip the env var so the predictor's helper picks
            # it up, then restore. We avoid duplicating the load logic so
            # the failure paths stay identical between the global and
            # per-symbol forms.
            saved = os.environ.get("REGIME_STORE_PATH")
            os.environ["REGIME_STORE_PATH"] = override
            try:
                predictor._maybe_init_regime_lookup()
            finally:
                if saved is None:
                    os.environ.pop("REGIME_STORE_PATH", None)
                else:
                    os.environ["REGIME_STORE_PATH"] = saved

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

    def predict_full(
        self, symbol: str, ticker: Any
    ) -> PredictorResult:
        """Route to the per-symbol predictor's ``predict_full`` if available.

        Older / stub predictors that only implement ``__call__`` (the
        2-tuple form) are handled by projecting their tuple back into a
        ``PredictorResult`` with empty rich fields. That keeps existing
        wiring + tests working while still letting A1 pull rich fields
        from real ``XGBoostPredictor`` instances.
        """
        predictor = self.model_map.get(symbol)
        if predictor is None:
            LOGGER.warning(
                "multi-symbol predictor: no model wired for %s; returning neutral",
                symbol,
            )
            return _NEUTRAL_RICH_RESULT
        # Real per-symbol predictors expose predict_full; stubs in tests
        # may only expose __call__. Route accordingly.
        rich = getattr(predictor, "predict_full", None)
        if callable(rich):
            return rich(symbol, ticker)
        side, conf = predictor(symbol, ticker)
        return PredictorResult(side=side, confidence=conf)

    def model_meta_for(self, symbol: str) -> Dict[str, Any]:
        """Return the per-symbol meta blob for the given symbol or ``{}``.

        Useful for A1 in a multi-symbol setup: each symbol has its own
        meta.json and A1 needs to load the *right* one for the trade
        being investigated.
        """
        predictor = self.model_map.get(symbol)
        if predictor is None:
            return {}
        meta = getattr(predictor, "model_meta", None)
        if isinstance(meta, dict):
            return dict(meta)
        if callable(meta):  # property fall-through if attribute lookup quirks
            try:
                resolved = meta()
                if isinstance(resolved, dict):
                    return dict(resolved)
            except Exception:  # noqa: BLE001
                return {}
        return {}


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
                # Configured-but-missing is a human error. Don't silently
                # downgrade the symbol to neutral -- fail so the operator
                # notices.
                raise FileNotFoundError(
                    f"CRYPTO_MODEL_MAP entry for {sym!r}: path {path!r} does "
                    f"not exist on disk. Fix the path or remove the entry."
                )
            loaded[sym] = XGBoostPredictor(
                model_dir=path, exchange=exchange, thr_long=thr
            )
        if loaded:
            return MultiSymbolXGBoostPredictor(model_map=loaded)
        LOGGER.warning(
            "predictor: CRYPTO_MODEL_MAP set but no models loaded; falling back"
        )

    # 2. Single XGBoost (preferred when present -- USD-native model).
    crypto_dir = os.getenv("CRYPTO_MODEL_DIR", "").strip()
    if crypto_dir:
        # Configured but missing on disk = operator error. Raise loudly
        # rather than silently falling through to a legacy path the
        # operator didn't intend to use.
        if not Path(crypto_dir).expanduser().exists():
            raise FileNotFoundError(
                f"CRYPTO_MODEL_DIR is configured but does not exist on disk: "
                f"{crypto_dir}. Refusing to silently fall back; either fix "
                f"the path or unset the env var."
            )
        thr_env = os.getenv("CRYPTO_MODEL_THR_LONG", "").strip()
        thr_long = float(thr_env) if thr_env else None
        LOGGER.info(
            "predictor: trying XGBoost bundle at %s (thr_long=%s)",
            crypto_dir,
            thr_long if thr_long is not None else "from-meta-or-default",
        )
        return XGBoostPredictor(
            model_dir=crypto_dir, exchange=exchange, thr_long=thr_long
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
