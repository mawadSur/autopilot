from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


class StrategyGate:
    """Translate model probabilities into class signals with optional hard gating."""

    def __init__(
        self,
        *,
        thr_long: float,
        thr_short: float,
        margin: float,
        feature_cols: Optional[Sequence[str]] = None,
        use_hard_gating: bool = True,
        sweep_lookback_bars: int = 5,
        confluence_lookback_bars: Optional[int] = None,
        avwap_proximity_pct: float = 0.0015,
    ) -> None:
        self.thr_long = float(thr_long)
        self.thr_short = float(thr_short)
        self.margin = float(margin)
        self.feature_index = {str(name): idx for idx, name in enumerate(feature_cols or [])}
        self.use_hard_gating = bool(use_hard_gating)
        self.sweep_lookback_bars = int(max(1, sweep_lookback_bars))
        self.confluence_lookback_bars = int(
            max(1, confluence_lookback_bars or self.sweep_lookback_bars)
        )
        self.avwap_proximity_pct = float(max(0.0, avwap_proximity_pct))

    @staticmethod
    def _coerce_float(value: Any, default: float = np.nan) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out if np.isfinite(out) else float(default)

    def _window_feature_values(self, window: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if window is None or name not in self.feature_index:
            return None
        arr = np.asarray(window)
        if arr.ndim != 2:
            return None
        col_idx = self.feature_index[name]
        if col_idx < 0 or col_idx >= arr.shape[1]:
            return None
        try:
            values = arr[:, col_idx].astype(np.float64, copy=False)
        except Exception:
            return None
        return values

    def _current_feature_value(
        self,
        name: str,
        *,
        window: Optional[np.ndarray] = None,
        feature_row: Optional[Mapping[str, Any]] = None,
        default: float = np.nan,
    ) -> float:
        values = self._window_feature_values(window, name)
        if values is not None and values.size:
            return self._coerce_float(values[-1], default=default)
        if feature_row is not None:
            return self._coerce_float(feature_row.get(name, default), default=default)
        return float(default)

    def _recent_feature_true(
        self,
        name: str,
        *,
        window: Optional[np.ndarray] = None,
        feature_row: Optional[Mapping[str, Any]] = None,
        lookback: Optional[int] = None,
        threshold: float = 0.5,
    ) -> bool:
        values = self._window_feature_values(window, name)
        if values is not None and values.size:
            tail = values[-int(max(1, lookback or self.sweep_lookback_bars)) :]
            valid = tail[np.isfinite(tail)]
            return bool(valid.size and np.any(valid > float(threshold)))
        current = self._current_feature_value(name, feature_row=feature_row, default=0.0)
        return bool(np.isfinite(current) and current > float(threshold))

    def _recent_feature_abs_leq(
        self,
        name: str,
        *,
        window: Optional[np.ndarray] = None,
        feature_row: Optional[Mapping[str, Any]] = None,
        lookback: Optional[int] = None,
        threshold: float,
    ) -> bool:
        values = self._window_feature_values(window, name)
        if values is not None and values.size:
            tail = values[-int(max(1, lookback or self.confluence_lookback_bars)) :]
            valid = tail[np.isfinite(tail)]
            return bool(valid.size and np.any(np.abs(valid) <= float(threshold)))
        current = self._current_feature_value(name, feature_row=feature_row, default=np.nan)
        return bool(np.isfinite(current) and abs(current) <= float(threshold))

    def long_confluences(
        self,
        *,
        window: Optional[np.ndarray] = None,
        feature_row: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, bool]:
        lookback = self.confluence_lookback_bars
        return {
            "recent_liq_sweep_low": self._recent_feature_true(
                "liq_sweep_low",
                window=window,
                feature_row=feature_row,
                lookback=self.sweep_lookback_bars,
            ),
            "near_avwap_cycle": self._recent_feature_abs_leq(
                "close_over_avwap_cycle",
                window=window,
                feature_row=feature_row,
                lookback=lookback,
                threshold=self.avwap_proximity_pct,
            ),
            "near_avwap_spike": self._recent_feature_abs_leq(
                "close_over_avwap_spike",
                window=window,
                feature_row=feature_row,
                lookback=lookback,
                threshold=self.avwap_proximity_pct,
            ),
            "in_golden_pocket": self._recent_feature_true(
                "in_golden_pocket",
                window=window,
                feature_row=feature_row,
                lookback=lookback,
            ),
        }

    def apply_hard_gate(
        self,
        sig: int,
        *,
        window: Optional[np.ndarray] = None,
        feature_row: Optional[Mapping[str, Any]] = None,
    ) -> int:
        sig = int(sig)
        if not self.use_hard_gating or sig != 2:
            return sig
        confluences = self.long_confluences(window=window, feature_row=feature_row)
        return 2 if any(confluences.values()) else 1

    def raw_signal_from_probs(self, probabilities: np.ndarray) -> int:
        row = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if row.size >= 3:
            p_short, _p_hold, p_long = float(row[0]), float(row[1]), float(row[2])
            if p_long >= self.thr_long and (p_long - max(p_short, float(row[1]))) >= self.margin:
                return 2
            if p_short >= self.thr_short and (p_short - max(p_long, float(row[1]))) >= self.margin:
                return 0
            return 1
        if row.size == 2:
            p_short, p_long = float(row[0]), float(row[1])
            gap = abs(p_long - p_short)
            if p_long >= p_short and p_long >= self.thr_long and gap >= self.margin:
                return 2
            if p_short > p_long and p_short >= self.thr_short and gap >= self.margin:
                return 0
            return 1
        if row.size == 1:
            return 2 if float(row[0]) >= self.thr_long else 1
        return 1

    def signal_from_probs(
        self,
        probabilities: np.ndarray,
        *,
        window: Optional[np.ndarray] = None,
        feature_row: Optional[Mapping[str, Any]] = None,
    ) -> int:
        raw_sig = self.raw_signal_from_probs(probabilities)
        return self.apply_hard_gate(raw_sig, window=window, feature_row=feature_row)
