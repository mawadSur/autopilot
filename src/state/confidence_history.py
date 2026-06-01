"""Rolling confidence history for the auto-pause gate.

The auto-pause gate (``risk/auto_pause.py``) needs a baseline (mean, std)
for the model's recent confidence outputs to detect distribution shift —
"the model still produces signals but they're unusually low-confidence"
is a different signal than "today is a bad day on PnL terms".

Storage seam
------------
We use the same Redis client the position store already owns. Each symbol
has its own LIST (Redis ``LPUSH`` + ``LTRIM``) capped at ``window_size``
entries, keyed under ``{ns}:confidence_baseline:{symbol}``. The list is
fixed-size (most-recent at the head) so reads are O(N) bounded, and the
history survives process restarts.

If Redis is unreachable the helper degrades to a process-local deque
fallback so the supervisor still gets a baseline (just less stable across
restarts). All public methods swallow Redis failures silently — never
raise.
"""
from __future__ import annotations

import logging
import math
import statistics
from collections import deque
from typing import Any, Deque, Dict, List, Optional


LOGGER = logging.getLogger(__name__)


DEFAULT_WINDOW_SIZE = 200
DEFAULT_NAMESPACE = "autopilot"


def _is_finite_float(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


class ConfidenceHistory:
    """Rolling per-symbol confidence buffer with mean+std reads.

    Construct with the same Redis client the supervisor uses, OR pass
    ``redis_client=None`` to run with the process-local fallback only
    (useful in tests).
    """

    def __init__(
        self,
        *,
        redis_client: Any | None = None,
        namespace: str = DEFAULT_NAMESPACE,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        self._redis = redis_client
        self.namespace = namespace
        self.window_size = int(window_size)
        # Process-local fallback used when Redis is unavailable. Keyed by
        # symbol so per-symbol baselines stay isolated.
        self._fallback: Dict[str, Deque[float]] = {}

    def _key(self, symbol: str) -> str:
        return f"{self.namespace}:confidence_baseline:{symbol}"

    def _fallback_buf(self, symbol: str) -> Deque[float]:
        buf = self._fallback.get(symbol)
        if buf is None:
            buf = deque(maxlen=self.window_size)
            self._fallback[symbol] = buf
        return buf

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def record(self, symbol: str, confidence: float) -> None:
        """Append a confidence value to the per-symbol rolling buffer."""
        if not _is_finite_float(confidence):
            return
        value = float(confidence)
        if self._redis is not None:
            try:
                key = self._key(symbol)
                pipe = self._redis.pipeline()
                pipe.lpush(key, value)
                pipe.ltrim(key, 0, self.window_size - 1)
                pipe.execute()
                return
            except Exception as exc:  # noqa: BLE001 - fall through to local
                LOGGER.warning(
                    "ConfidenceHistory.record(%s) Redis call failed: %s; "
                    "using process-local fallback",
                    symbol,
                    exc,
                )
        self._fallback_buf(symbol).append(value)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def values(self, symbol: str) -> List[float]:
        """Return the current rolling buffer (oldest -> newest)."""
        if self._redis is not None:
            try:
                raw = self._redis.lrange(self._key(symbol), 0, self.window_size - 1)
                if raw is None:
                    return []
                # LPUSH ordering is newest-first; reverse for oldest-first.
                values = [
                    float(x) for x in raw if _is_finite_float(x)
                ]
                values.reverse()
                if values:
                    return values
            except Exception as exc:  # noqa: BLE001 - fall through to local
                LOGGER.warning(
                    "ConfidenceHistory.values(%s) Redis call failed: %s; "
                    "using process-local fallback",
                    symbol,
                    exc,
                )
        return list(self._fallback_buf(symbol))

    def baseline(self, symbol: str) -> tuple[float, float, int]:
        """Return (mean, stdev, n) for the rolling buffer.

        Returns ``(0.0, 0.0, 0)`` if there are fewer than two samples,
        because stdev is undefined on a one-element population.
        """
        values = self.values(symbol)
        n = len(values)
        if n < 2:
            return (
                float(values[0]) if n == 1 else 0.0,
                0.0,
                n,
            )
        try:
            mean = statistics.fmean(values)
            std = statistics.pstdev(values, mu=mean)
        except statistics.StatisticsError:
            return (0.0, 0.0, n)
        return (mean, std, n)


__all__ = [
    "DEFAULT_NAMESPACE",
    "DEFAULT_WINDOW_SIZE",
    "ConfidenceHistory",
]
