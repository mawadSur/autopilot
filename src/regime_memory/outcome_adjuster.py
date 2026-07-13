"""Regime-scoped threshold adjuster driven by closed-trade outcomes.

Sprint 2 item #6 (CEO review 2026-05-17). This is a NEW component that is
DISTINCT from ``src/calibration_agent/outcome_weight_adjuster.py`` (which
adjusts Polymarket ensemble weights). The :class:`OutcomeAdjuster` here
nudges the ``optimal_threshold`` returned by
:class:`regime_memory.lookup.RegimeLookup` — per regime label, in response
to consecutive wins/losses within that regime.

Wiring
------
* :class:`RegimeLookup` accepts an optional ``outcome_adjuster`` and applies
  its current delta to the resolved threshold (clipped to [0, 1]).
* A daily CLI (``scripts/run_outcome_adjuster.py``) walks the day's closed
  positions, recomputes per-regime streaks, and writes the new adjustments
  to Redis.

Redis schema
------------
Single hash: ``{namespace}:regime_outcome_adjustment``. Fields are
human-readable regime labels (``"trend_up"``, ``"chop"``, ``"trend_down"``,
or numeric fallbacks like ``"label_0.50"``); values are float-as-string
deltas, clipped to ``[-max_adjustment, +max_adjustment]``. The hash has NO
TTL — adjustments are slow-moving operator state, not session data.

Streak math
-----------
Streaks are recomputed each run from the day's closed positions, oldest →
newest. The current adjustment is mutated incrementally:

* every ``losses_to_raise`` (default 3) consecutive losses bumps the
  adjustment by ``+per_event_delta`` (default 0.01), clipped to ``+max``.
* every ``wins_to_relax`` (default 5) consecutive wins bumps the
  adjustment by ``-per_event_delta``, clipped to ``-max``. The natural
  case is for adjustments to converge back toward zero on a winning run.
* a winning trade resets the loss streak; a losing trade resets the win
  streak. Boundaries are crossed only when the streak length passes a new
  multiple of N (or M).

Worked example (per_event_delta=0.01, max=0.05, losses_to_raise=3,
wins_to_relax=5)::

    starting adjustment for "high_vol": 0.000
    losses: 3 in a row     →  +0.010   (crossed 1st multiple)
    losses: 6 in a row     →  +0.020   (crossed 2nd multiple)
    losses: 9 in a row     →  +0.030
    wins:   5 in a row     →  +0.020   (relaxed by one per_event_delta)
    wins:  10 in a row     →  +0.010   (and another)
    wins:  15 in a row     →   0.000   (back to neutral)
    losses: 3 again        →  +0.010

The streak is per-regime-label, NOT per-symbol. Two symbols both in
``"trend_down"`` share the same Redis field — that's intentional: the
underlying market regime is the unit we want to learn about, not the
specific symbol bot. (When the operator wants per-symbol isolation, the
upstream regime store should encode it in the label; the adjuster only
cares about the label string it sees.)

Failure modes
-------------
* Corrupted Redis hash value (non-numeric, NaN, Inf) → read returns 0.0
  with WARN log; the bracket clip on write means a subsequent write
  rewrites a clean value.
* ``redis.exceptions.RedisError`` raised on read inside :class:`RegimeLookup`
  is caught at the lookup layer and falls back to delta=0.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Callable, Dict, Iterable, Optional


LOGGER = logging.getLogger(__name__)

DEFAULT_HASH_KEY = "regime_outcome_adjustment"

# Stable string labels for the v0 numeric regime labels written by
# ``regime_memory.backfill`` (0.0 = trend_down, 1.0 = chop, 2.0 = trend_up).
# Anything outside this set falls back to a stable numeric tag so the
# adjuster doesn't lose data when the upstream label scheme evolves.
_NUMERIC_LABEL_MAP: Dict[float, str] = {
    0.0: "trend_down",
    1.0: "chop",
    2.0: "trend_up",
}


def normalize_label(raw: Any) -> Optional[str]:
    """Coerce a regime label value (float, int, or string) to a stable string.

    Returns None when ``raw`` is None, NaN, or otherwise unrepresentable.
    Strings are returned trimmed; numerics are mapped to the v0 string
    aliases when they're a recognised whole number, else to a
    ``"label_<f.2f>"`` tag so distinct numeric labels don't collide.
    """
    if raw is None:
        return None
    # Allow already-stringified labels to pass through (operator-edited
    # adjuster keys, future v1 schemes that emit strings, etc.).
    if isinstance(raw, str):
        s = raw.strip()
        return s or None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    # Map exact whole-number labels to the v0 names. We accept a small
    # epsilon so a closest-neighbor lookup that returns 0.999999 still
    # snaps to ``"trend_up"`` rather than producing a unique tag.
    if abs(v - round(v)) < 1e-6:
        rounded = round(v)
        if float(rounded) in _NUMERIC_LABEL_MAP:
            return _NUMERIC_LABEL_MAP[float(rounded)]
    return f"label_{v:.2f}"


class OutcomeAdjuster:
    """Per-regime threshold adjustment store.

    State lives in a single Redis hash so all fields can be read in one
    HGETALL on the lookup hot path. Atomic writes use ``MULTI/EXEC`` so an
    aborted process leaves the hash in a consistent state.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        namespace: str = "autopilot",
        hash_key: str = DEFAULT_HASH_KEY,
        max_adjustment: float = 0.05,
        losses_to_raise: int = 3,
        wins_to_relax: int = 5,
        per_event_delta: float = 0.01,
    ) -> None:
        if max_adjustment < 0:
            raise ValueError(
                f"max_adjustment must be >= 0, got {max_adjustment!r}"
            )
        if losses_to_raise < 1:
            raise ValueError(
                f"losses_to_raise must be >= 1, got {losses_to_raise!r}"
            )
        if wins_to_relax < 1:
            raise ValueError(
                f"wins_to_relax must be >= 1, got {wins_to_relax!r}"
            )
        if per_event_delta < 0:
            raise ValueError(
                f"per_event_delta must be >= 0, got {per_event_delta!r}"
            )

        self._redis = redis_client
        self.namespace = namespace
        self._hash_key = hash_key
        self.max_adjustment = float(max_adjustment)
        self.losses_to_raise = int(losses_to_raise)
        self.wins_to_relax = int(wins_to_relax)
        self.per_event_delta = float(per_event_delta)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # key helpers
    # ------------------------------------------------------------------
    @property
    def full_hash_key(self) -> str:
        """Return the fully-namespaced Redis hash key."""
        return f"{self.namespace}:{self._hash_key}"

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def _clip(self, value: float) -> float:
        """Bracket clip a delta into ``[-max_adjustment, +max_adjustment]``."""
        if not math.isfinite(value):
            return 0.0
        return max(-self.max_adjustment, min(self.max_adjustment, float(value)))

    def current_adjustment(self, label: str) -> float:
        """Return the current delta for ``label``. 0.0 when missing or corrupt.

        Raises :class:`redis.exceptions.RedisError` on transport failures
        (consumer responsibility — :class:`RegimeLookup` catches and falls
        back to 0.0). Non-numeric / NaN / Inf values stored in the hash are
        treated as missing and logged at WARN level so an operator can spot
        manual-edit damage.
        """
        if not label:
            return 0.0
        raw = self._redis.hget(self.full_hash_key, label)
        if raw is None:
            return 0.0
        try:
            v = float(raw)
        except (TypeError, ValueError):
            LOGGER.warning(
                "outcome_adjuster: corrupt value at %s[%s]=%r; treating as 0.0",
                self.full_hash_key,
                label,
                raw,
            )
            return 0.0
        if not math.isfinite(v):
            LOGGER.warning(
                "outcome_adjuster: non-finite value at %s[%s]=%r; treating as 0.0",
                self.full_hash_key,
                label,
                raw,
            )
            return 0.0
        return self._clip(v)

    def all_adjustments(self) -> Dict[str, float]:
        """Snapshot the entire adjustment hash. Empty dict when unset.

        Corrupted / non-finite values are skipped with a WARN log so a
        single bad row doesn't poison the digest.
        """
        raw = self._redis.hgetall(self.full_hash_key) or {}
        out: Dict[str, float] = {}
        for label, value in raw.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "outcome_adjuster: corrupt value at %s[%s]=%r; skipping",
                    self.full_hash_key,
                    label,
                    value,
                )
                continue
            if not math.isfinite(v):
                LOGGER.warning(
                    "outcome_adjuster: non-finite value at %s[%s]=%r; skipping",
                    self.full_hash_key,
                    label,
                    value,
                )
                continue
            out[str(label)] = self._clip(v)
        return out

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def _write_hash(self, mapping: Dict[str, float]) -> None:
        """Replace the hash contents with ``mapping`` atomically.

        Implementation: DEL the hash first then HSET every field inside a
        single MULTI/EXEC block. The DEL handles the case where a regime
        label that previously had a non-zero adjustment no longer appears
        in ``mapping`` (otherwise it would leak).
        """
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.delete(self.full_hash_key)
            for label, delta in mapping.items():
                clipped = self._clip(float(delta))
                # Skip zero entries so the hash stays compact; a missing
                # field has the same semantic as a 0.0 stored field.
                if clipped == 0.0:
                    continue
                pipe.hset(self.full_hash_key, label, f"{clipped:.6f}")
            pipe.execute()

    def reset(self, label: Optional[str] = None) -> None:
        """Clear adjustments. ``None`` wipes the whole hash; else single label."""
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            if label is None:
                pipe.delete(self.full_hash_key)
            else:
                pipe.hdel(self.full_hash_key, label)
            pipe.execute()

    # ------------------------------------------------------------------
    # streak recompute
    # ------------------------------------------------------------------
    def apply_closed_positions(
        self,
        positions: Iterable[Any],
        *,
        label_resolver: Callable[[Any], Optional[str]],
    ) -> Dict[str, float]:
        """Recompute streak-driven adjustments from the day's closed positions.

        Positions are walked oldest → newest (callers SHOULD pre-sort by
        ``closed_at_utc`` ascending; if they don't, the streak math is
        still per-label but the iteration order then determines the
        interleaving). Each position's outcome (win when
        ``realized_pnl_usd > 0``, loss otherwise) extends the per-label
        win or loss streak. When the streak length crosses a new multiple
        of ``losses_to_raise`` we add ``+per_event_delta`` to that label's
        adjustment; multiples of ``wins_to_relax`` subtract one
        ``per_event_delta`` (toward zero, then negative, clipped at
        ``-max``).

        Positions for which ``label_resolver`` returns ``None`` are
        skipped (logged at INFO). Positions with ``realized_pnl_usd is
        None`` are also skipped (they aren't really closed yet).

        Returns the new full adjustments dict (post-write).
        """
        # Start from the current persisted state so adjustments accumulate
        # across daily runs. If the operator wants a clean slate they call
        # ``reset()`` first.
        new_state: Dict[str, float] = self.all_adjustments()

        # Per-label streak counters (resets on opposite outcome).
        win_streaks: Dict[str, int] = {}
        loss_streaks: Dict[str, int] = {}
        skipped = 0

        for position in positions:
            label = label_resolver(position)
            if label is None:
                skipped += 1
                LOGGER.info(
                    "outcome_adjuster: skipping position with no resolvable regime label"
                )
                continue
            pnl = getattr(position, "realized_pnl_usd", None)
            if pnl is None:
                skipped += 1
                LOGGER.info(
                    "outcome_adjuster: skipping position %r with no realized_pnl_usd",
                    getattr(position, "position_id", "<unknown>"),
                )
                continue
            try:
                pnl_f = float(pnl)
            except (TypeError, ValueError):
                skipped += 1
                continue

            if pnl_f > 0.0:
                # Winner — extend win streak, reset loss streak.
                loss_streaks[label] = 0
                new_count = win_streaks.get(label, 0) + 1
                win_streaks[label] = new_count
                # Crossed a new multiple of wins_to_relax?
                if (
                    new_count >= self.wins_to_relax
                    and new_count % self.wins_to_relax == 0
                ):
                    current = new_state.get(label, 0.0)
                    new_state[label] = self._clip(current - self.per_event_delta)
            else:
                # Loss or even-money — extend loss streak. Zero PnL is a
                # tie, treated as a loss for streak purposes (we never
                # learn from break-even closes; if the bot is in a
                # break-even-only regime the threshold should still creep
                # up to suppress entries).
                win_streaks[label] = 0
                new_count = loss_streaks.get(label, 0) + 1
                loss_streaks[label] = new_count
                if (
                    new_count >= self.losses_to_raise
                    and new_count % self.losses_to_raise == 0
                ):
                    current = new_state.get(label, 0.0)
                    new_state[label] = self._clip(current + self.per_event_delta)

        if skipped:
            LOGGER.info(
                "outcome_adjuster: skipped %d position(s) during apply", skipped
            )

        self._write_hash(new_state)
        # all_adjustments() reads back through the clip + corruption
        # filter so the returned dict mirrors what a future read will see.
        return self.all_adjustments()

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"OutcomeAdjuster(namespace={self.namespace!r}, "
            f"max={self.max_adjustment}, "
            f"losses_to_raise={self.losses_to_raise}, "
            f"wins_to_relax={self.wins_to_relax}, "
            f"per_event_delta={self.per_event_delta})"
        )


__all__ = [
    "OutcomeAdjuster",
    "DEFAULT_HASH_KEY",
    "normalize_label",
]
