"""Redis-backed trade context snapshot store (Lane E foundation, E1).

The :class:`TradeContextStore` records per-trade snapshots at three lifecycle
moments — ``signal``, ``fill``, and ``breaker`` — so the future loss-postmortem
forensics swarm can reconstruct WHY a trade went the way it did without
scraping logs. Each snapshot lives at::

    {ns}:trade_ctx:{trade_id}:{phase}

with a 30-day TTL. Keys live in their own ``trade_ctx`` namespace so they
don't collide with :class:`PositionStore`'s ``positions:`` / ``open_set`` /
``closed:`` / ``errors:by_symbol:`` keys but reuse the same Redis client.

The companion :class:`PostmortemQueue` Protocol + :class:`RedisPostmortemQueue`
impl let :class:`PositionStore.record_close` enqueue trades that should be
investigated by the swarm (D5 trigger gate, E2). The queue is a Redis LIST
keyed at ``{ns}:postmortem:queue`` so a worker can ``BRPOP`` it.

NaN / Inf serialization
-----------------------
Python's :mod:`json` raises on NaN/Inf when ``allow_nan=False`` and silently
emits invalid JSON otherwise. We pick the third option: convert NaN/Inf to
``None`` on the way in (with a sentinel field-name suffix in the stored blob
to make round-trip explicit if any reader cares to recover the marker).
The chosen pattern: NaN/Inf round-trip as ``None``. Forensics agents must
treat ``None`` in feature_buffer as "feature missing or non-finite" — which
is exactly the semantic the agents need (they care that a value was bad,
not whether it was specifically NaN vs +Inf).
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Protocol

LOGGER = logging.getLogger(__name__)

DEFAULT_NAMESPACE = "autopilot"
DEFAULT_TTL_SECONDS = 30 * 24 * 3600  # 30 days
TRADE_CTX_KEY_PREFIX = "trade_ctx"
POSTMORTEM_QUEUE_KEY = "postmortem:queue"

SnapshotPhase = Literal["signal", "fill", "breaker", "close"]
_VALID_PHASES: tuple[str, ...] = ("signal", "fill", "breaker", "close")


# ---------------------------------------------------------------------------
# JSON sanitisation
# ---------------------------------------------------------------------------


def _sanitize(value: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON safety.

    Tuples are converted to lists (JSON has no tuple). Sets are converted to
    sorted lists. Anything else is returned as-is — Pydantic models inside
    the snapshot's ``risk_metrics_*`` dicts must be pre-dumped by the caller.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, set):
        return [_sanitize(v) for v in sorted(value, key=str)]
    return value


# ---------------------------------------------------------------------------
# Snapshot dataclass
# ---------------------------------------------------------------------------


@dataclass
class TradeContextSnapshot:
    """A single moment-in-time snapshot of trade context.

    The store keys snapshots by ``(trade_id, phase)`` so a single trade can
    have up to four snapshots over its lifecycle: ``signal`` (decision time,
    BEFORE order placement), ``fill`` (after the exchange confirms the fill),
    ``breaker`` (a circuit-breaker forced exit), and ``close`` (the final
    book-closing snapshot, optional — usually fill or breaker is enough).

    Phase-16 canonical fields
    -------------------------
    The three optional ``kill_switch_reason``, ``stop_loss_trigger_price``,
    and ``breaker_decision`` fields are the documented seam the Process-
    Integrity (A5) and Execution-Forensics (A2) agents read instead of
    substring-scanning ``breaker_context`` / ``notes``. Pre-Phase-16
    snapshots stored without these fields deserialize cleanly because they
    default to ``None`` (Pydantic's ``cls(**data)`` resolution).

    Fill-side structured metadata (also Phase-16) lives on the position
    record, not the snapshot — see ``state.position_store.Position``.
    """

    trade_id: str
    symbol: str
    captured_at_utc: str
    phase: SnapshotPhase
    feature_buffer: Dict[str, float] = field(default_factory=dict)
    feature_window: Optional[List[Dict[str, float]]] = None
    model_probs: Dict[str, float] = field(default_factory=dict)
    model_confidence: float = 0.0
    risk_metrics_input: Dict[str, Any] = field(default_factory=dict)
    risk_metrics_output: Dict[str, Any] = field(default_factory=dict)
    breaker_context: Dict[str, Any] = field(default_factory=dict)
    ticker_buffer: List[Dict[str, float]] = field(default_factory=list)
    notes: Optional[str] = None
    # Phase-16 canonical breaker fields. A5 ProcessIntegrityAgent prefers
    # these over substring-matching ``breaker_context``/``notes``. Default
    # to None so legacy snapshots round-trip unchanged.
    kill_switch_reason: Optional[str] = None
    stop_loss_trigger_price: Optional[float] = None
    breaker_decision: Optional[str] = None
    # Sprint 2.6: regime label resolved by the predictor's regime_lookup at
    # signal time. The ``OutcomeAdjuster`` resolver (per
    # ``scripts/run_outcome_adjuster.py`` doc lines 15-22) treats this
    # top-level key as the canonical seam; today its concrete impl also
    # probes ``risk_metrics_input["regime_label"]`` (belt-and-suspenders the
    # supervisor mirrors). Defaults to None so legacy snapshots round-trip
    # cleanly through ``cls(**data)``.
    regime_label: Optional[str] = None

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dict (NaN/Inf → None)."""
        return _sanitize(asdict(self))

    def to_json(self) -> str:
        # ``allow_nan=False`` is belt-and-suspenders: _sanitize already replaced
        # them but if someone slipped a NaN in via a nested object that asdict
        # didn't recurse into we want to fail loudly here, not silently emit
        # invalid JSON.
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "TradeContextSnapshot":
        data = json.loads(blob)
        return cls(**data)


# ---------------------------------------------------------------------------
# PostmortemQueue protocol + Redis impl
# ---------------------------------------------------------------------------


class PostmortemQueue(Protocol):
    """Minimal interface a postmortem queue must satisfy.

    Implementations are expected to be process-safe and crash-tolerant. A
    bare in-memory list is fine for tests; Redis LPUSH is the production
    impl below.
    """

    def enqueue(self, trade_id: str) -> None:
        ...


class RedisPostmortemQueue:
    """Concrete :class:`PostmortemQueue` backed by a Redis LIST.

    Trade IDs are LPUSHed onto ``{ns}:postmortem:queue``. A worker can BRPOP
    the list to pop the oldest trade for investigation (FIFO).
    """

    def __init__(
        self,
        *,
        redis_client: Any,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self._redis = redis_client
        self._namespace = namespace
        self._lock = threading.Lock()

    @property
    def queue_key(self) -> str:
        return f"{self._namespace}:{POSTMORTEM_QUEUE_KEY}"

    def enqueue(self, trade_id: str) -> None:
        if not trade_id:
            raise ValueError("trade_id must be a non-empty string")
        with self._lock:
            try:
                self._redis.lpush(self.queue_key, str(trade_id))
            except Exception as exc:  # noqa: BLE001 - queue is best-effort
                LOGGER.warning(
                    "postmortem queue lpush failed for trade_id=%s: %r",
                    trade_id,
                    exc,
                )

    def queue_length(self) -> int:
        try:
            return int(self._redis.llen(self.queue_key))
        except Exception:  # noqa: BLE001 - tolerate flaky stubs
            return 0


# ---------------------------------------------------------------------------
# TradeContextStore
# ---------------------------------------------------------------------------


class TradeContextStore:
    """Redis-backed snapshot store for trade lifecycle context.

    Construction mirrors :class:`PositionStore`: pass either a ``redis_url``
    or a pre-built ``redis_client`` (tests pass ``fakeredis.FakeRedis()``).
    Keys are namespaced and prefixed so they don't collide with
    :class:`PositionStore` keys living in the same Redis instance.

    The default ``ttl_seconds`` is 30 days — postmortem agents need the full
    snapshot for as long as a postmortem could plausibly be re-run. After
    30 days Redis evicts automatically and the trade context is gone.
    """

    def __init__(
        self,
        *,
        redis_client: Optional[Any] = None,
        redis_url: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.namespace = namespace
        self.ttl_seconds = int(ttl_seconds)
        self._lock = threading.Lock()

        if redis_client is not None:
            self._redis = redis_client
        else:
            import redis  # type: ignore[import-not-found]

            url = redis_url or "redis://localhost:6379/0"
            self._redis = redis.Redis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------------
    # key helpers
    # ------------------------------------------------------------------
    def _key(self, trade_id: str, phase: str) -> str:
        return f"{self.namespace}:{TRADE_CTX_KEY_PREFIX}:{trade_id}:{phase}"

    def _scan_pattern(self, trade_id: str) -> str:
        return f"{self.namespace}:{TRADE_CTX_KEY_PREFIX}:{trade_id}:*"

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def record_snapshot(self, snap: TradeContextSnapshot) -> None:
        """Atomically write a snapshot with TTL applied.

        If the same ``(trade_id, phase)`` key already exists it's overwritten.
        Practical case: fill snapshots can be re-recorded if the fill price
        is amended after partial fills.
        """
        if snap.phase not in _VALID_PHASES:
            raise ValueError(
                f"snapshot.phase must be one of {_VALID_PHASES!r}, got {snap.phase!r}"
            )
        if not snap.trade_id:
            raise ValueError("snapshot.trade_id must be a non-empty string")

        blob = snap.to_json()
        key = self._key(snap.trade_id, snap.phase)
        with self._lock:
            pipe = self._redis.pipeline()
            pipe.multi()
            pipe.set(key, blob)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def get_snapshots(self, trade_id: str) -> Dict[str, TradeContextSnapshot]:
        """Return all phases recorded for ``trade_id`` keyed by phase."""
        out: Dict[str, TradeContextSnapshot] = {}
        keys = list(self._redis.scan_iter(match=self._scan_pattern(trade_id)))
        for key in keys:
            blob = self._redis.get(key)
            if blob is None:
                continue
            try:
                snap = TradeContextSnapshot.from_json(blob)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                LOGGER.warning(
                    "trade_ctx: skipping unparseable snapshot at %s: %r",
                    key,
                    exc,
                )
                continue
            out[snap.phase] = snap
        return out

    def get_snapshot(
        self, trade_id: str, phase: str
    ) -> Optional[TradeContextSnapshot]:
        """Fetch a single (trade_id, phase) snapshot or return None."""
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"phase must be one of {_VALID_PHASES!r}, got {phase!r}"
            )
        blob = self._redis.get(self._key(trade_id, phase))
        if blob is None:
            return None
        try:
            return TradeContextSnapshot.from_json(blob)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning(
                "trade_ctx: corrupt snapshot for %s/%s: %r", trade_id, phase, exc
            )
            return None

    def get_signal_snapshot(
        self, trade_id: str
    ) -> Optional[TradeContextSnapshot]:
        return self.get_snapshot(trade_id, "signal")

    def get_fill_snapshot(
        self, trade_id: str
    ) -> Optional[TradeContextSnapshot]:
        return self.get_snapshot(trade_id, "fill")

    def get_ttl(self, trade_id: str, phase: str) -> int:
        """Return remaining TTL in seconds for the given snapshot key.

        Returns ``-2`` when the key is missing, ``-1`` when no expiry is set
        (mirrors Redis ``TTL`` semantics).
        """
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"phase must be one of {_VALID_PHASES!r}, got {phase!r}"
            )
        try:
            return int(self._redis.ttl(self._key(trade_id, phase)))
        except Exception:  # noqa: BLE001 - tolerate flaky stubs
            return -2

    # ------------------------------------------------------------------
    # deletion
    # ------------------------------------------------------------------
    def delete_snapshots(self, trade_id: str) -> int:
        """Delete every phase snapshot for ``trade_id``. Returns count deleted."""
        keys = list(self._redis.scan_iter(match=self._scan_pattern(trade_id)))
        if not keys:
            return 0
        with self._lock:
            return int(self._redis.delete(*keys))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Convenience: ISO-8601 UTC timestamp with seconds precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_NAMESPACE",
    "DEFAULT_TTL_SECONDS",
    "POSTMORTEM_QUEUE_KEY",
    "PostmortemQueue",
    "RedisPostmortemQueue",
    "SnapshotPhase",
    "TRADE_CTX_KEY_PREFIX",
    "TradeContextSnapshot",
    "TradeContextStore",
    "utc_now_iso",
]
