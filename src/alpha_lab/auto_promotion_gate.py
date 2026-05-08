"""Auto-promotion gate — emit human-review candidates when rolling rank-IC > threshold.

Phase 4 / E2 commit 2. The gate consumes :class:`CorrelationResult` objects
from :class:`alpha_lab.correlation_miner.CorrelationMiner` (typically nightly)
and maintains a rolling 30-sample window per :class:`FeaturePair`. When a
pair's mean ``|rank_ic|`` exceeds ``threshold_rank_ic`` over at least
``min_samples`` observations, the gate emits a :class:`PromotionCandidate`
into a JSONL queue at ``runs/alpha_lab/promotion_queue.jsonl`` for human
review.

Design choices (mirroring :mod:`calibration_agent.outcome_weight_adjuster`):

* **Conservative — never auto-applies.** The gate only emits candidates;
  promotion into the live feature pipeline is the operator's responsibility.
* **Bounded mutations + audit log.** Every promotion is appended to the
  JSONL queue atomically (write-tmp-then-rename so a kill mid-write can
  never corrupt the file). Anything more invasive needs a follow-up PR.
* **Storage seam, not a hard dependency.** State lives in Redis when a
  ``redis_url`` (or ``redis_client``) is provided; otherwise the gate keeps
  an in-memory deque per pair. Tests use the in-memory path; production
  passes a real ``redis://`` URL so multiple nightly-runner processes share
  history.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
)

from alpha_lab.correlation_miner import CorrelationResult, FeaturePair

LOGGER = logging.getLogger(__name__)


# Default rolling window size. 30 nightly samples ~ 30 days of history at one
# run per day. Below this count the gate refuses to promote — conservative,
# avoids promoting on a few lucky observations.
DEFAULT_MIN_SAMPLES: int = 30

# Default threshold. CEO plan calls for ``rank-IC > 0.05 over 30 days``;
# we apply it on |rank_ic| so signals that consistently anti-correlate are
# also surfaced (the sign is preserved in the candidate payload so the
# operator can decide).
DEFAULT_THRESHOLD: float = 0.05

# Hard cap on the in-memory deque. Even if min_samples is huge we don't
# want a runaway pair to consume unbounded memory.
_MAX_HISTORY: int = 1024

# Redis key prefix — namespaced so the gate doesn't collide with the
# position-store / postmortem-queue keys already living in the same Redis.
_REDIS_KEY_PREFIX: str = "alpha_lab:rank_ic_history"

# Lock backoff schedule for the promotion-queue jsonl writer. Mirrors the
# outcome-weight adjuster so the discipline is consistent across modules.
_LOCK_BACKOFF_S: Tuple[float, ...] = (0.010, 0.020, 0.040, 0.080, 0.160)


__all__ = [
    "AutoPromotionGate",
    "DEFAULT_MIN_SAMPLES",
    "DEFAULT_THRESHOLD",
    "PromotionCandidate",
]


@dataclass(frozen=True)
class PromotionCandidate:
    """Surfaced when a pair's rolling |rank_ic| crosses the threshold.

    Attributes:
        pair: the :class:`FeaturePair` being recommended for promotion.
        rank_ic_30d_avg: signed mean of ``rank_ic`` across the rolling
            window (NOT ``|rank_ic|``). The threshold was applied on the
            absolute value but we report the sign so the operator knows
            whether to interpret the pair as a co-mover or anti-mover.
        rank_ic_30d_count: number of samples in the rolling window.
        first_seen_utc: ISO-8601 timestamp of the oldest sample.
        last_seen_utc: ISO-8601 timestamp of the most-recent sample.
    """

    pair: FeaturePair
    rank_ic_30d_avg: float
    rank_ic_30d_count: int
    first_seen_utc: str
    last_seen_utc: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict, used by the audit-log writer."""
        return {
            "pair": {
                "feature_a": self.pair.feature_a,
                "feature_b": self.pair.feature_b,
                "horizon_bars": self.pair.horizon_bars,
                "asset_class_a": self.pair.asset_class_a,
                "asset_class_b": self.pair.asset_class_b,
                "stable_id": self.pair.stable_id(),
            },
            "rank_ic_30d_avg": self.rank_ic_30d_avg,
            "rank_ic_30d_count": self.rank_ic_30d_count,
            "first_seen_utc": self.first_seen_utc,
            "last_seen_utc": self.last_seen_utc,
        }


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------
class _InMemoryHistoryStore:
    """Thread-safe per-pair deque, used when no Redis is configured."""

    def __init__(self, maxlen: int = _MAX_HISTORY) -> None:
        # Each entry is a (rank_ic, computed_at_utc) tuple. We don't bother
        # capping by time because the miner caps by sample-count semantically:
        # 30 samples ~ 30 days at one nightly run. If a caller bumps the
        # cadence, they should also bump min_samples.
        self._maxlen = int(maxlen)
        self._data: Dict[str, Deque[Tuple[float, str]]] = {}
        self._pairs: Dict[str, FeaturePair] = {}
        self._lock = threading.Lock()

    def append(self, pair: FeaturePair, rank_ic: float, computed_at_utc: str) -> None:
        sid = pair.stable_id()
        with self._lock:
            if sid not in self._data:
                self._data[sid] = deque(maxlen=self._maxlen)
                self._pairs[sid] = pair
            self._data[sid].append((float(rank_ic), str(computed_at_utc)))

    def all_pairs(self) -> List[Tuple[FeaturePair, List[Tuple[float, str]]]]:
        with self._lock:
            return [
                (self._pairs[sid], list(self._data[sid]))
                for sid in self._data
            ]


class _RedisHistoryStore:
    """Redis-backed history. Each pair is a LIST under ``alpha_lab:rank_ic_history:<sid>``.

    Redis LISTs are append-only via LPUSH/RPUSH and trimmable via LTRIM, which
    is exactly the semantics we want for a fixed-size rolling buffer. We track
    pair metadata in a parallel HASH so we can rehydrate :class:`FeaturePair`
    instances on read without re-deriving them from the stable_id.
    """

    def __init__(self, redis_client: Any, prefix: str = _REDIS_KEY_PREFIX) -> None:
        self._redis = redis_client
        self._prefix = prefix.rstrip(":")

    def _list_key(self, sid: str) -> str:
        return f"{self._prefix}:{sid}"

    @property
    def _meta_key(self) -> str:
        return f"{self._prefix}:meta"

    def append(self, pair: FeaturePair, rank_ic: float, computed_at_utc: str) -> None:
        sid = pair.stable_id()
        entry = json.dumps({"rank_ic": float(rank_ic), "ts": str(computed_at_utc)})
        meta = json.dumps(
            {
                "feature_a": pair.feature_a,
                "feature_b": pair.feature_b,
                "horizon_bars": pair.horizon_bars,
                "asset_class_a": pair.asset_class_a,
                "asset_class_b": pair.asset_class_b,
            }
        )
        pipe = self._redis.pipeline()
        pipe.rpush(self._list_key(sid), entry)
        pipe.ltrim(self._list_key(sid), -_MAX_HISTORY, -1)
        pipe.hset(self._meta_key, sid, meta)
        pipe.execute()

    def all_pairs(self) -> List[Tuple[FeaturePair, List[Tuple[float, str]]]]:
        meta_blob = self._redis.hgetall(self._meta_key) or {}
        out: List[Tuple[FeaturePair, List[Tuple[float, str]]]] = []
        for sid_raw, meta_raw in meta_blob.items():
            sid = sid_raw.decode() if isinstance(sid_raw, bytes) else str(sid_raw)
            meta_str = meta_raw.decode() if isinstance(meta_raw, bytes) else str(meta_raw)
            try:
                meta = json.loads(meta_str)
            except json.JSONDecodeError:
                LOGGER.warning("alpha_lab gate: malformed meta blob for sid=%s", sid)
                continue
            entries_raw = self._redis.lrange(self._list_key(sid), 0, -1) or []
            entries: List[Tuple[float, str]] = []
            for raw in entries_raw:
                txt = raw.decode() if isinstance(raw, bytes) else str(raw)
                try:
                    parsed = json.loads(txt)
                    entries.append((float(parsed["rank_ic"]), str(parsed["ts"])))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
            pair = FeaturePair(
                feature_a=str(meta.get("feature_a", "")),
                feature_b=str(meta.get("feature_b", "")),
                horizon_bars=int(meta.get("horizon_bars", 0)),
                asset_class_a=str(meta.get("asset_class_a", "unknown")),
                asset_class_b=str(meta.get("asset_class_b", "unknown")),
            )
            out.append((pair, entries))
        return out


# ---------------------------------------------------------------------------
# JSONL queue (audit log)
# ---------------------------------------------------------------------------
class _NullLock:
    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


class _FileLock:
    """``fcntl.flock`` advisory lock with bounded retries.

    Mirrors :class:`calibration_agent.outcome_weight_adjuster._FileLock` so the
    locking discipline is consistent across the additive modules.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        try:
            import fcntl  # type: ignore[import-not-found]
        except ImportError:
            return self
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o644)
        op = fcntl.LOCK_EX | fcntl.LOCK_NB
        for delay in _LOCK_BACKOFF_S:
            try:
                fcntl.flock(self._fd, op)
                return self
            except OSError:
                time.sleep(delay)
        try:
            fcntl.flock(self._fd, op)
        except OSError as exc:
            LOGGER.warning(
                "alpha_lab gate: queue lock contended for %s after %d retries (%s); "
                "proceeding without lock",
                self._path,
                len(_LOCK_BACKOFF_S),
                exc,
            )
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._fd is None:
            return
        try:
            import fcntl  # type: ignore[import-not-found]
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - best-effort unlock
            pass
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None


def _acquire_file_lock(path: Path) -> Any:
    try:
        import fcntl  # noqa: F401
    except ImportError:
        return _NullLock()
    return _FileLock(path)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
class AutoPromotionGate:
    """Rolling-window rank-IC threshold gate.

    Construct with optional ``redis_url`` (or ``redis_client`` for tests) for
    multi-process state sharing. When neither is supplied the gate keeps an
    in-memory deque per pair.

    Args:
        threshold_rank_ic: the |rank_ic| floor a pair must clear (over the
            rolling window mean) to be emitted. CEO plan default: 0.05.
        min_samples: minimum samples in the rolling window before a pair
            is even considered. Default 30 ~ "30 days of nightly runs".
        redis_url: optional Redis connection string. When provided, history
            persists across runs in Redis LISTs.
        redis_client: alternative to ``redis_url`` — pass a pre-built client
            (typically ``fakeredis.FakeRedis()`` in tests). Mutually exclusive.
        promotion_queue_path: path to the JSONL audit log. Defaults to
            ``runs/alpha_lab/promotion_queue.jsonl`` relative to CWD.
    """

    def __init__(
        self,
        *,
        threshold_rank_ic: float = DEFAULT_THRESHOLD,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        redis_url: Optional[str] = None,
        redis_client: Any = None,
        promotion_queue_path: Optional[Path] = None,
    ) -> None:
        if threshold_rank_ic < 0:
            raise ValueError("threshold_rank_ic must be >= 0")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")

        self.threshold_rank_ic = float(threshold_rank_ic)
        self.min_samples = int(min_samples)

        if redis_client is not None:
            self._store: Any = _RedisHistoryStore(redis_client)
        elif redis_url:
            try:
                import redis  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "redis_url provided but the 'redis' package is not installed"
                ) from exc
            self._store = _RedisHistoryStore(
                redis.Redis.from_url(redis_url, decode_responses=False)
            )
        else:
            self._store = _InMemoryHistoryStore()

        self.promotion_queue_path = (
            Path(promotion_queue_path)
            if promotion_queue_path is not None
            else Path("runs/alpha_lab/promotion_queue.jsonl")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record(self, result: CorrelationResult) -> None:
        """Append a single :class:`CorrelationResult` to the rolling window.

        The gate stores both the signed ``rank_ic`` and the timestamp so
        :meth:`check_promotions` can report the window's first/last seen
        UTC times in the candidate payload.
        """
        if not isinstance(result, CorrelationResult):
            raise TypeError("record expects a CorrelationResult")
        self._store.append(result.pair, result.rank_ic, result.computed_at_utc)

    def check_promotions(self) -> List[PromotionCandidate]:
        """Scan the rolling window and emit candidates above the threshold.

        For each pair with at least ``min_samples`` observations, compute the
        mean signed ``rank_ic``. If ``|mean| > threshold``, emit a
        :class:`PromotionCandidate` and append it to the JSONL queue.

        Returns the list of candidates emitted on this call. Idempotent in
        the sense that re-calling without further :meth:`record` calls will
        re-emit the same candidates (the operator's responsibility to
        dedupe-on-consume from the JSONL).
        """
        candidates: List[PromotionCandidate] = []
        for pair, history in self._store.all_pairs():
            if len(history) < self.min_samples:
                continue
            ric_values = [r for r, _ts in history]
            mean = float(sum(ric_values) / len(ric_values))
            if abs(mean) <= self.threshold_rank_ic:
                continue
            timestamps = sorted(ts for _r, ts in history if ts)
            first = timestamps[0] if timestamps else ""
            last = timestamps[-1] if timestamps else ""
            candidate = PromotionCandidate(
                pair=pair,
                rank_ic_30d_avg=mean,
                rank_ic_30d_count=len(history),
                first_seen_utc=first,
                last_seen_utc=last,
            )
            candidates.append(candidate)

        if candidates:
            self._append_to_queue(candidates)
        return candidates

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _append_to_queue(self, candidates: List[PromotionCandidate]) -> None:
        """Append candidates to the JSONL queue atomically.

        Uses an exclusive flock on a sidecar lock file so concurrent
        nightly-runner processes don't interleave half-written lines. We
        write each candidate as a single JSON object per line.
        """
        path = self.promotion_queue_path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        ts = datetime.now(timezone.utc).isoformat()
        with _acquire_file_lock(lock_path):
            with path.open("a", encoding="utf-8") as f:
                for cand in candidates:
                    payload = {"emitted_at_utc": ts, **cand.to_dict()}
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
