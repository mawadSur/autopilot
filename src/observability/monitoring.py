"""Sentry + Prometheus push-gateway helpers.

Both surfaces are optional and env-var toggled. None of the helpers in this
module may raise for monitoring-related failures: every external call is
wrapped in ``try/except`` because a flaky observability path must never kill
the live trader.

Conventions
-----------
* Every Prometheus metric name is prefixed with ``autopilot_``.
* Collectors (Gauge / Counter / Histogram) are cached on the
  ``MetricsPusher`` keyed by ``(name, frozenset(label_keys))`` so repeated
  calls reuse the same instance. A subsequent call with a *different* set of
  label keys logs a WARNING and is dropped, because Prometheus collectors
  cannot retroactively change their label_names.
* Histograms use a default bucket layout tuned for trading-loop intervals:
  ``(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, FrozenSet, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRIC_NAME_PREFIX = "autopilot_"

# Bucket layout tuned for trading tick intervals + fill latency. Anything
# slower than ~30s on a single tick is degenerate; finer-grained sub-second
# buckets matter for fill-latency histograms.
DEFAULT_HISTOGRAM_BUCKETS: Tuple[float, ...] = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

DEFAULT_JOB_NAME = "autopilot"

# Module-level latch so init_sentry() is idempotent across repeated calls.
_SENTRY_INITIALIZED: bool = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Metric(BaseModel):
    """Structured handoff between the supervisor and the pusher.

    Used when the caller wants to construct metrics declaratively (eg out
    of a tick result) rather than calling ``gauge`` / ``counter`` /
    ``histogram`` imperatively.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sentry bootstrap
# ---------------------------------------------------------------------------


def init_sentry(
    *,
    dsn: Optional[str] = None,
    environment: str = "dev",
    release: Optional[str] = None,
) -> bool:
    """Initialize the Sentry SDK if a DSN is available.

    The DSN is taken from the ``dsn`` argument first, then falls back to
    the ``SENTRY_DSN`` env var. If neither is set, the function returns
    ``False`` and Sentry is not initialized.

    The function is idempotent -- repeat calls after a successful init are
    no-ops. Any failure during initialization is swallowed and logged at
    WARNING level: monitoring failures must never crash the trader.
    """
    global _SENTRY_INITIALIZED

    if _SENTRY_INITIALIZED:
        return True

    # Treat bare `SENTRY_DSN=` (empty string) as "not configured", same as unset.
    # This is a common .env footgun — without the strip, sentry_sdk.init parses
    # the empty string as a URL and fails with "Unsupported scheme ''".
    raw_dsn = dsn if dsn is not None else os.getenv("SENTRY_DSN")
    effective_dsn = raw_dsn.strip() if isinstance(raw_dsn, str) else raw_dsn
    if not effective_dsn:
        return False

    try:
        import sentry_sdk  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - missing dep, never crash
        LOGGER.warning("sentry_sdk import failed: %s", exc)
        return False

    try:
        sentry_sdk.init(
            dsn=effective_dsn,
            environment=environment,
            release=release,
        )
        try:
            sentry_sdk.set_tag("service", "autopilot")
        except Exception as exc:  # noqa: BLE001 - tag setting is best-effort
            LOGGER.warning("sentry_sdk.set_tag failed: %s", exc)
    except Exception as exc:  # noqa: BLE001 - never raise from monitoring
        LOGGER.warning("sentry_sdk.init failed: %s", exc)
        return False

    _SENTRY_INITIALIZED = True
    return True


def _reset_sentry_for_tests() -> None:
    """Test helper: reset the idempotency latch between tests."""
    global _SENTRY_INITIALIZED
    _SENTRY_INITIALIZED = False


# ---------------------------------------------------------------------------
# MetricsPusher
# ---------------------------------------------------------------------------


def _validate_metric_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("metric name must be a non-empty string")


def _full_name(name: str) -> str:
    return name if name.startswith(METRIC_NAME_PREFIX) else f"{METRIC_NAME_PREFIX}{name}"


def _label_key(name: str, label_keys: FrozenSet[str]) -> Tuple[str, FrozenSet[str]]:
    return (name, label_keys)


class MetricsPusher:
    """Lightweight Prometheus push-gateway client with collector caching.

    Construct one per process. If ``push_url`` (or env ``PROMETHEUS_PUSH_URL``)
    is not set, the pusher is a no-op: collector mutations are skipped and
    ``push()`` returns ``False``.

    All collectors live in a single :class:`CollectorRegistry` and are
    cached by ``(metric_name, frozenset(label_keys))`` so repeated calls
    reuse the same instance.
    """

    def __init__(
        self,
        *,
        push_url: Optional[str] = None,
        job: Optional[str] = None,
        timeout_s: float = 5.0,
        registry: Any = None,
        push_fn: Any = None,
    ) -> None:
        self.push_url: Optional[str] = push_url or os.getenv("PROMETHEUS_PUSH_URL")
        self.job: str = job or os.getenv("PROMETHEUS_PUSH_JOB") or DEFAULT_JOB_NAME
        self.timeout_s = float(timeout_s)

        # Lazy import: prometheus_client may not be installed in trimmed envs.
        # The pusher remains usable as a no-op even if the import fails.
        self._prom: Any = None
        try:
            import prometheus_client  # type: ignore[import-not-found]

            self._prom = prometheus_client
        except Exception as exc:  # noqa: BLE001 - tolerate missing dep
            LOGGER.warning("prometheus_client import failed: %s", exc)

        if registry is None and self._prom is not None:
            registry = self._prom.CollectorRegistry()
        self._registry: Any = registry

        if push_fn is None and self._prom is not None:
            push_fn = self._prom.push_to_gateway
        self._push_fn = push_fn

        # Collector caches.
        self._gauges: Dict[Tuple[str, FrozenSet[str]], Any] = {}
        self._counters: Dict[Tuple[str, FrozenSet[str]], Any] = {}
        self._histograms: Dict[Tuple[str, FrozenSet[str]], Any] = {}

        # Latches so we don't spam warnings on every tick when a metric was
        # initially registered with one set of label keys and a later call
        # tries a different set.
        self._label_mismatch_warned: set = set()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------
    def is_enabled(self) -> bool:
        """True iff the pusher has a push URL AND prometheus_client is importable."""
        return bool(self.push_url) and self._prom is not None and self._registry is not None

    # ------------------------------------------------------------------
    # Collector mutation
    # ------------------------------------------------------------------
    def gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge to ``value``.

        Use for sampled-state metrics: equity, daily PnL, open-position
        counts, etc.
        """
        _validate_metric_name(name)
        if not self.is_enabled():
            return
        try:
            collector = self._get_or_create(
                kind="gauge", name=name, labels=labels or {}
            )
            if collector is None:
                return
            if labels:
                collector.labels(**labels).set(float(value))
            else:
                collector.set(float(value))
        except Exception as exc:  # noqa: BLE001 - never crash
            LOGGER.warning("gauge(%s) failed: %s", name, exc)

    def counter(
        self,
        name: str,
        increment: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter by ``increment`` (default 1.0)."""
        _validate_metric_name(name)
        if not self.is_enabled():
            return
        try:
            collector = self._get_or_create(
                kind="counter", name=name, labels=labels or {}
            )
            if collector is None:
                return
            if labels:
                collector.labels(**labels).inc(float(increment))
            else:
                collector.inc(float(increment))
        except Exception as exc:  # noqa: BLE001 - never crash
            LOGGER.warning("counter(%s) failed: %s", name, exc)

    def histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe ``value`` into the histogram. Buckets are the trading-loop default."""
        _validate_metric_name(name)
        if not self.is_enabled():
            return
        try:
            collector = self._get_or_create(
                kind="histogram", name=name, labels=labels or {}
            )
            if collector is None:
                return
            if labels:
                collector.labels(**labels).observe(float(value))
            else:
                collector.observe(float(value))
        except Exception as exc:  # noqa: BLE001 - never crash
            LOGGER.warning("histogram(%s) failed: %s", name, exc)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------
    def push(self) -> bool:
        """Push the registry to the configured gateway.

        Returns ``True`` on success, ``False`` if disabled or if the push
        failed for any reason. Failure is logged at WARNING level and
        never re-raised.
        """
        if not self.is_enabled() or self._push_fn is None:
            return False
        try:
            self._push_fn(
                self.push_url,
                job=self.job,
                registry=self._registry,
                timeout=self.timeout_s,
            )
            return True
        except TypeError:
            # Some push_fn stubs / older signatures don't accept ``timeout``.
            try:
                self._push_fn(
                    self.push_url,
                    job=self.job,
                    registry=self._registry,
                )
                return True
            except Exception as exc:  # noqa: BLE001 - never crash
                LOGGER.warning("metrics push to %s failed: %s", self.push_url, exc)
                return False
        except Exception as exc:  # noqa: BLE001 - never crash
            LOGGER.warning("metrics push to %s failed: %s", self.push_url, exc)
            return False

    # ------------------------------------------------------------------
    # Internal: collector cache
    # ------------------------------------------------------------------
    def _get_or_create(
        self,
        *,
        kind: str,
        name: str,
        labels: Dict[str, str],
    ) -> Any:
        """Lookup or register a Gauge/Counter/Histogram in the registry.

        Returns ``None`` if the call provided a different set of label_keys
        than the originally registered collector (logged once per metric).
        """
        if self._prom is None or self._registry is None:
            return None

        full_name = _full_name(name)
        label_keys = frozenset(labels.keys())
        cache = {
            "gauge": self._gauges,
            "counter": self._counters,
            "histogram": self._histograms,
        }[kind]

        key = _label_key(full_name, label_keys)
        if key in cache:
            return cache[key]

        # Detect "same name, different label keys" -- a Prometheus error.
        existing_keys_for_name = [
            existing for existing in cache if existing[0] == full_name
        ]
        if existing_keys_for_name:
            if full_name not in self._label_mismatch_warned:
                LOGGER.warning(
                    "metric %s already registered with labels %s; "
                    "dropping call with labels %s",
                    full_name,
                    sorted(existing_keys_for_name[0][1]),
                    sorted(label_keys),
                )
                self._label_mismatch_warned.add(full_name)
            return None

        label_names = sorted(labels.keys())
        if kind == "gauge":
            collector = self._prom.Gauge(
                full_name, full_name, labelnames=label_names, registry=self._registry
            )
        elif kind == "counter":
            collector = self._prom.Counter(
                full_name, full_name, labelnames=label_names, registry=self._registry
            )
        else:  # histogram
            collector = self._prom.Histogram(
                full_name,
                full_name,
                labelnames=label_names,
                buckets=DEFAULT_HISTOGRAM_BUCKETS,
                registry=self._registry,
            )
        cache[key] = collector
        return collector


# ---------------------------------------------------------------------------
# Sentry helpers (best-effort, never raise)
# ---------------------------------------------------------------------------


def breadcrumb(
    *,
    category: str,
    message: str,
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a Sentry breadcrumb. No-op if Sentry isn't configured.

    Used by the supervisor for tick start, order placed, fill received,
    breaker decision, and daily close. The breadcrumb call must NEVER
    raise — observability must not crash the trader.
    """
    try:
        import sentry_sdk  # type: ignore[import-not-found]

        sentry_sdk.add_breadcrumb(
            category=category,
            message=message,
            level=level,
            data=data or {},
        )
    except Exception:  # noqa: BLE001 - never raise from monitoring
        pass


def capture_message(message: str, level: str = "warning") -> None:
    """Forward a message to Sentry. Best-effort, never raises."""
    try:
        import sentry_sdk  # type: ignore[import-not-found]

        sentry_sdk.capture_message(message, level=level)
    except Exception:  # noqa: BLE001 - never raise from monitoring
        pass


__all__ = [
    "DEFAULT_HISTOGRAM_BUCKETS",
    "DEFAULT_JOB_NAME",
    "METRIC_NAME_PREFIX",
    "Metric",
    "MetricsPusher",
    "breadcrumb",
    "capture_message",
    "init_sentry",
]
