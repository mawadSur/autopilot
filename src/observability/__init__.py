"""Observability hooks for the autopilot live trader.

This package wires Sentry error tracking and a Prometheus push-gateway
metrics pipeline behind thin, optional hooks. Both surfaces are env-var
toggled; if the relevant env var is unset, the helpers degrade to safe
no-ops so the trader continues to run with zero monitoring overhead.

Public API:
    * :func:`init_sentry` - idempotent Sentry SDK bootstrap.
    * :class:`MetricsPusher` - cached Prometheus collector + push-gateway.
    * :class:`Metric` - structured handoff between supervisor and pusher.
"""

from __future__ import annotations

from .monitoring import Metric, MetricsPusher, init_sentry

__all__ = ["Metric", "MetricsPusher", "init_sentry"]
