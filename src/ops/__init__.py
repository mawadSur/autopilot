"""Operational hardening utilities for the live trader.

Modules under this package provide cron-friendly + on-demand operator tools:

* :mod:`ops.reconciliation` — compare ``PositionStore`` against the exchange
  and surface drift, orphans, and ghost positions.
"""
