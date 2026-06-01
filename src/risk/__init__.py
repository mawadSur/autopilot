"""Hard safety gates between model decisions and exchange execution.

This package contains *circuit breakers* — binary go/no-go gates that sit
downstream of the soft-penalty risk layer in
``src.risk_management_agent.risk_engine``. The soft layer shrinks position
size based on Kelly + correlation + liquidity penalties; the breakers in
this module can override any allow-trade verdict and force a halt.

Public API:
    - ``CircuitBreakerSet`` — container holding the configured breakers.
    - ``CircuitBreakerVerdict`` — pydantic model returned by ``check``.
    - ``DecisionContext`` — pydantic snapshot of the trade + portfolio
      state, evaluated against each breaker.
"""
from __future__ import annotations

from .circuit_breakers import (
    CircuitBreakerSet,
    CircuitBreakerVerdict,
    DecisionContext,
)

__all__ = [
    "CircuitBreakerSet",
    "CircuitBreakerVerdict",
    "DecisionContext",
]
