"""Loss postmortem swarm — Lane E foundation.

This package houses the forensics swarm that investigates losing trades.
The base class + finding dataclass live in :mod:`.base`; the five specialist
agents (signal / execution / sizing / context / process) will land in their
own modules in the next round (E4-E8 of the plan) and the synthesizer
glue lands in E9.
"""

from __future__ import annotations

from .base import BaseForensicsAgent, ForensicsFinding

__all__ = ["BaseForensicsAgent", "ForensicsFinding"]
