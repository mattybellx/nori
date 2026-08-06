"""Confidence-based dynamic model routing (cost/latency optimization).

Rationale: for an orchestration engine the relevant lever is choosing *which
model / how much effort* to spend per prompt — the same adaptive allocation
principle as compute-optimal test-time compute (Snell et al. 2024). Escalation
is driven by verifier confidence, not by the agent's self-reported confidence
(per the honesty rules: never trust unmeasured confidence).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import EffortTier, RouterConfig


@dataclass
class RouteDecision:
    model: str
    tier: EffortTier
    escalated: bool
    reason: str


class Router:
    """Start cheap; escalate to the expensive tier when confidence is low."""

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._escalations = 0

    def reset(self) -> None:
        self._escalations = 0

    def decide(self, probe_score: float) -> RouteDecision:
        cfg = self._config
        if (
            probe_score < cfg.escalate_confidence_below
            and self._escalations < cfg.max_escalations
        ):
            self._escalations += 1
            return RouteDecision(
                model=cfg.expensive.name,
                tier=EffortTier.HIGH,
                escalated=True,
                reason=f"probe_score={probe_score:.2f} below {cfg.escalate_confidence_below}",
            )
        return RouteDecision(
            model=cfg.cheap.name,
            tier=EffortTier.LOW,
            escalated=False,
            reason=f"probe_score={probe_score:.2f} sufficient or escalation budget spent",
        )

    @property
    def escalation_count(self) -> int:
        return self._escalations
