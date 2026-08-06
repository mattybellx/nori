"""Typed telemetry records emitted by every agent run.

These records are the raw material for the benchmark harness and for honest
reporting (latency, tokens, attempts, verifier scores) per the Master Brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepEvent:
    """A single step inside an agent run (thought/action/observation/...)."""

    kind: str                       # "thought" | "action" | "observation" | "verify" | "reflect" | "route"
    content: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    model: str = ""


@dataclass
class RunResult:
    """Outcome of one agent run over one task."""

    task_id: str
    strategy: str
    success: bool
    answer: str
    verifier_score: float = 0.0
    verifier_meta: dict = field(default_factory=dict)
    steps: list[StepEvent] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    route_tier: str = "low"

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out
