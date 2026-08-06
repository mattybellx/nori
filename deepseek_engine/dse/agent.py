"""Agent protocol and shared base infrastructure.

Strategies share a uniform ``solve(task, budget) -> RunResult`` interface so the
harness can compare them causally on the same suite. All LLM calls are recorded
as ``StepEvent`` telemetry (tokens, latency, model) so no measurement is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import EngineConfig, ModelConfig
from .environment import Environment, Task
from .events import RunResult, StepEvent
from .llm import LLM
from .verifier import Verdict, Verifier


@dataclass
class Budget:
    max_trials: int = 3
    max_search_nodes: int = 64
    max_tokens: int = 8192


class Agent(Protocol):
    name: str

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult: ...


def feedback_lines(failed_steps: list[int]) -> str:
    """Convert per-step test failures into the mock's external-signal syntax."""
    if not failed_steps:
        return ""
    return "\n".join(f"FEEDBACK: step {i} incorrect" for i in failed_steps)


class BaseAgent:
    """Shared plumbing for all strategies: LLM calls, telemetry, finalization."""

    name = "base"

    def __init__(
        self,
        llm: LLM,
        verifier: Verifier,
        config: EngineConfig,
        models: dict[str, ModelConfig],
        env: Environment | None = None,
    ) -> None:
        self.llm = llm
        self.verifier = verifier
        self.config = config
        self.models = models
        self.env = env

    # -- helpers ------------------------------------------------------------
    def _system(self, task: Task, extra: str = "") -> str:
        return f"TASK_ID: {task.id}\n{extra}".rstrip()

    def _call(
        self,
        run: RunResult,
        kind: str,
        system: str,
        user: str,
        model: str,
    ) -> str:
        completion = self.llm.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
            max_tokens=self.config.max_tokens_per_call,
        )
        run.steps.append(
            StepEvent(
                kind=kind,
                content=completion.text,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
                latency_s=completion.latency_s,
                model=completion.model,
            )
        )
        run.tokens_in += completion.tokens_in
        run.tokens_out += completion.tokens_out
        run.latency_s += completion.latency_s
        return completion.text

    def _verify(self, run: RunResult, answer: str, task: Task) -> Verdict:
        verdict = self.verifier.score(answer, task)
        run.steps.append(
            StepEvent(
                kind="verify",
                content=f"score={verdict.score:.3f} passed={verdict.passed}",
                model="verifier",
            )
        )
        return verdict

    def _finalize(
        self,
        run: RunResult,
        task: Task,
        verdict: Verdict,
        answer: str,
        attempts: int,
    ) -> RunResult:
        run.task_id = task.id
        run.strategy = self.name
        run.success = verdict.passed
        run.answer = answer
        run.verifier_score = verdict.score
        run.verifier_meta = dict(verdict.details or {})
        run.attempts = attempts
        return run
