"""LLM protocol and the calibrated MockLLM.

The MockLLM makes benchmarks deterministic and strategy comparisons causal:

- It is calibrated to a ``ModelConfig.step_accuracy``: each reasoning step is
  produced correctly with probability ``p`` (and otherwise a distractor).
- It encodes the evidence-backed mechanism that **external feedback enables
  targeted repair** (Huang et al. 2023): a step marked wrong via a
  ``FEEDBACK: step k incorrect`` line is re-attempted with probability
  ``p_fix`` (>= p). Without such external signal, retries do NOT improve —
  which is exactly the empirical finding that intrinsic self-correction fails.

Modes (selected via the system prompt, keeping the ``LLM`` protocol uniform):

- full solution: ``TASK_ID: <id>`` (optionally with ``FEEDBACK: ...`` lines)
- next step for search: ``MODE: next_step`` + ``PREFIX: <k>``
- judge/value function: ``MODE: judge`` + ``DRAFT: <answer>``

Token counts and latencies are simulated (seeded) so cost/latency telemetry is
meaningful and reproducible.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol

from .config import ModelConfig
from .environment import Task


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    model: str
    reasoning: str = ""  # hidden chain-of-thought from reasoning models


class LLM(Protocol):
    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Completion: ...


_SYSTEM_TASK_RE = re.compile(r"TASK_ID:\s*(\S+)")
_PREFIX_RE = re.compile(r"PREFIX:\s*(\d+)")
_FEEDBACK_RE = re.compile(r"FEEDBACK:\s*step\s+(\d+)\s+incorrect", re.IGNORECASE)


class MockLLM:
    """Deterministic, calibrated stand-in for a real model."""

    def __init__(
        self,
        catalog: dict[str, Task],
        models: dict[str, ModelConfig],
        seed: int = 0,
        p_fix: float = 0.9,
    ) -> None:
        self._catalog = catalog
        self._models = models
        self._p_fix = p_fix
        self._rng = random.Random(seed)
        self.calls = 0

    def reset(self, seed_str: str) -> None:
        """Re-seed the RNG deterministically.

        Used by the harness to give every (agent, task) pair the *same* RNG
        stream, so paired comparisons are valid: each strategy faces identical
        stochastic draws on the same task.
        """
        self._rng = random.Random(seed_str)
        self.calls = 0

    # -- internals ----------------------------------------------------------
    def _task(self, system: str) -> Task | None:
        match = _SYSTEM_TASK_RE.search(system)
        if not match:
            return None
        return self._catalog.get(match.group(1))

    def _sample_step(self, task: Task, index: int, p: float) -> str:
        """Return the correct step value with prob ``p``, else a distractor.

        Defensive: if ``index`` is outside the task's step range (possible only
        when caller and mock disagree on the catalog), return a non-matching
        placeholder instead of crashing.
        """
        if not (0 <= index < task.num_steps):
            return f"<out-of-range:{index}>"
        if self._rng.random() < p:
            return task.correct_steps[index]
        return self._rng.choice(task.distractors[index])

    def _full_solution(self, task: Task, cfg: ModelConfig, system: str) -> str:
        feedback = {int(m) for m in _FEEDBACK_RE.findall(system)}
        steps: list[str] = []
        for i in range(task.num_steps):
            p = self._p_fix if i in feedback else cfg.step_accuracy
            steps.append(self._sample_step(task, i, p))
        answer = "".join(steps)
        lines = [f"STEP {i}: {v}" for i, v in enumerate(steps)]
        return "\n".join(lines) + f"\nANSWER: {answer}"

    def _next_step(self, task: Task, prefix: int, cfg: ModelConfig, system: str) -> str:
        feedback = {int(m) for m in _FEEDBACK_RE.findall(system)}
        p = self._p_fix if prefix in feedback else cfg.step_accuracy
        value = self._sample_step(task, prefix, p)
        return f"STEP {prefix}: {value}"

    def _judge(self, task: Task, draft: str, cfg: ModelConfig) -> str:
        answer_match = re.search(r"ANSWER:\s*(\S+)", draft)
        answer = answer_match.group(1) if answer_match else draft.strip()
        correct = answer == task.gold
        true_score = 1.0 if correct else 0.0
        if self._rng.random() >= cfg.judge_accuracy:
            true_score = 1.0 - true_score
        noisy = min(1.0, max(0.0, true_score + self._rng.uniform(-0.05, 0.05)))
        return f"SCORE: {noisy:.3f}"

    def _step_judge(self, task: Task, step: str, prefix: int, cfg: ModelConfig) -> str:
        """Noisy per-step value estimate (models a real LLM judging a step)."""
        correct = (0 <= prefix < task.num_steps) and step == task.correct_steps[prefix]
        true_score = 1.0 if correct else 0.0
        if self._rng.random() >= cfg.judge_accuracy:
            true_score = 1.0 - true_score
        noisy = min(1.0, max(0.0, true_score + self._rng.uniform(-0.05, 0.05)))
        return f"SCORE: {noisy:.3f}"

    # -- public API ---------------------------------------------------------
    def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Completion:
        self.calls += 1
        system = messages[0]["content"] if messages else ""
        task = self._task(system)
        if task is None:
            text = "[mock: no task context]"
        else:
            cfg = self._models[model]
            if "MODE: judge" in system:
                draft = re.search(r"DRAFT:\s*(.+)", system, re.DOTALL)
                text = self._judge(task, draft.group(1) if draft else "", cfg)
            elif "MODE: grade" in system:
                # free-form grading (no gold available): neutral-ish score
                text = f"SCORE: {5 + self._rng.random() * 5:.1f}"
            elif "MODE: step_judge" in system:
                m = _PREFIX_RE.search(system)
                prefix = int(m.group(1)) if m else 0
                step_m = re.search(r"STEP:\s*(\S+)", system)
                text = self._step_judge(
                    task, step_m.group(1) if step_m else "", prefix, cfg
                )
            elif "MODE: next_step" in system:
                m = _PREFIX_RE.search(system)
                prefix = int(m.group(1)) if m else 0
                text = self._next_step(task, prefix, cfg, system)
            else:
                text = self._full_solution(task, cfg, system)
        tokens_in = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        tokens_out = max(1, len(text) // 4)
        latency_s = round(0.03 + self._rng.random() * 0.25, 4)
        return Completion(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=latency_s,
            model=model,
        )


def default_models() -> dict[str, ModelConfig]:
    """The two model tiers used by the engine/router in benchmarks."""
    from .config import RouterConfig

    router = RouterConfig()
    return {"cheap": router.cheap, "expensive": router.expensive}
