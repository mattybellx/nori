"""Verifier abstraction — the linchpin of the engine.

Evidence (RESEARCH.md §6): test-time compute is only as good as the verifier it
searches against (Snell et al. 2024); process/exact signals beat weak heuristics
(Lightman et al. 2023); self-consistency voting is a cheap weak verifier
(Wang et al. 2022). This module provides pluggable, composable verifiers and an
aggregator. The aggregate score drives compute allocation, search value
functions, and routing.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Protocol

from .environment import Environment, Task
from .llm import LLM


def extract_answer(text: str) -> str | None:
    """Pull the final ``ANSWER: <value>`` from a solution text."""
    match = re.search(r"ANSWER:\s*(\S+)", text)
    return match.group(1) if match else None


@dataclass
class Verdict:
    score: float  # 0..1
    passed: bool
    details: dict = field(default_factory=dict)


class GradeableTask(Protocol):
    """Minimal task surface every verifier needs (both ``Task`` and the
    free-form ``PromptTask`` satisfy this structurally)."""

    id: str
    prompt: str


class Verifier(Protocol):
    def score(self, answer: str, task: GradeableTask, context: dict | None = None) -> Verdict: ...


class ExactVerifier:
    """Deterministic: compare the extracted ANSWER with the task ground truth."""

    name = "exact"

    def score(self, answer: str, task: Task, context: dict | None = None) -> Verdict:
        extracted = extract_answer(answer)
        passed = extracted is not None and extracted == task.gold
        return Verdict(1.0 if passed else 0.0, passed, {"answer": extracted})


class TestVerifier:
    """AlphaCodium-style: run the task's 'tests' and use the pass rate.

    Provides per-step feedback (``failed_steps``) which strategies convert into
    ``FEEDBACK`` lines — the external signal that enables targeted repair.
    """

    name = "test"

    def __init__(self, env: Environment) -> None:
        self._env = env

    def score(self, answer: str, task: Task, context: dict | None = None) -> Verdict:
        report = self._env.run_tests(task, answer)
        return Verdict(
            report.score,
            report.passed == report.total,
            {"failed_steps": report.failed_steps, "passed": report.passed, "total": report.total},
        )


class LLMJudge:
    """LM-powered value function (LATS); noisy by design (judge_accuracy < 1)."""

    name = "llm_judge"

    def __init__(self, llm: LLM, model: str = "cheap") -> None:
        self._llm = llm
        self._model = model

    def score(self, answer: str, task: Task, context: dict | None = None) -> Verdict:
        system = f"TASK_ID: {task.id}\nMODE: judge\nDRAFT: {answer}"
        completion = self._llm.complete(
            [{"role": "system", "content": system}], model=self._model
        )
        match = re.search(r"SCORE:\s*([\d.]+)", completion.text)
        score = float(match.group(1)) if match else 0.0
        return Verdict(score, score >= 0.5, {"judge_score": score})


class SelfConsistencyVerifier:
    """Weak-but-cheap verifier: majority vote over N sampled drafts."""

    name = "self_consistency"

    def __init__(self, llm: LLM, model: str = "cheap", samples: int = 3) -> None:
        self._llm = llm
        self._model = model
        self._samples = samples

    def score(self, answer: str, task: Task, context: dict | None = None) -> Verdict:
        drafts: list[str] = []
        for _ in range(self._samples):
            system = f"TASK_ID: {task.id}\nMODE: sample"
            completion = self._llm.complete(
                [{"role": "system", "content": system}], model=self._model
            )
            draft = extract_answer(completion.text)
            if draft is not None:
                drafts.append(draft)
        if not drafts:
            return Verdict(0.0, False, {"drafts": 0})
        mode = statistics.mode(drafts)
        agreement = drafts.count(mode) / len(drafts)
        return Verdict(agreement, agreement >= 0.5, {"mode": mode, "agreement": agreement})


class RealTaskVerifier:
    """Verifier for the real (natural-language) suite: delegates to the task's
    deterministic checker and surfaces its feedback text for retries."""

    name = "real"

    def score(self, answer: str, task: Task, context: dict | None = None) -> Verdict:
        passed, feedback = task.check(answer)  # type: ignore[attr-defined]
        return Verdict(1.0 if passed else 0.0, passed, {"feedback": feedback})


class AggregateVerifier:
    """Weighted combination of multiple verifiers (with per-verifier details)."""

    name = "aggregate"

    def __init__(self, verifiers: list[Verifier], weights: list[float], pass_threshold: float = 0.5) -> None:
        if len(verifiers) != len(weights):
            raise ValueError("verifiers and weights must have equal length")
        self._verifiers = verifiers
        self._weights = weights
        self._pass_threshold = pass_threshold

    def score(self, answer: str, task: Task, context: dict | None = None) -> Verdict:
        total = 0.0
        weight_sum = 0.0
        details: dict = {}
        for verifier, weight in zip(self._verifiers, self._weights):
            verdict = verifier.score(answer, task, context)
            details[getattr(verifier, "name", "verifier")] = verdict.score
            total += weight * verdict.score
            weight_sum += weight
        score = total / weight_sum if weight_sum else 0.0
        return Verdict(score, score >= self._pass_threshold, details)
