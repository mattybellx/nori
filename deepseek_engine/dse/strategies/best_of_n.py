"""Best-of-N with verifier selection — the canonical test-time-compute baseline.

Per Snell et al. 2024 (RESEARCH.md §1), best-of-N is the *baseline* that
compute-optimal scaling must beat, not the target. This agent samples N
independent drafts on the cheap tier and returns the one the verifier scores
highest. It isolates the effect of "more independent samples" from "smarter
allocation" — a necessary control so we never over-claim for search/reflection.
"""

from __future__ import annotations

from ..agent import BaseAgent, Budget
from ..environment import Task
from ..events import RunResult


class BestOfNAgent(BaseAgent):
    name = "best_of_n"

    def __init__(self, llm, verifier, config, models, env=None, n: int = 3) -> None:
        super().__init__(llm, verifier, config, models, env)
        self._n = n

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")
        best_score = -1.0
        best = ""
        for _ in range(self._n):
            text = self._call(run, "sample", self._system(task), task.prompt, model="cheap")
            score = self.verifier.score(text, task).score
            if score > best_score:
                best_score, best = score, text
        verdict = self._verify(run, best, task)
        return self._finalize(run, task, verdict, best, attempts=self._n)
