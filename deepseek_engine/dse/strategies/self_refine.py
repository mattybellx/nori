"""Self-Refine agent (Madaan et al. 2023), gated on external verifier signals.

Per Huang et al. 2023, open-loop self-critique without external feedback often
degrades performance. This implementation therefore only refines when the
verifier has failed, and it feeds the *external* per-step failures back into the
refinement prompt. The model keeps its verified steps and repairs the rest.
"""

from __future__ import annotations

from ..agent import BaseAgent, Budget, feedback_lines
from ..environment import Task
from ..events import RunResult


class SelfRefineAgent(BaseAgent):
    name = "self_refine"

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        max_iter = budget.max_trials if budget else self.config.max_trials
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")

        text = self._call(run, "generate", self._system(task), task.prompt, model="cheap")
        verdict = self._verify(run, text, task)
        iterations = 1

        while not verdict.passed and iterations < max_iter:
            failed = verdict.details.get("failed_steps", [])
            extra = feedback_lines(failed)
            free_text = verdict.details.get("feedback", "")
            if failed:
                extra += "\nCRITIQUE: keep the verified steps; correct only the failing ones."
            elif free_text:
                extra += f"\nCRITIQUE: your answer was wrong. Feedback: {free_text}"
            system = self._system(task, extra)
            user = f"Previous draft:\n{text}\n\nProduce the refined draft."
            text = self._call(run, "refine", system, user, model="cheap")
            verdict = self._verify(run, text, task)
            iterations += 1

        return self._finalize(run, task, verdict, text, attempts=iterations)
