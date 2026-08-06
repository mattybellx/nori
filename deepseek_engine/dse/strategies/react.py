"""ReAct-style baseline agent (Yao et al. 2022).

Control condition for the harness: generate a solution, then observe the test
feedback. No retries, no memory, no search. Any strategy must beat this baseline
on the measured metrics to justify its added compute.
"""

from __future__ import annotations

from ..agent import BaseAgent, Budget
from ..environment import Task
from ..events import RunResult


class ReactAgent(BaseAgent):
    name = "react"

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")
        text = self._call(run, "thought", self._system(task), task.prompt, model="cheap")
        verdict = self._verify(run, text, task)
        return self._finalize(run, task, verdict, text, attempts=1)
