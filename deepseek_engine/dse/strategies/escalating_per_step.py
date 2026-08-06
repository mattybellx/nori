"""Per-step model escalation.

Weakness of whole-task escalation (measured in BENCHMARKS.md): one expensive
full re-draft loses to cheap feedback-driven retries on hard tasks, because
``0.8^steps`` decays fast. This variant escalates only the *failing steps*: the
cheap tier drafts, the verifier pinpoints the wrong steps, and only those steps
are re-proposed by the expensive tier (keeping the verified ones). For a task
with k cheap-correct steps, success becomes ``p_exp ** (steps - k)`` instead of
``p_exp ** steps``.
"""

from __future__ import annotations

import re

from ..agent import BaseAgent, Budget
from ..environment import Task
from ..events import RunResult


class EscalatingPerStepAgent(BaseAgent):
    name = "escalating_per_step"

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")

        text = self._call(run, "sample", self._system(task), task.prompt, model="cheap")
        verdict = self._verify(run, text, task)
        if verdict.passed:
            return self._finalize(run, task, verdict, text, attempts=1)

        steps = self._parse_steps(text)
        failed = verdict.details.get("failed_steps", [])
        for index in failed:
            system = self._system(task, f"MODE: next_step\nPREFIX: {index}")
            out = self._call(
                run, "escalate_step", system, "Propose this step.", model="expensive"
            )
            match = re.search(rf"STEP\s+{index}\s*:\s*(\S+)", out)
            if match:
                steps[index] = match.group(1)

        rebuilt = self._render(task, steps)
        verdict = self._verify(run, rebuilt, task)
        return self._finalize(run, task, verdict, rebuilt, attempts=2)

    @staticmethod
    def _parse_steps(text: str) -> dict[int, str]:
        steps: dict[int, str] = {}
        for match in re.finditer(r"^\s*STEP\s+(\d+)\s*:\s*(\S+)", text, re.MULTILINE):
            steps[int(match.group(1))] = match.group(2)
        return steps

    @staticmethod
    def _render(task: Task, steps: dict[int, str]) -> str:
        values = [steps.get(i, "?") for i in range(task.num_steps)]
        lines = [f"STEP {i}: {v}" for i, v in enumerate(values)]
        return "\n".join(lines) + f"\nANSWER: {''.join(values)}"
