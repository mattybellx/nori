"""Confidence-based model escalation (dynamic model routing, end-to-end).

This agent exercises the ``Router``: it starts on the cheap tier, and if the
verifier's confidence on the first draft is low, it escalates once to the
expensive tier (bounded by ``RouterConfig.max_escalations``). The benchmark
measures whether the expensive tier's higher step accuracy repays its higher
cost — the core dynamic-routing question (RESEARCH.md §7).
"""

from __future__ import annotations

from ..agent import BaseAgent, Budget
from ..environment import Task
from ..events import RunResult
from ..router import Router


class EscalatingAgent(BaseAgent):
    name = "escalating"

    def __init__(self, llm, verifier, config, models, env=None, router: Router | None = None) -> None:
        super().__init__(llm, verifier, config, models, env)
        self._router = router or Router(config.router)

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")
        self._router.reset()

        text = self._call(run, "sample", self._system(task), task.prompt, model="cheap")
        verdict = self._verify(run, text, task)
        attempts = 1

        if not verdict.passed:
            decision = self._router.decide(verdict.score)
            if decision.escalated:
                text = self._call(
                    run, "escalate", self._system(task), task.prompt, model=decision.model
                )
                verdict = self._verify(run, text, task)
                attempts += 1
                run.route_tier = "high"
            else:
                run.route_tier = "low"

        return self._finalize(run, task, verdict, text, attempts=attempts)
