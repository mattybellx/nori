"""Multi-agent orchestration (MoA-lite), behind the ``multi_agent`` flag.

Evidence (RESEARCH.md §4): layered proposer+aggregator improves quality at Nx
token cost (Wang et al. 2024); naive chaining cascades errors unless handoffs
are verifier-gated (Hong et al. 2023, MetaGPT).

This implementation: N proposers each draft a solution; the aggregator picks the
draft with the highest verifier score. **Honest caveat**: with a real LLM the
aggregator would *merge* proposals (true MoA); our mock uses verifier-selection
as the aggregation policy — a documented proxy that preserves the cost/quality
trade-off we want to measure. The verifier gate means a low-quality proposal
never reaches the output.
"""

from __future__ import annotations

from .agent import BaseAgent, Budget
from .environment import Task
from .events import RunResult


class MoAAgent(BaseAgent):
    name = "multi_agent"

    def __init__(self, llm, verifier, config, models, env=None, n_proposers: int | None = None) -> None:
        super().__init__(llm, verifier, config, models, env)
        self._n_proposers = n_proposers or config.multi_agent_proposers

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")
        proposals: list[tuple[float, str]] = []
        for _ in range(self._n_proposers):
            text = self._call(run, "propose", self._system(task), task.prompt, model="cheap")
            score = self.verifier.score(text, task).score
            proposals.append((score, text))

        best_score, best = max(proposals, key=lambda item: item[0])
        verdict = self._verify(run, best, task)
        return self._finalize(run, task, verdict, best, attempts=self._n_proposers)
