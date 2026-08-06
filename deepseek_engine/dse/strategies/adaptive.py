"""Adaptive compute allocation (compute-optimal test-time compute).

Per Snell et al. 2024 (RESEARCH.md §1): the effectiveness of any scaling method
depends on prompt difficulty, so compute should be *allocated adaptively per
prompt*. This strategy:

1. Runs one cheap **difficulty probe** (one cheap attempt + verifier score).
2. Routes by probe result:
   - probe passes            → low tier: done (cheapest possible path);
   - probe partially correct → medium tier: Reflexion (retry + feedback);
   - probe mostly wrong      → high tier: TreeSearch (LATS-lite).

The probe's telemetry is merged into the returned run so costs are fully
accounted for — no hidden compute.
"""

from __future__ import annotations

from ..agent import Agent, BaseAgent, Budget
from ..environment import Task
from ..events import RunResult


class AdaptiveAgent(BaseAgent):
    name = "adaptive"

    def __init__(
        self,
        llm,
        verifier,
        config,
        models,
        env=None,
        sub_agents: dict[str, Agent] | None = None,
        policy: str = "difficulty",
    ) -> None:
        super().__init__(llm, verifier, config, models, env)
        if sub_agents is None:
            from .reflexion import ReflexionAgent
            from .tree_search import TreeSearchAgent

            sub_agents = {
                "reflexion": ReflexionAgent(llm, verifier, config, models, env),
                "tree_search": TreeSearchAgent(llm, verifier, config, models, env),
            }
        self._sub_agents = sub_agents
        if policy not in {"score", "difficulty"}:
            raise ValueError(f"unknown adaptive policy: {policy!r}")
        self._policy = policy

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")

        # -- difficulty probe (one cheap call) --------------------------------
        text = self._call(run, "probe", self._system(task), task.prompt, model="cheap")
        verdict = self._verify(run, text, task)

        if verdict.passed:
            return self._finalize(run, task, verdict, text, attempts=1)

        # Re-tuned policy ("difficulty"): hard tasks go straight to search —
        # the score-based gate routed many hard tasks to reflexion and capped
        # success (measured in BENCHMARKS.md). The "score" policy is retained
        # as the v1 baseline for comparison.
        if self._policy == "difficulty" and task.num_steps >= 3:
            tier, key = "high", "tree_search"
        elif verdict.score >= 0.5:
            tier, key = "medium", "reflexion"
        else:
            tier, key = "high", "tree_search"

        sub_run = self._sub_agents[key].solve(task, budget)
        sub_run.route_tier = tier
        # the outer strategy owns the reported identity; internals are visible
        # in the step log (probe/thought/reflect/action events)
        sub_run.strategy = self.name
        # merge probe telemetry so the full cost is visible
        sub_run.steps = run.steps + sub_run.steps
        sub_run.tokens_in += run.tokens_in
        sub_run.tokens_out += run.tokens_out
        sub_run.latency_s += run.latency_s
        sub_run.attempts += 1
        return sub_run
