"""Reflexion agent (Shinn et al. 2023) with episodic memory.

Mechanism: run a ReAct trial; if the verifier fails, derive a reflection from
the *external* per-step feedback, store it in episodic memory, convert failures
into ``FEEDBACK`` lines for the next trial, and retry.

Important (Huang et al. 2023): the improvement comes from the external signal,
never from pure self-critique. The reflection text is recorded for telemetry and
future real-model use; the MockLLM's repair capability is driven by the
``FEEDBACK`` lines, which encode exactly that external signal.
"""

from __future__ import annotations

from ..agent import BaseAgent, Budget, feedback_lines
from ..environment import Task
from ..events import RunResult, StepEvent
from ..memory import Memory


class ReflexionAgent(BaseAgent):
    name = "reflexion"

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        max_trials = budget.max_trials if budget else self.config.max_trials
        memory = Memory(
            window_tokens=self.config.memory_window_tokens,
            summary_every=self.config.memory_summary_every,
        )
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")
        last_failed: list[int] = []
        text = ""
        verdict = None

        for trial in range(max_trials):
            extra = feedback_lines(last_failed)
            # real-task verifiers return free-text feedback instead of failed
            # steps; feed it to the next attempt so the model can use it
            free_text = verdict.details.get("feedback", "") if verdict else ""
            if free_text:
                extra += f"\nFEEDBACK: {free_text}"
            if memory.reflections:
                extra += "\n" + "\n".join(f"REFLECTION: {r}" for r in memory.reflections)
            system = self._system(task, extra)
            text = self._call(run, "thought", system, task.prompt, model="cheap")
            verdict = self._verify(run, text, task)

            if verdict.passed:
                return self._finalize(run, task, verdict, text, attempts=trial + 1)

            last_failed = verdict.details.get("failed_steps", [])
            reflection = (
                f"Trial {trial + 1} failed: {free_text or ('steps ' + str(last_failed))}; "
                f"re-attempt, fixing the reported problem."
            )
            memory.add_reflection(reflection)
            run.steps.append(
                StepEvent(kind="reflect", content=reflection, model="reflexion")
            )

        return self._finalize(run, task, verdict, text, attempts=max_trials)
