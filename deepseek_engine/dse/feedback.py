"""Compiler feedback loop — build → test → lint → security → perf.

Per RESEARCH.md §5, agents receive *parsed, structured* results (SWE-agent /
AlphaCodium lesson), not raw terminal dumps. Each stage is real but lightweight
here; deeper integrations (real compilers, real linters, real SCA scanners) are
documented limitations. ``test`` is the primary signal; the others are thin but
honest heuristics.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .environment import Environment, Task


@dataclass
class StageResult:
    stage: str
    passed: bool
    detail: str


@dataclass
class FeedbackReport:
    stages: list[StageResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.stages) and all(s.passed for s in self.stages)

    def failures(self) -> list[StageResult]:
        return [s for s in self.stages if not s.passed]

    def detail(self, stage: str) -> str:
        for s in self.stages:
            if s.stage == stage:
                return s.detail
        return ""


# Security patterns a real SCA pass would flag; kept as a documented heuristic.
_SECURITY_PATTERNS = [
    (r"eval\s*\(", "eval() call"),
    (r"exec\s*\(", "exec() call"),
    (r"subprocess.*shell\s*=\s*True", "shell=True"),
    (r"os\.system\s*\(", "os.system() call"),
]


class CompilerFeedbackLoop:
    """Runs the feedback stages over a candidate solution text."""

    def __init__(self, env: Environment) -> None:
        self._env = env

    def run(self, task: Task, solution_text: str) -> FeedbackReport:
        report = FeedbackReport()
        start = time.perf_counter()

        # -- build: structural parse -------------------------------------
        has_answer = re.search(r"ANSWER:\s*\S+", solution_text) is not None
        step_lines = re.findall(r"^\s*STEP\s+\d+\s*:\s*\S+", solution_text, re.MULTILINE)
        report.stages.append(
            StageResult(
                "build",
                has_answer and len(step_lines) == task.num_steps,
                f"answer_present={has_answer}, step_lines={len(step_lines)}/{task.num_steps}",
            )
        )

        # -- test: the primary signal ------------------------------------
        test_report = self._env.run_tests(task, solution_text)
        report.stages.append(
            StageResult(
                "test",
                test_report.passed == test_report.total,
                f"tests passed {test_report.passed}/{test_report.total}, "
                f"failed_steps={test_report.failed_steps}",
            )
        )

        # -- lint: formatting heuristic ----------------------------------
        malformed = [
            line for line in solution_text.splitlines()
            if re.match(r"^\s*(STEP|ANSWER):", line) and not re.match(r"^\s*(STEP\s+\d+\s*:\s*\S+|ANSWER:\s*\S+)", line)
        ]
        report.stages.append(
            StageResult("lint", not malformed, f"malformed_lines={len(malformed)}")
        )

        # -- security: documented heuristic (real SCA is a limitation) ---
        findings = [
            pattern for pattern, _ in _SECURITY_PATTERNS if re.search(pattern, solution_text)
        ]
        report.stages.append(
            StageResult("security", not findings, f"findings={findings}")
        )

        # -- perf: measured evaluation time --------------------------------
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        report.stages.append(
            StageResult("perf", True, f"evaluation took {elapsed_ms:.1f} ms")
        )
        return report
