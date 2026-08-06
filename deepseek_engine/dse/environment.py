"""Task model, deterministic task catalog, and the environment command surface.

The task suite is synthetic but *calibrated*: every task decomposes into
``num_steps`` independently testable steps, each with a known ground-truth value
and distractors. This lets the MockLLM (see ``dse/llm.py``) produce step-wise
outputs with a controllable accuracy, so strategy comparisons in the harness are
causal and reproducible rather than dependent on opaque API variance.

The environment exposes a small command surface (``run_tests``, ``search``,
``read``, ``edit``) per the SWE-agent ACI lesson: agents interact through a
structured interface and receive *parsed* feedback, not raw dumps.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Task:
    id: str
    domain: str
    prompt: str
    num_steps: int
    correct_steps: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def gold(self) -> str:
        return "".join(self.correct_steps)

    def check_step(self, index: int, value: str) -> bool:
        if not (0 <= index < self.num_steps):
            return False
        return value == self.correct_steps[index]


@dataclass
class TestReport:
    passed: int
    total: int
    failed_steps: list[int] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Deterministic catalog generation
# ---------------------------------------------------------------------------
_DOMAIN_POOLS = {
    "arithmetic": [f"a{i:02d}" for i in range(40)],
    "logic": [f"l{i:02d}" for i in range(40)],
    "code": [f"c{i:02d}" for i in range(40)],
}

_PROMPT_TEMPLATES = {
    "arithmetic": (
        "Evaluate the arithmetic expression. Work in exactly {n} intermediate "
        "steps, each on its own line as 'STEP i: <value>', then finish with "
        "'ANSWER: <final>'."
    ),
    "logic": (
        "Solve the logic puzzle. Work in exactly {n} intermediate steps, each "
        "on its own line as 'STEP i: <value>', then finish with 'ANSWER: <final>'."
    ),
    "code": (
        "Implement the function. Work in exactly {n} intermediate steps, each on "
        "its own line as 'STEP i: <value>', then finish with 'ANSWER: <final>'."
    ),
}


def make_catalog(
    seed: int = 0,
    n_tasks: int = 48,
    domains: tuple[str, ...] = ("arithmetic", "logic", "code"),
    min_steps: int = 1,
    max_steps: int = 5,
    distractors_per_step: int = 3,
) -> dict[str, Task]:
    """Build a deterministic catalog of calibrated tasks.

    Steps are drawn from per-domain token pools; the 'answer' is the
    concatenation of correct step values. Difficulty scales with ``num_steps``
    (base success probability = p ** num_steps for step accuracy p).
    """
    rng = random.Random(seed)
    catalog: dict[str, Task] = {}
    for idx in range(n_tasks):
        domain = domains[idx % len(domains)]
        pool = _DOMAIN_POOLS[domain]
        num_steps = rng.randint(min_steps, max_steps)
        correct = tuple(rng.choice(pool) for _ in range(num_steps))
        distractors: list[tuple[str, ...]] = []
        for _ in range(num_steps):
            options = [v for v in pool if v not in correct]
            rng.shuffle(options)
            distractors.append(tuple(options[:distractors_per_step]))
        task_id = f"{domain}-{idx:03d}"
        catalog[task_id] = Task(
            id=task_id,
            domain=domain,
            prompt=_PROMPT_TEMPLATES[domain].format(n=num_steps),
            num_steps=num_steps,
            correct_steps=correct,
            distractors=tuple(distractors),
        )
    return catalog


def split_by_difficulty(
    catalog: dict[str, Task], threshold: int = 3
) -> tuple[dict[str, Task], dict[str, Task]]:
    """Split the catalog into easy (num_steps < threshold) and hard (>= threshold)."""
    easy = {k: v for k, v in catalog.items() if v.num_steps < threshold}
    hard = {k: v for k, v in catalog.items() if v.num_steps >= threshold}
    return easy, hard


# ---------------------------------------------------------------------------
# Environment / agent-computer interface (ACI)
# ---------------------------------------------------------------------------
class Environment:
    """Structured interface an agent uses against a task.

    ``run_tests`` is the real signal in this engine: it parses the agent's
    solution and reports per-step pass/fail (AlphaCodium-style test feedback).
    ``search``/``read``/``edit`` exist for ACI completeness and are documented
    stubs (real repo integration is a stated limitation).
    """

    def __init__(self, catalog: dict[str, Task]) -> None:
        self.catalog = catalog

    def run_tests(self, task: Task, solution_text: str) -> TestReport:
        """Parse 'STEP i: <value>' lines and report per-step pass/fail."""
        steps: dict[int, str] = {}
        for match in re.finditer(r"^\s*STEP\s+(\d+)\s*:\s*(\S+)", solution_text, re.MULTILINE):
            steps[int(match.group(1))] = match.group(2)
        failed: list[int] = []
        passed = 0
        for i in range(task.num_steps):
            value = steps.get(i)
            if value is not None and task.check_step(i, value):
                passed += 1
            else:
                failed.append(i)
        return TestReport(passed=passed, total=task.num_steps, failed_steps=failed)

    def search(self, task: Task, query: str) -> str:
        """Stub: returns task metadata. Real repo search is a limitation."""
        return f"[search] {task.id} ({task.domain}, {task.num_steps} steps) :: {query}"

    def read(self, task: Task, path: str) -> str:
        """Stub: returns task prompt as the 'file' to read."""
        return f"[read] {path}\n{task.prompt}"

    def edit(self, task: Task, path: str, content: str) -> str:
        """Stub: records an edit (returns content length as confirmation)."""
        return f"[edit] {path} wrote {len(content)} chars"
