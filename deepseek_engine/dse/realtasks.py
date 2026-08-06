"""Real (natural-language) task suite for provider-backed runs.

Unlike the synthetic calibrated suite — whose step tokens are opaque and can
only be produced by the MockLLM — these tasks are answerable by a *real* LLM
and checked deterministically, with **no code execution** (a documented safety
choice; execution-based code checking is future work).

``RealTask`` mirrors the ``Task`` interface (``num_steps=1``,
``correct_steps=(gold,)``, ``distractors``, ``check_step``) so the existing
strategies and the MockLLM can consume it unchanged; the real-suite verifier
(``RealTaskVerifier``) uses the ``checker`` instead of per-step exact-match.

Answer format the prompt asks for: a final ``ANSWER: <value>`` line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Checkers (deterministic; no execution)
# ---------------------------------------------------------------------------
def _extract_number(text: str) -> float | None:
    # prefer the number on the ANSWER line (agents emit 'ANSWER: <value>'),
    # falling back to the first number anywhere in the text
    answer_match = re.search(r"ANSWER:\s*(-?\d+(?:[.,]\d+)?)", text, re.IGNORECASE)
    if answer_match:
        return float(answer_match.group(1).replace(",", ""))
    match = re.search(r"-?\d+(?:[.,]\d+)?", text)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def numeric_checker(gold: str, tolerance: float = 1e-6) -> Callable[[str], tuple[bool, str]]:
    gold_f = float(gold)

    def check(text: str) -> tuple[bool, str]:
        value = _extract_number(text)
        if value is None:
            return False, "no numeric answer found in the response"
        if abs(value - gold_f) <= tolerance:
            return True, ""
        return False, f"expected {gold_f:g}, got {value:g}"

    return check


def normalized_checker(gold: str) -> Callable[[str], tuple[bool, str]]:
    g = gold.strip().lower()

    def check(text: str) -> tuple[bool, str]:
        match = re.search(r"ANSWER:\s*(.+)", text, re.MULTILINE)
        candidate = match.group(1) if match else text
        norm = candidate.strip().lower().strip('"').strip("'").strip(".")
        if norm == g:
            return True, ""
        return False, f"expected {gold!r}, got {candidate.strip()[:40]!r}"

    return check


# ---------------------------------------------------------------------------
# RealTask
# ---------------------------------------------------------------------------
@dataclass
class RealTask:
    id: str
    domain: str
    prompt: str
    gold: str
    checker: Callable[[str], tuple[bool, str]] | None = None
    # --- Task-compatible surface (for strategies + MockLLM) ----------------
    num_steps: int = 1
    correct_steps: tuple[str, ...] = ()
    distractors: tuple[tuple[str, ...], ...] = ()
    # --- metadata -----------------------------------------------------------
    difficulty: float = 0.5  # 0..1; base real suite ~0.5, hard suite 0.75-0.9

    def __post_init__(self) -> None:
        self.correct_steps = (self.gold,)
        self.distractors = ((f"<wrong-{self.id}>",),)

    def check_step(self, index: int, value: str) -> bool:
        return index == 0 and value == self.gold

    def check(self, answer_text: str) -> tuple[bool, str]:
        if self.checker is not None:
            return self.checker(answer_text)
        return normalized_checker(self.gold)(answer_text)


# ---------------------------------------------------------------------------
# The suite
# ---------------------------------------------------------------------------
def _t() -> list[RealTask]:
    return [
        RealTask("math-001", "math",
                 "What is 17 × 23? Respond with a final line 'ANSWER: <number>'.",
                 "391", numeric_checker("391")),
        RealTask("math-002", "math",
                 "A train travels at 60 km/h for 2.5 hours. How far does it "
                 "travel in km? Respond with a final line 'ANSWER: <number>'.",
                 "150", numeric_checker("150")),
        RealTask("math-003", "math",
                 "What is the sum of the first 10 positive integers? Respond "
                 "with a final line 'ANSWER: <number>'.",
                 "55", numeric_checker("55")),
        RealTask("math-004", "math",
                 "Solve for x: 3x + 5 = 20. Respond with a final line "
                 "'ANSWER: <number>'.",
                 "5", numeric_checker("5")),
        RealTask("math-005", "math",
                 "A rectangle is 8 cm by 5 cm. What is its area in square cm? "
                 "Respond with a final line 'ANSWER: <number>'.",
                 "40", numeric_checker("40")),
        RealTask("logic-001", "logic",
                 "What is the next number in the sequence 2, 4, 8, 16, ...? "
                 "Respond with a final line 'ANSWER: <number>'.",
                 "32", numeric_checker("32")),
        RealTask("logic-002", "logic",
                 "A farmer has 17 sheep and all but 9 run away. How many sheep "
                 "are left? Respond with a final line 'ANSWER: <number>'.",
                 "9", numeric_checker("9")),
        RealTask("logic-003", "logic",
                 "If all Bloops are Razzies and all Razzies are Lazzies, then "
                 "are all Bloops Lazzies? Respond with a final line "
                 "'ANSWER: yes' or 'ANSWER: no'.",
                 "yes", normalized_checker("yes")),
        RealTask("logic-004", "logic",
                 "Which is larger, 0.5 or 1/3? Respond with a final line "
                 "'ANSWER: <value>' containing the larger value.",
                 "0.5", normalized_checker("0.5")),
        RealTask("code-001", "code",
                 "In Python, what built-in function returns the number of "
                 "items in a container? Respond with a final line "
                 "'ANSWER: <name>'.",
                 "len", normalized_checker("len")),
        RealTask("code-002", "code",
                 "In Python, what keyword introduces a function definition? "
                 "Respond with a final line 'ANSWER: <keyword>'.",
                 "def", normalized_checker("def")),
        RealTask("code-003", "code",
                 "In Python, what is the result of 7 // 2 (integer division)? "
                 "Respond with a final line 'ANSWER: <number>'.",
                 "3", numeric_checker("3")),
    ]


def make_real_catalog(seed: int = 0) -> dict[str, RealTask]:
    """Build the real suite. ``seed`` only affects ordering (all tasks are
    included; there are no stochastic draws in the catalog itself)."""
    tasks = _t()
    return {task.id: task for task in tasks}


# Strategies that require per-step decomposition are excluded from the real
# suite (single-unit tasks have no meaningful 'steps' to search over).
REAL_SUITE_AGENTS = frozenset(
    {"react", "best_of_n", "reflexion", "self_refine", "escalating"}
)
