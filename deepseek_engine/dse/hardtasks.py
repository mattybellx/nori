"""Hard real (natural-language) tasks: multi-step reasoning with deterministic
checkers and NO code execution.

These tasks are deliberately harder than the base real suite: each requires
several steps of reasoning (compound interest, work rates, inclusion-exclusion,
algorithm tracing, classic traps). A single-shot model frequently fails them;
strategies that verify against the deterministic checker and repair with
feedback should do measurably better — that is the delta we want to be able to
measure on real models.

Mirrors the ``RealTask`` interface, so the existing strategies, the MockLLM
and the ``RealTaskVerifier`` consume it unchanged.
"""

from __future__ import annotations

from .realtasks import RealTask, numeric_checker, normalized_checker


def _h() -> list[RealTask]:
    return [
        # --- multi-step math -------------------------------------------------
        RealTask(
            "hard-math-001", "math",
            "You deposit $1000 in an account earning 5% annual interest, "
            "compounded annually. What is the balance after 3 years, rounded "
            "to the nearest dollar? Respond with a final line "
            "'ANSWER: <number>'.",
            "1158", numeric_checker("1158"), difficulty=0.8),
        RealTask(
            "hard-math-002", "math",
            "Pipe A fills a tank in 6 hours; Pipe B fills the same tank in 3 "
            "hours. Working together, how many minutes does it take to fill "
            "the tank? Respond with a final line 'ANSWER: <number>'.",
            "120", numeric_checker("120"), difficulty=0.8),
        RealTask(
            "hard-math-003", "math",
            "A chemist mixes 3 liters of a 20% solution with 2 liters of a "
            "50% solution. What is the concentration of the mixture? Give the "
            "answer as a number (e.g., 32 for 32%). Respond with a final line "
            "'ANSWER: <number>'.",
            "32", numeric_checker("32"), difficulty=0.8),
        RealTask(
            "hard-math-004", "math",
            "A number is doubled, then 10 is added, then the result is divided "
            "by 3, giving 12. What was the original number? Respond with a "
            "final line 'ANSWER: <number>'.",
            "13", numeric_checker("13"), difficulty=0.75),
        RealTask(
            "hard-math-005", "math",
            "How many integers from 1 to 100 inclusive are divisible by 3 or "
            "by 5? Respond with a final line 'ANSWER: <number>'.",
            "47", numeric_checker("47"), difficulty=0.85),

        # --- algorithm tracing (manual trace; no execution) -----------------
        RealTask(
            "hard-code-001", "code",
            "Consider this Python function:\n"
            "def f(n):\n"
            "    if n < 2: return n\n"
            "    return f(n - 1) + f(n - 2)\n"
            "What is the value of f(8)? Respond with a final line "
            "'ANSWER: <number>'.",
            "21", numeric_checker("21"), difficulty=0.85),
        RealTask(
            "hard-code-002", "code",
            "Consider this Python function:\n"
            "def mystery(s):\n"
            "    r = ''\n"
            "    for ch in s:\n"
            "        r = ch + r\n"
            "    return r\n"
            "What does mystery('hello') return? Respond with a final line "
            "'ANSWER: <value>'.",
            "olleh", normalized_checker("olleh"), difficulty=0.75),
        RealTask(
            "hard-code-003", "code",
            "Consider this Python function:\n"
            "def g(n):\n"
            "    total = 0\n"
            "    i = 1\n"
            "    while i <= n:\n"
            "        total += i\n"
            "        i *= 2\n"
            "    return total\n"
            "What is the value of g(10)? Respond with a final line "
            "'ANSWER: <number>'.",
            "15", numeric_checker("15"), difficulty=0.8),

        # --- logic / counting / classic traps -------------------------------
        RealTask(
            "hard-logic-001", "logic",
            "How many three-digit numbers have an odd hundreds digit AND are "
            "divisible by 5? Respond with a final line 'ANSWER: <number>'.",
            "100", numeric_checker("100"), difficulty=0.85),
        RealTask(
            "hard-logic-002", "logic",
            "In a room, 5 people each shake hands with every other person "
            "exactly once. How many handshakes happen in total? Respond with "
            "a final line 'ANSWER: <number>'.",
            "10", numeric_checker("10"), difficulty=0.75),
        RealTask(
            "hard-logic-003", "logic",
            "A clock shows 3:15. What is the smaller angle between the hour "
            "hand and the minute hand, in degrees? Respond with a final line "
            "'ANSWER: <number>'.",
            "7.5", numeric_checker("7.5", tolerance=0.01), difficulty=0.9),

        # --- chained multi-part (a wrong step cascades) ---------------------
        RealTask(
            "hard-chain-001", "chain",
            "A train 100 meters long passes a pole in 5 seconds and passes a "
            "platform in 20 seconds (both at the same constant speed). What "
            "is the length of the platform in meters? Respond with a final "
            "line 'ANSWER: <number>'.",
            "300", numeric_checker("300"), difficulty=0.85),
    ]


def make_hard_catalog(seed: int = 0) -> dict[str, RealTask]:
    """Build the hard real suite. ``seed`` only affects ordering (all tasks are
    included; there are no stochastic draws in the catalog itself)."""
    return {task.id: task for task in _h()}


# ---------------------------------------------------------------------------
# hard-tuned: tasks targeted at a STRONG reasoning model's known weak spots.
# The base hard-real suite (multi-step but small numbers) was solved 100%
# single-shot by deepseek-v4-flash. These tasks deliberately attack where
# single-shot reasoning fails even on strong models: large exact arithmetic,
# long operation chains, double-counting overlaps, percent-off-then-tax
# ordering, CRT-style remainder search, expected value, and classic traps.
# Deterministic checkers, no code execution.
# ---------------------------------------------------------------------------
def _h2() -> list[RealTask]:
    return [
        # --- large exact arithmetic (single-shot hallucination) ------------
        RealTask(
            "hard2-math-001", "math",
            "What is 1234 × 5678? Give the exact integer. Respond with a "
            "final line 'ANSWER: <number>'.",
            "7006652", numeric_checker("7006652"), difficulty=0.95),
        RealTask(
            "hard2-math-002", "math",
            "What is the remainder when 37^5 is divided by 100? Respond with "
            "a final line 'ANSWER: <number>'.",
            "57", numeric_checker("57"), difficulty=0.9),
        RealTask(
            "hard2-math-003", "math",
            "Start with 0. Add 7, subtract 3, multiply by 2, add 11, subtract "
            "9, multiply by 3, add 5. What is the final value? Respond with a "
            "final line 'ANSWER: <number>'.",
            "35", numeric_checker("35"), difficulty=0.85),
        # --- double-counting / constraints ----------------------------------
        RealTask(
            "hard2-math-004", "math",
            "How many integers from 1 to 200 inclusive are divisible by 2, 3, "
            "or 5? Respond with a final line 'ANSWER: <number>'.",
            "146", numeric_checker("146"), difficulty=0.9),
        RealTask(
            "hard2-math-005", "math",
            "A store takes 10% off the marked price, then adds 8% sales tax "
            "on the discounted price. The item is marked $80. What is the "
            "final price in dollars, rounded to the nearest cent? Respond "
            "with a final line 'ANSWER: <number>'.",
            "77.76", numeric_checker("77.76", tolerance=0.01), difficulty=0.85),
        # --- traps -----------------------------------------------------------
        RealTask(
            "hard2-logic-001", "logic",
            "A snail climbs 3 meters up a wall during the day and slips 2 "
            "meters down at night. The wall is 10 meters high. On which day "
            "does it first reach the top? Respond with a final line "
            "'ANSWER: <number>'.",
            "8", numeric_checker("8"), difficulty=0.9),
        RealTask(
            "hard2-logic-002", "logic",
            "How many 4-digit PINs from 0000 to 9999 contain exactly one "
            "digit '7'? Respond with a final line 'ANSWER: <number>'.",
            "2916", numeric_checker("2916"), difficulty=0.85),
        RealTask(
            "hard2-logic-003", "logic",
            "Two fair six-sided dice are rolled and their values are "
            "multiplied. What is the expected value of the product? Respond "
            "with a final line 'ANSWER: <number>'.",
            "12.25", numeric_checker("12.25", tolerance=1e-4), difficulty=0.9),
        # --- modular arithmetic / digit tricks ------------------------------
        RealTask(
            "hard2-code-001", "code",
            "What is the remainder when 2^50 is divided by 7? Respond with a "
            "final line 'ANSWER: <number>'.",
            "4", numeric_checker("4"), difficulty=0.9),
        RealTask(
            "hard2-code-002", "code",
            "What is the sum of the digits of 99 × 99? Respond with a final "
            "line 'ANSWER: <number>'.",
            "18", numeric_checker("18"), difficulty=0.8),
        # --- cascading chains -------------------------------------------------
        RealTask(
            "hard2-chain-001", "chain",
            "A tank fills in 6 hours through pipe A alone, and drains empty "
            "in 9 hours through a leak at the bottom. With pipe A open and "
            "the leak unplugged, how many HOURS does it take to fill the "
            "tank? Respond with a final line 'ANSWER: <number>'.",
            "18", numeric_checker("18"), difficulty=0.85),
        RealTask(
            "hard2-chain-002", "chain",
            "Find the smallest positive integer greater than 20 that leaves "
            "remainder 3 when divided by 5 and remainder 4 when divided by "
            "7. Respond with a final line 'ANSWER: <number>'.",
            "53", numeric_checker("53"), difficulty=0.9),
    ]


def make_hard_tuned_catalog(seed: int = 0) -> dict[str, RealTask]:
    """Build the hard-tuned suite (targeted at strong reasoning models)."""
    return {task.id: task for task in _h2()}
