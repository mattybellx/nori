"""Fail suite: tasks deliberately hard for a STRONG reasoning model.

These are the "harder-than-frontier" tasks the honest-limits discussion asked
for — long exact arithmetic, modular exponentiation, CRT, calendar counting,
trap mixtures and exact probabilities. A frontier model (DeepSeek V4 Flash)
solves the hard-real / hard-tuned suites 100% single-shot, so those suites
cannot show "sometimes better". This suite is engineered so a strong model
genuinely slips single-shot (expected baseline well below 1.0), which is the
regime where the never-worse guards and synthesis can demonstrate real
recovery.

Every gold answer below is verified by computation (see the session log);
``RealTaskVerifier`` scores the extracted ``ANSWER: <value>`` line.

Mirrors the ``RealTask`` interface — strategies, MockLLM and the verifier
consume it unchanged.
"""

from __future__ import annotations

from .realtasks import RealTask, numeric_checker, normalized_checker


def make_fail_catalog(seed: int = 0) -> dict[str, RealTask]:
    del seed  # deterministic catalog (no seeding needed)
    tasks = [
        # --- long exact arithmetic (carry-chain / multi-digit) --------------
        RealTask(
            "fail-arith-001", "math",
            "Compute exactly: 4829537 × 71 − 12345. Show your work, then "
            "respond with a final line 'ANSWER: <number>'.",
            "342884782", numeric_checker("342884782"), difficulty=0.95),
        RealTask(
            "fail-arith-002", "math",
            "Compute exactly: 123456789 × 45 + 98765. Respond with a final "
            "line 'ANSWER: <number>'.",
            "5555654270", numeric_checker("5555654270"), difficulty=0.95),
        RealTask(
            "fail-arith-003", "math",
            "What is the sum of the digits of 2^40 (two to the fortieth "
            "power)? Respond with a final line 'ANSWER: <number>'.",
            "61", numeric_checker("61"), difficulty=0.9),
        RealTask(
            "fail-arith-004", "math",
            "Compute exactly: (19 × 23 × 29) − (17 × 31). Respond with a "
            "final line 'ANSWER: <number>'.",
            "12146", numeric_checker("12146"), difficulty=0.9),

        # --- modular exponentiation / CRT ------------------------------------
        RealTask(
            "fail-mod-001", "math",
            "Compute 7^11 modulo 1000 (the remainder when 7 to the 11th "
            "power is divided by 1000). Respond with a final line "
            "'ANSWER: <number>'.",
            "743", numeric_checker("743"), difficulty=0.95),
        RealTask(
            "fail-mod-002", "math",
            "Compute 3^17 modulo 100 (the last two digits of 3^17). Respond "
            "with a final line 'ANSWER: <number>'.",
            "63", numeric_checker("63"), difficulty=0.95),
        RealTask(
            "fail-crt-001", "math",
            "Find the smallest positive integer x such that x leaves "
            "remainder 2 when divided by 3, remainder 3 when divided by 5, "
            "and remainder 2 when divided by 7. Respond with a final line "
            "'ANSWER: <number>'.",
            "23", numeric_checker("23"), difficulty=0.95),

        # --- exact counting (inclusion-exclusion, digit counting) -----------
        RealTask(
            "fail-count-001", "math",
            "How many integers from 1 to 500 inclusive are divisible by 7 or "
            "by 11, but NOT by both? Respond with a final line "
            "'ANSWER: <number>'.",
            "104", numeric_checker("104"), difficulty=0.9),
        RealTask(
            "fail-count-002", "math",
            "How many times does the digit 7 appear when you write out every "
            "integer from 1 to 1000 inclusive? Respond with a final line "
            "'ANSWER: <number>'.",
            "300", numeric_checker("300"), difficulty=0.9),

        # --- calendar / date arithmetic --------------------------------------
        RealTask(
            "fail-calendar-001", "math",
            "How many days are there from March 14, 2026 through November 2, "
            "2026 inclusive (counting both the start and end date)? Respond "
            "with a final line 'ANSWER: <number>'.",
            "234", numeric_checker("234"), difficulty=0.95),

        # --- nested percentages / reverse mixtures ---------------------------
        RealTask(
            "fail-percent-001", "math",
            "An item costs $240. It is discounted 15%, then 8% sales tax is "
            "added to the discounted price. What is the final price in "
            "dollars (two decimal places)? Respond with a final line "
            "'ANSWER: <number>'.",
            "220.32", numeric_checker("220.32", tolerance=0.01), difficulty=0.9),
        RealTask(
            "fail-mix-001", "math",
            "You have 4 liters of an 80% acid solution. How many liters of "
            "pure water must you add so the mixture is 50% acid? Respond with "
            "a final line 'ANSWER: <number>'.",
            "2.4", numeric_checker("2.4", tolerance=0.01), difficulty=0.9),

        # --- sequence / probability with exact answer -------------------------
        RealTask(
            "fail-seq-001", "math",
            "What is the next number in this sequence: 2, 3, 5, 9, 17, ...? "
            "Respond with a final line 'ANSWER: <number>'.",
            "33", numeric_checker("33"), difficulty=0.85),
        RealTask(
            "fail-prob-001", "math",
            "Two fair six-sided dice are rolled. What is the probability that "
            "their sum is a prime number? Give the answer as a reduced "
            "fraction a/b (e.g., 5/12). Respond with a final line "
            "'ANSWER: a/b'.",
            "5/12", normalized_checker("5/12"), difficulty=0.9),

        # --- brutal: long exact chains a frontier model can still slip on ----
        RealTask(
            "brutal-arith-001", "math",
            "Compute exactly: 987654321 × 47382. Show your working, then "
            "respond with a final line 'ANSWER: <number>'.",
            "46797037037622", numeric_checker("46797037037622"), difficulty=1.0),
        RealTask(
            "brutal-arith-002", "math",
            "What is the sum of the digits of 3^60 (three to the sixtieth "
            "power)? Respond with a final line 'ANSWER: <number>'.",
            "99", numeric_checker("99"), difficulty=0.95),
        RealTask(
            "brutal-interest-001", "math",
            "You invest $5000 at 6% annual interest compounded MONTHLY for 5 "
            "years. What is the balance, to the nearest cent? Respond with a "
            "final line 'ANSWER: <number>'.",
            "6744.25", numeric_checker("6744.25", tolerance=0.01), difficulty=1.0),
        RealTask(
            "brutal-chain-001", "math",
            "Compute exactly, keeping all intermediate results: take 1000, "
            "add 7%, subtract 137.5, multiply by 1.12, subtract 88, then "
            "multiply by 0.915, then divide by 3. Give the final value to two "
            "decimal places. Respond with a final line 'ANSWER: <number>'.",
            "291.70", numeric_checker("291.70", tolerance=0.01), difficulty=1.0),
        RealTask(
            "brutal-mod-001", "math",
            "Compute 2^50 modulo 97 (the remainder when 2 to the 50th power "
            "is divided by 97). Respond with a final line 'ANSWER: <number>'.",
            "4", numeric_checker("4"), difficulty=0.95),
        RealTask(
            "brutal-count-001", "math",
            "How many integers from 1 to 2026 inclusive are divisible by 6, "
            "by 8, or by 9 (at least one of the three)? Respond with a final "
            "line 'ANSWER: <number>'.",
            "619", numeric_checker("619"), difficulty=1.0),
        RealTask(
            "brutal-date-001", "math",
            "How many days are there from February 28, 2024 through June 15, "
            "2026 inclusive (counting both the start and end date)? Remember "
            "2024 is a leap year. Respond with a final line 'ANSWER: <number>'.",
            "839", numeric_checker("839"), difficulty=1.0),
    ]
    return {t.id: t for t in tasks}
