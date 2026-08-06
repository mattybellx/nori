"""Code-generation suite with EXECUTION-based checking (the missing piece).

The text math/logic suites (real, hard-real, hard-tuned) all hit a 100% ceiling
on deepseek-v4-flash: single-shot solves them. Code generation is the one task
class where single-shot genuinely fails — code that "looks right" frequently
fails hidden test cases — and where verify-and-repair (reflexion / self_refine)
has real signal to work with: the checker runs the model's code and returns
concrete FAIL got=... want=... messages.

**Execution model (honest):** the model's code is executed with
``subprocess.run([sys.executable, "-c", harness])`` under a timeout, in a temp
cwd, with stdout redirected. This is a LOCAL single-user tool: this is a
resource timeout + temp-dir guardrail, NOT a security sandbox. Do NOT use the
``code`` suite in a hosted/multi-tenant setting without OS-level isolation
(containers/seccomp). That limitation is documented in LIMITATIONS.md.

Tasks mirror the ``RealTask`` interface so the existing strategies, the
``RealTaskVerifier`` and the MockLLM consume them unchanged; the checker is a
``code_checker`` built from the task's test cases.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
import tempfile
from typing import Callable

from .realtasks import RealTask


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------
def extract_code(text: str) -> str:
    """Pull the model's Python code out of its answer.

    Prefers a fenced ```python ...``` block; falls back to the whole text
    (prompts instruct the model to output only code, so the fallback is safe
    enough for a local tool).
    """
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# The sandboxed (timeout-only) checker
# ---------------------------------------------------------------------------
_HARNESS = (
    "import io, contextlib\n"
    "__DSE_CODE__\n"
    "_TESTS = %(tests)r\n"
    "_out = []\n"
    "for _args, _want in _TESTS:\n"
    "    try:\n"
    "        _buf = io.StringIO()\n"
    "        with contextlib.redirect_stdout(_buf):\n"
    "            _got = str(%(func)s(*_args))\n"
    "    except Exception:\n"
    "        _got = 'ERROR'\n"
    "    _out.append('PASS' if _got == _want else 'FAIL got=%%r want=%%r' %% (_got, _want))\n"
    "print('\\n'.join(_out))\n"
)


def code_checker(
    func_name: str,
    tests: list[tuple[tuple, str]],
    timeout: float = 10.0,
) -> Callable[[str], tuple[bool, str]]:
    """Build a deterministic checker that RUNS the model's code on ``tests``.

    ``tests`` is a list of ``((args...), expected_str)``. Returns
    ``(passed, feedback)`` where feedback contains the concrete failing cases
    (``FAIL got=... want=...``) so repair strategies get real signal.
    """
    # Format the template ONCE (tests/func are controlled by us); the model's
    # code is injected afterwards with a literal replace so it can contain any
    # characters (% , braces, quotes) safely. ``%(tests)r`` renders the list
    # literal directly (pass the list, NOT its repr).
    template = _HARNESS % {"tests": tests, "func": func_name}

    def check(text: str) -> tuple[bool, str]:
        code = extract_code(text)
        if not code:
            return False, "no code found in the response"
        if len(code) > 20000:
            return False, "generated code too large (>20000 chars)"
        runner = template.replace("__DSE_CODE__", code)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", runner],
                capture_output=True, text=True,
                timeout=timeout, cwd=tempfile.gettempdir(),
            )
        except subprocess.TimeoutExpired:
            return False, f"execution timed out (> {timeout:.0f}s)"
        except OSError as exc:
            return False, f"could not run interpreter: {exc}"
        if proc.returncode != 0:
            return False, "runtime error: " + (proc.stderr or "").strip()[:300]
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        if len(lines) != len(tests):
            return False, (
                f"expected {len(tests)} result lines, got {len(lines)}"
            )
        fails = [ln for ln in lines if not ln.startswith("PASS")]
        if not fails:
            return True, ""
        return False, "; ".join(fails[:3])

    return check


# ---------------------------------------------------------------------------
# The suite (tasks chosen because single-shot code often fails hidden tests)
# ---------------------------------------------------------------------------
def _c() -> list[RealTask]:
    return [
        RealTask(
            "code-fizzbuzz", "code",
            "Write a Python function named fizzbuzz(n) that returns the string "
            "'Fizz' if n is divisible by 3, 'Buzz' if divisible by 5, "
            "'FizzBuzz' if divisible by both, otherwise the number as a "
            "string. Output ONLY the function definition in a ```python code "
            "block.",
            "fizzbuzz",
            code_checker("fizzbuzz", [
                ((1,), "1"), ((2,), "2"), ((3,), "Fizz"),
                ((5,), "Buzz"), ((15,), "FizzBuzz"),
            ]), difficulty=0.85),
        RealTask(
            "code-compress", "code",
            "Write a Python function named compress(s) that performs run-"
            "length encoding: return a string where each run of identical "
            "characters is replaced by the character followed by its count "
            "(e.g. 'aabcccccaaa' -> 'a2b1c5a3'). Output ONLY the function "
            "definition in a ```python code block.",
            "compress",
            code_checker("compress", [
                (("aabcccccaaa",), "a2b1c5a3"), (("abc",), "a1b1c1"),
                (("aaaa",), "a4"), (("",), ""),
            ]), difficulty=0.9),
        RealTask(
            "code-palindrome", "code",
            "Write a Python function named is_palindrome(s) that returns True "
            "if s is a palindrome ignoring case, spaces and punctuation, "
            "else False. Output ONLY the function definition in a ```python "
            "code block.",
            "is_palindrome",
            code_checker("is_palindrome", [
                (("A man, a plan, a canal: Panama",), "True"),
                (("race a car",), "False"),
                (("abba",), "True"), (("",), "True"),
            ]), difficulty=0.85),
        RealTask(
            "code-two-sum", "code",
            "Write a Python function named two_sum(nums, target) that returns "
            "a list of the two indices whose values sum to target. The answer "
            "is guaranteed to exist; return the indices in ascending order. "
            "Output ONLY the function definition in a ```python code block.",
            "two_sum",
            code_checker("two_sum", [
                (([2, 7, 11, 15], 9), "[0, 1]"),
                (([3, 2, 4], 6), "[1, 2]"),
                (([3, 3], 6), "[0, 1]"),
            ]), difficulty=0.85),
        RealTask(
            "code-reverse-words", "code",
            "Write a Python function named reverse_words(s) that reverses the "
            "order of words in s, where words are separated by any amount of "
            "whitespace, and returns them joined by single spaces with no "
            "leading or trailing space. Output ONLY the function definition "
            "in a ```python code block.",
            "reverse_words",
            code_checker("reverse_words", [
                (("the sky is blue",), "blue is sky the"),
                (("  hello world  ",), "world hello"),
                (("a good   example",), "example good a"),
            ]), difficulty=0.85),
        RealTask(
            "code-longest-substring", "code",
            "Write a Python function named length_of_longest_substring(s) "
            "that returns the length of the longest substring without "
            "repeating characters. Output ONLY the function definition in a "
            "```python code block.",
            "length_of_longest_substring",
            code_checker("length_of_longest_substring", [
                (("abcabcbb",), "3"), (("bbbbb",), "1"),
                (("pwwkew",), "3"), (("",), "0"),
            ]), difficulty=0.9),
        RealTask(
            "code-min-coins", "code",
            "Write a Python function named min_coins(coins, amount) that "
            "returns the minimum number of coins (from the list coins) needed "
            "to make amount, or -1 if it is impossible. Output ONLY the "
            "function definition in a ```python code block.",
            "min_coins",
            code_checker("min_coins", [
                (([1, 2, 5], 11), "3"), (([2], 3), "-1"),
                (([1], 0), "0"), (([1, 5, 10], 18), "5"),
            ]), difficulty=0.95),
        RealTask(
            "code-binary-search", "code",
            "Write a Python function named binary_search(nums, target) that "
            "returns the index of target in the sorted list nums, or -1 if "
            "absent. Output ONLY the function definition in a ```python code "
            "block.",
            "binary_search",
            code_checker("binary_search", [
                (([1, 3, 5, 7, 9], 5), "2"), (([1, 3, 5, 7, 9], 4), "-1"),
                (([2], 2), "0"), (([], 1), "-1"),
            ]), difficulty=0.85),
    ]


def make_code_catalog(seed: int = 0) -> dict[str, RealTask]:
    """Build the code-execution suite."""
    return {task.id: task for task in _c()}
