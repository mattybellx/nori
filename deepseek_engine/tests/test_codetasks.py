"""Tests for the code-execution suite (runs real code through the checker)."""

import pytest

from dse.codetasks import extract_code, make_code_catalog
from dse.factory import build_stack
from dse.realtasks import REAL_SUITE_AGENTS


def test_extract_code_prefers_fenced_block():
    text = "Here is my code:\n```python\ndef f():\n    return 1\n```\nDone."
    assert extract_code(text) == "def f():\n    return 1"
    assert extract_code("def g(): return 2") == "def g(): return 2"


def test_code_checker_passes_correct_code():
    from dse.codetasks import code_checker

    check = code_checker("fizzbuzz", [
        ((1,), "1"), ((3,), "Fizz"), ((5,), "Buzz"), ((15,), "FizzBuzz"),
    ])
    good = (
        "def fizzbuzz(n):\n"
        "    if n % 15 == 0: return 'FizzBuzz'\n"
        "    if n % 3 == 0: return 'Fizz'\n"
        "    if n % 5 == 0: return 'Buzz'\n"
        "    return str(n)"
    )
    passed, feedback = check(f"```python\n{good}\n```")
    assert passed, feedback


def test_code_checker_rejects_wrong_code_with_concrete_feedback():
    from dse.codetasks import code_checker

    check = code_checker("fizzbuzz", [((3,), "Fizz"), ((5,), "Buzz")])
    wrong = "def fizzbuzz(n):\n    return str(n)"  # never Fizz/Buzz
    passed, feedback = check(wrong)
    assert not passed
    assert "FAIL" in feedback and "got=" in feedback and "want=" in feedback


def test_code_checker_reports_runtime_errors():
    from dse.codetasks import code_checker

    check = code_checker("f", [((1,), "1")])
    broken = "def f(n):\n    return n + undefined_name"
    passed, feedback = check(broken)
    assert not passed
    # the harness catches per-test exceptions and reports them concretely
    # (that granular feedback is exactly what repair strategies consume)
    assert "FAIL" in feedback and "ERROR" in feedback


def test_code_checker_times_out():
    from dse.codetasks import code_checker

    check = code_checker("f", [((1,), "1")], timeout=0.5)
    infinite = "def f(n):\n    while True:\n        pass"
    passed, feedback = check(infinite)
    assert not passed
    assert "timed out" in feedback


def test_code_catalog_checkers_pass_gold_and_reject_wrong():
    catalog = make_code_catalog()
    assert len(catalog) >= 8
    golds = {
        "code-fizzbuzz": "def fizzbuzz(n):\n"
                         "    if n % 15 == 0: return 'FizzBuzz'\n"
                         "    if n % 3 == 0: return 'Fizz'\n"
                         "    if n % 5 == 0: return 'Buzz'\n"
                         "    return str(n)",
        "code-compress": "def compress(s):\n"
                         "    if not s: return ''\n"
                         "    out, run = [], 1\n"
                         "    for i in range(1, len(s) + 1):\n"
                         "        if i < len(s) and s[i] == s[i - 1]:\n"
                         "            run += 1\n"
                         "        else:\n"
                         "            out.append(s[i - 1] + str(run))\n"
                         "            run = 1\n"
                         "    return ''.join(out)",
        "code-palindrome": "def is_palindrome(s):\n"
                           "    t = ''.join(c for c in s.lower() if c.isalnum())\n"
                           "    return t == t[::-1]",
        "code-two-sum": "def two_sum(nums, target):\n"
                        "    seen = {}\n"
                        "    for i, v in enumerate(nums):\n"
                        "        if target - v in seen:\n"
                        "            return sorted([seen[target - v], i])\n"
                        "        seen[v] = i",
        "code-reverse-words": "def reverse_words(s):\n"
                              "    return ' '.join(s.split()[::-1])",
        "code-longest-substring": "def length_of_longest_substring(s):\n"
                                  "    seen, start, best = {}, 0, 0\n"
                                  "    for i, c in enumerate(s):\n"
                                  "        if c in seen and seen[c] >= start:\n"
                                  "            start = seen[c] + 1\n"
                                  "        seen[c] = i\n"
                                  "        best = max(best, i - start + 1)\n"
                                  "    return best",
        "code-min-coins": "def min_coins(coins, amount):\n"
                          "    INF = float('inf')\n"
                          "    dp = [0] + [INF] * amount\n"
                          "    for a in range(1, amount + 1):\n"
                          "        for c in coins:\n"
                          "            if c <= a:\n"
                          "                dp[a] = min(dp[a], dp[a - c] + 1)\n"
                          "    return dp[amount] if dp[amount] != INF else -1",
        "code-binary-search": "def binary_search(nums, target):\n"
                              "    lo, hi = 0, len(nums) - 1\n"
                              "    while lo <= hi:\n"
                              "        mid = (lo + hi) // 2\n"
                              "        if nums[mid] == target: return mid\n"
                              "        if nums[mid] < target: lo = mid + 1\n"
                              "        else: hi = mid - 1\n"
                              "    return -1",
    }
    for task in catalog.values():
        gold = golds[task.id]
        passed, feedback = task.check(f"```python\n{gold}\n```")
        assert passed, f"{task.id}: {feedback}"
        wrong, _ = task.check("def wrong(): return 0")
        assert not wrong, task.id


def test_code_suite_stack_excludes_per_step_agents():
    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="code"
    )
    names = {a.name for a in agents}
    assert names == set(REAL_SUITE_AGENTS)
    assert "tree_search" not in names
    assert "adaptive" not in names


def test_code_suite_mock_run_is_wellformed():
    from dse.benchmarks.harness import run_benchmark

    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="code"
    )
    results = run_benchmark(agents, catalog, seed=0, models=models, budget=budget, llm=llm)
    assert set(results) == set(REAL_SUITE_AGENTS)
    for name, runs in results.items():
        assert len(runs) == len(catalog)
        assert all(isinstance(r.success, bool) for r in runs)
