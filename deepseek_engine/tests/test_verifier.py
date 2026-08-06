"""Tests for verifiers and the aggregate."""

import pytest

from dse.environment import make_catalog
from dse.llm import MockLLM
from dse.verifier import (
    AggregateVerifier,
    ExactVerifier,
    LLMJudge,
    SelfConsistencyVerifier,
    TestVerifier,
    extract_answer,
)


def _catalog_and_task():
    catalog = make_catalog(seed=0, n_tasks=3, max_steps=2)
    task = catalog[next(iter(catalog))]
    return catalog, task


def _correct_solution(task):
    lines = [f"STEP {i}: {v}" for i, v in enumerate(task.correct_steps)]
    return "\n".join(lines) + f"\nANSWER: {task.gold}"


def test_extract_answer():
    assert extract_answer("foo\nANSWER: abc") == "abc"
    assert extract_answer("no answer here") is None


def test_exact_verifier():
    _, task = _catalog_and_task()
    verifier = ExactVerifier()
    ok = verifier.score(_correct_solution(task), task)
    assert ok.passed and ok.score == 1.0
    bad = verifier.score("ANSWER: wrong", task)
    assert not bad.passed and bad.score == 0.0


def test_test_verifier_detects_failed_steps():
    _, task = _catalog_and_task()
    verifier = TestVerifier(_env_for(task))
    solution = _correct_solution(task)
    report = verifier.score(solution, task)
    assert report.passed and report.score == 1.0
    # corrupt the first step
    broken = solution.replace(task.correct_steps[0], "zz", 1)
    report2 = verifier.score(broken, task)
    assert not report2.passed
    assert 0 in report2.details["failed_steps"]


def test_llm_judge_perfect_is_accurate():
    _, task = _catalog_and_task()
    models = {"cheap": _cfg(judge_accuracy=1.0)}
    llm = MockLLM({task.id: task}, models, seed=0)
    judge = LLMJudge(llm, model="cheap")
    good = judge.score(_correct_solution(task), task)
    assert good.passed and good.score >= 0.9
    bad = judge.score("ANSWER: nope", task)
    assert not bad.passed


def test_self_consistency_verifier_perfect():
    _, task = _catalog_and_task()
    models = {"cheap": _cfg(step_accuracy=1.0)}
    llm = MockLLM({task.id: task}, models, seed=0)
    vc = SelfConsistencyVerifier(llm, model="cheap", samples=3)
    verdict = vc.score(_correct_solution(task), task)
    assert verdict.score >= 0.5


def test_aggregate_verifier_weights():
    _, task = _catalog_and_task()
    exact = ExactVerifier()
    # weighted: exact dominates
    agg = AggregateVerifier([exact, exact], [0.9, 0.1])
    ok = agg.score(_correct_solution(task), task)
    assert ok.passed and abs(ok.score - 1.0) < 1e-9
    bad = agg.score("ANSWER: wrong", task)
    assert not bad.passed


def test_aggregate_requires_equal_lengths():
    with pytest.raises(ValueError):
        AggregateVerifier([ExactVerifier()], [0.5, 0.5])


def _cfg(step_accuracy=0.5, judge_accuracy=0.8):
    from dse.config import ModelConfig

    return ModelConfig(name="cheap", step_accuracy=step_accuracy, judge_accuracy=judge_accuracy)


def _env_for(task):
    from dse.environment import Environment

    return Environment({task.id: task})
