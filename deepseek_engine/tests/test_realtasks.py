"""Tests for the real (natural-language) task suite and its wiring."""

import pytest

from dse.config import ModelConfig, default_flags
from dse.factory import build_stack
from dse.llm import MockLLM
from dse.realtasks import REAL_SUITE_AGENTS, make_real_catalog
from dse.verifier import RealTaskVerifier


def test_real_catalog_checkers_pass_gold_and_reject_wrong():
    catalog = make_real_catalog()
    assert len(catalog) >= 10
    for task in catalog.values():
        assert task.num_steps == 1
        assert task.correct_steps == (task.gold,)
        passed, _ = task.check(f"STEP 0: {task.gold}\nANSWER: {task.gold}")
        assert passed, task.id
        wrong, _ = task.check("ANSWER: definitely-wrong")
        assert not wrong, task.id


def test_real_task_verifier():
    catalog = make_real_catalog()
    verifier = RealTaskVerifier()
    task = catalog["math-001"]
    ok = verifier.score("ANSWER: 391", task)
    assert ok.passed and ok.score == 1.0
    bad = verifier.score("ANSWER: 999", task)
    assert not bad.passed and bad.score == 0.0
    assert bad.details["feedback"]  # free-text feedback for retries


def test_mock_llm_solves_real_task_with_perfect_model():
    catalog = make_real_catalog()
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=1.0)}
    llm = MockLLM(catalog, models, seed=0)
    task = catalog["math-001"]
    out = llm.complete([{"role": "system", "content": f"TASK_ID: {task.id}"}], model="cheap")
    assert task.check(out.text)[0]


def test_real_suite_stack_excludes_per_step_agents():
    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="real"
    )
    names = {a.name for a in agents}
    assert names == set(REAL_SUITE_AGENTS)
    assert "tree_search" not in names
    assert "adaptive" not in names
    assert "escalating_per_step" not in names


def test_real_suite_full_mock_run_is_wellformed():
    """End-to-end: the real suite runs through the harness with the MockLLM
    (smoke test for wiring; real numbers come from a real provider)."""
    from dse.benchmarks.harness import run_benchmark

    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="real"
    )
    results = run_benchmark(agents, catalog, seed=0, models=models, budget=budget, llm=llm)
    assert set(results) == set(REAL_SUITE_AGENTS)
    for name, runs in results.items():
        assert len(runs) == len(catalog)
        assert all(isinstance(r.success, bool) for r in runs)


# --- hard-real suite ---------------------------------------------------------


def test_hard_catalog_checkers_pass_gold_and_reject_wrong():
    from dse.hardtasks import make_hard_catalog

    catalog = make_hard_catalog()
    assert len(catalog) >= 10
    for task in catalog.values():
        assert task.num_steps == 1
        assert task.difficulty >= 0.75  # genuinely harder than the base suite
        passed, _ = task.check(f"ANSWER: {task.gold}")
        assert passed, task.id
        wrong, _ = task.check("ANSWER: definitely-wrong")
        assert not wrong, task.id


def test_hard_tasks_are_distinct_from_base_real_suite():
    from dse.hardtasks import make_hard_catalog
    from dse.realtasks import make_real_catalog

    hard_ids = set(make_hard_catalog())
    real_ids = set(make_real_catalog())
    assert hard_ids.isdisjoint(real_ids)


def test_hard_suite_stack_excludes_per_step_agents():
    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="hard-real"
    )
    names = {a.name for a in agents}
    assert names == set(REAL_SUITE_AGENTS)
    assert "tree_search" not in names
    assert "adaptive" not in names
    assert "escalating_per_step" not in names


def test_hard_suite_full_mock_run_is_wellformed():
    from dse.benchmarks.harness import run_benchmark

    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="hard-real"
    )
    results = run_benchmark(agents, catalog, seed=0, models=models, budget=budget, llm=llm)
    assert set(results) == set(REAL_SUITE_AGENTS)
    for name, runs in results.items():
        assert len(runs) == len(catalog)
        assert all(isinstance(r.success, bool) for r in runs)


# --- hard-tuned suite ---------------------------------------------------------


def test_hard_tuned_catalog_checkers_pass_gold_and_reject_wrong():
    from dse.hardtasks import make_hard_tuned_catalog

    catalog = make_hard_tuned_catalog()
    assert len(catalog) >= 10
    for task in catalog.values():
        assert task.difficulty >= 0.8
        passed, _ = task.check(f"ANSWER: {task.gold}")
        assert passed, task.id
        wrong, _ = task.check("ANSWER: definitely-wrong")
        assert not wrong, task.id


def test_hard_tuned_suite_stack_excludes_per_step_agents():
    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="hard-tuned"
    )
    names = {a.name for a in agents}
    assert names == set(REAL_SUITE_AGENTS)
    assert "tree_search" not in names


def test_hard_tuned_full_mock_run_is_wellformed():
    from dse.benchmarks.harness import run_benchmark

    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=0, suite="hard-tuned"
    )
    results = run_benchmark(agents, catalog, seed=0, models=models, budget=budget, llm=llm)
    assert set(results) == set(REAL_SUITE_AGENTS)
    for name, runs in results.items():
        assert len(runs) == len(catalog)
        assert all(isinstance(r.success, bool) for r in runs)
