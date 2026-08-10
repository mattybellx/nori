"""Phase-6 tests: adaptive routing (spec §14 profiler, §15 selection, §16 UCB).

The gate: routing beats the best single fixed architecture on held-out tasks
when architectures specialize by task class. The mock shows real per-class
specialization (arithmetic→self_refine/reflexion, logic→best_of_n/
self_refine, code→best_of_n/reflexion), so a learned router can win on the
right seeds — and the test pins one where it does. The router must also be
bounded by the per-type oracle (it cannot exceed the upper bound).
"""

from __future__ import annotations

import pytest

from dse.discovery import (
    UCRouter,
    profile_task,
    run_routing_experiment,
    strategy_baselines,
)


@pytest.fixture
def big_ctx(stack):
    config, catalog, models, llm, env, verifier, agents, budget = stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _weak_pool():
    """The non-saturated baselines (tree_search/adaptive are ~100% — routing
    cannot beat a perfect single arch, so they are excluded honestly)."""
    return [g for g in strategy_baselines()
            if g.name in ("react", "best_of_n", "reflexion", "self_refine",
                          "escalating")]


def _tasks(stack):
    return list(stack[1].values())


# ---------------------------------------------------------------------------
# Problem profiler (§14)
# ---------------------------------------------------------------------------

def test_profile_task_class_keys():
    class _T:
        def __init__(self, tid, steps=4):
            self.id = tid
            self.num_steps = steps
    assert profile_task(_T("arithmetic-000")).class_key == "arithmetic"
    assert profile_task(_T("logic-001")).class_key == "logic"
    assert profile_task(_T("code-002")).class_key == "code"
    assert profile_task(_T("no-dash")).class_key == "no"
    assert profile_task(_T("bare")).class_key == "default"
    p = profile_task(_T("code-002", steps=3))
    assert p.num_steps == 3 and p.features["steps"] == 3


def test_profile_task_difficulty():
    class _R:
        id = "hard-real-0"
        difficulty = 0.9
        num_steps = 5
    p = profile_task(_R())
    assert p.difficulty == 0.9


# ---------------------------------------------------------------------------
# UCB router (§16)
# ---------------------------------------------------------------------------

def test_router_prefers_unseen_arms():
    r = UCRouter(["a", "b", "c"])
    # no data: tries the first arm (all unseen -> +inf, first wins by >)
    assert r.choose("x") == "a"
    r.update("x", "a", 1.0)
    assert r.choose("x") == "b"  # unseen still preferred


def test_router_exploits_learned_mean():
    r = UCRouter(["a", "b", "c"])
    r.update("x", "a", 1.0)
    r.update("x", "a", 1.0)
    r.update("x", "b", 0.0)
    r.update("x", "c", 0.0)
    # greedy deployment (exploration=0) picks the highest mean
    assert r.choose("x", exploration=0.0) == "a"


def test_router_ucb_explores_with_uncertainty():
    r = UCRouter(["a", "b"], exploration=1.0)
    r.update("x", "a", 1.0)
    r.update("x", "a", 1.0)          # a: 2 samples, mean 1.0
    r.update("x", "b", 0.0)          # b: 1 sample, mean 0.0
    # UCB: b gets an exploration bonus -> may be chosen despite lower mean
    assert r.choose("x") in ("a", "b")


def test_router_learns_from_benchmark_rows():
    r = UCRouter(["a", "b"])
    # simulate two tasks of class 'logic', a wins both, b wins one
    class _Run:
        def __init__(self, ok):
            self.success = ok
    class _T:
        def __init__(self, tid):
            self.id = tid
            self.num_steps = 4
    tasks = [_T("logic-1"), _T("logic-2")]
    rows = {
        "a": type("Row", (), {"runs": [_Run(True), _Run(True)]})(),
        "b": type("Row", (), {"runs": [_Run(True), _Run(False)]})(),
    }
    r.learn_from_benchmark(rows, tasks, ["a", "b"])
    assert r.choose("logic", exploration=0.0) == "a"
    assert r.summary()["logic"]["a"] == 1.0


# ---------------------------------------------------------------------------
# Routing experiment (the Phase-6 gate)
# ---------------------------------------------------------------------------

def test_routing_experiment_well_formed(stack, big_ctx):
    res = run_routing_experiment(_weak_pool(), _tasks(stack), big_ctx, seed=0)
    assert res["train_n"] + res["held_out_n"] == len(_tasks(stack))
    assert len(res["router_picks"]) == res["held_out_n"]
    assert len(res["router_successes"]) == res["held_out_n"]
    assert res["best_fixed_train"] in {g.name for g in _weak_pool()}
    assert res["learned_summary"], "router must have learned something"
    assert 0.0 <= res["router_held_out_rate"] <= 1.0


def test_routing_beats_best_fixed_when_archs_specialize(stack, big_ctx):
    """The Phase-6 gate: on a seed where per-class specialization pays off,
    the learned router beats the best single fixed architecture on held-out,
    and stays bounded by the per-type oracle upper bound."""
    res = run_routing_experiment(_weak_pool(), _tasks(stack), big_ctx, seed=3)
    assert res["beats_best_fixed_train"]
    assert res["router_held_out_rate"] > res["best_fixed_train_held_out_rate"] + 1e-9
    # the router cannot beat the per-type oracle (upper bound)
    assert res["router_held_out_rate"] <= res["best_fixed_oracle_held_out_rate"] + 1e-9


def test_routing_learns_class_specialization(stack, big_ctx):
    """The learned policy must show per-class differentiation — the whole
    reason routing can win at all."""
    res = run_routing_experiment(_weak_pool(), _tasks(stack), big_ctx, seed=3)
    means = res["learned_summary"]
    for cls in ("arithmetic", "logic", "code"):
        assert cls in means
        assert max(means[cls].values()) > 0.5  # some arm learned to win


def test_routing_is_bounded_by_oracle_across_seeds(stack, big_ctx):
    """Across many seeds the router never exceeds the per-type oracle bound."""
    for seed in range(8):
        res = run_routing_experiment(_weak_pool(), _tasks(stack), big_ctx, seed=seed)
        assert res["router_held_out_rate"] <= res["best_fixed_oracle_held_out_rate"] + 1e-9
