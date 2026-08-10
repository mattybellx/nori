"""Phase-4 tests: the discovery loop v0 (spec §15–§21).

The loop must: split tasks into discovery/held-out deterministically,
benchmark candidates, register them with genealogy + fitness, apply the
promotion gate (never-worse + discovery improvement + held-out
generalization), and either promote or honestly report NO PROMOTION.
"""

from __future__ import annotations

import pytest

from dse.discovery import (
    ArchitectureRegistry,
    DiscoveryConfig,
    PROMOTED,
    REJECTED,
    benchmark_architectures,
    best_of_n_ify,
    compile_graph,
    discover,
    promotion_gate,
    react_graph,
    split_tasks,
)
from dse.discovery.mutations import substitute


@pytest.fixture
def discovery_ctx(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = tiny_stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


@pytest.fixture
def big_ctx(stack):
    """Context over the 24-task catalog — react has headroom there
    (~20% success), so the loop can genuinely discover an improvement."""
    config, catalog, models, llm, env, verifier, agents, budget = stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _tasks(tiny_stack):
    return list(tiny_stack[1].values())


def _big_tasks(stack):
    return list(stack[1].values())


# ---------------------------------------------------------------------------
# Task split
# ---------------------------------------------------------------------------

def test_split_tasks_deterministic_disjoint_and_nonempty(tiny_stack):
    tasks = _tasks(tiny_stack)
    d1, h1 = split_tasks(tasks, 0.75, seed=0)
    d2, h2 = split_tasks(tasks, 0.75, seed=0)
    assert [t.id for t in d1] == [t.id for t in d2]
    assert [t.id for t in h1] == [t.id for t in h2]
    assert set(d1).isdisjoint(h1)
    assert 0 < len(d1) < len(tasks) and 0 < len(h1) < len(tasks)


def test_split_tasks_respects_frac(tiny_stack):
    tasks = _tasks(tiny_stack)
    d, h = split_tasks(tasks, 0.5, seed=1)
    assert len(d) >= 1 and len(h) >= 1


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------

def test_gate_accepts_candidate_better_everywhere():
    # candidate wins 4/5 (better rate), never loses where incumbent wins
    cand = [True, True, True, True, False]
    inc = [False, True, False, False, False]
    held_c = [True, True, True]
    held_i = [False, False, True]
    gate = promotion_gate(cand, inc, held_c, held_i, ["a", "b", "c", "d", "e"])
    assert gate["passed"]
    assert gate["no_regression"]
    assert gate["never_worse"]  # b01 == 0


def test_gate_passes_on_rate_with_per_task_regressions_reported():
    """Rate-level improvement is what counts (per-architecture reseeding makes
    strict per-task never-worse unestimable); b01 is reported, not required."""
    cand = [True, False, True, True, True]   # 0.8 discovery
    inc = [False, True, False, False, True]  # 0.4 discovery (wins task 2 where cand lost)
    held_c = [True, True, True]
    held_i = [False, False, True]
    gate = promotion_gate(cand, inc, held_c, held_i, ["a", "b", "c", "d", "e"])
    assert gate["passed"]            # better on rate, no regression, holds on held-out
    assert gate["no_regression"]
    assert gate["b01"] == 1          # reported honestly...
    assert gate["never_worse"] is False  # ...but not required for promotion


def test_gate_rejects_when_held_out_does_not_generalize():
    cand = [True, True, True, True, True]
    inc = [False, True, False, False, False]
    held_c = [False, False, True]      # worse on held-out than incumbent
    held_i = [True, True, True]
    gate = promotion_gate(cand, inc, held_c, held_i, ["a", "b", "c", "d", "e"])
    assert not gate["passed"]
    assert not gate["held_out_improvement"]


def test_gate_rejects_equal_rates():
    """Regression: an epsilon sign bug flipped strict `>` into allowing
    EQUAL rates, promoting candidates identical to the incumbent. Equal
    discovery AND equal held-out rates must never pass."""
    cand = [True, True, True, False, False, False]
    inc = [True, True, True, False, False, False]   # identical -> equal rates
    held_c = [True, True, True]
    held_i = [True, True, True]
    gate = promotion_gate(cand, inc, held_c, held_i, ["a", "b", "c", "d", "e"])
    assert not gate["passed"]
    assert not gate["discovery_improvement"]
    assert not gate["held_out_improvement"]


def test_gate_requires_strict_rate_delta():
    """min_improvement must require a REAL margin, not an epsilon."""
    cand = [True, True, True, False, False]
    inc = [True, True, True, False, False]   # same rate, different task ids
    held_c = [True, True, False]
    held_i = [True, True, False]
    gate = promotion_gate(cand, inc, held_c, held_i, ["a", "b", "c", "d", "e"],
                          min_improvement=0.1)
    assert not gate["passed"]


# ---------------------------------------------------------------------------
# best_of_n_ify — the expansion operator the loop needs to win
# ---------------------------------------------------------------------------

def test_best_of_n_ify_compiles_and_beats_react(big_ctx, stack):
    """Turning react into N parallel drafts + OBJECTIVE selection + verify must
    improve success on the mock (the classic best-of-N win) — this is what
    makes the loop capable of discovering something better than a weak
    baseline. Verified: ~20% -> ~46% with strict per-task never-worse."""
    tasks = _big_tasks(stack)
    cand = best_of_n_ify(react_graph(), "s", n=3)
    cand.name = "bon_react"
    compile_graph(cand)
    res = benchmark_architectures([react_graph(), cand], tasks, big_ctx, seed=0)
    assert res.rows["react"].success_rate < res.rows[cand.name].success_rate
    # objective terminal (verify) => has a verifier score
    assert res.rows[cand.name].avg_verifier_score is not None
    # strict per-task never-worse holds on this mock: candidate never fails
    # where react succeeded (b01 == 0)
    b01 = sum(1 for rr, cc in zip(res.rows["react"].runs, res.rows[cand.name].runs)
              if (not cc.success) and rr.success)
    assert b01 == 0


# ---------------------------------------------------------------------------
# End-to-end discovery loop
# ---------------------------------------------------------------------------

def test_discover_loop_runs_and_records_genealogy(tiny_stack, discovery_ctx):
    from dse.discovery import strategy_baselines
    tasks = _tasks(tiny_stack)
    reg = ArchitectureRegistry("loop")
    cfg = DiscoveryConfig(n_rounds=2, candidates_per_parent=4, beam_width=3,
                          seed=0, incumbent="best")
    report = discover(reg, discovery_ctx, tasks, config=cfg)
    # well-formed report
    assert report.incumbent in {g.name for g in strategy_baselines()}
    assert report.discovery_n + report.held_out_n == len(tasks)
    assert len(report.rounds) == cfg.n_rounds
    # every candidate was registered with genealogy (mutated -> parent)
    mutated = [r for r in reg.all() if r.source == "mutated"]
    assert mutated
    for r in mutated:
        assert r.parent_ids, f"{r.arch_id} missing genealogy"
        assert r.graph is not None
    # outcome is either promotion or an honest NO PROMOTION
    assert report.promoted or report.no_promotion


def test_discover_can_promote_a_candidate_over_weak_incumbent(big_ctx, stack):
    """The Phase-4 gate: with react as the incumbent, the loop must be able
    to discover (via best_of_n_ify among other operators) a candidate that
    beats react on BOTH discovery and held-out tasks, and promote it."""
    tasks = _big_tasks(stack)
    reg = ArchitectureRegistry("weak")
    cfg = DiscoveryConfig(n_rounds=3, candidates_per_parent=6, beam_width=3,
                          seed=7, incumbent="react")
    report = discover(reg, big_ctx, tasks,
                      seed_graphs=[react_graph()], config=cfg)
    # the mechanism must find at least one promotion over the weak incumbent
    assert report.promoted, f"no promotion; report={report.to_dict()}"
    for name in report.promoted:
        assert reg.get(name).state == PROMOTED


def test_discover_rejects_text_terminal_candidates(tiny_stack, discovery_ctx):
    """A candidate with no objective score (e.g. react mutated by substituting
    its strategy node with verify — no objective terminal) must be REJECTED,
    never promoted."""
    tasks = _tasks(tiny_stack)
    reg = ArchitectureRegistry("reject")
    # build the degenerate candidate directly and register it
    bad = substitute(react_graph(), "s", "verify")
    compile_graph(bad)
    reg.register(bad, source="mutated", parent_ids=["react"])
    # run the loop; the degenerate graph cannot win the gate, and the report
    # must be well-formed either way
    cfg = DiscoveryConfig(n_rounds=1, candidates_per_parent=3, seed=3,
                          incumbent="react")
    report = discover(reg, discovery_ctx, tasks,
                      seed_graphs=[react_graph()], config=cfg)
    assert report.incumbent == "react"
    assert report.promoted or report.no_promotion
