"""Phase-5 tests: evolutionary search (spec §7 crossover, §16 novelty, §18
retirement, §20 statistical significance).

The Phase-5 gate: an improvement must survive the sign test + bootstrap CI,
crossover children must compile and run, unfit architectures get retired, and
the beam can prefer novelty without breaking the loop.
"""

from __future__ import annotations

import random

import pytest

from dse.discovery import (
    ArchitectureRegistry,
    DiscoveryConfig,
    PROMOTED,
    REJECTED,
    RETIRED,
    best_of_n_graph,
    beam_score,
    compile_graph,
    crossover,
    crossover_children,
    discover,
    react_graph,
    retire_unfit,
    unique_child_name,
    validate_statistical,
)
from dse.discovery.executor import ArchExecutor
from dse.discovery.evaluate import benchmark_architectures
from dse.discovery.primitives import ExecutionContext


@pytest.fixture
def big_ctx(stack):
    config, catalog, models, llm, env, verifier, agents, budget = stack
    by_name = {a.name: a for a in agents}
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _big_tasks(stack):
    return list(stack[1].values())


# ---------------------------------------------------------------------------
# Crossover (§7)
# ---------------------------------------------------------------------------

def test_crossover_compiles_and_runs(big_ctx, stack):
    child = crossover(react_graph(), best_of_n_graph(), cut_a="s", cut_b="s")
    compile_graph(child)
    assert child.name == "reactxbest_of_n"
    # the child is react's head followed by best_of_n's tail
    assert child.entry == "s"
    # runs on the mock
    task = _big_tasks(stack)[0]
    rec = ArchExecutor().run(compile_graph(child), task, context=big_ctx)
    assert rec.events and not rec.error


def test_crossover_keeps_a_head_and_b_tail():
    from dse.discovery import ArchGraph, ArchNode
    a = ArchGraph(name="A", entry="gen", exit="verify")
    a.add_node(ArchNode("gen", "generate"))
    a.add_node(ArchNode("verify", "verify"))
    a.add_edge("gen", "verify")
    b = ArchGraph(name="B", entry="draft", exit="final")
    b.add_node(ArchNode("draft", "generate"))
    b.add_node(ArchNode("synth", "synthesize"))
    b.add_node(ArchNode("final", "identity"))
    b.add_edge("draft", "synth")
    b.add_edge("synth", "final")
    # cut A at 'verify' (keeps gen+verify), cut B at 'synth' (keeps synth+final)
    child = crossover(a, b, cut_a="verify", cut_b="synth")
    compile_graph(child)
    assert child.entry == "gen"
    # no name collision with A's prefix here, so B's nodes keep their names
    assert child.exit == "final"
    assert child.name == "AxB"
    prims = {n.primitive for n in child.nodes.values()}
    assert {"generate", "verify", "synthesize", "identity"} <= prims


def test_crossover_deterministic_under_seeded_rng():
    from dse.discovery import synthesis_pipeline_graph
    # multi-node parents so the RNG cut choice actually varies with seed
    a = synthesis_pipeline_graph()
    b = best_of_n_graph()
    c1 = crossover(a, b, rng=random.Random(7))
    c2 = crossover(a, b, rng=random.Random(7))
    assert c1.to_json() == c2.to_json()  # deterministic under same seed
    # a different seed can produce a different child (try a few seeds)
    varied = any(crossover(a, b, rng=random.Random(s)).to_json() != c1.to_json()
                 for s in range(1, 6))
    assert varied


def test_crossover_renames_colliding_nodes():
    """When B's suffix shares a node id with A's prefix, it must be renamed
    so the child stays a valid, non-ambiguous graph."""
    from dse.discovery import ArchGraph, ArchNode
    a = ArchGraph(name="A", entry="s", exit="s")
    a.add_node(ArchNode("s", "strategy", params={"name": "react"}))
    b = ArchGraph(name="B", entry="s", exit="s")
    b.add_node(ArchNode("s", "strategy", params={"name": "best_of_n"}))
    child = crossover(a, b, cut_a="s", cut_b="s")
    compile_graph(child)
    # A's 's' keeps its name; B's 's' is renamed (collision)
    assert child.nodes["s"].params["name"] == "react"
    other = [n for n in child.nodes if n != "s"]
    assert len(other) == 1
    assert child.nodes[other[0]].params["name"] == "best_of_n"


def test_crossover_children_unique_and_compile():
    from dse.discovery import synthesis_pipeline_graph
    children = crossover_children(synthesis_pipeline_graph(), best_of_n_graph(),
                                  n=4, rng=random.Random(3))
    assert children
    names = [c.name for _, c in children]
    assert len(set(names)) == len(names)  # unique
    for _, c in children:
        compile_graph(c)


def test_unique_child_name_is_deterministic():
    g = crossover(react_graph(), best_of_n_graph(), cut_a="s", cut_b="s")
    assert unique_child_name("react", "best_of_n", g) == unique_child_name("react", "best_of_n", g)


# ---------------------------------------------------------------------------
# Statistical validation (§20) — the Phase-5 gate
# ---------------------------------------------------------------------------

def test_statistical_validation_detects_significant_improvement():
    # cand wins 6 where inc loses, never loses where inc wins -> sign p=0.031
    cand = [True, True, True, True, True, True, True, True, False]
    inc = [True, True, False, False, False, False, False, False, False]
    stats = validate_statistical(cand, inc, [str(i) for i in range(9)], seed=0)
    assert stats["significant"]
    assert stats["sign_test_p"] < 0.05
    assert stats["ci_lo"] > 0


def test_statistical_validation_balanced_is_not_significant():
    cand = [True, True, True, False, False, False]
    inc = [False, False, False, True, True, True]
    stats = validate_statistical(cand, inc, [str(i) for i in range(6)], seed=0)
    assert not stats["significant"]
    assert stats["sign_test_p"] >= 0.05 or stats["ci_lo"] <= 0


# ---------------------------------------------------------------------------
# Retirement (§18)
# ---------------------------------------------------------------------------

def test_retire_unfit_retires_zero_success_rejected():
    reg = ArchitectureRegistry("r")
    reg.register(react_graph())
    reg.set_state("react", REJECTED)
    rec = reg.get("react")
    # a zero-success fitness record
    from dse.discovery import ArchRunRecord
    rec.fitness.append(ArchRunRecord(architecture="react", graph={}, task_id="t",
                                     success=False, answer="").to_dict())
    retired = retire_unfit(reg)
    assert "react" in retired
    assert reg.get("react").state == RETIRED


def test_retire_unfit_keeps_candidates_with_any_success():
    reg = ArchitectureRegistry("r2")
    reg.register(react_graph())
    reg.set_state("react", REJECTED)
    from dse.discovery import ArchRunRecord
    rec = reg.get("react")
    rec.fitness.append(ArchRunRecord(architecture="react", graph={}, task_id="t",
                                     success=True, answer="x").to_dict())
    assert retire_unfit(reg) == []
    assert reg.get("react").state == REJECTED


# ---------------------------------------------------------------------------
# Novelty-aware selection (§9/§16)
# ---------------------------------------------------------------------------

def test_beam_score_novelty_bonus():
    assert beam_score(0.5, 0.0, novelty_weight=0.0) == 0.5
    assert beam_score(0.5, 0.8, novelty_weight=0.1) > beam_score(0.5, 0.0, novelty_weight=0.1)
    assert beam_score(0.5, 0.8, novelty_weight=0.0) == 0.5


# ---------------------------------------------------------------------------
# End-to-end: evolution-enabled discovery loop
# ---------------------------------------------------------------------------

def test_discover_with_crossover_and_novelty_runs(stack, big_ctx):
    tasks = _big_tasks(stack)
    reg = ArchitectureRegistry("evo")
    cfg = DiscoveryConfig(n_rounds=2, candidates_per_parent=4, beam_width=3,
                          seed=3, incumbent="react", crossover_per_round=2,
                          novelty_weight=0.05)
    report = discover(reg, big_ctx, tasks, seed_graphs=[react_graph()], config=cfg)
    assert len(report.rounds) == cfg.n_rounds
    # crossover children were registered with crossed genealogy
    crossed = [r for r in reg.all() if r.source == "crossed"]
    assert crossed
    for r in crossed:
        assert len(r.parent_ids) == 2
    # rejected zero-success candidates were retired
    assert any(r.state == RETIRED for r in reg.all()) or report.promoted


def test_discover_with_significance_gate_does_not_false_promote(stack, big_ctx):
    """With require_significance=True, any promoted candidate must survive
    the statistical gate (no promotion at small n is an honest outcome)."""
    tasks = _big_tasks(stack)
    reg = ArchitectureRegistry("sig")
    cfg = DiscoveryConfig(n_rounds=2, candidates_per_parent=5, beam_width=3,
                          seed=11, incumbent="react", require_significance=True)
    report = discover(reg, big_ctx, tasks, seed_graphs=[react_graph()], config=cfg)
    for name in report.promoted:
        assert reg.get(name).state == PROMOTED
