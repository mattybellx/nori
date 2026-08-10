"""Phase-2 tests: baseline graphs must reproduce the strategies they wrap.

This is the gate BEFORE any discovery: if a graph-executed baseline diverges
from the strategy it wraps, every later "discovered improvement" would be an
executor artifact. All tests are deterministic on the seeded mock stack with
the per-(strategy, task) reseeding invariant preserved.
"""

from __future__ import annotations

import pytest

from dse.discovery import (
    ArchGraph,
    ArchNode,
    benchmark_architectures,
    best_of_n_graph,
    default_baselines,
    react_graph,
    strategy_baselines,
    synthesis_pipeline_graph,
    validate_all_baselines,
    validate_equivalence,
)


@pytest.fixture
def discovery_ctx(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = tiny_stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _tasks(tiny_stack):
    return list(tiny_stack[1].values())


# ---------------------------------------------------------------------------
# Baseline coverage
# ---------------------------------------------------------------------------

def test_strategy_baselines_cover_the_agent_set(discovery_ctx):
    names = {g.nodes["s"].params["name"] for g in strategy_baselines()}
    assert {"react", "best_of_n", "reflexion", "self_refine", "tree_search",
            "escalating", "adaptive"} <= names
    # every baseline graph is a single strategy node named after itself
    for g in strategy_baselines():
        assert g.nodes["s"].params["name"] == g.name


# ---------------------------------------------------------------------------
# Equivalence: graph == wrapped strategy (the Phase-2 gate)
# ---------------------------------------------------------------------------

def test_equivalence_react(tiny_stack, discovery_ctx):
    tasks = _tasks(tiny_stack)
    report = validate_equivalence(react_graph(), "react", discovery_ctx, tasks, seed=3)
    assert report.passes, report.summary()
    assert report.n_tasks == len(tasks)


def test_equivalence_all_strategy_baselines(tiny_stack, discovery_ctx):
    """Every strategy baseline graph must reproduce its strategy exactly
    (answer, success, tokens) under identical reseeds."""
    reports = validate_all_baselines(discovery_ctx, _tasks(tiny_stack), seed=3)
    assert reports, "no baseline graphs were validated (agents missing?)"
    for rep in reports:
        assert rep.passes, rep.summary()


def test_equivalence_detects_divergence(tiny_stack, discovery_ctx):
    """The validator must actually catch a graph that does NOT reproduce the
    strategy — here a 'react'-named graph that secretly runs best_of_n."""
    g = ArchGraph(name="react", entry="s", exit="s")
    g.add_node(ArchNode("s", "strategy", params={"name": "best_of_n"}))
    report = validate_equivalence(g, "react", discovery_ctx, _tasks(tiny_stack)[:3], seed=3)
    assert not report.passes
    fields = {m.field for m in report.mismatches}
    assert "answer" in fields or "success" in fields or "tokens_in" in fields


# ---------------------------------------------------------------------------
# Head-to-head benchmark (the discovery evaluation loop)
# ---------------------------------------------------------------------------

def test_benchmark_architectures_shape_and_determinism(tiny_stack, discovery_ctx):
    tasks = _tasks(tiny_stack)[:4]
    graphs = [react_graph(), best_of_n_graph()]
    res1 = benchmark_architectures(graphs, tasks, discovery_ctx, seed=0)
    res2 = benchmark_architectures(graphs, tasks, discovery_ctx, seed=0)
    assert set(res1.rows) == {"react", "best_of_n"}
    assert res1.skipped == {}
    for name, row in res1.rows.items():
        d = row.to_dict()
        assert d["n"] == len(tasks)
        assert 0.0 <= d["success_rate"] <= 1.0
        assert d["avg_tokens"] >= 0 and d["total_latency_s"] >= 0
        # deterministic: same seed -> identical results
        assert res2.rows[name].to_dict() == d


def test_benchmark_skips_unavailable_strategies(tiny_stack, discovery_ctx):
    """A graph needing an Agent not in the context must be skipped, not run
    as a fake failure row (the multi_agent case on the default stack)."""
    from dse.discovery import multi_agent_graph
    tasks = _tasks(tiny_stack)[:2]
    res = benchmark_architectures([multi_agent_graph(), react_graph()],
                                  tasks, discovery_ctx, seed=0)
    if "multi_agent" not in discovery_ctx.agents:
        assert "multi_agent" in res.skipped
        assert "multi_agent" not in res.rows
    assert "react" in res.rows


def test_benchmark_includes_atomic_pipelines(tiny_stack, discovery_ctx):
    tasks = _tasks(tiny_stack)[:3]
    res = benchmark_architectures([synthesis_pipeline_graph(), react_graph()],
                                  tasks, discovery_ctx, seed=0)
    assert "synthesis_pipeline" in res.rows and "react" in res.rows
    assert res.rows["synthesis_pipeline"].n == len(tasks)


def test_benchmark_all_default_baselines_compiles_and_runs(tiny_stack, discovery_ctx):
    """Phase-2 sanity: every baseline in the initial library either benchmarks
    or is explicitly skipped — never silently faked."""
    res = benchmark_architectures(default_baselines(), _tasks(tiny_stack)[:2],
                                  discovery_ctx, seed=0)
    assert set(res.rows) | set(res.skipped) == {g.name for g in default_baselines()}
