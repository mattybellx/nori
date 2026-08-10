"""Tests for the Phase-1 architecture-discovery foundation.

Everything is deterministic: the mock stack (tiny_stack) is seeded and the
per-(strategy, task) reseeding invariant is preserved, so the executor runs
are reproducible.
"""

from __future__ import annotations

import pytest

from dse.agent import Budget
from dse.discovery import (
    ArchitectureRegistry,
    ArchGraph,
    ArchNode,
    BENCHMARKED,
    CompileError,
    ExecutionContext,
    INDEPENDENTLY_VERIFIED,
    PROMOTED,
    compile_graph,
    default_baselines,
    novelty_against,
    react_graph,
    structural_similarity,
    synthesis_pipeline_graph,
    disagreement_pipeline_graph,
)
from dse.discovery.executor import ArchExecutor
from dse.strategies import (
    BestOfNAgent,
    ReactAgent,
    ReflexionAgent,
    SelfRefineAgent,
)


# ---------------------------------------------------------------------------
# Graph representation & serialization
# ---------------------------------------------------------------------------

def test_graph_roundtrip_dict_and_json():
    g = synthesis_pipeline_graph()
    g2 = ArchGraph.from_dict(g.to_dict())
    g3 = ArchGraph.from_json(g.to_json())
    assert g2.name == g.name and g3.name == g.name
    assert sorted(g2.nodes) == sorted(g.nodes)
    assert len(g2.edges) == len(g.edges)
    assert g2.entry == g.entry and g2.exit == g.exit
    # node params survive the round trip
    assert g2.nodes["select"].params == {"baseline": "gen1"}


def test_graph_duplicate_node_rejected():
    g = ArchGraph(name="x", entry="a")
    g.add_node(ArchNode("a", "generate"))
    with pytest.raises(ValueError):
        g.add_node(ArchNode("a", "verify"))


# ---------------------------------------------------------------------------
# Compiler validation (spec §32)
# ---------------------------------------------------------------------------

def test_compile_rejects_cycle():
    g = ArchGraph(name="cyc", entry="a")
    g.add_node(ArchNode("a", "generate"))
    g.add_node(ArchNode("b", "verify"))
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    with pytest.raises(CompileError) as exc:
        compile_graph(g)
    assert any("cycle" in p for p in exc.value.problems)


def test_compile_rejects_unknown_primitive():
    g = ArchGraph(name="bad", entry="a")
    g.add_node(ArchNode("a", "no_such_primitive"))
    with pytest.raises(CompileError) as exc:
        compile_graph(g)
    assert any("unknown primitive" in p for p in exc.value.problems)


def test_compile_rejects_unreachable_node():
    # a disconnected subgraph (b<->d cycle with no external edge) is neither a
    # source (both have in-edges) nor reachable -> rejected as unreachable
    g = ArchGraph(name="un", entry="a", exit="c")
    g.add_node(ArchNode("a", "generate"))
    g.add_node(ArchNode("b", "verify"))
    g.add_node(ArchNode("c", "identity"))
    g.add_node(ArchNode("d", "verify"))
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("d", "b")
    with pytest.raises(CompileError) as exc:
        compile_graph(g)
    assert any("unreachable" in p for p in exc.value.problems)


def test_compile_rejects_unknown_edge_target():
    g = ArchGraph(name="edge", entry="a")
    g.add_node(ArchNode("a", "generate"))
    # build the bad edge directly to bypass add_edge's own existence check
    from dse.discovery.graph import ArchEdge
    g.edges.append(ArchEdge(source="a", target="ghost"))
    with pytest.raises(CompileError) as exc:
        compile_graph(g)
    assert any("target" in p for p in exc.value.problems)


def test_compile_rejects_budget_max_nodes():
    g = ArchGraph(name="big", entry="a", params={"max_nodes": 1})
    g.add_node(ArchNode("a", "generate"))
    g.add_node(ArchNode("b", "verify"))
    g.add_edge("a", "b")
    with pytest.raises(CompileError) as exc:
        compile_graph(g)
    assert any("max_nodes" in p for p in exc.value.problems)


def test_compile_accepts_single_node_arch():
    c = compile_graph(react_graph())
    assert c.order == ["s"] and c.entry == "s" and c.exit == "s"


# ---------------------------------------------------------------------------
# Executor (deterministic on tiny_stack)
# ---------------------------------------------------------------------------

@pytest.fixture
def discovery_ctx(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = tiny_stack
    by_name = {a.name: a for a in agents}
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _task(tiny_stack, index=0):
    config, catalog, models, llm, env, verifier, agents, budget = tiny_stack
    return list(catalog.values())[index]


def test_executor_react_matches_standalone_agent(tiny_stack, discovery_ctx):
    config, catalog, models, llm, env, verifier, agents, budget = tiny_stack
    task = _task(tiny_stack)
    by_name = {a.name: a for a in agents}
    ex = ArchExecutor()

    # standalone react with the same per-(agent, task) reseed
    llm.reset(f"3:react:{task.id}")
    standalone = by_name["react"].solve(task, budget)

    # graph react with the same reseed
    llm.reset(f"3:react:{task.id}")
    rec = ex.run(compile_graph(react_graph()), task, context=discovery_ctx)

    assert rec.answer == standalone.answer
    assert rec.success == standalone.success
    assert rec.task_id == task.id
    assert any(e.primitive == "strategy" for e in rec.events)


def test_executor_best_of_n_graph_runs(tiny_stack, discovery_ctx):
    task = _task(tiny_stack, 1)
    g = default_baselines()[1]  # best_of_n
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    assert rec.architecture == "best_of_n"
    assert rec.success in (True, False)  # deterministic either way
    assert rec.tokens_total >= 0


def test_executor_strategy_reflexion_and_self_refine(tiny_stack, discovery_ctx):
    for g in (default_baselines()[2], default_baselines()[3]):  # reflexion, self_refine
        task = _task(tiny_stack, 2)
        rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
        assert rec.architecture in ("reflexion", "self_refine")
        assert rec.answer != "" or rec.success is False


def test_executor_synthesis_pipeline_on_catalog_task(tiny_stack, discovery_ctx):
    """The production never-worse pipeline expressed as a graph must produce
    an answer with all stages recorded (generate x3, score, select, synth,
    guard)."""
    task = _task(tiny_stack, 0)
    rec = ArchExecutor().run(compile_graph(synthesis_pipeline_graph()),
                             task, context=discovery_ctx)
    kinds = [e.primitive for e in rec.events]
    assert "generate" in kinds and "score_items" in kinds
    assert "selection_guard" in kinds and "synthesize" in kinds
    assert "synthesis_guard" in kinds
    assert rec.answer != ""  # guard falls back to the winner if judge is missing


def test_executor_best_of_n_ify_preserves_answer_text(tiny_stack, discovery_ctx):
    """REGRESSION (real-domain bridge): a best_of_N expansion whose exit is a
    ``verify`` node must still record the answer TEXT. The terminal Verdict
    has no text, so the executor recovers it from the last textual output on
    the path. Previously the answer was always "" — harmless while benchmarks
    measured success rate only, fatal for the real-domain free-form
    experiment where the answer text is the deliverable."""
    from dse.discovery.mutations import best_of_n_ify

    g = best_of_n_ify(react_graph(), "s", n=2)
    task = _task(tiny_stack, 1)
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    assert any(e.primitive == "verify" for e in rec.events)
    assert any(e.primitive == "extract" for e in rec.events)
    assert rec.answer != ""
    assert rec.answer is not None


def test_executor_disagreement_routing_low_path(tiny_stack, discovery_ctx):
    """threshold > 0 with agreeing drafts -> 'low' path only; the targeted
    high-compute node must NOT execute."""
    task = _task(tiny_stack, 3)
    g = disagreement_pipeline_graph(threshold=5.0)
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    assert rec.routed.get("route") == "low"
    ran = {e.node_id for e in rec.events}
    assert "gen_targeted" not in ran
    assert "synth_low" in ran and "done" in ran


def test_executor_disagreement_routing_high_path(tiny_stack, discovery_ctx):
    """threshold < 0 forces 'high' -> targeted search runs and merges."""
    task = _task(tiny_stack, 3)
    g = disagreement_pipeline_graph(threshold=-1.0)
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    assert rec.routed.get("route") == "high"
    ran = {e.node_id for e in rec.events}
    assert "gen_targeted" in ran and "synth_high" in ran
    assert "synth_low" not in ran


def test_executor_records_provenance_and_cost(tiny_stack, discovery_ctx):
    task = _task(tiny_stack, 0)
    rec = ArchExecutor().run(compile_graph(react_graph()), task,
                             context=discovery_ctx)
    d = rec.to_dict()
    assert d["architecture"] == "react"
    assert d["graph"]["name"] == "react"
    assert d["task_id"] == task.id
    assert isinstance(d["events"], list) and d["events"]
    assert d["tokens_in"] >= 0 and d["latency_s"] >= 0


def test_executor_error_recorded_not_hidden(tiny_stack, discovery_ctx):
    """A failing node must surface in the record (spec: never hide failures).

    The compiler already rejects unknown primitives at compile time (that is
    the correct gate); here we use a *registered* primitive that fails at
    runtime — a strategy node naming a strategy not present in the context.
    """
    g = ArchGraph(name="boom", entry="a")
    g.add_node(ArchNode("a", "generate"))
    g.add_node(ArchNode("b", "strategy", params={"name": "ghost_strategy"}))
    g.add_edge("a", "b")
    task = _task(tiny_stack, 0)
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    assert rec.error is not None and "ghost_strategy" in rec.error
    # the failure is recorded, not silently swallowed
    assert not rec.success


# ---------------------------------------------------------------------------
# Registry: lifecycle, genealogy, novelty (§9/§18/§29)
# ---------------------------------------------------------------------------

def test_registry_baselines_and_promotion_gate():
    reg = ArchitectureRegistry("t")
    for g in default_baselines():
        reg.register(g)  # source defaults to baseline
    assert len(reg.all()) == len(default_baselines())
    assert "react" in reg and "synthesis_pipeline" in reg

    # promotion gate: cannot promote without passing required states
    with pytest.raises(ValueError):
        reg.promote("react")
    reg.set_state("react", BENCHMARKED)
    reg.set_state("react", INDEPENDENTLY_VERIFIED)
    reg.promote("react")
    assert reg.get("react").state == PROMOTED
    assert reg.get("react").promoted_at is not None


def test_registry_genealogy():
    reg = ArchitectureRegistry("g")
    reg.register(react_graph())
    child = ArchGraph(name="react_verified", entry="s", exit="s")
    child.add_node(ArchNode("s", "strategy", params={"name": "react"}))
    reg.register(child, source="mutated", parent_ids=["react"])
    chain = reg.genealogy("react_verified")
    assert [c["arch_id"] for c in chain] == ["react", "react_verified"]
    assert chain[1]["source"] == "mutated" and chain[1]["parents"] == ["react"]


def test_registry_novelty_against_baselines():
    reg = ArchitectureRegistry("n")
    for g in default_baselines():
        reg.register(g)
    # a brand-new graph vs the baselines should have some novelty
    novel = ArchGraph(name="novel_thing", entry="a")
    novel.add_node(ArchNode("a", "generate"))
    novel.add_node(ArchNode("b", "route_disagreement"))
    novel.add_edge("a", "b")
    info = reg.novelty(novel)
    assert 0.0 <= info["novelty_score"] <= 1.0
    assert info["nearest"] is not None


def test_structural_similarity_identity_and_disjoint():
    assert structural_similarity(react_graph(), react_graph()) == 1.0
    a = ArchGraph(name="a", entry="x")
    a.add_node(ArchNode("x", "generate"))
    b = ArchGraph(name="b", entry="y")
    b.add_node(ArchNode("y", "verify"))
    assert structural_similarity(a, b) < 1.0
    assert novelty_against(react_graph(), [react_graph()])["novelty_score"] == 0.0


def test_registry_fitness_history_serializes(tiny_stack, discovery_ctx):
    reg = ArchitectureRegistry("f")
    reg.register(react_graph())
    task = _task(tiny_stack, 0)
    rec = ArchExecutor().run(compile_graph(react_graph()), task,
                             context=discovery_ctx)
    reg.record_fitness("react", rec)
    d = reg.to_dict()
    arch = next(a for a in d["architectures"] if a["arch_id"] == "react")
    assert arch["n_fitness_records"] == 1
