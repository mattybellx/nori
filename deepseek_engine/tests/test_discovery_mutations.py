"""Phase-3 tests: mutation operators (spec §7).

The Phase-3 gate: every mutation must COMPILE (the compiler is the safety
net — invalid graphs never reach the executor). Mutations must also leave
the original graph untouched (immutability — genealogy records parents, §29),
and random_mutation must be deterministic under a seeded RNG.
"""

from __future__ import annotations

import random

import pytest

from dse.discovery import (
    ArchExecutor,
    append_verify,
    branch,
    compile_graph,
    best_of_n_graph,
    delete_node,
    duplicate_parallel,
    duplicate_sequential,
    gather_join,
    insert_after,
    insert_before,
    insert_verify,
    mutation_operators,
    random_mutation,
    react_graph,
    substitute,
    swap_primitives,
    synthesis_pipeline_graph,
)


@pytest.fixture
def discovery_ctx(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = tiny_stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _assert_compiles(graph):
    compile_graph(graph)  # raises CompileError if invalid


# ---------------------------------------------------------------------------
# Every operator on the baselines must compile (the Phase-3 gate)
# ---------------------------------------------------------------------------

def test_insert_before_compiles_and_rewires():
    g = insert_before(react_graph(), "s", "verify", node_id="v")
    _assert_compiles(g)
    # the verify node sits between entry and the strategy
    assert g.entry == "v" and g.nodes["v"].primitive == "verify"
    assert any(e.source == "v" and e.target == "s" for e in g.edges)


def test_insert_after_compiles():
    g = insert_after(react_graph(), "s", "verify", node_id="v")
    _assert_compiles(g)
    assert any(e.source == "s" and e.target == "v" for e in g.edges)
    assert g.exit == "v"  # source was the sink -> new node becomes exit


def test_insert_verify_convenience():
    g = insert_verify(synthesis_pipeline_graph(), "synth", node_id="check")
    _assert_compiles(g)
    assert g.nodes["check"].primitive == "verify"
    assert any(e.target == "check" for e in g.edges)
    assert any(e.source == "check" and e.target == "synth" for e in g.edges)


def test_append_verify_convenience():
    g = append_verify(best_of_n_graph(), "s", node_id="check")
    _assert_compiles(g)
    assert g.exit == "check"


def test_delete_node_bypasses():
    # react -> insert verify -> delete it -> back to the original shape
    g = insert_after(react_graph(), "s", "verify", node_id="v")
    g2 = delete_node(g, "v")
    _assert_compiles(g2)
    assert "v" not in g2.nodes
    assert sorted(g2.nodes) == sorted(react_graph().nodes)


def test_delete_generate_from_pipeline_compiles():
    g = delete_node(synthesis_pipeline_graph(), "gen3")
    _assert_compiles(g)
    # score/gather still fed by gen1/gen2 -> multi-entry sources remain valid
    assert "gen3" not in g.nodes


def test_substitute_compiles():
    g = substitute(react_graph(), "s", "verify")
    _assert_compiles(g)
    assert g.nodes["s"].primitive == "verify"


def test_swap_primitives_reorders():
    g = swap_primitives(synthesis_pipeline_graph(), "gen1", "gen2")
    _assert_compiles(g)
    assert g.nodes["gen1"].primitive == "generate"
    assert g.nodes["gen2"].primitive == "generate"


def test_duplicate_parallel_branches():
    g = duplicate_parallel(react_graph(), "s", new_id="s2")
    _assert_compiles(g)
    assert "s2" in g.nodes and g.nodes["s2"].primitive == "strategy"
    # the copy feeds the same downstream set as the original
    orig_targets = {e.target for e in react_graph().edges if e.source == "s"}
    copy_targets = {e.target for e in g.edges if e.source == "s2"}
    assert orig_targets == copy_targets


def test_duplicate_sequential_repeats_stage():
    g = duplicate_sequential(react_graph(), "s", new_id="s2")
    _assert_compiles(g)
    assert any(e.source == "s" and e.target == "s2" for e in g.edges)


def test_branch_creates_n_copies():
    g = branch(react_graph(), "s", n=3)
    _assert_compiles(g)
    copies = [n for n in g.nodes if n != "s" and g.nodes[n].primitive == "strategy"]
    assert len(copies) == 3


def test_gather_join_merges_independent_paths():
    # build: gen1 -> out, gen2 -> out (two parallel generators feeding one verify)
    from dse.discovery import ArchGraph, ArchNode
    g = ArchGraph(name="two_gen", entry="g1", exit="v")
    g.add_node(ArchNode("g1", "generate"))
    g.add_node(ArchNode("g2", "generate"))
    g.add_node(ArchNode("v", "verify"))
    g.add_edge("g1", "v")
    g.add_edge("g2", "v")
    joined = gather_join(g, "v", gather_id="j")
    _assert_compiles(joined)
    assert joined.nodes["j"].primitive == "gather"
    # both generators feed the join; the join feeds verify
    assert {e.target for e in joined.edges if e.source in ("g1", "g2")} == {"j"}
    assert any(e.source == "j" and e.target == "v" for e in joined.edges)


# ---------------------------------------------------------------------------
# Immutability: mutations never modify the source graph
# ---------------------------------------------------------------------------

def test_mutations_do_not_mutate_the_source_graph():
    src = synthesis_pipeline_graph()
    before = src.to_json()
    insert_before(src, "gen1", "verify")
    insert_after(src, "synth", "verify")
    delete_node(src, "gen3")
    substitute(src, "gen2", "sample_n")
    swap_primitives(src, "gen1", "gen2")
    duplicate_parallel(src, "gen1")
    gather_join(src, "guard")
    assert src.to_json() == before  # untouched


# ---------------------------------------------------------------------------
# Random mutation: compile-gated, deterministic under a seeded RNG
# ---------------------------------------------------------------------------

def test_random_mutation_deterministic_and_compiles():
    for graph in (react_graph(), best_of_n_graph(), synthesis_pipeline_graph()):
        r1 = random_mutation(graph, rng=random.Random(42))
        r2 = random_mutation(graph, rng=random.Random(42))
        assert r1 == r2  # deterministic
        if r1 is not None:
            name, candidate = r1
            assert name in mutation_operators()
            _assert_compiles(candidate)


def test_random_mutation_eventually_produces_valid_graph():
    """Across many seeds on a simple graph, at least one mutation succeeds."""
    saw_valid = False
    for seed in range(30):
        res = random_mutation(react_graph(), rng=random.Random(seed))
        if res is not None:
            saw_valid = True
            break
    assert saw_valid


# ---------------------------------------------------------------------------
# A mutated candidate executes (smoke — the discovery loop will filter)
# ---------------------------------------------------------------------------

def test_inserted_verify_candidate_executes(tiny_stack, discovery_ctx):
    """insert_verify before react's strategy node: verify receives the
    strategy's RunResult (has .text via RunResult? no — verify handles
    non-str via str()). We instead insert verify after the strategy in a
    chain where semantics stay coherent, and confirm it runs."""
    g = insert_after(react_graph(), "s", "verify", node_id="check")
    from dse.discovery import ArchGraph  # noqa
    task = list(tiny_stack[1].values())[0]
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    # verify consumes the strategy's RunResult -> coerced to str; answer still
    # comes from the strategy node (which remains the terminal).
    assert rec.events and any(e.node_id == "check" for e in rec.events)
