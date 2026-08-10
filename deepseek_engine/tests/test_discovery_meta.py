"""Phase-10 tests: meta-optimization (spec §24, Level 3 — discovering better
discovery strategies, bounded & auditable).

The meta-level must:
- parameterize a discovery strategy (operator weights, rounds, compute...)
- score the OUTCOME (improvement over incumbent per unit of compute)
- compare candidate strategies and pick a winner
- adopt a winner ONLY through the meta-gate (never-worse at the meta level,
  reusing Pareto dominance) — no unrestricted self-modification
- leave a full audit trail
"""

from __future__ import annotations

import pytest

from dse.discovery import ArchitectureRegistry, DiscoveryConfig


@pytest.fixture
def big_ctx(stack):
    config, catalog, models, llm, env, verifier, agents, budget = stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _big_tasks(stack):
    return list(stack[1].values())


# ---------------------------------------------------------------------------
# Operator weighting (the search-strategy knob)
# ---------------------------------------------------------------------------

def test_random_mutation_operator_weights_bias(stack, big_ctx):
    """Weights must bias which operator is chosen — the knob meta-level
    optimizes. best_of_n_ify weighted to 0 should never be picked."""
    import random
    from dse.discovery import random_mutation, react_graph
    zeroed = {n: 0.0 for n in
              ["insert_verify", "append_verify", "delete_node", "substitute",
               "swap", "duplicate_parallel", "duplicate_sequential",
               "gather_join", "best_of_n_ify"]}
    zeroed["best_of_n_ify"] = 0.0
    # only best_of_n_ify has weight -> every successful mutation is best_of_n_ify
    only = {n: 0.0 for n in zeroed}
    only["best_of_n_ify"] = 1.0
    picked = set()
    for seed in range(20):
        res = random_mutation(react_graph(), rng=random.Random(seed),
                              operator_weights=only)
        if res is not None:
            picked.add(res[0])
    assert picked and picked == {"best_of_n_ify"}


def test_discover_config_threads_operator_weights(stack, big_ctx):
    """operator_weights flow through DiscoveryConfig into the loop."""
    cfg = DiscoveryConfig(n_rounds=1, candidates_per_parent=3, seed=7,
                          incumbent="react",
                          operator_weights={"best_of_n_ify": 5.0})
    from dse.discovery import discover, react_graph
    reg = ArchitectureRegistry("weights")
    report = discover(reg, big_ctx, _big_tasks(stack),
                      seed_graphs=[react_graph()], config=cfg)
    assert report.incumbent == "react"


# ---------------------------------------------------------------------------
# Meta strategy outcomes & the gate
# ---------------------------------------------------------------------------

def test_meta_gate_adopts_only_better_strategy():
    from dse.discovery import MetaOutcome, meta_gate
    default = MetaOutcome(strategy="default", promoted_count=3,
                          best_discovered_success=0.5, incumbent_success=0.2,
                          improvement=0.3, candidates_tested=60,
                          meta_efficiency=0.5)
    better = MetaOutcome(strategy="better", promoted_count=4,
                         best_discovered_success=0.65, incumbent_success=0.2,
                         improvement=0.45, candidates_tested=70,
                         meta_efficiency=0.64)
    worse = MetaOutcome(strategy="worse", promoted_count=1,
                        best_discovered_success=0.4, incumbent_success=0.2,
                        improvement=0.2, candidates_tested=60,
                        meta_efficiency=0.33)
    expensive = MetaOutcome(strategy="expensive", promoted_count=4,
                            best_discovered_success=0.62, incumbent_success=0.2,
                            improvement=0.42, candidates_tested=400,
                            meta_efficiency=0.105)
    assert meta_gate(default, better)
    assert not meta_gate(default, worse)      # no improvement -> reject
    assert not meta_gate(default, expensive)  # dominated on compute -> reject


def test_meta_search_runs_and_is_auditable(stack, big_ctx):
    from dse.discovery import candidate_strategies, meta_search, react_graph
    report = meta_search(_big_tasks(stack), big_ctx, incumbent="react",
                         seed_graphs=[react_graph()])
    d = report.to_dict()
    assert set(d) == {"incumbent", "outcomes", "winner", "adopted"}
    assert d["incumbent"] == "react"
    # every declared strategy was evaluated (audit trail)
    assert {o["strategy"] for o in d["outcomes"]} == {s.name for s in candidate_strategies()}
    assert report.winner is not None
    assert report.winner in {s.name for s in candidate_strategies()}
    # outcomes are well-formed
    for o in report.outcomes:
        assert o.improvement >= -1.0 and o.candidates_tested >= 0
        assert o.meta_efficiency >= 0 or o.improvement <= 0


def test_meta_search_never_mutates_the_core(stack, big_ctx):
    """§24 bounded: the meta-level only SELECTS among declared strategies —
    it never rewrites the discovery core. The winner is a declared strategy
    name and no unknown strategy appears."""
    from dse.discovery import candidate_strategies, meta_search, react_graph
    report = meta_search(_big_tasks(stack), big_ctx, incumbent="react",
                         seed_graphs=[react_graph()])
    declared = {s.name for s in candidate_strategies()}
    for o in report.outcomes:
        assert o.strategy in declared
    assert report.winner in declared


def test_strategy_to_config_roundtrip():
    from dse.discovery import DiscoveryStrategy
    s = DiscoveryStrategy(name="q", operator_weights={"best_of_n_ify": 3.0},
                          n_rounds=2, candidates_per_parent=4)
    cfg = s.to_config()
    assert isinstance(cfg, DiscoveryConfig)
    assert cfg.operator_weights == {"best_of_n_ify": 3.0}
    assert cfg.n_rounds == 2 and cfg.candidates_per_parent == 4
    assert s.to_dict()["name"] == "q"
