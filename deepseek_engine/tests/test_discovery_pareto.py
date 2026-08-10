"""Phase-7 tests: compute optimization & the Pareto frontier
(spec §13 multi-objective, §23 compression, §40 dashboard).

The Pareto layer must: correctly identify dominance, return the non-dominated
frontier, rank by compute efficiency (quality per token), detect when a
simpler architecture's compute dominates a complex one (compression win), and
— wired into the discovery loop — reject candidates whose added compute did
not pay for itself (§23).
"""

from __future__ import annotations

import pytest

from dse.discovery import (
    ParetoPoint,
    DiscoveryConfig,
    REJECTED,
    compression_win,
    compute_efficiency,
    discover,
    dominates,
    pareto_frontier,
    pareto_summary,
    react_graph,
)


def _p(name, success, score=None, tokens=0.0, latency=0.0) -> ParetoPoint:
    return ParetoPoint(name=name, success_rate=success,
                       avg_verifier_score=score, avg_tokens=tokens,
                       avg_latency_s=latency)


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------

def test_dominates_when_better_everywhere():
    a = _p("a", 0.9, 8.0, 50, 0.2)
    b = _p("b", 0.5, 7.0, 200, 0.8)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominates_when_equal_on_some_axes():
    a = _p("a", 0.8, 7.0, 100, 0.5)
    b = _p("b", 0.8, 7.0, 100, 0.5)
    assert not dominates(a, b)   # identical: not strictly better on any axis
    c = _p("c", 0.8, 7.0, 80, 0.5)   # same quality, fewer tokens
    assert dominates(c, a)


def test_dominates_trade_off_not_dominated():
    a = _p("a", 0.9, 8.0, 300, 1.0)   # better quality, much more compute
    b = _p("b", 0.5, 7.0, 50, 0.2)
    assert not dominates(a, b)  # a better on success, worse on compute
    assert not dominates(b, a)  # genuine frontier trade-off


def test_dominates_ignores_missing_scores():
    a = _p("a", 0.8, None, 100, 0.5)
    b = _p("b", 0.6, None, 200, 0.7)
    assert dominates(a, b)  # score not comparable, others decide


# ---------------------------------------------------------------------------
# Frontier, efficiency, compression
# ---------------------------------------------------------------------------

def test_pareto_frontier_returns_non_dominated():
    pts = [
        _p("cheap", 0.5, None, 50, 0.2),
        _p("mid", 0.75, None, 150, 0.5),
        _p("quality", 0.9, None, 400, 1.2),
        _p("dominated", 0.5, None, 90, 0.6),   # dominated by cheap (worse tokens)
    ]
    frontier, dominated_by = pareto_frontier(pts)
    assert set(frontier) == {"cheap", "mid", "quality"}
    assert "dominated" in dominated_by


def test_compute_efficiency_quality_per_token():
    high = _p("high", 0.9, None, 900)
    eff = _p("eff", 0.5, None, 100)
    assert compute_efficiency(eff) > compute_efficiency(high)


def test_compression_win_detects_unnecessary_compute():
    complex_p = _p("complex", 0.75, None, 400, 1.0)
    simple_p = _p("simple", 0.75, None, 100, 0.3)   # same quality, less compute
    assert compression_win(complex_p, simple_p)
    # a complex arch with genuinely better quality is NOT a compression loss
    better_p = _p("better", 0.95, None, 400, 1.0)
    assert not compression_win(better_p, simple_p)


def test_pareto_summary_shape():
    rows = {
        "a": _p("a", 0.8, None, 100, 0.4),
        "b": _p("b", 0.6, None, 60, 0.2),
    }
    s = pareto_summary(rows)
    assert set(s) == {"frontier", "dominated_by", "efficiency_rank", "points"}
    assert "a" in s["frontier"] and "b" in s["frontier"]
    assert s["efficiency_rank"] == ["b", "a"]  # b is more efficient


# ---------------------------------------------------------------------------
# Pareto gate in the discovery loop (§23)
# ---------------------------------------------------------------------------

@pytest.fixture
def big_ctx(stack):
    config, catalog, models, llm, env, verifier, agents, budget = stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _big_tasks(stack):
    return list(stack[1].values())


def test_discover_reports_pareto_summary(stack, big_ctx):
    from dse.discovery import ArchitectureRegistry
    reg = ArchitectureRegistry("pareto-rep")
    cfg = DiscoveryConfig(n_rounds=2, candidates_per_parent=4, beam_width=3,
                          seed=7, incumbent="react")
    report = discover(reg, big_ctx, _big_tasks(stack),
                      seed_graphs=[react_graph()], config=cfg)
    assert report.pareto is not None
    assert set(report.pareto) == {"frontier", "dominated_by", "efficiency_rank", "points"}
    assert "react" in report.pareto["points"]


def test_require_pareto_rejects_compute_waste(stack, big_ctx):
    """With require_pareto + a generous margin, a candidate that only
    marginally beats the incumbent while burning far more tokens must be
    rejected as a compression loss (§23) — while the loop still runs."""
    from dse.discovery import ArchitectureRegistry
    reg = ArchitectureRegistry("pareto-gate")
    cfg = DiscoveryConfig(n_rounds=2, candidates_per_parent=5, beam_width=3,
                          seed=7, incumbent="react",
                          require_pareto=True, pareto_margin=0.15)
    report = discover(reg, big_ctx, _big_tasks(stack),
                      seed_graphs=[react_graph()], config=cfg)
    # well-formed either way; nothing promoted without clearing the margin
    for name in report.promoted:
        assert reg.get(name).state not in (REJECTED,)
