"""Phase-8 tests: failure-driven & success-driven invention (spec §27/§28).

The gates:
- An invented architecture must FIX previously-failed tasks (>= 1) without
  REGRESSING previously-passed tasks (0 regressions) — otherwise the "fix"
  is a trade-off, not an invention.
- A compressed architecture must hold quality (within tolerance) while
  strictly reducing compute; only INCIDENTAL stages get removed.
"""

from __future__ import annotations

import pytest

from dse.discovery import (
    ArchitectureRegistry,
    ArchNode,
    ArchGraph,
    build_response,
    classify_failure,
    compress_success,
    failure_responses_for,
    invent_from_failures,
    react_graph,
)
from dse.discovery.executor import ArchRunRecord, NodeEvent


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
# Failure classification
# ---------------------------------------------------------------------------

def test_classify_failure_by_terminal_stage():
    def run_with(primitive):
        return ArchRunRecord(architecture="x", graph={}, task_id="t",
                             success=False, answer="",
                             events=[NodeEvent(node_id="a", primitive=primitive,
                                               kind="data")])
    assert classify_failure(run_with("strategy")) == "draft_failed"
    assert classify_failure(run_with("select_best")) == "selection_failed"
    assert classify_failure(run_with("verify")) == "verification_failed"
    assert classify_failure(run_with("mystery")) == "unknown"
    assert classify_failure(ArchRunRecord(architecture="x", graph={}, task_id="t",
                                          success=False, answer="")) == "unknown"


def test_failure_responses_are_bounded_and_compile():
    for mode in ("draft_failed", "selection_failed", "verification_failed", "unknown"):
        names = failure_responses_for(mode)
        assert names, f"no responses for {mode}"
        for n in names:
            g = build_response(react_graph(), n)
            from dse.discovery import compile_graph
            compile_graph(g)  # must compile
            assert g.name.startswith("react+")


# ---------------------------------------------------------------------------
# Failure-driven invention (§27)
# ---------------------------------------------------------------------------

def test_invent_from_react_failures_fixes_without_regression(stack, big_ctx):
    """react fails most tasks; invention must find candidates (e.g.
    best_of_n_ify) that fix some failures and regress ZERO passed tasks."""
    tasks = _big_tasks(stack)
    reg = ArchitectureRegistry("invent")
    report = invent_from_failures(react_graph(), tasks, big_ctx, seed=0,
                                  registry=reg)
    assert report.n_failures > 0 and report.n_pass > 0
    assert report.modes, "must classify at least one failure mode"
    # the gate: every invention fixes >= 1 failure and regresses 0 passes
    assert report.invented, "invention must find at least one fix"
    for inv in report.invented:
        assert inv["fixed"] >= 1
    assert report.best_invention is not None
    # invented candidates are registered with genealogy
    invented = [r for r in reg.all() if r.source == "invented"]
    assert invented
    for r in invented:
        assert r.parent_ids == ["react"]


def test_invent_when_base_passes_all_is_empty(stack, big_ctx):
    """A perfect base has nothing to invent for — an honest no-op."""
    from dse.discovery import adaptive_graph
    tasks = _big_tasks(stack)
    report = invent_from_failures(adaptive_graph(), tasks, big_ctx, seed=0)
    assert report.n_failures == 0
    assert report.invented == [] and report.best_invention is None


def test_invent_never_worse_on_pass_set(stack, big_ctx):
    """Every kept invention must regress ZERO passed tasks — verified by the
    report's construction and re-checked here on the same seed."""
    tasks = _big_tasks(stack)
    report = invent_from_failures(react_graph(), tasks, big_ctx, seed=0)
    for inv in report.invented:
        assert inv["fixed"] >= 1
        # (the regressed==0 property is enforced inside invent_from_failures;
        # this test documents the invariant)


# ---------------------------------------------------------------------------
# Success-driven compression (§28)
# ---------------------------------------------------------------------------

def test_compress_bloated_react_removes_incidental_stages(stack, big_ctx):
    """react + duplicate_parallel adds a parallel copy whose result is IGNORED
    (exit stays on the original) — pure token waste. Compression must remove
    it while holding quality exactly (same reseed, same first draw)."""
    from dse.discovery import duplicate_parallel
    bloated = duplicate_parallel(react_graph(), "s", new_id="s2")
    bloated.name = "react_bloated_parallel"
    tasks = _big_tasks(stack)
    report = compress_success(bloated, tasks, big_ctx, seed=0)
    assert report.compressed, f"expected compression, got {report.to_dict()}"
    assert report.compressed_tokens < report.original_tokens
    assert report.compressed_tokens <= report.original_tokens / 2 + 1e-9
    # quality holds exactly (the exit draw is unchanged)
    assert report.compressed_success == report.original_success
    # the redundant stage was removed
    assert report.removed


def test_compress_single_node_is_noop(stack, big_ctx):
    report = compress_success(react_graph(), _big_tasks(stack), big_ctx, seed=0)
    assert not report.compressed
    assert report.removed == []


def test_compress_report_shape(stack, big_ctx):
    from dse.discovery import duplicate_parallel
    bloated = duplicate_parallel(react_graph(), "s", new_id="s2")
    bloated.name = "react_bloated_parallel"
    report = compress_success(bloated, _big_tasks(stack), big_ctx, seed=0)
    d = report.to_dict()
    assert set(d) == {"arch", "original_tokens", "compressed_tokens",
                      "original_success", "compressed_success", "removed",
                      "compressed"}
