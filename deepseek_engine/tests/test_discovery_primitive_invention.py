"""Phase-9 tests: novel primitive discovery (spec §38/§39).

Gates: a proposed primitive must pass STATIC, UNIT, ADVERSARIAL, REFERENCE
and SANDBOX checks before it is promoted into the live registry. Generated
code never silently replaces trusted infrastructure — promotion is explicit.
"""

from __future__ import annotations

import pytest

from dse.discovery import (
    ArchNode,
    NodeOutput,
    propose_majority_vote,
    promote_primitive,
    registered_names,
    validate_primitive,
)
from dse.discovery.primitive_invention import (
    PrimitiveProposal,
    _majority_vote_impl,
)


@pytest.fixture
def big_ctx(stack):
    config, catalog, models, llm, env, verifier, agents, budget = stack
    by_name = {a.name: a for a in agents}
    from dse.discovery.primitives import ExecutionContext
    return ExecutionContext(llm=llm, verifier=verifier, models=models,
                            config=config, agents=by_name, budget=budget)


def _task(stack):
    return list(stack[1].values())[0]


# ---------------------------------------------------------------------------
# majority_vote behaviour (unit level)
# ---------------------------------------------------------------------------

def test_majority_vote_returns_majority():
    node = ArchNode("n", "majority_vote")
    ctx = type("C", (), {})()
    inputs = {"a": NodeOutput("X"), "b": NodeOutput("Y"), "c": NodeOutput("X")}
    out = _majority_vote_impl(ctx, node, inputs)
    assert out.value == "X"
    assert out.meta == {"n": 3, "votes": 2}


def test_majority_vote_empty_and_tie():
    node = ArchNode("n", "majority_vote")
    ctx = type("C", (), {})()
    assert _majority_vote_impl(ctx, node, {}).value == ""
    out = _majority_vote_impl(ctx, node,
                              {"a": NodeOutput("B"), "b": NodeOutput("A")})
    assert out.value == "A"  # deterministic tie-break


# ---------------------------------------------------------------------------
# The validation pipeline (§39)
# ---------------------------------------------------------------------------

def test_majority_vote_proposal_passes_all_gates(stack, big_ctx):
    proposal = propose_majority_vote()
    assert proposal.name == "majority_vote"
    assert proposal.name not in registered_names()  # genuinely new
    v = validate_primitive(proposal, big_ctx, _task(stack))
    assert v.passes, v.errors
    assert v.gates["static"] and v.gates["unit"]
    assert v.gates["adversarial"] and v.gates["reference"]
    assert v.gates["sandbox"]


def test_validation_report_shape(stack, big_ctx):
    v = validate_primitive(propose_majority_vote(), big_ctx, _task(stack))
    d = v.to_dict()
    assert set(d) == {"name", "passes", "gates", "errors", "cost"}
    assert d["name"] == "majority_vote"
    # order-independent: the sandbox cost is only present when the sandbox ran
    if v.gates.get("sandbox"):
        assert d["cost"]["tokens_total"] >= 0


def test_promote_primitive_registers_only_after_gates(stack, big_ctx):
    proposal = propose_majority_vote()
    if proposal.name in registered_names():
        pytest.skip("already registered by an earlier test")
    ok, v = promote_primitive(proposal, big_ctx, _task(stack))
    assert ok and v.passes
    assert "majority_vote" in registered_names()
    # now usable in a graph
    from dse.discovery import ArchGraph, compile_graph
    g = ArchGraph(name="vote_graph", entry="a", exit="m")
    g.add_node(ArchNode("a", "generate"))
    g.add_node(ArchNode("m", "majority_vote"))
    g.add_edge("a", "m")
    compile_graph(g)  # the new primitive is in the search space


def test_proposal_fails_gates_without_tests(big_ctx, stack):
    bad = PrimitiveProposal(
        name="no_tests_prim",
        purpose="missing tests",
        input_contract={}, output_contract="x",
        implementation=_majority_vote_impl,
    )
    v = validate_primitive(bad, big_ctx, _task(stack))
    assert not v.passes
    assert "unit" in v.errors          # no contract tests -> reject
    assert v.gates["static"]            # static still checked first


def test_proposal_fails_static_on_bad_signature(big_ctx, stack):
    def wrong(a, b):  # two args, not a primitive
        return a
    bad = PrimitiveProposal(
        name="bad_sig", purpose="x", input_contract={}, output_contract="x",
        implementation=wrong, tests=[lambda i: None],
    )
    v = validate_primitive(bad, big_ctx, _task(stack))
    assert not v.passes
    assert "static" in v.errors
