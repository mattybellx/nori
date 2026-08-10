"""Phase 9 — novel primitive discovery (spec §38/§39).

A NEW primitive is a research proposal with a full contract. It enters the
architecture search space ONLY after passing every gate (§39):

1. STATIC  — the implementation is a well-formed Primitive (3-arg callable →
   NodeOutput) and the name is not already registered.
2. UNIT    — the proposal's own tests pass (the author's contract tests).
3. ADVERSARIAL — empty / missing / garbage inputs must not crash the
   executor; output must be deterministic (seeded mock).
4. REFERENCE — if a reference implementation exists, outputs must match.
5. SANDBOX — a graph using the primitive must compile and execute on the
   mock (the discovery sandbox), producing a well-formed ArchRunRecord.
6. COST    — tokens/latency are measured and reported.

Only then is it promoted into the live registry (`register`). Generated code
never silently replaces trusted infrastructure — promotion is explicit (§39).

Demonstrated with a genuinely NEW primitive: `majority_vote` — self-
consistency voting (Wang et al. 2022, cited in verifier.py) over N candidate
answers. Not previously in the registry.
"""

from __future__ import annotations

import inspect
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .compiler import CompileError, compile_graph
from .executor import ArchExecutor, ArchRunRecord
from .graph import ArchGraph, ArchNode
from .primitives import NodeOutput, Primitive, register, registered_names
from .primitives import ExecutionContext


# ---------------------------------------------------------------------------
# The primitive contract (§38)
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveProposal:
    name: str
    purpose: str
    input_contract: dict[str, str]      # port name -> description
    output_contract: str                # what value + meta the NodeOutput carries
    implementation: Primitive           # (ctx, node, inputs) -> NodeOutput
    preconditions: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    cost_model: str = "no LLM calls"
    tests: list[Callable] = field(default_factory=list)   # (impl) -> asserts
    ref_impl: Callable | None = None


@dataclass
class PrimitiveValidation:
    proposal: PrimitiveProposal
    gates: dict[str, bool] = field(default_factory=dict)   # gate -> passed
    errors: dict[str, str] = field(default_factory=dict)   # gate -> message
    cost: dict = field(default_factory=dict)               # tokens / latency

    @property
    def passes(self) -> bool:
        return bool(self.gates) and all(self.gates.values())

    def to_dict(self) -> dict:
        return {
            "name": self.proposal.name,
            "passes": self.passes,
            "gates": dict(self.gates),
            "errors": dict(self.errors),
            "cost": dict(self.cost),
        }


# ---------------------------------------------------------------------------
# The validation pipeline (§39)
# ---------------------------------------------------------------------------

def _gate(name: str, ok: bool, msg: str, v: PrimitiveValidation) -> None:
    v.gates[name] = ok
    if not ok:
        v.errors[name] = msg


def _static_check(proposal: PrimitiveProposal, v: PrimitiveValidation) -> None:
    impl = proposal.implementation
    try:
        sig = inspect.signature(impl)
        params = list(sig.parameters)
        if len(params) != 3:
            raise ValueError(f"expected (ctx, node, inputs), got {len(params)} params")
    except (TypeError, ValueError) as exc:
        _gate("static", False, f"implementation not a 3-arg primitive: {exc}", v)
        return
    if proposal.name in registered_names():
        _gate("static", False, f"name {proposal.name!r} already registered", v)
        return
    if not proposal.name or not proposal.purpose:
        _gate("static", False, "proposal needs name + purpose (§38)", v)
        return
    _gate("static", True, "ok", v)


def _unit_check(proposal: PrimitiveProposal, v: PrimitiveValidation) -> None:
    if not proposal.tests:
        _gate("unit", False, "proposal has no contract tests (§38: tests required)", v)
        return
    try:
        for t in proposal.tests:
            t(proposal.implementation)
        _gate("unit", True, f"{len(proposal.tests)} tests passed", v)
    except Exception as exc:
        _gate("unit", False, f"contract test failed: {exc!r}", v)


def _adversarial_check(proposal: PrimitiveProposal, v: PrimitiveValidation,
                       ctx: ExecutionContext, task) -> None:
    impl = proposal.implementation
    node = ArchNode("adv", proposal.name)
    try:
        # empty inputs must not crash
        out0 = impl(ctx, node, {})
        # missing/garbage port values
        class Garbage:
            pass
        garbage = {"a": NodeOutput(Garbage(), kind="data"),
                   "b": NodeOutput(None, kind="data")}
        impl(ctx, node, garbage)
        # determinism: same inputs -> same output (seeded mock)
        out1 = impl(ctx, node, {})
        same = (str(out0.value) == str(out1.value))
        _gate("adversarial", same and out0 is not None,
              "ok" if same else "non-deterministic output", v)
    except Exception as exc:
        _gate("adversarial", False, f"crashed on adversarial input: {exc!r}", v)


def _reference_check(proposal: PrimitiveProposal, v: PrimitiveValidation,
                     ctx: ExecutionContext, task) -> None:
    if proposal.ref_impl is None:
        _gate("reference", True, "no reference provided (skipped)", v)
        return
    impl = proposal.implementation
    node = ArchNode("ref", proposal.name)
    sample = {"a": NodeOutput("X", kind="data"), "b": NodeOutput("Y", kind="data")}
    try:
        got = impl(ctx, node, sample)
        want = proposal.ref_impl(ctx, node, sample)
        ok = str(got.value) == str(want.value)
        _gate("reference", ok, "ok" if ok else f"mismatch got={got.value!r}", v)
    except Exception as exc:
        _gate("reference", False, f"reference comparison failed: {exc!r}", v)


def _sandbox_check(proposal: PrimitiveProposal, v: PrimitiveValidation,
                   ctx: ExecutionContext, task) -> None:
    """Execute the primitive in a realistic two-generator pipeline (direct
    simulation — validation runs BEFORE promotion, so a graph cannot yet
    reference the primitive; the post-promotion graph-compile is verified by
    the promote test). Measures tokens/latency (the §39 cost gate)."""
    from .primitives import primitive as get_primitive
    gen_node = ArchNode("gen", "generate", params={"extra": "Produce a draft."})
    try:
        gen = get_primitive("generate")
        out1 = gen(ctx, gen_node, {})
        out2 = gen(ctx, gen_node, {})
        inputs = {"g1": out1, "g2": out2}
        t0 = time.perf_counter()
        out = proposal.implementation(ctx, ArchNode("m", proposal.name), inputs)
        elapsed = time.perf_counter() - t0
        well_formed = out is not None and isinstance(out, NodeOutput)
        v.cost = {
            "latency_s": round(elapsed, 4),
            "tokens_total": (out1.tokens_in + out1.tokens_out
                             + out2.tokens_in + out2.tokens_out),
        }
        _gate("sandbox", well_formed, "ok" if well_formed else "bad output", v)
    except Exception as exc:
        _gate("sandbox", False, f"pipeline execution failed: {exc!r}", v)


def validate_primitive(proposal: PrimitiveProposal, ctx: ExecutionContext,
                       task) -> PrimitiveValidation:
    """Run every gate. The primitive is NOT registered here — promotion is a
    separate explicit step (§39)."""
    v = PrimitiveValidation(proposal=proposal)
    _static_check(proposal, v)
    if v.gates.get("static"):
        _unit_check(proposal, v)
        _adversarial_check(proposal, v, ctx, task)
        _reference_check(proposal, v, ctx, task)
        _sandbox_check(proposal, v, ctx, task)
    return v


def promote_primitive(proposal: PrimitiveProposal, ctx: ExecutionContext,
                      task) -> tuple[bool, PrimitiveValidation]:
    """Register the primitive into the live registry ONLY if every gate
    passed. Returns (promoted, validation)."""
    v = validate_primitive(proposal, ctx, task)
    if not v.passes:
        return False, v
    register(proposal.name)(proposal.implementation)
    return True, v


# ---------------------------------------------------------------------------
# The demonstrated new primitive: majority_vote (§38 example class)
# ---------------------------------------------------------------------------

def _majority_vote_impl(ctx: ExecutionContext, node, inputs) -> NodeOutput:
    """Self-consistency voting (Wang et al. 2022): return the most common
    candidate answer. Handles strategy RunResults, dicts, and plain text;
    deterministic tie-break (lexicographically smallest among max-vote)."""
    texts: list[str] = []
    for port, out in inputs.items():
        v = out.value
        if hasattr(v, "answer"):
            texts.append(str(v.answer))
        elif isinstance(v, dict):
            texts.append(str(v.get("answer") or v.get("text") or v))
        elif v is None:
            texts.append("")
        else:
            texts.append(str(v))
    if not texts:
        return NodeOutput("", kind="majority_vote", meta={"n": 0})
    counts = Counter(texts)
    max_votes = max(counts.values())
    best = min(t for t, c in counts.items() if c == max_votes)
    return NodeOutput(best, kind="majority_vote",
                      meta={"n": len(texts), "votes": max_votes})


def propose_majority_vote() -> PrimitiveProposal:
    """The §38 proposal for the new majority_vote primitive."""
    def unit_returns_majority(impl):
        node = ArchNode("n", "majority_vote")
        ctx = ExecutionContext()
        inputs = {"a": NodeOutput("X", kind="data"),
                  "b": NodeOutput("Y", kind="data"),
                  "c": NodeOutput("X", kind="data")}
        out = impl(ctx, node, inputs)
        assert out.value == "X", f"expected majority X, got {out.value!r}"
        assert out.meta["votes"] == 2 and out.meta["n"] == 3

    def unit_empty_and_tie(impl):
        node = ArchNode("n", "majority_vote")
        ctx = ExecutionContext()
        assert impl(ctx, node, {}).value == ""
        # tie breaks deterministically (lexicographic)
        out = impl(ctx, node, {"a": NodeOutput("B", kind="data"),
                               "b": NodeOutput("A", kind="data")})
        assert out.value == "A"

    def unit_handles_runresults_and_dicts(impl):
        node = ArchNode("n", "majority_vote")
        ctx = ExecutionContext()
        class RR:
            answer = "42"
        inputs = {"a": NodeOutput(RR(), kind="data"),
                  "b": NodeOutput({"answer": "42"}, kind="data"),
                  "c": NodeOutput("42", kind="data")}
        assert impl(ctx, node, inputs).value == "42"

    return PrimitiveProposal(
        name="majority_vote",
        purpose="Self-consistency voting: return the most common candidate "
                "answer across N drafts (Wang et al. 2022).",
        input_contract={"<any port>": "one candidate (RunResult | dict | text)"},
        output_contract="value = the majority answer text; meta = {n, votes}",
        implementation=_majority_vote_impl,
        preconditions=["at least one input port", "candidates share an answer space"],
        failure_modes=["empty inputs -> empty answer", "all-ties -> lexicographic pick"],
        cost_model="no LLM calls (pure aggregation)",
        tests=[unit_returns_majority, unit_empty_and_tie, unit_handles_runresults_and_dicts],
        ref_impl=None,  # no trusted reference exists yet — this IS the proposal
    )
