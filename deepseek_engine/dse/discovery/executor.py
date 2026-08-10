"""Architecture executor (Phase 1, spec §37/§17).

Runs a compiled architecture graph against a task, streaming data along
edges and recording full provenance. Supports:
- straight-line DAGs, fan-in (gather), fan-out (branch → merge)
- conditional routing: a primitive that sets ``meta["__route"]`` makes the
  executor follow only the out-edge whose ``port`` matches (spec §7 routing)
- bounded loops are expressed as primitives (e.g. ``strategy:reflexion``
  already loops internally) — graph-level cycles are rejected by the compiler

Every run produces an ``ArchRunRecord`` — the seed of the §17 fitness record
and §29 genealogy: graph snapshot, per-node events, tokens/latency/cost,
final answer, success, and any routing decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .compiler import CompiledArch
from .graph import ArchGraph
from .primitives import ExecutionContext, NodeOutput, primitive


@dataclass
class NodeEvent:
    node_id: str
    primitive: str
    kind: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArchRunRecord:
    """Provenance record for one architecture execution (spec §17/§30)."""

    architecture: str
    graph: dict
    task_id: str
    success: bool
    answer: str
    events: list[NodeEvent] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    routed: dict[str, str] = field(default_factory=dict)   # node_id -> route taken
    error: str | None = None

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "graph": self.graph,
            "task_id": self.task_id,
            "success": self.success,
            "answer": self.answer,
            "events": [{"node": e.node_id, "primitive": e.primitive, "kind": e.kind,
                        "tokens_in": e.tokens_in, "tokens_out": e.tokens_out,
                        "latency_s": round(e.latency_s, 4), "model": e.model,
                        "meta": e.meta} for e in self.events],
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "latency_s": round(self.latency_s, 4), "cost_usd": round(self.cost_usd, 6),
            "routed": self.routed, "error": self.error,
        }


def _resolve_answer(out: NodeOutput | None) -> tuple[str, bool]:
    """Derive (answer_text, success) from the terminal node's output."""
    if out is None:
        return "", False
    value = out.value
    if out.meta.get("passed") is not None:
        return out.text, bool(out.meta["passed"])
    if hasattr(value, "passed") and hasattr(value, "score"):      # Verdict
        return "", bool(value.passed)
    if hasattr(value, "answer"):                                   # RunResult
        return value.answer, bool(value.success)
    if isinstance(value, dict) and "answer" in value:              # scored item
        return value.get("answer") or value.get("text") or "", True
    if isinstance(value, str):
        return value, True
    return "", True


class ArchExecutor:
    """Execute compiled architectures. Stateless — build one per context or
    reuse; the context is built fresh per run for isolation."""

    def __init__(self, context: ExecutionContext | None = None) -> None:
        self._context = context

    def _fresh_context(self, **overrides) -> ExecutionContext:
        base = self._context or ExecutionContext()
        kw = {k: getattr(base, k) for k in ("llm", "verifier", "models", "config",
                                            "agents", "task", "budget",
                                            "cost_per_1m_in", "cost_per_1m_out")}
        kw.update(overrides)
        return ExecutionContext(**kw)

    def run(self, arch: CompiledArch, task=None, context=None) -> ArchRunRecord:
        ctx = context or self._fresh_context()
        if task is not None:
            ctx.task = task
        if ctx.task is None:
            raise ValueError("executor needs a task (set on context or passed to run)")
        ctx.t_start = time.perf_counter()
        ctx.outputs = {}

        g = arch.graph
        record = ArchRunRecord(
            architecture=g.name, graph=g.to_dict(),
            task_id=getattr(ctx.task, "id", "?"), success=False, answer="")

        # active_edges: (source, target, port) edges the routing decisions
        # have enabled. An edge with port=None is always enabled once its
        # source runs (unless the source made a routing decision).
        active_edges: set[tuple[str, str, str | None]] = set()
        executed: set[str] = set()
        last_out: NodeOutput | None = None

        def in_edge_active(src: str, nid: str, port: str | None) -> bool:
            return ((src, nid, port) in active_edges or (src, nid, None) in active_edges)

        for nid in arch.order:
            fan_in = arch.fan_in.get(nid, [])
            if fan_in:
                # has in-edges: run only if at least one is active (not pruned)
                if not any(in_edge_active(src, nid, port) for port, src in fan_in):
                    continue
            # no in-edges -> implicit entry source, always runs

            # gather inputs only from active in-edges
            inputs: dict[str, NodeOutput] = {}
            for port, src in fan_in:
                if in_edge_active(src, nid, port) and src in ctx.outputs:
                    inputs[port] = ctx.outputs[src]

            node = g.nodes[nid]
            t0 = time.perf_counter()
            try:
                out = primitive(node.primitive)(ctx, node, inputs)
            except Exception as exc:  # record and stop (spec: never hide failures)
                record.error = f"{nid} ({node.primitive}): {exc!r}"
                ctx.outputs[nid] = NodeOutput("", kind="error")
                break

            lat = time.perf_counter() - t0
            ctx.outputs[nid] = out
            executed.add(nid)
            last_out = out
            record.events.append(NodeEvent(
                node_id=nid, primitive=node.primitive, kind=out.kind,
                tokens_in=out.tokens_in, tokens_out=out.tokens_out,
                latency_s=lat, model=out.model,
                meta={k: v for k, v in out.meta.items() if k != "__route"}))
            record.tokens_in += out.tokens_in
            record.tokens_out += out.tokens_out
            record.latency_s += out.latency_s

            # conditional routing: follow only the out-edge whose port matches
            route = out.meta.get("__route")
            if route is not None:
                record.routed[nid] = str(route)
            for edge in arch.fan_out.get(nid, []):
                if route is None or edge.key == str(route):
                    active_edges.add((edge.source, edge.target, edge.port))

        # final answer + success from the terminal node
        exit_id = arch.exit if arch.exit in executed else None
        terminal = exit_id or (executed and arch.order[-1]) or None
        final_out = ctx.outputs.get(terminal) if terminal else None
        if final_out is None and executed:
            final_out = ctx.outputs[sorted(executed)[-1]]
        answer, success = _resolve_answer(final_out)
        record.answer = answer
        record.success = success and record.error is None
        record.cost_usd = ctx.estimate_cost(record.tokens_in, record.tokens_out)
        return record
