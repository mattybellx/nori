"""Architecture compiler & validator (Phase 1, spec §32).

Rejects invalid graphs BEFORE execution:
- unknown primitives or node references
- cycles (topological ordering must succeed)
- unreachable nodes / unreachable exit
- graph-level budget violations (too many nodes, too deep)

Only compiled graphs execute. A compiled graph carries a deterministic
topological order plus per-node fan-in/fan-out so the executor can stream
data without re-deriving the graph structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import ArchEdge, ArchGraph
from .primitives import registered_names


class CompileError(ValueError):
    """Graph failed validation; ``problems`` lists every issue found."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems) if problems else "invalid architecture graph")


@dataclass
class CompiledArch:
    graph: ArchGraph
    order: list[str] = field(default_factory=list)        # topo order of node ids
    fan_in: dict[str, list[tuple[str, str]]] = field(default_factory=dict)   # node -> [(port, source)]
    fan_out: dict[str, list[ArchEdge]] = field(default_factory=dict)         # node -> [edges]

    @property
    def entry(self) -> str:
        if self.graph.entry:
            return self.graph.entry
        # implicit entry: first node with no incoming edges (multi-entry DAG)
        has_in = {e.target for e in self.graph.edges}
        sources = sorted(n for n in self.graph.nodes if n not in has_in)
        return sources[0] if sources else self.order[0]

    @property
    def exit(self) -> str:
        return self.graph.exit or self.order[-1]


def _reachable(graph: ArchGraph, starts: list[str]) -> set[str]:
    """All nodes reachable from the given start nodes (static edges)."""
    seen = set(starts)
    frontier = list(starts)
    out: dict[str, list[str]] = {}
    for e in graph.edges:
        out.setdefault(e.source, []).append(e.target)
    while frontier:
        cur = frontier.pop()
        for nxt in out.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def _sources(graph: ArchGraph) -> list[str]:
    """Implicit entry nodes: any node with no incoming edges (§4 — parallel
    generators like best-of-N naturally have multiple starts)."""
    has_in = {e.target for e in graph.edges}
    return sorted(n for n in graph.nodes if n not in has_in)


def validate(graph: ArchGraph) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems: list[str] = []
    known = set(registered_names())

    if not graph.nodes:
        problems.append("graph has no nodes")
        return problems

    entry = graph.entry
    exit_ = graph.exit
    if entry is not None and entry not in graph.nodes:
        problems.append(f"entry node {entry!r} not in graph")
    if exit_ is not None and exit_ not in graph.nodes:
        problems.append(f"exit node {exit_!r} not in graph")

    for nid, node in graph.nodes.items():
        if node.primitive not in known:
            problems.append(f"node {nid!r}: unknown primitive {node.primitive!r}")
        if not nid or not isinstance(nid, str):
            problems.append(f"node id must be a non-empty string, got {nid!r}")

    for e in graph.edges:
        if e.source not in graph.nodes:
            problems.append(f"edge source {e.source!r} not in graph")
        if e.target not in graph.nodes:
            problems.append(f"edge target {e.target!r} not in graph")
        if e.source == e.target:
            problems.append(f"self-loop on {e.source!r} (cycles are not allowed)")

    # cycle detection via Kahn's algorithm (deterministic order) — only when
    # every edge endpoint is a known node (otherwise already reported above)
    endpoints_known = all(e.source in graph.nodes and e.target in graph.nodes
                          for e in graph.edges)
    indeg = {nid: 0 for nid in graph.nodes}
    out_edges: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if endpoints_known:
            indeg[e.target] += 1
            out_edges[e.source].append(e.target)
    queue = sorted([n for n in graph.nodes if indeg[n] == 0])
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(out_edges[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(graph.nodes):
        cyclic = sorted(n for n in graph.nodes if indeg[n] > 0)
        problems.append(f"graph contains a cycle involving: {cyclic}")

    # reachability from the implicit entry sources (multi-entry DAGs)
    sources = _sources(graph)
    reach = _reachable(graph, sources) if sources else set()
    unreachable = sorted(set(graph.nodes) - reach)
    if unreachable:
        problems.append(f"nodes unreachable from any entry source: {unreachable}")
    if exit_ is not None and exit_ not in reach:
        problems.append(f"exit node {exit_!r} unreachable")

    # graph-level budget (spec §33): max_nodes / max_depth
    max_nodes = graph.params.get("max_nodes")
    if max_nodes is not None and len(graph.nodes) > int(max_nodes):
        problems.append(f"graph has {len(graph.nodes)} nodes > max_nodes {max_nodes}")
    max_depth = graph.params.get("max_depth")
    if max_depth is not None:
        depth = {n: 0 for n in graph.nodes}
        for n in order:
            for e in graph.edges:
                if e.source == n:
                    depth[e.target] = max(depth[e.target], depth[n] + 1)
        if any(d >= int(max_depth) for d in depth.values()):
            problems.append(f"graph depth exceeds max_depth {max_depth}")
    return problems


def compile_graph(graph: ArchGraph) -> CompiledArch:
    """Validate and topologically order ``graph``; raise CompileError if bad."""
    problems = validate(graph)
    if problems:
        raise CompileError(problems)

    indeg = {nid: 0 for nid in graph.nodes}
    out_edges: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        indeg[e.target] += 1
        out_edges[e.source].append(e.target)
    queue = sorted([n for n in graph.nodes if indeg[n] == 0])
    order: list[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(out_edges[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    fan_in: dict[str, list[tuple[str, str]]] = {}
    fan_out: dict[str, list[ArchEdge]] = {}
    for e in graph.edges:
        fan_in.setdefault(e.target, []).append((e.key, e.source))
        fan_out.setdefault(e.source, []).append(e)
    return CompiledArch(graph=graph, order=order, fan_in=fan_in, fan_out=fan_out)
