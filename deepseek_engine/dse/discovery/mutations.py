"""Phase 3 — mutation operators (spec §7).

Candidate generation: every operator takes an ``ArchGraph`` and returns a NEW
graph (architectures are immutable in the library — genealogy records parents,
§29). Operators are structural; the Phase-3 gate is that every mutation still
COMPILES (``compile_graph``). Runtime viability is the discovery loop's job
(Phase 4): an evaluation filters out candidates that fail or regress.

Supported operators (spec §7): insertion, deletion, substitution, reordering
(swap), duplication (parallel + sequential), branching, merging (gather join),
verify-stage insertion. Conditional routing is already expressed by the
``route_disagreement`` primitive + the ``disagreement_pipeline`` baseline;
crossover is Phase 5.
"""

from __future__ import annotations

import random

from .compiler import CompileError, compile_graph
from .graph import ArchGraph, ArchNode


def _copy(graph: ArchGraph) -> ArchGraph:
    return ArchGraph.from_dict(graph.to_dict())


def _fresh_id(graph: ArchGraph, prefix: str = "n") -> str:
    i = 1
    while f"{prefix}_{i}" in graph.nodes:
        i += 1
    return f"{prefix}_{i}"


def _has_edge(graph: ArchGraph, source: str, target: str, port: str | None = None) -> bool:
    return any(e.source == source and e.target == target and e.port == port
               for e in graph.edges)


def _add_edge(graph: ArchGraph, source: str, target: str, port: str | None = None) -> None:
    if not _has_edge(graph, source, target, port):
        graph.add_edge(source, target, port)


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------

def insert_before(graph: ArchGraph, target_id: str, primitive: str,
                  node_id: str | None = None, params: dict | None = None,
                  port: str | None = None) -> ArchGraph:
    """Insert a new node immediately BEFORE ``target_id``.

    In-edges of the target are re-routed into the new node (ports preserved);
    the new node then feeds the target. If the target was an entry source (no
    in-edges), the new node becomes the implicit entry.
    """
    g = _copy(graph)
    if target_id not in g.nodes:
        raise ValueError(f"target {target_id!r} not in graph")
    nid = node_id or _fresh_id(g, "ins")
    g.add_node(ArchNode(nid, primitive, params=dict(params or {})))
    # re-route in-edges: src -> new node (keep the port on the edge)
    in_edges = [e for e in g.edges if e.target == target_id]
    for e in in_edges:
        g.edges.remove(e)
        _add_edge(g, e.source, nid, e.port)
    _add_edge(g, nid, target_id, port)
    if g.entry == target_id:
        g.entry = nid
    return g


def insert_after(graph: ArchGraph, source_id: str, primitive: str,
                 node_id: str | None = None, params: dict | None = None,
                 port: str | None = None) -> ArchGraph:
    """Insert a new node immediately AFTER ``source_id``.

    Out-edges of the source are re-routed out of the new node (ports kept);
    the source then feeds the new node. If the source was a sink/exit, the
    new node becomes the new exit.
    """
    g = _copy(graph)
    if source_id not in g.nodes:
        raise ValueError(f"source {source_id!r} not in graph")
    nid = node_id or _fresh_id(g, "ins")
    g.add_node(ArchNode(nid, primitive, params=dict(params or {})))
    out_edges = [e for e in g.edges if e.source == source_id]
    for e in out_edges:
        g.edges.remove(e)
        _add_edge(g, nid, e.target, e.port)
    _add_edge(g, source_id, nid, port)
    if g.exit == source_id:
        g.exit = nid
    return g


def insert_verify(graph: ArchGraph, target_id: str, node_id: str | None = None) -> ArchGraph:
    """Convenience: insert a ``verify`` stage before ``target_id`` (the spec's
    canonical example: ``A -> B -> C`` becomes ``A -> Verify -> B -> C``)."""
    return insert_before(graph, target_id, "verify", node_id=node_id)


def append_verify(graph: ArchGraph, source_id: str, node_id: str | None = None) -> ArchGraph:
    """Convenience: append a ``verify`` stage after ``source_id`` (the spec's
    ``A -> B -> C`` becomes ``A -> B -> C -> Verify``)."""
    return insert_after(graph, source_id, "verify", node_id=node_id)


# ---------------------------------------------------------------------------
# Deletion (spec: remove a stage, test whether performance holds)
# ---------------------------------------------------------------------------

def delete_node(graph: ArchGraph, node_id: str) -> ArchGraph:
    """Splice ``node_id`` out of the graph (bypass: in-edges -> out-edges).

    Entry/exit nodes that are deleted leave the graph to derive implicit
    sources / terminal from the topology. Multi-entry semantics keep the
    result valid: a target left with no in-edges becomes an implicit source.
    """
    g = _copy(graph)
    if node_id not in g.nodes:
        raise ValueError(f"node {node_id!r} not in graph")
    in_edges = [e for e in g.edges if e.target == node_id]
    out_edges = [e for e in g.edges if e.source == node_id]
    # bypass: every in-edge connects to every out-edge (dedup)
    for ie in in_edges:
        g.edges.remove(ie)
        for oe in out_edges:
            _add_edge(g, ie.source, oe.target, None)
    for oe in out_edges:
        if oe in g.edges:
            g.edges.remove(oe)
    del g.nodes[node_id]
    if g.entry == node_id:
        g.entry = None
    if g.exit == node_id:
        g.exit = None
    return g


# ---------------------------------------------------------------------------
# Substitution & reordering
# ---------------------------------------------------------------------------

def substitute(graph: ArchGraph, node_id: str, primitive: str,
               params: dict | None = None) -> ArchGraph:
    """Replace a node's primitive (spec: ``MCTS -> beam search``)."""
    g = _copy(graph)
    if node_id not in g.nodes:
        raise ValueError(f"node {node_id!r} not in graph")
    g.nodes[node_id].primitive = primitive
    if params is not None:
        g.nodes[node_id].params = dict(params)
    return g


def swap_primitives(graph: ArchGraph, a: str, b: str) -> ArchGraph:
    """Reorder two operations by swapping their primitives+params in place
    (wiring is untouched — the two stages effectively change order)."""
    g = _copy(graph)
    for nid in (a, b):
        if nid not in g.nodes:
            raise ValueError(f"node {nid!r} not in graph")
    pa, pp = g.nodes[a].primitive, g.nodes[a].params
    qa, qp = g.nodes[b].primitive, g.nodes[b].params
    g.nodes[a].primitive, g.nodes[a].params = qa, qp
    g.nodes[b].primitive, g.nodes[b].params = pa, pp
    return g


# ---------------------------------------------------------------------------
# Duplication & branching (spec: repeat a strategically important stage;
# create parallel reasoning paths)
# ---------------------------------------------------------------------------

def duplicate_parallel(graph: ArchGraph, node_id: str, new_id: str | None = None) -> ArchGraph:
    """Add a PARALLEL copy of ``node_id`` that feeds the same downstream
    nodes — a second reasoning path (§7 branching)."""
    g = _copy(graph)
    if node_id not in g.nodes:
        raise ValueError(f"node {node_id!r} not in graph")
    src = g.nodes[node_id]
    nid = new_id or _fresh_id(g, "dup")
    g.add_node(ArchNode(nid, src.primitive, params=dict(src.params)))
    for oe in [e for e in g.edges if e.source == node_id]:
        _add_edge(g, nid, oe.target, oe.port)
    return g


def duplicate_sequential(graph: ArchGraph, node_id: str, new_id: str | None = None) -> ArchGraph:
    """Insert a copy of ``node_id`` immediately AFTER it — the same stage runs
    twice in sequence (spec §7 duplication)."""
    g = _copy(graph)
    if node_id not in g.nodes:
        raise ValueError(f"node {node_id!r} not in graph")
    src = g.nodes[node_id]
    return insert_after(g, node_id, src.primitive, node_id=new_id,
                        params=dict(src.params))


def branch(graph: ArchGraph, node_id: str, n: int = 2) -> ArchGraph:
    """Create ``n`` parallel copies of ``node_id`` (branching, §7)."""
    g = _copy(graph)
    for _ in range(n):
        g = duplicate_parallel(g, node_id)
    return g


# ---------------------------------------------------------------------------
# Merging (spec: combine independent paths via a gather join)
# ---------------------------------------------------------------------------

def gather_join(graph: ArchGraph, target_id: str, gather_id: str | None = None) -> ArchGraph:
    """Insert a ``gather`` node in front of ``target_id`` so its multiple
    in-edges are collected into a list before the target (spec §7 merging).

    Only applied when the target has >1 in-edge (a real join point).
    """
    g = _copy(graph)
    if target_id not in g.nodes:
        raise ValueError(f"target {target_id!r} not in graph")
    in_edges = [e for e in g.edges if e.target == target_id]
    if len(in_edges) < 2:
        return g  # nothing to merge
    gid = gather_id or _fresh_id(g, "join")
    g.add_node(ArchNode(gid, "gather"))
    for e in in_edges:
        g.edges.remove(e)
        _add_edge(g, e.source, gid, e.port)
    _add_edge(g, gid, target_id)
    return g


# ---------------------------------------------------------------------------
# Random mutation (for the discovery loop, Phase 4)
# ---------------------------------------------------------------------------

_OPERATORS = [
    ("insert_verify", lambda g, rng: insert_verify(g, _pick_single_input_target(g, rng))),
    ("append_verify", lambda g, rng: append_verify(g, _pick_node(g, rng))),
    ("delete_node", lambda g, rng: delete_node(g, _pick_node(g, rng))),
    ("substitute", lambda g, rng: substitute(g, _pick_node(g, rng), "verify")),
    ("swap", lambda g, rng: swap_primitives(g, *_pick_two_nodes(g, rng))),
    ("duplicate_parallel", lambda g, rng: duplicate_parallel(g, _pick_node(g, rng))),
    ("duplicate_sequential", lambda g, rng: duplicate_sequential(g, _pick_node(g, rng))),
    ("gather_join", lambda g, rng: gather_join(g, _pick_multi_input_target(g, rng))),
]


def _pick_node(graph: ArchGraph, rng: random.Random) -> str:
    return rng.choice(sorted(graph.nodes))


def _pick_two_nodes(graph: ArchGraph, rng: random.Random) -> tuple[str, str]:
    a, b = rng.sample(sorted(graph.nodes), 2)
    return a, b


def _pick_single_input_target(graph: ArchGraph, rng: random.Random) -> str:
    """A node with exactly one in-edge (safe insert-before target)."""
    cands = [n for n in graph.nodes
             if sum(1 for e in graph.edges if e.target == n) == 1]
    return rng.choice(cands) if cands else _pick_node(graph, rng)


def _pick_multi_input_target(graph: ArchGraph, rng: random.Random) -> str:
    cands = [n for n in graph.nodes
             if sum(1 for e in graph.edges if e.target == n) >= 2]
    return rng.choice(cands) if cands else _pick_node(graph, rng)


def _unique_name(graph: ArchGraph, op: str) -> str:
    """Deterministic unique candidate id: ``<parent>+<op>-<graphdigest>``.

    Unique per structure (so the benchmark/registry never collide) and
    reproducible (same mutation -> same id). The digest encodes the graph
    content, so distinct mutations of the same parent never share an id.
    """
    import hashlib
    digest = hashlib.md5(graph.to_json().encode("utf-8")).hexdigest()[:6]
    return f"{graph.name}+{op}-{digest}"


def random_mutation(graph: ArchGraph, rng: random.Random | None = None,
                    max_attempts: int = 8) -> tuple[str, ArchGraph] | None:
    """Apply one random operator, keeping only mutations that COMPILE (the
    Phase-3 gate). Returns ``(operator_name, new_graph)`` or None if no
    operator produced a valid graph. Deterministic when ``rng`` is seeded.

    The candidate is renamed to a deterministic unique id (the parent is
    never mutated — genealogy records the lineage, §29).
    """
    if rng is None:
        rng = random.Random()
    for _ in range(max_attempts):
        name, op = rng.choice(_OPERATORS)
        try:
            candidate = op(graph, rng)
            compile_graph(candidate)  # gate: must still compile
            candidate.name = _unique_name(candidate, name)
            return name, candidate
        except (CompileError, ValueError, KeyError):
            continue
    return None


def mutation_operators() -> list[str]:
    return [name for name, _ in _OPERATORS]
