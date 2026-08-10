"""Executable inference-architecture graphs (Phase 1 of the discovery spec).

Architectures are represented as machine-readable DAGs — NOT natural-language
prompts — so they can be compiled, executed, mutated and compared
programmatically. This is the substrate every later phase (discovery,
evolution, meta-optimization) builds on.

Design notes (adapted from the self-discovering-inference spec, §4/§9):
- nodes = primitives (generate, verify, synthesize, a whole strategy, ...)
- edges = data flow; every edge may name a ``port`` used for conditional
  routing (e.g. the ``high``/``low`` exit of a disagreement detector)
- graphs are JSON-serializable (provenance, §30)
- structural novelty is measured against a reference set (§9): never claim
  "nobody thought of this" — only report distance to known architectures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArchNode:
    """A single node in an architecture graph.

    ``primitive`` names an entry in the primitive registry (see primitives.py).
    ``params`` are passed to the primitive (model tier, sample count, budget,
    thresholds...). ``meta`` is free-form provenance/labels.
    """

    id: str
    primitive: str
    params: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "primitive": self.primitive,
                "params": dict(self.params), "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> "ArchNode":
        return cls(id=d["id"], primitive=d["primitive"],
                   params=dict(d.get("params", {})), meta=dict(d.get("meta", {})))


@dataclass
class ArchEdge:
    """Data-flow edge: ``source`` -> ``target``.

    ``port`` names the input slot on the target (default: the source node id).
    For conditional routing, the executor follows only the out-edge whose
    ``port`` matches the primitive's routing decision.
    """

    source: str
    target: str
    port: str | None = None

    @property
    def key(self) -> str:
        return self.port or self.source

    def to_dict(self) -> dict:
        return {"source": self.source, "target": self.target, "port": self.port}

    @classmethod
    def from_dict(cls, d: dict) -> "ArchEdge":
        return cls(source=d["source"], target=d["target"], port=d.get("port"))


@dataclass
class ArchGraph:
    """A named, executable inference-architecture DAG."""

    name: str
    nodes: dict[str, ArchNode] = field(default_factory=dict)
    edges: list[ArchEdge] = field(default_factory=list)
    entry: str | None = None      # single entry node
    exit: str | None = None       # optional exit node (default: last in topo)
    params: dict[str, Any] = field(default_factory=dict)  # graph-level defaults

    # -- construction ------------------------------------------------------
    def add_node(self, node: ArchNode) -> "ArchGraph":
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return self

    def add_edge(self, source: str, target: str, port: str | None = None) -> "ArchGraph":
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"edge references unknown node ({source}->{target})")
        self.edges.append(ArchEdge(source, target, port))
        return self

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entry": self.entry,
            "exit": self.exit,
            "params": dict(self.params),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchGraph":
        g = cls(name=d["name"], entry=d.get("entry"), exit=d.get("exit"),
                params=dict(d.get("params", {})))
        for nd in d.get("nodes", []):
            g.add_node(ArchNode.from_dict(nd))
        for ed in d.get("edges", []):
            g.add_edge(ArchEdge.from_dict(ed).source, ArchEdge.from_dict(ed).target,
                       ArchEdge.from_dict(ed).port)
        return g

    @classmethod
    def from_json(cls, s: str) -> "ArchGraph":
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ArchGraph({self.name!r}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


# ---------------------------------------------------------------------------
# Structural novelty (§9 of the spec)
# ---------------------------------------------------------------------------

def structural_similarity(a: ArchGraph, b: ArchGraph) -> float:
    """Jaccard similarity over primitive-node ids and edge pairs.

    1.0 = identical structure; 0.0 = nothing shared. Used ONLY to report
    ``novelty_score = 1 - max similarity vs a reference set`` — never to
    claim absolute novelty.
    """
    a_nodes = {n.primitive for n in a.nodes.values()}
    b_nodes = {n.primitive for n in b.nodes.values()}
    a_edges = {(e.source, e.target, e.key) for e in a.edges}
    b_edges = {(e.source, e.target, e.key) for e in b.edges}
    if not a_nodes and not b_nodes:
        return 1.0
    node_sim = (len(a_nodes & b_nodes) / len(a_nodes | b_nodes)) if (a_nodes | b_nodes) else 0.0
    if a_edges or b_edges:
        edge_sim = len(a_edges & b_edges) / len(a_edges | b_edges)
    else:
        edge_sim = 1.0  # both have no edges -> trivially identical structure
    return 0.5 * node_sim + 0.5 * edge_sim


def novelty_against(graph: ArchGraph, reference: list[ArchGraph]) -> dict:
    """Report novelty of ``graph`` vs a reference set (§9).

    Returns ``novelty_score`` (1 - max similarity), the nearest reference
    architecture, and the similarity to it. Classification is the caller's
    job (KNOWN/VARIANT/COMBINATION/STRUCTURALLY NOVEL) — this only measures.
    """
    if not reference:
        return {"novelty_score": 1.0, "nearest": None, "similarity": 0.0}
    best = max(reference, key=lambda r: structural_similarity(graph, r))
    sim = structural_similarity(graph, best)
    return {
        "novelty_score": round(1.0 - sim, 4),
        "nearest": best.name,
        "similarity": round(sim, 4),
        "shared_primitives": sorted({n.primitive for n in graph.nodes.values()}
                                    & {n.primitive for n in best.nodes.values()}),
    }
