"""Phase 5 — evolutionary search (spec §7 crossover, §9 novelty, §16
exploration, §18 retirement, §20 statistics).

Layered on the Phase-4 discovery loop:

- ``crossover`` — combine the HEAD of one successful architecture with the
  TAIL of another (spec §7 crossover: combine sections of two successful
  architectures). Well-defined on our DAG family: A keeps everything upstream
  of a cut node, B keeps everything downstream of a cut node, and the child
  joins them. B's suffix is renamed to avoid collisions; the child is
  deterministic under a seeded RNG.
- ``validate_statistical`` — the Phase-5 gate: an improvement must survive
  the two-sided binomial SIGN test on paired wins AND a bootstrap CI on the
  success-rate difference whose lower bound is above zero (§20).
- ``retire_unfit`` — retire candidates that were rejected with zero success
  over their fitness history (§18 terminal states).
- ``beam_score`` — novelty-aware selection: a small novelty bonus so the
  beam does not greedily collapse onto one structure (§9/§16).
"""

from __future__ import annotations

import hashlib
import random

from ..benchmarks.harness import bootstrap_ci, sign_test
from ..events import RunResult
from .compiler import CompileError, compile_graph
from .graph import ArchGraph, ArchNode
from .registry import ArchitectureRegistry, REJECTED, RETIRED


# ---------------------------------------------------------------------------
# Crossover (spec §7)
# ---------------------------------------------------------------------------

def _ancestors(graph: ArchGraph, node: str) -> set[str]:
    """Every node that can reach ``node`` via in-edges (inclusive)."""
    result = {node}
    frontier = [node]
    in_map: dict[str, list[str]] = {}
    for e in graph.edges:
        in_map.setdefault(e.target, []).append(e.source)
    while frontier:
        cur = frontier.pop()
        for src in in_map.get(cur, []):
            if src not in result:
                result.add(src)
                frontier.append(src)
    return result


def _descendants(graph: ArchGraph, node: str) -> set[str]:
    """Every node reachable from ``node`` via out-edges (inclusive)."""
    result = {node}
    frontier = [node]
    out_map: dict[str, list[str]] = {}
    for e in graph.edges:
        out_map.setdefault(e.source, []).append(e.target)
    while frontier:
        cur = frontier.pop()
        for tgt in out_map.get(cur, []):
            if tgt not in result:
                result.add(tgt)
                frontier.append(tgt)
    return result


def crossover(graph_a: ArchGraph, graph_b: ArchGraph,
              cut_a: str | None = None, cut_b: str | None = None,
              rng: random.Random | None = None) -> ArchGraph:
    """Combine the HEAD of A with the TAIL of B.

    ``cut_a`` splits A (A keeps everything upstream of it); ``cut_b`` splits
    B (B keeps everything downstream of it). The child is A's prefix -> join
    -> B's suffix. B's suffix nodes are renamed to avoid collisions with A's
    prefix. Deterministic under a seeded ``rng``.
    """
    if rng is None:
        rng = random.Random()
    for g, name in ((graph_a, "graph_a"), (graph_b, "graph_b")):
        if not g.nodes:
            raise ValueError(f"{name} has no nodes to crossover")
    ca = cut_a or rng.choice(sorted(graph_a.nodes))
    cb = cut_b or rng.choice(sorted(graph_b.nodes))
    keep_a = _ancestors(graph_a, ca)
    keep_b = _descendants(graph_b, cb)

    child = ArchGraph(name=f"{graph_a.name}x{graph_b.name}",
                      params=dict(graph_a.params))
    # A's prefix (keep names)
    for nid in sorted(keep_a):
        n = graph_a.nodes[nid]
        child.add_node(ArchNode(nid, n.primitive, params=dict(n.params),
                                meta=dict(n.meta)))
    for e in graph_a.edges:
        if e.source in keep_a and e.target in keep_a:
            child.add_edge(e.source, e.target, e.port)
    # B's suffix, renamed to avoid collisions
    rename: dict[str, str] = {}
    counter = 1
    for nid in sorted(keep_b):
        n = graph_b.nodes[nid]
        new = nid
        while new in child.nodes:
            new = f"y{counter}_{nid}"
            counter += 1
        rename[nid] = new
        child.add_node(ArchNode(new, n.primitive, params=dict(n.params),
                                meta=dict(n.meta)))
    for e in graph_b.edges:
        if e.source in keep_b and e.target in keep_b:
            child.add_edge(rename[e.source], rename[e.target], e.port)
    # join: A's cut -> B's cut
    child.add_edge(ca, rename[cb])
    # entry/exit
    if graph_a.entry in keep_a:
        child.entry = graph_a.entry
    if graph_b.exit in keep_b:
        child.exit = rename[graph_b.exit]
    return child


def unique_child_name(a_name: str, b_name: str, graph: ArchGraph) -> str:
    digest = hashlib.md5(graph.to_json().encode("utf-8")).hexdigest()[:6]
    return f"{a_name}x{b_name}-{digest}"


# ---------------------------------------------------------------------------
# Statistical validation (§20) — the Phase-5 significance gate
# ---------------------------------------------------------------------------

def validate_statistical(cand_oks, inc_oks, task_ids, seed: int = 0,
                         alpha: float = 0.05, ci_iters: int = 1000) -> dict:
    """Does the candidate's edge over the incumbent survive statistics?

    - two-sided binomial SIGN test on paired wins (ties excluded)
    - bootstrap 95% CI on the success-rate difference; requires the lower
      bound > 0 AND sign-test p < alpha for ``significant``.
    """
    favor = sum(1 for c, i in zip(cand_oks, inc_oks) if c and not i)
    against = sum(1 for c, i in zip(cand_oks, inc_oks) if (not c) and i)
    p = sign_test(favor, favor + against)

    def _runs(name, oks):
        return [RunResult(task_id=tid, strategy=name, success=bool(ok), answer="")
                for tid, ok in zip(task_ids, oks)]

    lo, hi, point = bootstrap_ci(_runs("cand", cand_oks), _runs("inc", inc_oks),
                                 iters=ci_iters, seed=seed)
    significant = bool(p < alpha and lo > 0)
    return {
        "sign_test_p": round(p, 6),
        "favor": favor, "against": against,
        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
        "diff": round(point, 4),
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# Retirement (§18) & novelty-aware selection (§9/§16)
# ---------------------------------------------------------------------------

def retire_unfit(registry: ArchitectureRegistry, min_records: int = 1,
                 require_zero_success: bool = True) -> list[str]:
    """Retire candidates in the REJECTED terminal state that never succeeded
    in their fitness history. Returns the retired ids."""
    retired: list[str] = []
    for rec in registry.all():
        if rec.state != REJECTED:
            continue
        if len(rec.fitness) < min_records:
            continue
        successes = [bool(f.get("success")) for f in rec.fitness]
        if require_zero_success and any(successes):
            continue
        registry.set_state(rec.arch_id, RETIRED)
        retired.append(rec.arch_id)
    return retired


def beam_score(success_rate: float, novelty: float, novelty_weight: float = 0.05) -> float:
    """Novelty-aware selection: base fitness + a small novelty bonus so the
    beam does not greedily collapse onto one structure (§16 exploration vs
    exploitation — a bounded epsilon in novelty space)."""
    return success_rate + novelty_weight * novelty


def crossover_children(a: ArchGraph, b: ArchGraph, n: int,
                       rng: random.Random | None = None) -> list[tuple[str, ArchGraph]]:
    """Generate ``n`` compile-valid crossover children of ``a`` and ``b``
    with unique names. Invalid children are skipped (the compiler gate)."""
    rng = rng or random.Random()
    out: list[tuple[str, ArchGraph]] = []
    for _ in range(n):
        child = crossover(a, b, rng=rng)
        try:
            compile_graph(child)
        except CompileError:
            continue
        child.name = unique_child_name(a.name, b.name, child)
        out.append(("crossover", child))
    return out
