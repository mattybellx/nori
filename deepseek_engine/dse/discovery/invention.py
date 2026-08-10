"""Phase 8 — failure-driven & success-driven invention (spec §27/§28).

§27 FAILURE-DRIVEN: when an architecture fails, classify the failure mode,
generate ARCHITECTURAL responses (not just retries), and test them on the
exact failure set. An invention is kept only if it fixes failures WITHOUT
regressing the tasks the baseline already passed (never-worse on the pass
set) — otherwise the "fix" is a trade-off, not an invention.

§28 SUCCESS-DRIVEN COMPRESSION: for a successful architecture, ablate stages
one at a time; if removing a stage keeps quality while reducing compute, the
stage was INCIDENTAL — prefer the compressed architecture (the spec's
"minimum computation required for the desired reliability").
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .compiler import CompileError, compile_graph
from .evaluate import benchmark_architectures
from .executor import ArchRunRecord
from .graph import ArchGraph
from .mutations import (
    append_verify,
    best_of_n_ify,
    delete_node,
    duplicate_sequential,
    gather_join,
    insert_verify,
)
from .primitives import ExecutionContext
from .registry import ArchitectureRegistry


# ---------------------------------------------------------------------------
# Failure classification (§27)
# ---------------------------------------------------------------------------

def classify_failure(run: ArchRunRecord) -> str:
    """Classify a failed run by its TERMINAL stage — where the pipeline broke.

    - draft_failed: the generator/strategy produced a wrong answer
    - selection_failed: objective selection picked a wrong candidate
    - verification_failed: a verify/guard stage rejected the outcome
    - unknown: anything else (or no events)
    """
    if not run.events:
        return "unknown"
    last = run.events[-1].primitive
    if last in ("strategy", "generate", "sample_n"):
        return "draft_failed"
    if last in ("select_best", "verify_items", "score_items"):
        return "selection_failed"
    if last in ("verify", "synthesis_guard", "extract", "identity"):
        return "verification_failed"
    return "unknown"


# ---------------------------------------------------------------------------
# Failure -> architectural responses (§27 table)
# ---------------------------------------------------------------------------

# Each response is a (name, builder) pair; the builder maps a base graph to a
# NEW candidate graph. These are structural, compile-gated responses to a
# specific failure mode — not generic retries.
FAILURE_RESPONSES: dict[str, list[tuple[str, callable]]] = {
    "draft_failed": [
        ("best_of_n_ify", lambda g: best_of_n_ify(g, _source_node(g), n=3)),
        ("duplicate_sequential", lambda g: duplicate_sequential(g, _source_node(g))),
        ("append_verify", lambda g: append_verify(g, _any_node(g))),
    ],
    "selection_failed": [
        ("insert_verify", lambda g: insert_verify(g, _any_node(g))),
        ("append_verify", lambda g: append_verify(g, _any_node(g))),
        ("gather_join", lambda g: gather_join(g, _any_multi_target(g))),
    ],
    "verification_failed": [
        ("best_of_n_ify", lambda g: best_of_n_ify(g, _source_node(g), n=3)),
        ("insert_verify", lambda g: insert_verify(g, _any_node(g))),
    ],
    "unknown": [
        ("best_of_n_ify", lambda g: best_of_n_ify(g, _source_node(g), n=3)),
        ("append_verify", lambda g: append_verify(g, _any_node(g))),
    ],
}


def _source_node(g: ArchGraph) -> str:
    has_in = {e.target for e in g.edges}
    sources = sorted(n for n in g.nodes if n not in has_in)
    return sources[0] if sources else sorted(g.nodes)[0]


def _any_node(g: ArchGraph) -> str:
    return sorted(g.nodes)[0]


def _any_multi_target(g: ArchGraph) -> str:
    cands = [n for n in g.nodes if sum(1 for e in g.edges if e.target == n) >= 2]
    return cands[0] if cands else _any_node(g)


def build_response(base: ArchGraph, response_name: str) -> ArchGraph:
    """Build the candidate graph for a named response, renamed uniquely."""
    builder = None
    for responses in FAILURE_RESPONSES.values():
        for name, fn in responses:
            if name == response_name:
                builder = fn
    if builder is None:
        raise ValueError(f"unknown failure response {response_name!r}")
    cand = builder(base)
    digest = hashlib.md5(cand.to_json().encode("utf-8")).hexdigest()[:6]
    cand.name = f"{base.name}+{response_name}-{digest}"
    return cand


def failure_responses_for(mode: str) -> list[str]:
    return [name for name, _ in FAILURE_RESPONSES.get(mode, FAILURE_RESPONSES["unknown"])]


# ---------------------------------------------------------------------------
# Failure-driven invention (§27)
# ---------------------------------------------------------------------------

@dataclass
class InventionReport:
    base: str
    n_tasks: int
    n_failures: int
    n_pass: int
    modes: list[str] = field(default_factory=list)
    candidates_tested: int = 0
    invented: list[dict] = field(default_factory=list)   # {response, name, fixed, failures}
    best_invention: str | None = None

    def to_dict(self) -> dict:
        return {
            "base": self.base, "n_tasks": self.n_tasks,
            "n_failures": self.n_failures, "n_pass": self.n_pass,
            "modes": self.modes, "candidates_tested": self.candidates_tested,
            "invented": list(self.invented), "best_invention": self.best_invention,
        }


def invent_from_failures(
    base_graph: ArchGraph,
    tasks,
    ctx: ExecutionContext,
    seed: int = 0,
    max_candidates: int = 6,
    registry: ArchitectureRegistry | None = None,
) -> InventionReport:
    """Classify the base's failures, build response candidates, and keep any
    that FIX failures without REGRESSING passed tasks (§27).

    The gate is strict: a candidate must succeed on >= 1 previously-failed
    task (fix) AND fail on 0 previously-passed tasks (never-worse on pass).
    """
    base_res = benchmark_architectures([base_graph], tasks, ctx, seed=seed)
    base_row = base_res.rows[base_graph.name]
    fail_tasks = [t for t, r in zip(tasks, base_row.runs) if not r.success]
    pass_tasks = [t for t, r in zip(tasks, base_row.runs) if r.success]

    report = InventionReport(base=base_graph.name, n_tasks=len(tasks),
                             n_failures=len(fail_tasks), n_pass=len(pass_tasks))
    if not fail_tasks:
        return report  # nothing to invent for — the base already passes all

    # classify the failure modes from the failed runs
    modes = {classify_failure(r) for t, r in zip(tasks, base_row.runs) if not r.success}
    report.modes = sorted(modes)

    # build response candidates (bounded, compile-gated)
    candidates: list[tuple[str, ArchGraph]] = []
    seen: set[str] = set()
    for mode in sorted(modes) or ["unknown"]:
        for resp in failure_responses_for(mode):
            try:
                cand = build_response(base_graph, resp)
                compile_graph(cand)
            except (CompileError, ValueError):
                continue
            if cand.name in seen:
                continue
            seen.add(cand.name)
            candidates.append((resp, cand))
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    report.candidates_tested = len(candidates)
    if not candidates:
        return report

    # measure on the FAIL set and the PASS set separately
    cand_graphs = [c for _, c in candidates]
    fail_res = benchmark_architectures(cand_graphs, fail_tasks, ctx, seed=seed)
    pass_res = (benchmark_architectures(cand_graphs, pass_tasks, ctx, seed=seed)
                if pass_tasks else None)

    # keep candidates that fix failures without regressing passes
    invented: list[dict] = []
    for resp, cand in candidates:
        fr = fail_res.rows.get(cand.name)
        if fr is None:
            continue
        fixed = sum(1 for r in fr.runs if r.success)
        regressed = 0
        if pass_res is not None:
            pr = pass_res.rows.get(cand.name)
            regressed = sum(1 for r in pr.runs if not r.success) if pr else 0
        if fixed > 0 and regressed == 0:
            invented.append({
                "response": resp, "name": cand.name,
                "fixed": fixed, "failures": len(fail_tasks),
                "tokens": round(fr.avg_tokens, 1),
            })
            if registry is not None:
                if cand.name not in registry:
                    registry.register(cand, source="invented",
                                      parent_ids=[base_graph.name])
                    registry.record_fitness(cand.name, _first_run(fail_res, cand.name))

    report.invented = invented
    if invented:
        report.best_invention = max(invented, key=lambda d: (d["fixed"], -d["tokens"]))["name"]
    return report


def _first_run(res, name):
    row = res.rows.get(name)
    return row.runs[0] if row and row.runs else None


# ---------------------------------------------------------------------------
# Success-driven compression (§28)
# ---------------------------------------------------------------------------

@dataclass
class CompressionReport:
    arch: str
    original_tokens: float
    compressed_tokens: float
    original_success: float
    compressed_success: float
    removed: list[str] = field(default_factory=list)
    compressed_graph: ArchGraph | None = None

    @property
    def compressed(self) -> bool:
        return bool(self.removed)

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "original_tokens": round(self.original_tokens, 1),
            "compressed_tokens": round(self.compressed_tokens, 1),
            "original_success": round(self.original_success, 4),
            "compressed_success": round(self.compressed_success, 4),
            "removed": list(self.removed),
            "compressed": self.compressed,
        }


def compress_success(
    arch: ArchGraph,
    tasks,
    ctx: ExecutionContext,
    seed: int = 0,
    tolerance: float = 0.0,
    node_order: list[str] | None = None,
) -> CompressionReport:
    """§28: greedily ablate stages; keep a deletion only if success holds
    (within ``tolerance`` of the ORIGINAL) and compute drops below the
    original. Returns the most-compressed graph that still matches quality."""
    from .graph import ArchGraph as _G
    base_res = benchmark_architectures([arch], tasks, ctx, seed=seed)
    base_row = base_res.rows[arch.name]
    original_rate = base_row.success_rate
    original_tokens = base_row.avg_tokens

    current = _G.from_dict(arch.to_dict())
    removed: list[str] = []
    for nid in node_order or sorted(arch.nodes):
        if nid not in current.nodes:
            continue
        if len(current.nodes) <= 1:
            break
        try:
            cand = delete_node(current, nid)
            compile_graph(cand)
        except CompileError:
            continue
        cand_res = benchmark_architectures([cand], tasks, ctx, seed=seed)
        cr = cand_res.rows[cand.name]
        if (cr.success_rate >= original_rate - tolerance - 1e-9
                and cr.avg_tokens < original_tokens - 1e-9):
            current = cand
            removed.append(nid)

    final_res = benchmark_architectures([current], tasks, ctx, seed=seed)
    fr = final_res.rows[current.name]
    return CompressionReport(
        arch=arch.name,
        original_tokens=original_tokens,
        compressed_tokens=fr.avg_tokens,
        original_success=original_rate,
        compressed_success=fr.success_rate,
        removed=removed,
        compressed_graph=current,
    )
