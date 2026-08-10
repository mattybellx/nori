"""Phase 2 — baseline validation & head-to-head evaluation (spec §44/§45).

Before ANY discovery claim is possible, two things must be true:

1. **Equivalence (the Phase-2 gate):** a baseline graph that wraps an
   existing strategy must reproduce that strategy EXACTLY — same answer,
   same success, same token count — when both run with the identical
   per-(strategy, task) reseed. If the executor changes behavior, every
   later "discovered improvement" would be an artifact of the executor, not
   a real architectural gain.

2. **Head-to-head benchmarking:** the evaluation loop that Phase 4 (discovery)
   will use — run a set of architectures over a task catalog with paired
   reseeding and aggregate quality/reliability/tokens/latency into a
   leaderboard row (spec §40). This reuses the project's paired-comparison
   invariant: every (architecture, task) faces identical stochastic draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compiler import CompiledArch, compile_graph
from .executor import ArchExecutor, ArchRunRecord
from .graph import ArchGraph
from .primitives import ExecutionContext


@dataclass
class EquivalenceMismatch:
    task_id: str
    strategy: str
    field: str            # answer | success | tokens_in | tokens_out | attempts
    graph_value: object = None
    strategy_value: object = None


@dataclass
class EquivalenceReport:
    strategy: str
    n_tasks: int
    mismatches: list[EquivalenceMismatch] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        if self.passes:
            return f"{self.strategy}: graph == strategy on {self.n_tasks} tasks (no mismatches)"
        return (f"{self.strategy}: {len(self.mismatches)} mismatches "
                f"({sorted({m.field for m in self.mismatches})})")


def validate_equivalence(
    graph: ArchGraph,
    strategy_name: str,
    ctx: ExecutionContext,
    tasks,
    seed: int = 0,
) -> EquivalenceReport:
    """Run ``graph`` (which must wrap ``strategy_name``) and the strategy
    itself with identical reseeds; report every divergence.

    The graph's ``strategy`` node delegates to ``Agent.solve``, so with the
    same RNG state the two MUST be byte-identical. Any mismatch is a bug in
    the executor/context, not in the strategy.
    """
    assert graph.name == strategy_name, "graph name must match the wrapped strategy"
    arch = compile_graph(graph)
    ex = ArchExecutor()
    report = EquivalenceReport(strategy=strategy_name, n_tasks=len(tasks))
    agent = ctx.agents[strategy_name]
    for task in tasks:
        key = f"{seed}:{strategy_name}:{task.id}"
        # standalone strategy
        ctx.llm.reset(key)
        rr = agent.solve(task, ctx.budget)
        # graph
        ctx.llm.reset(key)
        rec = ex.run(arch, task, context=ctx)

        if rec.answer != rr.answer:
            report.mismatches.append(EquivalenceMismatch(
                task.id, strategy_name, "answer", rec.answer, rr.answer))
        if rec.success != rr.success:
            report.mismatches.append(EquivalenceMismatch(
                task.id, strategy_name, "success", rec.success, rr.success))
        if rec.tokens_in != rr.tokens_in:
            report.mismatches.append(EquivalenceMismatch(
                task.id, strategy_name, "tokens_in", rec.tokens_in, rr.tokens_in))
        if rec.tokens_out != rr.tokens_out:
            report.mismatches.append(EquivalenceMismatch(
                task.id, strategy_name, "tokens_out", rec.tokens_out, rr.tokens_out))
    return report


def validate_all_baselines(ctx: ExecutionContext, tasks,
                           seed: int = 0) -> list[EquivalenceReport]:
    """Validate every strategy baseline graph against the strategy it wraps."""
    from .baselines import strategy_baselines

    reports = []
    for g in strategy_baselines():
        name = g.nodes["s"].params["name"]
        if name not in ctx.agents:
            continue
        reports.append(validate_equivalence(g, name, ctx, tasks, seed=seed))
    return reports


# ---------------------------------------------------------------------------
# Head-to-head benchmark (the evaluation loop discovery will use)
# ---------------------------------------------------------------------------


@dataclass
class ArchBenchRow:
    arch: str
    n: int
    success_rate: float
    avg_verifier_score: float | None
    avg_tokens: float
    avg_latency_s: float
    total_tokens: int
    total_latency_s: float
    runs: list[ArchRunRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arch": self.arch, "n": self.n,
            "success_rate": round(self.success_rate, 4),
            "avg_verifier_score": (round(self.avg_verifier_score, 4)
                                   if self.avg_verifier_score is not None else None),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_latency_s": round(self.avg_latency_s, 3),
            "total_tokens": self.total_tokens,
            "total_latency_s": round(self.total_latency_s, 3),
        }


@dataclass
class BenchmarkResult:
    """Rows for every runnable architecture + the ones that were skipped.

    Skipped means the graph needs a resource this context doesn't have (e.g.
    a ``strategy`` node whose Agent isn't in the context) — the caller must
    see this rather than mistaking an error row for a real result.
    """

    rows: dict[str, ArchBenchRow] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # arch name -> reason


def _strategy_nodes_needed(graph: ArchGraph) -> list[str]:
    return [n.params.get("name") for n in graph.nodes.values()
            if n.primitive == "strategy" and n.params.get("name")]


def benchmark_architectures(
    graphs: list[ArchGraph],
    tasks,
    ctx: ExecutionContext,
    seed: int = 0,
    reset_llm: bool = True,
) -> BenchmarkResult:
    """Run every architecture over every task with paired reseeding and
    aggregate into leaderboard rows (§40).

    Architectures that need a strategy/Agent not present in ``ctx.agents`` are
    SKIPPED (reported in ``result.skipped``) — never run as fake failures.

    ``reset_llm`` reseeds the mock before each (architecture, task) pair so
    comparisons are causal (the project's paired-comparison invariant). For
    real LLMs this is a no-op (their RNG is not mock-controlled).
    """
    ex = ArchExecutor()
    result = BenchmarkResult()
    for graph in graphs:
        missing = [s for s in _strategy_nodes_needed(graph) if s not in ctx.agents]
        if missing:
            result.skipped[graph.name] = f"missing strategies: {sorted(set(missing))}"
            continue
        arch = compile_graph(graph)
        runs: list[ArchRunRecord] = []
        for task in tasks:
            if reset_llm and hasattr(ctx.llm, "reset"):
                ctx.llm.reset(f"{seed}:{graph.name}:{task.id}")
            runs.append(ex.run(arch, task, context=ctx))
        n = len(runs)
        succ = sum(1 for r in runs if r.success)
        result.rows[graph.name] = ArchBenchRow(
            arch=graph.name, n=n,
            success_rate=succ / n if n else 0.0,
            avg_verifier_score=_avg_score(runs),
            avg_tokens=sum(r.tokens_total for r in runs) / n if n else 0.0,
            avg_latency_s=sum(r.latency_s for r in runs) / n if n else 0.0,
            total_tokens=sum(r.tokens_total for r in runs),
            total_latency_s=sum(r.latency_s for r in runs),
            runs=runs,
        )
    return result


def _avg_score(runs: list[ArchRunRecord]) -> float | None:
    """Mean verifier score across runs (from the terminal node's verdict).

    Returns None when NO run recorded a score (e.g. text-terminal pipelines
    whose exit node is a synthesis/identity — absence of a score is not a
    zero score).
    """
    scores = []
    for r in runs:
        terminal = r.events[-1].meta if r.events else {}
        score = terminal.get("score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    return (sum(scores) / len(scores)) if scores else None
