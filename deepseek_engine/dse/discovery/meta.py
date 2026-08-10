"""Phase 10 — meta-optimization (spec §24, Level 3).

Level 1 solves tasks. Level 2 discovers architectures. Level 3 discovers
better ways TO DISCOVER — i.e. better DISCOVERY STRATEGIES. This module is
strictly BOUNDED and AUDITABLE (§24): it searches only a declared catalog of
strategy parameterizations (it never rewrites its own core), records every
outcome, and ADOPTS a better strategy only through an explicit gate — the
meta-level analogue of the never-worse promotion gate.

A discovery strategy parameterizes the Phase-4/5/7 loop: which mutation
operators are weighted, how many rounds/candidates, beam width, novelty
weight, crossover, and the improvement margin. ``run_discovery_strategy``
runs the loop and scores the OUTCOME (improvement over the incumbent per unit
of compute). ``meta_search`` compares candidate strategies and, via
``meta_gate``, adopts one only if it is not dominated by the default on
(improvement up, compute down) — Pareto at the meta level, reusing pareto.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compiler import compile_graph
from .evaluate import benchmark_architectures
from .loop import DiscoveryConfig, discover, split_tasks
from .pareto import ParetoPoint, dominates
from .primitives import ExecutionContext
from .registry import ArchitectureRegistry
from .graph import ArchGraph


# ---------------------------------------------------------------------------
# Discovery strategy (§24 — the search-space parameterization)
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryStrategy:
    name: str
    operator_weights: dict | None = None    # bias WHICH mutation operator
    n_rounds: int = 3
    candidates_per_parent: int = 6
    beam_width: int = 3
    novelty_weight: float = 0.0
    crossover_per_round: int = 2
    min_improvement: float = 0.0
    seed: int = 7

    def to_config(self) -> DiscoveryConfig:
        return DiscoveryConfig(
            n_rounds=self.n_rounds, candidates_per_parent=self.candidates_per_parent,
            beam_width=self.beam_width, novelty_weight=self.novelty_weight,
            crossover_per_round=self.crossover_per_round,
            min_improvement=self.min_improvement, seed=self.seed,
            operator_weights=self.operator_weights,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "operator_weights": dict(self.operator_weights or {}),
            "n_rounds": self.n_rounds, "candidates_per_parent": self.candidates_per_parent,
            "beam_width": self.beam_width, "novelty_weight": self.novelty_weight,
            "crossover_per_round": self.crossover_per_round,
            "min_improvement": self.min_improvement, "seed": self.seed,
        }


def default_strategy() -> DiscoveryStrategy:
    """The project's default discovery strategy (the current Phase-4/5/7
    defaults, uniform operator choice)."""
    return DiscoveryStrategy(name="default")


def quality_focused_strategy() -> DiscoveryStrategy:
    """A candidate strategy that biases the search toward the operators that
    expand compute (best_of_n_ify / duplicate) — a plausible better way."""
    return DiscoveryStrategy(
        name="quality_focused",
        operator_weights={"best_of_n_ify": 4.0, "duplicate_sequential": 2.0,
                          "append_verify": 1.5},
    )


def cheap_strategy() -> DiscoveryStrategy:
    """A candidate strategy with fewer candidates per round (less search
    compute — tests whether the same quality can be found more cheaply)."""
    return DiscoveryStrategy(name="cheap", candidates_per_parent=3, n_rounds=2)


def candidate_strategies() -> list[DiscoveryStrategy]:
    """The DECLARED, bounded strategy catalog (§24 — the meta-level never
    searches outside this set)."""
    return [default_strategy(), quality_focused_strategy(), cheap_strategy()]


# ---------------------------------------------------------------------------
# Running & scoring a discovery strategy (the meta-level fitness)
# ---------------------------------------------------------------------------

@dataclass
class MetaOutcome:
    strategy: str
    promoted_count: int
    best_discovered_success: float | None
    incumbent_success: float
    improvement: float                       # best discovered - incumbent (held-out)
    candidates_tested: int
    meta_efficiency: float                   # improvement per 100 candidates
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "promoted_count": self.promoted_count,
            "best_discovered_success": (round(self.best_discovered_success, 4)
                                        if self.best_discovered_success is not None else None),
            "incumbent_success": round(self.incumbent_success, 4),
            "improvement": round(self.improvement, 4),
            "candidates_tested": self.candidates_tested,
            "meta_efficiency": round(self.meta_efficiency, 6),
        }


def run_discovery_strategy(
    strategy: DiscoveryStrategy,
    tasks,
    ctx: ExecutionContext,
    incumbent: str = "react",
    seed_graphs: list[ArchGraph] | None = None,
    registry: ArchitectureRegistry | None = None,
) -> MetaOutcome:
    """Run the discovery loop with this strategy; score the OUTCOME on the
    held-out split (best promoted success vs the incumbent)."""
    reg = registry or ArchitectureRegistry(f"meta_{strategy.name}")
    report = discover(reg, ctx, tasks, seed_graphs=seed_graphs,
                      config=strategy.to_config())
    inc_success = report.incumbent_success

    # measure the best promoted candidate on the held-out split
    _, held = split_tasks(tasks, strategy.to_config().discovery_frac,
                          strategy.to_config().seed)
    best_held = 0.0
    if report.promoted:
        cand_graphs = [reg.get(name).graph for name in report.promoted]
        hres = benchmark_architectures(cand_graphs, held, ctx,
                                       seed=strategy.to_config().seed)
        best_held = max((hres.rows[g.name].success_rate for g in cand_graphs
                         if g.name in hres.rows), default=0.0)
    improvement = best_held - inc_success
    candidates = report.total_candidates
    return MetaOutcome(
        strategy=strategy.name,
        promoted_count=len(report.promoted),
        best_discovered_success=best_held if report.promoted else None,
        incumbent_success=inc_success,
        improvement=improvement,
        candidates_tested=candidates,
        meta_efficiency=improvement / (max(candidates, 1) / 100.0),
        detail={"report": report.to_dict()},
    )


# ---------------------------------------------------------------------------
# The meta-level gate (§24 — bounded, never-worse at the meta level)
# ---------------------------------------------------------------------------

def meta_gate(default: MetaOutcome, candidate: MetaOutcome) -> bool:
    """Adopt a candidate discovery strategy ONLY if it is clearly better, not
    just differently-expensive (§24 bounded):

    - strictly better quality (improvement > default), AND
    - at least as efficient (quality-per-compute >= default).

    A compute-heavy strategy with a small quality gain is a frontier
    TRADE-OFF, not a clearly-better way to discover — the meta-level stays
    conservative (never-worse at the meta level).
    """
    if candidate.improvement <= default.improvement + 1e-9:
        return False
    if candidate.meta_efficiency < default.meta_efficiency - 1e-9:
        return False
    return True


@dataclass
class MetaReport:
    incumbent: str
    outcomes: list[MetaOutcome] = field(default_factory=list)
    winner: str | None = None
    adopted: bool = False

    def to_dict(self) -> dict:
        return {
            "incumbent": self.incumbent,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "winner": self.winner,
            "adopted": self.adopted,
        }


def meta_search(
    tasks,
    ctx: ExecutionContext,
    incumbent: str = "react",
    seed_graphs: list[ArchGraph] | None = None,
    strategies: list[DiscoveryStrategy] | None = None,
    registry_factory=None,
) -> MetaReport:
    """Level-3 meta-optimization: run every declared discovery strategy, pick
    the best by meta-efficiency (tiebreak improvement), and ADOPT it only if
    it passes the meta-gate vs the default. Fully auditable — the report
    records every strategy's outcome and the decision."""
    strategies = strategies or candidate_strategies()
    outcomes = [run_discovery_strategy(s, tasks, ctx, incumbent=incumbent,
                                       seed_graphs=seed_graphs,
                                       registry=(registry_factory(s) if registry_factory else None))
                for s in strategies]
    report = MetaReport(incumbent=incumbent, outcomes=outcomes)
    if not outcomes:
        return report

    best = max(outcomes, key=lambda o: (o.meta_efficiency, o.improvement))
    report.winner = best.strategy
    default = next((o for o in outcomes if o.strategy == "default"), None)
    if default is not None and best.strategy != "default":
        report.adopted = meta_gate(default, best)
    return report
