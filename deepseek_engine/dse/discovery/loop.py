"""Phase 4 — the discovery loop v0 (spec §15–§21).

The loop turns the mutation/evaluation machinery into an actual discovery
process:

    1. split tasks into a DISCOVERY set and a HELD-OUT set (deterministic)
    2. benchmark the seed baselines on the discovery set -> INCUMBENT
    3. each round: mutate the population -> benchmark candidates on the
       discovery set -> register with genealogy + fitness -> apply the
       promotion gate (never-worse + discovery improvement + HELD-OUT
       generalization) -> keep the best beam as the next population
    4. report promoted candidates, or honestly report NO PROMOTION (§41)

Honest scope (spec §47): on the checkable mock suites, "independent
verification" means the held-out task split (§20/§21 — never optimize the
final test set). Text-terminal architectures (no objective score) cannot be
promoted on checkable tasks — they are for free-form evaluation in later
phases. If the incumbent is already saturated (100%), the loop says
NO PROMOTION — that is a valid outcome, not a failure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..events import RunResult
from ..benchmarks.harness import mcnemar
from .baselines import strategy_baselines
from .compiler import compile_graph
from .evaluate import ArchBenchRow, BenchmarkResult, benchmark_architectures
from .graph import ArchGraph, structural_similarity
from .mutations import random_mutation
from .evolution import (
    beam_score,
    crossover_children,
    retire_unfit,
    validate_statistical,
)
from .pareto import pareto_summary, point_from_row
from .primitives import ExecutionContext
from .registry import (
    ArchitectureRegistry,
    BENCHMARKED,
    INDEPENDENTLY_VERIFIED,
    PROMOTED,
    REJECTED,
)


@dataclass
class DiscoveryConfig:
    n_rounds: int = 3
    candidates_per_parent: int = 6
    beam_width: int = 4
    discovery_frac: float = 0.75          # rest is held out (§20)
    seed: int = 0
    incumbent: str = "best"               # "best" or an architecture name
    token_budget_mult: float = 3.0        # candidate tokens <= mult x incumbent
    min_improvement: float = 0.0          # success-rate delta required
    mutation_attempts: int = 6            # random_mutation max_attempts
    # Phase 5 (evolution)
    crossover_per_round: int = 2          # crossover children per round (§7)
    require_significance: bool = False    # §20 gate: sign test + bootstrap CI
    alpha: float = 0.05
    novelty_weight: float = 0.0           # >0 adds a novelty bonus to beam (§16)
    retire_after: bool = True             # retire rejected zero-success archs (§18)
    # Phase 7 (compute optimization / Pareto, §13/§23)
    require_pareto: bool = False          # reject candidates whose compute didn't pay
    pareto_margin: float = 0.05           # quality-equivalence tolerance (success-rate)
    # Phase 10 (meta-optimization, §24): biases WHICH mutation operator is
    # chosen — the search-strategy knob the meta-level optimizes.
    operator_weights: dict | None = None  # name -> weight (None = uniform)


@dataclass
class RoundRecord:
    round: int
    candidates_tested: int
    beam: list[str] = field(default_factory=list)
    best_candidate: str | None = None
    best_success: float = 0.0

    def to_dict(self) -> dict:
        return {"round": self.round, "candidates_tested": self.candidates_tested,
                "beam": list(self.beam), "best_candidate": self.best_candidate,
                "best_success": round(self.best_success, 4)}


@dataclass
class DiscoveryReport:
    incumbent: str
    incumbent_success: float
    discovery_n: int
    held_out_n: int
    rounds: list[RoundRecord] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    total_candidates: int = 0
    pareto: dict | None = None            # §13/§40 frontier summary of last round

    @property
    def no_promotion(self) -> bool:
        return not self.promoted

    def to_dict(self) -> dict:
        return {
            "incumbent": self.incumbent,
            "incumbent_success": round(self.incumbent_success, 4),
            "discovery_n": self.discovery_n, "held_out_n": self.held_out_n,
            "rounds": [r.to_dict() for r in self.rounds],
            "promoted": list(self.promoted),
            "rejected": list(self.rejected),
            "total_candidates": self.total_candidates,
            "no_promotion": self.no_promotion,
            "pareto": self.pareto,
        }


def split_tasks(tasks, discovery_frac: float = 0.75, seed: int = 0):
    """Deterministic train/holdout split of a task list (§20)."""
    ordered = sorted(tasks, key=lambda t: getattr(t, "id", str(t)))
    rng = random.Random(f"{seed}:split")
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    k = max(1, int(round(len(ordered) * discovery_frac)))
    discovery = [ordered[i] for i in idx[:k]]
    held_out = [ordered[i] for i in idx[k:]]
    return discovery, held_out


def _rr(name: str, task_ids, oks) -> list[RunResult]:
    return [RunResult(task_id=tid, strategy=name, success=bool(ok), answer="")
            for tid, ok in zip(task_ids, oks)]


def promotion_gate(cand_oks, inc_oks, held_cand_oks, held_inc_oks,
                   task_ids, min_improvement: float = 0.0) -> dict:
    """The §19/§20 promotion gate.

    Requires ALL of:
    - discovery improvement: candidate success-rate > incumbent + delta
    - no_regression: candidate discovery success-rate >= incumbent (the
      honest "no unacceptable regression" — §19)
    - held-out improvement: candidate success-rate > incumbent on the
      held-out split (generalizes — §21 anti-overfitting)

    ``never_worse`` (strict per-task b01 == 0) is reported but NOT required:
    with per-architecture reseeding each architecture faces its own RNG
    stream, so strict per-task never-worse is not estimable in expectation —
    rate-level no-regression is the honest measure. McNemar significance is
    reported as information (v0 does not hard-require it).
    """
    def rate(oks):
        return (sum(1 for o in oks if o) / len(oks)) if oks else 0.0

    # strictly greater than the incumbent by MORE than the delta + epsilon
    # (a bare ``- 1e-9`` here flipped `>` into allowing EQUALITY and promoted
    # equal-rate candidates — caught by the Phase-5 statistical gate)
    disc_improve = rate(cand_oks) > rate(inc_oks) + min_improvement + 1e-9
    no_regression = rate(cand_oks) >= rate(inc_oks) - 1e-9
    held_improve = rate(held_cand_oks) > rate(held_inc_oks) + min_improvement + 1e-9
    b01 = sum(1 for c, i in zip(cand_oks, inc_oks) if (not c) and i)
    never_worse = b01 == 0
    m = mcnemar(_rr("cand", task_ids, cand_oks), _rr("inc", task_ids, inc_oks))
    return {
        "passed": bool(disc_improve and no_regression and held_improve),
        "discovery_improvement": bool(disc_improve),
        "no_regression": bool(no_regression),
        "held_out_improvement": bool(held_improve),
        "never_worse": bool(never_worse),       # strict per-task (informational)
        "b01": b01,
        "mcnemar_p": round(m.p_value, 6),
        "cand_discovery": round(rate(cand_oks), 4),
        "inc_discovery": round(rate(inc_oks), 4),
        "cand_held_out": round(rate(held_cand_oks), 4),
        "inc_held_out": round(rate(held_inc_oks), 4),
    }


def _objective_rows(res: BenchmarkResult) -> dict[str, ArchBenchRow]:
    """Only rows with an objective verifier score are promotion-eligible on
    checkable tasks (text-terminal pipelines are excluded honestly)."""
    return {k: v for k, v in res.rows.items() if v.avg_verifier_score is not None}


def _pick_incumbent(rows: dict[str, ArchBenchRow], incumbent: str) -> str:
    if incumbent in rows:
        return incumbent
    return max(rows, key=lambda k: (rows[k].success_rate, rows[k].avg_verifier_score or 0.0))


def discover(
    registry: ArchitectureRegistry,
    ctx: ExecutionContext,
    tasks,
    seed_graphs: list[ArchGraph] | None = None,
    config: DiscoveryConfig | None = None,
) -> DiscoveryReport:
    """Run the discovery loop. ``registry`` records every candidate with its
    genealogy and fitness; promotion is explicit and gate-gated (§31)."""
    cfg = config or DiscoveryConfig()
    rng = random.Random(f"{cfg.seed}:discovery")
    seed_graphs = seed_graphs or strategy_baselines()
    discovery, held_out = split_tasks(tasks, cfg.discovery_frac, cfg.seed)

    # incumbent on the discovery set
    base = benchmark_architectures(seed_graphs, discovery, ctx, seed=cfg.seed)
    base_rows = _objective_rows(base)
    if not base_rows:
        raise ValueError("no objectively-scored baseline to act as incumbent")
    inc_name = _pick_incumbent(base_rows, cfg.incumbent)
    inc_row = base_rows[inc_name]

    report = DiscoveryReport(incumbent=inc_name, incumbent_success=inc_row.success_rate,
                             discovery_n=len(discovery), held_out_n=len(held_out))

    # register the seed baselines so genealogy has roots
    graph_by_name = {g.name: g for g in seed_graphs}
    for g in seed_graphs:
        if g.name not in registry:
            registry.register(g)
    incumbent_graph = graph_by_name.get(inc_name)

    # population: (graph, parent_name) — starts with the objective baselines
    population: list[tuple[ArchGraph, str | None]] = [
        (graph_by_name[n], None) for n in base_rows if n in graph_by_name]

    for round_idx in range(cfg.n_rounds):
        candidates: list[tuple[str, ArchGraph, str | None]] = []  # (op, graph, parent)
        for parent, _parent_name in population:
            for _ in range(cfg.candidates_per_parent):
                res = random_mutation(parent, rng=rng, max_attempts=cfg.mutation_attempts,
                                      operator_weights=cfg.operator_weights)
                if res is not None:
                    op, cand = res
                    candidates.append((op, cand, parent.name))

        if not candidates:
            report.rounds.append(RoundRecord(round_idx + 1, 0))
            break

        cand_graphs = [c for _, c, _ in candidates]
        cres = benchmark_architectures(cand_graphs, discovery, ctx, seed=cfg.seed)

        # register + record fitness; filter to promotion-eligible
        eligible: list[tuple[str, ArchGraph, str, ArchBenchRow]] = []
        for op, cand, parent in candidates:
            name = cand.name
            if name in registry:
                continue
            registry.register(cand, source="mutated", parent_ids=[parent])
            registry.record_fitness(name, _first_run(cres, name))
            row = cres.rows.get(name)
            if row is None or row.avg_verifier_score is None:
                registry.set_state(name, REJECTED)   # no objective score on checkable tasks
                report.rejected.append(name)
                continue
            if row.avg_tokens > cfg.token_budget_mult * inc_row.avg_tokens + 1e-9:
                registry.set_state(name, REJECTED)   # cost regression (§19)
                report.rejected.append(name)
                continue
            # structural-distinctness guard: a candidate that is structurally
            # IDENTICAL to the incumbent is the same architecture — any
            # "improvement" is reseed luck, not discovery (§47: never promote
            # on a lucky run). Only strictly-different structures compete.
            if incumbent_graph is not None and structural_similarity(cand, incumbent_graph) >= 1.0 - 1e-9:
                registry.set_state(name, REJECTED)   # same architecture
                report.rejected.append(name)
                continue
            registry.set_state(name, BENCHMARKED)
            eligible.append((op, cand, parent, row))
            report.total_candidates += 1

        # held-out check + promotion gate for each eligible candidate
        for op, cand, parent, row in eligible:
            name = cand.name
            # held-out benchmark: candidate vs incumbent
            hres = benchmark_architectures([cand, graph_by_name.get(inc_name, _first_graph(seed_graphs, inc_name))],
                                           held_out, ctx, seed=cfg.seed)
            h_cand = hres.rows.get(name)
            h_inc = hres.rows.get(inc_name)
            if h_cand is None or h_inc is None:
                registry.set_state(name, REJECTED)
                continue
            gate = promotion_gate(
                _successes(row.runs), _successes(inc_row.runs),
                _successes(h_cand.runs), _successes(h_inc.runs),
                [getattr(t, "id", str(t)) for t in discovery],
                min_improvement=cfg.min_improvement)
            # Phase-5 statistical gate: the edge must survive sign test + CI
            if cfg.require_significance:
                stats = validate_statistical(
                    _successes(row.runs), _successes(inc_row.runs),
                    [getattr(t, "id", str(t)) for t in discovery],
                    seed=cfg.seed, alpha=cfg.alpha)
                gate["statistics"] = stats
                if not stats["significant"]:
                    gate["passed"] = False
            # Phase-7 Pareto gate (§13/§23): if the candidate's quality gain
            # is within the margin of the incumbent BUT costs strictly more
            # compute, the added compute did not pay for itself — reject
            # (compression: prefer minimum compute for the desired quality).
            if cfg.require_pareto and gate["passed"]:
                if (inc_row.success_rate >= row.success_rate - cfg.pareto_margin - 1e-9
                        and inc_row.avg_tokens < row.avg_tokens - 1e-9):
                    gate["passed"] = False
                    gate["pareto"] = {
                        "rejected_reason": "compression loss",
                        "inc_success": round(inc_row.success_rate, 4),
                        "cand_success": round(row.success_rate, 4),
                        "inc_tokens": round(inc_row.avg_tokens, 1),
                        "cand_tokens": round(row.avg_tokens, 1),
                    }
            if gate["passed"]:
                registry.set_state(name, INDEPENDENTLY_VERIFIED)
                registry.promote(name)
                report.promoted.append(name)
            else:
                registry.set_state(name, REJECTED)
                report.rejected.append(name)

        # next population = best beam from (eligible candidates + current),
        # with an optional novelty bonus, plus crossover children (§7)
        def _key(item):
            rate, avg, cand, parent = item
            if cfg.novelty_weight > 0:
                nov = registry.novelty(cand)["novelty_score"]
                return (-beam_score(rate, nov, cfg.novelty_weight), -avg)
            return (-rate, -avg)

        scored = [(row.success_rate, row.avg_verifier_score or 0.0, cand, parent)
                  for op, cand, parent, row in eligible]
        scored.sort(key=_key)
        next_pop = [(cand, parent) for _, _, cand, parent in scored[:cfg.beam_width]]
        population = next_pop or population

        # crossover: combine the top two of the current beam (§7)
        crossed: list[tuple[ArchGraph, str | None]] = []
        if cfg.crossover_per_round and len(population) >= 2:
            for child_op, child in crossover_children(
                    population[0][0], population[1][0],
                    cfg.crossover_per_round, rng=rng):
                if child.name not in registry:
                    registry.register(child, source="crossed",
                                      parent_ids=[population[0][0].name, population[1][0].name])
                crossed.append((child, None))
        if crossed:
            population = population + crossed

        report.rounds.append(RoundRecord(
            round_idx + 1, len(eligible),
            beam=[g.name for g, _ in population],
            best_candidate=(scored[0][2].name if scored else None),
            best_success=(scored[0][0] if scored else 0.0)))

        # §13/§40: Pareto frontier summary over incumbent + this round's
        # objective candidates (quality vs compute).
        round_rows = {inc_name: inc_row}
        for _op, _cand, _parent, _row in eligible:
            round_rows[_cand.name] = _row
        report.pareto = pareto_summary(round_rows)

    if cfg.retire_after:
        retire_unfit(registry)
    return report


def _successes(runs) -> list[bool]:
    return [bool(r.success) for r in runs]


def _first_run(res: BenchmarkResult, name: str):
    row = res.rows.get(name)
    return row.runs[0] if row and row.runs else None


def _first_graph(graphs, name):
    return next(g for g in graphs if g.name == name)
