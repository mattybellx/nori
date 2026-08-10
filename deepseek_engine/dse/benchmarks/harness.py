"""Benchmark harness — paired, deterministic, statistically grounded.

Per the Master Brief Honesty Rules, every comparison reports: n, success rate,
mean/median/p90 latency, mean tokens, mean cost (USD), plus full metadata
(models, seed, config). Statistical significance uses McNemar's test on the
paired discordant counts and a bootstrap CI on the difference of rates —
implemented here without external dependencies (no scipy).
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field

from ..agent import Agent, Budget
from ..config import EngineConfig, ModelConfig
from ..environment import Task
from ..events import RunResult
from ..telemetry import MetricSummary, summarize


# ---------------------------------------------------------------------------
# Statistics (dependency-free)
# ---------------------------------------------------------------------------
def _lower_incomplete_gamma(a: float, x: float, iters: int = 300) -> float:
    """P(a, x) via series expansion; valid for moderate x."""
    if x <= 0:
        return 0.0
    s = 1.0 / a
    term = 1.0 / a
    for n in range(1, iters):
        term *= x / (a + n)
        s += term
        if abs(term) < 1e-14:
            break
    return math.exp(a * math.log(x) - x) * s


def chi2_cdf(x: float, k: int) -> float:
    """CDF of the chi-squared distribution with k degrees of freedom.

    For k == 1 (the only case McNemar needs) we use the exact closed form
    ``erf(sqrt(x/2))``; the series expansion is retained for other k.
    """
    if x <= 0:
        return 0.0
    if k == 1:
        return max(0.0, min(1.0, math.erf(math.sqrt(x / 2.0))))
    a = k / 2.0
    return min(1.0, _lower_incomplete_gamma(a, x / 2.0) / math.gamma(a))


@dataclass
class McNemarResult:
    strategy_a: str
    strategy_b: str
    b01: int  # A ok, B fail
    b10: int  # A fail, B ok
    statistic: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def mcnemar(a: list[RunResult], b: list[RunResult]) -> McNemarResult:
    """Paired McNemar test with Edwards continuity correction.

    ``a`` and ``b`` must be paired (same task order / same task ids).
    """
    if len(a) != len(b):
        raise ValueError("mcnemar requires paired runs of equal length")
    b01 = sum(1 for ra, rb in zip(a, b) if ra.success and not rb.success)
    b10 = sum(1 for ra, rb in zip(a, b) if not ra.success and rb.success)
    total = b01 + b10
    if total == 0:
        statistic, p_value = 0.0, 1.0
    else:
        statistic = (abs(b01 - b10) - 1.0) ** 2 / total
        p_value = 1.0 - chi2_cdf(statistic, 1)
    return McNemarResult(
        strategy_a=a[0].strategy if a else "?",
        strategy_b=b[0].strategy if b else "?",
        b01=b01,
        b10=b10,
        statistic=statistic,
        p_value=p_value,
    )


def bootstrap_ci(
    a: list[RunResult], b: list[RunResult], iters: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap CI (95%) for the difference of success rates (a - b)."""
    if len(a) != len(b):
        raise ValueError("bootstrap_ci requires paired runs of equal length")
    n = len(a)
    a_bin = [1 if r.success else 0 for r in a]
    b_bin = [1 if r.success else 0 for r in b]
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = statistics.fmean(a_bin[i] for i in idx)
        db = statistics.fmean(b_bin[i] for i in idx)
        diffs.append(da - db)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[int(0.975 * iters)]
    point = statistics.fmean(diffs)
    return lo, hi, point


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via ``math.erf`` (dependency-free)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(
    a: list[float], b: list[float], exact_max_n: int = 16, seed: int = 0
) -> tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test on continuous scores.

    Returns ``(w_plus, p_value)`` — the sum of ranks of the positive
    differences and the two-sided p-value against "no difference".

    Zero-difference pairs are dropped; tied absolute differences get average
    ranks. For ``n <= exact_max_n`` the p-value is exact (enumerates all 2**n
    sign assignments); otherwise a normal approximation with continuity and
    tie corrections is used. This is the right test for the free-form
    benchmark, where the quality metric is a (de-noised) judge score rather
    than a pass/fail bit — McNemar can't see a 6 -> 8 improvement, Wilcoxon
    can.
    """
    if len(a) != len(b):
        raise ValueError("wilcoxon requires paired samples of equal length")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    diffs = [d for d in diffs if abs(d) > 1e-12]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    # rank |differences|, using average ranks for ties
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    tie_sizes: list[int] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(diffs[order[j + 1]]) - abs(diffs[order[i]])) <= 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        tie_sizes.append(j - i + 1)
        i = j + 1
    w_plus = sum(ranks[k] for k in range(n) if diffs[k] > 0)
    w_obs = min(w_plus, n * (n + 1) / 2.0 - w_plus)

    if n <= exact_max_n:
        total = 2 ** n
        count = 0
        for mask in range(total):
            wp = 0.0
            for k in range(n):
                if (mask >> k) & 1:
                    wp += ranks[k]
            if wp <= w_obs + 1e-12:
                count += 1
        return w_plus, min(1.0, 2.0 * count / total)

    # normal approximation: E = n(n+1)/4, tie-corrected variance, CC 0.5
    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    for t in tie_sizes:
        if t > 1:
            var -= (t ** 3 - t) / 48.0
    z = (w_obs - mu + 0.5) / math.sqrt(max(var, 1e-12))
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return w_plus, min(1.0, p)


def sign_test(favor: int, total: int) -> float:
    """Two-sided binomial sign test on paired preferences (dependency-free).

    ``favor`` of ``total`` paired comparisons preferred the treatment
    (ties excluded). Under H0 each comparison favors either side with p=0.5;
    the two-sided p-value is ``2 * P(X <= min(favor, total-favor))``.
    """
    total = int(total)
    favor = int(favor)
    if total <= 0:
        return 1.0
    k = min(favor, total - favor)
    p_lower = 0.0
    for i in range(k + 1):
        p_lower += math.comb(total, i) * (0.5 ** total)
    return min(1.0, 2.0 * p_lower)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def run_benchmark(
    agents: list[Agent],
    catalog: dict[str, Task],
    *,
    seed: int = 0,
    models: dict[str, ModelConfig] | None = None,
    budget: Budget | None = None,
    llm: object | None = None,
    progress=None,
) -> dict[str, list[RunResult]]:
    """Run every agent over the same catalog (shared order) with the same seed.

    Methodological note: for a valid paired comparison, every agent must face
    *identical* stochastic draws on each task. If ``llm`` (a MockLLM) is given,
    it is re-seeded per (agent, task) so all strategies see the same draws.

    ``progress(done, message)`` is called after each task (optional UI hook).
    """
    models = models or {}
    tasks = list(catalog.values())
    total = len(agents) * len(tasks)
    done = 0
    results: dict[str, list[RunResult]] = {}
    for agent in agents:
        runs: list[RunResult] = []
        for task in tasks:
            if llm is not None and hasattr(llm, "reset"):
                llm.reset(f"{seed}:{agent.name}:{task.id}")
            runs.append(agent.solve(task, budget))
            done += 1
            if progress is not None:
                progress(done, f"{agent.name}:{task.id}")
        results[agent.name] = runs
    return results


def run_multi_seed(
    agent_factory,
    catalog: dict[str, Task],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    models: dict[str, ModelConfig] | None = None,
    budget: Budget | None = None,
    progress=None,
) -> dict[str, list[RunResult]]:
    """Run the paired benchmark across several seeds and combine the runs.

    ``agent_factory(seed)`` must return ``(llm, agents)`` — a fresh LLM and a
    set of agents **bound to that same LLM** (the agents' ``.llm`` must be the
    object ``run_benchmark`` reseeds). Sharing agents built on a different LLM
    would silently couple the RNG streams across agents — a real pairing bug
    that was found and fixed here.

    Because the catalog is shared and every (agent, task, seed) is reseeded via
    ``reset(f"{seed}:{agent}:{task}")``, concatenating the per-seed runs gives
    one larger, still-paired dataset for McNemar/bootstrap — more power than a
    single seed (n × len(seeds)).
    """
    models = models or {}
    combined: dict[str, list[RunResult]] = {}
    for seed in seeds:
        llm, agents = agent_factory(seed)
        runs = run_benchmark(
            agents, catalog, seed=seed, models=models, budget=budget, llm=llm,
            progress=progress,
        )
        for name, strategy_runs in runs.items():
            combined.setdefault(name, []).extend(strategy_runs)
    return combined


def compare(results: dict[str, list[RunResult]]) -> list[McNemarResult]:
    """Pairwise McNemar comparisons over all strategy pairs."""
    names = list(results)
    comparisons: list[McNemarResult] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            comparisons.append(mcnemar(results[name_a], results[name_b]))
    return comparisons


def render_table(results: dict[str, list[RunResult]], models: dict[str, ModelConfig]) -> str:
    width = max(14, max(len(name) for name in results)) + 2
    header = (
        f"{'strategy':<{width}}{'n':>5}{'success':>10}{'mean_lat':>10}"
        f"{'med_lat':>10}{'p90_lat':>10}{'mean_tok':>10}{'cost_usd':>11}{'attempts':>10}"
    )
    rows = [header, "-" * len(header)]
    for name, runs in results.items():
        s = summarize(runs, models)
        rows.append(
            f"{name:<{width}}{s.n:>5}{s.success_rate:>10.3f}{s.mean_latency_s:>10.3f}"
            f"{s.median_latency_s:>10.3f}{s.p90_latency_s:>10.3f}"
            f"{s.mean_tokens_total:>10.1f}{s.mean_cost_usd:>11.6f}{s.mean_attempts:>10.2f}"
        )
    return "\n".join(rows)


def render_comparisons(comparisons: list[McNemarResult]) -> str:
    names = [n for c in comparisons for n in (c.strategy_a, c.strategy_b)]
    width = max(14, len(max(names, default="strategy"))) + 2
    header = f"{'A':<{width}}{'B':<{width}}{'b01':>5}{'b10':>5}{'chi2':>8}{'p':>10}{'sig':>6}"
    rows = [header, "-" * len(header)]
    for c in comparisons:
        rows.append(
            f"{c.strategy_a:<{width}}{c.strategy_b:<{width}}{c.b01:>5}{c.b10:>5}"
            f"{c.statistic:>8.2f}{c.p_value:>10.4f}{'YES' if c.significant else 'no':>6}"
        )
    return "\n".join(rows)


def serialize_results(
    results: dict[str, list[RunResult]],
    models: dict[str, ModelConfig],
    config: EngineConfig,
    seed: int | None = None,
    seeds: tuple[int, ...] | None = None,
    provider: str = "mock",
) -> dict:
    """Serialize the full record: summaries + comparisons + metadata."""
    comparisons = compare(results)
    if seeds is not None:
        seeds_list = tuple(seeds)
    elif seed is not None:
        seeds_list = (seed,)
    else:
        seeds_list = ()
    return {
        "metadata": {
            "seed": seeds_list[0] if len(seeds_list) == 1 else seed,
            "seeds": list(seeds_list),
            "provider": provider,
            "engine_config": {
                "max_trials": config.max_trials,
                "max_search_nodes": config.max_search_nodes,
                "multi_agent_proposers": config.multi_agent_proposers,
                "flags": config.flags,
            },
            "models": {k: v.__dict__ for k, v in models.items()},
            "hardware": _hardware_note(),
        },
        "summaries": {
            name: summarize(runs, models).as_dict() for name, runs in results.items()
        },
        "comparisons": [
            {
                "A": c.strategy_a,
                "B": c.strategy_b,
                "b01": c.b01,
                "b10": c.b10,
                "chi2": c.statistic,
                "p_value": c.p_value,
                "significant": c.significant,
            }
            for c in comparisons
        ],
    }


def _hardware_note() -> str:
    import platform

    return f"{platform.system()} {platform.machine()} (python {platform.python_version()})"


def save_results(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
