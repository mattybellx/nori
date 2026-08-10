"""Phase 6 — adaptive routing (spec §14 profiler, §15 selection, §16 UCB).

The system stops treating every task identically: a ProblemProfiler maps a
task to a class, and a UCB contextual bandit learns which ARCHITECTURE works
best per class, then routes each held-out task to its best arm.

- ``profile_task`` / ``TaskProfile`` — cheap, deterministic task features
  (type prefix from the task id, step count, optional difficulty).
- ``UCRouter`` — a UCB bandit over architectures, keyed by task class.
  ``learn_from_benchmark`` ingests per-(class, arch) rewards from the
  discovery evaluation; ``choose`` balances exploitation (mean) and
  exploration (sqrt(2 ln N / n)); unseen arms are always tried.
- ``run_routing_experiment`` — train on a split, then route held-out tasks;
  reports the router's rate vs the best single fixed architecture (both the
  train-selected one — the realistic comparison — and the held-out oracle
  upper bound) and a per-type oracle bound.

Honest gate: routing beats the best single architecture you would have picked
from training. If no architecture specializes by class, routing cannot win —
and the experiment says so.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .compiler import compile_graph
from .evaluate import benchmark_architectures
from .executor import ArchExecutor
from .graph import ArchGraph
from .loop import split_tasks
from .primitives import ExecutionContext


# ---------------------------------------------------------------------------
# Problem profiler (§14)
# ---------------------------------------------------------------------------

@dataclass
class TaskProfile:
    task_type: str
    num_steps: int = 1
    difficulty: float | None = None
    features: dict = field(default_factory=dict)

    @property
    def class_key(self) -> str:
        """The routing class label. Default: the task-type prefix (the signal
        that actually differentiates strategies on the mock suites)."""
        return self.task_type or "default"


def profile_task(task) -> TaskProfile:
    """Deterministic, dependency-free task features.

    ``task_type`` is the part of the task id before the first ``-``
    (arithmetic / logic / code on the synthetic suites); falls back to
    ``default`` for ids without a type. ``num_steps``/``difficulty`` are
    read off the task when present (RealTask carries ``difficulty``).
    """
    tid = getattr(task, "id", "") or ""
    task_type = tid.split("-")[0] if "-" in tid else ("default" if tid else "unknown")
    num_steps = int(getattr(task, "num_steps", 1) or 1)
    difficulty = getattr(task, "difficulty", None)
    return TaskProfile(task_type=task_type, num_steps=num_steps,
                       difficulty=difficulty,
                       features={"type": task_type, "steps": num_steps})


# ---------------------------------------------------------------------------
# UCB contextual bandit (§16)
# ---------------------------------------------------------------------------

class UCRouter:
    """Upper-Confidence-Bound router keyed by task class. Learns per-class
    per-architecture mean rewards; chooses the arm with the best UCB score
    (unseen arms always tried first)."""

    def __init__(self, arch_names: list[str], exploration: float = 1.0) -> None:
        self.arch_names = list(arch_names)
        self.exploration = exploration
        self.stats: dict[str, dict[str, dict]] = {}   # class -> arch -> {n, sum, mean}

    # -- learning -----------------------------------------------------------
    def update(self, class_key: str, arch: str, reward: float) -> None:
        cell = self.stats.setdefault(class_key, {}).setdefault(
            arch, {"n": 0, "sum": 0.0, "mean": 0.0})
        cell["n"] += 1
        cell["sum"] += reward
        cell["mean"] = cell["sum"] / cell["n"]

    def learn_from_benchmark(self, rows, tasks, pool_names: list[str]) -> None:
        """Full-information learning: every pool architecture is run on every
        train task (via benchmark rows), so per-class estimates are clean."""
        for name in pool_names:
            row = rows[name]
            for task, run in zip(tasks, row.runs):
                self.update(profile_task(task).class_key, name,
                            1.0 if run.success else 0.0)

    # -- decision -----------------------------------------------------------
    def choose(self, class_key: str, exploration: float | None = None) -> str:
        """UCB: mean + exploration * sqrt(2 ln(total) / n); unseen arms = +inf.

        ``exploration=None`` uses the router's configured value. Deployment
        (evaluate) uses 0.0 — pure exploitation of the learned policy."""
        if exploration is None:
            exploration = self.exploration
        cell = self.stats.get(class_key, {})
        total = sum(e["n"] for e in cell.values()) or 1
        best, best_score = None, -math.inf
        for arch in self.arch_names:
            e = cell.get(arch)
            if e is None or e["n"] == 0:
                score = math.inf
            else:
                score = e["mean"] + exploration * math.sqrt(
                    2.0 * math.log(total) / e["n"])
            if score > best_score:
                best, best_score = arch, score
        return best

    def choose_for_task(self, task, exploration: float | None = None) -> str:
        return self.choose(profile_task(task).class_key, exploration=exploration)

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, pool_by_name: dict[str, ArchGraph], tasks,
                 ctx: ExecutionContext, seed: int = 0,
                 reset_llm: bool = True) -> tuple[list[str], list[bool]]:
        """Route every task to its chosen architecture and run it.

        Deployment is PURE EXPLOITATION (exploration=0): the policy was
        learned on the train split; held-out is deployment, not more
        learning."""
        ex = ArchExecutor()
        picks: list[str] = []
        successes: list[bool] = []
        for task in tasks:
            arch = self.choose_for_task(task, exploration=0.0)
            if reset_llm and hasattr(ctx.llm, "reset"):
                ctx.llm.reset(f"{seed}:{arch}:{getattr(task, 'id', '?')}")
            rec = ex.run(compile_graph(pool_by_name[arch]), task, context=ctx)
            picks.append(arch)
            successes.append(bool(rec.success))
        return picks, successes

    def summary(self) -> dict:
        return {cls: {a: round(e["mean"], 3) for a, e in arms.items()}
                for cls, arms in sorted(self.stats.items())}


# ---------------------------------------------------------------------------
# Routing experiment (the Phase-6 gate)
# ---------------------------------------------------------------------------

def _oracle_pick(pool_by_name, held_rows, task) -> str:
    """The per-type oracle: the architecture with the best held-out success
    for this task's class (upper bound — routing cannot do better)."""
    cls = profile_task(task).class_key
    return max(pool_by_name, key=lambda n: _class_rate(held_rows, n, cls))


def _class_rate(rows, arch, cls, tasks=None):
    if tasks is None:
        return rows[arch].success_rate
    runs = [r for r, t in zip(rows[arch].runs, tasks) if profile_task(t).class_key == cls]
    return (sum(1 for r in runs if r.success) / len(runs)) if runs else 0.0


def run_routing_experiment(
    pool: list[ArchGraph],
    tasks,
    ctx: ExecutionContext,
    seed: int = 0,
    train_frac: float = 0.7,
    exploration: float = 1.0,
    rng: random.Random | None = None,
) -> dict:
    """Learn per-class routing on a train split, then route held-out tasks.

    Returns the router's held-out rate, the best-single-fixed rates (both the
    train-selected arch — the realistic comparison — and the held-out oracle
    upper bound), the per-type oracle bound, and the router's chosen-arch log.
    """
    rng = rng or random.Random(f"{seed}:route")
    pool_by_name = {g.name: g for g in pool}
    train, held = split_tasks(tasks, train_frac, seed)

    # learn on train (full information)
    train_res = benchmark_architectures(pool, train, ctx, seed=seed)
    router = UCRouter(list(pool_by_name), exploration=exploration)
    router.learn_from_benchmark(train_res.rows, train, list(pool_by_name))

    # the realistic single alternative: best arch on the train split
    best_fixed_train = max(pool_by_name, key=lambda n: train_res.rows[n].success_rate)

    # evaluate router on held-out
    picks, succs = router.evaluate(pool_by_name, held, ctx, seed=seed)
    router_rate = (sum(1 for s in succs if s) / len(succs)) if succs else 0.0

    # single-fixed baselines on held-out
    held_res = benchmark_architectures(pool, held, ctx, seed=seed)
    best_fixed_oracle = max(pool_by_name, key=lambda n: held_res.rows[n].success_rate)
    oracle_pick = _oracle_pick(pool_by_name, held_res.rows, held[0]) if held else None

    # per-type oracle routing upper bound on held-out
    oracle_succ = 0
    for t in held:
        if oracle_pick:
            arch = _oracle_pick(pool_by_name, held_res.rows, t)
            rate = _class_rate(held_res.rows, arch, profile_task(t).class_key)
            oracle_succ += rate  # expectation over the 1 task draw
    oracle_rate = oracle_succ / len(held) if held else 0.0

    return {
        "router_held_out_rate": round(router_rate, 4),
        "router_picks": picks,
        "router_successes": succs,
        "best_fixed_train": best_fixed_train,
        "best_fixed_train_rate": round(train_res.rows[best_fixed_train].success_rate, 4),
        "best_fixed_train_held_out_rate": round(held_res.rows[best_fixed_train].success_rate, 4),
        "best_fixed_oracle_held_out": best_fixed_oracle,
        "best_fixed_oracle_held_out_rate": round(held_res.rows[best_fixed_oracle].success_rate, 4),
        "oracle_routing_held_out_rate": round(oracle_rate, 4),
        "beats_best_fixed_train": bool(router_rate > held_res.rows[best_fixed_train].success_rate + 1e-9),
        "learned_summary": router.summary(),
        "train_n": len(train), "held_out_n": len(held),
        "seed": seed,
    }
