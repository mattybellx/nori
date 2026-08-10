"""Phase 7 — compute optimization & the Pareto frontier (spec §13/§23/§40).

An architecture is valuable when it finds a NEW point on the Pareto frontier
(quality/reliability vs cost/tokens/latency) — not merely when it is "better"
at any cost. This module provides:

- ``ParetoPoint`` / ``point_from_row`` — map a benchmark row to the
  multi-objective point (maximize success_rate & verifier score; minimize
  tokens & latency).
- ``dominates`` — strict multi-objective dominance.
- ``pareto_frontier`` — the non-dominated set (the §40 leaderboard/frontier).
- ``compute_efficiency`` — quality per unit compute (the §13 "more per
  token" view).
- ``compression_win`` (§23) — when a SIMPLER architecture achieves
  equivalent-or-better quality at lower cost, the extra compute in the
  complex one was unnecessary. This is the "minimum computation required for
  the desired reliability" check.
- ``pareto_summary`` — a dashboard-shaped dict (frontier / dominated /
  efficiency ranking) for reports and the future §40 dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_EPS = 1e-9


@dataclass
class ParetoPoint:
    name: str
    success_rate: float
    avg_verifier_score: float | None = None
    avg_tokens: float = 0.0
    avg_latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success_rate": round(self.success_rate, 4),
            "avg_verifier_score": (round(self.avg_verifier_score, 4)
                                   if self.avg_verifier_score is not None else None),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_latency_s": round(self.avg_latency_s, 4),
            "efficiency": round(compute_efficiency(self), 6),
        }


def point_from_row(name: str, row) -> ParetoPoint:
    """Build a Pareto point from an ArchBenchRow (or anything with those
    attributes)."""
    return ParetoPoint(
        name=name,
        success_rate=float(row.success_rate),
        avg_verifier_score=(float(row.avg_verifier_score)
                            if row.avg_verifier_score is not None else None),
        avg_tokens=float(row.avg_tokens),
        avg_latency_s=float(row.avg_latency_s),
    )


def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
    """Does ``a`` strictly dominate ``b``?

    Strictly better on at least one objective and not worse on any.
    Objectives: success_rate (max), avg_verifier_score (max, only compared
    when BOTH are non-None), avg_tokens (min), avg_latency_s (min).
    """
    better = 0
    if a.success_rate > b.success_rate + _EPS:
        better += 1
    if (a.avg_verifier_score is not None and b.avg_verifier_score is not None
            and a.avg_verifier_score > b.avg_verifier_score + _EPS):
        better += 1
    if a.avg_tokens < b.avg_tokens - _EPS:
        better += 1
    if a.avg_latency_s < b.avg_latency_s - _EPS:
        better += 1
    if better == 0:
        return False
    not_worse = (
        a.success_rate >= b.success_rate - _EPS
        and a.avg_tokens <= b.avg_tokens + _EPS
        and a.avg_latency_s <= b.avg_latency_s + _EPS
        and (a.avg_verifier_score is None or b.avg_verifier_score is None
             or a.avg_verifier_score >= b.avg_verifier_score - _EPS)
    )
    return not_worse


def pareto_frontier(points: list[ParetoPoint]):
    """Return (frontier_names, dominated_by) — the non-dominated set and a
    map of every dominated point to the names that dominate it."""
    frontier = []
    dominated_by: dict[str, list[str]] = {}
    for p in points:
        dominators = [q.name for q in points
                      if q is not p and dominates(q, p)]
        if dominators:
            dominated_by[p.name] = dominators
        else:
            frontier.append(p.name)
    return frontier, dominated_by


def compute_efficiency(p: ParetoPoint) -> float:
    """Success per unit of compute: success_rate / avg_tokens (quality per
    token — the §13 "more for less" view)."""
    tokens = max(p.avg_tokens, 1e-9)
    return p.success_rate / tokens


def compression_win(complex_p: ParetoPoint, simple_p: ParetoPoint) -> bool:
    """§23: did the complex architecture's extra compute pay for itself?

    True when the SIMPLE architecture dominates the complex one — i.e. it
    achieves equivalent-or-better quality/reliability at equal-or-lower
    compute (and strictly lower on at least one axis). That means the
    complex architecture's added stages were unnecessary.
    """
    return dominates(simple_p, complex_p)


def pareto_summary(rows: dict[str, Any]) -> dict:
    """Dashboard-shaped summary of a benchmark-rows dict (§40)."""
    points = [point_from_row(name, row) for name, row in rows.items()]
    frontier, dominated_by = pareto_frontier(points)
    efficiency_rank = sorted(points, key=compute_efficiency, reverse=True)
    return {
        "frontier": list(frontier),
        "dominated_by": {k: v for k, v in sorted(dominated_by.items())},
        "efficiency_rank": [p.name for p in efficiency_rank],
        "points": {p.name: p.to_dict() for p in points},
    }
