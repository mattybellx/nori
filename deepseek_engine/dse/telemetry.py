"""Metrics recorder: accumulates per-run telemetry and computes summaries.

Every meaningful change must be measured (Master Brief). This module provides the
accumulator used by the benchmark harness; it also computes cost from model
configs so token usage is translated into dollars, not hand-waved.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .config import ModelConfig
from .events import RunResult


@dataclass
class MetricSummary:
    n: int = 0
    success_rate: float = 0.0
    mean_latency_s: float = 0.0
    median_latency_s: float = 0.0
    p90_latency_s: float = 0.0
    mean_tokens_in: float = 0.0
    mean_tokens_out: float = 0.0
    mean_tokens_total: float = 0.0
    mean_cost_usd: float = 0.0
    mean_attempts: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "success_rate": round(self.success_rate, 4),
            "mean_latency_s": round(self.mean_latency_s, 4),
            "median_latency_s": round(self.median_latency_s, 4),
            "p90_latency_s": round(self.p90_latency_s, 4),
            "mean_tokens_in": round(self.mean_tokens_in, 1),
            "mean_tokens_out": round(self.mean_tokens_out, 1),
            "mean_tokens_total": round(self.mean_tokens_total, 1),
            "mean_cost_usd": round(self.mean_cost_usd, 6),
            "mean_attempts": round(self.mean_attempts, 2),
        }


def estimate_cost(result: RunResult, models: dict[str, ModelConfig]) -> float:
    """Translate token usage into USD using per-model prices."""
    total = 0.0
    for step in result.steps:
        cfg = models.get(step.model)
        if cfg is None:
            continue
        total += (step.tokens_in / 1000.0) * cfg.cost_per_1k_in
        total += (step.tokens_out / 1000.0) * cfg.cost_per_1k_out
    return total


def summarize(results: list[RunResult], models: dict[str, ModelConfig] | None = None) -> MetricSummary:
    """Aggregate a list of run results into a summary."""
    models = models or {}
    if not results:
        return MetricSummary()
    latencies = [r.latency_s for r in results]
    costs = [estimate_cost(r, models) for r in results]
    return MetricSummary(
        n=len(results),
        success_rate=sum(1 for r in results if r.success) / len(results),
        mean_latency_s=statistics.fmean(latencies),
        median_latency_s=statistics.median(latencies),
        p90_latency_s=_percentile(latencies, 90),
        mean_tokens_in=statistics.fmean(r.tokens_in for r in results),
        mean_tokens_out=statistics.fmean(r.tokens_out for r in results),
        mean_tokens_total=statistics.fmean(r.tokens_total for r in results),
        mean_cost_usd=statistics.fmean(costs),
        mean_attempts=statistics.fmean(r.attempts for r in results),
    )


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
