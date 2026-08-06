"""Benchmark package: deterministic task suite + harness + CLI."""

from .harness import (
    bootstrap_ci,
    compare,
    mcnemar,
    render_comparisons,
    render_table,
    run_benchmark,
    save_results,
    serialize_results,
)
from .tasks import make_catalog, split_by_difficulty

__all__ = [
    "make_catalog",
    "split_by_difficulty",
    "run_benchmark",
    "compare",
    "mcnemar",
    "bootstrap_ci",
    "render_table",
    "render_comparisons",
    "serialize_results",
    "save_results",
]
