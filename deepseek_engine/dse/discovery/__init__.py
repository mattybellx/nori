"""Automated inference-architecture discovery — Phase 1.

See ``ARCHITECTURE_DISCOVERY.md`` for the phased roadmap. This package turns
the project's orchestration into a controlled architecture-discovery
laboratory: architectures are executable graphs (not prompts), baselines are
the existing strategies, and evaluation reuses the project's independent-judge
+ never-worse measurement stack.
"""

from .baselines import (
    adaptive_graph,
    best_of_n_graph,
    default_baselines,
    disagreement_pipeline_graph,
    escalating_graph,
    multi_agent_graph,
    react_graph,
    reflexion_graph,
    self_refine_graph,
    strategy_baselines,
    synthesis_pipeline_graph,
    tree_search_graph,
)
from .compiler import CompileError, compile_graph, validate
from .evaluate import (
    ArchBenchRow,
    BenchmarkResult,
    EquivalenceMismatch,
    EquivalenceReport,
    benchmark_architectures,
    validate_all_baselines,
    validate_equivalence,
)
from .executor import ArchExecutor, ArchRunRecord, NodeEvent
from .graph import ArchEdge, ArchGraph, ArchNode, novelty_against, structural_similarity
from .evolution import (
    beam_score,
    crossover,
    crossover_children,
    retire_unfit,
    unique_child_name,
    validate_statistical,
)
from .invention import (
    CompressionReport,
    InventionReport,
    build_response,
    classify_failure,
    compress_success,
    failure_responses_for,
    invent_from_failures,
)
from .loop import (
    DiscoveryConfig,
    DiscoveryReport,
    RoundRecord,
    discover,
    promotion_gate,
    split_tasks,
)
from .meta import (
    DiscoveryStrategy,
    MetaOutcome,
    MetaReport,
    candidate_strategies,
    cheap_strategy,
    default_strategy,
    meta_gate,
    meta_search,
    quality_focused_strategy,
    run_discovery_strategy,
)
from .mutations import (
    append_verify,
    best_of_n_ify,
    branch,
    delete_node,
    duplicate_parallel,
    duplicate_sequential,
    gather_join,
    insert_after,
    insert_before,
    insert_verify,
    mutation_operators,
    random_mutation,
    substitute,
    swap_primitives,
)
from .pareto import (
    ParetoPoint,
    compression_win,
    compute_efficiency,
    dominates,
    pareto_frontier,
    pareto_summary,
    point_from_row,
)
from .primitive_invention import (
    PrimitiveProposal,
    PrimitiveValidation,
    propose_majority_vote,
    promote_primitive,
    validate_primitive,
)
from .primitives import (
    ExecutionContext,
    NodeOutput,
    Primitive,
    registered_names,
    register,
)
from .registry import (
    ArchitectureRegistry,
    ArchRecord,
    BENCHMARKED,
    CANDIDATE,
    COMPILED,
    FAILED,
    GENERATED,
    INDEPENDENTLY_VERIFIED,
    PROMOTED,
    REGRESSED,
    REJECTED,
    RETIRED,
    SANITY_CHECKED,
    STATISTICALLY_VALIDATED,
    SUPERSEDED,
    TERMINAL,
)
from .routing import (
    TaskProfile,
    UCRouter,
    profile_task,
    run_routing_experiment,
)

# The real-domain bridge (real DeepSeek + independent judge) is exported
# lazily: it imports the live provider stack, and eager import would break
# `python -m dse.discovery.real_domain` (module already in sys.modules).
_REAL_DOMAIN_EXPORTS = {
    "RealDomainReport",
    "build_freeform_context",
    "build_summary",
    "preference_real",
    "run_architectures_real",
    "run_real_experiment",
    "score_real",
}


def __getattr__(name: str):
    if name in _REAL_DOMAIN_EXPORTS:
        from . import real_domain
        return getattr(real_domain, name)
    raise AttributeError(f"module 'dse.discovery' has no attribute {name!r}")

__all__ = [
    "ArchEdge", "ArchGraph", "ArchNode",
    "ArchExecutor", "ArchRunRecord", "ArchitectureRegistry", "ArchRecord",
    "ArchBenchRow", "BenchmarkResult", "EquivalenceMismatch", "EquivalenceReport",
    "benchmark_architectures", "validate_all_baselines", "validate_equivalence",
    "CompileError", "compile_graph", "validate",
    "ExecutionContext", "NodeOutput", "Primitive",
    "register", "registered_names",
    "DiscoveryConfig", "DiscoveryReport", "RoundRecord",
    "discover", "promotion_gate", "split_tasks",
    "crossover", "crossover_children", "unique_child_name",
    "validate_statistical", "retire_unfit", "beam_score",
    "TaskProfile", "UCRouter", "profile_task", "run_routing_experiment",
    "ParetoPoint", "compression_win", "compute_efficiency", "dominates",
    "pareto_frontier", "pareto_summary", "point_from_row",
    "classify_failure", "build_response", "failure_responses_for",
    "invent_from_failures", "InventionReport",
    "compress_success", "CompressionReport",
    "PrimitiveProposal", "PrimitiveValidation",
    "validate_primitive", "promote_primitive", "propose_majority_vote",
    "DiscoveryStrategy", "MetaOutcome", "MetaReport",
    "default_strategy", "quality_focused_strategy", "cheap_strategy",
    "candidate_strategies", "run_discovery_strategy", "meta_search",
    "meta_gate",
    "insert_before", "insert_after", "insert_verify", "append_verify",
    "delete_node", "substitute", "swap_primitives",
    "duplicate_parallel", "duplicate_sequential", "branch", "gather_join",
    "best_of_n_ify",
    "random_mutation", "mutation_operators",
    "novelty_against", "structural_similarity",
    "react_graph", "best_of_n_graph", "reflexion_graph", "self_refine_graph",
    "tree_search_graph", "escalating_graph", "adaptive_graph", "multi_agent_graph",
    "synthesis_pipeline_graph", "disagreement_pipeline_graph",
    "default_baselines", "strategy_baselines",
    # real-domain bridge (real DeepSeek + independent judge)
    "RealDomainReport", "build_freeform_context", "build_summary",
    "preference_real", "run_architectures_real", "run_real_experiment",
    "score_real",
    # lifecycle states
    "GENERATED", "COMPILED", "SANITY_CHECKED", "BENCHMARKED",
    "INDEPENDENTLY_VERIFIED", "STATISTICALLY_VALIDATED", "CANDIDATE", "PROMOTED",
    "REJECTED", "FAILED", "REGRESSED", "SUPERSEDED", "RETIRED", "TERMINAL",
]
