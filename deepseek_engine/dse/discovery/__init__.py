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
from .loop import (
    DiscoveryConfig,
    DiscoveryReport,
    RoundRecord,
    discover,
    promotion_gate,
    split_tasks,
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
    # lifecycle states
    "GENERATED", "COMPILED", "SANITY_CHECKED", "BENCHMARKED",
    "INDEPENDENTLY_VERIFIED", "STATISTICALLY_VALIDATED", "CANDIDATE", "PROMOTED",
    "REJECTED", "FAILED", "REGRESSED", "SUPERSEDED", "RETIRED", "TERMINAL",
]
