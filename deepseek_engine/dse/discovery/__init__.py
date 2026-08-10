"""Automated inference-architecture discovery — Phase 1.

See ``ARCHITECTURE_DISCOVERY.md`` for the phased roadmap. This package turns
the project's orchestration into a controlled architecture-discovery
laboratory: architectures are executable graphs (not prompts), baselines are
the existing strategies, and evaluation reuses the project's independent-judge
+ never-worse measurement stack.
"""

from .baselines import (
    best_of_n_graph,
    default_baselines,
    disagreement_pipeline_graph,
    react_graph,
    reflexion_graph,
    self_refine_graph,
    synthesis_pipeline_graph,
    tree_search_graph,
)
from .compiler import CompileError, compile_graph, validate
from .executor import ArchExecutor, ArchRunRecord, NodeEvent
from .graph import ArchEdge, ArchGraph, ArchNode, novelty_against, structural_similarity
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
    "CompileError", "compile_graph", "validate",
    "ExecutionContext", "NodeOutput", "Primitive",
    "register", "registered_names",
    "novelty_against", "structural_similarity",
    "react_graph", "best_of_n_graph", "reflexion_graph", "self_refine_graph",
    "tree_search_graph", "synthesis_pipeline_graph", "disagreement_pipeline_graph",
    "default_baselines",
    # lifecycle states
    "GENERATED", "COMPILED", "SANITY_CHECKED", "BENCHMARKED",
    "INDEPENDENTLY_VERIFIED", "STATISTICALLY_VALIDATED", "CANDIDATE", "PROMOTED",
    "REJECTED", "FAILED", "REGRESSED", "SUPERSEDED", "RETIRED", "TERMINAL",
]
