"""Baseline architectures as executable graphs (Phase 1, spec §6).

The project's existing strategies are the INITIAL architecture library. Two
kinds of baseline graphs are provided:

1. **Strategy baselines** — one node runs the whole existing Agent
   (``strategy`` primitive): react, best_of_n, reflexion, self_refine,
   tree_search. Exact reuse: the node IS the measured strategy.
2. **Atomic pipelines** — composed from fine-grained primitives to prove the
   executor expresses real compositions: the production never-worse
   synthesis pipeline, and a conditional-routing disagreement pipeline
   (spec §7 routing, §35 architecture-specific compute).

These are static graph definitions — no runtime dependencies.
"""

from __future__ import annotations

from .graph import ArchGraph, ArchNode


def _strategy_arch(name: str, strategy: str, label: str) -> ArchGraph:
    """A single-node architecture whose node IS the existing strategy.

    The node's output is the strategy's RunResult, so the recorded success and
    answer come straight from the measured strategy (no lossy passthrough).
    """
    g = ArchGraph(name=name, entry="s", exit="s")
    g.add_node(ArchNode("s", "strategy", params={"name": strategy},
                        meta={"label": label}))
    return g


def react_graph() -> ArchGraph:
    return _strategy_arch("react", "react", "single generate→verify")


def best_of_n_graph() -> ArchGraph:
    return _strategy_arch("best_of_n", "best_of_n", "N drafts, verifier-selected")


def reflexion_graph() -> ArchGraph:
    return _strategy_arch("reflexion", "reflexion", "retry with episodic reflection")


def self_refine_graph() -> ArchGraph:
    return _strategy_arch("self_refine", "self_refine", "verifier-gated refinement")


def tree_search_graph() -> ArchGraph:
    return _strategy_arch("tree_search", "tree_search", "MCTS with per-step tests")


def synthesis_pipeline_graph() -> ArchGraph:
    """The production never-worse pipeline as an executable graph:

    generate ×3 (react/reflexion/self_refine)
      → score_items        (independent-judge each candidate)
      → selection_guard    (never ship a candidate below the baseline floor)
      → synthesize         (best-of-all merge)
      → synthesis_guard    (ship only if grounded AND scored >= winner)
    """
    g = ArchGraph(name="synthesis_pipeline", entry="gen1", exit="guard")
    g.add_node(ArchNode("gen1", "generate", params={"extra": "Produce a draft."},
                        meta={"label": "react-style draft"}))
    g.add_node(ArchNode("gen2", "generate", params={"extra": "Produce a draft."},
                        meta={"label": "reflexion-style draft"}))
    g.add_node(ArchNode("gen3", "generate", params={"extra": "Produce a draft."},
                        meta={"label": "self_refine-style draft"}))
    g.add_node(ArchNode("score", "score_items", params={"samples": 3},
                        meta={"label": "judge each candidate"}))
    g.add_node(ArchNode("select", "selection_guard", params={"baseline": "gen1"},
                        meta={"label": "never-worse selection"}))
    g.add_node(ArchNode("gather", "gather", meta={"label": "collect candidates"}))
    g.add_node(ArchNode("synth", "synthesize", meta={"label": "best-of-all merge"}))
    g.add_node(ArchNode("guard", "synthesis_guard", params={"samples": 3},
                        meta={"label": "no-regression gate"}))
    for src in ("gen1", "gen2", "gen3"):
        g.add_edge(src, "score", port=src)
        g.add_edge(src, "gather")
    g.add_edge("score", "select")
    g.add_edge("gather", "synth")
    g.add_edge("synth", "guard", port="synth")
    g.add_edge("gather", "guard", port="candidates")
    g.add_edge("select", "guard", port="winner")
    return g


def disagreement_pipeline_graph(threshold: float = 1.0) -> ArchGraph:
    """Architecture-specific compute (§35) via conditional routing (§7):

    generate ×3 → score_items → route_disagreement
        ├── low  → synthesize
        └── high → targeted generate → synthesize
    both → done
    """
    g = ArchGraph(name="disagreement_pipeline", entry="gen1", exit="done")
    g.add_node(ArchNode("gen1", "generate", meta={"label": "draft A"}))
    g.add_node(ArchNode("gen2", "generate", meta={"label": "draft B"}))
    g.add_node(ArchNode("score", "score_items", params={"samples": 3}))
    g.add_node(ArchNode("route", "route_disagreement",
                        params={"threshold": threshold}))
    g.add_node(ArchNode("synth_low", "synthesize", meta={"label": "cheap merge"}))
    g.add_node(ArchNode("gen_targeted", "generate",
                        params={"extra": "Carefully reconsider; produce a stronger draft."},
                        meta={"label": "targeted search"}))
    g.add_node(ArchNode("synth_high", "synthesize", meta={"label": "merge after search"}))
    g.add_node(ArchNode("done", "identity", meta={"label": "final answer"}))
    for src in ("gen1", "gen2"):
        g.add_edge(src, "score", port=src)
    g.add_edge("score", "route")
    g.add_edge("route", "synth_low", port="low")
    g.add_edge("route", "gen_targeted", port="high")
    g.add_edge("gen_targeted", "synth_high")
    g.add_edge("synth_low", "done")
    g.add_edge("synth_high", "done")
    return g


def default_baselines() -> list[ArchGraph]:
    """The initial architecture library (spec §6)."""
    return [
        react_graph(),
        best_of_n_graph(),
        reflexion_graph(),
        self_refine_graph(),
        tree_search_graph(),
        synthesis_pipeline_graph(),
        disagreement_pipeline_graph(),
    ]
