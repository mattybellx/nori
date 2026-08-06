"""Agent strategies. Each implements the ``Agent`` protocol from dse/agent.py.

Evidence-driven notes (see RESEARCH.md):
- ReAct: the control condition — a single generate→observe cycle, no retries.
- Best-of-N: N independent drafts, verifier-selected (Snell et al. 2024 baseline).
- Reflexion: retry with episodic reflections + external feedback (Shinn et al.).
- Self-Refine: verifier-gated refinement (Madaan et al.; gated per Huang et al.).
- TreeSearch: LATS-lite MCTS with per-step tests as the value signal.
- Escalating: confidence-based cheap→expensive routing (dynamic model routing).
- Adaptive: compute-optimal allocation across tiers (Snell et al. 2024).
"""

from .adaptive import AdaptiveAgent
from .best_of_n import BestOfNAgent
from .escalating import EscalatingAgent
from .escalating_per_step import EscalatingPerStepAgent
from .react import ReactAgent
from .reflexion import ReflexionAgent
from .self_refine import SelfRefineAgent
from .tree_search import TreeSearchAgent

__all__ = [
    "ReactAgent",
    "BestOfNAgent",
    "ReflexionAgent",
    "SelfRefineAgent",
    "TreeSearchAgent",
    "EscalatingAgent",
    "EscalatingPerStepAgent",
    "AdaptiveAgent",
]
