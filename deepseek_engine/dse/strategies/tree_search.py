"""LATS-lite: MCTS over reasoning steps with per-step tests as the value signal.

Design (Zhou et al. 2023; RESEARCH.md §2):

- *Selection*: UCT descent to the best *expandable* leaf; fully-evaluated
  subtrees are skipped so alternate branches keep being explored (true MCTS,
  not a single-path beam).
- *Expansion*: propose B distinct candidate next steps via the LLM
  (``next_step`` mode); duplicates are skipped.
- *Evaluation*: each proposed step is checked against the task's per-step test
  (the environment's deterministic unit tests per step).
- *Backprop*: propagate step values up the tree.
- *Terminal reflection* (flag ``search_reflection``): if the best path is
  incomplete, convert per-step failures into ``FEEDBACK`` and run one repair
  trial (LATS reflection → informed re-attempt).

Soundness note: with deterministic per-step tests, a branch containing a wrong
step can never be part of the correct answer, so UCT concentrates budget on
verified-correct prefixes.
"""

from __future__ import annotations

import math
import re

from ..agent import BaseAgent, Budget, feedback_lines
from ..environment import Task
from ..events import RunResult


class _Node:
    __slots__ = ("step", "value", "parent", "children", "visits", "accum", "depth", "expanded")

    def __init__(self, step: str, value: float, parent: "_Node | None", depth: int) -> None:
        self.step = step
        self.value = value  # edge value: 1.0 correct step, 0.0 wrong step
        self.parent = parent
        self.children: list[_Node] = []
        self.visits = 0
        self.accum = 0.0
        self.depth = depth
        self.expanded = False  # True once an expansion attempt was made

    @property
    def avg(self) -> float:
        return self.accum / self.visits if self.visits else 0.0


class TreeSearchAgent(BaseAgent):
    name = "tree_search"
    EXPLORATION = 1.0
    BRANCH = 3

    def __init__(self, llm, verifier, config, models, env=None) -> None:
        super().__init__(llm, verifier, config, models, env)
        # With the ``llm_judge`` flag the per-step value comes from a noisy LLM
        # judge instead of the deterministic per-step test. Pruning is only
        # sound with a deterministic oracle (see ``_is_live``).
        self._noisy = self.config.enabled("llm_judge")

    def solve(self, task: Task, budget: Budget | None = None) -> RunResult:
        max_nodes = budget.max_search_nodes if budget else self.config.max_search_nodes
        run = RunResult(task_id=task.id, strategy=self.name, success=False, answer="")
        root = _Node(step="<root>", value=0.0, parent=None, depth=0)
        nodes = 1

        while nodes < max_nodes:
            leaf = self._select(root, task.num_steps)
            if leaf is None:
                break  # no expandable leaf remains: search space exhausted
            children = self._expand(task, leaf, run)
            if not children:
                continue  # leaf marked expanded; next _select moves elsewhere
            nodes += len(children)
            for child in children:
                if self._noisy or child.value == 1.0:
                    # deterministic: only live edges contribute (sound pruning);
                    # noisy: backprop every estimate (no pruning possible)
                    self._backprop(child, child.value)

        path = self._best_path(root, task.num_steps)
        steps = [n.step for n in path[1:]]
        answer = self._render(task, steps)
        verdict = self._verify(run, answer, task)

        if not verdict.passed and self.config.enabled("search_reflection"):
            failed = verdict.details.get("failed_steps", [])
            extra = feedback_lines(failed)
            if extra:
                repaired = self._call(
                    run, "reflect", self._system(task, extra),
                    "Apply the feedback and produce the corrected solution.",
                    model="cheap",
                )
                verdict = self._verify(run, repaired, task)
                answer = repaired

        return self._finalize(run, task, verdict, answer, attempts=1)

    # -- search internals ---------------------------------------------------
    def _is_live(self, node: _Node) -> bool:
        """A node is live if it lies on an all-correct prefix (or is the root).

        Sound pruning: with deterministic per-step tests, a branch containing a
        wrong step can never be part of the correct answer, so dead branches are
        never expanded. With a noisy evaluator (``llm_judge``) pruning is NOT
        sound — a step judged wrong may actually be right — so every branch
        stays live.
        """
        if self._noisy:
            return True
        return node.parent is None or node.value == 1.0

    def _is_expandable(self, node: _Node, num_steps: int) -> bool:
        return (
            node.depth < num_steps
            and not node.children
            and not node.expanded
            and self._is_live(node)
        )

    def _has_expandable(self, node: _Node, num_steps: int) -> bool:
        if self._is_expandable(node, num_steps):
            return True
        return any(self._has_expandable(c, num_steps) for c in node.children)

    def _select(self, root: _Node, num_steps: int) -> _Node | None:
        """UCT descent to the best *expandable* leaf.

        Fully-evaluated (terminal or dead) subtrees are skipped, so the search
        keeps exploring alternate branches instead of re-visiting the first
        success — this is what makes it MCTS rather than a single-path beam.
        Returns ``None`` when no expandable leaf remains.
        """
        node = root
        while node.children:
            best: _Node | None = None
            best_key = float("-inf")
            for child in node.children:
                if self._has_expandable(child, num_steps):
                    key = self._uct(node, child)
                    if key > best_key:
                        best, best_key = child, key
            if best is None:
                return None
            node = best
        return node if self._is_expandable(node, num_steps) else None

    @staticmethod
    def _uct(parent: _Node, child: _Node) -> float:
        return child.avg + TreeSearchAgent.EXPLORATION * math.sqrt(
            math.log(max(1, parent.visits) + 1) / (child.visits + 1)
        )

    def _expand(self, task: Task, node: _Node, run: RunResult) -> list[_Node]:
        node.expanded = True
        children: list[_Node] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = self.BRANCH * 2
        while len(children) < self.BRANCH and attempts < max_attempts:
            attempts += 1
            system = self._system(task, f"MODE: next_step\nPREFIX: {node.depth}")
            text = self._call(run, "action", system, "Propose the next step.", model="cheap")
            match = re.search(r"STEP\s+(\d+)\s*:\s*(\S+)", text)
            if not match:
                continue
            value = match.group(2)
            if value in seen:
                continue  # duplicate proposals add no information
            seen.add(value)
            if self._noisy:
                node_value = self._judge_step(task, node.depth, value, run)
                correct = node_value >= 0.5
            else:
                correct = task.check_step(node.depth, value)
                node_value = 1.0 if correct else 0.0
            child = _Node(
                step=value,
                value=node_value,
                parent=node,
                depth=node.depth + 1,
            )
            node.children.append(child)
            children.append(child)
        return children

    def _judge_step(self, task: Task, index: int, value: str, run: RunResult) -> float:
        """Noisy per-step value via the LLM judge (LATS-style value function)."""
        system = self._system(task, f"MODE: step_judge\nSTEP: {value}\nPREFIX: {index}")
        text = self._call(run, "judge", system, "Score this step 0..1.", model="cheap")
        match = re.search(r"SCORE:\s*([\d.]+)", text)
        return float(match.group(1)) if match else 0.0

    @staticmethod
    def _backprop(node: _Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.accum += value
            node = node.parent

    @staticmethod
    def _best_path(root: _Node, num_steps: int) -> list[_Node]:
        path = [root]
        node = root
        while node.children and len(path) <= num_steps:
            node = max(node.children, key=lambda c: c.avg)
            path.append(node)
        return path

    @staticmethod
    def _render(task: Task, steps: list[str]) -> str:
        lines = [f"STEP {i}: {v}" for i, v in enumerate(steps)]
        return "\n".join(lines) + f"\nANSWER: {''.join(steps)}"
