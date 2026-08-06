"""Context management and memory (window + summary + episodic reflections).

Design (RESEARCH.md §8): rolling window with bounded tokens, a trigger-based
rolling summary to keep context bounded, and an episodic buffer of reflections
(Reflexion, Shinn et al. 2023). The summary is approximate by design — with a
real model the summary is an LLM-produced digest; the MockLLM does not read it,
which is documented rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import StepEvent


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Memory:
    window_tokens: int = 4096
    summary_every: int = 8
    window: list[StepEvent] = field(default_factory=list)
    summary: str = ""
    reflections: list[str] = field(default_factory=list)
    max_reflections: int = 8

    def add_step(self, event: StepEvent) -> None:
        self.window.append(event)
        self._trim()
        if len(self.window) % self.summary_every == 0:
            self._fold_summary()

    def add_reflection(self, text: str) -> None:
        self.reflections.append(text)
        if len(self.reflections) > self.max_reflections:
            self.reflections.pop(0)

    def _trim(self) -> None:
        used = sum(_approx_tokens(e.content) for e in self.window)
        while self.window and used > self.window_tokens:
            dropped = self.window.pop(0)
            used -= _approx_tokens(dropped.content)

    def _fold_summary(self) -> None:
        recent = "\n".join(f"[{e.kind}] {e.content}" for e in self.window[-self.summary_every :])
        self.summary = f"(summary of last {len(self.window)} steps) {recent[-400:]}"

    def render_messages(self, system: str, user: str) -> list[dict]:
        """Build the message list the agent sends: system + summary + window +
        reflections + current user prompt."""
        parts = [system]
        if self.summary:
            parts.append(f"CONTEXT SUMMARY: {self.summary}")
        if self.reflections:
            parts.append("EPISODIC MEMORY (reflections):")
            parts.extend(f"- {r}" for r in self.reflections)
        return [
            {"role": "system", "content": "\n".join(parts)},
            {"role": "user", "content": user},
        ]

    def __len__(self) -> int:
        return len(self.window)
