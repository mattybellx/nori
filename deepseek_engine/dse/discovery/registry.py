"""Architecture library & registry (Phase 1, spec §18/§29).

Every architecture has a lifecycle state and full lineage. Promotion is a
deliberate, evidence-gated step — the registry does NOT auto-promote (§19/§31):
promotion must be called explicitly by whoever validated the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .executor import ArchRunRecord
from .graph import ArchGraph, novelty_against

# lifecycle states (spec §18)
GENERATED = "generated"
COMPILED = "compiled"
SANITY_CHECKED = "sanity_checked"
BENCHMARKED = "benchmarked"
INDEPENDENTLY_VERIFIED = "independently_verified"
STATISTICALLY_VALIDATED = "statistically_validated"
CANDIDATE = "candidate"
PROMOTED = "promoted"
REJECTED = "rejected"
FAILED = "failed"
REGRESSED = "regressed"
SUPERSEDED = "superseded"
RETIRED = "retired"

LIFECYCLE = [
    GENERATED, COMPILED, SANITY_CHECKED, BENCHMARKED, INDEPENDENTLY_VERIFIED,
    STATISTICALLY_VALIDATED, CANDIDATE, PROMOTED,
]
TERMINAL = {REJECTED, FAILED, REGRESSED, SUPERSEDED, RETIRED}


@dataclass
class ArchRecord:
    """One entry in the library (spec §18/§29)."""

    arch_id: str
    graph: ArchGraph
    state: str = GENERATED
    source: str = "baseline"          # baseline | mutated | crossed | generated
    parent_ids: list[str] = field(default_factory=list)
    fitness: list[dict] = field(default_factory=list)   # ArchRunRecord.to_dict()
    promoted_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "arch_id": self.arch_id,
            "state": self.state,
            "source": self.source,
            "parents": list(self.parent_ids),
            "n_fitness_records": len(self.fitness),
            "promoted_at": self.promoted_at,
            "graph": self.graph.to_dict(),
        }


class ArchitectureRegistry:
    """The evolving architecture library: baselines + discovered candidates
    with genealogy, lifecycle and fitness history."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._records: dict[str, ArchRecord] = {}

    # -- library ------------------------------------------------------------
    def register(self, graph: ArchGraph, source: str = "baseline",
                 parent_ids: list[str] | None = None, state: str = GENERATED) -> str:
        if graph.name in self._records:
            raise ValueError(f"architecture {graph.name!r} already registered")
        self._records[graph.name] = ArchRecord(
            arch_id=graph.name, graph=graph, state=state, source=source,
            parent_ids=list(parent_ids or []))
        return graph.name

    def get(self, arch_id: str) -> ArchRecord:
        return self._records[arch_id]

    def __contains__(self, arch_id: str) -> bool:
        return arch_id in self._records

    def all(self) -> list[ArchRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def baseline_ids(self) -> list[str]:
        return [r.arch_id for r in self.all() if r.source == "baseline"]

    # -- lifecycle ----------------------------------------------------------
    def set_state(self, arch_id: str, state: str) -> None:
        rec = self._records[arch_id]
        rec.state = state
        if state == PROMOTED:
            import datetime
            rec.promoted_at = datetime.datetime.now().isoformat(timespec="seconds")

    def promote(self, arch_id: str, require: list[str] | None = None) -> None:
        """Explicit promotion gate (§19/§31): only from a validated state and
        only when the required gates (default: validated states) are present."""
        rec = self._records[arch_id]
        required = require or [BENCHMARKED, INDEPENDENTLY_VERIFIED]
        missing = [g for g in required if g not in LIFECYCLE[:LIFECYCLE.index(rec.state) + 1]]
        if missing:
            raise ValueError(
                f"{arch_id!r} cannot be promoted from state {rec.state!r}: "
                f"missing gates {missing}")
        rec.state = PROMOTED
        import datetime
        rec.promoted_at = datetime.datetime.now().isoformat(timespec="seconds")

    # -- fitness history ----------------------------------------------------
    def record_fitness(self, arch_id: str, run: ArchRunRecord) -> None:
        self._records[arch_id].fitness.append(run.to_dict())

    # -- genealogy (§29) ----------------------------------------------------
    def genealogy(self, arch_id: str) -> list[dict]:
        """Lineage chain, oldest -> newest, walking ``parents`` recursively."""
        chain: list[dict] = []
        seen: set[str] = set()

        def walk(aid: str, depth: int = 0) -> None:
            if aid in seen or depth > 20:
                return
            seen.add(aid)
            rec = self._records.get(aid)
            if rec is None:
                chain.append({"arch_id": aid, "source": "unknown", "parents": []})
                return
            for p in rec.parent_ids:
                walk(p, depth + 1)
            chain.append({"arch_id": aid, "source": rec.source,
                          "parents": list(rec.parent_ids), "state": rec.state})
        walk(arch_id)
        return chain

    # -- novelty (§9) -------------------------------------------------------
    def novelty(self, graph: ArchGraph, reference_ids: list[str] | None = None) -> dict:
        refs = [self._records[r].graph for r in (reference_ids or self.baseline_ids())
                if r in self._records]
        return novelty_against(graph, refs)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name,
                "architectures": [r.to_dict() for r in self.all()]}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"ArchitectureRegistry({self.name!r}, {len(self._records)} "
                f"architectures, {len(self.baseline_ids())} baselines)")
