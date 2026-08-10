# Architecture Discovery — Roadmap (adapted from the research spec)

> Source: `SELF-DISCOVERING INFERENCE ARCHITECTURE ENGINE` (the master spec).
> This file adapts it to **nori's existing codebase** — the spec's §49 rule:
> *inspect the repo, reuse existing infrastructure, don't duplicate, don't
> delete working systems.* Every phase is test-gated (§50): no phase continues
> while its tests are failing.

---

## The research question (spec §53)

> Can an AI system autonomously discover, construct, evaluate, optimize, and
> reuse inference architectures that are structurally novel relative to its
> initial human-designed library, while achieving better quality/reliability
> at lower or equal cost on independently evaluated unseen tasks?

Either outcome is valuable. We never claim it before evidence exists (§3).

## What already exists in nori (reuse, don't rebuild)

| Spec concept | Existing implementation | Reuse for |
|---|---|---|
| Baselines (§6) | `dse/strategies/*`: react, best_of_n, reflexion, self_refine, tree_search, escalating, adaptive, multi_agent | The initial architecture library |
| Objective verifiers (§10) | `dse/verifier.py`: Exact/Test/LLMJudge; `code` suite runs real tests | Promotion gates |
| Independent evaluation (§10/§11) | `--judge-model`, preference judge, `robust_score` (median-N) | Independent gate |
| Self-grading inflation (§11) | Measured: self p=0.011 vs independent p=0.32 (n=32) | The inflation baseline |
| Grounded reward (§12) | `guards.grounded_score` (substance-weighted) | Reward formulation |
| Never-worse guards (§19) | `guards.selection_guard` / `synthesis_guard` (property-tested) | Promotion gate |
| Statistics (§20) | `harness.py`: wilcoxon, sign_test, bootstrap CI, McNemar | Significance |
| Telemetry (§30) | `events.StepEvent` / `RunResult` | Provenance seed |

## What Phase 1 added (2026-08-11, DONE — 21 tests)

`dse/discovery/` — the executable-graph substrate every later phase needs:

- `graph.py` — `ArchGraph`/`ArchNode`/`ArchEdge` DAG representation,
  JSON-serializable, multi-entry (zero-in-degree nodes are implicit sources),
  `structural_similarity` + `novelty_against` (§9 — never claim absolute
  novelty, only distance to a reference set).
- `primitives.py` — composable primitive registry wrapping existing code:
  `generate`, `sample_n`, `verify`, `judge` (median-N robust), `score_items`,
  `select_best`, `selection_guard`, `synthesize`, `synthesis_guard`,
  `gather`, `identity`, `route_disagreement`, `strategy` (runs a whole
  existing Agent), `extract`. Each output carries tokens/latency/verdicts.
- `compiler.py` — validates before execution (§32): unknown primitives/nodes,
  cycles, unreachable nodes/exits, self-loops, `max_nodes`/`max_depth`
  budgets; produces a deterministic topological order.
- `executor.py` — streams data along edges, supports fan-in/fan-out and
  conditional routing (a node sets `__route` → only the matching port edge
  is followed, §7/§35), records full `ArchRunRecord` provenance (§17), never
  hides failures (§47.13).
- `registry.py` — architecture library with lifecycle states (§18:
  generated→…→promoted; terminal: rejected/retired/…), explicit promotion
  gate (§19/§31: cannot promote without passing required gates), genealogy
  (§29), fitness history (§17), novelty vs baselines (§9).
- `baselines.py` — the existing strategies expressed as baseline graphs
  (react, best_of_n, reflexion, self_refine, tree_search) PLUS two atomic
  compositions proving the substrate: `synthesis_pipeline` (the production
  never-worse pipeline as a graph) and `disagreement_pipeline`
  (architecture-specific compute via conditional routing).

Verified: **202 tests passing** (21 new in `tests/test_discovery.py`), all
deterministic on the seeded mock stack (per-(strategy, task) reseeding
preserved).

## The phased roadmap (spec §42 — adapt, don't boil the ocean)

| Phase | Scope | Reuses | Gate |
|---|---|---|---|
| **1 ✅** | Executable graphs, primitives, compiler, executor, registry, baselines | strategies, guards, verifiers, harness | 21 tests |
| 2 | Benchmark the baseline graphs head-to-head on the existing suites | `run_benchmark`, `run_never_worse` | baselines match the strategy results they wrap |
| 3 | Candidate generation: mutation/insertion/deletion/substitution/reordering/duplication | `ArchGraph` + registry | each mutation compiles; graph-level unit tests |
| 4 | Discovery loop v0: population + beam of candidates, fitness from `ArchRunRecord`, independent-judge gate, never-worse promotion gate | registry, `harness` stats | a discovered candidate can beat a baseline on held-out mock tasks |
| 5 | Evolutionary search: selection, crossover, novelty preservation, retirement | Phase 4 loop | improvement survives bootstrap CI + sign test vs baselines |
| 6 | Adaptive routing: learn task-class → architecture via the profiler + bandit/UCB | `ProblemProfiler` (new) + Phase 5 | routing beats any single fixed architecture on held-out tasks |
| 7 | Compute optimization: quality/reliability vs cost/tokens/latency Pareto (spec §13/§23) | `ArchRunRecord` costs | a Pareto-optimal point vs human baselines |
| 8 | Failure-driven + success-driven invention (spec §27/§28): generate architectures from observed failures, compress bloated successes | Phase 4 + failure classification | invented arch beats the failing baseline |
| 9 | Novel primitive discovery with compile/unit/adversarial/sandbox gates (spec §38/§39) | compiler + sandbox | new primitives pass their contract tests |
| 10 | Meta-optimization: discover better discovery strategies (spec §24) | everything | bounded, auditable, gated — no unrestricted self-modification |

## Non-negotiable rules (spec §47 — the project's honesty rules, applied to discovery)

1. Never manipulate evaluation or ground truth. 2. Never delete failed
experiments. 3. Never promote on one lucky run. 4. Never use self-evaluation
as sole evidence — the independent-judge gate is mandatory (we already proved
self-grading inflates significance: p=0.011 → p=0.32). 5. Never claim novelty
without the novelty reference set. 6. Never optimize the locked final test
set. 7. Never report an unverified improvement as fact. 8. Every promoted
architecture must have: benchmark improvement + independent verification +
no unacceptable regression + cost/latency within budget + reproducibility
(§19).

## The system must say "I don't know" (spec §41)

NO PROMOTION / INCONCLUSIVE / REQUIRES INVESTIGATION / NOVEL FAILURE are
first-class outcomes. Not every experiment must succeed.

## How to run Phase 1

```python
from dse.discovery import (ArchitectureRegistry, default_baselines,
                           compile_graph, ArchExecutor)
from dse.discovery.primitives import ExecutionContext
from dse.factory import build_stack

config, catalog, models, llm, env, verifier, agents, budget = build_stack(seed=0)
ctx = ExecutionContext(llm=llm, verifier=verifier, models=models, config=config,
                       agents={a.name: a for a in agents}, budget=budget)
reg = ArchitectureRegistry("nori")
for g in default_baselines():
    reg.register(g)
task = list(catalog.values())[0]
rec = ArchExecutor().run(compile_graph(reg.get("synthesis_pipeline").graph), task, context=ctx)
print(rec.answer, rec.tokens_total, rec.success)
```

## Honest status

Phase 1 proves the *representation and execution* substrate works and reuses
existing infrastructure. It does NOT yet claim the research hypothesis — no
architecture has been automatically discovered yet. That begins in Phase 3–4,
and nothing is promoted until the independent-judge + never-worse gates pass.
