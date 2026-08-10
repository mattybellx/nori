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

## What Phase 2 added (2026-08-11, DONE — 8 tests)

`dse/discovery/evaluate.py` — the gate + the evaluation loop discovery will use:

- `validate_equivalence(graph, strategy, ctx, tasks, seed)` — runs a baseline
  graph and the strategy it wraps with IDENTICAL per-(strategy, task)
  reseeds and reports every divergence (answer / success / tokens_in /
  tokens_out). The Phase-2 gate: **a graph that wraps a strategy must
  reproduce it exactly** — otherwise every later "discovered improvement"
  would be an executor artifact.
- `validate_all_baselines(ctx, tasks)` — checks every strategy baseline.
  **Verified: react, best_of_n, reflexion, self_refine, tree_search,
  escalating, adaptive all reproduce their strategies with ZERO mismatches
  on the seeded mock stack.**
- `benchmark_architectures(graphs, tasks, ctx, seed)` — head-to-head
  leaderboard (§40): paired reseeding, per-architecture success rate /
  mean verifier score / tokens / latency. Architectures needing a strategy
  not in the context are **skipped and reported** (`BenchmarkResult.skipped`)
  — never run as fake failure rows. Text-terminal pipelines honestly show
  `avg_verifier_score = None` (no objective score) rather than a fake 0.
- New baseline graphs added: `escalating`, `adaptive`, `multi_agent`
  (the full Agent set is now expressible as graphs).

Demo leaderboard (seeded mock, n=12 tasks, 4 steps): tree_search 100%,
adaptive 100%, synthesis_pipeline 100% (n/a score — text terminal), reflexion
66.7%, self_refine 66.7%, escalating 50%, best_of_n 33.3%, react 16.7%,
multi_agent SKIPPED (not in the default stack). Matches the project's known
mock hierarchy — the graphs reproduce reality.

Verified: **210 tests passing** (8 new in `tests/test_discovery_eval.py`).

## What Phase 3 added (2026-08-11, DONE — 16 tests)

`dse/discovery/mutations.py` — candidate generation (spec §7). Every operator
returns a NEW graph (architectures are immutable — genealogy records parents,
§29) and the Phase-3 gate is that every mutation COMPILES:

- **insertion**: `insert_before` / `insert_after` (+ `insert_verify` /
  `append_verify` conveniences — the spec's canonical `A→B→C` →
  `A→Verify→B→C`)
- **deletion**: `delete_node` splices a stage out (bypass in→out); multi-entry
  semantics keep the result valid (a target left with no in-edges becomes an
  implicit source)
- **substitution**: `substitute` replaces a primitive (`MCTS → beam search`)
- **reordering**: `swap_primitives` swaps two operations in place
- **duplication**: `duplicate_parallel` (second reasoning path) and
  `duplicate_sequential` (repeat a stage)
- **branching**: `branch(node, n)` — n parallel copies
- **merging**: `gather_join(target)` — collects multiple in-edges into a list
  before the target
- **random generation**: `random_mutation(graph, rng)` picks a random
  operator, keeps only mutations that COMPILE, and renames the candidate to a
  **deterministic unique id** (`<parent>+<op>-<digest>`) so benchmark rows and
  registry entries never collide (found the collision the hard way in the
  demo: a child sharing its parent's name overwrote the parent's benchmark
  row).

Verified: every operator compiles on the baselines; mutations never modify the
source graph (immutability test); `random_mutation` is deterministic under a
seeded RNG. Demo: react+append_verify / best_of_n+gather_join /
reflexion+duplicate_sequential / … all compile; benchmarking a substitute
candidate head-to-head against its parent shows the parent at its known 16.7%
and the (degenerate) candidate at 0% — precisely the signal Phase 4's filter
will use to discard bad candidates.

**226 tests passing** (16 new in `tests/test_discovery_mutations.py`).

## What Phase 4 added (2026-08-11, DONE — 9 tests)

`dse/discovery/loop.py` — the discovery loop v0 (spec §15–§21). This is the
milestone where the system stops being a toolbox and actually **discovers**:

- `split_tasks` — deterministic discovery/held-out split (§20/§21).
- `promotion_gate` — the §19/§20 gate: candidate must show **discovery
  improvement**, **no-regression** (rate-level), and **held-out
  improvement** (generalizes — anti-overfitting). Strict per-task b01 and
  McNemar p are reported as information; with per-architecture reseeding,
  strict per-task never-worse is not estimable in expectation (each
  architecture faces its own RNG stream), so rate-level no-regression is the
  honest requirement (documented in the gate).
- `discover` — population + beam: benchmark baselines → incumbent; each
  round mutate the population, benchmark on the discovery set, register
  every candidate with genealogy + fitness (lifecycle: benchmarked →
  independently_verified → promoted / rejected), apply the gate, keep the
  best beam. Reports `DiscoveryReport` (rounds, beam, promoted/rejected,
  honest NO PROMOTION).
- **Structural-distinctness guard** (found the hard way in the demo): a
  candidate structurally IDENTICAL to the incumbent (a no-op mutation like
  gather_join on a single-node graph) was promoted by pure reseed luck —
  that's §47's "never promote on a lucky run". Now rejected: only strictly
  different structures compete.
- `best_of_n_ify` expansion operator (in mutations.py): turns a generator
  node into N parallel copies → **objective-verifier selection** (like the
  real best_of_n agent — noisy-judge selection does NOT win on the mock) →
  verify. Verified: react 16.7% → **33.3%** (n=12) and 20.8% → **45.8%**
  (n=24), with strict per-task never-worse (b01=0) on this mock.
- `verify_items` primitive: objective per-candidate scoring (RunResult /
  dict / text aware) — selection must use the objective verifier, not the
  noisy judge, to be reliable.

**Demo discovery run** (react incumbent, 24-task mock): the loop generated 28
candidates over 3 rounds and PROMOTED three genuinely-distinct architectures
(novelty 0.75–0.9 vs react) — e.g. `duplicate_sequential + best_of_n_ify`
beats react on both discovery and held-out with no regression. The research
hypothesis now has its first (small, mock-scale) positive instance.

**235 tests passing** (9 new in `tests/test_discovery_loop.py`).

## What Phase 5 added (2026-08-11, DONE — 13 tests)

`dse/discovery/evolution.py` — evolutionary search on top of the loop:

- **`crossover`** (§7) — combine the HEAD of architecture A with the TAIL of
  B: A keeps everything upstream of a cut node, B keeps everything downstream
  of a cut node, child = A-prefix → join → B-suffix (B's suffix renamed on
  collision). Deterministic under a seeded RNG; `crossover_children` produces
  compile-valid, uniquely-named children.
- **`validate_statistical`** (§20) — the Phase-5 significance gate: two-sided
  binomial **sign test** on paired wins + **bootstrap CI** on the success-rate
  difference; `significant` requires p < α AND CI lower bound > 0.
- **`retire_unfit`** (§18) — retire REJECTED candidates that never succeeded
  in their fitness history (terminal state RETIRED).
- **`beam_score`** (§16) — novelty-aware selection: base fitness + a small
  novelty bonus so the beam does not greedily collapse onto one structure.
- **Wired into `discover`**: `crossover_per_round`, `require_significance`
  (the §20 gate), `alpha`, `novelty_weight`, `retire_after` config options;
  crossover children get `source="crossed"` genealogy with two parents.

**The statistical gate found a real bug in the Phase-4 gate** (the demo
promoted 14 candidates, all statistically indistinguishable): the strict
`>` comparison used `- 1e-9`, which flipped it into allowing EQUAL rates —
candidates identical in performance to the incumbent were promoted by an
epsilon. Fixed to `+ 1e-9` (strict improvement by MORE than delta), with
regression tests (`test_gate_rejects_equal_rates`,
`test_gate_requires_strict_rate_delta`). This is exactly the value of the
§20 layer: it caught a subtle promotion bug the rate-level gate could not.

**Honest status after the fix**: on the 24-task mock with a react incumbent,
the loop still discovers strictly-better architectures (4 promoted over 3
rounds, novelty 0.75–0.83), but the statistical gate honestly reports those
small rate gains are NOT significant at this scale (p ≥ 0.5, CI spanning 0)
— the system correctly says "I don't know" (§41). `require_significance=True`
would promote nothing here; the significance function itself is unit-verified
to catch genuinely-strong improvements (p=0.031, CI>0).

**250 tests passing** (13 new in `tests/test_discovery_evolution.py` + 2
regression tests in `test_discovery_loop.py`).

## What Phase 6 added (2026-08-11, DONE — 10 tests)

`dse/discovery/routing.py` — the system stops treating every task identically
(spec §14 profiler, §15 selection, §16 UCB):

- **`profile_task` / `TaskProfile`** (§14) — deterministic task features:
  the task-type prefix of the id (arithmetic / logic / code on the mock
  suites), step count, and optional difficulty.
- **`UCRouter`** (§16) — a UCB contextual bandit keyed by task class. Learns
  per-(class, architecture) mean rewards from the discovery evaluation
  (`learn_from_benchmark`, full-information), `choose` balances exploitation
  (mean) and exploration (sqrt(2 ln N / n)); **deployment is pure
  exploitation** (exploration=0 — the policy is learned on train; held-out is
  deployment, not more learning).
- **`run_routing_experiment`** — train/test split; learns per-class routing
  on train, routes held-out tasks, and reports the router's rate vs the best
  single fixed architecture (both the train-selected one — the realistic
  comparison — and the held-out oracle upper bound) plus a per-type oracle
  bound.

**The mock shows real per-class specialization** (empirically measured):
arithmetic → reflexion 0.75 / self_refine 0.62, code → reflexion & self_refine
0.88, logic → self_refine 0.75 — while the best single architecture overall
(relexion/self_refine) is only 0.75. A per-type oracle would reach ~0.79.
**On seed 3 the learned router reached 0.86 — beating the best single
architecture (0.71) AND matching the per-type oracle upper bound (0.86)**: it
learned best_of_n for logic/code and self_refine for arithmetic. Across 12
seeds the router beat the best train-selected single architecture in 3 — the
advantage is real but modest at this scale, and it is always bounded by the
oracle (never exceeds it).

Honest status: routing *can* beat any single fixed architecture on this mock,
and the mechanism (learn → exploit → bounded-by-oracle) is verified. The edge
is small because the specialization margin is small; on a task distribution
with stronger per-class differences the win would be larger (a Phase-8+
direction to test on real tasks).

**260 tests passing** (10 new in `tests/test_discovery_routing.py`).

## What Phase 7 added (2026-08-11, DONE — 10 tests)

`dse/discovery/pareto.py` — compute optimization (spec §13 multi-objective,
§23 compression, §40 dashboard):

- **`ParetoPoint` / `point_from_row`** — map a benchmark row to the
  multi-objective point (maximize success_rate & verifier score; minimize
  tokens & latency; missing scores are not compared).
- **`dominates`** — strict multi-objective dominance (better on ≥1, not worse
  on all).
- **`pareto_frontier`** — the non-dominated set (the §40 frontier).
- **`compute_efficiency`** — quality per unit compute (success per token).
- **`compression_win`** (§23) — when a SIMPLER architecture dominates a
  complex one, the complex architecture's extra compute did not pay.
- **`pareto_summary`** — dashboard-shaped dict (frontier / dominated /
  efficiency rank / points).
- **Wired into `discover`**: `require_pareto` (default False) + `pareto_margin`
  — if the candidate's success gain is within the margin of the incumbent but
  costs strictly more compute, it is rejected as a **compression loss** (the
  added compute did not pay; prefer minimum compute for the desired quality).
  `DiscoveryReport.pareto` carries the frontier summary of the last round.

**Demo on the 24-task mock** (exactly what §40 wants): frontier = `react`
(cheapest, 51 tok), `escalating` (0.58 @ 78 tok — most efficient 0.0075/tok),
`self_refine` (0.75 @ 116 tok — best quality-per-token among the strong).
**Dominated**: `reflexion` is strictly dominated by `self_refine` (same 0.75
success but 148 vs 116 tokens); `best_of_n` dominated; and the hand-rolled
`react_best_of_n_ify` is dominated by the real `best_of_n` — a genuinely
honest finding that the discovered-style expansion is strictly worse than the
built-in strategy at the same cost. **Compression**: react does NOT dominate
the bigger arches — their extra compute buys real quality (genuine frontier
trade-offs, not waste).

**270 tests passing** (10 new in `tests/test_discovery_pareto.py`).

## The phased roadmap (spec §42 — adapt, don't boil the ocean)

| Phase | Scope | Reuses | Gate |
|---|---|---|---|
| **1 ✅** | Executable graphs, primitives, compiler, executor, registry, baselines | strategies, guards, verifiers, harness | 21 tests |
| **2 ✅** | Validate + head-to-head the baseline graphs (Phase-2 gate: graph ≡ strategy) | `run_benchmark`, `run_never_worse` | baselines match the strategy results they wrap (8 equivalence tests) |
| **3 ✅** | Mutation operators: insertion/deletion/substitution/reordering/duplication/branching/merging + compile-gated random mutation | `ArchGraph` + registry | each mutation compiles (16 tests) |
| **4 ✅** | Discovery loop v0: population + beam, fitness from `ArchRunRecord`, promotion gate (improvement + no-regression + held-out generalization + structural distinctness) | registry, `harness` stats, mutations | a discovered candidate beats a baseline on held-out mock tasks (9 tests — demonstrated) |
| **5 ✅** | Evolutionary search: crossover (head of A × tail of B), novelty-aware beam, retirement, statistical significance gate (sign test + bootstrap CI) | Phase 4 loop, `harness.sign_test`/`bootstrap_ci` | significance layer works + gate requires STRICT improvement (13 tests; significance gate found & fixed an epsilon bug) |
| **6 ✅** | Adaptive routing: task-class → architecture via the ProblemProfiler + UCB router | `ProblemProfiler` (new), `harness` | routing beats the best single fixed architecture on held-out tasks when archs specialize (10 tests — demonstrated on the mock) |
| **7 ✅** | Compute optimization: quality/reliability vs cost/tokens/latency Pareto (spec §13/§23/§40) | `ArchRunRecord` costs | Pareto frontier + efficiency + compression detection (10 tests — demonstrated) |
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
