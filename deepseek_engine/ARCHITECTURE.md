# nori — Architecture

This document specifies the module design of the engine and traces every decision
to `RESEARCH.md`. It is the implementation contract.

---

## 1. Design goals

- **Verifier-first**: all test-time compute is allocated against a verifier
  (Snell et al. 2024). Nothing improves output unless a signal can measure it.
- **Adaptive compute**: per-prompt difficulty gates the compute budget
  (low / medium / high), instead of a uniform budget.
- **Deterministic by default**: a `MockLLM` with controllable per-task accuracy
  makes tests and benchmarks reproducible and strategy comparisons causal.
- **Modular and flagged**: every experimental technique sits behind a feature
  flag with an explicit hypothesis and rollback condition.
- **Measured, not claimed**: the benchmark harness reports success, latency,
  tokens, cost, and statistical significance for every comparison.

---

## 2. Module map

```text
deepseek_engine/
├── RESEARCH.md                  # evidence base (written first)
├── ARCHITECTURE.md              # this file
├── README.md                    # user guide
├── LIMITATIONS.md               # honest limitations
├── pyproject.toml
├── dse/
│   ├── __init__.py
│   ├── config.py                # EngineConfig, Experiment, FeatureFlags
│   ├── events.py                # StepEvent, RunResult (typed telemetry records)
│   ├── telemetry.py             # MetricsRecorder (latency, tokens, attempts)
│   ├── llm.py                   # LLM protocol + calibrated MockLLM
│   ├── providers.py             # OpenAI-compatible client (DeepSeek/Ollama/OpenAI/GitHub)
│   ├── memory.py                # rolling window + summary + episodic memory
│   ├── verifier.py              # Verifier protocol + implementations + aggregation
│   ├── router.py                # confidence-based model escalation
│   ├── environment.py           # Task + Environment protocol (feedback signals)
│   ├── feedback.py              # CompilerFeedbackLoop (build→test→lint→sec→perf)
│   ├── agent.py                 # Agent protocol + BaseAgent (ReAct loop)
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── react.py             # baseline: ReAct
│   │   ├── best_of_n.py         # N drafts, verifier-selected (Snell et al. baseline)
│   │   ├── reflexion.py         # episodic-memory retry (Shinn et al.)
│   │   ├── self_refine.py       # verifier-gated refinement (Madaan et al.)
│   │   ├── tree_search.py       # LATS-lite MCTS + sound pruning (Zhou et al.)
│   │   ├── escalating.py        # whole-task cheap→expensive routing (Router)
│   │   ├── escalating_per_step.py # escalate only failing steps
│   │   └── adaptive.py          # compute-optimal allocation (Snell et al.)
│   ├── orchestration.py         # multi-agent (MoA-lite), feature-flagged
│   └── benchmarks/
│       ├── __init__.py
│       ├── tasks.py             # synthetic task suite (controllable difficulty)
│       ├── harness.py           # paired benchmark + McNemar + bootstrap CI
│       └── run_benchmark.py     # CLI entry point
└── tests/                       # pytest suite (one file per module)
```

---

## 3. Core abstractions

### 3.1 `LLM` (protocol)

```python
class LLM(Protocol):
    def complete(self, messages, *, temperature, max_tokens) -> Completion: ...
```

`Completion` carries `text`, `tokens_in`, `tokens_out`, `latency_s`, `model`.

`MockLLM` implements a **calibrated toy reasoner** over a task suite: for each
task it has a configurable per-step accuracy `p`. It is deterministic under a
seed, so benchmarks compare strategies under controlled model quality.

### 3.2 `Task` and `Environment`

```python
class Task(Protocol):
    id: str
    difficulty: float          # 0..1, controls expected base success
    def evaluate(self, answer) -> TaskResult: ...   # exact, deterministic signal
```

`Environment` wraps a task and exposes the small **command surface** (SWE-agent
ACI lesson): `search`, `read`, `edit`, `run`. It returns parsed feedback, never
raw dumps.

### 3.3 `Verifier` (the linchpin)

```python
class Verifier(Protocol):
    def score(self, draft, task, context) -> Verdict: ...
```

Implementations:

- `ExactVerifier` — deterministic string/exact-match signal (ground truth).
- `TestVerifier` — runs the task's test suite via the environment, returns
  pass-rate and per-test results (AlphaCodium lesson).
- `LLMJudge` — an LLM scores the draft (value function in LATS terms).
- `SelfConsistencyVerifier` — majority vote over N samples (Wang et al. 2022).

`AggregateVerifier` weights and combines multiple sources. The aggregate score
drives: compute allocation, search value estimates, and routing decisions.

### 3.4 `Memory`

- Rolling window of recent steps (bounded tokens).
- Rolling summary (trigger-based).
- Episodic buffer of reflections (Reflexion).

### 3.5 `Router`

- `RouteDecision` picks model + effort tier.
- Confidence-based escalation: if the cheap model's verifier score is below
  threshold, escalate to the expensive model (bounded by cost budget).

---

## 4. Strategies

All strategies share the `Agent` protocol:

```python
class Agent(Protocol):
    def solve(self, task, budget) -> RunResult: ...
```

- **ReAct** (`react.py`) — baseline. Thought → action → observation loop, no
  retries, no search. The control condition.
- **Best-of-N** (`best_of_n.py`) — N independent cheap drafts; the verifier
  selects the best. The canonical test-time-compute *baseline* that smarter
  allocation must beat (Snell et al. 2024).
- **Reflexion** (`reflexion.py`) — ReAct loop plus: on verifier failure, write a
  reflection into episodic memory, retry up to `max_trials`. Reflection is
  **always derived from external verifier signals** (Huang et al. 2023 caveat).
- **Self-Refine** (`self_refine.py`) — generate → verify; only if the verifier
  fails does the model critique+refine, then re-verify. Gated refinement.
- **Escalating** (`escalating.py`) — cheap first; if verifier confidence is
  low, escalate once to the expensive tier via the `Router` (bounded).
- **TreeSearch** (`tree_search.py`) — LATS-lite MCTS:
  - *Selection*: UCT descent to the best expandable leaf; fully-evaluated
    subtrees are skipped so alternate branches are explored (true MCTS).
  - *Expansion*: LLM proposes B distinct next steps (duplicates skipped).
  - *Evaluation*: each proposed step checked against the per-step test.
  - *Sound pruning*: branches containing a wrong step are never expanded
    (deterministic per-step tests ⇒ they can never be part of the answer).
  - *Backprop*: only live (correct) edges contribute to path value.
  - *Reflection*: on terminal failure, convert failures into feedback and run
    one repair trial (LATS).
  Budget-capped by `max_search_nodes`.
- **Adaptive** (`adaptive.py`) — orchestrates the above by difficulty:
  - easy → single ReAct attempt;
  - medium → Self-Refine or Reflexion;
  - hard → TreeSearch.
  Difficulty probe = verifier score on the cheap first attempt (Snell et al.
  compute-optimal allocation).

---

## 5. Feature flags & experiments

Every experimental feature is declared in `config.py` as an `Experiment` with:

- `flag` — the feature-flag key (e.g. `multi_agent`, `adaptive_compute`).
- `hypothesis` — what we expect to gain.
- `expected_benefit` — the measurable metric.
- `possible_downside` — cost/latency/regression risk.
- `benchmark_plan` — which comparison validates it.
- `rollback_condition` — the measured threshold that removes the experiment.

Experiments that do not beat the baseline in the harness are removed (brief §Experimentation).

---

## 6. Benchmark harness

`benchmarks/harness.py`:

- `run_benchmark` — paired runs over a shared suite with fixed seeds.
- `run_multi_seed` — aggregates several seeds into one larger paired dataset
  (n × seeds). **Correctness invariant:** each seed's agents must be bound to
  that seed's LLM (sharing agents across seeds couples their RNG streams — a
  real bug found and fixed during review).
- Runs a set of strategies over a shared task suite with fixed seeds.
- Reports per-strategy: n, success rate, mean/median/p90 latency, mean tokens,
  estimated cost, and the **full metadata** (model config, seed, hardware).
- **Paired significance**: McNemar's test on the discordant pair counts for every
  strategy pair; 95% bootstrap CI on the difference of rates.
- Serializes the full record to `benchmarks/results/<timestamp>.json` and prints
  a human-readable table.

---

## 7. Honesty rules (from the brief)

- No "better"/"smarter" claims without harness numbers.
- Every reported number ships with hardware, model, seed, n, and CI.
- Known limitations live in `LIMITATIONS.md`, including: no trained PRM (value
  function is heuristic/LLM), MockLLM is a calibrated toy (not a frontier model),
  and multi-agent is experimental.
