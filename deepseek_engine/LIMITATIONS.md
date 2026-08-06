# Limitations

Honest inventory of what this engine does **not** do, and where its
measurements must not be over-read. Per the brief: never exaggerate capability.

## Model fidelity

- **The benchmark uses a calibrated MockLLM, not a frontier model.** The mock
  encodes the "external feedback ⇒ targeted repair" mechanism (Huang et al.
  2023) as ground truth with a configurable probability. Results measure
  *orchestration mechanics* (does retrying with feedback help? does search pay
  for itself?) — they do **not** measure any real model's capability.
- **Real-LLM providers are wired; the real suite exists, but real-model
  results are not yet published.** `dse/providers.py` implements an
  OpenAI-compatible client (DeepSeek API, Ollama, OpenAI, GitHub Models),
  unit-tested against a local HTTP server, and `--suite real` provides 12
  natural-language tasks with deterministic checkers (no code execution).
  Running `--provider deepseek --suite real` with your token is the next step
  (real-model scores depend on your key/plan). Code-execution checking and a
  larger real suite are future work.
- **The noisy-value gap is now measured, not assumed.** With a realistic noisy
  per-step judge (`--flag llm_judge:true`), tree_search collapses from 0.924
  to 0.444 at ~8× cost (see BENCHMARKS §8). The search advantage holds only
  under a deterministic per-step oracle — an explicit, measured boundary on
  where search wins.
- **The blank-completion quirk hits answer GENERATION, not just the judge —
  root cause found + fixed (2026-08-06).** ``deepseek-v4-flash`` is a
  **reasoning model**: it emits hidden ``reasoning_content`` and only writes
  to visible ``content`` once reasoning finishes. For prompts that trigger
  long chain-of-thought (e.g. "design N tasks"), it exhausts ``max_tokens``
  on reasoning (`finish_reason: length`) and returns **empty content** —
  consistently, not intermittently. This caused 12 consecutive empty answers
  through ``dse.ask`` (3 runs × 4 strategies), billing tokens for nothing.
  **Fix (in `dse/providers.py`)**: (1) retry empty completions up to 3× at the
  single provider choke point (all paths: agents, chat, verifier); (2) when
  content is still empty, fall back to ``reasoning_content`` so the engine
  returns what the model actually generated instead of a blank. Validated
  live: the same `dse.ask` request went from 12 empty answers to non-empty,
  judge-graded 9.2/9.2/9.2/7.0. Caveat: the fallback returns reasoning-
  flavored text (verbose working-out), not a clean final answer.

## Environment & tasks

- The task suite is **synthetic and calibrated**, not a coding benchmark
  (no SWE-bench/HumanEval). The environment's `search`/`read`/`edit` commands
  are documented stubs; only `run_tests` is real. Repository-level agent
  behavior (file editing, builds) is therefore out of scope.
- **A code-execution suite now EXISTS (`--suite code`, dse/codetasks.py):**
  the model's generated code is run against hidden tests with a timeout, in a
  temp cwd, stdout redirected. Honest caveat: this is a **resource timeout +
  temp-dir guardrail, NOT a security sandbox** — fine for a local single-user
  tool, NOT for hosting without OS-level isolation (containers/seccomp).
- **Code suite result (2026-08-06): 4th consecutive 100% ceiling** — V4 Flash
  solved all 8 classic coding tasks single-shot (incl. DP min_coins,
  sliding-window longest substring). The strategies add zero measurable gain
  on a frontier model with classic tasks. **BUT a weak-model run (llama3.2:1b)
  finally showed the delta: single-shot 33–50%, repair recovers to 63–67%
  (hard-real reflexion 0.667 = 2× react; code self_refine 0.625).**
  **Aggregated across all four real suites this is now STATISTICALLY
  SIGNIFICANT AND REPLICATED: two independent runs gave identical rates;
  combined react vs reflexion, McNemar p = 0.000062 (n = 88), p = 0.0015
  (hard-only n = 64), b01 = 0 (never worse), 18/18 recoveries; self_refine
  p = 0.041 (all); best_of_n = react (p = 1.0); bootstrap 95% CI on the gain
  [+0.09, +0.32].** Scope: n modest, model weak 1b, tasks short.
  Still below strong single-shot (0.636 < 1.0) — "cheap beats expensive" is
  NOT yet made, only "repair recovers a weak model's failures, significantly,
  at ~2× tokens".
- The compiler feedback loop's lint and security stages are **heuristic
  placeholders** (real linters / SCA scanners are a stated TODO). The `test`
  stage is the only primary signal.

## Statistics

- Default n = 48 tasks; `--seeds N` aggregates to n × N for more power.
  A non-significant result means "not detected here", not "equal".
- The bootstrap CI and McNemar implementation are dependency-free. The
  chi-square CDF uses the exact erf closed form for 1 dof (the only case
  McNemar needs) and is validated at known quantiles; the series expansion for
  other k has not been cross-checked against scipy across its domain.
- **Multi-seed pairing invariant:** each seed's agents must be bound to that
  seed's LLM. A violation (sharing agents across seeds) silently couples RNG
  streams and was found and fixed during review; a regression test now guards
  it (`test_run_multi_seed_aggregates_and_pairs`).

## Experiments with negative results (must not be over-claimed)

- **adaptive_compute (score policy, v1)**: the original probe-score gating lost
  to plain search (0.722 vs 0.924, p = 0.0006). The re-tuned **difficulty
  policy (v2, now default)** fixes this: 0.910, no difference vs search
  (p = 0.823, n = 144). The score policy is retained as a benchmarked control.
- **escalating (whole-task routing)**: one expensive re-draft loses to cheap
  feedback-driven retries (0.306, ≈ best_of_n) — but **per-step escalation**
  (escalating_per_step, 0.708 @ 79 tokens) is the routing that works, and is
  the best strategy under noisy evaluation (BENCHMARKS §7–§8).
- **multi_agent (MoA-lite)**: uses verifier-selection as the aggregation policy
  (a documented proxy), not a true merging aggregator LLM. It does not justify
  its cost on this suite. Keep disabled by default.

## Scope

- Mixture-of-Experts (parameter-level routing) is out of scope — it is a model
  training/infra concern, not an orchestration concern.
- No trained classifiers for difficulty gating (the probe is one cheap LLM
  call); no online learning; no production serving concerns (retries, rate
  limits, streaming).
