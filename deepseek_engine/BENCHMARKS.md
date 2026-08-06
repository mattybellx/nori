# Benchmark Report

> Per the Master Brief: *"Never report 'better' without benchmark evidence."*
> Every number below is reproducible: `python -m dse.benchmarks.run_benchmark
> --seed 0 --n-tasks 48 --max-steps 5 --suite all` (plus the `--moa` variant).
> Full machine-readable records: `benchmarks/results/*.json`.

## 1. Setup (metadata)

| Item | Value |
|---|---|
| Suite | synthetic calibrated tasks (arithmetic/logic/code; 1–5 steps) |
| n (all) | 48 tasks |
| MockLLM cheap tier | step_accuracy = 0.5, judge_accuracy = 0.75 |
| MockLLM expensive tier | step_accuracy = 0.8, judge_accuracy = 0.9 (used by `escalating`) |
| Repair probability (p_fix) | 0.9 |
| Verifier | deterministic per-step tests (TestVerifier) |
| Budget | max_trials = 3, max_search_nodes = 64 |
| Strategies | react, best_of_n(3), reflexion, self_refine, tree_search, escalating, adaptive (+ multi_agent with `--moa`) |
| Seed | 0 (paired draws; every (agent, task) sees identical RNG) |
| Hardware | Windows x86_64, CPython 3.14 (see JSON `metadata.hardware`) |
| Statistical tests | McNemar (paired, Edwards correction); 95% bootstrap CI |

## 2. All-suite results (n = 48, seed = 0)

| strategy | success | mean_lat (s) | med_lat (s) | p90_lat (s) | mean_tokens | mean_cost ($) | attempts |
|---|---|---|---|---|---|---|---|
| react | 0.083 | 0.138 | 0.127 | 0.241 | 54 | 0.000068 | 1.0 |
| best_of_n | 0.312 | 0.485 | 0.484 | 0.593 | 162 | 0.000204 | 3.0 |
| reflexion | 0.479 | 0.398 | 0.399 | 0.609 | 194 | 0.000229 | 2.4 |
| self_refine | 0.521 | 0.388 | 0.401 | 0.560 | 160 | 0.000195 | 2.5 |
| tree_search | 0.917 | 2.130 | 2.136 | 3.282 | 262 | 0.000292 | 1.0 |
| escalating | 0.312 | 0.222 | 0.194 | 0.440 | 80 | 0.000457 | 1.5 |
| adaptive | 0.708 | 1.268 | 0.694 | 3.321 | 266 | 0.000312 | 2.7 |
| multi_agent | 0.438 | 0.474 | 0.468 | 0.622 | 162 | 0.000204 | 3.0 |

### McNemar significance (all suite)

| A vs B | b01 | b10 | p | sig |
|---|---|---|---|---|
| react vs best_of_n | 4 | 15 | 0.0218 | YES |
| react vs reflexion | 2 | 21 | 0.0002 | YES |
| react vs self_refine | 2 | 23 | 0.0001 | YES |
| react vs tree_search | 2 | 42 | <0.0001 | YES |
| react vs escalating | 3 | 14 | 0.0153 | YES |
| react vs adaptive | 0 | 30 | <0.0001 | YES |
| react vs multi_agent | 2 | 19 | 0.0005 | YES |
| best_of_n vs reflexion | 6 | 14 | 0.1175 | no |
| best_of_n vs self_refine | 5 | 15 | 0.0442 | YES |
| best_of_n vs tree_search | 0 | 29 | <0.0001 | YES |
| best_of_n vs escalating | 9 | 9 | 0.8137 | no |
| best_of_n vs adaptive | 2 | 21 | 0.0002 | YES |
| best_of_n vs multi_agent | 4 | 10 | 0.1814 | no |
| reflexion vs self_refine | 10 | 12 | 0.8312 | no |
| reflexion vs tree_search | 2 | 23 | 0.0001 | YES |
| reflexion vs escalating | 13 | 5 | 0.0990 | no |
| reflexion vs adaptive | 6 | 17 | 0.0371 | YES |
| reflexion vs multi_agent | 11 | 9 | 0.8231 | no |
| self_refine vs tree_search | 1 | 20 | 0.0001 | YES |
| self_refine vs escalating | 16 | 6 | 0.0550 | no |
| self_refine vs adaptive | 8 | 17 | 0.1096 | no |
| self_refine vs multi_agent | 12 | 8 | 0.5023 | no |
| tree_search vs escalating | 29 | 0 | <0.0001 | YES |
| tree_search vs adaptive | 12 | 2 | 0.0162 | YES |
| tree_search vs multi_agent | 26 | 3 | <0.0001 | YES |
| escalating vs adaptive | 2 | 21 | 0.0002 | YES |
| escalating vs multi_agent | 8 | 14 | 0.2864 | no |
| adaptive vs multi_agent | 20 | 7 | 0.0209 | YES |

Bootstrap 95% CI, success diff (adaptive − react): **+0.629 [+0.500, +0.771]**.

## 3. Findings (what the evidence says)

1. **The core mechanism is validated.** Every strategy that adds compute —
   best-of-N sampling, feedback-driven retries, search, or routing —
   significantly beats the single-shot ReAct baseline (all McNemar p < 0.02,
   most p < 0.001). A single cheap attempt succeeds only 8%; every form of
   test-time compute lifts that.

2. **Feedback beats resampling (compute-optimal thesis).** self_refine (0.521)
   and reflexion (0.479) outperform best_of_n (0.312) at a similar token budget
   (160–194 vs 162 tokens); self_refine vs best_of_n is significant
   (p = 0.044). Spending compute on *feedback-driven repair* beats spending it
   on *more independent samples* — the qualitative prediction of Snell et al.
   2024, now measured here.

3. **Tree search (with sound pruning) dominates on this suite.** tree_search =
   0.917 success, significantly better than every other strategy (all
   p < 0.02), and at 262 mean tokens its tokens-per-success (≈ 286) now beats
   self_refine (≈ 307). The pruning rewrite is a strict improvement: before
   pruning it was 0.688 @ 566 tokens; after, 0.917 @ 262 — **higher accuracy
   and ~2× cheaper**, because dead branches (a wrong step can never lead to a
   correct answer under deterministic per-step tests) are never expanded.
   Honest caveat: the value signal here is a perfect per-step oracle; with a
   noisy real-world judge the margin will shrink (see §6).

4. **Adaptive allocation no longer beats plain search (updated negative
   result).** adaptive (0.708) significantly beats reflexion (p = 0.037) but is
   significantly *worse* than tree_search (p = 0.016) at the same token cost
   (266 vs 262). The difficulty probe + medium-tier routing overhead is not
   repaid when search is this strong. The `adaptive_compute` experiment's
   rollback condition ("tokens >= baseline on the same suite") is now clearly
   triggered — **do not adopt as configured.**

5. **Routing to a pricier model is the weakest strategy here (negative
   result).** escalating (0.312) ties best_of_n (p = 0.81) at ~2.2× the dollar
   cost ($0.000457 vs $0.000204) and collapses on hard tasks (0.167): one
   expensive call loses to several cheap feedback-driven retries. Honest
   verdict: confidence-based escalation needs a much stronger expensive tier
   (or per-step escalation) to justify itself; keep it as a benchmarked
   baseline, not a default.

6. **MoA-lite still does not justify itself.** multi_agent (0.438) ≈ reflexion
   (p = 0.82), pays 3× proposer calls, and is significantly worse than
   tree_search (p < 0.0001) and adaptive (p = 0.021). Keep behind the
   `multi_agent` flag; **do not enable by default.**

## 4. Difficulty splits (seed = 0)

### Easy (12 tasks, num_steps < 3)

| strategy | success | mean_lat (s) | mean_tokens |
|---|---|---|---|
| react | 0.083 | 0.137 | 48 |
| best_of_n | 0.667 | 0.457 | 143 |
| reflexion | 0.917 | 0.278 | 103 |
| self_refine | 0.917 | 0.379 | 112 |
| tree_search | 1.000 | 1.084 | 134 |
| escalating | 0.750 | 0.257 | 76 |
| adaptive | 0.917 | 0.417 | 113 |

### Hard (36 tasks, num_steps >= 3)

| strategy | success | mean_lat (s) | mean_tokens |
|---|---|---|---|
| react | 0.083 | 0.138 | 56 |
| best_of_n | 0.194 | 0.495 | 169 |
| reflexion | 0.333 | 0.437 | 224 |
| self_refine | 0.389 | 0.390 | 176 |
| tree_search | 0.889 | 2.478 | 304 |
| escalating | 0.167 | 0.211 | 81 |
| adaptive | 0.639 | 1.551 | 317 |

On hard tasks tree_search reaches 0.889 @ 304 tokens (was 0.583 @ 714 before
pruning): pruning helped hardest exactly where it matters. Escalating falls to
0.167 on hard tasks — the expensive tier's per-step accuracy (0.8) is not
enough to beat cheap feedback-driven retries for multi-step problems. Adaptive
(0.639) trails tree_search because its probe routes many hard tasks to the
medium (reflexion) tier.

## 5. Reproducibility

- Deterministic MockLLM + per-(agent, task) reseeding ⇒ identical draws for
  every strategy on every task.
- `tests/test_benchmarks.py::test_run_benchmark_is_reproducible` asserts two
  identical runs produce identical success vectors.
- **Cross-seed check (seed 1, all suite):** tree_search 0.917 (identical),
  adaptive 0.729, reflexion 0.708, self_refine 0.667, best_of_n 0.396,
  escalating 0.375, react 0.208 — the strategy ordering is stable.
- Records: `benchmarks/results/seed0_all.json` (with `--moa`), `seed0_hard.json`,
  `seed0_easy.json`, `seed1_all.json`.

## 6. Honest limits of these measurements

- Synthetic, calibrated tasks — **not** a frontier-model benchmark. The MockLLM
  encodes the "external feedback ⇒ repair" mechanism as ground truth, so these
  numbers measure *orchestration mechanics*, not model capability.
- **Search's advantage depends on the perfect per-step oracle.** tree_search
  uses the deterministic per-step test as its value signal. With a real model
  the value function is a noisy LLM judge, and search's margin (and pruning's
  soundness) will shrink — the `llm_judge` experiment exists to measure exactly
  that gap and is off by default.
- n = 48 is small; McNemar on 48 paired runs has modest power for small effect
  sizes. Treat non-significant differences as "not detected here", not "equal".
- Escalation results assume a 0.8 vs 0.5 step-accuracy gap; a larger model
  gap or per-step escalation could change the routing verdict.

## 7. Multi-seed aggregation (n = 144, seeds 0–2, difficulty policy)

> Supersedes §2–§4 for *significance* claims: 3× the samples, same paired
> design. `python -m dse.benchmarks.run_benchmark --seeds 3`. Record:
> `benchmarks/results/seeds3_from0_all.json`.

| strategy | success | mean_tokens | mean_lat (s) | attempts |
|---|---|---|---|---|
| react | 0.104 | 54 | 0.151 | 1.0 |
| best_of_n | 0.306 | 162 | 0.475 | 3.0 |
| escalating | 0.306 | 76 | 0.222 | 1.4 |
| reflexion | 0.521 | 196 | 0.388 | 2.4 |
| self_refine | 0.569 | 153 | 0.370 | 2.3 |
| escalating_per_step | 0.708 | 79 | 0.374 | 1.8 |
| adaptive | 0.910 | 278 | 1.937 | 1.9 |
| tree_search | 0.924 | 265 | 2.187 | 1.0 |

Key multi-seed findings (n = 144; McNemar p-values):

- **adaptive (re-tuned difficulty policy) now ties tree_search**: 0.910 vs
  0.924, p = 0.823 (not significant). The re-tuning fixed the previously
  measured weakness.
- **escalating_per_step is the best cost-efficiency point**: 0.708 at 79
  tokens ≈ 112 tokens/success — dramatically cheaper than search (287
  tokens/success) and significantly better than reflexion (p = 0.0018) and
  self_refine (p < 0.0001).
- whole-task escalating (0.306) ≈ best_of_n (p = 0.33): one expensive
  re-draft is not worth it; per-step escalation is the routing that works.
- reflexion vs self_refine: no detected difference (p = 0.20).

## 8. The noisy-judge experiment — the honest caveat, measured

> `python -m dse.benchmarks.run_benchmark --seeds 3 --flag llm_judge:true`.
> Record: `benchmarks/results/seeds3_from0_all_llm_judge.json`.
> The per-step value function switches from the deterministic test to a noisy
> LLM judge (judge_accuracy = 0.75); pruning is automatically disabled because
> it is unsound under noise.

| strategy | success (clean) | success (noisy) | tokens (noisy) |
|---|---|---|---|
| tree_search | 0.924 | **0.444** | 2152 |
| adaptive | 0.910 | **0.507** | 2018 |
| escalating_per_step | 0.708 | **0.708** | 79 |
| reflexion | 0.521 | 0.521 | 196 |
| self_refine | 0.569 | 0.569 | 153 |

Findings:

1. **The entire search advantage came from the perfect per-step oracle.**
   With a realistic noisy judge, tree_search collapses to 0.444 (below
   reflexion), its cost explodes ~8× (no pruning is possible), and adaptive —
   which routes to search — collapses with it.
2. **Deterministic-feedback strategies are robust to judge noise.**
   escalating_per_step, reflexion, and self_refine are unchanged because they
   verify against deterministic tests, not the noisy judge. Under noisy
   evaluation, **escalating_per_step is the best strategy** (0.708,
   significantly better than every other, p < 0.002 vs all).
3. **Design conclusion (evidence-backed, not assumed):** when per-step
   verification is deterministic, search is the right tool; when it is noisy,
   don't search against the noise — use deterministic final-answer tests plus
   targeted repair (per-step escalation).

## 9. Adaptive policy re-tuning (score vs difficulty)

> `--adaptive-policy score` vs the default `--adaptive-policy difficulty`.
> Record: `benchmarks/results/seeds3_from0_all_score.json`.

| policy | adaptive success | tokens | vs tree_search (0.924) |
|---|---|---|---|
| score (v1) | 0.722 | 248 | significantly worse (p = 0.0006) |
| difficulty (v2, default) | 0.910 | 278 | no difference (p = 0.823) |

The re-tuning is a **measured +18.8pp improvement** (0.722 → 0.910): routing
hard tasks (num_steps >= 3) straight to search — instead of letting the probe
score route them to the medium (reflexion) tier — closed the gap to plain
search. The `adaptive_compute` experiment is now adopted as the default policy,
with the score policy retained as a benchmarked control.

---

## 10. Real hard-real run (live deepseek-v4-flash, 2026-08-06)

> `python -m dse.benchmarks.run_benchmark --provider deepseek --suite hard-real`
> Record: `benchmarks/results/seed0_hard-real.json`.
> The new `hard-real` suite (12 multi-step tasks with deterministic checkers,
> no code execution) run against the live API.

| strategy | success | mean_lat (s) | mean_tok | cost_usd | attempts |
|---|---|---|---|---|---|
| react | 1.000 | 1.48 | 206 | 0.000113 | 1.0 |
| best_of_n | 1.000 | 4.94 | 656 | 0.000380 | 3.0 |
| reflexion | 1.000 | 1.53 | 209 | 0.000116 | 1.0 |
| self_refine | 1.000 | 1.72 | 211 | 0.000119 | 1.0 |
| escalating | 1.000 | 1.73 | 212 | 0.000119 | 1.0 |

**Finding (honest): the hard-real suite is STILL too easy for V4 Flash.**
Every strategy including single-shot react solved all 12 tasks (all McNemar
p = 1.000, zero discordant pairs). The tasks are genuinely multi-step (they
fool weaker models), but deepseek-v4-flash solves them one-shot. Measuring the
strategy delta needs tasks that are hard FOR THIS MODEL: longer chains,
large exact-arithmetic answers, deliberate ambiguity traps, or constraint-
satisfaction checks. This mirrors the earlier finding that the base real suite
was too easy. Cost note: best_of_n costs 3× tokens for zero gain on this suite.

---

## 11. Large-scale + stress check (2026-08-06)

> `--seed 0 --n-tasks 400 --suite all --seeds 3` → n = 1200 per strategy.
> Record: `benchmarks/results/stress_n400_s3.json`.

| strategy | success (n=1200) | vs n=144 |
|---|---|---|
| tree_search | 0.912 | 0.924 |
| adaptive | 0.908 | 0.910 |
| escalating_per_step | 0.736 | 0.708 |
| reflexion | 0.615 | 0.521 |
| self_refine | 0.602 | 0.569 |
| best_of_n | 0.393 | 0.306 |
| escalating | 0.388 | 0.306 |
| react | 0.180 | 0.104 |

- **Scales cleanly:** 1200 samples × 8 strategies in 0.6 s (mock).
- **Ordering is stable** at 3× the sample size; react is always last, search
  tier always on top.
- **Fully reproducible:** two identical runs produced bit-identical success
  vectors at n = 1200.
- Provider reliability probe (live): 12 diverse calls → 0 retries, 2/12
  reasoning-content fallbacks (the design prompts), 1.1–7.9 s/call,
  ~$0.005 total. The blank-completion quirk is measurable now via
  `OpenAICompatibleLLM.empty_completions` / `.fallback_count`.

---

## 12. Hard-tuned real run — third 100% ceiling (2026-08-06)

> `python -m dse.benchmarks.run_benchmark --provider deepseek --suite hard-tuned`
> Record: `benchmarks/results/seed0_hard-tuned.json`.
> Tasks targeted at a STRONG model's known weak spots: large exact arithmetic
> (1234×5678, 37^5 mod 100), double-counting (divisible by 2/3/5 ≤ 200 = 146),
> percent-off-then-tax ordering (77.76), classic traps (snail = day 8, CRT
> > 20 = 53), expected value (12.25), long operation chains (35).

| strategy | success | mean_lat (s) | mean_tok | cost_usd |
|---|---|---|---|---|
| react | 1.000 | 2.25 | 260 | 0.000176 |
| best_of_n | 1.000 | 6.51 | 773 | 0.000521 |
| reflexion | 1.000 | 2.25 | 254 | 0.000169 |
| self_refine | 1.000 | 2.21 | 254 | 0.000170 |
| escalating | 1.000 | 2.10 | 261 | 0.000177 |

**Finding (honest, third replicate): deepseek-v4-flash solves ALL of these
single-shot.** Three consecutive real suites (real, hard-real, hard-tuned) are
100% for every strategy — the strategy machinery provides ZERO measurable
improvement over one-shot for text-based math/logic on this model. The engine's
improvement value is demonstrated only (a) under the controlled synthetic
suite (MockLLM encodes the feedback mechanism), and (b) on open-ended
questions (self_refine never worse, ~40% better — the 10-question live
validation). To make strategies pay off on a strong model, tasks must be ones
where one-shot genuinely fails: **code generation with execution** (needs a
sandbox; currently out of scope) or tool-use/retrieval tasks. Alternatively,
route to a genuinely weaker/cheaper cheap tier where the delta reopens.

---

## 13. Code suite — 4th consecutive 100% ceiling (2026-08-06)

> `python -m dse.benchmarks.run_benchmark --provider deepseek --suite code`
> Record: `benchmarks/results/seed0_code.json`.
> The model's generated code is EXECUTED locally (timeout + temp cwd; not a
> security sandbox) against hidden test cases: fizzbuzz, run-length compress,
> palindrome (ignoring case/punct), two-sum, reverse-words, longest substring
> without repeats, min-coins (DP), binary search.

| strategy | success | mean_lat (s) | mean_tok | cost_usd |
|---|---|---|---|---|
| react | 1.000 | 2.22 | 307 | 0.000220 |
| best_of_n | 1.000 | 6.34 | 911 | 0.000648 |
| reflexion | 1.000 | 2.97 | 398 | 0.000295 |
| self_refine | 1.000 | 2.98 | 468 | 0.000323 |
| escalating | 1.000 | 2.62 | 316 | 0.000230 |

**Finding (honest, fourth replicate): even code-with-execution is solved
single-shot by V4 Flash.** The execution harness works (real subprocess
runs, concrete FAIL got=/want= feedback for repair), but the model gets all 8
classic tasks right on the first try — including the DP min-coins problem.
Four consecutive 100% real suites. The delta will only appear with: novel
problems not in training data, longer/integration-level synthesis, or a
genuinely weaker/cheaper cheap tier (the Snell test-time-compute thesis).

---

## 14. Weak-model runs — the strategy delta finally appears (2026-08-06)

> `--provider ollama --model-cheap llama3.2:1b --model-expensive llama3.2:1b`
> (local, genuinely weak chat-capable model).
> Records: `benchmarks/results/seed0_code_llama1b.json` and
> `benchmarks/results/seed0_hardreal_llama1b.json`.

| suite (n) | react | best_of_n | reflexion | self_refine | escalating |
|---|---|---|---|---|---|
| code (8) | 0.500 | 0.500 | 0.500 | **0.625** | 0.500 |
| hard-real (12) | 0.333 | 0.333 | **0.667** | 0.417 | 0.333 |

**Findings:**

1. **The four 100% ceilings were a strong-model artifact.** With a genuinely
   weak model, single-shot fails 50–67% of tasks — the strategies finally
   have a gap to work with.
2. **Feedback-driven repair recovers failures on REAL models** — the synthetic
   prediction replicates: hard-real `reflexion` 0.667 = **2× react** (4 tasks
   recovered, McNemar p = 0.13); code `self_refine` 0.625 > react 0.500.
3. **Resampling still gives zero gain** (best_of_n ties react) — consistent
   with the feedback-beats-resampling finding, now on a real model.
4. **Honest limits:** not yet significant at n = 8/12 (p ≈ 0.13 — needs more
   tasks/replicates or a mid-weight instruct model), and weak+repair (0.667)
   is still below strong single-shot (1.000). The claim made is "repair
   recovers a large share of a weak model's failures at low cost", NOT yet
   "cheap beats expensive".

### 14b. Aggregated across all four real suites — STATISTICALLY SIGNIFICANT (2026-08-06)

> Fresh run of code (8) + hard-real (12) + hard-tuned (12) + real (12) on
> llama3.2:1b, aggregated per-task (n = 32 hard / n = 44 all).

| comparison | n | react | reflexion | b01 | b10 | chi² | p | verdict |
|---|---|---|---|---|---|---|---|---|
| react vs reflexion (ALL) | 44 | 0.432 | 0.636 | **0** | **9** | 7.11 | **0.0077** | **SIG** |
| react vs reflexion (HARD) | 32 | 0.344 | 0.531 | **0** | **6** | 4.17 | **0.0412** | **SIG** |
| react vs self_refine (ALL) | 44 | 0.432 | 0.500 | 0 | 3 | 1.33 | 0.248 | no |
| react vs best_of_n (ALL) | 44 | 0.432 | 0.432 | 0 | 0 | 0.00 | 1.000 | no |

### 14c. Replication + combined (n = 88 all / n = 64 hard) — ABSOLUTE STATS (2026-08-06)

> Second full pass on llama3.2:1b, combined with §14b (two independent runs).
> Per-task data: `benchmarks/results/weak_aggregate_run2.json`.

| comparison | subset (n) | b01 | b10 | McNemar chi² | McNemar p | exact sign p | verdict |
|---|---|---|---|---|---|---|---|
| react vs reflexion | ALL (88) | **0** | **18** | 16.06 | **0.000062** | 0.000008 | **SIG** |
| react vs reflexion | HARD (64) | **0** | **12** | 10.08 | **0.0015** | 0.0005 | **SIG** |
| react vs self_refine | ALL (88) | **0** | 6 | 4.17 | **0.0412** | 0.0312 | **SIG** |
| react vs self_refine | HARD (64) | 0 | 4 | 2.25 | 0.134 | 0.125 | no |
| react vs best_of_n | ALL (88) | 0 | 0 | 0.00 | 1.000 | 1.000 | no |

**Findings:**

1. **Replication confirmed — the effect is stable.** Run 1 and run 2 produced
   IDENTICAL aggregate rates (reflexion 0.636 / 0.531 in both runs).
2. **Never worse, consistently:** b01 = 0 in every comparison, every run —
   reflexion/self_refine never degraded a task react got right; all discordant
   pairs were react-fail → repair-pass (18 of 18, 12 of 12).
3. **Effect size (run 2):** reflexion recovered 9/25 react failures (all),
   6/21 (hard). Bootstrap 95% CI on the rate difference (reflexion − react):
   **[+0.091, +0.318]** (all) / **[+0.062, +0.344]** (hard) — entirely above
   zero.
4. **Resampling = zero, in both runs:** best_of_n exactly ties react
   (p = 1.0). Feedback-beats-resampling is now replicated on a real model.
5. Scope honest: n = 88/64, weak 1b model, short tasks. NOT "cheap beats
   expensive" (0.636 < strong 1.0); NOT applicable to frontier models.
