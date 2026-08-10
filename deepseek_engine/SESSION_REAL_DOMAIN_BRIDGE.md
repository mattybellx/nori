# Session record — the REAL-DOMAIN BRIDGE (2026-08-10/11)

> **What this document is**: the complete, line-by-line account of taking the
> architecture-discovery laboratory to the REAL DeepSeek model with an
> INDEPENDENT judge on FREE-FORM questions. Every file, every function, every
> design decision, every bug, every number. This is the "honest frontier"
> made runnable — the next chapter after all 10 discovery phases.
>
> Companion docs: `ARCHITECTURE_DISCOVERY.md` (the full 10-phase lab),
> `MASTER_PROJECT_DOC.md` (the hub), `BENCHMARKS.md` (measured numbers).

---

## §0. TL;DR

- Built `dse/discovery/real_domain.py` — runs discovered architectures on the
  real DeepSeek model over free-form questions, graded by the **independent
  pro judge** (median-N, question-aware), with the A/B preference judge,
  groundedness weighting, and the sign test.
- **13 new mock-safe tests** + **1 regression test** → **305 tests passing**.
- **Found and fixed a real bug**: any architecture whose exit node is a
  `verify` node silently recorded an EMPTY answer text. Harmless while
  benchmarks only measured success rate; fatal for the real-domain
  experiment. Fixed in `executor.py` (last-text recovery) with a regression
  test.
- First real experiment: `react` (baseline) vs `best_of_n_ify(react, 3)`
  (the architecture the discovery loop invented on the mock), 6 open-ended
  questions, judged by `deepseek-v4-pro`. **Result: NO PROMOTION,
  inconclusive with a negative lean — mean 8.33 vs 9.83 (−1.5), never-worse
  4/6, preference 3/0/3 with p=0.25.** Full analysis in §8.
- Rename to **nori** completed for pip metadata (`pyproject.toml`); directory
  rename deferred as a dedicated follow-up (too risky mid-experiment).

---

## §1. Why this session exists (the honest frontier)

The 10 discovery phases proved — on the **seeded mock** with **objective
verifiers** — that:

- architectures are executable and faithfully reproduce the strategies they
  wrap (Phases 1–2),
- the loop + evolution discover architectures that beat weak baselines
  (Phases 3–5),
- routing specializes per task class (Phase 6),
- the Pareto frontier flags dominated/compute-wasting designs (Phase 7),
- failure-driven invention fixes real failures, compression halves compute
  (Phase 8),
- new primitives pass contract gates before promotion (Phase 9),
- the meta-level discovers better ways to discover (Phase 10).

**What was NOT claimed** (from `ARCHITECTURE_DISCOVERY.md` "Honest status"):

> The research hypothesis is NOT proven: nothing has survived
> **real-model, free-form, independent-judge** evaluation at scale.

That is exactly what this session builds: the bridge to the real domain, using
the machinery already validated in the free-form benchmark — the **independent
judge** (breaking the self-grading circularity that inflates significance:
we measured p=0.011 self-judge vs p=0.32 independent), the **preference
judge**, **groundedness**, and the **sign test**.

---

## §2. Session timeline (in order)

1. Created `dse/discovery/real_domain.py` (the bridge: context builder, arch
   runner, independent scoring, preference comparison, experiment orchestrator,
   CLI with JSON-first persistence).
2. Simplified the CLI (dropped an unnecessary best-effort `majority_vote`
   registration that the pool doesn't need).
3. Exported the module from `dse/discovery/__init__.py` — first eagerly, then
   **lazily via `__getattr__`** after discovering that the eager import breaks
   `python -m dse.discovery.real_domain` (module already in `sys.modules`
   → runpy RuntimeWarning → exit code 1).
4. Wrote `tests/test_discovery_real_domain.py` (13 tests). Hit the project's
   `python_classes = ["*Tests"]` pytest rule — renamed `Test*` classes to
   `*Tests`.
5. Full suite: **304 passing**.
6. Smoke-tested the real wiring on ONE question (react graph, real DeepSeek):
   worked — real answer produced. (Debug script bug only: `rec.tokens` doesn't
   exist; the module itself only uses `rec.answer`.)
7. **First real run launched** — react produced 6 real answers, then the
   candidate produced **six EMPTY answers (0 chars)**.
8. **Diagnosed the empty-answer bug** (see §4). Killed the run.
9. **Fixed `executor.py`** + added regression test → **305 passing**.
10. **Restarted the experiment** with the fix (still running — §8).
11. Renamed pip metadata to `nori`; wrote this doc; recorded the rename
    preference in memory.

---

## §3. The module, in full detail

### 3.1 `dse/discovery/real_domain.py` — imports & reuse

```python
from ..benchmarks.harness import sign_test
from ..benchmarks.run_never_worse import (
    _FREE_QUESTIONS, _build_judge_llm, _make_robust_judge, _preference_judge,
)
from ..env import load_env
from ..freeform import FreeFormJudge, PromptTask
from ..guards import grounded_score
from .compiler import compile_graph
from .executor import ArchExecutor
from .graph import ArchGraph
from .primitives import ExecutionContext
```

The design principle: **never rebuild what the free-form benchmark already
validated.** The judge-side machinery (`_build_judge_llm`, `_make_robust_judge`,
`_preference_judge`, `_FREE_QUESTIONS`) is reused verbatim from
`run_never_worse.py`. The only genuinely new code is: (a) building an
`ExecutionContext` for the real free-form domain, (b) running graphs against
real questions, (c) orchestrating the experiment, (d) the honest summary.

### 3.2 `build_freeform_context(provider="deepseek")` → `ExecutionContext`

```python
def build_freeform_context(provider: str = "deepseek", budget=None) -> ExecutionContext:
    from ..ask import build_ask_pipeline
    llm, models, _judge, strategies, budget = build_ask_pipeline(provider, None, None)
    verifier = FreeFormJudge(llm, model="expensive")
    return ExecutionContext(
        llm=llm, verifier=verifier, models=models,
        agents={s.name: s for s in strategies}, budget=budget,
        config=None,
    )
```

- `build_ask_pipeline` raises `ValueError` for `"mock"` — the bridge is real
  provider only (this is deliberate; mock determinism is for the 10-phase
  tests, not for this).
- The verifier is the calibrated `FreeFormJudge` (0–10 scale, `score/10.0`,
  `passed = score >= 5.0`, full retry ladder for the flash blank-quirk).
- `agents` maps strategy names → Agent objects so the `strategy` primitive
  (which the baseline graphs use) runs the EXACT existing strategy
  implementation.
- The `strategy` primitive's output is the agent's `RunResult`, so recorded
  success/answer come straight from the measured strategy.

### 3.3 `run_architectures_real(graphs, questions, ctx)` → `{arch: [answer...]}`

```python
def run_architectures_real(graphs, questions, ctx):
    ex = ArchExecutor()
    out = {}
    for graph in graphs:
        compiled = compile_graph(graph)
        answers = []
        for qi, question in enumerate(questions, 1):
            task = PromptTask(id=f"real-{graph.name}-{qi}", prompt=question)
            try:
                rec = ex.run(compiled, task, context=ctx)
                answers.append(rec.answer or "")
            except Exception as exc:
                answers.append(f"<error: {exc!r}>")   # never silently drop
        out[graph.name] = answers
    return out
```

- Every architecture answers every question.
- **Errors are recorded, never swallowed** — an `<error: ...>` string keeps
  the row aligned so scoring can't silently shift.
- Progress is printed per (arch, question) with answer length — this is how
  the empty-answer bug was spotted.

### 3.4 `score_real(...)` — the INDEPENDENT judge

```python
def score_real(answers_by_arch, questions, provider, judge_model, samples=3, main_llm=None):
    judge_llm = _build_judge_llm(provider, judge_model, main_llm)
    scores = {}
    for name, answers in answers_by_arch.items():
        row = []
        for qi, (question, answer) in enumerate(zip(questions, answers), 1):
            robust = _make_robust_judge(judge_llm, samples, question)
            score = robust(answer)
            row.append(round(score, 3) if score is not None else None)
        scores[name] = row
    return scores
```

- `_build_judge_llm`: when `judge_model` is given (e.g. `deepseek-v4-pro`), a
  **separate provider client** is built — grading is done by a DIFFERENT model
  than the one that wrote the answers. The self-grading circularity is broken
  exactly as in the free-form benchmark.
- `_make_robust_judge(llm, samples, question)` returns a median-of-N,
  question-aware scorer (median de-noising against judge noise).
- Missing judge scores → `None` (never a fake 0).

### 3.5 `preference_real(...)` — the A/B relative judge

```python
def preference_real(answers_by_arch, questions, baseline_name, provider, judge_model, main_llm=None):
    judge_llm = _build_judge_llm(provider, judge_model, main_llm)
    base = answers_by_arch[baseline_name]
    out = {}
    for name, answers in answers_by_arch.items():
        if name == baseline_name:
            continue
        row = []
        for qi, (question, arch_a, base_b) in enumerate(zip(questions, answers, base), 1):
            pick = _preference_judge(judge_llm, question, arch_a, base_b)
            row.append(pick)
        out[name] = row
    return out
```

- `_preference_judge` returns `"A"` (the candidate is better), `"B"` (the
  baseline is better), or `"TIE"`, with 3 retries and a default of `TIE` on
  parse failure. Relative comparison is the most direct attack on judge noise.

### 3.6 The report & summary — `RealDomainReport`, `build_summary`

`RealDomainReport` is a dataclass holding: provider, judge model, n questions,
judge samples, baseline name, the questions, the answers, the scores, the
preferences, the summary dict, and elapsed seconds. `to_dict()` → fully JSON-
serializable (results persist FIRST — the data-loss lesson).

`build_summary` computes, per candidate architecture:

| field | meaning |
|---|---|
| `mean_judge_score` | mean of the independent judge's median-N scores (0–10) |
| `baseline_mean` | same for the baseline |
| `delta` | mean(candidate) − mean(baseline) |
| `never_worse_by_judge` | count of questions where candidate ≥ baseline − 0.5 (judge noise floor) |
| `n` | valid scored questions |
| `grounded_avg` | mean of `grounded_score(candidate_answer, [baseline_answer], judge_score)` — the substance-weighted quality |
| `preference` | `{favor_arch, favor_baseline, ties, win_rate, sign_test_p}` from `sign_test(favor, total)` |

`_pref_stats`: `win_rate = favor / (favor + against)`; `sign_test_p` = the
two-sided binomial sign test from `harness.sign_test`. **6/6 favor → p =
0.03125** (the smallest run size at which the preference test can reach
significance at α=0.05).

### 3.7 `run_real_experiment(...)` → `RealDomainReport`

```python
questions = questions or _FREE_QUESTIONS
ctx = build_freeform_context(provider)
answers  = run_architectures_real(pool_graphs, questions, ctx)
scores   = score_real(answers, questions, provider, judge_model, samples, main_llm=ctx.llm)
prefs    = preference_real(answers, questions, baseline, provider, judge_model, main_llm=ctx.llm)
report.summary = build_summary(report)
```

### 3.8 CLI (`main()`)

```bash
python -m dse.discovery.real_domain --provider deepseek --judge-model deepseek-v4-pro --n 6 --samples 3 --out <ABS PATH> [--baseline react]
```

- Loads `.env`; errors if `DSE_PROVIDER_KEY` missing.
- Pool: `react_graph()` (single-shot baseline) + `best_of_n_ify(react_graph(), "s", n=3)`
  renamed to `best_of_n_ify_react` — 3 drafts, judge-selected (the architecture
  the discovery loop invented on the mock).
- **JSON is written BEFORE printing** (the data-loss lesson: a crash while
  printing must never destroy the results).

---

## §4. The bug: verify-node exits silently drop the answer TEXT

### 4.1 What happened

The first run: react produced 6 real answers (1345–2564 chars), then
`best_of_n_ify_react` produced **six empty answers (0 chars)**. Because the
candidate's answers were empty, the independent judge would score them ~0 and
the preference judge would always pick the baseline — a wasted, misleading run.

### 4.2 Root-cause chain (traced line by line)

1. `best_of_n_ify` (in `mutations.py`) rewires the graph so the **exit node is
   the final `verify` node** (`g.exit = vid`).
2. The `verify` primitive returns `NodeOutput(verdict, ...)` — its `value` is a
   `Verdict` object, **not a string**.
3. `NodeOutput.text` is a property: `self.value if isinstance(self.value, str) else ""`.
   → For the verify node, `out.text == ""`.
4. `_resolve_answer` in `executor.py` checks, in order:
   - `if out.meta.get("passed") is not None: return out.text, ...` — the verify
     node HAS `meta["passed"]` → returns `("", passed)`.
   - (the `Verdict` branch `return "", bool(value.passed)` is unreachable for a
     verify-node exit — the meta branch fires first, and is equally empty.)
5. Result: **`rec.answer == ""` for EVERY graph whose exit is a verify node.**

### 4.3 Why the mock never caught it

- The discovery benchmark (`benchmark_architectures`) measured **success rate**
  (the `passed` boolean), not answer text.
- `validate_equivalence` compared graph vs strategy on **success**, not text.
- `_resolve_answer` for verify-node exits returned `("", passed)` — success was
  always correct, so every metric that mattered on the mock was fine.

The real-domain experiment is the first place where the **answer text itself**
is the deliverable — and it immediately surfaced the bug.

### 4.4 The fix (`dse/discovery/executor.py`)

Track the most recent textual output on the executed path, and recover it when
the terminal resolves to an empty answer:

```python
last_text = ""                          # before the node loop
...
if out.text:                            # inside the loop, after storing the output
    last_text = out.text
...
answer, success = _resolve_answer(final_out)
if not answer and last_text:            # verify-node exit -> recover the text
    answer = last_text
record.answer = answer
```

Why this is correct and general:
- In a `best_of_N`-style graph, the last text node is exactly the `extract`
  node feeding the terminal `verify` — the selected draft's text. Recovered
  correctly.
- In graphs whose terminal already produces text (react's `strategy` node →
  RunResult), `answer` is non-empty → no fallback → behavior unchanged.
- It fixes every verdict-terminal shape (current and future), not just this
  one graph.

### 4.5 Regression test (`tests/test_discovery.py`)

```python
def test_executor_best_of_n_ify_preserves_answer_text(tiny_stack, discovery_ctx):
    """REGRESSION (real-domain bridge): a best_of_N expansion whose exit is a
    ``verify`` node must still record the answer TEXT. ..."""
    from dse.discovery.mutations import best_of_n_ify
    g = best_of_n_ify(react_graph(), "s", n=2)
    task = _task(tiny_stack, 1)
    rec = ArchExecutor().run(compile_graph(g), task, context=discovery_ctx)
    assert any(e.primitive == "verify" for e in rec.events)
    assert any(e.primitive == "extract" for e in rec.events)
    assert rec.answer != ""
    assert rec.answer is not None
```

It asserts the exit IS a verify node AND the answer text is preserved.

---

## §5. The lazy-export lesson (`dse/discovery/__init__.py`)

`real_domain` imports the live provider stack. Eagerly importing it from the
package `__init__` broke `python -m dse.discovery.real_domain` (runpy sees the
module already in `sys.modules` → RuntimeWarning → exit 1). Fix: export lazily
via a module-level `__getattr__` over an explicit `_REAL_DOMAIN_EXPORTS` set.
`from dse.discovery import run_real_experiment` still works; the package stays
cheap to import.

---

## §6. Tests (14 files, 305 passing)

### 6.1 `tests/test_discovery_real_domain.py` (13 tests, mock-safe)

| class | tests | what they verify |
|---|---|---|
| `MeanTests` | 3 | `_mean`: all-None → None; ignores None; empty → None |
| `PrefStatsTests` | 3 | preference aggregation: favor/against/ties, win rate; **sign-test p = 0.03125 for 6/6**; no-decisions → None |
| `BuildSummaryTests` | 6 | means + delta; never-worse counting; preference stats; groundedness averaging (0–10 bound); missing candidate scores → None/0; baseline excluded from summary |
| `ReportSerializationTests` | 1 | `to_dict()` round-trip of the full report shape |

All mock-safe by construction — they exercise report/stat logic with fabricated
data; the live experiment is a CLI run, not a unit test.

### 6.2 `tests/test_discovery.py` (+1 regression)

`test_executor_best_of_n_ify_preserves_answer_text` (§4.5).

---

## §7. The experiment protocol (honesty, step by step)

1. **Pool**: `react` (single-shot baseline) vs `best_of_n_ify_react` (3 drafts
   → objective judge-selection → extract → verify). The candidate's INTERNAL
   selection may use the **self judge** (flash) — that is its design.
2. **Answers**: both architectures answer the same 6 open-ended questions
   (first 6 of the 64-question free-form set) on real `deepseek-v4-flash`.
3. **Claims**: every score comes from the **independent `deepseek-v4-pro`**
   judge — median of 3 question-aware calls per (arch, question). Self-grading
   circularity is broken (the measured p=0.011 → p=0.32 lesson).
4. **Preferences**: per-question A/B "which answer is better" (pro judge),
   aggregated with the two-sided sign test.
5. **Groundedness**: substance-weighted score of each candidate answer vs its
   baseline answer (`grounded_score`).
6. **Interpretation rule**: n=6 can only reach p=0.03125 if ALL 6 favor the
   candidate. Anything less → **"I don't know"** is the honest verdict (spec
   §41: NO PROMOTION / INCONCLUSIVE are first-class outcomes).
7. **Persistence**: JSON written first; the run's full transcript (answers,
   scores, preferences, summary) survives any print crash.

---

## §8. RESULTS — first real-domain, independent-judge run (2026-08-11)

**Setup**: react (single-shot baseline) vs `best_of_n_ify_react` (3 drafts,
self-judge-selected → extract → verify) on the first 6 free-form questions;
every claim graded by the **independent `deepseek-v4-pro`** judge, median of 3
question-aware calls; A/B preference judge per question; groundedness vs the
baseline answer. Full transcript:
`benchmarks/results/real_domain_archs6_projudge.json`.

### Per-question independent-judge scores (0–10)

| # | question | react | best_of_n_ify | pref |
|---|---|---|---|---|
| 1 | quantum entanglement | 10.0 | 10.0 | A |
| 2 | remote work pros/cons | 10.0 | 10.0 | A |
| 3 | ML vs DL | 10.0 | 10.0 | A |
| 4 | recursion to a 5-yo | 10.0 | 10.0 | TIE |
| 5 | HTTP/1.1 vs HTTP/2 | 10.0 | **6.0** | TIE |
| 6 | solar vs wind for a town | 9.0 | **4.0** | TIE |

### Summary

| metric | value |
|---|---|
| mean judge (candidate) | **8.333** |
| mean judge (baseline) | **9.833** |
| delta | **−1.5** (candidate WORSE) |
| never-worse by judge | 4/6 |
| grounded-avg | 5.83 |
| preference (A/B/TIE) | **3 / 0 / 3** |
| preference sign-test p | **0.25** (n=3 decisions — NOT significant) |
| elapsed | 754.9 s |

### Honest verdict (spec §41)

**NO PROMOTION. INCONCLUSIVE with a negative lean on the absolute judge.**

What is TRUE and measured:
- The discovered `best_of_N` expansion did **NOT** beat single-shot react on
  these 6 questions under the independent judge. Its mean was 1.5 points
  lower, and it was **never-worse in only 4/6** — it lost q5 (HTTP/2) and q6
  (solar/wind) badly (6.0 and 4.0 vs 10.0 and 9.0).
- The **relative judge never preferred the baseline** (3 A / 0 B / 3 TIE) —
  a mild positive signal — but with only 3 decisive comparisons the sign test
  gives p=0.25, which is nowhere near significant (6/6 would be needed for
  p=0.03125). **"I don't know" is the honest verdict.**
- The failure mode is visible and instructive: on the four EASY questions
  (q1–q4) both architectures saturate at 10.0 — no headroom for best_of_N —
  and on the two questions where react was still strong, the candidate's
  INTERNAL self-judge selection picked a draft the independent judge scored
  6.0 and 4.0. The self-selected draft can be worse than the single-shot
  answer: best_of_N only pays off when the internal selector is reliable AND
  there is headroom to gain.
- This is the first real-domain, independent-judge measurement of a
  discovered architecture — and it keeps the research hypothesis
  **unproven**, exactly as it should. The bridge works; the claim does not
  get to lean on this run.

### What this means for next steps

1. The 64-question set starts with easy questions (saturation at 10.0) —
   a meaningful test needs questions with more headroom (harder/longer-form),
   and/or more of them.
2. The internal selector is the weak link for best_of_N on real free-form:
   self-judge (flash) selection mis-ranked on q5/q6. A synthesis/merge step
   (Phase-8 style) or an objective-selection guard may behave differently.
3. n=6 cannot reach significance; the honest scaling is n=8+ (8/8 → p=0.0078)
   on a harder question subset.

<!-- RESULTS (append: summary table, per-question scores, preferences, verdict) -->

---

## §9. Rename to nori — status

- **Brand**: already fully "nori" from the prior rebrand session (UI, CLI
  argparse/prints, HTML, bat files, doc H1s, tests).
- **Done this session**: `pyproject.toml` `name = "deepseek-engine"` →
  `"nori"` (pip metadata).
- **Deferred** (dedicated follow-up after this experiment): renaming the
  `deepseek_engine/` **directory** to `nori/` — touches `ask.bat`/`chat.bat`
  cd-paths, `.gitignore`, every doc path reference, `pip install -e` metadata.
  The Python package is `dse`, so imports are unaffected. The rename was
  deferred rather than done mid-experiment because the running experiment
  writes to an absolute path under `deepseek_engine\` and a mid-flight rename
  would break the write.
- **Model IDs are NOT renamed** — `deepseek-v4-flash`/`deepseek-v4-pro` are the
  real DeepSeek model names.

---

## §10. Lessons learned (this session)

1. **The mock hides text-correctness bugs.** Success-rate benchmarks never
   look at the answer text — the verify-node exit dropped it silently for the
   whole lab. The real domain immediately exposed it. Lesson: any architecture
   that will ever produce user-facing answers needs at least one
   answer-text-preserving test, even on the mock.
2. **Watch the exit node.** A terminal `verify`/verdict node is a
   success-measurement shape, not an answer-delivery shape. If the exit is a
   verdict, the executor must recover the text from the feeding node.
3. **Never trust a 0-char answer** — it is either the flash blank-quirk or a
   pipeline bug; investigate before spending judge calls on it.
4. **Kill-and-fix beats let-it-finish** when a run is producing garbage: the
   empty candidate answers would have yielded a misleading "candidate loses"
   result that we'd have had to discard anyway.
5. **Eager imports break `-m` execution** for runnable submodules — use lazy
   `__getattr__` exports.
6. **`python_classes = ["*Tests"]`** — this project only collects classes
   ending in "Tests"; new test classes must follow it.
7. **JSON-first persistence** — write results before printing (the data-loss
   lesson, reinforced).

---

## §11. Files touched

| file | change |
|---|---|
| `dse/discovery/real_domain.py` | **NEW** — the bridge (module + CLI) |
| `dse/discovery/executor.py` | **FIX** — last-text recovery for verdict terminals |
| `dse/discovery/__init__.py` | lazy exports for the bridge |
| `tests/test_discovery_real_domain.py` | **NEW** — 13 mock-safe tests |
| `tests/test_discovery.py` | +1 regression test (answer-text preservation) |
| `pyproject.toml` | package name → `nori` |
| `ARCHITECTURE_DISCOVERY.md` | + "real-domain bridge" section |
| `MASTER_PROJECT_DOC.md` | quick-facts 304 → 305 (after regression) |
| `/memories/nori-rename.md` | **NEW** — rename preference + pending directory rename |
| `/memories/repo/deepseek-engine.md` | session notes |
