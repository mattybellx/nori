# nori — Research Synthesis & Evidence Base

> Per the Master Brief: *"Never assume an approach is best because it is common."*
> This document records what was researched, what the evidence says, and why each
> architectural decision was made. It is the audit trail for the implementation.
>
> **Status of each claim is labelled:**
> - **[ESTABLISHED]** — replicated, peer-reviewed, or official-documentation-backed consensus.
> - **[EXPERIMENTAL]** — promising single-study or early evidence; needs local measurement.
> - **[HYPOTHESIS]** — our own conjecture, to be validated or rejected by our benchmark harness.

---

## 0. Scope

The objective (per the brief) is an **evidence-driven agent orchestration engine**:
test-time compute allocation, planning/search, reflection, verification, routing,
multi-agent orchestration, feedback loops, telemetry, and benchmarking. All
experimental capabilities sit behind feature flags with explicit hypotheses and
rollback conditions.

---

## 1. Test-time compute & inference-time scaling

### Evidence
- **Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally..." (arXiv:2408.03314) [ESTABLISHED]**
  - Two mechanisms for scaling test-time compute: (1) search against a dense
    process-based verifier, (2) adaptively revising/updating the response distribution.
  - **The effectiveness of every scaling method depends critically on prompt
    difficulty.** A *compute-optimal* strategy that allocates compute adaptively
    per prompt is **>4x more compute-efficient than a best-of-N baseline**.
  - FLOPs-matched: test-time compute on a smaller model can **outperform a 14x
    larger model** on prompts where the small model has non-trivial baseline success.

### Design implications (adopted)
1. Compute budget must be **allocated adaptively per prompt**, not uniformly.
   → We implement a *difficulty probe* (cheap first attempt + verifier confidence)
   and route to low/medium/high compute regimes. **[ESTABLISHED]**
2. Test-time compute is only as good as the **verifier** it searches against.
   → The Verifier is a first-class abstraction with pluggable sources. **[ESTABLISHED]**
3. Best-of-N is the **baseline**, not the target. Search with a verifier and
   adaptive revision are the candidates we benchmark against it. **[ESTABLISHED]**

---

## 2. Tree search (ToT, LATS)

### Evidence
- **Yao et al. 2023, "Tree of Thoughts" (arXiv:2305.10601) [ESTABLISHED]**
  - Explicit search over coherent "thought" units with self-evaluation and
    lookahead/backtracking. Game of 24: **4% (CoT) → 74% (ToT)** with GPT-4.
  - Strongest on tasks that require exploration / strategic lookahead; *costly*
    and not always worth it on simple tasks.
- **Zhou et al. 2023, "Language Agent Tree Search (LATS)" (arXiv:2310.04406) [ESTABLISHED]**
  - Unifies reasoning + acting + planning via **Monte Carlo Tree Search** with
    **LM-powered value functions**, **self-reflections**, and **environment feedback**.
  - **92.7% pass@1 on HumanEval (GPT-4)**; 75.9 average on WebShop (GPT-3.5),
    comparable to gradient-based fine-tuning.
  - Key ingredients we adopt: environment feedback, value function from verifier,
    reflection on failed branches, UCT selection.

### Design implications (adopted)
- Default search strategy = **MCTS ("LATS-lite")**: UCT selection → expansion via
  LLM proposals → value estimation via verifier → reflection on terminal failures.
  **[ESTABLISHED]**
- Search is **budget-capped** and only engaged for medium/hard prompts (per §1).
  **[ESTABLISHED, from Snell et al. compute-optimal]**

---

## 3. Verbal reinforcement / reflection

### Evidence
- **Shinn et al. 2023, "Reflexion" (arXiv:2303.11366) [ESTABLISHED]**
  - Agents verbally reflect on feedback, store reflections in **episodic memory**,
    and reuse them in later trials. **91% pass@1 HumanEval** vs 80% for GPT-4 at the time.
- **Madaan et al. 2023, "Self-Refine" (arXiv:2303.17651) [ESTABLISHED]**
  - Same model generates → critiques → refines. ~**20% absolute average gain** across
    7 tasks with GPT-3.5/4.
- **CRITICAL CAVEAT — Huang et al. 2023, "Large Language Models Cannot
  Self-Correct Reasoning Yet" (arXiv:2310.01798) [ESTABLISHED, replicated finding]**
  - **Intrinsic self-correction** (no external feedback) often *degrades* accuracy;
    the *apparent* gains in Self-Refine are frequently attributable to the external
    verifier selecting a better draft, not to the refinement itself.

### Design implications (adopted)
1. **Reflection/refinement MUST be driven by external verifier signals**
   (tests, rules, environment), never pure self-critique. **[ESTABLISHED]**
2. Reflexion-style **episodic memory** of failures is kept and replayed. **[ESTABLISHED]**
3. Self-Refine is implemented but **gated**: refinement only proceeds when an
   external signal says the current draft fails; never as an open loop. **[ESTABLISHED]**

---

## 4. Multi-agent orchestration (Mixture-of-Agents)

### Evidence
- **Wang et al. 2024, "Mixture-of-Agents" (arXiv:2406.04692) [ESTABLISHED]**
  - Layered architecture: proposers → aggregator. Open-source-only MoA hit
    **65.1% on AlpacaEval 2.0** vs GPT-4 Omni 57.5%.
  - Cost is **N× the token budget** — real and material.
- **Hong et al. 2023, "MetaGPT" (arXiv:2308.00352) [ESTABLISHED]**
  - SOP-driven role decomposition with **verification of intermediate results**
    reduces "cascading hallucinations" from naive chaining.

### Design implications (adopted)
- MoA-style aggregation is **[EXPERIMENTAL]** in our engine: behind a feature flag,
  with a benchmark plan and a token-cost gate. Default is single-agent + search.
- When multi-agent is on, it uses **verifier-gated handoffs** (MetaGPT lesson):
  an output only propagates if the verifier passes it. **[ESTABLISHED]**

---

## 5. Agent–computer interfaces & coding feedback loops

### Evidence
- **Yang et al. 2024, "SWE-agent" (arXiv:2405.15793) [ESTABLISHED]**
  - **ACI design materially changes agent performance.** Structured commands beat
    free-form tool use; 12.5% pass@1 on SWE-bench, 87.7% on HumanEvalFix.
- **AlphaCodium (Ridnik et al. 2024, arXiv:2401.08500) [ESTABLISHED]**
  - "Flow engineering": two-phase (pre-processing + iterative coding) with
    **test-based feedback**. ~**44% absolute gain on CodeContests** with GPT-4
    (19% → 63% pass@5).
  - Note: arXiv 2401.16196 is an unrelated algebraic-geometry paper; AlphaCodium's
    correct ID is 2401.08500 (verified against the abstract at retrieval time of this
    document; provenance recorded because a fetch of the wrong ID returned a mismatch).

### Design implications (adopted)
- The compiler feedback loop returns **parsed, structured results**
  (pass/fail per test, lint/security findings, perf deltas), not raw terminal dumps.
  **[ESTABLISHED]**
- Agents interact through a small **command surface** (read/search/edit/run),
  per the SWE-agent ACI lesson. **[ESTABLISHED]**

---

## 6. Verifiers

### Evidence
- **Snell et al. 2024** (above): search against **process verifiers** is one of the
  two effective scaling mechanisms. **[ESTABLISHED]**
- **Lightman et al. 2023, "Let's Verify Step by Step" (arXiv:2305.20050) [ESTABLISHED]**
  - Process supervision (per-step) beats outcome supervision for math reasoning.
- **Wang et al. 2022, "Self-Consistency" (arXiv:2203.11171) [ESTABLISHED]**
  - Majority vote over sampled reasoning paths is a cheap, weak-but-robust signal.

### Design implications (adopted)
- `Verifier` protocol with pluggable sources: **exact/rule** (deterministic),
  **test-execution** (AlphaCodium-style), **LLM judge**, **self-consistency vote**.
- Verifiers **aggregate** (weighted) when multiple signals exist; the aggregate
  drives compute allocation and search value functions. **[ESTABLISHED]**
- We do **not** implement a trained PRM (needs fine-tuning data); instead the value
  function is an LLM judge + heuristic composition — an acknowledged limitation.

---

## 7. Dynamic model routing; MoE vs MoA

### Evidence
- MoE = **parameter-level** sparse routing inside a model (infrastructure concern,
  established in production LLM serving). MoA = **output-level** ensembling
  (Wang et al. 2024). They solve different problems. **[ESTABLISHED]**
- For an orchestration engine, the relevant lever is **choosing which model/effort
  to spend per prompt** — consistent with compute-optimal scaling (§1).

### Design implications (adopted)
- `Router` implements **confidence-based escalation**: start cheap, escalate to
  expensive model when verifier confidence is low (or difficulty probe is high).
  Cost/latency goals are explicit inputs. **[HYPOTHESIS → benchmarked]**
- MoE itself is out of scope (it is a model-training/infra concern).

---

## 8. Context management & memory

### Evidence
- Sliding-window + summarization is the de-facto practice in long-horizon agent
  systems (OpenHands, MemGPT lineage) to bound context growth. **[ESTABLISHED, practice]**
- Episodic memory of reflections is the mechanism in Reflexion (§3). **[ESTABLISHED]**

### Design implications (adopted)
- `Memory`: rolling window + rolling summary + episodic reflection buffer.
  Budgets are explicit (token caps), summaries are triggers, not silent. **[ESTABLISHED]**

---

## 9. Benchmark methodology & statistical significance

### Evidence / practice
- Report the full context (hardware, models, latency, tokens, success, CIs). **[BRIEF REQUIREMENT]**
- Paired binary outcomes (same task set, two strategies) → **McNemar's test**;
  difference-of-rates CIs via **bootstrap**. This is the standard, accepted approach
  for LLM-agent strategy comparisons. **[ESTABLISHED, practice]**
- Determinism: fixed seeds + a **MockLLM** with controllable per-task accuracy make
  benchmarks reproducible and strategy comparisons causal (no API variance). **[ESTABLISHED]**

### Design implications (adopted)
- Benchmark harness reports: n, success rate, mean/median/p90 latency, mean tokens,
  estimated cost, McNemar p-value and bootstrap CI for every strategy pair, plus
  hardware/model/seed metadata. Results serialize to JSON for the record.

---

## 10. Decisions summary

| Decision | Choice | Evidence |
|---|---|---|
| Compute allocation | Adaptive per-prompt (difficulty-gated) | Snell et al. 2024 [ESTABLISHED] |
| Search | MCTS/LATS-lite, budget-capped | Zhou et al. 2023 [ESTABLISHED] |
| Reflection | External-feedback-driven only | Huang et al. 2023 [ESTABLISHED] |
| Self-Refine | Gated on verifier failure signals | Huang et al. 2023 [ESTABLISHED] |
| Verifier | Pluggable sources, aggregated | Snell 2024; Lightman 2023 [ESTABLISHED] |
| Routing | Confidence-based escalation | Hypothesis, benchmarked |
| Multi-agent | Behind flag, verifier-gated handoffs | Wang 2024; Hong 2023 [EXPERIMENTAL] |
| Feedback loop | Parsed/structured signals, small command surface | Yang 2024; Ridnik 2024 [ESTABLISHED] |
| Context | Window + summary + episodic memory | Reflexion; agent practice |
| Benchmarks | Paired, McNemar + bootstrap, deterministic | Practice |

---

## 11. Known failure modes & mitigations

1. **Self-critique without external signal** degrades performance → we gate all
   refinement on verifier signals. [ESTABLISHED]
2. **Search cost explosion** on easy prompts → difficulty-gated budgets. [ESTABLISHED]
3. **Verifier bias / weak verifier** → always benchmark strategy × verifier, never
   assume; verifier quality is reported alongside success.
4. **Unreproducible benchmarks** (API variance) → MockLLM + fixed seeds for CI;
   real-LLM runs reported separately with full metadata.
5. **Multi-agent token blowup** → cost gate and verifier-gated handoffs.

---

## 12. References (verified at retrieval)

1. Snell, Lee, Xu, Kumar — *Scaling LLM Test-Time Compute Optimally* — arXiv:2408.03314
2. Zhou et al. — *Language Agent Tree Search* — arXiv:2310.04406
3. Shinn et al. — *Reflexion: Language Agents with Verbal Reinforcement Learning* — arXiv:2303.11366
4. Madaan et al. — *Self-Refine: Iterative Refinement with Self-Feedback* — arXiv:2303.17651
5. Huang et al. — *Large Language Models Cannot Self-Correct Reasoning Yet* — arXiv:2310.01798
6. Yao et al. — *Tree of Thoughts* — arXiv:2305.10601
7. Wang et al. — *Mixture-of-Agents Enhances LLM Capabilities* — arXiv:2406.04692
8. Hong et al. — *MetaGPT* — arXiv:2308.00352
9. Yang et al. — *SWE-agent: Agent-Computer Interfaces* — arXiv:2405.15793
10. Ridnik et al. — *Code Generation with AlphaCodium* — arXiv:2401.08500
11. Lightman et al. — *Let's Verify Step by Step* — arXiv:2305.20050
12. Wang et al. — *Self-Consistency Improves Chain of Thought Reasoning* — arXiv:2203.11171
13. Yao et al. — *ReAct: Synergizing Reasoning and Acting* — arXiv:2210.03629
