# Nori

**Ask once. Get the best answer — with the thinking shown.**

Nori is an evidence-driven AI answer engine with a polished, bring-your-own-key
chat UI. Instead of one blind guess, it runs several strategies in parallel,
streams each model's real chain-of-thought live, ranks the results, and then
**synthesizes the best parts of every candidate into a single final reply** —
so the answer you actually read is better than any single attempt.

![Nori chat UI](docs/nori-screenshot.png)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Dependencies: stdlib only](https://img.shields.io/badge/dependencies-stdlib%20only-4caf50)](pyproject.toml)
[![Status: beta](https://img.shields.io/badge/status-beta-yellow)]

## Why it's different

- **Visible thinking** — each strategy's real hidden reasoning streams live into
  independently collapsible **Reasoning** blocks. You see *how* it got there,
  not just the answer.
- **Best-of-all answers** — the workflow merges the strongest parts of every
  candidate into one final synthesized reply (with a provenance note), instead
  of just returning whichever single answer scored highest.
- **Bring your own key** — paste a DeepSeek, OpenAI, GitHub Models or Ollama
  key into the Settings panel: it's connection-tested, stored locally, and
  hot-swapped with no restart.
- **Zero dependencies** — the entire runtime is Python's standard library.
  No `pip install` of packages required.
- **Evidence-driven, not claimed** — every strategy decision is traced to
  peer-reviewed research in [RESEARCH.md](RESEARCH.md); the design contract is
  [ARCHITECTURE.md](ARCHITECTURE.md); measured results — including honest
  negative results — are in [BENCHMARKS.md](BENCHMARKS.md).
- **Deterministic by default** — a calibrated `MockLLM` makes tests and
  benchmarks reproducible, so strategy comparisons are causal, not anecdotal.

## What it is

A modular framework for allocating test-time compute across agent strategies:

- **Verifier-first**: nothing improves unless a signal can measure it.
- **Strategies**: ReAct (baseline), Best-of-N (canonical baseline), Reflexion,
  Self-Refine (verifier-gated), LATS-lite tree search (MCTS + sound pruning),
  whole-task & per-step model escalation, adaptive compute allocation, and
  MoA-lite (feature-flagged).
- **Real-LLM providers**: OpenAI-compatible client (stdlib-only) for DeepSeek,
  Ollama, OpenAI, and GitHub Models — see `dse/providers.py`.
- **Measured, not claimed**: a paired benchmark harness with McNemar + bootstrap
  significance reporting, latency, tokens, cost, and multi-seed aggregation.

## Quickstart

```bash
# One-time: make `dse` usable from ANY folder (adds pytest for the test suite)
python -m pip install -e ".[dev]"

cd deepseek_engine
python -m pytest tests -q        # 124 tests
python -m dse.benchmarks.run_benchmark --seed 0 --n-tasks 48 --suite all
python -m dse.benchmarks.run_benchmark --seed 0 --seeds 3                # multi-seed
python -m dse.benchmarks.run_benchmark --flag llm_judge:true             # noisy-judge experiment
python -m dse.benchmarks.run_benchmark --provider ollama                 # real LLM (local)
python -m dse.chat                                                       # Grok-style chat UI
```

> After `pip install -e .` you can run `python -m dse.chat`, `python -m dse.ask`,
> and `python -m dse.benchmarks.run_benchmark` from **any** directory. There are
> also double-click launchers: `chat.bat` and `ask.bat` in the project root
> (and `chat.bat` one level up).

## Real-LLM providers (DeepSeek / Ollama / GitHub Models)

The engine speaks OpenAI-compatible chat-completions (stdlib only, no deps).
Configure via environment variables (see `dse/providers.py`):

```bash
# DeepSeek API (V4 Flash by default)
$env:DSE_PROVIDER_KEY = "sk-..."
python -m dse.benchmarks.run_benchmark --provider deepseek --suite real
python -m dse.benchmarks.run_benchmark --provider deepseek --suite hard-real  # harder multi-step tasks

# Ollama (local, no key)
python -m dse.benchmarks.run_benchmark --provider ollama

# GitHub Models (Copilot-class models with a GitHub PAT)
$env:DSE_PROVIDER_KEY = "github_pat_..."
python -m dse.benchmarks.run_benchmark --provider github --model-cheap gpt-4o-mini
```

> **Copilot note (honest):** VS Code Copilot has no public third-party API, so
> the engine cannot call Copilot directly. GitHub Models is OpenAI-compatible
> and serves Copilot-class models with a GitHub PAT — that is the supported
> path.

## Run with your DeepSeek token (step by step)

1. **Paste your token into `.env`** (already created, git-ignored — open it and
   fill in `DSE_PROVIDER_KEY=`):

   ```bash
   # .env
   DSE_PROVIDER_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
   ```

   Get a token at <https://platform.deepseek.com> → API Keys. (A documented
   template lives in `.env.example`.)

2. **Run the real suite against DeepSeek** — 12 natural-language math/logic/code
   tasks with deterministic checkers (no code execution):

   ```bash
   python -m dse.benchmarks.run_benchmark --provider deepseek --suite real
   ```

   The engine reads the token from `.env` automatically. Both tiers use
   **DeepSeek V4 Flash** (`deepseek-v4-flash`) — the only model used by default.
   `--seeds N` works too; pass `--judge-model deepseek-v4-pro` for a stronger
   grader/judge.

3. **Or run locally for free with Ollama** (no token):

   ```bash
   ollama pull deepseek-r1:8b   # or any model you have
   python -m dse.benchmarks.run_benchmark --provider ollama --suite real
   ```

> The **synthetic** suite (`--suite all` / `easy` / `hard`) is calibrated for
> the MockLLM (opaque step tokens) and is for reproducible strategy
> comparisons. The **real** suites are what you benchmark real models on:
> `--suite real` (12 easy single-step tasks) and **`--suite hard-real`**
> (12 deliberately harder multi-step tasks — compound interest, work rates,
> inclusion-exclusion, algorithm tracing, classic traps — all with
> deterministic checkers, no code execution). Start with `hard-real`: the base
> real suite is too easy to tell strategies apart.
> **`--suite hard-tuned`** (12 tasks aimed at a strong model's weak spots) and
> **`--suite code`** (8 code-generation tasks whose output is EXECUTED
> locally under a timeout against hidden tests — opt-in, not a security
> sandbox).

## Ask questions, audit answers, track improvement

**Easiest: the Nori chat UI.** (Nori is the local web app — a Grok-style
front-end for the engine.)

```bash
python -m dse.chat     # opens http://127.0.0.1:8787 in your browser
```

Type a question; each strategy's answer appears as a clean **lighter cyan-blue
gradient card**, with the full answer text and buttons: **Good / Bad / ★1-5
rating**, plus **"vs baseline: better / worse / same"** comparison against the
plain single-DeepSeek answer.

**Two answer modes** (bottom of the sidebar → **Settings**): **Auto** (default)
runs the full workflow and sends back a single **synthesized** answer — one
clean card whose reply is the merged best-of-all version (marked with a small
**✦ Best of all answers** pill). **Dev** shows all four strategy cards, scores,
ratings and the manual "Pick best" flow.

Auto mode keeps things clean for normal users: the sidebar starts closed, the
step guide and the redundant standalone compare are hidden, and strategy names
(react / best_of_n / …) don't appear on the answer bubble — you only see time,
tokens and cost. While the strategies run, the **thinking** panel shows each
one's real hidden chain-of-thought live: an independently collapsible block
per strategy with **Reasoning** (the model's actual deliberation) and
**Final answer** sub-sections. After the arbiter crowns a winner, the workflow
automatically **synthesizes** the answer — it merges the strongest parts of
every candidate into one final version and that merged answer is the reply in
the blue bubble (saved to the session, so reopening it later shows the same
merged answer). The winner bubble has exactly one **All** button that opens
every ORIGINAL candidate; each entry is clickable and pops the answer
full-screen with **‹ Prev / Next ›** (and ←/→ arrow keys) to step through all
answers seamlessly, and the **✦ Synthesize best answer** button inside the
panel re-merges on demand with a note on which parts came from where.

**Bring your own key**: in Settings you can paste any DeepSeek / OpenAI / GitHub
Models / Ollama key (plus an optional base URL). Nori tests the connection,
auto-detects the provider and shows it as a badge, saves it to
`settings.json`, and switches the live engine to it without a restart. A
failing key is never saved, and the full key is never sent back to the page.
Each card's footer shows a small muted line (strategy · time · tokens · cost) —
no loud tags, no badges. Everything is saved to `sessions/` while the
strategies run.

**AI pick best** — after the four strategies answer, one click asks the model
to rank every saved answer and crowns the winner with a subtle "★ Best answer"
pill plus a plain-language *reason* and *what could still improve*.

**Ghost compare** — opens a full side-by-side overlay: **NORMAL** (the plain
one-shot answer) on the left, **BEST** (the one Nori judged best) on the right,
both fully scrollable so you can read both end to end and judge for yourself.
The AI's "why" note is pinned at the top.

Every answer card also has its own **View** and **Compare** buttons: **View**
opens that answer full-screen in a popup, and **Compare** opens it side-by-side
with the plain one-shot answer so you can see what the strategy actually
changed. Escape or ✕ closes any popup.

The UI is a polished single page: the **Geomini** typeface (via Google Fonts,
with system fallbacks) used for every piece of text, a **red colourway**, a
pure-black dark mode with a clean **SVG** light/dark toggle (no emoji,
remembers your choice), a **Bionic reading** toggle next to it — bolds the
first ~55% of each word in answers (cards, compare columns, verdicts; code
blocks are skipped) so long replies are easier to skim, on/off and remembers
your choice — a collapsible history sidebar (☰ toggles it, remembers
your choice), and a live **Insights** panel in the sidebar — per-strategy
average judge score, rating, and the first-half-vs-second-half trend, so you
can watch training improve from inside the app. A premium HD pass keeps it
crisp: film-grain overlay, vignette, ligatures off, themed scrollbars.

The comparison flow is a simple 3-step guide shown under every answer set:
**1 Ask → 2 Pick best → 3 Compare**. "Pick best answer" asks the model to crown
the winner (BEST badge + a plain-language reason), then "Compare side by side"
opens a two-column view — **Normal answer** (the plain one-shot) on the left
and **Worked-on answer** (the one Nori judged best) on the right — both fully
scrollable, with the AI verdict pinned at the top. Type, press Enter or click
**↑**, and cards stream in over SSE. The page lives in `dse/chat_page.html`,
so you can restyle it without touching any Python (edits are picked up on the
next reload — no server restart).

Terminal alternative (same saved sessions):

```bash
python -m dse.ask                                   # interactive REPL
python -m dse.ask --question "why is the sky blue?" # one-shot
python -m dse.ask --stats                           # is it getting better?
python -m dse.ask --audit sessions/ask_2026-08-05.jsonl  # replay a session
```

In the REPL you type a question; `react`, `best_of_n`, `reflexion`, and
`self_refine` each answer it, an LLM judge grades every answer 0-10 with
feedback, and you rate each 1-5. `--stats` shows per-strategy averages and a
first-half-vs-second-half trend — the "is it getting better" answer in one
screen. Judge defaults to the fast answer model; pass `--judge-model
deepseek-v4-pro` for a stronger grader.

## Fancy terminal output

All commands print color-coded tables, section headers, and live progress bars
(`█▓▒░` with elapsed/ETA). On a real terminal you get the full Unicode/color
experience; when piped to a file or non-UTF-8 console, output automatically
falls back to clean ASCII so nothing is ever mojibaked. Set `NO_COLOR=1` (or
`NO_UNICODE=1`) to force plain output.

## Package layout

```text
dse/
├── config.py          # EngineConfig, feature flags, experiment registry
├── llm.py             # LLM protocol + calibrated MockLLM
├── providers.py       # OpenAI-compatible client (DeepSeek/Ollama/OpenAI/GitHub)
├── environment.py     # Task model, deterministic catalog, ACI surface
├── verifier.py        # exact / test / LLM-judge / self-consistency + aggregate
├── router.py          # confidence-based model escalation
├── memory.py          # rolling window + summary + episodic reflections
├── feedback.py        # compiler feedback loop (build→test→lint→sec→perf)
├── agent.py           # Agent protocol + BaseAgent plumbing
├── strategies/        # react, best_of_n, reflexion, self_refine, tree_search,
│                      # escalating, escalating_per_step, adaptive
├── orchestration.py   # MoA-lite (flag: multi_agent)
├── ask.py             # interactive Q&A + sessions + stats (terminal)
├── chat.py            # local Grok-style chat UI (buttons, SSE, zero deps)
├── ui.py              # colors, tables, progress bars (TTY-aware)
├── freeform.py        # PromptTask + LLM judge for free-form Q&A
├── factory.py         # one-call stack assembly (config→agents)
└── benchmarks/        # harness + multi-seed + CLI + JSON records
```

## Adding a strategy

1. Subclass `BaseAgent` (or implement the `Agent` protocol) in
   `dse/strategies/`.
2. Register it in `dse/factory.build_default_agents`.
3. Add a hypothesis to `ENGINE_EXPERIMENTS` in `dse/config.py` if it is
   experimental (with a rollback condition).
4. Benchmark it against the baseline — never claim an improvement without the
   harness numbers.

## Feature flags

All experimental techniques are gated:

| Flag | Default | Evidence |
|---|---|---|
| `adaptive_compute` | on | Snell et al. 2024 (see BENCHMARKS §3.4 for the honest negative result) |
| `self_consistency` | on | Wang et al. 2022 |
| `search_reflection` | on | Zhou et al. 2023 (LATS) |
| `llm_judge` | off | experimental (noisy value function) |
| `multi_agent` | off | experimental (cost-heavy; see BENCHMARKS §3.5) |

## Honesty rules (from the brief)

- No "better" / "smarter" / "more accurate" without harness numbers.
- Every reported number ships with hardware, models, seed, n, and CIs.
- Known limitations are listed in [LIMITATIONS.md](LIMITATIONS.md).
- Experiments that do not beat the baseline are removed or flagged, never
  silently kept.
