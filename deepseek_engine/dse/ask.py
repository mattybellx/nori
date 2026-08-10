"""Interactive Q&A against the engine's strategies.

Ask a question; several strategies answer it (graded by an LLM judge on the
expensive tier); you rate each answer; everything is saved to ``sessions/`` so
you can audit later and track whether answers are getting better over time.

Usage:
    python -m dse.ask                                    # interactive REPL
    python -m dse.ask --question "why is the sky blue?"  # one-shot (scriptable)
    python -m dse.ask --stats                            # improvement over time
    python -m dse.ask --audit sessions/ask_2026-08-05.jsonl

Requires a real provider (deepseek / ollama / openai / github). Put your token
in ``.env`` (DSE_PROVIDER_KEY) — see README.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

from . import ui
from .agent import BaseAgent, Budget
from .config import EngineConfig, default_flags, provider_models
from .env import load_env
from .freeform import FreeFormJudge, PromptTask
from .providers import make_provider
from .strategies import BestOfNAgent, ReactAgent, ReflexionAgent, SelfRefineAgent
from .telemetry import estimate_cost

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"

DEFAULT_STRATEGIES = ("react", "best_of_n", "reflexion", "self_refine")


# ---------------------------------------------------------------------------
# Session storage (JSONL: one record per question × strategy)
# ---------------------------------------------------------------------------
def _record_path() -> Path:
    SESSIONS_DIR.mkdir(exist_ok=True)
    return SESSIONS_DIR / f"ask_{datetime.now().date().isoformat()}.jsonl"


def save_records(records: list[dict]) -> Path:
    path = _record_path()
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_records(glob: str = "ask_*.jsonl") -> list[dict]:
    records: list[dict] = []
    for path in sorted(SESSIONS_DIR.glob(glob)):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def recent_questions(limit: int = 50) -> list[str]:
    """Distinct questions, most recent first (for the chat UI sidebar)."""
    seen: set[str] = set()
    result: list[str] = []
    for record in reversed(load_records()):
        question = record.get("question")
        if question and question not in seen:
            seen.add(question)
            result.append(question)
            if len(result) >= limit:
                break
    return result


def update_record(question: str, strategy: str, **fields) -> int:
    """Update rating/comparison fields on matching saved records (by question
    text + strategy). Returns the number of records updated. Used by the chat
    UI's buttons so audits persist to the same session files."""
    updated = 0
    for path in sorted(SESSIONS_DIR.glob("ask_*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = False
        for record in records:
            if record.get("question") == question and record.get("strategy") == strategy:
                record.update(fields)
                changed = True
                updated += 1
        if changed:
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                encoding="utf-8",
            )
    return updated


# ---------------------------------------------------------------------------
# Building the ask pipeline
# ---------------------------------------------------------------------------
def build_ask_pipeline(provider: str, model: str | None, judge_model: str | None):
    load_env()
    model = model or os.environ.get("DSE_MODEL_CHEAP", "deepseek-v4-flash")
    # judge defaults to the fast answer model unless explicitly overridden
    # (deepseek-reasoner is slower; pass --judge-model deepseek-reasoner for
    # a stronger grader)
    judge_model = judge_model or model
    models = provider_models(cheap_model=model, expensive_model=judge_model)
    llm = make_provider(provider, models=models)
    if llm is None:
        raise ValueError("ask requires a real provider (deepseek/ollama/...); got 'mock'")
    config = EngineConfig(seed=0, flags=default_flags())
    judge = FreeFormJudge(llm, model="expensive")
    strategies = [
        ReactAgent(llm, judge, config, models),
        BestOfNAgent(llm, judge, config, models, n=3),
        ReflexionAgent(llm, judge, config, models),
        SelfRefineAgent(llm, judge, config, models),
    ]
    budget = Budget(max_trials=config.max_trials, max_search_nodes=config.max_search_nodes)
    return llm, models, judge, strategies, budget


def ask_question(strategies, models, question: str, qid: str, budget: Budget,
                 progress=None) -> list[dict]:
    task = PromptTask(id=qid, prompt=question)
    records: list[dict] = []
    for index, strategy in enumerate(strategies):
        if progress is not None:
            progress(index, f"asking {strategy.name}...")
        result = strategy.solve(task, budget)
        meta = getattr(result, "verifier_meta", {}) or {}
        jscore = meta.get("judge_score")
        records.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "question": question,
                "strategy": strategy.name,
                "answer": result.answer,
                "judge_score": round(jscore, 1) if jscore is not None else None,
                "passed": result.success,
                "latency_s": round(result.latency_s, 2),
                "tokens": result.tokens_total,
                "cost_usd": round(estimate_cost(result, models), 6),
                "rating": None,
            }
        )
        if progress is not None:
            progress(index + 1, f"{strategy.name} done")
    return records


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def _score_style(score: float | None):
    if score is None:
        return ui.dim("n/a")
    if score >= 7:
        return ui.bright_green(f"{score:.1f}")
    if score >= 4:
        return ui.bright_yellow(f"{score:.1f}")
    return ui.bright_red(f"{score:.1f}")


def _score_style_fn(score: float | None):
    """Style *function* for table cells (color decided by value)."""
    if score is None:
        return ui.dim
    if score >= 7:
        return ui.bright_green
    if score >= 4:
        return ui.bright_yellow
    return ui.bright_red


def display_answer(record: dict, index: int) -> None:
    score = record.get("judge_score")
    rating = record.get("rating")
    rating_text = ui.ok(f"you: {rating}/5") if rating else ui.dim("unrated")
    header = (
        f"[{index}] {record['strategy']:<12} "
        f"{ui.bold('judge')} {_score_style(score)}/10   "
        f"{record['latency_s']:>5.1f}s  {record['tokens']:>4} tok  "
        f"${record['cost_usd']:.6f}  {rating_text}"
    )
    body = record["answer"]  # full answer, never truncated
    print(ui.panel(header, body, title_color=ui.cyan))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def stats_json(records: list[dict]) -> dict:
    """Structured stats for the chat UI Insights panel (mirrors print_stats)."""
    strategies = sorted({r.get("strategy") for r in records})
    per = []
    for name in strategies:
        rs = [r for r in records if r.get("strategy") == name]
        scored = [r["judge_score"] for r in rs if r.get("judge_score") is not None]
        rated = [r["rating"] for r in rs if r.get("rating") is not None]
        per.append({
            "strategy": name,
            "n": len(rs),
            "judge_avg": round(statistics.fmean(scored), 1) if scored else None,
            "rating_avg": round(statistics.fmean(rated), 1) if rated else None,
            "passed": sum(bool(r.get("passed")) for r in rs),
            "lat_avg": round(statistics.fmean(r["latency_s"] for r in rs), 1),
            "tok_avg": round(statistics.fmean(r["tokens"] for r in rs)),
            "cost_avg": statistics.fmean(r["cost_usd"] for r in rs),
        })
    trend = None
    scored_all = [r["judge_score"] for r in records if r.get("judge_score") is not None]
    if len(scored_all) >= 4:
        half = len(scored_all) // 2
        first = statistics.fmean(scored_all[:half])
        second = statistics.fmean(scored_all[half:])
        trend = {"first": round(first, 2), "second": round(second, 2), "delta": round(second - first, 2)}
    return {
        "n": len(records),
        "days": sum(1 for _ in SESSIONS_DIR.glob("ask_*.jsonl")),
        "strategies": per,
        "trend": trend,
    }


def print_stats(records: list[dict]) -> None:
    ui.section("Ask history — is it getting better?")
    if not records:
        print(ui.warn("no sessions yet — run `python -m dse.ask` to ask something"))
        return
    print(ui.dim(f"{len(records)} answer records across {sum(1 for _ in SESSIONS_DIR.glob('ask_*.jsonl'))} day(s)\n"))

    strategies = sorted({r["strategy"] for r in records})
    headers = ["strategy", "n", "judge_avg", "rating_avg", "passed", "lat_avg", "tok_avg", "cost_avg"]
    rows, styles = [], []
    for name in strategies:
        rs = [r for r in records if r["strategy"] == name]
        scored = [r["judge_score"] for r in rs if r.get("judge_score") is not None]
        rated = [r["rating"] for r in rs if r.get("rating") is not None]
        rows.append([
            name, len(rs),
            f"{statistics.fmean(scored):.1f}" if scored else "—",
            f"{statistics.fmean(rated):.1f}" if rated else "—",
            f"{sum(r['passed'] for r in rs)}/{len(rs)}",
            f"{statistics.fmean(r['latency_s'] for r in rs):.1f}s",
            f"{statistics.fmean(r['tokens'] for r in rs):.0f}",
            f"${statistics.fmean(r['cost_usd'] for r in rs):.6f}",
        ])
        styles.append([
            ui.bold, None, _score_style_fn(statistics.fmean(scored)) if scored else ui.dim,
            ui.green if rated else ui.dim, None, None, None, None,
        ])
    print(ui.table(headers, rows, header_style=ui.compose(ui.bold, ui.cyan), cell_styles=styles))

    # trend: average judge score over time (first half vs second half)
    scored = [r["judge_score"] for r in records if r.get("judge_score") is not None]
    if len(scored) >= 4:
        half = len(scored) // 2
        first = statistics.fmean(scored[:half])
        second = statistics.fmean(scored[half:])
        delta = second - first
        if delta > 0.1:
            arrow = ui.ok(f"{ui.G.up} +{delta:.2f}")
        elif delta < -0.1:
            arrow = ui.fail(f"{ui.G.down} {delta:.2f}")
        else:
            arrow = ui.dim(f"{ui.G.dot} {delta:.2f}")
        print(f"\n{ui.bold('Trend:')} first half avg {first:.2f} -> second half avg {second:.2f}  {arrow}")


# ---------------------------------------------------------------------------
# Audit (replay a session)
# ---------------------------------------------------------------------------
def audit(path: Path) -> None:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    ui.section(f"Audit — {path.name}")
    last_question = None
    for index, record in enumerate(records, 1):
        if record["question"] != last_question:
            print(ui.bold("\nQ: ") + record["question"])
            last_question = record["question"]
        display_answer(record, index)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="nori - ask & audit")
    parser.add_argument("--question", help="ask a single question and exit")
    parser.add_argument("--provider", default="deepseek",
                        choices=["deepseek", "ollama", "openai", "github"])
    parser.add_argument("--model", help="answer model (cheap tier)")
    parser.add_argument("--judge-model", help="judge model (expensive tier)")
    parser.add_argument("--stats", action="store_true", help="show improvement stats")
    parser.add_argument("--audit", metavar="FILE", help="replay a session file")
    args = parser.parse_args(argv)

    if args.stats:
        print_stats(load_records())
        return 0
    if args.audit:
        audit(Path(args.audit))
        return 0

    llm, models, judge, strategies, budget = build_ask_pipeline(
        args.provider, args.model, args.judge_model
    )

    if args.question:
        progress_update, progress_finish = ui.progress(len(strategies), "asking", width=20)
        records = ask_question(
            strategies, models, args.question, "q-1", budget, progress=progress_update
        )
        progress_finish()
        ui.section("Answers")
        for i, record in enumerate(records, 1):
            display_answer(record, i)
        path = save_records(records)
        print(ui.ok(f"saved {len(records)} records -> {path.name}"))
        return 0

    # -- interactive REPL ----------------------------------------------------
    ui.section("nori — Ask  (type 'quit', 'history', or 'help')")
    print(ui.dim("provider=") + ui.bold(args.provider) +
          ui.dim("  model=") + ui.bold(models["cheap"].provider_model) +
          ui.dim("  judge=") + ui.bold(models["expensive"].provider_model))
    qid = 0
    while True:
        question = ui.user_input(ui.bright_cyan("\nYou> "))
        if question in {"quit", "exit", "q"}:
            print(ui.dim("bye"))
            return 0
        if question in {"help", "h"}:
            print(ui.dim("Ask anything; each strategy answers, a judge grades them, "
                         "then you rate 1-5. 'history' shows past Q&A, 'quit' exits."))
            continue
        if question in {"history", "hist"}:
            print_stats(load_records())
            continue
        if not question:
            continue
        qid += 1
        ui.section(f"Q{qid}: {question[:70]}", color=ui.bright_cyan)
        progress_update, progress_finish = ui.progress(len(strategies), "asking", width=20)
        records = ask_question(
            strategies, models, question, f"q{qid}", budget, progress=progress_update
        )
        progress_finish()
        for i, record in enumerate(records, 1):
            display_answer(record, i)
        answer = ui.user_input("\n" + ui.bold("Rate each answer 1-5 (comma-separated, Enter=skip): "))
        ratings = [int(x) for x in answer.replace(";", ",").split(",") if x.strip().isdigit()]
        for i, record in enumerate(records):
            if i < len(ratings):
                record["rating"] = max(1, min(5, ratings[i]))
        path = save_records(records)
        print(ui.ok(f"saved → {path.name}"))


if __name__ == "__main__":
    raise SystemExit(main())
