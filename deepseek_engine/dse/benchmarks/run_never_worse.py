"""CLI: never-worse benchmark — does the guarded pipeline really never ship an
answer worse than its best candidate?

Usage:
    python -m dse.benchmarks.run_never_worse --provider mock --suite all --n-tasks 48
    python -m dse.benchmarks.run_never_worse --provider deepseek --suite hard-real

For each task it:
  1. runs react / reflexion / self_refine and scores every answer with the
     calibrated judge (0-10),
  2. applies the SELECTION guard to choose the winner (never ship a candidate
     the judge scored meaningfully below the baseline),
  3. runs synthesis and applies the NO-REGRESSION guard (never ship a merge
     that is ungrounded in a candidate or scored below the winner),
  4. measures, by GROUND TRUTH, whether the final answer is never worse than
     the guarded winner and sometimes better than the baseline.

Output: per-stage success rates, never-worse / sometimes-better rates, guard
fire rates, the judge-metric guarantee (final score >= winner score - margin),
and paired McNemar tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from .. import ui
from ..env import load_env
from ..events import RunResult
from ..factory import build_stack
from ..freeform import FreeFormJudge, PromptTask
from ..guards import robust_score, selection_guard, synthesis_guard
from .harness import bootstrap_ci, mcnemar, wilcoxon_signed_rank

_SYNTH_PROMPT = (
    "You are a careful answer synthesizer. Below are several candidate "
    "answers to the same question. Combine the STRONGEST parts of all "
    "candidates into ONE final, polished, complete answer. Keep the best "
    "explanations, examples, and wording; drop anything weaker or "
    "redundant.\n\n"
    "QUESTION: {q}\n\n{answers}\n\n"
    "Reply with exactly two sections and nothing else:\n"
    "FINAL ANSWER:\n<the merged answer>\n"
    "PARTS:\n<one short paragraph: which candidate(s) contributed which parts>"
)

_JUDGE_PROMPT = (
    "You are a strict, calibrated answer grader. Reply with EXACTLY one line "
    "and nothing else: SCORE: <0-10>. Be harsh; use the full range.\n"
    "ANSWER:\n{text}"
)

# Open-ended questions (no ground truth): quality is the de-noised judge's
# score. This is the regime where a frontier model's answers genuinely differ
# and synthesis can add value — the honest place to show "sometimes better".
# 32 questions across science, tech, health, finance, career and society so
# the paired significance test has enough power (n=32, not a pilot n=8).
_FREE_QUESTIONS = [
    "Explain how quantum entanglement works in simple terms.",
    "What are the main pros and cons of remote work?",
    "Explain the difference between machine learning and deep learning.",
    "How would you explain recursion to a five-year-old?",
    "What are the key differences between HTTP/1.1 and HTTP/2?",
    "Compare solar power and wind power for powering a small town.",
    "Explain compound interest to a teenager who wants to start saving.",
    "What causes inflation, and what are its main effects on everyday people?",
    "What are the best strategies for building a habit that actually sticks?",
    "Explain the concept of a blockchain in plain language, and where it genuinely adds value.",
    "What are the trade-offs between renting and buying a home for a young professional?",
    "How does a search engine decide which results to show first?",
    "What makes a good apology, and why do most apologies fail?",
    "Explain the difference between a physical therapist and a chiropractor for back pain.",
    "What should someone consider before switching careers in their thirties?",
    "How do vaccines work, and why do some need boosters?",
    "What are the strongest arguments for and against nuclear power today?",
    "Explain how a neural network learns, without using math.",
    "What are the most important things to know before starting a small online business?",
    "How would you explain the concept of opportunity cost to someone who hates economics?",
    "What are the main differences between mediation and arbitration in a dispute?",
    "Why do some people find it hard to sleep, and what actually helps?",
    "Explain the pros and cons of open-source software versus proprietary software for a small company.",
    "How does inflation affect someone's personal savings, and what can they do about it?",
    "What is the best way to prepare for a job interview in a technical field?",
    "Explain the concept of a carbon footprint and whether individual actions matter.",
    "What are the key considerations when choosing between a traditional car and an electric car in 2026?",
    "How would you explain the stock market to a complete beginner?",
    "What are the trade-offs between eating organic and conventionally grown food?",
    "Explain the difference between a manager and a leader, with examples.",
    "What are the main causes of climate change, and what are the most effective responses?",
    "How would you design a personal budget that you can actually stick to?",
]


def _build_judge_llm(provider: str, judge_model: str | None, main_llm):
    """The LLM used to GRADE answers.

    By default this is the same model that wrote the answers (self-grading —
    the project's known circularity). When ``--judge-model`` names a different
    model, a separate provider client is built so grading is done by an
    independent model and the circularity is broken.
    """
    if not judge_model:
        return main_llm
    from ..providers import make_provider, provider_models

    models = provider_models(cheap_model=judge_model, expensive_model=judge_model)
    judge_llm = make_provider(provider, models=models)
    return judge_llm if judge_llm is not None else main_llm


def _score_text(llm, text: str) -> float | None:
    try:
        resp = llm.complete([{"role": "system", "content": _JUDGE_PROMPT.format(text=text or "")}],
                            model="expensive", max_tokens=40).text or ""
    except Exception:
        return None
    m = re.search(r"SCORE\s*:\s*(\d+(?:\.\d+)?)", resp, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _judge_candidate(judge, task, answer: str) -> float | None:
    v = judge.score(answer or "", PromptTask(getattr(task, "id", "?"), getattr(task, "prompt", "")))
    return v.details.get("judge_score")


def _make_robust_judge(llm, samples: int):
    """A judge whose score is the median of ``samples`` calls — de-noised."""
    return lambda text: robust_score(lambda t: _score_text(llm, t), text, samples=samples)


def _synthesize(llm, question: str, candidates: list[dict], winner_name: str) -> str:
    parts = []
    for i, c in enumerate(candidates, 1):
        mark = "WINNER" if c["strategy"] == winner_name else "candidate"
        parts.append(f"[{i}] strategy={c['strategy']} ({mark})\n"
                     f"{' '.join((c['answer'] or '').split())}")
    prompt = _SYNTH_PROMPT.format(q=question, answers="\n\n".join(parts))
    text = ""
    for _ in range(3):
        try:
            text = llm.complete([{"role": "system", "content": prompt}],
                                model="expensive", max_tokens=1200).text or ""
        except Exception:
            continue
        if text.strip():
            break
    m = re.search(r"FINAL ANSWER\s*:\s*(.*?)(?:\nPARTS\s*:|$)", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else text.strip())


def _verify(verifier, task, answer: str) -> bool:
    try:
        return bool(verifier.score(answer or "", task).passed)
    except Exception:
        return False


def run(provider: str, suite: str, n_tasks: int, seed: int, max_steps: int,
        judge_samples: int = 3, judge_model: str | None = None) -> dict:
    if suite == "free":
        return run_free(provider, judge_samples, n_tasks, judge_model)
    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=seed, n_tasks=n_tasks, max_steps=max_steps, provider=provider, suite=suite)
    by_name = {a.name: a for a in agents}
    wanted = [n for n in ("react", "reflexion", "self_refine") if n in by_name]
    tasks = list(catalog.values())[:n_tasks]
    judge_llm = _build_judge_llm(provider, judge_model, llm)
    judge = FreeFormJudge(judge_llm, model="expensive")
    robust = _make_robust_judge(judge_llm, judge_samples)

    rows = []
    t_start = time.time()
    for ti, task in enumerate(tasks, 1):
        cands: dict[str, dict] = {}
        for name in wanted:
            if hasattr(llm, "reset"):
                llm.reset(f"{seed}:{name}:{task.id}")
            rr = by_name[name].solve(task, budget)
            cands[name] = {
                "strategy": name,
                "answer": rr.answer or "",
                "correct": bool(rr.success),
                "judge_score": _judge_candidate(judge, task, rr.answer),
            }

        # ---- selection guard ----
        records = [{"strategy": k, "judge_score": v["judge_score"]} for k, v in cands.items()]
        scored = [r for r in records if r["judge_score"] is not None]
        arbiter_pick = (max(scored, key=lambda r: r["judge_score"])["strategy"] if scored
                        else "react")
        winner_name = selection_guard(records, arbiter_pick)
        sel_fired = winner_name != arbiter_pick
        win = cands[winner_name]

        # ---- synthesis + no-regression guard ----
        candidates_text = [c["answer"] for c in cands.values()]
        question = getattr(task, "prompt", "")
        synth = _synthesize(llm, question, list(cands.values()), winner_name)
        final, used_synth, guard_why = synthesis_guard(
            synth, candidates_text, win["answer"], judge=robust)
        synth_fired = not used_synth
        final_correct = _verify(verifier, task, final)
        # judge-metric: if we fell back, the final IS the winner (identical
        # answer — no new noisy judge call). If we shipped, score it with the
        # SAME robust judge the guard used.
        winner_judge = robust(win["answer"])
        final_judge = robust(final) if used_synth else winner_judge

        rows.append({
            "task": getattr(task, "id", ti),
            "baseline_correct": cands["react"]["correct"] if "react" in cands else None,
            "arbiter_correct": cands[arbiter_pick]["correct"] if arbiter_pick in cands else None,
            "winner_correct": win["correct"],
            "winner_strategy": winner_name,
            "final_correct": final_correct,
            "used_synth": used_synth,
            "selection_fired": sel_fired,
            "synthesis_fired": synth_fired,
            "winner_judge": winner_judge,
            "final_judge": final_judge,
            "guard": guard_why,
        })
        if ti % 5 == 0 or ti == len(tasks):
            print(f"  [{ti}/{len(tasks)}] {getattr(task,'id','')}")

    # ---- aggregate ----
    n = len(rows)
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else 0.0

    baseline_ok = [bool(r["baseline_correct"]) for r in rows if r["baseline_correct"] is not None]
    winner_ok = [bool(r["winner_correct"]) for r in rows]
    final_ok = [bool(r["final_correct"]) for r in rows]

    never_worse = sum(1 for w, f in zip(winner_ok, final_ok) if (not w) or f)
    sometimes_better = sum(1 for w, f in zip(winner_ok, final_ok) if (not w) and f)
    never_worse_rate = never_worse / n if n else 0.0
    # judge-metric guarantee: final judge score >= winner judge score - margin
    judge_guarantee = sum(
        1 for r in rows
        if r["final_judge"] is None or r["winner_judge"] is None
        or r["final_judge"] >= r["winner_judge"] - 0.5 - 1e-9)

    def res(strategy: str, ok_list) -> list[RunResult]:
        return [RunResult(task_id=f"t{i}", strategy=strategy, success=ok, answer="")
                for i, ok in enumerate(ok_list)]

    cmp_final_baseline = mcnemar(res("final", final_ok), res("react", baseline_ok))
    cmp_final_winner = mcnemar(res("final", final_ok), res("winner", winner_ok))
    ci = bootstrap_ci(res("final", final_ok), res("react", baseline_ok)) if len(baseline_ok) == n else None

    summary = {
        "n": n,
        "provider": provider,
        "suite": suite,
        "seed": seed,
        "baseline_success": round(rate("baseline_correct"), 4),
        "arbiter_winner_success": round(rate("arbiter_correct"), 4),
        "guarded_winner_success": round(rate("winner_correct"), 4),
        "final_success": round(rate("final_correct"), 4),
        "never_worse_rate": round(never_worse_rate, 4),
        "never_worse_count": never_worse,
        "sometimes_better": sometimes_better,
        "selection_guard_fired": sum(1 for r in rows if r["selection_fired"]),
        "synthesis_guard_fired": sum(1 for r in rows if r["synthesis_fired"]),
        "judge_metric_guarantee_held": judge_guarantee,
        "mcnemar_final_vs_baseline": {
            "b01": cmp_final_baseline.b01, "b10": cmp_final_baseline.b10,
            "p": round(cmp_final_baseline.p_value, 6),
        },
        "mcnemar_final_vs_winner": {
            "b01": cmp_final_winner.b01, "b10": cmp_final_winner.b10,
            "p": round(cmp_final_winner.p_value, 6),
        },
        "bootstrap_95_ci_final_minus_baseline": (
            [round(ci[0], 4), round(ci[1], 4)] if ci else None),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    return {"summary": summary, "rows": rows}


def run_free(provider: str, judge_samples: int, n_questions: int,
             judge_model: str | None = None) -> dict:
    """Never-worse on open-ended questions: no ground truth exists, so the
    de-noised judge's score IS the quality metric. The guards still guarantee
    final >= winner (by the same robust judge), and synthesis has room to be
    genuinely better than the baseline.

    When ``judge_model`` is set, grading is done by an INDEPENDENT model (not
    the model that wrote the answers), breaking the self-grading circularity.
    """
    from ..ask import build_ask_pipeline

    llm, _models, _judge, strategies, budget = build_ask_pipeline(provider, None, None)
    by_name = {s.name: s for s in strategies}
    wanted = [n for n in ("react", "reflexion", "self_refine") if n in by_name]
    judge_llm = _build_judge_llm(provider, judge_model, llm)
    robust = _make_robust_judge(judge_llm, judge_samples)
    questions = _FREE_QUESTIONS[:n_questions]
    rows = []
    t_start = time.time()
    for qi, question in enumerate(questions, 1):
        task = PromptTask(id=f"free-{qi}", prompt=question)
        cands = {}
        for name in wanted:
            rr = by_name[name].solve(task, budget)
            meta = getattr(rr, "verifier_meta", {}) or {}
            cands[name] = {"strategy": name, "answer": rr.answer or "",
                           "judge_score": meta.get("judge_score")}
        records = [{"strategy": k, "judge_score": v["judge_score"]}
                   for k, v in cands.items()]
        scored = [r for r in records if r["judge_score"] is not None]
        arbiter_pick = (max(scored, key=lambda r: r["judge_score"])["strategy"]
                        if scored else "react")
        winner_name = selection_guard(records, arbiter_pick)
        sel_fired = winner_name != arbiter_pick
        win = cands[winner_name]
        candidates_text = [c["answer"] for c in cands.values()]
        synth = _synthesize(llm, question, list(cands.values()), winner_name)
        final, used_synth, guard_why = synthesis_guard(
            synth, candidates_text, win["answer"], judge=robust)
        synth_fired = not used_synth
        baseline_quality = robust(cands["react"]["answer"])
        winner_quality = robust(win["answer"])
        final_quality = robust(final) if used_synth else winner_quality
        rows.append({
            "question": question,
            "winner_strategy": winner_name,
            "used_synth": used_synth,
            "selection_fired": sel_fired,
            "synthesis_fired": synth_fired,
            "baseline_quality": baseline_quality,
            "winner_quality": winner_quality,
            "final_quality": final_quality,
            "guard": guard_why,
        })
        print(f"  [{qi}/{len(questions)}] {question[:42]}")
    n = len(rows)

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else 0.0

    never_worse = sum(1 for r in rows if r["final_quality"] is None
                      or r["winner_quality"] is None
                      or r["final_quality"] >= r["winner_quality"] - 0.5 - 1e-9)
    sometimes_better = sum(1 for r in rows
                           if r["final_quality"] is not None
                           and r["baseline_quality"] is not None
                           and r["final_quality"] > r["baseline_quality"] + 0.5)

    # paired significance: the scores are continuous, so McNemar can't see a
    # 6 -> 8 improvement — Wilcoxon signed-rank can.
    def _wilcox(pairs: list[tuple[float, float]]) -> dict:
        if not pairs:
            return {"w": None, "p": 1.0, "n": 0}
        w, p = wilcoxon_signed_rank([p[0] for p in pairs], [p[1] for p in pairs])
        return {"w": round(w, 1), "p": round(p, 6), "n": len(pairs)}

    fv_b = [(r["final_quality"], r["baseline_quality"]) for r in rows
            if r["final_quality"] is not None and r["baseline_quality"] is not None]
    fv_w = [(r["final_quality"], r["winner_quality"]) for r in rows
            if r["final_quality"] is not None and r["winner_quality"] is not None]
    wv_b = [(r["winner_quality"], r["baseline_quality"]) for r in rows
            if r["winner_quality"] is not None and r["baseline_quality"] is not None]

    summary = {
        "n": n,
        "provider": provider,
        "suite": "free",
        "judge_samples": judge_samples,
        "judge_model": judge_model or "self (same model as answers)",
        "baseline_quality_avg": round(avg("baseline_quality"), 3),
        "winner_quality_avg": round(avg("winner_quality"), 3),
        "final_quality_avg": round(avg("final_quality"), 3),
        "never_worse_judge": round(never_worse / n, 3) if n else 0.0,
        "never_worse_count": never_worse,
        "sometimes_better": sometimes_better,
        "wilcoxon_final_vs_baseline": _wilcox(fv_b),
        "wilcoxon_final_vs_winner": _wilcox(fv_w),
        "wilcoxon_winner_vs_baseline": _wilcox(wv_b),
        "selection_guard_fired": sum(1 for r in rows if r["selection_fired"]),
        "synthesis_guard_fired": sum(1 for r in rows if r["synthesis_fired"]),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="nori never-worse benchmark")
    parser.add_argument("--provider", choices=["mock", "deepseek", "ollama", "openai", "github"],
                        default="mock")
    parser.add_argument("--suite", choices=["all", "easy", "hard", "real", "hard-real", "hard-tuned", "code", "fail", "free"],
                        default="all")
    parser.add_argument("--n-tasks", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--judge-samples", type=int, default=3,
                        help="median-of-N judge calls to de-noise scores (1 = single call)")
    parser.add_argument("--judge-model", default=None,
                        help="grade answers with a DIFFERENT model (e.g. deepseek-v4-pro) "
                             "to break self-grading circularity; default = same model as answers")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    load_env()
    if args.provider in ("deepseek", "openai", "github") and not os.environ.get("DSE_PROVIDER_KEY"):
        parser.error("provider needs a token. Add DSE_PROVIDER_KEY to the .env file "
                     "(see .env.example) or set the environment variable.")

    result = run(args.provider, args.suite, args.n_tasks, args.seed, args.max_steps,
                 judge_samples=args.judge_samples, judge_model=args.judge_model)
    s = result["summary"]
    # persist the result IMMEDIATELY (before printing) so a late crash/kill
    # can never lose a multi-hour run — learned the hard way when the
    # independent-judge run exited 1 after printing but before writing.
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2)
            print(f"OK wrote {args.out}")
        except Exception as exc:  # pragma: no cover - filesystem edge
            print(f"! could not write {args.out}: {exc}")
    print()
    ui.section("never-worse benchmark (n=%d, provider=%s, suite=%s, judge-samples=%d%s)" % (
        s["n"], s["provider"], s["suite"], args.judge_samples,
        f", judge-model={args.judge_model}" if args.judge_model else ""))
    if args.suite == "free":
        fvb = s["wilcoxon_final_vs_baseline"]
        fvw = s["wilcoxon_final_vs_winner"]
        print(ui.table(
            ["metric", "value"],
            [
                ["judge model", s["judge_model"]],
                ["baseline (react) quality (robust judge)", f"{s['baseline_quality_avg']:.2f}/10"],
                ["guarded winner quality", f"{s['winner_quality_avg']:.2f}/10"],
                ["FINAL quality (with guards)", f"{s['final_quality_avg']:.2f}/10"],
                ["never-worse by robust judge (final >= winner)", f"{s['never_worse_judge']:.3f}  ({s['never_worse_count']}/{s['n']})"],
                ["sometimes better (final > baseline + 0.5)", str(s["sometimes_better"])],
                ["final vs baseline paired Wilcoxon", f"W={fvb['w']} p={fvb['p']} (n={fvb['n']})"],
                ["final vs winner paired Wilcoxon", f"W={fvw['w']} p={fvw['p']} (n={fvw['n']})"],
                ["selection guard fired", str(s["selection_guard_fired"])],
                ["synthesis guard fired (fell back)", str(s["synthesis_guard_fired"])],
                ["elapsed", f"{s['elapsed_s']}s"],
            ],
            header_style=ui.bold,
        ))
    else:
        print(ui.table(
            ["metric", "value"],
            [
                ["baseline (react) success", f"{s['baseline_success']:.3f}"],
                ["arbiter winner success (unguarded)", f"{s['arbiter_winner_success']:.3f}"],
                ["guarded winner success", f"{s['guarded_winner_success']:.3f}"],
                ["FINAL success (with guards)", f"{s['final_success']:.3f}"],
                ["never-worse rate (final >= winner)", f"{s['never_worse_rate']:.3f}  ({s['never_worse_count']}/{s['n']})"],
                ["sometimes better (final fixes wrong winner)", str(s["sometimes_better"])],
                ["selection guard fired", str(s["selection_guard_fired"])],
                ["synthesis guard fired (fell back)", str(s["synthesis_guard_fired"])],
                ["judge-metric guarantee held", f"{s['judge_metric_guarantee_held']}/{s['n']}"],
                ["final vs baseline McNemar p", f"{s['mcnemar_final_vs_baseline']['p']} (b01={s['mcnemar_final_vs_baseline']['b01']}, b10={s['mcnemar_final_vs_baseline']['b10']})"],
                ["final vs winner McNemar p", f"{s['mcnemar_final_vs_winner']['p']} (b01={s['mcnemar_final_vs_winner']['b01']}, b10={s['mcnemar_final_vs_winner']['b10']})"],
                ["bootstrap 95% CI (final - baseline)", str(s.get("bootstrap_95_ci_final_minus_baseline"))],
                ["elapsed", f"{s['elapsed_s']}s"],
            ],
            header_style=ui.bold,
        ))


if __name__ == "__main__":
    main()
