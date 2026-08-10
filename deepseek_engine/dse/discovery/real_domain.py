"""The real-domain frontier: take the discovery laboratory to the REAL
DeepSeek model on free-form questions, evaluated with the INDEPENDENT judge.

This is the "next chapter" — testing the research hypothesis where it
actually matters: does a discovered architecture (e.g. best_of_N_ify = N
drafts + judge-selected) beat the single-shot baseline on open-ended
questions, under a DIFFERENT model's grading (breaking the self-grading
circularity)?

Reuses the exact machinery we built and validated:
- free-form strategies from ``build_ask_pipeline`` (real provider)
- the robust median-N judge + question-aware scoring + the preference judge
  (from ``run_never_worse``)
- groundedness weighting (``guards.grounded_score``) and the sign test
  (``harness.sign_test``)

Honest protocol (spec §10/§11/§47):
- the architecture's INTERNAL selection may use the self judge (flash),
- but every CLAIM is graded by the independent judge (pro) with median-N,
- and preference comparisons use the relative A/B judge + sign test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

from ..benchmarks.harness import sign_test
from ..benchmarks.run_never_worse import (
    _FREE_QUESTIONS,
    _build_judge_llm,
    _make_robust_judge,
    _preference_judge,
)
from ..env import load_env
from ..freeform import FreeFormJudge, PromptTask
from ..guards import grounded_score
from .compiler import compile_graph
from .executor import ArchExecutor
from .graph import ArchGraph
from .primitives import ExecutionContext


def build_freeform_context(provider: str = "deepseek",
                           budget=None) -> ExecutionContext:
    """An ExecutionContext wired for the real free-form domain: the ask
    pipeline's LLM + strategies, and the FreeFormJudge as the verifier."""
    from ..ask import build_ask_pipeline
    llm, models, _judge, strategies, budget = build_ask_pipeline(provider, None, None)
    verifier = FreeFormJudge(llm, model="expensive")
    return ExecutionContext(
        llm=llm, verifier=verifier, models=models,
        agents={s.name: s for s in strategies}, budget=budget,
        config=None,
    )


def run_architectures_real(graphs: list[ArchGraph], questions: list[str],
                           ctx: ExecutionContext) -> dict[str, list[str]]:
    """Run each architecture on each question; return {arch: [answer...]}."""
    ex = ArchExecutor()
    out: dict[str, list[str]] = {}
    for graph in graphs:
        compiled = compile_graph(graph)
        answers: list[str] = []
        for qi, question in enumerate(questions, 1):
            task = PromptTask(id=f"real-{graph.name}-{qi}", prompt=question)
            try:
                rec = ex.run(compiled, task, context=ctx)
                answers.append(rec.answer or "")
            except Exception as exc:  # record, never silently drop
                answers.append(f"<error: {exc!r}>")
            print(f"  [{graph.name}] q{qi}/{len(questions)} answered ({len(answers[-1])} ch)")
        out[graph.name] = answers
    return out


def score_real(answers_by_arch: dict[str, list[str]], questions: list[str],
               provider: str, judge_model: str | None, samples: int = 3,
               main_llm=None) -> dict[str, list[float | None]]:
    """Robust median-N judge score per (arch, question) by the INDEPENDENT
    judge (question-aware)."""
    judge_llm = _build_judge_llm(provider, judge_model, main_llm)
    scores: dict[str, list[float | None]] = {}
    for name, answers in answers_by_arch.items():
        row: list[float | None] = []
        for qi, (question, answer) in enumerate(zip(questions, answers), 1):
            robust = _make_robust_judge(judge_llm, samples, question)
            score = robust(answer)
            row.append(round(score, 3) if score is not None else None)
            print(f"  [score:{name}] q{qi} = {score}")
        scores[name] = row
    return scores


def preference_real(answers_by_arch: dict[str, list[str]], questions: list[str],
                    baseline_name: str, provider: str, judge_model: str | None,
                    main_llm=None) -> dict[str, list[str]]:
    """Relative A/B judge: for each arch vs the baseline, per-question
    'BETTER: A|B|TIE' (A = the arch, B = the baseline)."""
    judge_llm = _build_judge_llm(provider, judge_model, main_llm)
    base = answers_by_arch[baseline_name]
    out: dict[str, list[str]] = {}
    for name, answers in answers_by_arch.items():
        if name == baseline_name:
            continue
        row: list[str] = []
        for qi, (question, arch_a, base_b) in enumerate(zip(questions, answers, base), 1):
            pick = _preference_judge(judge_llm, question, arch_a, base_b)
            row.append(pick)
            print(f"  [pref:{name}] q{qi} -> {pick}")
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class RealDomainReport:
    provider: str
    judge_model: str | None
    n_questions: int
    judge_samples: int
    baseline: str
    questions: list[str] = field(default_factory=list)
    answers: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, list[float | None]] = field(default_factory=dict)
    preferences: dict[str, list[str]] = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "judge_model": self.judge_model,
            "n_questions": self.n_questions,
            "judge_samples": self.judge_samples,
            "baseline": self.baseline,
            "questions": self.questions,
            "answers": self.answers,
            "scores": self.scores,
            "preferences": self.preferences,
            "summary": self.summary,
            "elapsed_s": round(self.elapsed_s, 1),
        }


def _mean(vals):
    xs = [v for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _pref_stats(prefs: list[str]) -> dict:
    favor = sum(1 for p in prefs if p == "A")
    against = sum(1 for p in prefs if p == "B")
    ties = sum(1 for p in prefs if p == "TIE")
    total = favor + against
    return {
        "favor_arch": favor, "favor_baseline": against, "ties": ties,
        "win_rate": round(favor / total, 3) if total else None,
        "sign_test_p": round(sign_test(favor, total), 6) if total else None,
    }


def build_summary(report: RealDomainReport) -> dict:
    base_name = report.baseline
    base_scores = report.scores.get(base_name, [])
    summary: dict = {}
    for name, scores in report.scores.items():
        if name == base_name:
            continue
        entry = {
            "arch": name,
            "mean_judge_score": round(_mean(scores), 3) if _mean(scores) is not None else None,
            "baseline_mean": round(_mean(base_scores), 3) if _mean(base_scores) is not None else None,
            "delta": (round(_mean(scores) - _mean(base_scores), 3)
                      if _mean(scores) is not None and _mean(base_scores) is not None else None),
            "never_worse_by_judge": sum(
                1 for s, b in zip(scores, base_scores)
                if s is not None and b is not None and s >= b - 0.5 - 1e-9),
            "n": sum(1 for s in scores if s is not None),
            # substance-weighted quality vs the baseline answers (§12)
            "grounded_avg": round(
                sum(grounded_score(a, [b], js) for a, b, js in zip(
                    report.answers.get(name, []), report.answers.get(base_name, []),
                    scores) if js is not None) /
                max(sum(1 for js in scores if js is not None), 1), 3),
        }
        prefs = report.preferences.get(name)
        if prefs:
            entry["preference"] = _pref_stats(prefs)
        summary[name] = entry
    return summary


def run_real_experiment(
    pool_graphs: list[ArchGraph],
    questions: list[str] | None = None,
    provider: str = "deepseek",
    judge_model: str | None = "deepseek-v4-pro",
    samples: int = 3,
    baseline: str | None = None,
) -> RealDomainReport:
    """The real-domain experiment: run the pool on free-form questions, score
    everything with the independent judge, run preference comparisons, and
    build the honest summary."""
    questions = questions or _FREE_QUESTIONS
    t0 = time.time()
    ctx = build_freeform_context(provider)
    names = [g.name for g in pool_graphs]
    baseline = baseline or "react"
    if baseline not in names:
        raise ValueError(f"baseline {baseline!r} not in pool {names}")
    answers = run_architectures_real(pool_graphs, questions, ctx)
    scores = score_real(answers, questions, provider, judge_model, samples,
                        main_llm=ctx.llm)
    prefs = preference_real(answers, questions, baseline, provider, judge_model,
                            main_llm=ctx.llm)
    report = RealDomainReport(
        provider=provider, judge_model=judge_model,
        n_questions=len(questions), judge_samples=samples, baseline=baseline,
        questions=list(questions), answers=answers, scores=scores,
        preferences=prefs, elapsed_s=time.time() - t0,
    )
    report.summary = build_summary(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="real-domain discovery experiment: discovered architectures "
                    "on free-form questions, graded by the independent judge")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--judge-model", default="deepseek-v4-pro",
                        help="independent grader (breaks self-grading circularity)")
    parser.add_argument("--n", type=int, default=6, help="number of free-form questions")
    parser.add_argument("--samples", type=int, default=3, help="median-N judge calls")
    parser.add_argument("--out", default=None)
    parser.add_argument("--baseline", default="react")
    args = parser.parse_args()

    load_env()
    if not os.environ.get("DSE_PROVIDER_KEY"):
        parser.error("provider needs a token. Add DSE_PROVIDER_KEY to .env")

    from .baselines import react_graph
    from .mutations import best_of_n_ify

    questions = _FREE_QUESTIONS[: args.n]
    # the pool: single-shot baseline + the discovered-style best_of_N expansion
    baseline_graph = react_graph()
    candidate = best_of_n_ify(react_graph(), "s", n=3)
    candidate.name = "best_of_n_ify_react"
    pool = [baseline_graph, candidate]

    report = run_real_experiment(pool, questions, provider=args.provider,
                                 judge_model=args.judge_model,
                                 samples=args.samples, baseline=args.baseline)
    # persist FIRST (the data-loss lesson: write results before printing)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
            print(f"OK wrote {args.out}")
        except Exception as exc:
            print(f"! could not write {args.out}: {exc}")

    print("\n=== REAL-DOMAIN DISCOVERY EXPERIMENT ===")
    print(f"provider={report.provider}  judge={report.judge_model}  "
          f"n={report.n_questions}  baseline={report.baseline}")
    for name, s in report.summary.items():
        pref = s.get("preference", {})
        pref_str = (f"  pref {pref['favor_arch']}/{pref['favor_baseline']}/{pref['ties']} "
                    f"p={pref['sign_test_p']}") if pref else ""
        print(f"\n  {name}:")
        print(f"    mean judge: {s['mean_judge_score']} vs baseline {s['baseline_mean']} "
              f"(delta {s['delta']})")
        print(f"    grounded-avg {s['grounded_avg']}   never-worse {s['never_worse_by_judge']}/{s['n']}"
              f"{pref_str}")


if __name__ == "__main__":
    main()
