"""CLI: run the benchmark suite and record results.

Usage:
    python -m dse.benchmarks.run_benchmark --seed 0 --n-tasks 60 --max-steps 5
    python -m dse.benchmarks.run_benchmark --seed 0 --seeds 5          # multi-seed aggregation
    python -m dse.benchmarks.run_benchmark --flag llm_judge:true       # run an experiment flag
    python -m dse.benchmarks.run_benchmark --provider ollama           # real LLM (local)
    python -m dse.benchmarks.run_benchmark --provider deepseek --model-cheap deepseek-chat

Prints the strategy table and pairwise McNemar comparisons, then serializes the
full record (summaries + comparisons + metadata) to JSON for the audit trail.
"""

from __future__ import annotations

import argparse
import os
import time

from .. import ui
from ..config import default_flags
from ..env import load_env
from ..factory import build_default_agents, build_stack
from ..telemetry import summarize
from .harness import (
    bootstrap_ci,
    compare,
    run_benchmark,
    save_results,
    serialize_results,
)
from .tasks import split_by_difficulty


def _parse_flag(value: str) -> tuple[str, bool]:
    key, sep, raw = value.partition(":")
    if not sep or raw.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError(
            f"--flag must be KEY:true|false, got {value!r}"
        )
    return key, raw.lower() == "true"


def provider_note(provider: str, suite: str) -> str | None:
    """Human note printed before a provider-backed run (None = no note).

    The synthetic suite's step tokens are opaque to a real model, so real-model
    scores on it are meaningless — but the real suites (``--suite real`` /
    ``--suite hard-real``) are exactly what a real model should be scored on.
    """
    if provider == "mock":
        return None
    if suite in ("real", "hard-real", "hard-tuned", "code"):
        return ("Real-model run: natural-language tasks with deterministic "
                "checkers (no code execution). Each strategy makes live API "
                "calls; the table appears when all tasks finish.")
    return ("WARNING: the synthetic task suite is calibrated for the MockLLM "
            "(step tokens are opaque). Real-model scores on it are NOT "
            "meaningful — use --suite real for real-model evaluation.")


def _fancy_results(results, models) -> str:
    headers = ["strategy", "n", "success", "mean_lat", "med_lat", "p90_lat", "mean_tok", "cost_usd", "attempts"]
    rows, styles = [], []
    for name, runs in results.items():
        s = summarize(runs, models)
        style = ui.bright_green if s.success_rate >= 0.7 else ui.bright_yellow if s.success_rate >= 0.4 else ui.bright_red
        rows.append([
            name, s.n, f"{s.success_rate:.3f}", f"{s.mean_latency_s:.2f}",
            f"{s.median_latency_s:.2f}", f"{s.p90_latency_s:.2f}",
            f"{s.mean_tokens_total:.0f}", f"${s.mean_cost_usd:.6f}", f"{s.mean_attempts:.2f}",
        ])
        styles.append([ui.bold, None, style, None, None, None, None, None, None])
    return ui.table(headers, rows, header_style=ui.compose(ui.bold, ui.cyan), cell_styles=styles)


def _fancy_comparisons(comparisons) -> str:
    headers = ["A", "B", "b01", "b10", "chi2", "p", "sig"]
    rows, styles = [], []
    for c in comparisons:
        p_style = ui.bright_red if c.p_value < 0.001 else ui.bright_yellow if c.p_value < 0.05 else ui.dim
        rows.append([
            c.strategy_a, c.strategy_b, c.b01, c.b10,
            f"{c.statistic:.2f}", f"{c.p_value:.4f}", "YES" if c.significant else "no",
        ])
        styles.append([None, None, None, None, None, p_style, ui.bright_green if c.significant else ui.dim])
    return ui.table(headers, rows, header_style=ui.compose(ui.bold, ui.cyan), cell_styles=styles)


def main() -> None:
    parser = argparse.ArgumentParser(description="nori benchmark runner")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1,
                        help="number of seeds to aggregate (mock provider only)")
    parser.add_argument("--n-tasks", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--suite", choices=["all", "easy", "hard", "real", "hard-real", "hard-tuned", "code"], default="all",
                        help="real/hard-real/hard-tuned = natural-language suites answerable by a real LLM "
                             "(hard-real = multi-step; hard-tuned = strong-model targeted); "
                             "code = code-generation suite checked by EXECUTING the model's code (local, opt-in)")
    parser.add_argument("--moa", action="store_true",
                        help="include the (experimental) MoA agent")
    parser.add_argument("--flag", action="append", default=[], type=_parse_flag,
                        metavar="KEY:BOOL", help="override a feature flag (repeatable)")
    parser.add_argument("--provider", choices=["mock", "deepseek", "ollama", "openai", "github"],
                        default="mock", help="LLM backend (see dse/providers.py)")
    parser.add_argument("--adaptive-policy", choices=["score", "difficulty"],
                        default="difficulty", help="adaptive allocation policy (experiment control)")
    parser.add_argument("--model-cheap", default=None, help="cheap-tier provider model")
    parser.add_argument("--model-expensive", default=None, help="expensive-tier provider model")
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args()

    load_env()  # pick up DSE_PROVIDER_KEY etc. from .env before anything else
    if args.provider in ("deepseek", "openai", "github") and not os.environ.get("DSE_PROVIDER_KEY"):
        parser.error(
            f"provider {args.provider!r} needs a token. Add DSE_PROVIDER_KEY to "
            "the .env file at the project root (see .env.example) or set the "
            "DSE_PROVIDER_KEY environment variable."
        )

    if args.seeds > 1 and args.provider != "mock":
        parser.error("--seeds > 1 is only valid with --provider mock (real APIs "
                     "cannot be re-seeded per (agent, task))")

    flags = default_flags()
    for key, value in args.flag:
        flags[key] = value
    if args.moa:
        flags["multi_agent"] = True

    if args.model_cheap:
        os.environ["DSE_MODEL_CHEAP"] = args.model_cheap
    if args.model_expensive:
        os.environ["DSE_MODEL_EXPENSIVE"] = args.model_expensive

    config, catalog, models, llm, env, verifier, agents, budget = build_stack(
        seed=args.seed, n_tasks=args.n_tasks, max_steps=args.max_steps,
        min_steps=args.min_steps, flags=flags, include_moa=args.moa,
        provider=args.provider, adaptive_policy=args.adaptive_policy,
        suite=args.suite,
    )

    if args.suite == "easy":
        catalog, _ = split_by_difficulty(catalog)
    elif args.suite == "hard":
        _, catalog = split_by_difficulty(catalog)
    elif args.suite in ("real", "hard-real", "hard-tuned", "code"):
        print("real suite: natural-language tasks with deterministic checkers "
              "(no code execution); per-step strategies excluded.")
    if args.suite == "code":
        print("CODE SUITE: the model's generated code is EXECUTED locally "
              "under a timeout to check it (opt-in; not a security sandbox).")

    # experiment flags that are ON (beyond defaults) and non-default controls
    # become part of the output filename so different runs never overwrite
    experiment_on = [k for k in ("llm_judge", "multi_agent") if flags.get(k)]
    flag_suffix = ("_" + "_".join(experiment_on)) if experiment_on else ""
    if args.adaptive_policy != "difficulty":
        flag_suffix += f"_{args.adaptive_policy}"

    print(f"suite={args.suite} tasks={len(catalog)} seed={args.seed} "
          f"provider={args.provider} flags={[k for k, v in flags.items() if v]} "
          f"models={[m.name for m in models.values()]}")
    note = provider_note(args.provider, args.suite)
    if note:
        print(note)
    t0 = time.perf_counter()

    total = len(agents) * len(catalog) * max(1, args.seeds)
    progress_update, progress_finish = ui.progress(total, "benchmark", width=30)

    if args.seeds > 1:
        from ..llm import MockLLM
        from .harness import run_multi_seed

        seeds = tuple(range(args.seeds))

        def agent_factory(seed):
            # agents MUST be bound to the same LLM that gets reseeded per
            # (agent, task) — sharing agents across seeds would couple RNGs
            llm = MockLLM(catalog, models, seed=seed)
            agents_for_seed = build_default_agents(
                llm, verifier, config, models, env,
                include_moa=args.moa, adaptive_policy=args.adaptive_policy,
            )
            return llm, agents_for_seed

        results = run_multi_seed(
            agent_factory, catalog, seeds=seeds, models=models, budget=budget,
            progress=progress_update,
        )
        elapsed = time.perf_counter() - t0
        payload = serialize_results(results, models, config, seeds=seeds,
                                    provider=args.provider)
        out = args.out or f"benchmarks/results/seeds{args.seeds}_from{args.seed}_{args.suite}{flag_suffix}.json"
    else:
        results = run_benchmark(agents, catalog, seed=args.seed, models=models,
                                budget=budget, llm=llm, progress=progress_update)
        elapsed = time.perf_counter() - t0
        payload = serialize_results(results, models, config, seed=args.seed,
                                    provider=args.provider)
        out = args.out or f"benchmarks/results/seed{args.seed}_{args.suite}{flag_suffix}.json"
    progress_finish()

    ui.section(f"Strategy results — {len(catalog)} tasks")
    print(_fancy_results(results, models))
    ui.section("Paired McNemar comparisons (b01 = A ok/B fail, b10 = A fail/B ok)")
    print(_fancy_comparisons(compare(results)))

    # bootstrap CI on the flagship comparison: adaptive vs react
    if "adaptive" in results and "react" in results:
        lo, hi, point = bootstrap_ci(results["adaptive"], results["react"], seed=args.seed)
        print(ui.dim(f"\nbootstrap 95% CI (adaptive - react): {point:+.3f} "
                     f"[{lo:+.3f}, {hi:+.3f}]"))

    payload["metadata"]["elapsed_s"] = round(elapsed, 3)
    payload["metadata"]["suite"] = args.suite
    payload["metadata"]["paired"] = args.provider == "mock"
    save_results(out, payload)
    print(ui.ok(f"wrote {out}"))


if __name__ == "__main__":
    main()
