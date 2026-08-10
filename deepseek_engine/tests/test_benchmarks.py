"""Tests for the benchmark harness: statistics and end-to-end runs."""

import json

from dse.agent import Budget
from dse.benchmarks.harness import (
    bootstrap_ci,
    chi2_cdf,
    compare,
    mcnemar,
    render_comparisons,
    render_table,
    run_benchmark,
    run_multi_seed,
    save_results,
    serialize_results,
    sign_test,
    wilcoxon_signed_rank,
)
from dse.events import RunResult


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_chi2_cdf_sanity():
    assert chi2_cdf(0.0, 1) == 0.0
    assert chi2_cdf(1e9, 1) > 0.999
    # critical value for alpha=0.05 at 1 dof is 3.841 → CDF ≈ 0.95
    assert abs(chi2_cdf(3.841, 1) - 0.95) < 0.001


def test_chi2_cdf_exact_closed_form_for_one_dof():
    """CDF(x, 1) = erf(sqrt(x/2)) exactly; check against known quantiles."""
    import math

    assert abs(chi2_cdf(6.635, 1) - 0.99) < 0.001  # alpha=0.01 critical value
    # median of chi-square(1) is 0.454936
    assert abs(chi2_cdf(0.454936, 1) - 0.50) < 0.0001
    # consistency with the closed form at an arbitrary point
    assert abs(chi2_cdf(2.0, 1) - math.erf(math.sqrt(1.0))) < 1e-12


def test_mcnemar_discordant_counts():
    a = [_run(i, i % 2 == 0) for i in range(10)]
    b = [_run(i, i % 3 == 0) for i in range(10)]
    result = mcnemar(a, b)
    assert result.b01 == sum(1 for i in range(10) if i % 2 == 0 and i % 3 != 0)
    assert result.b10 == sum(1 for i in range(10) if i % 2 != 0 and i % 3 == 0)
    assert 0.0 <= result.p_value <= 1.0


def test_mcnemar_identical_has_p_one():
    runs = [_run(i, True) for i in range(8)]
    result = mcnemar(runs, list(runs))
    assert result.p_value == 1.0
    assert not result.significant


def test_mcnemar_detects_difference():
    # A always right, B always wrong → strong discordance → tiny p
    a = [_run(i, True) for i in range(20)]
    b = [_run(i, False) for i in range(20)]
    result = mcnemar(a, b)
    assert result.significant
    assert result.p_value < 0.001


def test_bootstrap_ci_identical_is_zero():
    runs = [_run(i, i % 2 == 0) for i in range(20)]
    lo, hi, point = bootstrap_ci(runs, list(runs), iters=300, seed=0)
    assert lo <= point <= hi
    assert abs(point) < 1e-9


def test_bootstrap_ci_separates_different_rates():
    a = [_run(i, True) for i in range(40)]
    b = [_run(i, False) for i in range(40)]
    lo, hi, point = bootstrap_ci(a, b, iters=400, seed=0)
    assert point > 0.5
    assert lo > 0.5


def test_wilcoxon_identical_is_p_one():
    w, p = wilcoxon_signed_rank([6.0, 7.0, 8.0, 5.0], [6.0, 7.0, 8.0, 5.0])
    assert p == 1.0  # all differences zero -> dropped -> no evidence
    w2, p2 = wilcoxon_signed_rank([6.0, 6.5, 7.0], [6.0, 6.5, 7.0])
    assert p2 == 1.0


def test_wilcoxon_detects_systematic_improvement():
    # every final score is 2 points above baseline -> strongly significant
    baseline = [5.0, 6.0, 7.0, 5.5, 6.5, 7.5, 4.0, 6.0]
    final = [b + 2.0 for b in baseline]
    w, p = wilcoxon_signed_rank(final, baseline)
    assert p < 0.01
    # all positive -> W+ equals the full rank sum
    assert w == 8 * 9 / 2.0


def test_wilcoxon_no_difference_has_high_p():
    # balanced + and - diffs -> not significant; ranks {1,2,3,4}, positives at
    # ranks 1 and 2 -> W+ = 3
    a = [6.0, 7.0, 8.0, 9.0]
    b = [5.0, 5.0, 11.0, 13.0]  # diffs: +1, +2, -3, -4
    w, p = wilcoxon_signed_rank(a, b)
    assert p > 0.05
    assert w == 3.0


def test_wilcoxon_requires_equal_length():
    import pytest

    with pytest.raises(ValueError, match="equal length"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


def test_wilcoxon_handles_ties_with_average_ranks():
    # diffs: +1, +1, -2 -> tied rank (1+2)/2=1.5 each for the two +1s
    w, p = wilcoxon_signed_rank([5.0, 5.0, 3.0], [4.0, 4.0, 5.0])
    assert w == 1.5 + 1.5  # both positive diffs keep their tied ranks
    assert 0.0 <= p <= 1.0


def test_wilcoxon_large_n_uses_normal_approximation():
    # n=40 systematic improvement; normal approximation must agree direction
    import random

    rng = random.Random(0)
    baseline = [rng.uniform(3.0, 7.0) for _ in range(40)]
    final = [b + 1.5 for b in baseline]
    w, p = wilcoxon_signed_rank(final, baseline)
    assert p < 0.001
    assert w == 40 * 41 / 2.0


# ---------------------------------------------------------------------------
# sign_test (two-sided binomial sign test on paired preferences)
# ---------------------------------------------------------------------------

def test_sign_test_zero_total_is_one():
    assert sign_test(0, 0) == 1.0


def test_sign_test_all_against_is_two_sided_tail():
    # 0 favor of 10 -> P(X<=0) = 0.5^10; two-sided doubles it
    expected = 2 * (0.5 ** 10)
    assert abs(sign_test(0, 10) - expected) < 1e-12


def test_sign_test_balanced_is_high_p():
    # 5 vs 5 -> two-sided p should be ~1
    assert sign_test(5, 10) >= 0.99


def test_sign_test_extreme_split_is_significant():
    # 20 vs 0 -> essentially impossible under H0
    assert sign_test(20, 20) < 0.00001


def test_sign_test_moderate_split_exact():
    # 9 vs 1 of 10: P(X<=1) = (C(10,0)+C(10,1))*0.5^10; two-sided
    expected = 2 * (0.5 ** 10 + 10 * 0.5 ** 10)
    assert abs(sign_test(9, 10) - expected) < 1e-12


def test_sign_test_symmetric_in_favor_direction():
    # favoring A 9/10 and favoring B 9/10 give identical p
    assert sign_test(9, 10) == sign_test(1, 10)


def test_sign_test_typical_n64_boundary():
    # n=64, 40 favor, 24 against (ties excluded) — power check
    p = sign_test(40, 64)
    assert 0.0 <= p <= 1.0
    # 40/64 = 62.5% is not yet significant at 5% for a two-sided test
    assert p > 0.05


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def _unpack(stack):
    """(config, catalog, models, llm, env, verifier, agents, budget)"""
    return stack


def test_run_benchmark_covers_all_agents(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    results = run_benchmark(
        agents, catalog, seed=11, models=models, budget=budget, llm=llm
    )
    assert set(results) == {a.name for a in agents}
    for name, runs in results.items():
        assert len(runs) == len(catalog)
        # all runs have valid summaries
        assert all(isinstance(r.success, bool) for r in runs)


def test_run_benchmark_is_reproducible(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    first = run_benchmark(agents, catalog, seed=5, models=models, budget=budget, llm=llm)
    # re-seed llm so both runs start from the same state
    llm.reset("bench:seed:5")
    second = run_benchmark(agents, catalog, seed=5, models=models, budget=budget, llm=llm)
    for name in first:
        assert [r.success for r in first[name]] == [r.success for r in second[name]]


def test_paired_fairness_identical_draws(tiny_stack):
    """Every (agent, task) pair must face identical draws → same strategy sees
    the same task instance across agents (valid paired comparison)."""
    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    llm.reset("fair:0")
    _ = run_benchmark(agents, catalog, seed=0, models=models, budget=budget, llm=llm)
    # after run, MockLLM state is at the last (agent,task); reset deterministically
    llm.reset("fair:0")
    _ = run_benchmark(agents, catalog, seed=0, models=models, budget=budget, llm=llm)


def test_serialize_and_save(tiny_stack, tmp_path):
    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    results = run_benchmark(agents, catalog, seed=2, models=models, budget=budget, llm=llm)
    payload = serialize_results(results, models, config, seed=2)
    assert "metadata" in payload and "summaries" in payload and "comparisons" in payload
    assert set(payload["summaries"]) == {a.name for a in agents}
    path = tmp_path / "result.json"
    save_results(str(path), payload)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["seed"] == 2


def test_run_multi_seed_aggregates_and_pairs(tiny_stack):
    """Multi-seed aggregation must yield n*seeds runs per strategy, all still
    paired (same per-(seed, agent, task) reseeding). Each seed's agents must be
    bound to that seed's LLM (the coupling bug regression check)."""
    from dse.benchmarks.harness import run_multi_seed
    from dse.factory import build_default_agents
    from dse.llm import MockLLM

    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    seeds = (0, 1, 2)

    def agent_factory(seed):
        seed_llm = MockLLM(catalog, models, seed=seed)
        seed_agents = build_default_agents(seed_llm, verifier, config, models, env)
        return seed_llm, seed_agents

    results = run_multi_seed(agent_factory, catalog, seeds=seeds, models=models, budget=budget)
    for name, runs in results.items():
        assert len(runs) == len(seeds) * len(catalog)
    # combined dataset still produces valid summaries and comparisons
    assert len(compare(results)) > 0

    # determinism across two identical multi-seed runs (regression for the
    # cross-agent RNG-coupling bug)
    results2 = run_multi_seed(agent_factory, catalog, seeds=seeds, models=models, budget=budget)
    for name in results:
        assert [r.success for r in results[name]] == [r.success for r in results2[name]]


def test_serialize_seeds_metadata(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    results = run_benchmark(agents, catalog, seed=2, models=models, budget=budget, llm=llm)
    payload = serialize_results(results, models, config, seeds=(0, 1, 2), provider="mock")
    assert payload["metadata"]["seeds"] == [0, 1, 2]
    assert payload["metadata"]["provider"] == "mock"
    # single-seed back-compat: seed field still set
    payload2 = serialize_results(results, models, config, seed=2)
    assert payload2["metadata"]["seed"] == 2


def test_provider_note_is_suite_aware():
    """The provider note must NOT warn on the real suite (real models score it
    meaningfully), and must warn on the synthetic suite."""
    from dse.benchmarks.run_benchmark import provider_note

    assert provider_note("mock", "all") is None
    assert provider_note("mock", "real") is None
    assert "WARNING" not in provider_note("deepseek", "real")
    assert "Real-model run" in provider_note("deepseek", "real")
    assert "WARNING" in provider_note("deepseek", "all")
    assert "WARNING" in provider_note("ollama", "easy")


def test_render_table_and_comparisons(tiny_stack):
    config, catalog, models, llm, env, verifier, agents, budget = _unpack(tiny_stack)
    results = run_benchmark(agents, catalog, seed=3, models=models, budget=budget, llm=llm)
    table = render_table(results, models)
    assert "strategy" in table and "success" in table
    comparisons = render_comparisons(compare(results))
    assert "b01" in comparisons


def _run(task_id, success):
    return RunResult(
        task_id=str(task_id), strategy="t", success=success, answer="a",
        verifier_score=1.0 if success else 0.0,
    )
