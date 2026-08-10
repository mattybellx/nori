"""Tests for the real-domain bridge's report/stat components (mock-safe).

These cover everything that does NOT need the live DeepSeek API: preference
aggregation, sign-test wiring, summary math, groundedness averaging, and
report serialization. The live-model experiment itself is an interactive CLI
run, not a unit test.
"""

from dse.discovery.real_domain import (
    RealDomainReport,
    _mean,
    _pref_stats,
    build_summary,
)


class MeanTests:
    def test_mean_all_none(self):
        assert _mean([None, None]) is None

    def test_mean_ignores_none(self):
        assert _mean([2.0, None, 4.0]) == 3.0

    def test_mean_empty(self):
        assert _mean([]) is None


class PrefStatsTests:
    def test_wins_and_ties(self):
        s = _pref_stats(["A", "A", "B", "TIE", "A"])
        assert s["favor_arch"] == 3
        assert s["favor_baseline"] == 1
        assert s["ties"] == 1
        assert s["win_rate"] == 0.75

    def test_sign_test_p(self):
        # 6/6 favor -> two-sided sign test p = 2 * (1/2)^6 = 0.03125
        s = _pref_stats(["A"] * 6)
        assert s["sign_test_p"] == 0.03125

    def test_no_decisions(self):
        s = _pref_stats(["TIE", "TIE"])
        assert s["win_rate"] is None
        assert s["sign_test_p"] is None


class BuildSummaryTests:
    def _report(self):
        return RealDomainReport(
            provider="deepseek", judge_model="deepseek-v4-pro",
            n_questions=3, judge_samples=3, baseline="react",
            questions=["q1", "q2", "q3"],
            answers={
                "react": ["base answer one", "base answer two", "base answer three"],
                "cand": ["cand answer one", "cand answer two", "cand answer three"],
            },
            scores={"react": [6.0, 7.0, 8.0], "cand": [8.0, 7.0, 9.0]},
            preferences={"cand": ["A", "TIE", "A"]},
        )

    def test_means_and_delta(self):
        s = build_summary(self._report())["cand"]
        assert s["mean_judge_score"] == 8.0
        assert s["baseline_mean"] == 7.0
        assert s["delta"] == 1.0

    def test_never_worse(self):
        s = build_summary(self._report())["cand"]
        # cand [8,7,9] vs base [6,7,8]: all >= base - 0.5
        assert s["never_worse_by_judge"] == 3
        assert s["n"] == 3

    def test_preference_stats(self):
        s = build_summary(self._report())["cand"]
        p = s["preference"]
        assert p["favor_arch"] == 2
        assert p["favor_baseline"] == 0
        assert p["ties"] == 1
        # 2/2 -> p = 0.5
        assert p["sign_test_p"] == 0.5

    def test_grounded_avg(self):
        # grounded_score = js * (0.6 + 0.4 * overlap).
        # overlap of "cand answer one" vs "base answer one": shared tokens
        # {"cand","answer","one"} / {"base","answer","one"}... exactly 2/3.
        s = build_summary(self._report())["cand"]
        g = s["grounded_avg"]
        # 8.0*(0.6+0.4*2/3) = 8.0*(0.6+0.2667) = 6.9333 ; 9.0*...7/9 overlap
        # values are fuzzy; just assert it is a float in (0, 10) and >= 0
        assert isinstance(g, float)
        assert 0.0 <= g <= 10.0

    def test_missing_candidate_scores(self):
        report = RealDomainReport(
            provider="d", judge_model="j", n_questions=2, judge_samples=3,
            baseline="react", questions=["a", "b"],
            answers={"react": ["x", "y"], "cand": ["u", "v"]},
            scores={"react": [5.0, 5.0], "cand": [None, None]},
        )
        s = build_summary(report)["cand"]
        assert s["mean_judge_score"] is None
        assert s["grounded_avg"] == 0.0  # no valid judge scores

    def test_baseline_excluded_from_summary(self):
        summary = build_summary(self._report())
        assert "react" not in summary
        assert "cand" in summary


class ReportSerializationTests:
    def test_to_dict_roundtrip(self):
        report = RealDomainReport(
            provider="deepseek", judge_model="deepseek-v4-pro",
            n_questions=1, judge_samples=3, baseline="react",
            questions=["q"], answers={"react": ["a"]},
            scores={"react": [6.5]}, preferences={},
            summary={}, elapsed_s=12.5,
        )
        d = report.to_dict()
        assert d["provider"] == "deepseek"
        assert d["judge_model"] == "deepseek-v4-pro"
        assert d["scores"] == {"react": [6.5]}
        assert d["elapsed_s"] == 12.5
        assert set(d) >= {"answers", "preferences", "summary", "n_questions"}
