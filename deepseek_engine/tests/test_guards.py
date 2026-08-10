"""Tests for the never-worse guards (dse/guards.py).

The property test at the end is the important one: across many random
scenarios it asserts the guards *always* return something whose (simulated)
quality is never below the best candidate's by more than the configured
margin — i.e. "never worse" is enforced by construction.
"""

from __future__ import annotations

import random

from dse.guards import (
    consistency_score,
    grounded_score,
    robust_score,
    selection_guard,
    synthesis_guard,
)


# ---------------------------------------------------------------------------
# consistency_score
# ---------------------------------------------------------------------------

def test_consistency_identical_is_one():
    assert consistency_score("the quick brown fox jumps", ["the quick brown fox jumps"]) == 1.0


def test_consistency_disjoint_is_zero():
    assert consistency_score("alpha beta gamma delta", ["w x y z"]) == 0.0


def test_consistency_partial_is_between():
    # shares one bigram ("the quick") out of three -> partial
    score = consistency_score("the quick red fox", ["the quick brown fox jumps over the lazy dog"])
    assert 0.0 < score < 1.0
    assert abs(score - 1.0 / 3.0) < 1e-9


def test_consistency_takes_best_candidate():
    # shares nothing with candidate 0, everything with candidate 1
    synth = "hello world from nori"
    assert consistency_score(synth, ["zzz yyy", "hello world from nori and more"]) == 1.0


def test_consistency_empty_is_zero():
    assert consistency_score("", ["anything at all"]) == 0.0
    assert consistency_score(None, ["anything"]) == 0.0
    assert consistency_score("something", []) == 0.0


# ---------------------------------------------------------------------------
# grounded_score (substance-weighted quality — attacks the self-grading
# circularity: a score is only "worth" as much as the answer is grounded in
# the candidates, not merely styled like the judge's own output)
# ---------------------------------------------------------------------------

def test_grounded_fully_grounded_keeps_full_judge_score():
    # overlap = 1.0 -> weight = 1.0 -> score unchanged
    assert grounded_score("abc abc abc", ["abc abc abc"],
                          judge_score=9.0, min_weight=0.6) == 9.0


def test_grounded_zero_overlap_drops_to_min_weight():
    # overlap = 0.0 -> score * min_weight
    assert grounded_score("wxyz qrst", ["abc def ghi"],
                          judge_score=9.0, min_weight=0.6) == 9.0 * 0.6


def test_grounded_partial_overlap_discounts():
    # one shared bigram out of three -> overlap = 1/3
    # weight = 0.6 + 0.4*(1/3) = 0.7333...
    s = grounded_score("the quick red fox", ["the quick brown fox jumps"],
                       judge_score=9.0, min_weight=0.6)
    assert abs(s - 9.0 * (0.6 + 0.4 / 3.0)) < 1e-9


def test_grounded_no_judge_returns_overlap_only():
    assert grounded_score("abc abc", ["abc abc abc"]) == 1.0
    assert grounded_score("wxyz", ["abc def"]) == 0.0


def test_grounded_takes_best_candidate_and_defaults_min_weight():
    # default min_weight=0.6; best candidate gives overlap 1.0
    assert grounded_score("nori test phrase", ["zzz", "nori test phrase extra"],
                          judge_score=8.0) == 8.0


def test_grounded_empty_synth_is_zero():
    assert grounded_score("", ["anything"], judge_score=8.0) == 0.0


# ---------------------------------------------------------------------------
# selection_guard
# ---------------------------------------------------------------------------

def _rec(strategy, score):
    return {"strategy": strategy, "judge_score": score}


def test_selection_keeps_winner_when_above_baseline():
    records = [_rec("react", 7.0), _rec("reflexion", 8.5)]
    assert selection_guard(records, "reflexion") == "reflexion"


def test_selection_falls_back_to_baseline_when_winner_below_floor():
    records = [_rec("react", 8.0), _rec("reflexion", 6.5)]  # gap 1.5 > 0.5
    assert selection_guard(records, "reflexion") == "react"


def test_selection_keeps_winner_within_floor():
    records = [_rec("react", 8.0), _rec("reflexion", 7.7)]  # gap 0.3 <= 0.5
    assert selection_guard(records, "reflexion") == "reflexion"


def test_selection_baseline_winner_stays():
    records = [_rec("react", 7.0), _rec("reflexion", 8.5)]
    assert selection_guard(records, "react") == "react"


def test_selection_no_baseline_keeps_winner():
    records = [_rec("reflexion", 8.0), _rec("self_refine", 7.0)]
    assert selection_guard(records, "self_refine") == "self_refine"


def test_selection_missing_scores_trusts_arbiter():
    records = [_rec("react", None), _rec("reflexion", None)]
    assert selection_guard(records, "reflexion") == "reflexion"


def test_selection_unknown_winner_falls_to_highest_scored():
    records = [_rec("react", 7.0), _rec("reflexion", 8.0)]
    assert selection_guard(records, "ghost") == "reflexion"


# ---------------------------------------------------------------------------
# synthesis_guard
# ---------------------------------------------------------------------------

def test_synth_empty_falls_back():
    ans, used, why = synthesis_guard("", ["candidate answer text"], "winner text")
    assert used is False and ans == "winner text"


def test_synth_ungrounded_falls_back():
    ans, used, why = synthesis_guard("totally unrelated novel content here",
                                     ["the real candidate answer text"], "winner text")
    assert used is False and ans == "winner text"


def test_synth_grounded_no_judge_ships():
    ans, used, why = synthesis_guard("the real candidate answer text refined",
                                     ["the real candidate answer text"], "winner text")
    assert used is True and "grounded" in why


def test_synth_grounded_but_scored_below_winner_falls_back():
    judge = lambda t: 6.0 if "synth" in t else 9.0  # synth scores 6, winner 9
    ans, used, why = synthesis_guard("synth grounded answer", ["synth grounded answer"],
                                     "winner", judge=judge)
    assert used is False and ans == "winner"


def test_synth_grounded_and_scored_above_winner_ships():
    judge = lambda t: 8.5 if "synth" in t else 7.0
    ans, used, why = synthesis_guard("synth grounded answer", ["synth grounded answer"],
                                     "winner", judge=judge)
    assert used is True and ans == "synth grounded answer"


def test_synth_judge_error_falls_back_to_winner():
    """A judge failure means never-worse cannot be verified -> ship the winner
    (conservative). This closes the n=32 miss where an unverified merge shipped
    and the fresh judge scored it 0.0."""
    def boom(t):
        raise RuntimeError("judge down")
    ans, used, why = synthesis_guard("grounded answer text", ["grounded answer text"],
                                     "winner", judge=boom)
    assert used is False and ans == "winner"
    assert "judge failed" in why


def test_synth_judge_score_missing_falls_back_to_winner():
    """Judge returns no score -> cannot verify never-worse -> ship the winner."""
    judge = lambda t: None
    ans, used, why = synthesis_guard("grounded answer text", ["grounded answer text"],
                                     "winner", judge=judge)
    assert used is False and ans == "winner"
    assert "score missing" in why


# ---------------------------------------------------------------------------
# robust_score (median-of-N judge — de-noising)
# ---------------------------------------------------------------------------

def test_robust_score_median_smooths_outliers():
    calls = []
    def noisy(t):
        calls.append(1)
        return 7.0 if len(calls) <= 2 else 3.0  # one outlier low call
    assert robust_score(noisy, "x", samples=3) == 7.0


def test_robust_score_all_none_is_none():
    assert robust_score(lambda t: None, "x", samples=3) is None


def test_robust_score_mixed_none_keeps_real_scores():
    def mixed(t):
        return 5.0 if t == "a" else None
    assert robust_score(mixed, "a", samples=3) == 5.0


def test_robust_score_single_sample_is_plain_call():
    assert robust_score(lambda t: 4.0, "x", samples=1) == 4.0


# ---------------------------------------------------------------------------
# Property: never worse BY CONSTRUCTION across many random scenarios
# ---------------------------------------------------------------------------

def test_property_never_worse_by_construction():
    """For thousands of random scenarios the guards never return an answer
    whose quality is below the best candidate's by more than the margin."""
    rng = random.Random(42)
    best_worse_count = 0
    scenarios = 3000
    for _ in range(scenarios):
        # pick a baseline and a competing winner with random judge scores
        base_score = rng.uniform(0, 10)
        win_score = rng.uniform(0, 10)
        records = [_rec("react", base_score), _rec("reflexion", win_score)]

        # selection guard: shipped strategy's score must be >= baseline - floor
        shipped = selection_guard(records, "reflexion")
        shipped_score = next(r["judge_score"] for r in records if r["strategy"] == shipped)
        if shipped_score < base_score - 0.5 - 1e-9:
            best_worse_count += 1

        # synthesis guard: simulated quality of the returned answer must be
        # >= min(winner, synth) quality (it can only fall back to the winner)
        winner_q = win_score
        synth_q = rng.uniform(0, 10)
        # fake judge that reports the simulated qualities
        judge = lambda t, _q=synth_q, _w=winner_q: synth_q if "SYNTH" in t else winner_q
        grounded = rng.random() > 0.3  # sometimes the synthesis is grounded
        if grounded:
            candidates = ["SYNTH refined answer"]  # overlaps the synth -> grounded
            final, used, _why = synthesis_guard("SYNTH refined answer", candidates,
                                                "winner text", judge=judge)
            final_q = synth_q if used else winner_q
            # never worse than the winner (by more than the margin)
            if final_q < winner_q - 0.5 - 1e-9:
                best_worse_count += 1
    assert best_worse_count == 0, (
        f"guard violated never-worse in {best_worse_count}/{scenarios} scenarios")
