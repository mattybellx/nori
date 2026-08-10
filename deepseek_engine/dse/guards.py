"""Never-worse guards: selection guard + no-regression synthesis guard.

These make *"never worse than the best candidate"* a design property instead
of an empirical accident (the project's weak-model result b01 = 0 becomes a
guarantee, not a coincidence):

1. ``selection_guard`` — only crown a non-baseline winner when the calibrated
   judge score clears a noise floor; otherwise ship the baseline (react). The
   shipped answer can never be one the calibrated judge scored meaningfully
   *below* the baseline.
2. ``consistency_score`` — a cheap, embedding-free groundedness measure: the
   maximum n-gram overlap ratio of the synthesized answer with any single
   candidate. Stops merges from inventing content no candidate produced.
3. ``synthesis_guard`` — ship the synthesized answer only if it is grounded in
   at least one candidate AND scores at least as well as the winner (within a
   margin); otherwise fall back to the winner's answer verbatim.

Everything here is stdlib-only and deterministic except the optional judge
call in ``synthesis_guard`` (used by the live chat path and the benchmark).
The guarantee it enforces: the returned answer is never the synthesized
version when the calibrated judge scored that version below the winner by
more than the margin — and if the judge FAILS or returns no score, the
synthesis is never shipped (fall back to the winner), because an unverified
merge could not be guaranteed never-worse (the one miss in the n=32
free-form run).
"""

from __future__ import annotations

import re
from typing import Callable

DEFAULT_NOISE_FLOOR = 0.5   # calibrated-judge (0-10) gap a non-baseline winner must clear
DEFAULT_MIN_OVERLAP = 0.10  # min bigram-overlap ratio to call a synthesis "grounded"
DEFAULT_SCORE_MARGIN = 0.5  # judge (0-10) margin a synthesis must clear to replace the winner
DEFAULT_JUDGE_SAMPLES = 3   # median-of-N judge calls to de-noise the score-based check


def robust_score(judge: Callable[[str], float | None], text: str,
                 samples: int = DEFAULT_JUDGE_SAMPLES) -> float | None:
    """Median of ``samples`` independent judge calls — reduces judge noise.

    The live judge is noisy (a reasoning model grading itself), so a single
    call can move the score by a point or more. The median of several calls is
    a far more stable input to the guards (the property the guards enforce is
    only as good as the score they compare).
    """
    scores = [judge(text) for _ in range(max(1, samples))]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    scores.sort()
    return scores[len(scores) // 2]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def ngram_set(text: str, n: int = 2) -> set[tuple[str, ...]]:
    """Word n-grams as a set (default bigrams)."""
    toks = _tokens(text)
    if len(toks) < n:
        return set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def consistency_score(synth: str, candidates: list[str], n: int = 2) -> float:
    """Max n-gram overlap ratio of ``synth`` with any single candidate.

    1.0 = every n-gram of the synthesis appears in one candidate;
    0.0 = entirely novel content. Used as a cheap groundedness signal.
    """
    if not synth or not synth.strip() or not candidates:
        return 0.0
    s = ngram_set(synth, n)
    if not s:
        return 0.0
    best = 0.0
    for cand in candidates:
        if not cand:
            continue
        c = ngram_set(cand, n)
        if not c:
            continue
        overlap = len(s & c) / len(s)
        if overlap > best:
            best = overlap
    return best


def selection_guard(
    records: list[dict],
    winner_strategy: str | None,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
) -> str | None:
    """Pick the strategy to ship so the shipped answer is never scored
    meaningfully below the baseline by the calibrated judge.

    ``records``: latest record per strategy, each with ``strategy`` and
    ``judge_score`` (0-10 or None). ``react`` is treated as the baseline.
    """
    if not records:
        return winner_strategy
    base = next((r for r in records if r.get("strategy") == "react"), None)
    win = next((r for r in records if r.get("strategy") == winner_strategy), None)
    if win is None:
        # arbiter picked something we don't have — fall back to the highest-scored record
        scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]
        if scored:
            return max(scored, key=lambda r: r["judge_score"])["strategy"]
        return (records[0] or {}).get("strategy")
    if winner_strategy == "react" or base is None:
        return winner_strategy
    bs = base.get("judge_score")
    ws = win.get("judge_score")
    if bs is None or ws is None:
        # no judge scores to compare — trust the arbiter
        return winner_strategy
    if ws < bs - noise_floor:
        return base["strategy"]
    return winner_strategy


def synthesis_guard(
    synth: str | None,
    candidates: list[str],
    winner_answer: str,
    judge: Callable[[str], float | None] | None = None,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    score_margin: float = DEFAULT_SCORE_MARGIN,
) -> tuple[str, bool, str]:
    """Decide whether the synthesized answer is safe to ship.

    Returns ``(final_answer, used_synth, reason)``. Never ships a synthesis
    that (a) isn't grounded in at least one candidate, or (b) — when a judge
    is supplied — scores below the winner by more than ``score_margin``.
    """
    if not synth or not synth.strip():
        return winner_answer, False, "empty synthesis -> winner"
    grounded = consistency_score(synth, candidates)
    if grounded < min_overlap:
        return winner_answer, False, f"ungrounded synthesis (overlap={grounded:.2f}) -> winner"
    if judge is None:
        return synth, True, f"grounded synthesis (overlap={grounded:.2f})"
    try:
        s_score = judge(synth)
        w_score = judge(winner_answer)
    except Exception:
        # A judge failure means we CANNOT verify never-worse, so shipping an
        # unverified merge is unsafe. Fall back to the winner (conservative).
        # Lesson from the n=32 free-form run: the old "ship anyway" policy let
        # one unverified merge through and the fresh judge scored it 0.0.
        return winner_answer, False, "judge failed -> winner"
    if s_score is None or w_score is None:
        # No score = no verification of never-worse -> fall back to the winner.
        return winner_answer, False, (
            f"judge score missing (synth={s_score}, winner={w_score}) -> winner")
    if s_score < w_score - score_margin:
        return winner_answer, False, (
            f"synthesis {s_score:.1f} < winner {w_score:.1f} - margin {score_margin} -> winner")
    return synth, True, (
        f"grounded (overlap={grounded:.2f}) and score {s_score:.1f} >= winner {w_score:.1f}")
