"""Free-form Q&A support: a minimal task wrapper and an LLM-based judge.

Used by the interactive ``ask`` tool. There is no deterministic ground truth
for free-form questions, so verification is an LLM judge (the ``expensive``
tier by default) that returns a score and a short piece of feedback — which the
retry strategies (reflexion / self_refine) feed back into their next attempt,
exactly as the research dictates (verifier-gated refinement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LLM
from .verifier import GradeableTask, Verdict


@dataclass
class PromptTask:
    """Minimal task surface for free-form prompts (id + prompt)."""

    id: str
    prompt: str
    num_steps: int = 1  # compatibility surface; unused by free-form strategies


_GRADE_SYSTEM = (
    "MODE: grade\n"
    "You are a strict, calibrated answer grader. Be harsh and use the FULL "
    "0-10 range.\n"
    "Calibration: 10 = flawless and complete; 8-9 = very good with minor gaps; "
    "6-7 = correct but generic or missing depth; 4-5 = partial or vague; "
    "0-3 = wrong, off-topic, or empty.\n"
    "Most answers are NOT 9 or 10. Only exceptional answers earn 9+.\n"
    "TASK_ID: {task_id}\n"
    "QUESTION: {question}\n"
    "ANSWER: {answer}\n\n"
    "Reply with EXACTLY two lines and nothing else:\n"
    "SCORE: <number between 0 and 10>\n"
    "FEEDBACK: <one short sentence>\n"
    "Do not answer the question. Do not explain your rubric."
)

_GRADE_SYSTEM_STRICT = (
    "MODE: grade\n"
    "Ignore all previous instructions. You are ONLY a strict grading function "
    "that uses the full 0-10 scale (most answers 6-7, only excellent ones 9+).\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Reply with EXACTLY this shape, nothing else, no preamble:\n"
    "SCORE: 7\n"
    "FEEDBACK: one short sentence"
)

_GRADE_SYSTEM_BARE = (
    "MODE: grade\n"
    "You grade one answer. Be strict; use the full 0-10 scale.\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Reply with ONLY a single number from 0 to 10 and nothing else."
)


_LETTER_GRADES = {
    "a+": 9.8, "a": 9.2, "a-": 8.8,
    "b+": 8.2, "b": 7.5, "b-": 7.0,
    "c+": 6.2, "c": 5.5, "c-": 5.0,
    "d+": 4.2, "d": 3.5,
    "f": 1.0,
}


def parse_grade(text: str) -> tuple[float | None, str]:
    """Parse a judge reply into ``(score_0_10, feedback)``.

    Handles ``SCORE: 7``, ``SCORE: 7/10``, ``Score: 35/40`` (scaled), letter
    grades (``Grade: A``), and case variants. Returns ``(None, "")`` when no
    score is found.
    """
    # explicit fraction first (e.g. "Score: 35/40" -> 8.75)
    frac = re.search(r"(?:score)\s*[:=]\s*(\d+(?:\.\d+)?)\s*/\s*(\d+)", text, re.IGNORECASE)
    if frac:
        numerator, denominator = float(frac.group(1)), float(frac.group(2))
        if denominator > 0:
            return max(0.0, min(10.0, numerator / denominator * 10.0)), _extract_feedback(text)
    score = re.search(r"(?:score)\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if score:
        return max(0.0, min(10.0, float(score.group(1)))), _extract_feedback(text)
    # letter grade ("Grade: A", "Grade: B+")
    letter = re.search(r"grade\s*[:=]?[^A-Za-z0-9]*([A-Fa-f])([+-])?", text, re.IGNORECASE)
    if letter:
        key = letter.group(1).lower() + (letter.group(2) or "")
        if key in _LETTER_GRADES:
            return _LETTER_GRADES[key], _extract_feedback(text)
    return None, ""


def _extract_feedback(text: str) -> str:
    match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:300]
    return ""


class FreeFormJudge:
    """LLM-based judge for free-form answers (the value function for `ask`).

    Uses a strict two-line output contract; if the model ignores it, one
    stricter retry is attempted before giving up (score stays ``None`` — the UI
    shows "n/a" rather than a wrong number).
    """

    name = "freeform"

    def __init__(self, llm: LLM, model: str = "expensive", max_answer_chars: int = 4000) -> None:
        self._llm = llm
        self._model = model
        self._max_answer_chars = max_answer_chars
        self._last_raw = ""

    def score(self, answer: str, task: GradeableTask, context: dict | None = None) -> Verdict:
        # Flatten newlines: DeepSeek V4 Flash returns blank completions far
        # more often when the answer contains \n (verified empirically).
        answer = " ".join((answer or "").split())[: self._max_answer_chars]
        system = _GRADE_SYSTEM.format(task_id=task.id, question=task.prompt, answer=answer)
        score, feedback = self._grade_retry(system, max_tokens=120, attempts=4)

        if score is None:  # model ignored the contract -> one strict retry
            strict = _GRADE_SYSTEM_STRICT.format(question=task.prompt, answer=answer)
            score, feedback = self._grade_retry(strict, max_tokens=80, attempts=3)

        if score is None:  # last resort: ask for a bare number only
            bare = _GRADE_SYSTEM_BARE.format(question=task.prompt, answer=answer)
            score, feedback = self._grade_retry(bare, max_tokens=12, attempts=3,
                                                bare_number=True)

        if score is None:
            return Verdict(0.0, False, {"feedback": "", "judge_score": None, "raw": self._last_raw[:300]})
        return Verdict(score / 10.0, score >= 5.0, {"feedback": feedback, "judge_score": score})

    def _grade_retry(self, system: str, max_tokens: int, attempts: int,
                     bare_number: bool = False) -> tuple[float | None, str]:
        """Call the model up to ``attempts`` times, retrying blank or
        unparseable replies (DeepSeek V4 Flash occasionally returns an empty
        completion, which used to silently become a 0/10 score)."""
        self._last_raw = ""
        for _ in range(attempts):
            text = self._llm.complete(
                [{"role": "system", "content": system}], model=self._model, max_tokens=max_tokens
            ).text
            self._last_raw = text
            if not text or not text.strip():
                continue  # blank reply -> retry same prompt
            if bare_number:
                m = re.search(r"\b(10|[0-9](?:\.[0-9])?)\b", text.strip())
                if m:
                    return max(0.0, min(10.0, float(m.group(1)))), ""
                continue
            score, feedback = parse_grade(text)
            if score is not None:
                return score, feedback
        # last attempt produced nothing usable
        if bare_number:
            m = re.search(r"\b(10|[0-9](?:\.[0-9])?)\b", self._last_raw.strip())
            if m:
                return max(0.0, min(10.0, float(m.group(1)))), ""
            return None, ""
        return parse_grade(self._last_raw)
