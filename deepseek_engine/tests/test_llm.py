"""Tests for the calibrated MockLLM."""

from dse.config import ModelConfig
from dse.environment import make_catalog
from dse.llm import MockLLM
from dse.verifier import extract_answer


def _single_step_task():
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=1)
    return catalog[next(iter(catalog))]


def test_no_task_context_returns_placeholder():
    llm = MockLLM({}, {"cheap": ModelConfig(name="cheap")}, seed=0)
    out = llm.complete([{"role": "system", "content": "hello"}], model="cheap")
    assert "no task context" in out.text
    assert out.tokens_in >= 1 and out.tokens_out >= 1


def test_full_solution_is_correct_with_perfect_model():
    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=1.0)}
    llm = MockLLM(catalog, models, seed=0, p_fix=1.0)
    out = llm.complete(
        [{"role": "system", "content": f"TASK_ID: {task.id}"}], model="cheap"
    )
    assert extract_answer(out.text) == task.gold


def test_feedback_enables_repair():
    """With p=0 but p_fix=1.0, FEEDBACK lines make the mock answer correctly."""
    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=0.0)}
    llm = MockLLM(catalog, models, seed=0, p_fix=1.0)
    system = f"TASK_ID: {task.id}\nFEEDBACK: step 0 incorrect"
    out = llm.complete([{"role": "system", "content": system}], model="cheap")
    assert extract_answer(out.text) == task.gold


def test_without_feedback_low_accuracy_model_fails():
    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=0.0)}
    llm = MockLLM(catalog, models, seed=0, p_fix=0.0)
    out = llm.complete([{"role": "system", "content": f"TASK_ID: {task.id}"}], model="cheap")
    assert extract_answer(out.text) != task.gold


def test_next_step_mode():
    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=1.0)}
    llm = MockLLM(catalog, models, seed=0)
    system = f"TASK_ID: {task.id}\nMODE: next_step\nPREFIX: 0"
    out = llm.complete([{"role": "system", "content": system}], model="cheap")
    assert out.text == f"STEP 0: {task.correct_steps[0]}"


def test_judge_mode_returns_score():
    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=1.0, judge_accuracy=1.0)}
    llm = MockLLM(catalog, models, seed=0)
    system = f"TASK_ID: {task.id}\nMODE: judge\nDRAFT: {task.gold}"
    out = llm.complete([{"role": "system", "content": system}], model="cheap")
    assert "SCORE: 1.000" in out.text


def test_step_judge_mode_perfect():
    """With judge_accuracy=1.0 the noisy per-step judge must score a correct
    step ~1.0 and a wrong step ~0.0."""
    import re

    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", judge_accuracy=1.0)}
    llm = MockLLM(catalog, models, seed=0)

    good = llm.complete(
        [{"role": "system", "content": f"TASK_ID: {task.id}\nMODE: step_judge\nSTEP: {task.correct_steps[0]}\nPREFIX: 0"}],
        model="cheap",
    )
    good_score = float(re.search(r"SCORE:\s*([\d.]+)", good.text).group(1))
    assert good_score >= 0.9

    bad = llm.complete(
        [{"role": "system", "content": f"TASK_ID: {task.id}\nMODE: step_judge\nSTEP: wrong\nPREFIX: 0"}],
        model="cheap",
    )
    bad_score = float(re.search(r"SCORE:\s*([\d.]+)", bad.text).group(1))
    assert bad_score <= 0.1


def test_reset_makes_draws_reproducible():
    task = _single_step_task()
    catalog = {task.id: task}
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=0.5)}
    llm_a = MockLLM(catalog, models, seed=0)
    llm_b = MockLLM(catalog, models, seed=0)
    llm_a.reset("k:v")
    llm_b.reset("k:v")
    out_a = llm_a.complete([{"role": "system", "content": f"TASK_ID: {task.id}"}], model="cheap")
    out_b = llm_b.complete([{"role": "system", "content": f"TASK_ID: {task.id}"}], model="cheap")
    assert out_a.text == out_b.text
