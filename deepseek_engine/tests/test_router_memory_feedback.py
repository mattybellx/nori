"""Tests for the router, memory, and the compiler feedback loop."""

from dse.config import EffortTier, RouterConfig
from dse.events import StepEvent
from dse.feedback import CompilerFeedbackLoop
from dse.memory import Memory
from dse.router import Router


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def test_router_escalates_on_low_confidence():
    router = Router(RouterConfig(escalate_confidence_below=0.5, max_escalations=2))
    decision = router.decide(probe_score=0.2)
    assert decision.escalated
    assert decision.tier == EffortTier.HIGH
    assert decision.model == "expensive"


def test_router_stays_cheap_on_high_confidence():
    router = Router(RouterConfig(escalate_confidence_below=0.5, max_escalations=2))
    decision = router.decide(probe_score=0.9)
    assert not decision.escalated
    assert decision.model == "cheap"
    assert decision.tier == EffortTier.LOW


def test_router_respects_escalation_budget():
    router = Router(RouterConfig(escalate_confidence_below=0.5, max_escalations=1))
    assert router.decide(0.1).escalated
    # second low-confidence call: budget spent → back to cheap
    decision = router.decide(0.1)
    assert not decision.escalated
    assert decision.model == "cheap"


def test_router_reset():
    router = Router(RouterConfig(escalate_confidence_below=0.5, max_escalations=1))
    router.decide(0.1)
    assert router.escalation_count == 1
    router.reset()
    assert router.escalation_count == 0


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def test_memory_window_trimming():
    memory = Memory(window_tokens=40, summary_every=100)
    for i in range(10):
        memory.add_step(StepEvent(kind="thought", content="x" * 50, model="cheap"))
    used = sum(len(e.content) // 4 for e in memory.window)
    assert used <= 40


def test_memory_folds_summary():
    memory = Memory(window_tokens=100000, summary_every=3)
    for i in range(3):
        memory.add_step(StepEvent(kind="thought", content=f"step {i}", model="cheap"))
    assert memory.summary  # summary was triggered


def test_memory_reflections_bounded():
    memory = Memory(max_reflections=2)
    memory.add_reflection("r1")
    memory.add_reflection("r2")
    memory.add_reflection("r3")
    assert memory.reflections == ["r2", "r3"]


def test_memory_render_includes_system_and_reflections():
    memory = Memory(max_reflections=2)
    memory.add_reflection("careful with step 3")
    messages = memory.render_messages("TASK_ID: t1", "solve")
    assert "TASK_ID: t1" in messages[0]["content"]
    assert "careful with step 3" in messages[0]["content"]
    assert messages[1]["content"] == "solve"


# ---------------------------------------------------------------------------
# Compiler feedback loop
# ---------------------------------------------------------------------------
def _task_with_correct_solution():
    from dse.environment import Environment, make_catalog

    catalog = make_catalog(seed=1, n_tasks=1, max_steps=2)
    task = catalog[next(iter(catalog))]
    env = Environment(catalog)
    solution = "\n".join(f"STEP {i}: {v}" for i, v in enumerate(task.correct_steps))
    solution += f"\nANSWER: {task.gold}"
    return env, task, solution


def test_feedback_loop_all_stages_present():
    env, task, solution = _task_with_correct_solution()
    loop = CompilerFeedbackLoop(env)
    report = loop.run(task, solution)
    stages = {s.stage for s in report.stages}
    assert stages == {"build", "test", "lint", "security", "perf"}


def test_feedback_loop_passes_correct_solution():
    env, task, solution = _task_with_correct_solution()
    loop = CompilerFeedbackLoop(env)
    report = loop.run(task, solution)
    assert report.passed
    assert report.failures() == []


def test_feedback_loop_detects_test_failure():
    env, task, solution = _task_with_correct_solution()
    broken = solution.replace(task.correct_steps[0], "!!", 1)
    loop = CompilerFeedbackLoop(env)
    report = loop.run(task, broken)
    assert not report.passed
    assert report.detail("test").startswith("tests passed")
