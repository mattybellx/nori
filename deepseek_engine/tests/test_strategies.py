"""Behavioral + structural tests for the agent strategies."""

import pytest

from dse.agent import Budget
from dse.config import EngineConfig, ModelConfig, default_flags
from dse.environment import Environment, make_catalog
from dse.factory import build_default_agents
from dse.llm import MockLLM
from dse.orchestration import MoAAgent
from dse.strategies import (
    AdaptiveAgent,
    BestOfNAgent,
    EscalatingAgent,
    EscalatingPerStepAgent,
    ReactAgent,
    ReflexionAgent,
    SelfRefineAgent,
    TreeSearchAgent,
)
from dse.verifier import TestVerifier


def _task(catalog):
    return next(iter(catalog.values()))


def _agents(llm, catalog, models, config=None, include_moa=True):
    config = config or EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    return build_default_agents(llm, verifier, config, models, env, include_moa=include_moa)


# ---------------------------------------------------------------------------
# Structural invariants (shared by every strategy)
# ---------------------------------------------------------------------------
def test_all_agents_produce_valid_run_results(stack):
    catalog, models, _, env, verifier, agents, budget = (
        stack[1], stack[2], stack[3], stack[4], stack[5], stack[6], stack[7],
    )
    task = _task(catalog)
    for agent in agents:
        result = agent.solve(task, budget)
        assert result.strategy == agent.name
        assert isinstance(result.success, bool)
        assert isinstance(result.answer, str) and result.answer
        assert 0.0 <= result.verifier_score <= 1.0
        assert len(result.steps) >= 1
        assert result.tokens_in > 0 and result.tokens_out > 0
        assert result.latency_s >= 0.0
        assert result.attempts >= 1
        assert result.tokens_total == result.tokens_in + result.tokens_out


# ---------------------------------------------------------------------------
# Deterministic success with a perfect model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cls",
    [
        ReactAgent,
        BestOfNAgent,
        ReflexionAgent,
        SelfRefineAgent,
        TreeSearchAgent,
        EscalatingAgent,
        EscalatingPerStepAgent,
        AdaptiveAgent,
        MoAAgent,
    ],
)
def test_perfect_model_succeeds(cls, tiny_catalog):
    """With step_accuracy=1.0 the MockLLM must answer every step correctly, so
    every strategy must succeed. The mock is built over the *same* catalog as
    the task (catalog id must match the task's ground truth)."""
    models = {
        "cheap": ModelConfig(name="cheap", step_accuracy=1.0, judge_accuracy=1.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=1.0, judge_accuracy=1.0),
    }
    llm = MockLLM(tiny_catalog, models, seed=0, p_fix=1.0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(tiny_catalog)
    verifier = TestVerifier(env)
    agent = cls(llm, verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog))
    assert result.success is True


# ---------------------------------------------------------------------------
# External-feedback mechanism (Huang et al. 2023): retries only help WITH signal
# ---------------------------------------------------------------------------
def _feedback_llm(catalog, step_accuracy, p_fix):
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=step_accuracy)}
    return MockLLM(catalog, models, seed=4, p_fix=p_fix), models


def test_reflexion_repairs_with_external_feedback(tiny_catalog):
    """step_accuracy=0 (always wrong first), p_fix=1.0 (feedback repairs).
    Reflexion must succeed on trial 2, with attempts == 2."""
    llm, models = _feedback_llm(tiny_catalog, step_accuracy=0.0, p_fix=1.0)
    config = EngineConfig(seed=0, flags=default_flags(), max_trials=3)
    env = Environment(tiny_catalog)
    verifier = TestVerifier(env)
    agent = ReflexionAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog))
    assert result.success is True
    assert result.attempts == 2


def test_reflexion_does_not_improve_without_repair_signal(tiny_catalog):
    """p_fix=0 and step_accuracy=0 → retries are independent and all fail.
    This encodes the Huang et al. finding: no external signal, no improvement."""
    llm, models = _feedback_llm(tiny_catalog, step_accuracy=0.0, p_fix=0.0)
    config = EngineConfig(seed=0, flags=default_flags(), max_trials=3)
    env = Environment(tiny_catalog)
    verifier = TestVerifier(env)
    agent = ReflexionAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog))
    assert result.success is False
    assert result.attempts == 3


def test_self_refine_gated_refinement(tiny_catalog):
    """Refinement only improves when fed the external failure signal."""
    llm, models = _feedback_llm(tiny_catalog, step_accuracy=0.0, p_fix=1.0)
    config = EngineConfig(seed=0, flags=default_flags(), max_trials=3)
    env = Environment(tiny_catalog)
    verifier = TestVerifier(env)
    agent = SelfRefineAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog))
    assert result.success is True
    assert result.attempts == 2


# ---------------------------------------------------------------------------
# Tree search specifics
# ---------------------------------------------------------------------------
def test_tree_search_produces_step_lines(tiny_catalog, stack):
    llm = stack[3]
    config = stack[0]
    models = stack[2]
    env = stack[4]
    verifier = stack[5]
    agent = TreeSearchAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog), Budget(max_search_nodes=32))
    assert "STEP" in result.answer
    assert result.latency_s >= 0.0


def test_tree_search_respects_node_budget():
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=4)
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=0.9)}
    llm = MockLLM(catalog, models, seed=0)
    config = EngineConfig(seed=0, flags=default_flags(), max_search_nodes=1000)
    env = Environment(catalog)
    verifier = TestVerifier(env)
    agent = TreeSearchAgent(llm, verifier, config, models, env)
    # budget of 4 nodes → at most (max_nodes-1) expansions; each expansion burns
    # at most BRANCH*2 = 6 proposal calls (dedup attempts). Correct bound:
    # actions <= 6 * max_nodes.
    result = agent.solve(_task(catalog), Budget(max_search_nodes=4, max_trials=1))
    action_steps = [s for s in result.steps if s.kind == "action"]
    assert len(action_steps) <= 6 * 4


def test_tree_search_select_explores_alternate_branches():
    """True-MCTS property: a fully-evaluated (terminal) branch must not block
    selection of an expandable sibling. Build the tree by hand and assert the
    selection returns the expandable leaf, not the dead subtree."""
    from dse.strategies.tree_search import TreeSearchAgent, _Node

    config = EngineConfig(seed=0, flags=default_flags())
    agent = TreeSearchAgent(None, None, config, {})  # _select needs no llm/verifier

    root = _Node("<root>", 0.0, None, depth=0)
    # Branch A: correct edge but its only child is terminal (depth == num_steps)
    a = _Node("a", 1.0, root, depth=1)
    a_term = _Node("a1", 0.0, a, depth=2)
    a.children = [a_term]
    # Branch B: correct edge, expandable leaf (depth 1 < 2, no children)
    b = _Node("b", 1.0, root, depth=1)
    root.children = [a, b]

    selected = agent._select(root, num_steps=2)
    assert selected is b

    # once B is expanded and its children are terminal too, nothing remains
    b.children = [_Node("b1", 1.0, b, depth=2)]
    assert agent._select(root, num_steps=2) is None


# ---------------------------------------------------------------------------
# Adaptive allocation
# ---------------------------------------------------------------------------
def test_adaptive_sets_route_tier(tiny_catalog, stack):
    config = stack[0]
    models = stack[2]
    env = stack[4]
    verifier = stack[5]
    agent = AdaptiveAgent(stack[3], verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog))
    assert result.route_tier in {"low", "medium", "high"}


def test_adaptive_probe_telemetry_merged(tiny_catalog):
    """The probe's cost must be visible in the returned run."""
    llm, models = _feedback_llm(tiny_catalog, step_accuracy=0.0, p_fix=1.0)
    config = EngineConfig(seed=0, flags=default_flags(), max_trials=3)
    env = Environment(tiny_catalog)
    verifier = TestVerifier(env)
    agent = AdaptiveAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(tiny_catalog))
    kinds = [s.kind for s in result.steps]
    assert "probe" in kinds
    assert result.tokens_in > 0


# ---------------------------------------------------------------------------
# Best-of-N (canonical test-time-compute baseline)
# ---------------------------------------------------------------------------
def test_best_of_n_reports_n_attempts(tiny_catalog, stack):
    llm = stack[3]
    config = stack[0]
    models = stack[2]
    env = stack[4]
    verifier = stack[5]
    agent = BestOfNAgent(llm, verifier, config, models, env, n=4)
    result = agent.solve(_task(tiny_catalog))
    assert result.strategy == "best_of_n"
    assert result.attempts == 4
    assert len([s for s in result.steps if s.kind == "sample"]) == 4


def test_best_of_n_selects_best_verifier_draft(tiny_catalog):
    """Deterministic proof of the selection logic: with scripted drafts and
    per-answer verifier scores, the returned answer must be the max-score one."""
    from dse.llm import Completion
    from dse.verifier import Verdict

    outputs = ["draft-low", "draft-mid", "draft-high"]
    scores = {"draft-low": 0.2, "draft-mid": 0.5, "draft-high": 1.0}

    class _ScriptedLLM:
        def complete(self, messages, *, model="cheap", temperature=0.0, max_tokens=512):
            text = outputs[len(self._seen) % len(outputs)]
            self._seen.append(text)
            return Completion(text=text, tokens_in=10, tokens_out=10, latency_s=0.01, model=model)

        def __init__(self):
            self._seen = []

    class _ScriptedVerifier:
        def score(self, answer, task, context=None):
            s = scores.get(answer, 0.0)
            return Verdict(score=s, passed=s >= 1.0, details={})

    config = EngineConfig(seed=0, flags=default_flags())
    agent = BestOfNAgent(_ScriptedLLM(), _ScriptedVerifier(), config, {})
    result = agent.solve(_task(tiny_catalog))
    assert result.answer == "draft-high"
    assert result.success is True
    assert result.attempts == 3


# ---------------------------------------------------------------------------
# Escalating (dynamic model routing)
# ---------------------------------------------------------------------------
def test_escalating_escalates_on_low_confidence():
    """Cheap tier always wrong, expensive tier always right => the router must
    escalate and the run must succeed on attempt 2 at the high tier."""
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=2)
    models = {
        "cheap": ModelConfig(name="cheap", step_accuracy=0.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=1.0),
    }
    llm = MockLLM(catalog, models, seed=0, p_fix=0.0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    agent = EscalatingAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(catalog))
    assert result.success is True
    assert result.attempts == 2
    assert result.route_tier == "high"
    assert any(s.kind == "escalate" for s in result.steps)


def test_escalating_per_step_repairs_failed_steps():
    """Per-step escalation: cheap tier fails every step; the expensive tier
    re-proposes ONLY the failing steps → deterministic success on attempt 2."""
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=2)
    models = {
        "cheap": ModelConfig(name="cheap", step_accuracy=0.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=1.0),
    }
    llm = MockLLM(catalog, models, seed=0, p_fix=0.0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    agent = EscalatingPerStepAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(catalog))
    assert result.success is True
    assert result.attempts == 2
    assert any(s.kind == "escalate_step" for s in result.steps)


def test_escalating_per_step_keeps_verified_steps():
    """Only failing steps are escalated — one expensive call per failed step.
    With cheap step_accuracy=0.0 every step fails on the first draft, so the
    number of escalate_step events must equal the task's step count (and no
    extra full re-drafts happen)."""
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=3)
    models = {
        "cheap": ModelConfig(name="cheap", step_accuracy=0.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=1.0),
    }
    llm = MockLLM(catalog, models, seed=0, p_fix=0.0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    task = _task(catalog)
    agent = EscalatingPerStepAgent(llm, verifier, config, models, env)
    result = agent.solve(task)
    escalations = [s for s in result.steps if s.kind == "escalate_step"]
    # one expensive call per failed step, and every step failed on the cheap draft
    assert len(escalations) == task.num_steps
    assert result.success is True


def test_adaptive_unknown_policy_raises():
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=2)
    models = {"cheap": ModelConfig(name="cheap")}
    llm = MockLLM(catalog, models, seed=0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    with pytest.raises(ValueError, match="unknown adaptive policy"):
        AdaptiveAgent(llm, verifier, config, models, env, policy="bogus")


def test_adaptive_difficulty_policy_routes_hard_to_search():
    """Difficulty policy: a hard task (num_steps >= 3) whose probe fails must
    go straight to the high (search) tier."""
    catalog = make_catalog(seed=0, n_tasks=6, max_steps=4)
    task = next(t for t in catalog.values() if t.num_steps >= 3)
    models = {
        "cheap": ModelConfig(name="cheap", step_accuracy=0.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=0.0),
    }
    llm = MockLLM(catalog, models, seed=0, p_fix=0.0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    agent = AdaptiveAgent(llm, verifier, config, models, env, policy="difficulty")
    result = agent.solve(task)
    assert result.route_tier == "high"
    assert result.strategy == "adaptive"


def test_tree_search_noisy_uses_judge_when_flagged():
    """With the llm_judge flag, tree search must evaluate steps via the LLM
    judge (emitting 'judge' events); without the flag it must not."""
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=3)
    models = {"cheap": ModelConfig(name="cheap", step_accuracy=0.9, judge_accuracy=0.9)}
    llm = MockLLM(catalog, models, seed=0)
    env = Environment(catalog)
    verifier = TestVerifier(env)
    task = _task(catalog)

    noisy_config = EngineConfig(seed=0, flags={**default_flags(), "llm_judge": True})
    noisy = TreeSearchAgent(llm, verifier, noisy_config, models, env)
    noisy_result = noisy.solve(task, Budget(max_search_nodes=16))
    assert any(s.kind == "judge" for s in noisy_result.steps)

    clean_config = EngineConfig(seed=0, flags=default_flags())
    clean = TreeSearchAgent(llm, verifier, clean_config, models, env)
    clean_result = clean.solve(task, Budget(max_search_nodes=16))
    assert not any(s.kind == "judge" for s in clean_result.steps)


def test_escalating_stays_cheap_when_confident():
    """High-confidence first draft => no escalation, one attempt, low tier."""
    catalog = make_catalog(seed=0, n_tasks=1, max_steps=2)
    models = {
        "cheap": ModelConfig(name="cheap", step_accuracy=1.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=1.0),
    }
    llm = MockLLM(catalog, models, seed=0, p_fix=0.0)
    config = EngineConfig(seed=0, flags=default_flags())
    env = Environment(catalog)
    verifier = TestVerifier(env)
    agent = EscalatingAgent(llm, verifier, config, models, env)
    result = agent.solve(_task(catalog))
    assert result.success is True
    assert result.attempts == 1
    assert result.route_tier == "low"
    assert not any(s.kind == "escalate" for s in result.steps)
