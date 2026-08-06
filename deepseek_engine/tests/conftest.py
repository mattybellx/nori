"""Shared fixtures: the full benchmark stack and small deterministic catalogs."""

import pytest

from dse.config import EngineConfig, ModelConfig
from dse.environment import Environment, make_catalog
from dse.llm import MockLLM
from dse.verifier import TestVerifier


@pytest.fixture(scope="session")
def stack():
    """The canonical, fully-wired stack (same wiring as the CLI)."""
    from dse.factory import build_stack

    return build_stack(seed=7, n_tasks=24, max_steps=4)


@pytest.fixture(scope="session")
def tiny_stack():
    """A small, self-consistent stack over ``tiny_catalog`` (seed=3, 8 tasks).

    All components (llm, env, verifier, agents) share the SAME catalog — mixing
    components from different stacks breaks the mock's task-id lookup.
    """
    from dse.factory import build_stack

    return build_stack(seed=3, n_tasks=8, max_steps=3)


@pytest.fixture
def catalog(stack):
    return stack[1]


@pytest.fixture
def models(stack):
    return stack[2]


@pytest.fixture
def llm(stack):
    return stack[3]


@pytest.fixture
def env(stack):
    return stack[4]


@pytest.fixture
def verifier(stack):
    return stack[5]


@pytest.fixture
def agents(stack):
    return stack[6]


@pytest.fixture
def config(stack):
    return stack[0]


@pytest.fixture
def budget(stack):
    return stack[7]


@pytest.fixture
def perfect_models():
    """All-cheap model tier with step_accuracy=1.0 → deterministic success."""
    return {
        "cheap": ModelConfig(name="cheap", step_accuracy=1.0, judge_accuracy=1.0),
        "expensive": ModelConfig(name="expensive", step_accuracy=1.0, judge_accuracy=1.0),
    }


@pytest.fixture
def perfect_llm(catalog, perfect_models):
    """MockLLM that always answers every step correctly (p_fix=1.0)."""
    return MockLLM(catalog, perfect_models, seed=1, p_fix=1.0)


@pytest.fixture
def tiny_catalog():
    """A tiny deterministic catalog for fast end-to-end tests."""
    return make_catalog(seed=3, n_tasks=8, max_steps=3)
