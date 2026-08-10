"""Assembly factory: build a complete, reproducible benchmark stack.

This is the single place that wires config → catalog → models → MockLLM →
environment → verifier → agents, so tests and the CLI share identical wiring
(no drift between what is tested and what is benchmarked).
"""

from __future__ import annotations

import os

from .agent import Agent, Budget, BaseAgent
from .config import EngineConfig, ModelConfig, default_flags, provider_models, validate_flags
from .environment import Environment, Task
from .llm import LLM, MockLLM, default_models
from .verifier import TestVerifier, Verifier


def build_default_agents(
    llm: LLM,
    verifier: Verifier,
    config: EngineConfig,
    models: dict[str, ModelConfig],
    env: Environment | None = None,
    include_moa: bool = False,
    adaptive_policy: str = "difficulty",
) -> list[Agent]:
    """The default strategy set for benchmarks."""
    from .orchestration import MoAAgent
    from .strategies import (
        AdaptiveAgent,
        BestOfNAgent,
        EscalatingAgent,
        EscalatingPerStepAgent,
        ReactAgent,
        ReflexionAgent,
        SelfRefineAgent,
        TreeSearchAgent,
    )

    agents: list[Agent] = [
        ReactAgent(llm, verifier, config, models, env),
        BestOfNAgent(llm, verifier, config, models, env),
        ReflexionAgent(llm, verifier, config, models, env),
        SelfRefineAgent(llm, verifier, config, models, env),
        TreeSearchAgent(llm, verifier, config, models, env),
        EscalatingAgent(llm, verifier, config, models, env),
        EscalatingPerStepAgent(llm, verifier, config, models, env),
        AdaptiveAgent(llm, verifier, config, models, env, policy=adaptive_policy),
    ]
    if include_moa or config.enabled("multi_agent"):
        agents.append(MoAAgent(llm, verifier, config, models, env))
    return agents


def build_stack(
    seed: int = 0,
    n_tasks: int = 48,
    max_steps: int = 5,
    min_steps: int = 1,
    flags: dict[str, bool] | None = None,
    include_moa: bool = False,
    budget: Budget | None = None,
    provider: str = "mock",
    adaptive_policy: str = "difficulty",
    suite: str = "all",
) -> tuple[EngineConfig, dict[str, Task], dict[str, ModelConfig], LLM, Environment, Verifier, list[Agent], Budget]:
    """Build the full stack in one call.

    ``provider`` is one of "mock" (default), "deepseek", "ollama", "openai",
    "github" — non-mock providers build an OpenAI-compatible client backed by
    the ``DSE_*`` environment variables / ``.env`` (see dse/providers.py).

    ``suite`` is "all"/"easy"/"hard" (synthetic calibrated tasks), "real"
    (natural-language tasks answerable by a real LLM; tree_search/adaptive are
    excluded because single-unit tasks have no steps to search over), "hard-real"
    (multi-step tasks with deterministic checkers — see dse/hardtasks.py),
    "hard-tuned" (tasks targeted at a strong reasoning model's weak spots), or
    "code" (code-generation tasks checked by EXECUTING the model's code
    against hidden tests — see dse/codetasks.py).

    Returns ``(config, catalog, models, llm, env, verifier, agents, budget)``.
    """
    from .benchmarks.tasks import make_catalog
    from .codetasks import make_code_catalog
    from .env import load_env
    from .failtasks import make_fail_catalog
    from .hardtasks import make_hard_catalog, make_hard_tuned_catalog
    from .realtasks import REAL_SUITE_AGENTS, make_real_catalog
    from .verifier import RealTaskVerifier

    load_env()
    flags = flags if flags is not None else default_flags()
    validate_flags(flags)
    config = EngineConfig(seed=seed, flags=dict(flags))

    if suite == "real":
        catalog: dict[str, Task] = make_real_catalog(seed=seed)
        env = Environment({})
        verifier: Verifier = RealTaskVerifier()
    elif suite == "fail":
        catalog = make_fail_catalog(seed=seed)
        env = Environment({})
        verifier = RealTaskVerifier()
    elif suite == "code":
        catalog = make_code_catalog(seed=seed)
        env = Environment({})
        verifier = RealTaskVerifier()
    elif suite in ("hard-real", "hard-tuned"):
        catalog = (make_hard_tuned_catalog(seed=seed) if suite == "hard-tuned"
                   else make_hard_catalog(seed=seed))
        env = Environment({})
        verifier = RealTaskVerifier()
    else:
        catalog = make_catalog(
            seed=seed, n_tasks=n_tasks, min_steps=min_steps, max_steps=max_steps
        )
        env = Environment(catalog)
        verifier = TestVerifier(env)

    if provider == "mock":
        models = default_models()
        llm: LLM = MockLLM(catalog, models, seed=seed)
    else:
        from .providers import make_provider

        models = provider_models(
            cheap_model=os.environ.get("DSE_MODEL_CHEAP", "deepseek-chat"),
            expensive_model=os.environ.get("DSE_MODEL_EXPENSIVE", "deepseek-reasoner"),
        )
        llm = make_provider(provider, models=models)

    agents = build_default_agents(
        llm, verifier, config, models, env,
        include_moa=include_moa, adaptive_policy=adaptive_policy,
    )
    if suite in ("real", "hard-real", "hard-tuned", "code"):
        # per-step strategies are meaningless on single-unit real tasks
        agents = [a for a in agents if a.name in REAL_SUITE_AGENTS]
    budget = budget or Budget(
        max_trials=config.max_trials,
        max_search_nodes=config.max_search_nodes,
    )
    return config, catalog, models, llm, env, verifier, agents, budget


__all__ = ["build_default_agents", "build_stack", "BaseAgent"]
