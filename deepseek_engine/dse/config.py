"""Configuration, feature flags, and the experiment registry.

Every experimental technique is declared as an ``Experiment`` with an explicit
hypothesis, expected benefit, possible downside, benchmark plan, and rollback
condition (per the Master Brief's Experimentation section). Experiments that do
not beat the baseline in ``dse.benchmarks.harness`` must be removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EffortTier(str, Enum):
    """Compute regime chosen by the adaptive allocator (Snell et al. 2024)."""

    LOW = "low"        # single attempt, cheap model
    MEDIUM = "medium"  # retry / refine with feedback
    HIGH = "high"      # tree search


@dataclass(frozen=True)
class Experiment:
    """Metadata for an experimental feature (feature flag)."""

    flag: str
    hypothesis: str
    expected_benefit: str
    possible_downside: str
    benchmark_plan: str
    rollback_condition: str


@dataclass
class ModelConfig:
    """Cost/behaviour profile for a model tier used by the engine."""

    name: str
    cost_per_1k_in: float = 0.0   # USD per 1k input tokens
    cost_per_1k_out: float = 0.0  # USD per 1k output tokens
    # MockLLM calibration: probability a single reasoning step is correct.
    step_accuracy: float = 0.5
    # MockLLM calibration: probability a judge call scores a draft correctly.
    judge_accuracy: float = 0.8
    # Real-model name for provider-backed runs (e.g. "deepseek-chat"). When set,
    # the OpenAI-compatible provider sends THIS name; the tier key ("cheap" /
    # "expensive") is only an internal label. None ⇒ mock-only calibration.
    provider_model: str | None = None


def provider_models(
    cheap_model: str = "deepseek-v4-flash",
    expensive_model: str = "deepseek-v4-flash",
    cheap_cost_in: float = 0.00027,
    cheap_cost_out: float = 0.0011,
    expensive_cost_in: float = 0.00055,
    expensive_cost_out: float = 0.00219,
) -> dict[str, ModelConfig]:
    """Model tiers for provider-backed (real-LLM) runs.

    ``name`` stays the internal tier key used by strategies; ``provider_model``
    is what actually gets sent to the API. Both tiers default to
    ``deepseek-v4-flash`` (the user's verified fast model); pass a different
    ``expensive_model`` (e.g. ``deepseek-v4-pro``) for a stronger judge/escalation
    tier. The USD costs are *example* rates — update them to match your plan;
    they only affect cost telemetry, not behavior.
    """
    return {
        "cheap": ModelConfig(
            name="cheap",
            cost_per_1k_in=cheap_cost_in,
            cost_per_1k_out=cheap_cost_out,
            provider_model=cheap_model,
        ),
        "expensive": ModelConfig(
            name="expensive",
            cost_per_1k_in=expensive_cost_in,
            cost_per_1k_out=expensive_cost_out,
            provider_model=expensive_model,
        ),
    }


@dataclass
class RouterConfig:
    """Confidence-based model escalation (see dse/router.py).

    ``cheap.name`` / ``expensive.name`` are the keys strategies use when calling
    the LLM; they must match the model dict passed to the engine/benchmark.
    """

    cheap: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="cheap", cost_per_1k_in=0.001, cost_per_1k_out=0.002, step_accuracy=0.5, judge_accuracy=0.75)
    )
    expensive: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="expensive", cost_per_1k_in=0.01, cost_per_1k_out=0.03, step_accuracy=0.8, judge_accuracy=0.9)
    )
    escalate_confidence_below: float = 0.5
    max_escalations: int = 2


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    seed: int = 0
    max_trials: int = 3
    max_search_nodes: int = 64
    max_tokens_per_call: int = 512
    memory_window_tokens: int = 4096
    memory_summary_every: int = 8          # steps between summary triggers
    multi_agent_proposers: int = 3
    router: RouterConfig = field(default_factory=RouterConfig)
    flags: dict[str, bool] = field(default_factory=dict)

    def enabled(self, flag: str) -> bool:
        return bool(self.flags.get(flag, False))


# ---------------------------------------------------------------------------
# Experiment registry. Each entry is a feature flag consumed by the engine.
# ---------------------------------------------------------------------------
ENGINE_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        flag="adaptive_compute",
        hypothesis="Allocating test-time compute per-prompt difficulty is more "
        "compute-efficient than a uniform budget (Snell et al. 2024).",
        expected_benefit=">=2x fewer tokens for equal success on mixed-difficulty suites.",
        possible_downside="Difficulty probe adds one cheap call per task; mis-gating "
        "could under-allocate compute to hard tasks.",
        benchmark_plan="Adaptive vs uniform-budget baseline on mixed suite; compare "
        "success@tokens.",
        rollback_condition="Adaptive success rate < baseline - 2pp or tokens >= baseline "
        "on the same suite.",
    ),
    Experiment(
        flag="self_consistency",
        hypothesis="Majority vote over sampled paths is a cheap, robust weak verifier "
        "(Wang et al. 2022).",
        expected_benefit="Higher verifier precision than a single LLM judge for low cost.",
        possible_downside="Nx output tokens per verdict.",
        benchmark_plan="Vote-based selection vs single-draft selection on the suite.",
        rollback_condition="No accuracy gain at same token budget.",
    ),
    Experiment(
        flag="llm_judge",
        hypothesis="An LLM value function enables search when exact tests are absent "
        "(LATS).",
        expected_benefit="Search works on tasks without deterministic tests.",
        possible_downside="Judge noise propagates into search; costs tokens.",
        benchmark_plan="Search with LLM judge vs search with exact verifier; report "
        "gap attributable to judge noise.",
        rollback_condition="Judge-gated search < exact-gated search - 10pp.",
    ),
    Experiment(
        flag="multi_agent",
        hypothesis="Layered proposer+aggregator improves quality but costs Nx tokens "
        "(MoA, Wang et al. 2024).",
        expected_benefit="Higher success than single-agent at equal model tier.",
        possible_downside="Token cost scales with proposer count; cascading errors "
        "without verifier gating (MetaGPT).",
        benchmark_plan="MoA-lite vs single-agent on the suite; report success AND cost.",
        rollback_condition="Cost per win > 3x single-agent, or success < single-agent.",
    ),
    Experiment(
        flag="search_reflection",
        hypothesis="Reflecting on terminal failures improves subsequent search "
        "branches (LATS).",
        expected_benefit="Higher search success at equal node budget.",
        possible_downside="Reflection calls add latency/tokens.",
        benchmark_plan="TreeSearch with vs without reflection on hard tasks.",
        rollback_condition="No success gain at equal node budget.",
    ),
)


def default_flags() -> dict[str, bool]:
    """Defaults: established techniques on; experimental ones off.

    - ``adaptive_compute``: on (established, Snell et al. 2024).
    - ``self_consistency``: on (established, Wang et al. 2022).
    - ``llm_judge``: off (experimental; exact tests preferred when available).
    - ``multi_agent``: off (experimental; cost-heavy).
    - ``search_reflection``: on (established, LATS).
    """
    return {
        "adaptive_compute": True,
        "self_consistency": True,
        "llm_judge": False,
        "multi_agent": False,
        "search_reflection": True,
    }


def validate_flags(flags: dict[str, bool]) -> None:
    """Raise if a flag is unknown (typo protection)."""
    known = {e.flag for e in ENGINE_EXPERIMENTS}
    unknown = set(flags) - known
    if unknown:
        raise ValueError(f"Unknown feature flags: {sorted(unknown)}; known: {sorted(known)}")
