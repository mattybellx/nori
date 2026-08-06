"""Tests for configuration, feature flags, and the experiment registry."""

import pytest

from dse.config import (
    ENGINE_EXPERIMENTS,
    EffortTier,
    EngineConfig,
    default_flags,
    validate_flags,
)


def test_default_flags_are_known():
    flags = default_flags()
    validate_flags(flags)  # raises if any unknown flag
    assert set(flags) == {e.flag for e in ENGINE_EXPERIMENTS}


def test_validate_flags_rejects_unknown():
    with pytest.raises(ValueError):
        validate_flags({"not_a_real_flag": True})


def test_every_experiment_has_required_metadata():
    for exp in ENGINE_EXPERIMENTS:
        assert exp.flag
        assert exp.hypothesis
        assert exp.expected_benefit
        assert exp.possible_downside
        assert exp.benchmark_plan
        assert exp.rollback_condition


def test_engine_config_enabled():
    cfg = EngineConfig(flags={"adaptive_compute": True})
    assert cfg.enabled("adaptive_compute")
    assert not cfg.enabled("llm_judge")


def test_established_techniques_on_by_default():
    flags = default_flags()
    # established → on; experimental → off
    assert flags["adaptive_compute"] is True
    assert flags["search_reflection"] is True
    assert flags["self_consistency"] is True
    assert flags["multi_agent"] is False
    assert flags["llm_judge"] is False


def test_effort_tier_enum():
    assert EffortTier.LOW.value == "low"
    assert EffortTier.HIGH.value == "high"
