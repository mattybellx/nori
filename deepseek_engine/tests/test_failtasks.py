"""Fail-suite sanity: every gold answer must pass its own deterministic
checker, and the catalog must be stable regardless of seed."""

from __future__ import annotations

from dse.failtasks import make_fail_catalog
from dse.verifier import RealTaskVerifier


def test_fail_catalog_has_enough_tasks():
    cat = make_fail_catalog()
    assert len(cat) >= 12


def test_every_gold_passes_its_own_checker():
    v = RealTaskVerifier()
    cat = make_fail_catalog()
    for task in cat.values():
        verdict = v.score(f"ANSWER: {task.gold}", task)
        assert verdict.passed, f"{task.id}: gold {task.gold!r} failed its own checker"


def test_fail_catalog_is_deterministic():
    assert set(make_fail_catalog(seed=0)) == set(make_fail_catalog(seed=7))
