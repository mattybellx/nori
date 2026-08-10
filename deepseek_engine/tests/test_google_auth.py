"""Tests for the Google OAuth + Drive history-sync module (pure functions)."""

import json

from dse import google_auth


def test_build_auth_url_contains_required_params():
    url = google_auth.build_auth_url("cid-123", "http://127.0.0.1:8787/auth/callback")
    assert "client_id=cid-123" in url
    assert "redirect_uri=" in url
    assert "response_type=code" in url
    assert "access_type=offline" in url
    assert "drive.appdata" in url
    assert "http%3A%2F%2F127.0.0.1%3A8787%2Fauth%2Fcallback" in url


def test_configured_from_settings_or_env(monkeypatch):
    assert google_auth.configured({"google_client_id": "", "google_client_secret": ""}) is False
    assert google_auth.configured({"google_client_id": "abc", "google_client_secret": "x"}) is True
    monkeypatch.setenv("DSE_GOOGLE_CLIENT_ID", "env-cid")
    assert google_auth.configured({}) is True


def test_bundle_sessions(tmp_path):
    (tmp_path / "ask_2026-01-01.jsonl").write_text(
        '{"question": "q1", "strategy": "react", "answer": "a", "ts": "t"}\n'
        '{"question": "q1", "strategy": "reflexion", "answer": "b", "ts": "t"}\n',
        encoding="utf-8",
    )
    (tmp_path / "ask_2026-01-02.jsonl").write_text(
        '{"question": "q2", "strategy": "react", "answer": "c", "ts": "u"}\n'
        "not-json\n"
        "\n",
        encoding="utf-8",
    )
    bundle = google_auth.bundle_sessions(tmp_path)
    assert bundle["app"] == "nori"
    assert len(bundle["records"]) == 3  # invalid line skipped
    assert {r["question"] for r in bundle["records"]} == {"q1", "q2"}


def test_merge_pull_dedupes_by_question_strategy_ts():
    local = [
        {"question": "q1", "strategy": "react", "ts": "t1"},
        {"question": "q2", "strategy": "react", "ts": "t2"},
    ]
    remote = [
        {"question": "q1", "strategy": "react", "ts": "t1"},   # duplicate -> skip
        {"question": "q2", "strategy": "react", "ts": "t2"},   # duplicate -> skip
        {"question": "q3", "strategy": "react", "ts": "t3"},   # new -> keep
        {"question": "q2", "strategy": "reflexion", "ts": "t2"},  # new -> keep
    ]
    fresh = google_auth.merge_pull(remote, local)
    assert len(fresh) == 2
    assert {r["question"] for r in fresh} == {"q3", "q2"}


def test_sync_pull_no_file_is_empty():
    # no network, no auth file id -> must return empty without raising
    assert google_auth.sync_pull("") == {"ok": True, "records": [], "file_id": None}
