"""Tests for the tiny .env loader."""

import os
from pathlib import Path

from dse.env import load_env


def test_load_env_parses(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "DSE_PROVIDER_KEY=sk-abc123\n"
        "DSE_MODEL_CHEAP='deepseek-chat'\n"
        'DSE_TIMEOUT="90"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DSE_PROVIDER_KEY", raising=False)
    monkeypatch.delenv("DSE_MODEL_CHEAP", raising=False)
    monkeypatch.delenv("DSE_TIMEOUT", raising=False)

    loaded = load_env(paths=(env_file,))
    assert "DSE_PROVIDER_KEY" in loaded
    assert os.environ["DSE_PROVIDER_KEY"] == "sk-abc123"
    assert os.environ["DSE_MODEL_CHEAP"] == "deepseek-chat"
    assert os.environ["DSE_TIMEOUT"] == "90"


def test_load_env_does_not_override_existing(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DSE_PROVIDER_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("DSE_PROVIDER_KEY", "from-shell")
    loaded = load_env(paths=(env_file,))
    assert "DSE_PROVIDER_KEY" not in loaded
    assert os.environ["DSE_PROVIDER_KEY"] == "from-shell"


def test_load_env_missing_file_is_noop():
    assert load_env(paths=(Path("definitely-missing.env"),)) == []
