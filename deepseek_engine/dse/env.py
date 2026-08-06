"""Tiny .env loader (stdlib only).

Loads ``KEY=VALUE`` lines from ``.env`` files into ``os.environ`` without
overriding values that are already set (so a real shell environment always
wins). Comments (``#``) and blank lines are ignored; values may be wrapped in
single or double quotes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels up from this file (dse/env.py -> project root).
_DEFAULT_PATHS = (
    Path.cwd() / ".env",
    Path(__file__).resolve().parent.parent / ".env",
)


def load_env(paths: tuple[Path, ...] | None = None) -> list[str]:
    """Load the first existing ``.env`` file in ``paths``.

    Returns the list of keys loaded (already-set keys are skipped, so they are
    NOT included). Calling twice is safe and cheap.
    """
    loaded: list[str] = []
    for path in paths or _DEFAULT_PATHS:
        path = Path(path)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded
