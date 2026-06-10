"""Load environment variables from .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env() -> None:
    """Load .env from the project root if it exists."""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)


def get_env(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_env_int(key: str, default: int | None = None) -> int | None:
    value = get_env(key)
    if value is None:
        return default
    return int(value)
