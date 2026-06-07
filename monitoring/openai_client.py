"""Server-side OpenAI env loading and client factory (never expose keys to clients)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_local_dotenv() -> None:
    """Load project .env for local development. On Vercel, platform env vars are used."""
    if os.environ.get("VERCEL") == "1":
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)


def get_openai_api_key() -> str:
    """Return OPENAI_API_KEY from os.environ (after optional local .env load)."""
    ensure_local_dotenv()
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def log_openai_key_status(context: str = "") -> bool:
    """Log whether OPENAI_API_KEY is present — never log the key value."""
    loaded = bool(get_openai_api_key())
    if context:
        logger.info("%s: OPENAI_API_KEY loaded: %s", context, loaded)
    else:
        logger.info("OPENAI_API_KEY loaded: %s", loaded)
    return loaded


def create_openai_client():
    """Build an OpenAI SDK client or return None if the key is missing."""
    api_key = get_openai_api_key()
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)
