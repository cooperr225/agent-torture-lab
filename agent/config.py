"""Settings for the agent, loaded from the .env file.

The API key is stored as a SecretStr, so printing the settings object
(or an error that includes it) shows '**********' instead of the key.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when settings are missing or invalid. Never contains the key."""


class Settings(BaseModel):
    anthropic_api_key: SecretStr
    agent_model: str = "claude-haiku-5-5"
    judge_model: str = "claude-sonnet-5-5"
    max_steps: int = Field(default=10, ge=1, le=50)
    db_path: Path = PROJECT_ROOT / "runs" / "runs.db"


def load_settings(env_file: Path | None = None) -> Settings:
    """Read settings from .env (and the environment) and validate them."""
    load_dotenv(env_file or PROJECT_ROOT / ".env")

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key == "your-key-here":
        raise ConfigError("ANTHROPIC_API_KEY is missing. Set it in .env.")

    values: dict = {"anthropic_api_key": key}
    for field, env_name in [
        ("agent_model", "AGENT_MODEL"),
        ("judge_model", "JUDGE_MODEL"),
        ("max_steps", "MAX_STEPS"),
    ]:
        if os.environ.get(env_name):
            values[field] = os.environ[env_name]

    try:
        return Settings(**values)
    except ValueError as exc:
        # Report which fields are wrong, but never echo the input values.
        fields = ", ".join(str(e["loc"][0]) for e in exc.errors())  # type: ignore[attr-defined]
        raise ConfigError(f"Invalid settings: {fields}") from None
