"""Settings loaded from environment variables (prefix ``AGENT_``)."""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["demo", "openai", "anthropic", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    # "demo" is an offline, rule-based chat model so the service runs without API keys.
    llm_provider: LLMProvider = "demo"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_base_url: str = "http://localhost:11434"

    max_plan_steps: int = Field(default=4, ge=1, le=10)
    max_revisions: int = Field(default=2, ge=0, le=5)
    worker_max_iterations: int = Field(default=4, ge=1, le=10)

    checkpoint_db: Path = Path("data/checkpoints.sqlite")
    api_key: SecretStr | None = None  # when set, required in the X-API-Key header

    # Fixed "today" for reproducible demos/tests; defaults to the real date.
    today: date | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
