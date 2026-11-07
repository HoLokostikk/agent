"""Model registry: loads ``configs/models.yaml`` into validated configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Ollama and vLLM ignore the key, but the openai SDK refuses an empty one.
_LOCAL_API_KEY = "not-needed"


class ConfigError(Exception):
    """The model registry is missing, malformed, or refers to a missing secret."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Pricing(_Strict):
    """USD per 1M tokens."""

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)
    # Providers that discount cached prefixes; ``None`` means "same as input".
    cached_input_per_mtok: float | None = Field(default=None, ge=0)


class Sampling(_Strict):
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    presence_penalty: float | None = None
    # Not part of the OpenAI API; sent through ``extra_body``.
    top_k: int | None = Field(default=None, gt=0)


class Timeouts(_Strict):
    """``read_s`` bounds the gap between bytes, ``total_s`` the whole call."""

    connect_s: float = Field(default=5.0, gt=0)
    read_s: float = Field(default=60.0, gt=0)
    total_s: float = Field(default=300.0, gt=0)


class RetryPolicy(_Strict):
    max_attempts: int = Field(default=4, ge=1)
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=20.0, ge=0)


class ModelConfig(_Strict):
    alias: str
    base_url: str
    model: str
    revision: str = Field(min_length=1)
    context_window: int = Field(gt=0)
    pricing: Pricing
    api_key_env: str | None = None
    sampling: Sampling = Sampling()
    extra_body: dict[str, Any] = Field(default_factory=dict)
    timeouts: Timeouts = Timeouts()
    retry: RetryPolicy = RetryPolicy()

    def resolve_api_key(self) -> str:
        """Read the key from the environment; never stored in the config."""
        if self.api_key_env is None:
            return _LOCAL_API_KEY
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise ConfigError(
                f"model {self.alias!r} needs {self.api_key_env} in the environment or .env"
            )
        return key


def load_models(path: Path) -> dict[str, ModelConfig]:
    """Load and validate every model in the registry, keyed by alias."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"model registry not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"model registry is not valid YAML: {path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise ConfigError(f"{path}: expected a top-level 'models' mapping")

    models: dict[str, ModelConfig] = {}
    for alias, entry in raw["models"].items():
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: model {alias!r} must be a mapping")
        try:
            models[alias] = ModelConfig(alias=alias, **entry)
        except ValidationError as exc:
            raise ConfigError(f"{path}: model {alias!r} is invalid:\n{exc}") from exc
    return models


def get_model(path: Path, alias: str) -> ModelConfig:
    models = load_models(path)
    try:
        return models[alias]
    except KeyError:
        known = ", ".join(sorted(models)) or "none"
        raise ConfigError(f"unknown model alias {alias!r}; known: {known}") from None
