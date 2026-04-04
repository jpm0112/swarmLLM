from __future__ import annotations

"""Backend profile loading and normalization."""

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from swarmllm.config import BackendKind, Config, LLMEndpoint

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


class RoleProfile(BaseModel):
    model: str
    temperature: float = Field(default=0.7, ge=0.0)
    max_tokens: int = Field(default=4096, gt=0)


class EndpointProfile(BaseModel):
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    weight: int = Field(default=1, ge=1)
    label: str | None = None

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return normalize_openai_base_url(value)


class BackendProfile(BaseModel):
    name: str
    kind: Literal["ollama", "vllm-metal", "vllm", "mlx-lm", "groq", "together"]
    request_timeout: int = Field(default=300, gt=0)
    default_max_concurrent_agents: int | None = Field(default=None, gt=0)
    coordinator: RoleProfile
    worker: RoleProfile
    coordinator_endpoints: list[EndpointProfile]
    worker_endpoints: list[EndpointProfile]

    @model_validator(mode="after")
    def _validate_endpoints(self) -> "BackendProfile":
        if not self.coordinator_endpoints:
            raise ValueError("Backend profiles must define at least one coordinator endpoint.")
        if not self.worker_endpoints:
            raise ValueError("Backend profiles must define at least one worker endpoint.")
        return self


def normalize_openai_base_url(url: str) -> str:
    """Normalize OpenAI-compatible endpoint URLs to include `/v1`."""
    normalized = url.strip().rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def is_loopback_base_url(url: str) -> bool:
    """Return whether a base URL points at a local loopback interface."""
    hostname = urlsplit(normalize_openai_base_url(url)).hostname
    return hostname in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def loopback_base_url_candidates(url: str) -> list[str]:
    """Return reasonable local aliases for loopback URLs.

    Some local model servers on macOS bind in ways that make `localhost`
    work while `127.0.0.1` does not, or vice versa. Trying both helps make
    the example backend profiles more forgiving without changing remote URLs.
    """
    normalized = normalize_openai_base_url(url)
    parts = urlsplit(normalized)
    hostname = parts.hostname
    if hostname not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return [normalized]

    candidates = [normalized]
    alias_map = {
        "127.0.0.1": ["localhost"],
        "localhost": ["127.0.0.1"],
        "::1": ["localhost", "127.0.0.1"],
        "0.0.0.0": ["127.0.0.1", "localhost"],
    }
    for alias in alias_map.get(hostname, []):
        alias_host = f"[{alias}]" if ":" in alias else alias
        if parts.port is not None:
            netloc = f"{alias_host}:{parts.port}"
        else:
            netloc = alias_host
        candidates.append(urlunsplit(parts._replace(netloc=netloc)))
    return list(dict.fromkeys(candidates))


def resolve_api_key(endpoint: LLMEndpoint, backend_kind: BackendKind) -> str:
    """Resolve an API key from explicit value, environment, or backend defaults."""
    if endpoint.api_key_env:
        env_value = os.getenv(endpoint.api_key_env)
        if env_value:
            return env_value
    if endpoint.api_key:
        return endpoint.api_key
    if backend_kind == "ollama":
        return "ollama"
    return "EMPTY"


def load_backend_profile(path: str | os.PathLike[str]) -> BackendProfile:
    """Load and validate a backend profile from TOML."""
    profile_path = Path(path)
    with profile_path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return BackendProfile.model_validate(data)
    except ValidationError as exc:  # pragma: no cover - exercised in tests through string matching
        raise ValueError(f"Invalid backend profile at {profile_path}: {exc}") from exc


def apply_backend_profile(config: Config, path: str | os.PathLike[str]) -> BackendProfile:
    """Apply a backend profile to the runtime config."""
    profile = load_backend_profile(path)
    config.llm.backend_kind = profile.kind
    config.llm.backend_profile_path = str(path)
    config.llm.backend_profile_name = profile.name
    config.llm.coordinator_model = profile.coordinator.model
    config.llm.agent_model = profile.worker.model
    config.llm.temperature_coordinator = profile.coordinator.temperature
    config.llm.temperature_worker = profile.worker.temperature
    config.llm.max_tokens_coordinator = profile.coordinator.max_tokens
    config.llm.max_tokens_worker = profile.worker.max_tokens
    config.llm.request_timeout = profile.request_timeout
    config.llm.coordinator_endpoints = [
        LLMEndpoint(**endpoint.model_dump()) for endpoint in profile.coordinator_endpoints
    ]
    config.llm.worker_endpoints = [
        LLMEndpoint(**endpoint.model_dump()) for endpoint in profile.worker_endpoints
    ]
    if profile.default_max_concurrent_agents is not None:
        config.swarm.max_concurrent_agents = profile.default_max_concurrent_agents
    return profile
