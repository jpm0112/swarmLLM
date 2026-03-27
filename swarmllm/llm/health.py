from __future__ import annotations

"""Readiness checks for OpenAI-compatible model servers."""

from dataclasses import dataclass

import httpx

from swarmllm.config import LLMConfig, LLMEndpoint
from swarmllm.llm.profiles import normalize_openai_base_url, resolve_api_key


@dataclass
class BackendValidationResult:
    endpoint: str
    available_models: list[str]


async def fetch_available_models(
    endpoint: LLMEndpoint,
    config: LLMConfig,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BackendValidationResult:
    """Fetch available model ids from an OpenAI-compatible `/v1/models` endpoint."""
    base_url = normalize_openai_base_url(endpoint.base_url)
    api_key = resolve_api_key(endpoint, config.backend_kind)
    timeout = httpx.Timeout(config.request_timeout)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        response = await client.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        payload = response.json()
    model_ids = [item["id"] for item in payload.get("data", []) if "id" in item]
    return BackendValidationResult(endpoint=base_url, available_models=model_ids)


async def validate_backend_or_raise(
    config: LLMConfig,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Fail fast if configured coordinator or worker models are not available."""
    coordinator_endpoint = config.coordinator_endpoints[0]
    coordinator_result = await fetch_available_models(coordinator_endpoint, config, transport=transport)
    if config.coordinator_model not in coordinator_result.available_models:
        raise RuntimeError(
            f"Coordinator model '{config.coordinator_model}' is not available at "
            f"{coordinator_result.endpoint}. Found: {coordinator_result.available_models}"
        )

    for endpoint in config.worker_endpoints:
        worker_result = await fetch_available_models(endpoint, config, transport=transport)
        if config.agent_model not in worker_result.available_models:
            raise RuntimeError(
                f"Worker model '{config.agent_model}' is not available at "
                f"{worker_result.endpoint}. Found: {worker_result.available_models}"
            )
