from __future__ import annotations

"""Readiness checks for OpenAI-compatible model servers."""

from dataclasses import dataclass

import httpx

from swarmllm.config import LLMConfig, LLMEndpoint
from swarmllm.llm.profiles import (
    is_loopback_base_url,
    loopback_base_url_candidates,
    normalize_openai_base_url,
    resolve_api_key,
)


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
    api_key = resolve_api_key(endpoint, config.backend_kind)
    timeout = httpx.Timeout(config.request_timeout)
    attempted_urls: list[str] = []
    last_connect_error: httpx.ConnectError | None = None

    for base_url in loopback_base_url_candidates(endpoint.base_url):
        attempted_urls.append(base_url)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=transport,
                trust_env=not is_loopback_base_url(base_url),
            ) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
                payload = response.json()
            model_ids = [item["id"] for item in payload.get("data", []) if "id" in item]
            return BackendValidationResult(endpoint=base_url, available_models=model_ids)
        except httpx.ConnectError as exc:
            last_connect_error = exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(_format_status_error(base_url, exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to query {base_url}/models: {exc}") from exc

    attempted = ", ".join(attempted_urls)
    raise RuntimeError(_format_connect_error(config.backend_kind, endpoint.base_url, attempted)) from last_connect_error


async def validate_backend_or_raise(
    config: LLMConfig,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Fail fast if configured coordinator or worker models are not available."""
    coordinator_endpoint = config.coordinator_endpoints[0]
    coordinator_result = await fetch_available_models(coordinator_endpoint, config, transport=transport)
    coordinator_endpoint.base_url = coordinator_result.endpoint
    if config.coordinator_model not in coordinator_result.available_models:
        raise RuntimeError(
            f"Coordinator model '{config.coordinator_model}' is not available at "
            f"{coordinator_result.endpoint}. Found: {coordinator_result.available_models}"
        )

    for endpoint in config.worker_endpoints:
        worker_result = await fetch_available_models(endpoint, config, transport=transport)
        endpoint.base_url = worker_result.endpoint
        if config.agent_model not in worker_result.available_models:
            raise RuntimeError(
                f"Worker model '{config.agent_model}' is not available at "
                f"{worker_result.endpoint}. Found: {worker_result.available_models}"
            )


def _format_connect_error(backend_kind: str, configured_base_url: str, attempted_urls: str) -> str:
    normalized = normalize_openai_base_url(configured_base_url)
    backend_help = {
        "ollama": (
            "Make sure Ollama is running and exposing its OpenAI-compatible API, "
            "for example with `ollama serve`."
        ),
        "vllm-metal": (
            "Make sure your Apple Silicon vLLM server is running and serving an "
            "OpenAI-compatible API on the configured port."
        ),
        "vllm": (
            "Make sure your vLLM server is running and serving an OpenAI-compatible "
            "API on the configured port."
        ),
        "mlx-lm": (
            "Make sure your mlx-lm server is running (`mlx_lm.server --model <model>`) "
            "and serving an OpenAI-compatible API on the configured port."
        ),
    }[backend_kind]
    return (
        f"Could not connect to the configured {backend_kind} backend at {normalized}. "
        f"Tried: {attempted_urls}. {backend_help} "
        "If the server is already running on macOS, update the backend profile to use "
        "the address it actually binds to, usually `http://localhost:.../v1` or "
        "`http://127.0.0.1:.../v1`."
    )


def _format_status_error(base_url: str, exc: httpx.HTTPStatusError) -> str:
    return (
        f"The backend at {base_url} responded to `/models` with HTTP "
        f"{exc.response.status_code}. Verify the OpenAI-compatible base URL and API key."
    )
