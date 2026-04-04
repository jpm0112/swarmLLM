from __future__ import annotations

"""PydanticAI model and agent factory helpers."""

import json
from typing import Any, cast

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from swarmllm.config import BackendKind, Config, LLMConfig, LLMEndpoint
from swarmllm.llm.profiles import is_loopback_base_url, normalize_openai_base_url, resolve_api_key
from swarmllm.llm.schemas import CoordinatorRoundPlan, WorkerDraft


_OPENAI_COMPAT_PROFILE = OpenAIModelProfile(
    openai_supports_strict_tool_definition=False,
)

_GROQ_PROFILE = OpenAIModelProfile(
    openai_supports_strict_tool_definition=False,
    supports_tools=False,
    default_structured_output_mode="prompted",
)

_HTTP_CLIENT_CACHE: dict[tuple[str, str, int], httpx.AsyncClient] = {}
_MODEL_CACHE: dict[tuple[str, str, str, int], OpenAIChatModel] = {}
_AGENT_CACHE: dict[tuple[str, str, str, str], Agent[Any, Any]] = {}


def clear_caches() -> None:
    """Clear factory caches. Primarily useful for tests."""
    _AGENT_CACHE.clear()
    _MODEL_CACHE.clear()
    _HTTP_CLIENT_CACHE.clear()


def build_worker_agent(config: Config, endpoint: LLMEndpoint, system_prompt: str) -> Agent[None, WorkerDraft]:
    """Build or reuse a cached worker agent for a specific endpoint/model pair."""
    model = _get_chat_model(config.llm, endpoint, config.llm.agent_model)
    cache_key = ("worker", model.model_name, normalize_openai_base_url(endpoint.base_url), system_prompt)
    if cache_key not in _AGENT_CACHE:
        _AGENT_CACHE[cache_key] = Agent(
            model,
            output_type=WorkerDraft,
            system_prompt=system_prompt,
            name="swarmllm_worker",
            retries=3,
            defer_model_check=True,
        )
    return cast(Agent[None, WorkerDraft], _AGENT_CACHE[cache_key])


def build_coordinator_agent(
    config: Config,
    endpoint: LLMEndpoint,
    system_prompt: str,
) -> Agent[None, CoordinatorRoundPlan]:
    """Build or reuse a cached coordinator agent for a specific endpoint/model pair."""
    model = _get_chat_model(config.llm, endpoint, config.llm.coordinator_model)
    cache_key = ("coordinator", model.model_name, normalize_openai_base_url(endpoint.base_url), system_prompt)
    if cache_key not in _AGENT_CACHE:
        _AGENT_CACHE[cache_key] = Agent(
            model,
            output_type=CoordinatorRoundPlan,
            system_prompt=system_prompt,
            name="swarmllm_coordinator",
            retries=3,
            defer_model_check=True,
        )
    return cast(Agent[None, CoordinatorRoundPlan], _AGENT_CACHE[cache_key])


def _get_chat_model(config: LLMConfig, endpoint: LLMEndpoint, model_name: str) -> OpenAIChatModel:
    base_url = normalize_openai_base_url(endpoint.base_url)
    api_key = resolve_api_key(endpoint, config.backend_kind)
    timeout = config.request_timeout
    cache_key = (base_url, api_key, model_name, timeout)
    if cache_key not in _MODEL_CACHE:
        http_client = _get_http_client(base_url, api_key, timeout, config.backend_kind)
        provider = OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
        )
        # Cloud endpoints (non-loopback) use prompted JSON output since tool
        # calling support varies across providers.  Local backends use standard
        # OpenAI-compatible tool mode.
        if not is_loopback_base_url(base_url):
            profile = _GROQ_PROFILE
        else:
            profile = _OPENAI_COMPAT_PROFILE
        _MODEL_CACHE[cache_key] = OpenAIChatModel(
            model_name,
            provider=provider,
            profile=profile,
        )
    return _MODEL_CACHE[cache_key]


def _get_http_client(
    base_url: str, api_key: str, timeout_seconds: int, backend_kind: BackendKind = "ollama",
) -> httpx.AsyncClient:
    cache_key = (base_url, api_key, timeout_seconds)
    if cache_key not in _HTTP_CLIENT_CACHE:
        is_loopback = is_loopback_base_url(base_url)
        transport: httpx.AsyncBaseTransport | None = None
        inner = httpx.AsyncHTTPTransport(trust_env=not is_loopback)
        if is_loopback:
            # Ollama rejects messages with null content — wrap the transport
            # to patch those to empty strings before the request is sent.
            transport = _OllamaFixTransport(inner)
        elif backend_kind == "groq":
            # Groq returns service_tier: "on_demand" which PydanticAI's
            # strict OpenAI schema rejects.  Strip it from responses.
            transport = _GroqFixTransport(inner)
        _HTTP_CLIENT_CACHE[cache_key] = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=None,
                write=30.0,
                pool=30.0,
            ),
            trust_env=not is_loopback,
            transport=transport,
        )
    return _HTTP_CLIENT_CACHE[cache_key]


class _OllamaFixTransport(httpx.AsyncBaseTransport):
    """Wraps an httpx transport to fix null ``content`` fields in chat messages.

    Ollama's OpenAI-compatible ``/v1/chat/completions`` endpoint returns
    HTTP 400 when any message in the ``messages`` array has ``content: null``.
    PydanticAI legitimately sends ``null`` content on assistant messages that
    only carry tool calls, which is valid per the OpenAI spec but not
    accepted by Ollama.  This transport rewrites those to ``""`` on the fly.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport) -> None:
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions") and request.content:
            try:
                body = json.loads(request.content)
                patched = False
                for msg in body.get("messages", []):
                    if msg.get("content") is None:
                        msg["content"] = ""
                        patched = True
                if patched:
                    new_content = json.dumps(body).encode("utf-8")
                    # Rebuild headers with corrected Content-Length
                    patched_headers = {
                        k: v for k, v in request.headers.items()
                        if k.lower() != "content-length"
                    }
                    patched_headers["content-length"] = str(len(new_content))
                    request = httpx.Request(
                        method=request.method,
                        url=request.url,
                        headers=patched_headers,
                        content=new_content,
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return await self._wrapped.handle_async_request(request)

    async def aclose(self) -> None:
        await self._wrapped.aclose()


class _GroqFixTransport(httpx.AsyncBaseTransport):
    """Strips non-standard ``service_tier`` values from Groq API responses.

    Groq returns ``"service_tier": "on_demand"`` in chat completion responses,
    which PydanticAI's strict OpenAI ``ChatCompletion`` schema rejects.
    This transport removes the field from response bodies before they are parsed.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport) -> None:
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._wrapped.handle_async_request(request)
        if request.url.path.endswith("/chat/completions"):
            await response.aread()
            try:
                body = json.loads(response.content)
                if "service_tier" in body:
                    del body["service_tier"]
                    # Overwrite the already-decoded content in place so httpx
                    # does not attempt to re-decompress it.
                    response._content = json.dumps(body).encode("utf-8")
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return response

    async def aclose(self) -> None:
        await self._wrapped.aclose()
