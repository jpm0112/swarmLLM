from __future__ import annotations

import json

import httpx
import pytest

from swarmllm.config import Config, LLMEndpoint
from swarmllm.llm.health import validate_backend_or_raise


@pytest.mark.anyio
async def test_validate_backend_accepts_available_models():
    config = Config()
    config.llm.coordinator_model = "coord-model"
    config.llm.agent_model = "worker-model"
    config.llm.coordinator_endpoints = [LLMEndpoint(base_url="http://coord/v1", api_key="test")]
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://worker/v1", api_key="test")]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"data": [{"id": "coord-model"}, {"id": "worker-model"}]}
        return httpx.Response(200, text=json.dumps(payload))

    transport = httpx.MockTransport(handler)

    await validate_backend_or_raise(config.llm, transport=transport)


@pytest.mark.anyio
async def test_validate_backend_raises_when_worker_model_missing():
    config = Config()
    config.llm.coordinator_model = "coord-model"
    config.llm.agent_model = "worker-model"
    config.llm.coordinator_endpoints = [LLMEndpoint(base_url="http://coord/v1", api_key="test")]
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://worker/v1", api_key="test")]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = {"data": [{"id": "coord-model"}]}
        return httpx.Response(200, text=json.dumps(payload))

    transport = httpx.MockTransport(handler)

    with pytest.raises(RuntimeError):
        await validate_backend_or_raise(config.llm, transport=transport)


@pytest.mark.anyio
async def test_validate_backend_falls_back_to_localhost_alias():
    config = Config()
    config.llm.coordinator_model = "coord-model"
    config.llm.agent_model = "worker-model"
    config.llm.coordinator_endpoints = [LLMEndpoint(base_url="http://127.0.0.1:11434/v1", api_key="test")]
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://127.0.0.1:11434/v1", api_key="test")]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            raise httpx.ConnectError("connection refused", request=request)
        payload = {"data": [{"id": "coord-model"}, {"id": "worker-model"}]}
        return httpx.Response(200, text=json.dumps(payload))

    transport = httpx.MockTransport(handler)

    await validate_backend_or_raise(config.llm, transport=transport)

    assert config.llm.coordinator_endpoints[0].base_url == "http://localhost:11434/v1"
    assert config.llm.worker_endpoints[0].base_url == "http://localhost:11434/v1"


@pytest.mark.anyio
async def test_validate_backend_reports_actionable_connect_error():
    config = Config()
    config.llm.backend_kind = "ollama"
    config.llm.coordinator_model = "coord-model"
    config.llm.agent_model = "worker-model"
    config.llm.coordinator_endpoints = [LLMEndpoint(base_url="http://127.0.0.1:11434/v1", api_key="test")]
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://127.0.0.1:11434/v1", api_key="test")]

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handler)

    with pytest.raises(RuntimeError, match="Could not connect to the configured ollama backend") as excinfo:
        await validate_backend_or_raise(config.llm, transport=transport)

    message = str(excinfo.value)
    assert "http://127.0.0.1:11434/v1" in message
    assert "http://localhost:11434/v1" in message
    assert "ollama serve" in message
