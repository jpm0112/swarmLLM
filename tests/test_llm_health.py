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
