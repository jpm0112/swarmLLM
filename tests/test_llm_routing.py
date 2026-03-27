from __future__ import annotations

from swarmllm.config import Config, LLMEndpoint
from swarmllm.llm.routing import EndpointRouter


def test_endpoint_router_uses_primary_coordinator_endpoint():
    config = Config()
    config.llm.coordinator_endpoints = [
        LLMEndpoint(base_url="http://coord-primary/v1", api_key="x"),
        LLMEndpoint(base_url="http://coord-secondary/v1", api_key="x"),
    ]
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://worker/v1", api_key="x")]

    router = EndpointRouter(config.llm)

    assert router.coordinator_endpoint().base_url == "http://coord-primary/v1"


def test_endpoint_router_round_robins_worker_pool_with_weights():
    config = Config()
    config.llm.worker_endpoints = [
        LLMEndpoint(base_url="http://worker-a/v1", api_key="x", weight=1),
        LLMEndpoint(base_url="http://worker-b/v1", api_key="x", weight=2),
    ]

    router = EndpointRouter(config.llm)
    seen = [router.worker_endpoint().base_url for _ in range(6)]

    assert seen == [
        "http://worker-a/v1",
        "http://worker-b/v1",
        "http://worker-b/v1",
        "http://worker-a/v1",
        "http://worker-b/v1",
        "http://worker-b/v1",
    ]
