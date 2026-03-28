from __future__ import annotations

"""Role-aware endpoint selection for coordinator and worker traffic."""

import itertools

from swarmllm.config import LLMConfig, LLMEndpoint


class EndpointRouter:
    """Routes coordinator traffic to the primary pool and workers round-robin."""

    def __init__(self, config: LLMConfig):
        if not config.coordinator_endpoints:
            raise ValueError("LLM config must include at least one coordinator endpoint.")
        if not config.worker_endpoints:
            raise ValueError("LLM config must include at least one worker endpoint.")
        self._config = config
        self._worker_cycle = _expand_weighted_pool(config.worker_endpoints)
        self._worker_counter = itertools.count()

    def coordinator_endpoint(self) -> LLMEndpoint:
        return self._config.coordinator_endpoints[0]

    def worker_endpoint(self) -> LLMEndpoint:
        idx = next(self._worker_counter) % len(self._worker_cycle)
        return self._worker_cycle[idx]


def _expand_weighted_pool(endpoints: list[LLMEndpoint]) -> list[LLMEndpoint]:
    pool: list[LLMEndpoint] = []
    for endpoint in endpoints:
        pool.extend([endpoint] * max(endpoint.weight, 1))
    if not pool:
        raise ValueError("Worker endpoint pools must contain at least one endpoint.")
    return pool
