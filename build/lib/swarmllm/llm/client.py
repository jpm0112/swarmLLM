from __future__ import annotations

"""
Legacy compatibility exports for the LLM layer.

The repo now routes model access through PydanticAI plus backend profiles.
This module remains as a lightweight compatibility surface for imports that
historically referenced `swarmllm.llm.client`.
"""

from swarmllm.llm.factory import build_coordinator_agent, build_worker_agent
from swarmllm.llm.health import validate_backend_or_raise
from swarmllm.llm.profiles import (
    apply_backend_profile,
    load_backend_profile,
    normalize_openai_base_url,
    resolve_api_key,
)
from swarmllm.llm.routing import EndpointRouter

__all__ = [
    "EndpointRouter",
    "apply_backend_profile",
    "build_coordinator_agent",
    "build_worker_agent",
    "load_backend_profile",
    "normalize_openai_base_url",
    "resolve_api_key",
    "validate_backend_or_raise",
]
