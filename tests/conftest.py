from __future__ import annotations

import pytest
from pydantic_ai import models

from swarmllm.llm.factory import clear_caches


models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(autouse=True)
def _clear_llm_factory_caches():
    clear_caches()
    yield
    clear_caches()
