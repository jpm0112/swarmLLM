# swarmllm.llm

This package contains the boundary to model providers. It now targets OpenAI-compatible backends through PydanticAI, with backend profiles for Ollama, vLLM Metal, and vLLM.

## What Goes Here

- LLM client adapters
- Backend profile loading and validation
- Role-aware endpoint routing
- Retry, timeout, and load-balancing behavior for model calls
- Backend-specific request and response normalization

## What Should Not Go Here

- Prompt design and agent strategy logic from `swarmllm/core/`
- Logging concerns that belong in `swarmllm/tracking/`

## Hierarchy

`swarmllm/llm` is an infrastructure layer used by `swarmllm/core`. Future backend integrations should preserve the OpenAI-compatible boundary so orchestration code stays backend-agnostic.

## Environment Note

Repository command examples prefer `uv`, but the package guidance here also applies when the repo is used from an activated `pip` or `conda` environment. If `uv` is unavailable, run the equivalent `python -m ...` or `pytest` commands directly inside that environment.
