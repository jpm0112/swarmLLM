# swarmllm.llm

This package contains the boundary to model providers. Right now it wraps Ollama's OpenAI-compatible chat API.

## What Goes Here

- LLM client adapters
- Retry, timeout, and load-balancing behavior for model calls
- Backend-specific request and response normalization

## What Should Not Go Here

- Prompt design and agent strategy logic from `swarmllm/core/`
- Logging concerns that belong in `swarmllm/tracking/`

## Hierarchy

`swarmllm/llm` is an infrastructure layer used by `swarmllm/core`. Future backend integrations should follow the same separation so orchestration code stays backend-agnostic.
