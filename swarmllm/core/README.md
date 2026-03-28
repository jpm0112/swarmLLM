# swarmllm.core

This package holds the swarm control loop: worker-agent execution, coordinator direction-setting, and the main orchestration flow.

## What Goes Here

- Agent lifecycle code
- Coordinator prompt/response handling
- Iteration orchestration and concurrency control
- Parsing logic tied to coordinator or worker response formats

## What Should Not Go Here

- Problem-specific scoring rules that belong in `swarmllm/problems/`
- LLM transport details that belong in `swarmllm/llm/`
- Generic logging helpers that belong in `swarmllm/tracking/`

## Hierarchy

`swarmllm/core` is the top-level application layer inside the package. It calls into `swarmllm.llm`, `swarmllm.problems`, `swarmllm.sandbox`, and `swarmllm.tracking` to run a full swarm iteration.

## Environment Note

Repository command examples prefer `uv`, but the package guidance here also applies when the repo is used from an activated `pip` or `conda` environment. If `uv` is unavailable, run the equivalent `python -m ...` or `pytest` commands directly inside that environment.
