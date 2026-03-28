# swarmllm.sandbox

This package executes agent-generated code with restrictions so experiments can be scored without giving arbitrary code full access to the host environment.

## What Goes Here

- Restricted execution helpers
- Import allow/block rules
- Timeouts, subprocess handling, and execution wrappers

## What Should Not Go Here

- LLM prompt logic from `swarmllm/core/`
- Problem scoring rules from `swarmllm/problems/`

## Hierarchy

`swarmllm/sandbox` sits between generated code and the rest of the repo. `swarmllm/core` sends candidate code here before `swarmllm/problems` evaluates the returned schedule.

## Environment Note

Repository command examples prefer `uv`, but the package guidance here also applies when the repo is used from an activated `pip` or `conda` environment. If `uv` is unavailable, run the equivalent `python -m ...` or `pytest` commands directly inside that environment.
