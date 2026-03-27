# swarmllm.tracking

This package contains observability helpers for swarm runs: shared logs, prompt capture, and token accounting.

## What Goes Here

- Shared markdown log management
- Prompt and response logging
- Token usage aggregation and summaries
- Future run metadata or analytics helpers

## What Should Not Go Here

- Coordination logic from `swarmllm/core/`
- Problem-specific evaluation from `swarmllm/problems/`

## Hierarchy

`swarmllm/tracking` supports the whole repo. It is used by `swarmllm/core` to make runs inspectable and to preserve the shared-log workflow described in the design doc.

## Environment Note

Repository command examples prefer `uv`, but the package guidance here also applies when the repo is used from an activated `pip` or `conda` environment. If `uv` is unavailable, run the equivalent `python -m ...` or `pytest` commands directly inside that environment.
