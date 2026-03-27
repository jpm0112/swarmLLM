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
