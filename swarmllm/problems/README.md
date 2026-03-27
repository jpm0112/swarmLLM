# swarmllm.problems

This package defines optimization problems, instance generation, evaluation, and baseline solvers.

## What Goes Here

- Problem dataclasses and schemas
- Instance generators and loaders
- Objective functions and validation helpers
- Simple baseline strategies for comparison

## What Should Not Go Here

- Coordinator or agent orchestration logic from `swarmllm/core/`
- Sandbox execution code from `swarmllm/sandbox/`

## Hierarchy

`swarmllm/problems` is the domain layer. `swarmllm/core` depends on it to describe tasks for agents and score the solutions they produce.
