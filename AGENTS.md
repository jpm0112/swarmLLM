# AGENTS.md

This repository is a research codebase for a coordinator-led swarm of LLM agents solving optimization problems. The current implementation targets job scheduling, but the code is structured so new problem families, LLM backends, and coordination strategies can be added incrementally.

## Working Principles

- Keep changes aligned with the goal in [docs/DESIGN.md](docs/DESIGN.md): parallel worker exploration, coordinator-guided iteration, and inspectable shared logs.
- Prefer small, composable modules. Put problem-agnostic orchestration in `swarmllm/core`, problem definitions in `swarmllm/problems`, backend communication in `swarmllm/llm`, execution controls in `swarmllm/sandbox`, and observability helpers in `swarmllm/tracking`.
- Preserve deterministic seams. Anything that can be tested without a live LLM or external service should stay easy to exercise in isolation.

## Dependency Management

- Use `uv add <package>` to add runtime dependencies.
- Use `uv add --dev <package>` to add development-only tools such as test runners or linters.
- Treat `pyproject.toml` and `uv.lock` as the dependency source of truth.
- Do not edit dependency versions in multiple places by hand unless there is a strong reason and the change is explained in the PR or commit.

## Running Code

- Use `uv run ...` for scripts, entry points, and one-off commands.
- Preferred examples:
  - `uv run swarmllm --help`
  - `uv run python -m scripts.run --agents 10 --iterations 3`
  - `uv run pytest`
- When adding new scripts under `scripts/`, make them runnable through `uv run python -m ...` or a declared project entry point.

## Testing Expectations

- Keep an extensive test suite under `tests/`.
- Every behavior change should come with tests or a clear explanation of why automated coverage is not practical.
- Favor fast, deterministic unit tests for:
  - problem generation and evaluation
  - shared-log formatting and parsing
  - token tracking and prompt logging helpers
  - sandbox success and failure paths
  - response parsing in agent and coordinator workflows
- Avoid requiring a live Ollama server in the default test suite. Mock or isolate LLM calls when testing orchestration behavior.
- Run `uv run pytest` before wrapping up substantive changes.

## Code Organization Guidelines

- Keep optimization-problem-specific logic out of `swarmllm/core` when it can live in `swarmllm/problems`.
- Keep backend-specific API details inside `swarmllm/llm/client.py` or a future backend adapter module.
- Be careful with sandbox changes. Security restrictions, subprocess behavior, and package installation paths should be treated as high-risk surfaces and covered by tests.
- Treat logging and tracking output as part of the developer experience. Changes to markdown structure should remain readable by humans and stable enough for future parsing.

## Contributor Checklist

- Read the relevant folder README before adding a new module.
- Update docs when repository structure or workflow expectations change.
- Add tests in `tests/` with new features and bug fixes.
- Use `uv add` and `uv run` consistently so local workflows stay predictable for future collaborators.
