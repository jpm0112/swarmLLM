# SwarmLLM

SwarmLLM is a research prototype for running a coordinator-guided swarm of LLM agents against optimization problems. The goal is to let many worker agents explore solution ideas in parallel, log what they tried, and let a coordinator LLM steer the next round based on what worked, what failed, and what still looks unexplored.

Today the repo is focused on a job scheduling benchmark: agents generate Python scheduling heuristics, the sandbox executes them, and the orchestrator scores them across multiple problem instances with different sizes and deadline tightness.

## Goal

The design in [docs/DESIGN.md](docs/DESIGN.md) centers on three ideas:

- A swarm of worker agents exploring different algorithmic directions in parallel.
- A coordinator LLM assigning new directions after reading shared results.
- A shared, human-readable log so runs are debuggable and easy to inspect.

## Quickstart

Install and sync the project environment with `uv`:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest
```

Run the swarm from the project entry point:

```bash
uv run swarmllm --help
uv run swarmllm --agents 10 --iterations 3
```

You can also run the script module directly:

```bash
uv run python -m scripts.run --agents 10 --iterations 3
```

## Current Architecture

- `scripts/run.py` is the main CLI entry point.
- `swarmllm/core/` contains the orchestrator, worker agent loop, and coordinator logic.
- `swarmllm/problems/` contains optimization problem definitions and evaluation helpers.
- `swarmllm/sandbox/` runs agent-generated code under restrictions.
- `swarmllm/llm/` contains the LLM client wrapper.
- `swarmllm/tracking/` stores shared-log, prompt-log, and token-tracking helpers.
- `tests/` holds deterministic unit tests for local development and regression protection.

## Repository Layout

```text
swarmLLM/
├── AGENTS.md
├── README.md
├── docs/
│   └── DESIGN.md
├── scripts/
│   ├── run.py
│   └── setup_run.py
├── swarmllm/
│   ├── config.py
│   ├── core/
│   ├── llm/
│   ├── problems/
│   ├── sandbox/
│   └── tracking/
├── tests/
└── pyproject.toml
```

## Development Notes

- Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for developer tooling.
- Use `uv run ...` for scripts, CLIs, and tests so commands run inside the managed environment.
- Add or update tests whenever behavior changes, especially in `swarmllm/problems`, `swarmllm/sandbox`, `swarmllm/tracking`, and parser-heavy logic in `swarmllm/core`.
- Keep the deterministic core testable without requiring a live Ollama server.
