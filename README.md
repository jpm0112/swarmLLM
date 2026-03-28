# SwarmLLM

SwarmLLM is a research prototype for running a coordinator-guided swarm of LLM agents against optimization problems. The goal is to let many worker agents explore solution ideas in parallel, log what they tried, and let a coordinator LLM steer the next round based on what worked, what failed, and what still looks unexplored.

Today the repo is focused on a job scheduling benchmark: agents generate Python scheduling heuristics, the sandbox executes them, and the orchestrator scores them across multiple problem instances with different sizes and deadline tightness.

The inference layer now targets a unified OpenAI-compatible runtime surface, with backend profiles for `ollama`, `vllm-metal`, and `vllm`. The orchestration layer uses PydanticAI with typed coordinator and worker outputs while preserving the existing shared-log swarm loop.

## Goal

The design in [docs/DESIGN.md](docs/DESIGN.md) centers on three ideas:

- A swarm of worker agents exploring different algorithmic directions in parallel.
- A coordinator LLM assigning new directions after reading shared results.
- A shared, human-readable log so runs are debuggable and easy to inspect.

## Quickstart

The preferred workflow uses `uv`, but you can fall back to `pip` or `conda` if a `uv` environment is unavailable.

Preferred `uv` setup:

```bash
uv sync
```

Fallback `pip` setup inside an activated Python environment:

```bash
python -m pip install -U pip
python -m pip install -e .
python -m pip install pytest
```

Fallback `conda` setup:

```bash
conda create -n swarmllm python=3.11
conda activate swarmllm
python -m pip install -U pip
python -m pip install -e .
python -m pip install pytest
```

Run the test suite:

```bash
uv run pytest
# or, inside an activated pip or conda environment:
pytest
```

Run the swarm from the project entry point:

```bash
uv run swarmllm --help
uv run swarmllm --backend-profile configs/backends/ollama.local.example.toml --agents 10 --iterations 3
uv run swarmllm --backend-profile configs/backends/ollama.local.example.toml --agents 10 --iterations 3 --dashboard auto
# or, inside an activated pip or conda environment:
swarmllm --help
swarmllm --backend-profile configs/backends/ollama.local.example.toml --agents 10 --iterations 3
```

You can also run the script module directly:

```bash
uv run python -m scripts.run --backend-profile configs/backends/vllm.single-node.example.toml --agents 10 --iterations 3
# or, inside an activated pip or conda environment:
python -m scripts.run --backend-profile configs/backends/vllm.single-node.example.toml --agents 10 --iterations 3
```

## Live Monitoring

Swarm runs now support a built-in terminal monitor:

```bash
uv run swarmllm --backend-profile configs/backends/ollama.local.example.toml --dashboard auto
```

- `--dashboard auto` uses the Rich TUI when stdout is a TTY and falls back to plain logs otherwise.
- `--dashboard plain` forces plain terminal output while still writing telemetry files.
- `--dashboard tui` asks for the Rich TUI explicitly and falls back to plain mode if a TTY is unavailable.

Each run folder now includes:

- `events.jsonl` for append-only lifecycle telemetry
- `live_state.json` for the latest dashboard snapshot
- `run.log` for mirrored console output
- the existing `results_log.md`, `summary.txt`, `token_usage.json`, `prompts/`, `instances/`, and `config.json`

## Backends

- `ollama` is the default local-iteration and Windows-friendly path.
- `vllm-metal` is the Apple Silicon path for higher local throughput on macOS.
- `vllm` is the Linux/server/cluster/cloud path for higher parallel agent throughput.

App dependencies live in this repo. Model servers are external runtimes:

- Ollama: run an OpenAI-compatible Ollama server locally.
- vLLM Metal: install and run the Apple Silicon `vllm` CLI from the documented `~/.venv-vllm-metal` environment.
- vLLM: run a standard `vllm serve` deployment locally or remotely.

Example backend profiles live in `configs/backends/`, and example `vllm serve` YAML files live in `configs/vllm/`.

## Current Architecture

- `scripts/run.py` is the main CLI entry point.
- `swarmllm/core/` contains the orchestrator, worker agent loop, and coordinator logic.
- `swarmllm/problems/` contains optimization problem definitions and evaluation helpers.
- `swarmllm/sandbox/` runs agent-generated code under restrictions.
- `swarmllm/llm/` contains the LLM client wrapper.
- `configs/backends/` contains backend profile examples for Ollama, vLLM Metal, and vLLM.
- `configs/vllm/` contains `vllm serve` YAML templates.
- `swarmllm/tracking/` stores shared-log, prompt-log, and token-tracking helpers.
- `tests/` holds deterministic unit tests for local development and regression protection.

## Repository Layout

```text
swarmLLM/
├── AGENTS.md
├── README.md
├── docs/
│   └── DESIGN.md
│   └── LLM_INFRA_SPEC.md
├── configs/
│   ├── backends/
│   └── vllm/
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

- Prefer `uv add <package>` for runtime dependencies and `uv add --dev <package>` for developer tooling.
- If `uv` is unavailable, install the project into an activated `pip` or `conda` environment with `python -m pip install -e .`, install local dev tools such as `pytest` with `python -m pip install ...`, and run commands directly inside that environment.
- Prefer `uv run ...` for scripts, CLIs, and tests when `uv` is available; otherwise use the same commands without the `uv run` prefix inside the active `pip` or `conda` environment.
- Treat `pyproject.toml` and `uv.lock` as the canonical dependency definitions; `pip` and `conda` are compatibility workflows, not separate sources of truth.
- Add or update tests whenever behavior changes, especially in `swarmllm/problems`, `swarmllm/sandbox`, `swarmllm/tracking`, and parser-heavy logic in `swarmllm/core`.
- Keep the deterministic core testable without requiring a live Ollama server.
- Keep backend selection in TOML profiles instead of scattering endpoint details through code.
- Treat `vllm` and `vllm-metal` as external serving runtimes rather than mandatory Python package dependencies for repo users.
