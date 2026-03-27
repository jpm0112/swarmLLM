# LLM Infra Spec

## Goal

This document defines the next-generation inference architecture for SwarmLLM. The repo started with a single Ollama-oriented HTTP client and now needs an explicit backend/runtime layer that supports:

- `ollama` for Windows and local iteration
- `vllm-metal` for Apple Silicon
- `vllm` for Linux, server, cluster, and cloud deployments

The product behavior stays the same:

1. A coordinator assigns directions.
2. Worker agents run in parallel.
3. Generated code is sandboxed and evaluated.
4. Results are written to a shared markdown log.
5. The coordinator reads the log and assigns the next round.

The implementation objective is to move backend and orchestration concerns onto:

- OpenAI-compatible server interfaces for runtime compatibility
- PydanticAI for typed coordinator/worker runs
- backend profile files for deployment-specific wiring

## Current State

Before this change, the repo assumed:

- Ollama-only serving
- one custom `aiohttp` chat-completion client
- coordinator and worker responses parsed from markdown/text with regexes
- launcher/setup logic tightly coupled to local Ollama processes

The current codebase areas that this spec evolves are:

- [docs/DESIGN.md](docs/DESIGN.md)
- [swarmllm/core/orchestrator.py](/Users/zilikons/code/zilikons/projects/swarmLLM/swarmllm/core/orchestrator.py)
- [swarmllm/core/agent.py](/Users/zilikons/code/zilikons/projects/swarmLLM/swarmllm/core/agent.py)
- [swarmllm/core/coordinator.py](/Users/zilikons/code/zilikons/projects/swarmLLM/swarmllm/core/coordinator.py)
- [scripts/run.py](/Users/zilikons/code/zilikons/projects/swarmLLM/scripts/run.py)
- [scripts/setup_run.py](/Users/zilikons/code/zilikons/projects/swarmLLM/scripts/setup_run.py)

## Architecture Decisions

### 1. Common Runtime Boundary

All model backends are treated as OpenAI-compatible APIs.

Why:

- Ollama exposes OpenAI-compatible endpoints including `/v1/chat/completions` and `/v1/models`.
- vLLM exposes an OpenAI-compatible server via `vllm serve`.
- PydanticAI can target OpenAI-compatible providers through `OpenAIChatModel` plus `OpenAIProvider(base_url=..., api_key=...)`.

Consequence:

- SwarmLLM no longer embeds backend-specific HTTP request code in orchestration modules.
- Endpoint differences live in TOML profiles, launcher logic, and server templates.

### 2. PydanticAI Orchestration Pattern

SwarmLLM uses PydanticAI in programmatic hand-off mode, not graph mode.

Why:

- The current swarm loop is application-directed already.
- Coordinator and workers are distinct model calls controlled by application code.
- Programmatic hand-off preserves the current control flow without forcing a graph rewrite.

Consequence:

- Application code still decides when to call the coordinator and workers.
- PydanticAI supplies typed output contracts, usage accounting, and test doubles.

### 3. Structured Outputs Are Required

Coordinator and worker outputs are defined as typed schemas:

- `WorkerDraft`
- `DirectionAssignment`
- `CoordinatorRoundPlan`

Why:

- The old markdown parsing path was brittle.
- The shared log should remain human-readable, but the runtime contract should be machine-validated.

Consequence:

- Backends must use tool-capable instruction/chat models.
- Non-tool-capable local models are out of scope for this infra version.

### 4. Shared Markdown Log Remains Canonical

PydanticAI message history is useful for observability, but it does not replace `results_log.md`.

Why:

- The repo’s design emphasizes inspectable swarm behavior.
- The coordinator prompt still benefits from a readable shared experiment history.

Consequence:

- Prompt logs store structured message traces for debugging.
- The main run artifact remains the shared markdown log.

## Compatibility Matrix

| Backend | Primary OS / Use Case | Runtime Owner | Python Repo Dependency | Notes |
|---|---|---|---|---|
| `ollama` | Windows, local iteration | External local server | No extra runtime dep beyond repo app deps | Best compatibility and simplest setup |
| `vllm-metal` | macOS Apple Silicon | External local server | No bundled repo dep; installed externally | Uses Apple Silicon-specific vLLM Metal runtime |
| `vllm` | Linux, remote server, cluster, cloud | External local or remote server | No bundled repo dep; external deployment | Best throughput for large worker pools |

## Config Model

### Backend Profile TOML

Backend profiles live under `configs/backends/` and define:

- backend kind
- coordinator model alias
- worker model alias
- coordinator endpoint pool
- worker endpoint pool
- API key wiring
- request timeout
- default concurrency hint
- model-specific temperature and max-token defaults

The implemented schema is:

```toml
name = "profile-name"
kind = "ollama" # or "vllm-metal" or "vllm"
request_timeout = 300
default_max_concurrent_agents = 5

[coordinator]
model = "model-alias"
temperature = 0.4
max_tokens = 8192

[worker]
model = "model-alias"
temperature = 0.7
max_tokens = 4096

[[coordinator_endpoints]]
label = "primary"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"

[[worker_endpoints]]
label = "primary"
base_url = "http://127.0.0.1:11434/v1"
api_key = "ollama"
weight = 1
```

### CLI Precedence

Runtime precedence is:

1. CLI overrides
2. backend profile values
3. code defaults

Implemented CLI surface:

- `--backend-profile`
- `--coordinator-model`
- `--agent-model`
- `--agents`
- `--iterations`
- `--explore-ratio`
- `--max-concurrent`
- `--instance-sizes`
- `--seed`
- `--timeout`
- `--agent-retries`
- `--base-urls`
- `--output-dir`

`--base-urls` overrides endpoint pools as a convenience for quick local experiments.

## File Layout

### New Runtime Files

- `swarmllm/llm/profiles.py`
  Loads TOML backend profiles, validates them, and applies them to runtime config.
- `swarmllm/llm/routing.py`
  Routes coordinator calls to the primary endpoint and worker calls across a weighted pool.
- `swarmllm/llm/factory.py`
  Builds cached PydanticAI OpenAI-compatible models and coordinator/worker agents.
- `swarmllm/llm/health.py`
  Probes `/v1/models` and fails fast if model aliases are missing.
- `swarmllm/llm/schemas.py`
  Defines typed worker/coordinator output contracts.

### Supporting Config and Template Files

- `configs/backends/ollama.local.example.toml`
- `configs/backends/vllm-metal.local.example.toml`
- `configs/backends/vllm.single-node.example.toml`
- `configs/vllm/serve.single-node.example.yaml`
- `configs/vllm/serve.cluster.example.yaml`

## Runtime Behavior

### Startup Validation

Before the swarm starts:

1. Load the backend profile.
2. Normalize endpoint URLs to an OpenAI-compatible `/v1` base.
3. Resolve API keys from explicit values or environment variables.
4. Probe `/v1/models`.
5. Confirm:
   - the coordinator model alias exists on the coordinator endpoint
   - the worker model alias exists on every worker endpoint

The run fails immediately if any configured alias is missing.

### Coordinator Flow

Coordinator runs use:

- one primary coordinator endpoint
- one typed output schema: `CoordinatorRoundPlan`
- a structured analysis plus agent-direction assignments

The application then converts that typed result into the list-of-directions structure that the orchestrator already uses.

### Worker Flow

Worker runs use:

- a weighted round-robin worker endpoint pool
- typed output schema `WorkerDraft`
- a retry path that sends prior code and prior failure context back to the model if the smallest-instance pre-test fails

### Logging and Usage

Prompt logs now include:

- system prompt
- user prompt
- structured output JSON
- raw PydanticAI message history JSON

Token usage is derived from PydanticAI `RunUsage` and normalized into the repo’s existing token tracker summary format.

## Package Setup

### App Dependencies

SwarmLLM app dependencies now include:

- `pydantic-ai-slim[openai]`
- `pydantic>=2`
- `tomli` for Python 3.10 compatibility
- `numpy`

The old custom `aiohttp` transport dependency is removed because runtime calls now go through PydanticAI and its OpenAI-compatible provider stack.

### External Runtime Dependencies

These are intentionally not bundled as repo Python dependencies:

- `ollama`
- `vllm`
- `vllm-metal`

Reason:

- They are server runtimes, not application libraries for SwarmLLM itself.
- Different environments will provision them differently.

## Launcher Behavior

`scripts/setup_run.py` is now backend-aware and constrained by OS:

- Windows: `ollama`
- macOS Apple Silicon: `ollama` or `vllm-metal`
- Linux: `ollama` or `vllm`

Cluster/cloud behavior:

- point at remote `vllm` endpoints through a backend profile
- do not provision infrastructure from this repo in v1

## vLLM Template Notes

`vllm serve` supports YAML config files and CLI-over-config precedence.

The example YAML templates include:

- `model`
- `served-model-name`
- `host`
- `port`
- `api-key`
- `dtype`
- `max-model-len`
- `gpu-memory-utilization`
- `max-num-seqs`
- `enable-prefix-caching`
- `tensor-parallel-size`
- `pipeline-parallel-size`

These templates are intentionally examples, not generated runtime artifacts.

## Backend-Specific Operational Notes

### Ollama

- Use Ollama for Windows compatibility and local iteration.
- Treat Ollama as an OpenAI-compatible server at `http://127.0.0.1:11434/v1` by default.
- Ollama requires an API key field for OpenAI-compatible clients, but the default local value can safely be `"ollama"`.

### vLLM Metal

- Use vLLM Metal only on Apple Silicon.
- Treat the vLLM Metal runtime as an external install, typically under `~/.venv-vllm-metal`.
- Default memory guidance should assume `VLLM_METAL_MEMORY_FRACTION=auto`.
- Default compute guidance should assume the MLX path is enabled.
- Paged attention should be documented as experimental and opt-in.

### vLLM

- Use standard vLLM for Linux and remote/server deployments.
- Favor stable `served-model-name` aliases that match backend profile model names.
- Enable prefix caching when the served model/runtime combination supports it.
- Remote endpoint pools are supported in v1; deployment provisioning is not.

## Implementation Task Breakdown

1. Add the written infra spec and example config files.
2. Introduce backend profile loading and runtime config application.
3. Add endpoint routing for coordinator and worker traffic.
4. Add backend health validation through `/v1/models`.
5. Replace the old custom HTTP LLM client with PydanticAI-backed factories.
6. Refactor worker generation to use typed `WorkerDraft` output plus retry-on-pretest-failure.
7. Refactor coordinator planning to use typed `CoordinatorRoundPlan` output.
8. Update the CLI and interactive setup flow to select backend profiles.
9. Update logging and token tracking to capture structured outputs and PydanticAI usage.
10. Expand tests around profiles, routing, health checks, typed outputs, and launcher helpers.

## Testing Strategy

### Unit Tests

- backend profile parsing and validation
- precedence behavior when CLI overrides profile defaults
- routing behavior for coordinator and workers
- backend health checks with mocked OpenAI-compatible `/v1/models` responses
- deterministic sandbox, scheduling, and tracking behavior

### PydanticAI Tests

- happy-path coordinator and worker tests using `TestModel`
- edge cases using `FunctionModel`
- accidental live model requests disabled in tests

### Launcher Tests

- OS-to-backend compatibility matrix
- setup helper behavior around example profile selection

## Rollout Phases

### Phase 1

- introduce backend profiles
- add health checks
- add routing and PydanticAI factory

### Phase 2

- move coordinator and worker paths to typed outputs
- preserve shared-log orchestration
- update prompt logging and token tracking

### Phase 3

- add example profiles and server templates
- update launcher and README
- expand tests

## Non-Goals

This spec does not include:

- automatic cluster provisioning
- in-repo vLLM deployment management
- replacing the shared markdown log with a database
- switching the swarm loop to Pydantic Graph
- supporting non-tool-capable local models in the new runtime stack

## References

- PydanticAI OpenAI-compatible models: [ai.pydantic.dev/models/openai](https://ai.pydantic.dev/models/openai/)
- PydanticAI multi-agent patterns: [ai.pydantic.dev/multi-agent-applications](https://ai.pydantic.dev/multi-agent-applications/)
- PydanticAI testing: [ai.pydantic.dev/testing](https://ai.pydantic.dev/testing/)
- vLLM OpenAI-compatible server: [docs.vllm.ai/en/stable/serving/openai_compatible_server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- vLLM server args and YAML config: [docs.vllm.ai/en/stable/configuration/serve_args](https://docs.vllm.ai/en/stable/configuration/serve_args/)
- vLLM distributed serving background: [docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html](https://docs.vllm.ai/en/v0.7.2/serving/distributed_serving.html)
- vLLM Metal README: [github.com/vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)
- Ollama OpenAI compatibility: [docs.ollama.com/api/openai-compatibility](https://docs.ollama.com/api/openai-compatibility)
