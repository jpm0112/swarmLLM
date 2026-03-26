---
name: swarm-llm-optimization
description: Research project using a swarm of LLM agents coordinated by a larger LLM to solve optimization problems iteratively
type: project
---

# SwarmLLM — Multi-Agent LLM Swarm for Optimization

## What It Is

A framework where multiple LLM agents work in parallel to solve optimization problems. A coordinator LLM reads a shared log of all attempts, analyzes what worked and what didn't, and assigns diverse research directions to each agent for the next round.

## Architecture

```
Orchestrator (Python code, not an LLM)
    │
    ├── Coordinator LLM
    │   - Reads shared results log + top 5 solution codes
    │   - Sees per-instance scores, failure reasons, and error details
    │   - Analyzes progress across all instance sizes
    │   - Assigns unique direction to each agent
    │
    └── Agent LLMs (×N, run in parallel)
        - Receives a direction
        - Writes a Python schedule() function
        - Code runs in sandboxed subprocess
        - Tested on ALL problem instances (diverse sizes & characteristics)
        - Per-instance scores logged to shared file
```

**Important distinction:** The orchestrator (`orchestrator.py`) is Python code that runs the main loop. The coordinator is an LLM call that thinks about strategy. They are separate things.

## Design Decisions Made

| Decision | Choice |
|----------|--------|
| Coordinator strategy | Explore/Exploit split (50/50 default, tunable) |
| Log format | Flat markdown (append-only .md file) |
| Agent execution | Code generation + sandboxed subprocess |
| Communication | Hub-and-spoke (agents → log → coordinator) |
| Agent diversity | Explicit direction assignment by coordinator |
| Stopping criteria | Fixed iterations (default 5) |
| LLM backend | Ollama (local), separate models for coordinator vs agents |
| Default model | qwen2.5-coder:14b (fits on 3090 24GB) |
| Parallelism | OLLAMA_NUM_PARALLEL=4 set as user env var |
| Instance testing | Multi-instance benchmark with diverse profiles |
| Agent freedom | Minimal prompt injection — don't bias agent approaches |
| Schedule validation | No auto-repair — agent must produce valid permutation |
| Repeated directions | "Avoid repeating" (soft), not "Do NOT repeat" (hard) |
| Top solutions | Passed to coordinator only, not agents directly |

## Problem

Job scheduling: N jobs with processing times and due dates, minimize total tardiness. Agents write a `schedule(jobs) -> list[int]` function that returns a job ordering.

Baselines computed automatically per instance: FIFO, EDD (Earliest Due Date), SPT (Shortest Processing Time).

## Features

### Multi-Instance Evaluation
Each agent's code is tested on multiple diverse problem instances. Default profiles:

| Instance | Jobs | Deadline Tightness | Processing Time | Purpose |
|----------|------|--------------------|-----------------|---------|
| `small_tight` | 20 | 0.4 (tight) | 1-15 | Tests smart ordering under pressure |
| `medium_mixed` | 50 | 0.6 (moderate) | 1-20 | Balanced challenge |
| `large_loose` | 100 | 0.8 (loose) | 5-30 | Tests scalability with long jobs |

- Instances vary in size, deadline tightness, and processing time ranges
- Per-instance scores + aggregate score reported to coordinator
- Instances saved as JSON in `instances/` dir for reproducibility
- Coordinator sees which approaches scale and which don't
- Custom profiles easy to define in `config.py` (`InstanceProfile` dataclass)

### Dual Model Support
- Separate model selection for coordinator vs agents
- Coordinator can be a bigger/smarter model, agents can be smaller/faster
- Both selectable via interactive numbered menu in `run.bat`

### Sandboxed Code Execution
- Agent code runs in isolated subprocess with time limit (configurable, default 60s)
- **Blocklist approach**: dangerous modules blocked (os, sys, subprocess, socket, etc.), everything else allowed
- Internal library imports not blocked (scipy can use os internally) via `_import_depth` counter
- All stdlib modules available + numpy, scipy, networkx pre-installed

### Auto-Install Packages
- If agent code imports a package that's not installed, sandbox auto-installs it via pip and retries
- Installed packages persist in the .venv for future agents and runs
- Blocked modules still can't be imported directly by agent code

### Shared Memory (Top Solutions)
- Orchestrator tracks top 5 solutions (score + approach + code + per-instance scores)
- Passed to coordinator as context — it sees the actual winning code
- Coordinator decides how to reference it in directions (nothing hardcoded)
- Agents don't get the code directly unless the coordinator includes it in their direction

### Detailed Failure Reporting
Agent failures are categorized with actionable detail for the coordinator:
- `execution timed out — code took too long, needs a faster algorithm or early termination`
- `blocked import 'os' — agent tried to use a forbidden module`
- `syntax error — invalid Python syntax: unexpected indent`
- `undefined variable 'result' — code references something not defined`
- `type error — unsupported operand type(s) for +: 'int' and 'str'`
- `index error — code accessed a list/array out of bounds`
- `key error — missing key 'foo'`
- `value error — invalid value in computation`
- `division by zero — code divided by zero, needs a guard`
- `out of memory — algorithm used too much RAM, needs a lighter approach`
- `recursion limit — infinite or too-deep recursion`
- `missing module 'pulp' — library not available in sandbox`
- `invalid schedule — missing jobs: {2, 7, 15}, extra/duplicate jobs: {2, 2, 2}`
- `no code generated — LLM response had no valid python code block`
- `LLM unreachable — could not get a response from the model`
- Unknown errors show actual exception type + first 120 chars of message

Per-instance errors also logged separately (e.g. code works on 20 jobs but times out on 100).

### Logging & Output
Each run saves to a timestamped folder under `runs/`:
```
runs/2026-03-26_123045_coord-qwen2.5-coder_14b_agent-qwen2.5_3b_5agents_5iter/
    config.json           # exact settings used
    results_log.md        # all agent results, per-instance scores, coordinator summaries
    summary.txt           # final results comparison vs baselines
    instances/
        small_tight.json  # full job data + profile params
        medium_mixed.json
        large_loose.json
    prompts/
        iter_1/
            agent_00.md   # system prompt → user prompt → response
            agent_01.md
            ...
        iter_2/
            coordinator.md
            agent_00.md
            ...
```

Results log entries include:
- Direction, approach, aggregate score, runtime
- Per-instance scores (e.g. Score (small_tight): 336)
- Per-instance errors if partial failure
- Failure reason (categorized) + raw error
- Code in collapsible `<details>` block

### Interactive Setup (`run.bat` + `setup_run.py`)
- Creates and activates .venv automatically
- Shows numbered list of available Ollama models for coordinator and agent selection
- Auto-estimates recommended parallelism based on model size vs 24GB VRAM
- Restarts Ollama with correct OLLAMA_NUM_PARALLEL
- Installs extra pip packages for agents
- All parameters configurable with sensible defaults

### LLM Client
- Async wrapper around Ollama's OpenAI-compatible API
- Retry with exponential backoff: 3 attempts, 5s/15s/30s on connection errors and 5xx
- Does NOT retry on 4xx (bad request)
- Accepts explicit model parameter (coordinator vs agent)

### Configurable Parameters
All tunable via interactive prompt or CLI args:
- Coordinator model / Agent model
- Number of agents (default: max parallel based on VRAM)
- Number of iterations (default 5)
- Max concurrent agents / OLLAMA_NUM_PARALLEL
- Instance sizes (comma-separated, default "20,50,100") — auto-generates diverse profiles
- Explore/exploit ratio (default 0.5)
- Code execution timeout (default 60s)
- Random seed (default 1048596)

## File Structure

```
swarmLLM/
├── run.bat              # Entry point — venv setup + launch
├── setup_run.py         # Interactive parameter selection
├── run.py               # CLI entry point with argparse
├── config.py            # All configuration dataclasses + instance profiles
├── orchestrator.py      # Main loop (Python, not LLM)
├── coordinator.py       # Coordinator LLM prompts + parsing
├── agent.py             # Worker agent LLM prompts + parsing
├── llm_client.py        # Async Ollama API wrapper with retry
├── sandbox.py           # Sandboxed code execution in subprocess
├── problem.py           # Job scheduling problem + evaluator + baselines + save/load
├── logger.py            # Shared markdown log manager
├── prompt_logger.py     # Per-call prompt/response logging
├── requirements.txt     # aiohttp, numpy
├── .gitignore           # runs/, .venv/, __pycache__/
├── DESIGN.md            # Full design doc with alternatives + prior art
└── runs/                # Output from each run (gitignored)
```

## Hardware Setup

- GPU: NVIDIA RTX 3090 (24GB) + RTX 3080 Ti (12GB)
- Ollama with OLLAMA_NUM_PARALLEL=4 (set as user env var)
- Python 3.9 on Windows 11 (using `from __future__ import annotations`)

## First Run Results

20 agents × 5 iterations with qwen2.5-coder:14b (single instance, 20 jobs):
- **Best score: 336** (insertion-based heuristic)
- **Beat best baseline (SPT=446) by 24.7%**
- ~35% agent success rate (14b model)
- Best result came from iteration 1
- Coordinator direction parsing needs improvement for later iterations

## Known Issues & Improvements Needed

- Small models (3b) have very low success rates (~0-10%)
- Coordinator falls back to generic directions when parsing fails in later iterations
- Could use both GPUs (two Ollama instances) for more parallelism
- vLLM would give better batching but requires WSL2 on Windows

## Prior Art

FunSearch, AlphaEvolve, OPRO, ReEvo, ELM, EvoPrompt, LLaMEA, Model Swarms, SIER — all explored LLM-driven optimization. SwarmLLM differs in using explicit coordinator-assigned diversity (explore/exploit split) rather than pure evolutionary selection.
