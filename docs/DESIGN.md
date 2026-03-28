# SwarmLLM: Multi-Agent LLM Swarm for Optimization

## 1. Core Idea

A framework where multiple LLM agents work in parallel to solve optimization problems. A coordinator LLM reads a shared log of all attempts, analyzes what worked and what didn't, and assigns diverse research directions to each agent for the next round.

```
Orchestrator (Python code, not an LLM)
    │
    ├── Coordinator LLM (qwen3.5:27b — runs once per iteration, sequentially)
    │   - Reads shared results log + top 5 solution codes
    │   - Sees per-instance scores, failure reasons, and error details
    │   - Analyzes progress across all instance sizes
    │   - Assigns unique direction to each agent
    │
    └── Agent LLMs (×N, deepcoder:14b — run in parallel)
        - Receives a direction from coordinator
        - Writes a Python schedule() function
        - Code runs in sandboxed subprocess
        - Tested on ALL problem instances (diverse sizes & characteristics)
        - Per-instance scores logged to shared file
```

**Important distinction:** The orchestrator (`orchestrator.py`) is Python code that runs the main loop. The coordinator is an LLM call that thinks about strategy. They are separate things.

## 2. The Loop

```
1. INITIALIZE
   - Coordinator LLM assigns diverse initial directions (one per agent)

2. EXECUTE (parallel)
   - Each agent receives its direction
   - Proposes a solution approach
   - Implements it (writes code)
   - Pre-tests on all instances with retry feedback loop
   - Logs results to the shared file

3. COORDINATE
   - Coordinator reads the full shared log + top solutions
   - Analyzes: what scored highest? what failed? what's unexplored?
   - Assigns new directions for the next round

4. REPEAT steps 2-3 for N iterations (or until convergence)
```

## 3. Design Decisions

| Decision | Choice |
|----------|--------|
| Coordinator strategy | Explore/Exploit split (50/50 default, tunable) |
| Log format | Flat markdown (append-only .md file) |
| Agent execution | Code generation + sandboxed subprocess |
| Communication | Hub-and-spoke (agents → log → coordinator) |
| Agent diversity | Explicit direction assignment by coordinator |
| Stopping criteria | Fixed iterations (default 5) |
| LLM backend | Ollama (local), separate models for coordinator vs agents |
| Coordinator model | qwen3.5:27b (~17GB) — smarter strategy, runs sequentially |
| Agent model | deepcoder:14b (~9GB) — best coding perf at this size, reasoning backbone |
| Parallelism | Dual GPU: 3090 (4 slots) + 3080 Ti (1 slot) = 5 concurrent |
| Instance testing | Multi-instance benchmark with diverse profiles |
| Agent freedom | Minimal prompt injection — don't bias agent approaches |
| Schedule validation | No auto-repair — agent must produce valid permutation |
| Repeated directions | "Avoid repeating" (soft), not "Do NOT repeat" (hard) |
| Top solutions | Passed to coordinator only, not agents directly |
| Default agents | Multiple of max concurrent (e.g. 20 = 4 batches of 5) |

## 4. Design Alternatives

### 4.1 Coordinator Strategy

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Pure Exploration** | Always assign untried directions | Maximum coverage of solution space | Wastes time on bad regions; never refines |
| **Pure Exploitation** | All agents refine the current best | Fast convergence on a good solution | Gets stuck in local optima |
| **Explore/Exploit Split** | E.g., 10 agents explore, 10 exploit | Balances breadth and depth | Ratio is a hyperparameter to tune |
| **Adaptive Bandit** | Coordinator uses a bandit-like strategy | Dynamically adjusts explore/exploit balance | More complex coordinator prompt |
| **Island Model** | Group agents into "islands" with periodic migration | Maintains diversity naturally | More complex orchestration |
| **Tournament** | Bottom N reassigned to variations of top N | Strong selection pressure | Can lose diversity too fast |
| **Quality-Diversity (MAP-Elites)** | Maintain grid of solution niches | Diverse set of good solutions | Requires defining meaningful dimensions |

### 4.2 Shared Log Format

| Format | Pros | Cons |
|--------|------|------|
| **Flat Markdown** | Simple, human-readable, LLM-friendly | Gets huge |
| **Structured Markdown + Summary** | Coordinator can skim summary | Two files to maintain |
| **JSON Log** | Easy to parse programmatically | Less natural for LLM |
| **Database (SQLite)** | Scalable, queryable | LLMs can't directly read it |

### 4.3 Agent Execution Model

| Model | Pros | Cons |
|-------|------|------|
| **Code Generation + Sandbox** | Full flexibility | Security concerns; execution failures |
| **Template-Based** | Safe; consistent interface | Less creative freedom |
| **Prompt-Only** | No code execution needed | Limited to harness capabilities |

### 4.4 Communication Pattern

| Pattern | Pros | Cons |
|---------|------|------|
| **Hub-and-Spoke** | Simple; coordinator has full picture | Coordinator is bottleneck |
| **Peer-to-Peer** | Faster information spread | Chaotic; no central strategy |
| **Hierarchical** | Scales better | More complex; more LLM calls |
| **Blackboard** | Flexible | Needs conflict resolution |
| **Stigmergy** | Emergent coordination | Harder to control |

### 4.5 Agent Diversity Mechanism

| Mechanism | Description |
|-----------|-------------|
| **Explicit direction assignment** | Coordinator assigns distinct strategies |
| **Temperature variation** | Different agents use different LLM temperatures |
| **Persona assignment** | Each agent gets a different "persona" |
| **Constraint variation** | Each agent optimizes under different constraints |
| **Historical exclusion** | Each agent must try something different from history |

## 5. Problem: Job Scheduling

N jobs with processing times and due dates, minimize total tardiness. Agents write a `schedule(jobs) -> list[int]` function that returns a job ordering.

Baselines computed automatically per instance: FIFO, EDD (Earliest Due Date), SPT (Shortest Processing Time).

### Multi-Instance Evaluation

| Instance | Jobs | Deadline Tightness | Processing Time | Purpose |
|----------|------|--------------------|-----------------|---------|
| `small_tight` | 20 | 0.4 (tight) | 1-15 | Tests smart ordering under pressure |
| `medium_mixed` | 50 | 0.6 (moderate) | 1-20 | Balanced challenge |
| `large_loose` | 100 | 0.8 (loose) | 5-30 | Tests scalability with long jobs |

## 6. Features

### Dual GPU Support
- **GPU 0:** RTX 3090 (24GB) — runs agents in parallel (~4 slots with 14b model)
- **GPU 1:** RTX 3080 Ti (12GB) — runs agents in parallel (~1 slot with 14b model)
- Two Ollama instances on separate ports (11434, 11435), one per GPU
- Round-robin request distribution across instances
- Coordinator always runs on GPU 0 (largest VRAM)

### Dual Model Support
- Separate model selection for coordinator vs agents
- Coordinator: `qwen3.5:27b` — bigger/smarter for strategic thinking
- Agents: `deepcoder:14b` — coding-specialized with reasoning backbone (built on DeepSeek-R1)
- Both selectable via interactive numbered menu in `run.bat`

### Sandboxed Code Execution
- Agent code runs in isolated subprocess with time limit (configurable, default 120s)
- **Blocklist approach**: dangerous modules blocked (os, sys, subprocess, socket, etc.)
- Internal library imports not blocked (scipy can use os internally) via `_import_depth` counter
- All stdlib modules available + numpy, scipy, networkx pre-installed

### Auto-Install Packages
- If agent code imports a package that's not installed, sandbox auto-installs it via pip and retries
- Agents are told they can use any pip package in their system prompt
- Installed packages persist in the .venv for future agents and runs
- Blocked modules still can't be imported directly by agent code

### Agent Pre-Test Feedback Loop
- After code generation, test on all instances before submitting
- If any instance fails, feed the error back to the LLM for a fix attempt
- Configurable retries (`agent_retries`, default 1)
- Saves tokens and time vs always using max retries

### Shared Memory (Top Solutions)
- Orchestrator tracks top 5 solutions (score + approach + code + per-instance scores)
- Passed to coordinator as context — it sees the actual winning code
- Coordinator decides how to reference it in directions (nothing hardcoded)
- Agents don't get the code directly unless the coordinator includes it

### Detailed Failure Reporting
Agent failures are categorized with actionable detail:
- `execution timed out`, `blocked import`, `syntax error`, `undefined variable`
- `type error`, `index error`, `key error`, `value error`, `division by zero`
- `out of memory`, `recursion limit`, `missing module`, `invalid schedule`
- `no code generated`, `LLM unreachable`
- Per-instance errors logged separately (e.g. works on 20 jobs but times out on 100)

### Token Tracking
- Per-call token usage (prompt + completion) tracked
- Per-iteration and running totals
- Agent vs coordinator split
- Saved to `token_usage.json` in run output directory

### Timing Split
- LLM response time and sandbox execution time tracked separately per agent
- Shown in console output: `[LLM: 35.2s, exec: 9.4s, total: 44.6s]`
- Coordinator LLM time tracked per iteration

### Logging & Output
Each run saves to a timestamped folder under `runs/`:
```
runs/2026-03-26_123045_coord-qwen3.5_27b_agent-deepcoder_14b_20agents_5iter/
    config.json           # exact settings used
    results_log.md        # all agent results (read by coordinator)
    summary.txt           # final results comparison vs baselines
    token_usage.json      # token tracking data
    instances/
        small_tight.json  # full job data + profile params
        medium_mixed.json
        large_loose.json
    prompts/
        iter_1/
            coordinator.md
            agent_00.md   # system prompt → user prompt → response
            agent_01.md
            ...
        iter_2/
            coordinator.md
            agent_00.md
            ...
```

### Interactive Setup (`run.bat` + `setup_run.py`)
- Creates and activates .venv automatically
- Shows numbered list of available Ollama models for coordinator and agent selection
- Defaults: coordinator=qwen2.5-coder:14b, agents=qwen2.5-coder:14b
- Auto-estimates recommended parallelism based on model size vs GPU VRAM
- Dual GPU detection: calculates slots per GPU, asks to enable
- Default agents = max_concurrent × 4 (4 full batches)
- Restarts Ollama with correct OLLAMA_NUM_PARALLEL
- All parameters configurable with sensible defaults and explanations

### LLM Client
- Async wrapper around Ollama's OpenAI-compatible API
- Round-robin across multiple base URLs (for dual GPU)
- Retry with exponential backoff: 5 attempts, 5s/15s/30s/60s/60s
- Coordinator gets longer request timeout (600s vs 300s for agents)
- Does NOT retry on 4xx (bad request)

## 7. Configurable Parameters

All tunable via interactive prompt or CLI args:
- Coordinator model (default: qwen2.5-coder:14b)
- Agent model (default: qwen2.5-coder:14b)
- Max concurrent agents (auto-calculated from model size + GPUs)
- Number of agents (default: max_concurrent × 4)
- Number of iterations (default 5)
- Dual GPU mode (y/n)
- Instance sizes (comma-separated, default "20,50,100")
- Explore/exploit ratio (default 0.5)
- Code execution timeout (default 120s, same for every agent)
- Agent retries (default 1)
- Random seed (default 1048596)

## 8. File Structure

```
swarmLLM/
├── run.bat              # Entry point — venv setup + launch
├── setup_run.py         # Interactive parameter selection
├── run.py               # CLI entry point with argparse
├── config.py            # All configuration dataclasses + instance profiles
├── orchestrator.py      # Main loop (Python, not LLM)
├── coordinator.py       # Coordinator LLM prompts + parsing
├── agent.py             # Worker agent LLM prompts + parsing
├── llm_client.py        # Async Ollama API wrapper with retry + round-robin
├── sandbox.py           # Sandboxed code execution in subprocess
├── problem.py           # Job scheduling problem + evaluator + baselines + save/load
├── logger.py            # Shared markdown log manager
├── prompt_logger.py     # Per-call prompt/response logging (markdown files)
├── token_tracker.py     # Token usage tracking
├── requirements.txt     # aiohttp, numpy
├── .gitignore           # runs/, .venv/, __pycache__/
├── DESIGN.md            # This file — design doc + feature reference
└── runs/                # Output from each run (gitignored)
```

## 9. Hardware Setup

- **GPU 0:** NVIDIA RTX 3090 (24GB) — primary, runs agents + coordinator
- **GPU 1:** NVIDIA RTX 3080 Ti (12GB) — secondary, runs additional agent slots
- Dual Ollama instances: port 11434 (GPU 0), port 11435 (GPU 1)
- ~5 total concurrent agent slots with 14b model (Q4_K_M quantization)
- Python 3.9 on Windows 11 (using `from __future__ import annotations`)

## 10. Models

| Role | Model | Size | Quantization | Why |
|------|-------|------|--------------|-----|
| Coordinator | qwen3.5:27b | ~17GB | Q4_K_M | Newest general model, smart strategy |
| Agents | deepcoder:14b | ~9GB | Q4_K_M | Best coding benchmarks at 14B, DeepSeek-R1 backbone |
| Previous | qwen2.5-coder:14b | ~9GB | Q4_K_M | Good but older, less reasoning |

## 11. First Run Results

20 agents × 5 iterations with qwen2.5-coder:14b (single instance, 20 jobs):
- **Best score: 336** (insertion-based heuristic)
- **Beat best baseline (SPT=446) by 24.7%**
- ~35% agent success rate (14b model)
- Best result came from iteration 1

## 12. Known Issues

- Small models (3b) have very low success rates (~0-10%)
- Coordinator falls back to generic directions when parsing fails in later iterations
- Invalid permutation is the most common failure — agents return partial job lists
- Retries rarely fix fundamental algorithm bugs (1/11 success rate with 5 retries)
- vLLM would give better batching but requires WSL2 on Windows

## 13. Prior Art & Related Work

| Project | By | Year | Key Difference from SwarmLLM |
|---------|-----|------|------------------------------|
| **FunSearch** | DeepMind | 2023 | Single LLM, no coordinator — diversity via island model |
| **AlphaEvolve** | DeepMind | 2025 | Flash/Pro split, but coordination is implicit |
| **OPRO** | DeepMind | 2023 | Same accumulating-history pattern, but single agent |
| **ReEvo** | NeurIPS 2024 | 2024 | Self-reflection ≈ coordinator, but single agent |
| **ELM** | OpenAI | 2022 | Quality-diversity, but single LLM |
| **EvoPrompt** | Tsinghua/MS | 2023 | Optimizes prompts, not general solutions |
| **LLaMEA** | Leiden Univ. | 2024 | Meta-level (evolves algorithms). Single LLM |
| **Model Swarms** | Google | 2024 | PSO in LLM weight space |
| **SIER** | 2025 | Most similar, but uses algorithmic coordination, not a coordinator LLM |

**What makes SwarmLLM novel:** No existing system combines (1) parallel swarm of LLM agents, (2) explicit coordinator LLM reasoning about strategy, and (3) shared textual log as communication medium.

## 14. References

- FunSearch: https://www.nature.com/articles/s41586-023-06924-6
- AlphaEvolve: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- OPRO: https://arxiv.org/abs/2309.03409
- ELM / OpenELM: https://github.com/CarperAI/OpenELM
- EvoPrompt: https://arxiv.org/abs/2309.08532
- ReEvo: NeurIPS 2024
- LLaMEA: https://arxiv.org/abs/2405.20132
- Model Swarms: https://arxiv.org/abs/2410.11163
- SIER: https://arxiv.org/html/2505.17115v1
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve
