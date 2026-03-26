# SwarmLLM: Multi-Agent LLM Swarm for Optimization

## 1. Core Idea

Use a **swarm of 20 LLM agents** coordinated by a **single coordinator LLM** to solve optimization problems through parallel exploration, shared logging, and iterative refinement.

Each worker agent independently proposes, implements, and tests a solution. All results are recorded in a shared log file. The coordinator reads the log after each round, analyzes what worked and what didn't, and assigns new research directions to each agent for the next round.

```
                    ┌───────────────────────┐
                    │   Coordinator LLM     │
                    │  - Reads shared log   │
                    │  - Analyzes progress   │
                    │  - Assigns directions  │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                  │
        ┌─────▼─────┐   ┌─────▼─────┐     ┌─────▼─────┐
        │  Agent 1   │   │  Agent 2   │ ... │  Agent 20  │
        │ Direction A│   │ Direction B│     │ Direction T│
        └─────┬─────┘   └─────┬─────┘     └─────┬─────┘
              │                 │                  │
              └─────────────────┼──────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Shared Log (.md)    │
                    │  - What was tried     │
                    │  - Scores / results   │
                    │  - Failure analysis   │
                    └───────────────────────┘
```

## 2. The Loop

```
1. INITIALIZE
   - Coordinator reads the problem definition
   - Assigns 20 diverse initial directions (one per agent)

2. EXECUTE (parallel)
   - Each agent receives its direction
   - Proposes a solution approach
   - Implements it (writes code)
   - Tests it (runs against evaluation function)
   - Logs results to the shared file

3. COORDINATE
   - Coordinator reads the full shared log
   - Analyzes: what scored highest? what failed? what's unexplored?
   - Assigns new directions for the next round

4. REPEAT steps 2-3 for N iterations (or until convergence)
```

## 3. Design Alternatives

### 3.1 Coordinator Strategy

How the coordinator decides what to assign each agent in the next round.

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Pure Exploration** | Always assign untried directions | Maximum coverage of solution space | Wastes time on bad regions; never refines |
| **Pure Exploitation** | All agents refine the current best | Fast convergence on a good solution | Gets stuck in local optima |
| **Explore/Exploit Split** | E.g., 10 agents explore, 10 exploit | Balances breadth and depth | Ratio is a hyperparameter to tune |
| **Adaptive Bandit** | Coordinator uses a bandit-like strategy — allocate more agents to promising directions as confidence grows | Dynamically adjusts explore/exploit balance | More complex coordinator prompt |
| **Island Model** | Group agents into "islands" that evolve independently, with periodic migration of best ideas between islands | Maintains diversity naturally; robust to deception | More complex orchestration |
| **Tournament** | Bottom N agents are reassigned to variations of top N agents' approaches | Strong selection pressure | Can lose diversity too fast |
| **Quality-Diversity (MAP-Elites style)** | Coordinator maintains a grid of solution niches (e.g., by approach type, complexity, speed) and assigns agents to underfilled niches | Finds diverse set of good solutions, not just one | Requires defining meaningful dimensions |

### 3.2 Shared Log Format

How agents record their results.

| Format | Description | Pros | Cons |
|--------|-------------|------|------|
| **Flat Markdown** | One big .md file, append-only | Simple, human-readable, LLM-friendly | Gets huge; coordinator has to read everything |
| **Structured Markdown + Summary** | Detailed log + a separate summary table of (agent, approach, score) | Coordinator can skim summary, deep-dive when needed | Two files to maintain |
| **JSON Log** | Structured JSON entries | Easy to parse programmatically; sortable | Less natural for LLM to read/write |
| **Database (SQLite)** | Structured storage with queries | Scalable; can query top-K, filter by approach | Overhead; LLMs can't directly read it |
| **Hybrid: JSON + Markdown Renderer** | Store in JSON, render to .md for coordinator to read | Best of both worlds | More infrastructure |

### 3.3 Agent Execution Model

How agents implement and test solutions.

| Model | Description | Pros | Cons |
|-------|-------------|------|------|
| **Code Generation + Sandbox** | Agent writes Python code, runs in sandboxed environment, reports score | Full flexibility; agent can try any algorithm | Security concerns; execution failures |
| **Template-Based** | Agent fills in a function template (e.g., `def heuristic(problem) -> solution`) | Safe; consistent interface; easy to evaluate | Less creative freedom |
| **Prompt-Only** | Agent describes approach in natural language, a fixed harness interprets it | No code execution needed | Limited to what the harness can interpret |
| **Hybrid** | Agent writes code within a constrained API (imports whitelisted, time-limited) | Balance of safety and flexibility | More infrastructure to build |

### 3.4 Communication Pattern

How information flows between agents.

| Pattern | Description | Pros | Cons |
|---------|-------------|------|------|
| **Hub-and-Spoke (proposed)** | All agents write to shared log; only coordinator reads it | Simple; coordinator has full picture | Coordinator is bottleneck; agents don't learn from each other directly |
| **Peer-to-Peer** | Agents can read each other's results directly | Faster information spread | Chaotic; no central strategy |
| **Hierarchical** | Sub-coordinators manage groups of 5, report up to main coordinator | Scales better; specialized sub-strategies | More complex; more LLM calls |
| **Blackboard** | Shared workspace where anyone can read/write; coordinator prunes and organizes | Flexible; agents can build on each other | Needs careful conflict resolution |
| **Stigmergy** | Agents leave "traces" (scored solutions); other agents are attracted to high-scoring traces | Emergent coordination; truly swarm-like | Harder to control; unpredictable |

### 3.5 Agent Diversity Mechanism

How to prevent all 20 agents from converging on the same approach.

| Mechanism | Description |
|-----------|-------------|
| **Explicit direction assignment** | Coordinator assigns distinct strategies (e.g., "try genetic algorithms", "try gradient-free methods") |
| **Temperature variation** | Different agents use different LLM temperatures (some conservative, some wild) |
| **Persona assignment** | Each agent gets a different "persona" (e.g., "you are a mathematician", "you are a systems engineer") |
| **Constraint variation** | Each agent optimizes under different constraints (e.g., "optimize for speed", "optimize for simplicity") |
| **Historical exclusion** | Each agent is told what others have already tried and must try something different |
| **Random seed prompts** | Each agent gets a random "inspiration" snippet to push them in different directions |

### 3.6 Convergence / Stopping Criteria

| Criterion | Description |
|-----------|-------------|
| **Fixed iterations** | Run for N rounds |
| **Score plateau** | Stop when best score hasn't improved for K rounds |
| **Budget-based** | Stop after spending $X on API calls |
| **Coordinator decision** | Coordinator decides when to stop based on its analysis |
| **Diversity collapse** | Stop when all agents are converging to the same approach (nothing new to explore) |

## 4. Prior Art & Related Work

This idea builds on and differs from several existing lines of research:

### 4.1 Closest Related Work

| Project | By | Year | What It Does | Key Difference from SwarmLLM |
|---------|-----|------|-------------|------------------------------|
| **FunSearch** | DeepMind | 2023 | Evolves programs using LLM + island-based evolution. Discovered new math results. | Single LLM, no coordinator — diversity via island model, not explicit strategy assignment |
| **AlphaEvolve** | DeepMind | 2025 | Successor to FunSearch. Uses Gemini Flash (breadth) + Pro (depth) ensemble to evolve codebases. | Flash/Pro split resembles worker/coordinator, but coordination is implicit, not through a shared log |
| **OPRO** | DeepMind | 2023 | Single LLM optimizes by reading history of past attempts + scores in its prompt. | Same accumulating-history pattern as our shared log, but single agent, no parallelism, no coordinator |
| **ReEvo** | NeurIPS 2024 | 2024 | LLM reflects on why solutions succeeded/failed before mutating. State-of-the-art sample efficiency. | Self-reflection ≈ our coordinator's analysis, but single agent, no swarm |
| **ELM** | OpenAI | 2022 | LLM as mutation operator in MAP-Elites quality-diversity algorithm. | Quality-diversity approach is relevant for our coordinator, but single LLM, no multi-agent |
| **EvoPrompt** | Tsinghua/MS | 2023 | Evolutionary prompt optimization using LLM-based crossover/mutation. | Optimizes prompts specifically, not general solutions |
| **LLaMEA** | Leiden Univ. | 2024 | LLM generates entire metaheuristic algorithms, evolves them. | Meta-level (evolves algorithms, not solutions). Single LLM. |
| **Model Swarms** | Google | 2024 | PSO in LLM weight space — models are particles. | Optimizes model weights, not external solutions |
| **SIER** | 2025 | Multiple LLM agents explore solution space in parallel with density-based diversity. | Most architecturally similar, but uses algorithmic coordination (density), not a coordinator LLM |

### 4.2 Multi-Agent Frameworks (Infrastructure)

| Framework | Notes |
|-----------|-------|
| **Swarms** (kyegomez) | Supports hierarchical coordinator + worker patterns natively. Closest production framework. |
| **AutoGen** (Microsoft) | Conversational multi-agent; could implement our shared-log pattern |
| **CrewAI** | Role-based agent teams; role assignment ≈ coordinator giving directions |
| **LangGraph** (LangChain) | Graph-based stateful workflows; most flexible for complex coordination |
| **OpenAI Swarm/Agents SDK** | Lightweight agent handoffs; good for prototyping |

### 4.3 What Makes SwarmLLM Novel

No existing system combines all three of:

1. **Parallel swarm** of LLM agents (not a single LLM)
2. **Explicit coordinator LLM** that reasons about strategy (not algorithmic selection like PSO or fitness-proportional sampling)
3. **Shared textual log** as the communication medium (human-readable, inspectable, debuggable)

The closest systems either use a single LLM with evolutionary pressure (FunSearch, OPRO, ReEvo), multiple LLMs with algorithmic coordination (Model Swarms, SIER), or multi-agent frameworks without optimization-specific design (CrewAI, AutoGen).

## 5. Open Questions

1. **Which LLM for workers vs coordinator?** Small/cheap model for workers (e.g., Haiku, GPT-4o-mini) vs large model for coordinator (e.g., Opus, GPT-4)?
2. **What benchmark problem to start with?** TSP? Bin packing? Prompt optimization? Function optimization?
3. **How many iterations before diminishing returns?**
4. **Cost management** — 20 agents × N iterations × API cost per call adds up fast. How to budget?
5. **Can the coordinator learn to coordinate better over time?** (Meta-meta-optimization)

## 6. References

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
