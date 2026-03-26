from __future__ import annotations

"""
Coordinator LLM

Reads the shared log, analyzes results, and assigns research directions
to each agent for the next iteration.

Uses the Explore/Exploit split strategy:
- Half the agents explore new untried approaches
- Half the agents refine/improve the best approaches so far
"""

from config import Config
from llm_client import chat_completion
from token_tracker import TokenUsage
from prompt_logger import PromptLogger


COORDINATOR_SYSTEM_PROMPT = """\
You are the coordinator of a swarm of {num_agents} optimization agents working on
a job scheduling problem (minimize total tardiness).

Each agent's code is tested on {num_instances} problem instances of different sizes
and characteristics. Scores are reported per instance plus an aggregate (sum).
Pay attention to how approaches perform across different instance sizes — some
algorithms scale better than others.

Your job is to:
1. Analyze what has been tried so far and what worked best (across all instances)
2. Assign a UNIQUE research direction to each agent for the next iteration

STRATEGY (Explore/Exploit split):
- {num_explore} agents should EXPLORE new, untried approaches
- {num_exploit} agents should EXPLOIT and refine the best-performing approaches

For EXPLORE agents: assign creative, diverse directions that haven't been tried.
Think about different algorithmic families: greedy heuristics, metaheuristics,
dynamic programming, local search, genetic algorithms, simulated annealing,
constraint-based, hybrid approaches, mathematical relaxations, etc.

For EXPLOIT agents: take the top-performing approaches and ask agents to
improve them — tune parameters, combine with other ideas, fix weaknesses.

OUTPUT FORMAT (you must follow this exactly):

ANALYSIS:
<your 2-3 sentence analysis of what's working and what's not>

DIRECTIONS:
Agent 0 [EXPLORE]: <specific direction>
Agent 1 [EXPLOIT]: <specific direction>
Agent 2 [EXPLORE]: <specific direction>
...
(one line per agent, covering all {num_agents} agents)
"""

INITIAL_SYSTEM_PROMPT = """\
You are the coordinator of a swarm of {num_agents} optimization agents working on
a job scheduling problem (minimize total tardiness).

Each agent's code is tested on {num_instances} problem instances of different sizes
and characteristics. Scores are reported per instance plus an aggregate (sum).

This is the FIRST iteration — no results yet. Your job is to assign {num_agents}
UNIQUE and diverse research directions for the agents to explore.

Think about different algorithmic families: greedy heuristics, metaheuristics,
dynamic programming, local search, genetic algorithms, simulated annealing,
constraint-based, hybrid approaches, mathematical relaxations, constructive
heuristics, priority rules, etc.

OUTPUT FORMAT (you must follow this exactly):

DIRECTIONS:
Agent 0: <specific direction>
Agent 1: <specific direction>
Agent 2: <specific direction>
...
(one line per agent, covering all {num_agents} agents)
"""


async def get_initial_directions(
    config: Config,
    prompt_logger: PromptLogger | None = None,
) -> tuple[list[str], TokenUsage]:
    """Ask the coordinator LLM for initial directions (iteration 1)."""
    num_agents = config.swarm.num_agents

    system_prompt = INITIAL_SYSTEM_PROMPT.format(
        num_agents=num_agents,
        num_instances=len(config.problem.instances),
    )

    prompt = f"""This is the first iteration. No results yet.
Assign {num_agents} diverse research directions for the agents.
Each direction should be specific and actionable — tell the agent exactly what
algorithm or approach to implement. Cover a wide range of algorithmic families.
"""

    response, token_usage = await chat_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        config=config.llm,
        model=config.llm.coordinator_model,
        temperature=config.llm.temperature_coordinator,
        max_tokens=config.llm.max_tokens_coordinator,
    )

    if prompt_logger:
        prompt_logger.log("coordinator", None, 1, system_prompt, prompt, response)

    # Parse directions
    _, directions = _parse_coordinator_response(response, num_agents)

    return directions, token_usage


async def get_next_directions(
    iteration: int,
    log_content: str,
    config: Config,
    prompt_logger: PromptLogger | None = None,
    top_solutions: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """
    Ask the coordinator LLM to analyze results and assign new directions.

    Returns (analysis_summary, list_of_directions, token_usage).
    """
    num_agents = config.swarm.num_agents
    num_explore = int(num_agents * config.swarm.explore_ratio)
    num_exploit = num_agents - num_explore

    system_prompt = COORDINATOR_SYSTEM_PROMPT.format(
        num_agents=num_agents,
        num_explore=num_explore,
        num_exploit=num_exploit,
        num_instances=len(config.problem.instances),
    )

    # Build top solutions section
    top_section = ""
    if top_solutions:
        top_section = "\n## Top Solutions So Far (code included)\n\n"
        for i, sol in enumerate(top_solutions):
            top_section += f"### #{i+1} — Score: {sol['score']} — {sol['approach']}\n"
            top_section += f"```python\n{sol['code']}\n```\n\n"
        top_section += "You may reference this code in your directions. Agents can refine, combine, or build on it.\n"

    prompt = f"""## Results So Far

{log_content}

{top_section}
---

This is iteration {iteration}. Based on the results above, assign research
directions for the next round. Remember:
- {num_explore} agents should EXPLORE new approaches
- {num_exploit} agents should EXPLOIT/refine what's working
- Each direction must be specific and actionable
- Avoid repeating directions that have already been tried (unless refining them)
"""

    response, token_usage = await chat_completion(
        prompt=prompt,
        system_prompt=system_prompt,
        config=config.llm,
        model=config.llm.coordinator_model,
        temperature=config.llm.temperature_coordinator,
        max_tokens=config.llm.max_tokens_coordinator,
    )

    if prompt_logger:
        prompt_logger.log("coordinator", None, iteration, system_prompt, prompt, response)

    # Parse the response
    analysis, directions = _parse_coordinator_response(response, num_agents)

    return analysis, directions, token_usage


def _parse_coordinator_response(response: str, num_agents: int) -> tuple[str, list[str]]:
    """Parse coordinator response into analysis and list of directions."""
    # Extract analysis
    analysis = ""
    if "ANALYSIS:" in response:
        parts = response.split("ANALYSIS:", 1)
        after = parts[1]
        if "DIRECTIONS:" in after:
            analysis = after.split("DIRECTIONS:", 1)[0].strip()
        else:
            analysis = after.strip()

    # Extract directions
    directions = []
    if "DIRECTIONS:" in response:
        dir_section = response.split("DIRECTIONS:", 1)[1].strip()
        for line in dir_section.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Match lines like "Agent 0 [EXPLORE]: direction text"
            if ":" in line and ("Agent" in line or "agent" in line):
                direction = line.split(":", 1)[1].strip()
                directions.append(direction)

    # Pad or trim to match expected agent count
    if len(directions) < num_agents:
        # Fill remaining with generic exploration directions
        fallbacks = [
            "Try a novel heuristic approach not yet explored",
            "Combine the two best approaches seen so far",
            "Try the opposite of what's been working — explore a very different strategy",
            "Focus on reducing the maximum tardiness rather than total",
            "Try a randomized approach with multiple restarts",
        ]
        while len(directions) < num_agents:
            idx = len(directions) % len(fallbacks)
            directions.append(fallbacks[idx])

    return analysis, directions[:num_agents]
