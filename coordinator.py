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

INITIAL_DIRECTIONS = [
    "Implement Earliest Due Date (EDD) first scheduling — sort jobs by due date ascending",
    "Implement Shortest Processing Time (SPT) first — sort by processing time ascending",
    "Implement Weighted Shortest Job First — weight by due_date / processing_time ratio",
    "Try a greedy slack-based approach: schedule jobs by (due_date - processing_time) ascending",
    "Implement a simple genetic algorithm: random population, tournament selection, order crossover, swap mutation",
    "Try simulated annealing: start with a random schedule, swap adjacent jobs, accept worse with decreasing probability",
    "Implement tabu search: local search with swap neighborhood, keep a tabu list of recent moves",
    "Try a beam search approach: expand partial schedules, keep top-K at each step based on partial tardiness",
    "Implement ant colony optimization: ants build schedules probabilistically based on pheromone trails",
    "Try a dynamic programming approach on subsets or approximation if exact DP is too expensive",
    "Implement iterated local search: perturbation + local search with swap moves",
    "Try a constructive heuristic: insert each job into the best position in the current partial schedule",
    "Implement a priority rule combining processing time and due date: Apparent Tardiness Cost (ATC) rule",
    "Try a random restart hill climbing: many random starts, swap-based local search from each",
    "Implement a hybrid: EDD initial solution + simulated annealing refinement",
    "Try a decomposition approach: split jobs into early/late groups, optimize each separately",
    "Implement particle swarm optimization adapted for permutations",
    "Try a branch-and-bound approach with EDD-based lower bounds (may need pruning for 20 jobs)",
    "Implement a memetic algorithm: genetic algorithm + local search on each offspring",
    "Try a neural-inspired approach: use numpy to compute urgency scores and sort by them",
]


async def get_initial_directions(config: Config) -> list[str]:
    """Return the initial set of directions for iteration 0."""
    return INITIAL_DIRECTIONS[:config.swarm.num_agents]


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
