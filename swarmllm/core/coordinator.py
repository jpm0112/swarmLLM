from __future__ import annotations

"""
Coordinator LLM

Reads the shared log, analyzes results, and assigns research directions
to each agent for the next iteration.
"""

from pydantic_ai import AgentRunResult
from pydantic_ai.usage import RunUsage

from swarmllm.config import Config, LLMEndpoint
from swarmllm.llm.factory import build_coordinator_agent
from swarmllm.llm.schemas import CoordinatorRoundPlan, DirectionAssignment
from swarmllm.tracking.prompt_logger import PromptLogger
from swarmllm.tracking.token_tracker import TokenUsage


COORDINATOR_SYSTEM_PROMPT = """\
You are the coordinator of a swarm of optimization agents working on a job
scheduling problem where lower total tardiness is better.

Return structured output with:
- analysis: a short summary of what is working, failing, or still unexplored
- directions: one assignment per agent with agent_id, mode, and direction

Return only the structured result. Do not emit XML tags, <tools> wrappers,
markdown fences, or any prose outside the final structured output.

Modes:
- explore: try a meaningfully new idea or underexplored algorithm family
- exploit: refine or combine the strongest-performing approaches so far
"""


async def get_initial_directions(
    config: Config,
    endpoint: LLMEndpoint,
    prompt_logger: PromptLogger | None = None,
) -> tuple[list[str], TokenUsage | None]:
    """Ask the coordinator for initial directions for iteration 1."""
    num_agents = config.swarm.num_agents
    prompt = f"""This is the first iteration. There are no prior results.

Assign one specific research direction to each agent from 0 to {num_agents - 1}.
All assignments should use mode "explore" because the swarm is still mapping out
the search space.

Each direction should be concrete and actionable, with broad diversity across
algorithmic families.
"""
    usage = RunUsage()
    result = await _request_coordinator_plan(
        role="coordinator_initial",
        iteration=1,
        prompt=prompt,
        config=config,
        endpoint=endpoint,
        prompt_logger=prompt_logger,
        usage=usage,
    )
    _, directions = _normalize_round_plan(result.output, num_agents, initial=True)
    return directions, TokenUsage.from_run_usage(usage)


async def get_next_directions(
    iteration: int,
    log_content: str,
    config: Config,
    endpoint: LLMEndpoint,
    prompt_logger: PromptLogger | None = None,
    top_solutions: list[dict] | None = None,
) -> tuple[str, list[str], TokenUsage | None]:
    """
    Ask the coordinator to analyze prior rounds and assign directions.

    Returns (analysis_summary, list_of_directions, token_usage).
    """
    num_agents = config.swarm.num_agents
    num_explore = int(num_agents * config.swarm.explore_ratio)
    num_exploit = num_agents - num_explore

    top_section = ""
    if top_solutions:
        top_section = "\n## Top Solutions So Far\n\n"
        for i, solution in enumerate(top_solutions):
            top_section += f"### #{i + 1} — Score: {solution['score']} — {solution['approach']}\n"
            top_section += f"```python\n{solution['code']}\n```\n\n"
        top_section += "Agents may refine, combine, or contrast with these solutions.\n"

    prompt = f"""## Results So Far

{log_content}

{top_section}
---

This is iteration {iteration}. Assign one direction to each agent from 0 to
{num_agents - 1}.

Requirements:
- Use exactly {num_explore} explore assignments
- Use exactly {num_exploit} exploit assignments
- Exploit assignments should directly improve or hybridize the strongest results
- Explore assignments should avoid repeating already-tried directions unless they
  are meaningfully reframed
- Pay attention to behavior across different instance sizes, not just one score
"""
    usage = RunUsage()
    result = await _request_coordinator_plan(
        role="coordinator",
        iteration=iteration,
        prompt=prompt,
        config=config,
        endpoint=endpoint,
        prompt_logger=prompt_logger,
        usage=usage,
    )

    analysis, directions = _normalize_round_plan(result.output, num_agents, initial=False)
    return analysis, directions, TokenUsage.from_run_usage(usage)


async def _request_coordinator_plan(
    role: str,
    iteration: int,
    prompt: str,
    config: Config,
    endpoint: LLMEndpoint,
    prompt_logger: PromptLogger | None,
    usage: RunUsage,
) -> AgentRunResult[CoordinatorRoundPlan]:
    coordinator_agent = build_coordinator_agent(config, endpoint, COORDINATOR_SYSTEM_PROMPT)
    result = await coordinator_agent.run(
        prompt,
        usage=usage,
        model_settings={
            "temperature": config.llm.temperature_coordinator,
            "max_tokens": config.llm.max_tokens_coordinator,
        },
    )
    if prompt_logger:
        prompt_logger.log_structured(
            role=role,
            agent_id=None,
            iteration=iteration,
            system_prompt=COORDINATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            output=result.output,
            messages_json=result.all_messages_json(),
        )
    return result


def _normalize_round_plan(
    plan: CoordinatorRoundPlan,
    num_agents: int,
    initial: bool,
) -> tuple[str, list[str]]:
    """Normalize typed coordinator output into the orchestrator's direction list."""
    assignments: dict[int, DirectionAssignment] = {}
    for assignment in plan.directions:
        if 0 <= assignment.agent_id < num_agents and assignment.agent_id not in assignments:
            assignments[assignment.agent_id] = assignment

    fallbacks = [
        "Try a novel heuristic approach not yet explored",
        "Combine the two best approaches seen so far",
        "Try the opposite of what has been working to preserve diversity",
        "Focus on reducing maximum tardiness before total tardiness",
        "Try a randomized approach with multiple restarts",
    ]

    ordered_directions: list[str] = []
    for agent_id in range(num_agents):
        if agent_id in assignments:
            ordered_directions.append(assignments[agent_id].direction)
        else:
            ordered_directions.append(fallbacks[agent_id % len(fallbacks)])

    if initial:
        analysis = plan.analysis or "Initial exploration across diverse scheduling strategies."
    else:
        analysis = plan.analysis or "Coordinator returned no analysis; using fallback direction normalization."
    return analysis, ordered_directions
