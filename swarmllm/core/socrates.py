from __future__ import annotations

"""
Socrates Agent

Runs in parallel with coders. Reads the memory of the previous iteration
(coordinator decisions + coder attempts) plus any existing knowledge file,
and produces an updated knowledge document with abstract, generalizable
insights for the coordinator.
"""

import time

from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

from swarmllm.config import Config, LLMEndpoint
from swarmllm.llm.factory import _get_chat_model
from swarmllm.tracking.attempt_memory import AttemptMemory
from swarmllm.tracking.prompt_logger import PromptLogger
from swarmllm.tracking.telemetry import TelemetrySink
from swarmllm.tracking.token_tracker import TokenUsage


SOCRATES_SYSTEM_PROMPT = """\
You are Socrates, an epistemologist embedded in a swarm of optimization agents.

Your job is NOT to solve the problem directly. Your job is to observe what the \
agents tried, what worked, what failed, and WHY — then distill that into \
abstract, reusable knowledge that will help the coordinator make better \
decisions in future iterations.

Think like a scientist reviewing experimental results:
- What patterns emerge across successful vs. failed attempts?
- What algorithmic families are promising and which are dead ends?
- What implementation pitfalls keep recurring?
- What problem structure properties make certain approaches work or fail?
- What combinations or hybrids seem worth exploring?

Write concise, actionable knowledge — not a summary of what happened, but \
lessons learned that generalize beyond the specific iteration.

DO NOT give instructions, directives, or commands. Do NOT write things like \
"try X", "agents should do Y", "use approach Z". You are an observer and \
analyst, not a director. State what IS true, what WAS observed, what patterns \
EXIST — never what someone should do next. The coordinator reads your output \
and decides actions on its own.

Your output is a Markdown document with sections. BE BRIEF — 300 words max. \
Use bullet points, not paragraphs. Cut anything that is obvious or redundant. \
Every sentence must earn its place.
"""

SOCRATES_USER_PROMPT_TEMPLATE = """\
## Your Task

Read the data from the latest iteration below and update the accumulated \
knowledge document. Preserve still-valid insights, revise or remove outdated \
ones, and add new learnings from iteration {iteration}.

{existing_knowledge_section}

## Iteration {iteration} Data

### Coordinator Decision
{coordinator_summary}

### Agent Results
{agent_results_summary}

---

Write the updated knowledge document now. 300 words MAX — bullet points, \
no fluff. Output ONLY the markdown content — no preamble, no meta-commentary.
"""


def _format_coordinator_summary(coord: dict | None) -> str:
    if not coord:
        return "(no coordinator data available)"
    lines = []
    analysis = coord.get("analysis")
    if analysis:
        lines.append(f"**Analysis:** {analysis}")
    for a in coord.get("assignments", []):
        mode = a.get("mode", "?")
        direction = a.get("direction", "?")
        refs = a.get("source_refs", [])
        ref_str = ""
        if refs:
            ref_parts = [f"agent {r.get('agent_id')} iter {r.get('iteration')}" for r in refs]
            ref_str = f" (building on: {', '.join(ref_parts)})"
        lines.append(f"- Agent {a.get('agent_id')}: [{mode}] {direction}{ref_str}")
    return "\n".join(lines) or "(empty)"


def _format_agent_results(coders: list[dict]) -> str:
    if not coders:
        return "(no agent data available)"
    lines = []
    for c in coders:
        aid = c.get("agent_id", "?")
        approach = c.get("approach", "?")
        success = c.get("success", False)
        score = c.get("score")
        error = c.get("error", "")
        notes = c.get("notes", "")
        instance_scores = c.get("instance_scores", {})

        status = f"score={score}" if success else "FAILED"
        lines.append(f"### Agent {aid}: {approach}")
        lines.append(f"- Status: {status}")
        if instance_scores:
            parts = [f"{k}={v}" for k, v in sorted(instance_scores.items())]
            lines.append(f"- Per-instance: {', '.join(parts)}")
        if not success and error:
            # Keep error concise — last 2 lines
            err_lines = error.strip().splitlines()
            short_err = "\n".join(err_lines[-2:]) if len(err_lines) > 2 else error.strip()
            lines.append(f"- Error: `{short_err}`")
        if notes:
            lines.append(f"- Notes: {notes}")
        lines.append("")
    return "\n".join(lines)


async def run_socrates(
    iteration: int,
    memory: AttemptMemory,
    config: Config,
    endpoint: LLMEndpoint,
    prompt_logger: PromptLogger | None = None,
    telemetry: TelemetrySink | None = None,
) -> tuple[str, TokenUsage | None]:
    """
    Run the Socrates agent for a given iteration.

    Reads memory from iteration-1, reads the existing knowledge file,
    and produces an updated knowledge markdown document.

    Returns (knowledge_markdown, token_usage).
    """
    # Read previous iteration memory
    prev = iteration - 1
    coord_data = memory.read_coordinator(prev) if prev >= 1 else None
    coder_data = memory.read_coders(prev) if prev >= 1 else []

    # Read existing knowledge
    existing_knowledge = memory.read_knowledge()

    # Build the prompt
    if existing_knowledge:
        existing_section = f"## Existing Knowledge Document\n\n{existing_knowledge}"
    else:
        existing_section = (
            "## Existing Knowledge Document\n\n"
            "(This is the first iteration with results — no prior knowledge exists yet. "
            "Create the initial knowledge document.)"
        )

    coordinator_summary = _format_coordinator_summary(coord_data)
    agent_results_summary = _format_agent_results(coder_data)

    prompt = SOCRATES_USER_PROMPT_TEMPLATE.format(
        iteration=prev,
        existing_knowledge_section=existing_section,
        coordinator_summary=coordinator_summary,
        agent_results_summary=agent_results_summary,
    )

    endpoint_label = endpoint.label or endpoint.base_url
    if telemetry:
        telemetry.emit_event(
            "socrates_started",
            message=f"Socrates analyzing iteration {prev}",
            iteration=iteration,
            endpoint_label=endpoint_label,
        )

    usage = RunUsage()
    llm_start = time.time()
    try:
        model = _get_chat_model(config.llm, endpoint, config.llm.coordinator_model)
        agent = Agent(
            model,
            output_type=str,
            system_prompt=SOCRATES_SYSTEM_PROMPT,
            name="swarmllm_socrates",
            retries=2,
            defer_model_check=True,
        )
        result = await agent.run(
            prompt,
            usage=usage,
            model_settings={
                "temperature": config.llm.temperature_coordinator,
                "max_tokens": config.llm.max_tokens_coordinator,
            },
        )
        llm_time = time.time() - llm_start

        if prompt_logger:
            prompt_logger.log_structured(
                role="socrates",
                agent_id=None,
                iteration=iteration,
                system_prompt=SOCRATES_SYSTEM_PROMPT,
                user_prompt=prompt,
                output=result.output,
                messages_json=result.all_messages_json(),
            )

        knowledge_text = result.output
        memory.write_knowledge(knowledge_text)
        memory.write_knowledge_iteration(iteration, knowledge_text)

        token_usage = TokenUsage.from_run_usage(usage)

        if telemetry:
            telemetry.record_llm_call(
                role="socrates",
                iteration=iteration,
                agent_id=None,
                model=config.llm.coordinator_model,
                duration_seconds=llm_time,
                usage=token_usage,
                endpoint_label=endpoint_label,
            )
            telemetry.emit_event(
                "socrates_completed",
                message=f"Socrates updated knowledge ({len(knowledge_text)} chars, {llm_time:.1f}s)",
                iteration=iteration,
                llm_time=llm_time,
            )

        return knowledge_text, token_usage

    except Exception as exc:
        llm_time = time.time() - llm_start
        if telemetry:
            telemetry.emit_event(
                "socrates_failed",
                message=f"Socrates failed: {exc}",
                level="warning",
                iteration=iteration,
            )
        print(f"  Socrates failed ({type(exc).__name__}): {exc}")
        return existing_knowledge or "", TokenUsage.from_run_usage(usage)
