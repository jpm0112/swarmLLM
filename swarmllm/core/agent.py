from __future__ import annotations

"""
Worker Agent

Each worker receives a direction from the coordinator, asks the configured
OpenAI-compatible backend for a structured draft, executes the generated code
in the sandbox, and reports scored results back to the orchestrator.
"""

import time

from pydantic_ai import AgentRunResult
from pydantic_ai.usage import RunUsage

from swarmllm.config import Config, LLMEndpoint
from swarmllm.llm.factory import build_worker_agent
from swarmllm.llm.schemas import WorkerDraft
from swarmllm.problems import ProblemBase, ProblemInstance
from swarmllm.sandbox.executor import execute_agent_code_async
from swarmllm.tracking.prompt_logger import PromptLogger
from swarmllm.tracking.telemetry import TelemetrySink
from swarmllm.tracking.token_tracker import TokenUsage, UsageSnapshot


def _build_source_section(source_context: list[dict]) -> str:
    """Build the exploit context section for the agent prompt."""
    lines = ["## Prior Work to Build On", ""]
    lines.append(
        "The following solutions are provided as context. Refine, combine, or"
        " improve on these approaches as directed."
    )
    for ref in source_context:
        lines.append("")
        lines.append(
            f"### Agent {ref['agent_id']} — Iteration {ref['iteration']}"
            f" — {ref['approach']}"
        )
        if ref.get("notes"):
            lines.append(f"**Notes:** {ref['notes']}")
        lines.append("")
        lines.append("```python")
        lines.append(ref["code"])
        lines.append("```")
    lines.append("")
    return "\n".join(lines)



async def run_agent(
    agent_id: int,
    direction: str,
    problems: list[tuple[str, ProblemInstance]],
    config: Config,
    problem: ProblemBase,
    endpoint: LLMEndpoint,
    iteration: int = 0,
    prompt_logger: PromptLogger | None = None,
    source_context: list[dict] | None = None,
    telemetry: TelemetrySink | None = None,
) -> dict:
    """
    Run a single worker agent against all configured problem instances.

    Returns a dict with:
        agent_id, direction, approach, code, score, success, error, notes,
        instance_scores, instance_errors, token_usage, llm_time, exec_time

    source_context: list of prior agent results (agent_id, iteration, code,
        notes, approach) to inject into the prompt for exploit-mode agents.
    """
    _, example_problem = problems[0]
    instance_desc = ", ".join(
        f"{name} ({len(problem.prepare_input(p))} items)" for name, p in problems
    )
    source_section = _build_source_section(source_context) if source_context else ""
    problem_desc = problem.get_agent_user_prompt(example_problem, instance_desc)

    prompt = f"""{problem_desc}

Your code will be tested on {len(problems)} diverse instances: {instance_desc}.
They vary in size and characteristics.
Write a general algorithm — do not hardcode for a specific instance.

{source_section}## Research Direction

{direction}
"""

    pip_list = ", ".join(config.sandbox.pip_packages) if config.sandbox.pip_packages else "none"
    system_prompt = problem.get_agent_system_prompt(
        timeout=config.sandbox.timeout, pip_packages=pip_list
    )
    usage = RunUsage()
    llm_time = 0.0
    endpoint_label = endpoint.label or endpoint.base_url

    try:
        if telemetry:
            telemetry.set_agent_phase(
                agent_id,
                iteration,
                phase="llm",
                status="running",
                retry_count=0,
                endpoint_label=endpoint_label,
                direction=direction,
            )
            telemetry.emit_event(
                "agent_llm_started",
                message=f"Agent {agent_id} requesting initial draft",
                agent_id=agent_id,
                iteration=iteration,
                endpoint_label=endpoint_label,
            )
        initial_start = time.time()
        draft_result, call_usage = await _request_worker_draft(
            role="agent",
            agent_id=agent_id,
            iteration=iteration,
            prompt=prompt,
            system_prompt=system_prompt,
            endpoint=endpoint,
            config=config,
            prompt_logger=prompt_logger,
            usage=usage,
        )
        call_duration = time.time() - initial_start
        llm_time += call_duration
        if telemetry:
            telemetry.record_llm_call(
                role="agent",
                iteration=iteration,
                agent_id=agent_id,
                model=config.llm.agent_model,
                duration_seconds=call_duration,
                usage=call_usage,
                endpoint_label=endpoint_label,
            )
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"    Agent {agent_id} LLM error ({type(exc).__name__}): {exc}")
        print(f"    {tb.splitlines()[-2] if len(tb.splitlines()) >= 2 else tb}")
        return {
            "agent_id": agent_id,
            "direction": direction,
            "approach": "Failed to get LLM response",
            "code": "",
            "score": None,
            "success": False,
            "error": f"LLM error ({type(exc).__name__}): {exc}",
            "notes": "",
            "instance_scores": {},
            "instance_errors": {},
            "token_usage": TokenUsage.from_run_usage(usage),
            "llm_time": llm_time,
            "exec_time": 0.0,
        }

    draft = draft_result.output
    approach = draft.approach
    code = draft.code
    notes = draft.notes

    # Pre-test on smallest instance
    _, smallest_problem = problems[0]
    smallest_input = problem.prepare_input(smallest_problem)
    function_name = problem.get_function_name()

    if telemetry:
        telemetry.set_agent_phase(
            agent_id,
            iteration,
            phase="sandbox",
            status="running",
            retry_count=0,
            endpoint_label=endpoint_label,
            direction=direction,
        )
        telemetry.emit_event(
            "agent_sandbox_started",
            message=f"Agent {agent_id} validating draft on the smallest instance",
            agent_id=agent_id,
            iteration=iteration,
            stage="precheck",
        )
    pre_error = await _validate_generated_code(
        code,
        smallest_problem,
        smallest_input,
        problem,
        function_name,
        config,
        telemetry=telemetry,
        agent_id=agent_id,
        iteration=iteration,
        process_label=f"agent_{agent_id} sandbox precheck",
    )

    for retry in range(config.swarm.agent_retries):
        if pre_error is None:
            break

        if telemetry:
            telemetry.emit_event(
                "agent_precheck_failed",
                message=f"Agent {agent_id} precheck failed; requesting retry {retry + 1}",
                level="warning",
                agent_id=agent_id,
                iteration=iteration,
                retry_count=retry + 1,
                error=pre_error,
            )
            telemetry.set_agent_phase(
                agent_id,
                iteration,
                phase="llm",
                status="running",
                retry_count=retry + 1,
                endpoint_label=endpoint_label,
                direction=direction,
            )
        fix_prompt = problem.get_fix_prompt(error=pre_error, code=code)
        try:
            retry_start = time.time()
            fix_result, call_usage = await _request_worker_draft(
                role=f"agent_fix{retry + 1}",
                agent_id=agent_id,
                iteration=iteration,
                prompt=fix_prompt,
                system_prompt=system_prompt,
                endpoint=endpoint,
                config=config,
                prompt_logger=prompt_logger,
                usage=usage,
            )
            call_duration = time.time() - retry_start
            llm_time += call_duration
            if telemetry:
                telemetry.record_llm_call(
                    role="agent",
                    iteration=iteration,
                    agent_id=agent_id,
                    model=config.llm.agent_model,
                    duration_seconds=call_duration,
                    usage=call_usage,
                    endpoint_label=endpoint_label,
                )
        except Exception:
            break

        fixed = fix_result.output
        approach = fixed.approach or approach
        code = fixed.code or code
        if fixed.notes:
            notes = f"{fixed.notes} (fixed after pre-test retry {retry + 1})"

        if telemetry:
            telemetry.set_agent_phase(
                agent_id,
                iteration,
                phase="sandbox",
                status="running",
                retry_count=retry + 1,
                endpoint_label=endpoint_label,
                direction=direction,
            )
        pre_error = await _validate_generated_code(
            code,
            smallest_problem,
            smallest_input,
            problem,
            function_name,
            config,
            telemetry=telemetry,
            agent_id=agent_id,
            iteration=iteration,
            process_label=f"agent_{agent_id} sandbox precheck",
        )

    # Full evaluation across all instances
    exec_start = time.time()
    instance_scores: dict[str, float] = {}
    instance_errors: dict[str, str] = {}
    first_error: str | None = None

    if telemetry:
        telemetry.set_agent_phase(
            agent_id,
            iteration,
            phase="sandbox",
            status="running",
            retry_count=0 if pre_error is None else config.swarm.agent_retries,
            endpoint_label=endpoint_label,
            direction=direction,
        )
        telemetry.emit_event(
            "agent_sandbox_started",
            message=f"Agent {agent_id} evaluating across {len(problems)} instances",
            agent_id=agent_id,
            iteration=iteration,
            stage="benchmark",
        )
    for inst_name, inst in problems:
        input_data = problem.prepare_input(inst)

        exec_result = await execute_agent_code_async(
            code,
            input_data,
            config.sandbox,
            function_name=function_name,
            telemetry=telemetry,
            process_label=f"agent_{agent_id} sandbox {inst_name}",
            process_metadata={"agent_id": agent_id, "iteration": iteration, "instance": inst_name},
        )
        if not exec_result["success"]:
            instance_errors[inst_name] = exec_result["error"]
            if first_error is None:
                first_error = exec_result["error"]
            continue

        solution = problem.extract_solution(exec_result)
        eval_result = problem.evaluate(inst, solution)
        if not eval_result["valid"]:
            instance_errors[inst_name] = eval_result["error"]
            if first_error is None:
                first_error = eval_result["error"]
            continue

        instance_scores[inst_name] = eval_result["score"]

    exec_time = time.time() - exec_start
    token_usage = TokenUsage.from_run_usage(usage)

    all_passed = len(instance_scores) == len(problems) and not instance_errors
    if not instance_scores:
        return {
            "agent_id": agent_id,
            "direction": direction,
            "approach": approach,
            "code": code,
            "score": None,
            "success": False,
            "error": first_error,
            "notes": notes,
            "instance_scores": instance_scores,
            "instance_errors": instance_errors,
            "token_usage": token_usage,
            "llm_time": llm_time,
            "exec_time": exec_time,
        }

    aggregate_score = sum(instance_scores.values())
    notes_parts = [f"Aggregate={aggregate_score}"]
    for name in sorted(instance_scores.keys()):
        notes_parts.append(f"{name}={instance_scores[name]}")
    if instance_errors:
        failed_names = sorted(instance_errors.keys())
        reasons = []
        for failed_name in failed_names:
            err = instance_errors[failed_name]
            lines = err.strip().splitlines()
            last = lines[-1] if lines else err[:80]
            reasons.append(f"{failed_name}: {last[:80]}")
        notes_parts.append(f"Failed: {'; '.join(reasons)}")
    if notes:
        notes_parts.append(notes)

    return {
        "agent_id": agent_id,
        "direction": direction,
        "approach": approach,
        "code": code,
        "score": aggregate_score if all_passed else None,
        "success": all_passed,
        "error": first_error if not all_passed else None,
        "failure_reason": f"passed {len(instance_scores)}/{len(problems)} instances" if not all_passed else None,
        "notes": " | ".join(notes_parts),
        "instance_scores": instance_scores,
        "instance_errors": instance_errors,
        "token_usage": token_usage,
        "llm_time": llm_time,
        "exec_time": exec_time,
    }


async def _request_worker_draft(
    role: str,
    agent_id: int,
    iteration: int,
    prompt: str,
    system_prompt: str,
    endpoint: LLMEndpoint,
    config: Config,
    prompt_logger: PromptLogger | None,
    usage: RunUsage,
) -> tuple[AgentRunResult[WorkerDraft], TokenUsage | None]:
    worker_agent = build_worker_agent(config, endpoint, system_prompt)
    usage_before = UsageSnapshot.capture(usage)
    result = await worker_agent.run(
        prompt,
        usage=usage,
        model_settings={
            "temperature": config.llm.temperature_worker,
            "max_tokens": config.llm.max_tokens_worker,
        },
    )
    if prompt_logger:
        prompt_logger.log_structured(
            role=role,
            agent_id=agent_id,
            iteration=iteration,
            system_prompt=system_prompt,
            user_prompt=prompt,
            output=result.output,
            messages_json=result.all_messages_json(),
        )
    usage_after = UsageSnapshot.capture(usage)
    return result, TokenUsage.from_usage_delta(usage_before, usage_after)


async def _validate_generated_code(
    code: str,
    instance: ProblemInstance,
    input_data: list[dict],
    problem: ProblemBase,
    function_name: str,
    config: Config,
    telemetry: TelemetrySink | None = None,
    agent_id: int | None = None,
    iteration: int = 0,
    process_label: str | None = None,
) -> str | None:
    """Run the smallest-instance validation used before the full benchmark set."""
    pre_result = await execute_agent_code_async(
        code,
        input_data,
        config.sandbox,
        function_name=function_name,
        telemetry=telemetry,
        process_label=process_label or f"agent_{agent_id} sandbox precheck",
        process_metadata={"agent_id": agent_id, "iteration": iteration, "stage": "precheck"},
    )
    if not pre_result["success"]:
        return pre_result["error"]

    solution = problem.extract_solution(pre_result)
    eval_check = problem.evaluate(instance, solution)
    if not eval_check["valid"]:
        return eval_check["error"]
    return None
