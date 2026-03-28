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
from swarmllm.problems.scheduling import ProblemInstance, evaluate_schedule
from swarmllm.sandbox.executor import execute_agent_code
from swarmllm.tracking.prompt_logger import PromptLogger
from swarmllm.tracking.telemetry import TelemetrySink
from swarmllm.tracking.token_tracker import TokenUsage, UsageSnapshot


FIX_PROMPT = """\
Your previous draft failed when tested on the smallest instance ({num_jobs} jobs).

Full error traceback:
{error}

Previous code:
```python
{code}
```

Revise the implementation so it returns a valid permutation of all job IDs,
with no duplicates or omissions, while preserving the assigned research direction.
"""


AGENT_SYSTEM_PROMPT = """\
You are an optimization coder working on a job scheduling problem.
You will be given a problem description and a research direction to explore.
Your job is to write a Python function that produces a good schedule.

Rules:
1. You must define a function: def schedule(jobs: list[dict]) -> list[int]
2. Each job dict has keys: "id", "processing_time", "due_date"
3. Return a list of job IDs representing the full schedule order
4. The goal is to minimize total tardiness (lower is better)
5. You may import pip packages already available in the sandbox: {pip_packages}
6. Dangerous system/network modules are blocked
7. Your code has a {timeout}s time limit

Return structured output with:
- approach: a short one-line description of the algorithm
- code: the complete Python source defining schedule(jobs)
- notes: brief rationale, caveats, or what you changed on a retry

Return only the structured result. Do not emit XML tags, <tools> wrappers,
markdown fences, or any prose outside the final structured output.
"""


async def run_agent(
    agent_id: int,
    direction: str,
    problems: list[tuple[str, ProblemInstance]],
    config: Config,
    endpoint: LLMEndpoint,
    iteration: int = 0,
    prompt_logger: PromptLogger | None = None,
    top_solutions: list[dict] | None = None,
    telemetry: TelemetrySink | None = None,
) -> dict:
    """
    Run a single worker agent against all configured problem instances.

    Returns a dict with:
        agent_id, direction, approach, code, score, success, error, notes,
        instance_scores, instance_errors, token_usage, llm_time, exec_time
    """
    del top_solutions  # Reserved for future exploit-agent prompt tuning.

    _, example_problem = problems[0]
    instance_desc = ", ".join(f"{name} ({len(problem.jobs)} jobs)" for name, problem in problems)
    prompt = f"""{example_problem.to_description()}

Your code will be tested on {len(problems)} diverse instances: {instance_desc}.
They vary in size, deadline tightness, and processing time ranges.
Write a general algorithm and do not hardcode for a specific instance.

## Research Direction

{direction}
"""

    pip_list = ", ".join(config.sandbox.pip_packages) if config.sandbox.pip_packages else "none"
    system_prompt = AGENT_SYSTEM_PROMPT.format(timeout=config.sandbox.timeout, pip_packages=pip_list)
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
        return {
            "agent_id": agent_id,
            "direction": direction,
            "approach": "Failed to get LLM response",
            "code": "",
            "score": None,
            "success": False,
            "error": f"LLM error: {exc}",
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

    _, smallest_problem = problems[0]
    smallest_jobs = [
        {"id": job.id, "processing_time": job.processing_time, "due_date": job.due_date}
        for job in smallest_problem.jobs
    ]
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
    pre_error = _validate_generated_code(
        code,
        smallest_problem,
        smallest_jobs,
        config,
        telemetry=telemetry,
        agent_id=agent_id,
        iteration=iteration,
        process_label=f"agent {agent_id} precheck",
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
        fix_prompt = FIX_PROMPT.format(
            num_jobs=len(smallest_problem.jobs),
            error=pre_error,
            code=code,
        )
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
        pre_error = _validate_generated_code(
            code,
            smallest_problem,
            smallest_jobs,
            config,
            telemetry=telemetry,
            agent_id=agent_id,
            iteration=iteration,
            process_label=f"agent {agent_id} precheck",
        )

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
    for inst_name, problem in problems:
        job_data = [
            {"id": job.id, "processing_time": job.processing_time, "due_date": job.due_date}
            for job in problem.jobs
        ]

        exec_result = execute_agent_code(
            code,
            job_data,
            config.sandbox,
            telemetry=telemetry,
            process_label=f"agent {agent_id} {inst_name}",
            process_metadata={"agent_id": agent_id, "iteration": iteration, "instance": inst_name},
        )
        if not exec_result["success"]:
            instance_errors[inst_name] = exec_result["error"]
            if first_error is None:
                first_error = exec_result["error"]
            continue

        schedule = exec_result["schedule"]
        eval_result = evaluate_schedule(problem, schedule)
        if not eval_result["valid"]:
            instance_errors[inst_name] = eval_result["error"]
            if first_error is None:
                first_error = eval_result["error"]
            continue

        instance_scores[inst_name] = eval_result["total_tardiness"]

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


def _validate_generated_code(
    code: str,
    problem: ProblemInstance,
    jobs: list[dict],
    config: Config,
    telemetry: TelemetrySink | None = None,
    agent_id: int | None = None,
    iteration: int = 0,
    process_label: str | None = None,
) -> str | None:
    """Run the smallest-instance validation used before the full benchmark set."""
    pre_result = execute_agent_code(
        code,
        jobs,
        config.sandbox,
        telemetry=telemetry,
        process_label=process_label or f"agent {agent_id} precheck",
        process_metadata={"agent_id": agent_id, "iteration": iteration, "stage": "precheck"},
    )
    if not pre_result["success"]:
        return pre_result["error"]

    eval_check = evaluate_schedule(problem, pre_result["schedule"])
    if not eval_check["valid"]:
        return eval_check["error"]
    return None
