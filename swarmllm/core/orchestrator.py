from __future__ import annotations

"""
Orchestrator

Main loop that ties everything together:
1. Generate problem instance
2. Get initial directions from coordinator
3. For each iteration:
   a. Run all agents in parallel (with concurrency limit)
   b. Log results
   c. Ask coordinator for next directions
4. Print final summary
"""

import asyncio
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict

from swarmllm.config import Config
from swarmllm.llm.schemas import DirectionAssignment
from swarmllm.problems.scheduling import (
    generate_instance,
    save_instance,
    ProblemInstance,
    evaluate_schedule,
    baseline_fifo,
    baseline_edd,
    baseline_spt,
)
from swarmllm.core.agent import run_agent
from swarmllm.core.coordinator import get_initial_directions, get_next_directions
from swarmllm.llm.health import validate_backend_or_raise
from swarmllm.llm.routing import EndpointRouter
from swarmllm.tracking.shared_log import SharedLog
from swarmllm.tracking.prompt_logger import PromptLogger
from swarmllm.tracking.telemetry import DashboardRequest, RunTelemetry
from swarmllm.tracking.token_tracker import TokenTracker


async def run_swarm(
    config: Config,
    output_dir: str = ".",
    dashboard_mode: DashboardRequest = "auto",
):
    """Run the full swarm optimization loop."""
    telemetry = RunTelemetry(output_dir, requested_mode=dashboard_mode, is_tty=sys.stdout.isatty())
    telemetry.set_run_metadata(
        total_iterations=config.swarm.num_iterations,
        total_agents=config.swarm.num_agents,
        current_stage="starting",
    )
    telemetry.emit_event(
        "run_started",
        message="Swarm run started",
        backend_kind=config.llm.backend_kind,
        output_dir=output_dir,
    )
    _register_launcher_process(telemetry)

    try:
        with contextlib.redirect_stdout(telemetry.stdout_mirror()), contextlib.redirect_stderr(telemetry.stderr_mirror()):
            result = await _run_swarm_impl(config, output_dir, telemetry)
        telemetry.close(status="completed")
        return result
    except Exception as exc:
        telemetry.emit_event(
            "run_failed",
            message=f"Run failed: {exc}",
            level="error",
        )
        telemetry.close(status="failed")
        raise


async def _run_swarm_impl(config: Config, output_dir: str, telemetry: RunTelemetry):
    """Run the inner swarm loop with redirected logging and telemetry."""
    print("=" * 60)
    print("  SwarmLLM — Multi-Agent Optimization")
    print("=" * 60)
    print()

    telemetry.set_stage("backend validation")
    telemetry.emit_event("backend_validation_started", message=f"Validating {config.llm.backend_kind} backend profile")
    print(f"[0/3] Validating {config.llm.backend_kind} backend profile...")
    await validate_backend_or_raise(config.llm)
    router = EndpointRouter(config.llm)
    telemetry.emit_event("backend_validation_completed", message="Backend validation passed")
    print("  Backend validation passed.")
    print()

    # 1. Generate problem instances (one per profile)
    profiles = config.problem.instances
    telemetry.set_stage("problem generation")
    print(f"[1/3] Generating {len(profiles)} problem instances...")
    problems = []
    instance_names = []  # human-readable names for each instance
    for i, profile in enumerate(profiles):
        problem = generate_instance(
            num_jobs=profile.num_jobs,
            seed=config.problem.seed + i,  # different but reproducible per instance
            min_pt=profile.min_processing_time,
            max_pt=profile.max_processing_time,
            due_date_tightness=profile.due_date_tightness,
        )
        problems.append(problem)
        instance_names.append(profile.name)

    # Save instances to disk
    instances_dir = os.path.join(output_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)
    for problem, profile in zip(problems, profiles):
        path = os.path.join(instances_dir, f"{profile.name}.json")
        save_instance(problem, path, profile_name=profile.name, profile_params={
            "num_jobs": profile.num_jobs,
            "min_processing_time": profile.min_processing_time,
            "max_processing_time": profile.max_processing_time,
            "due_date_tightness": profile.due_date_tightness,
        })
    print(f"  Instances saved to: {instances_dir}")

    # Compute baselines per instance
    all_baselines = {}
    print()
    print(f"  {'=' * 56}")
    print(f"  PROBLEM INSTANCES & BASELINES")
    print(f"  Objective: minimize total tardiness (lower is better)")
    print(f"  {'=' * 56}")
    for problem, profile in zip(problems, profiles):
        baselines = _compute_baselines(problem)
        all_baselines[profile.name] = baselines
        best_baseline_name = min(baselines, key=baselines.get)
        print(f"\n  {profile.name}")
        print(f"    {profile.num_jobs} jobs | tightness={profile.due_date_tightness} "
              f"| processing time={profile.min_processing_time}-{profile.max_processing_time} "
              f"| total PT={problem.total_processing_time}")
        for name, score in baselines.items():
            marker = " <-- best" if name == best_baseline_name else ""
            print(f"    {name:>4}: {score:>10,} tardiness{marker}")

    # Aggregate baseline scores (sum across all instances, per strategy)
    agg_baselines = {}
    for name in ["FIFO", "EDD", "SPT"]:
        agg_baselines[name] = sum(all_baselines[n][name] for n in instance_names)

    best_agg_name = min(agg_baselines, key=agg_baselines.get)
    telemetry.set_baseline(min(agg_baselines.values()))
    telemetry.emit_event(
        "baselines_ready",
        message=f"Computed baselines across {len(instance_names)} instances",
        aggregate_baselines=agg_baselines,
    )
    print(f"\n  {'—' * 56}")
    print(f"  AGGREGATE (sum across all instances) — target to beat:")
    for name, score in agg_baselines.items():
        marker = " <-- best" if name == best_agg_name else ""
        print(f"    {name:>4}: {score:>10,} tardiness{marker}")
    print(f"  {'=' * 56}")
    print()

    # 2. Initialize log, prompt logger, and token tracker
    log = SharedLog(config.log, output_dir)
    prompt_logger = PromptLogger(output_dir)
    token_tracker = TokenTracker()

    # Save config for this run
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    # Record baselines in log
    log.append_coordinator_summary(
        iteration=0,
        summary=_format_baselines(all_baselines, agg_baselines),
    )

    # 3. Main loop
    best_score = min(agg_baselines.values())
    best_approach = "baseline"
    telemetry.set_best_score(best_score)
    best_instance_scores = {}  # per-instance scores for best solution
    # Track top solutions (score, approach, code) for exploit agents
    top_solutions = []
    # Track per-iteration stats for summary
    iteration_stats = []

    for iteration in range(1, config.swarm.num_iterations + 1):
        telemetry.start_iteration(iteration)
        telemetry.emit_event("iteration_started", message=f"Iteration {iteration} started", iteration=iteration)
        print(f"{'=' * 60}")
        print(f"  Iteration {iteration}/{config.swarm.num_iterations}")
        print(f"{'=' * 60}")

        # Get directions from coordinator LLM
        coord_endpoint = router.coordinator_endpoint()
        if iteration == 1:
            telemetry.set_stage("coordinator planning")
            telemetry.emit_event(
                "coordinator_request_started",
                message="Coordinator assigning initial directions",
                iteration=iteration,
                endpoint_label=coord_endpoint.label or coord_endpoint.base_url,
            )
            print("  Coordinator assigning initial directions...")
            coord_start = time.time()
            assignments, coord_tokens = await get_initial_directions(
                config=config,
                endpoint=coord_endpoint,
                prompt_logger=prompt_logger,
            )
            coord_duration = time.time() - coord_start
            if coord_tokens:
                token_tracker.record(
                    "coordinator",
                    iteration,
                    None,
                    config.llm.coordinator_model,
                    coord_tokens,
                    duration_seconds=coord_duration,
                )
                telemetry.record_llm_call(
                    role="coordinator",
                    iteration=iteration,
                    agent_id=None,
                    model=config.llm.coordinator_model,
                    duration_seconds=coord_duration,
                    usage=coord_tokens,
                    endpoint_label=coord_endpoint.label or coord_endpoint.base_url,
                )
            telemetry.emit_event(
                "coordinator_request_completed",
                message=f"Coordinator returned {len(assignments)} assignments",
                iteration=iteration,
                duration_seconds=coord_duration,
            )
        else:
            telemetry.set_stage("coordinator analysis")
            telemetry.emit_event(
                "coordinator_request_started",
                message=f"Coordinator analyzing iteration {iteration - 1}",
                iteration=iteration,
                endpoint_label=coord_endpoint.label or coord_endpoint.base_url,
            )
            print("  Coordinator analyzing results...")
            last_iter_content = log.read_iteration(iteration - 1)
            best_solution = top_solutions[0] if top_solutions else None
            coord_start = time.time()
            analysis, assignments, coord_tokens = await get_next_directions(
                iteration=iteration,
                last_iteration_content=last_iter_content,
                config=config,
                endpoint=coord_endpoint,
                prompt_logger=prompt_logger,
                best_solution=best_solution,
            )
            coord_duration = time.time() - coord_start
            if coord_tokens:
                token_tracker.record(
                    "coordinator",
                    iteration,
                    None,
                    config.llm.coordinator_model,
                    coord_tokens,
                    duration_seconds=coord_duration,
                )
                telemetry.record_llm_call(
                    role="coordinator",
                    iteration=iteration,
                    agent_id=None,
                    model=config.llm.coordinator_model,
                    duration_seconds=coord_duration,
                    usage=coord_tokens,
                    endpoint_label=coord_endpoint.label or coord_endpoint.base_url,
                )
            telemetry.emit_event(
                "coordinator_request_completed",
                message=f"Coordinator analyzed iteration {iteration - 1}",
                iteration=iteration,
                duration_seconds=coord_duration,
            )
            print(f"  Coordinator analysis: {analysis[:200]}...")
            log.append_coordinator_summary(iteration, analysis)
            telemetry.emit_event(
                "coordinator_analysis_logged",
                message=_truncate_analysis(analysis),
                iteration=iteration,
            )

        # Run agents with concurrency limit
        telemetry.set_stage("agent execution")
        print(f"  Running {config.swarm.num_agents} agents "
              f"(max {config.swarm.max_concurrent_agents} concurrent)...")
        print(f"  Testing on {len(profiles)} instances: {', '.join(instance_names)}")
        start_time = time.time()

        # Build (name, problem) tuples for agents
        named_problems = list(zip(instance_names, problems))

        results = await _run_agents_parallel(
            assignments=assignments,
            problems=named_problems,
            config=config,
            router=router,
            iteration=iteration,
            prompt_logger=prompt_logger,
            top_solutions=top_solutions,
            telemetry=telemetry,
        )

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")

        # Log results, track progress, record tokens
        successful = 0
        for result in results:
            if result.get("token_usage"):
                token_tracker.record(
                    "agent",
                    iteration,
                    result["agent_id"],
                    config.llm.agent_model,
                    result["token_usage"],
                    duration_seconds=result.get("llm_time"),
                )
            log.append_result(
                iteration=iteration,
                agent_id=result["agent_id"],
                direction=result["direction"],
                approach=result["approach"],
                code=result["code"],
                score=result["score"],
                success=result["success"],
                error=result["error"],
                notes=result.get("notes", ""),
                runtime=result.get("runtime"),
                failure_reason=result.get("failure_reason"),
                instance_scores=result.get("instance_scores"),
                instance_errors=result.get("instance_errors"),
                llm_time=result.get("llm_time"),
                exec_time=result.get("exec_time"),
            )

            if result["success"]:
                successful += 1
                top_solutions.append({
                    "score": result["score"],
                    "approach": result["approach"],
                    "code": result["code"],
                    "instance_scores": result.get("instance_scores", {}),
                })
                # Keep only top 5, sorted by aggregate score
                top_solutions.sort(key=lambda x: x["score"])
                top_solutions = top_solutions[:5]

                if result["score"] < best_score:
                    best_score = result["score"]
                    best_approach = result["approach"]
                    best_instance_scores = result.get("instance_scores", {})
                    telemetry.record_new_best(iteration, result["agent_id"], result["score"], result["approach"])
                    parts = []
                    for n in instance_names:
                        if n in result.get("instance_scores", {}):
                            parts.append(f"{n}={result['instance_scores'][n]}")
                        elif n in result.get("instance_errors", {}):
                            reason = _categorize_failure(result["instance_errors"][n])
                            # Short version for console — take first phrase before " —"
                            short = reason.split(" —")[0]
                            parts.append(f"{n}=FAIL({short})")
                        else:
                            parts.append(f"{n}=N/A")
                    scores_detail = ", ".join(parts)
                    print(f"  ** New best! Agent {result['agent_id']}: "
                          f"aggregate={result['score']} ({scores_detail}) "
                          f"({result['approach']})")

        # Collect failure reasons
        failure_counts = {}
        for r in results:
            if not r["success"]:
                reason = r.get("failure_reason") or _categorize_failure(r.get("error", ""))
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        # Collect timing stats
        llm_times = [r.get("llm_time", 0) for r in results if r.get("llm_time")]
        exec_times = [r.get("exec_time", 0) for r in results if r.get("exec_time")]
        successful_scores = [r["score"] for r in results if r["success"]]

        iter_stat = {
            "iteration": iteration,
            "num_agents": config.swarm.num_agents,
            "successful": successful,
            "failed": config.swarm.num_agents - successful,
            "failure_counts": failure_counts,
            "best_score_this_iter": min(successful_scores) if successful_scores else None,
            "best_score_overall": best_score,
            "best_approach": best_approach,
            "wall_time": elapsed,
            "llm_times": llm_times,
            "exec_times": exec_times,
            "avg_llm_time": sum(llm_times) / len(llm_times) if llm_times else 0,
            "avg_exec_time": sum(exec_times) / len(exec_times) if exec_times else 0,
        }
        iteration_stats.append(iter_stat)
        telemetry.finish_iteration(
            iteration,
            successful=successful,
            failed=config.swarm.num_agents - successful,
            best_score_this_iter=iter_stat["best_score_this_iter"],
            best_score_overall=best_score,
            wall_time_seconds=elapsed,
            avg_llm_time_seconds=iter_stat["avg_llm_time"],
            avg_exec_time_seconds=iter_stat["avg_exec_time"],
            failure_counts=failure_counts,
        )
        telemetry.emit_event(
            "iteration_completed",
            message=f"Iteration {iteration} completed",
            iteration=iteration,
            successful=successful,
            failed=config.swarm.num_agents - successful,
            best_score_overall=best_score,
        )

        print(f"  Results: {successful}/{config.swarm.num_agents} successful, "
              f"best aggregate: {best_score}")
        _print_iteration_summary(results, instance_names)
        token_tracker.print_iteration_tokens(iteration)
        token_tracker.print_running_total()
        print()

    # 4. Final summary
    _print_final_summary(log, all_baselines, agg_baselines, best_score, best_approach,
                         best_instance_scores, iteration_stats, config, output_dir,
                         token_tracker)

    return best_score, best_approach


async def _run_agents_parallel(
    assignments: list[DirectionAssignment],
    problems: list[tuple[str, ProblemInstance]],
    config: Config,
    router: EndpointRouter,
    iteration: int = 0,
    prompt_logger: PromptLogger | None = None,
    top_solutions: list[dict] | None = None,
    telemetry: RunTelemetry | None = None,
) -> list[dict]:
    """Run all agents with a concurrency semaphore."""
    semaphore = asyncio.Semaphore(config.swarm.max_concurrent_agents)
    scheduled = [(assignment, router.worker_endpoint()) for assignment in assignments]

    for assignment, endpoint in scheduled:
        endpoint_label = endpoint.label or endpoint.base_url
        if telemetry:
            telemetry.queue_agent(
                assignment.agent_id,
                iteration,
                assignment.mode,
                assignment.direction,
                endpoint_label,
            )
            telemetry.emit_event(
                "agent_assignment",
                message=f"Agent {assignment.agent_id} assigned {assignment.mode}",
                agent_id=assignment.agent_id,
                iteration=iteration,
                mode=assignment.mode,
                endpoint_label=endpoint_label,
                direction=assignment.direction,
            )

    async def run_with_limit(agent_id: int, assignment: DirectionAssignment, endpoint):
        async with semaphore:
            endpoint_label = endpoint.label or endpoint.base_url
            if telemetry:
                telemetry.set_agent_phase(
                    agent_id,
                    iteration,
                    phase="llm",
                    status="running",
                    endpoint_label=endpoint_label,
                    direction=assignment.direction,
                    mode=assignment.mode,
                )
            print(f"    Agent {agent_id:2d}: {assignment.direction[:100]}...")
            agent_start = time.time()
            result = await run_agent(
                agent_id, assignment.direction, problems, config,
                endpoint=endpoint,
                iteration=iteration, prompt_logger=prompt_logger,
                top_solutions=top_solutions,
                telemetry=telemetry,
            )
            elapsed = time.time() - agent_start
            result["runtime"] = elapsed
            llm_t = result.get("llm_time", 0)
            exec_t = result.get("exec_time", 0)
            timing = f"LLM: {llm_t:.1f}s, exec: {exec_t:.1f}s, total: {elapsed:.1f}s"
            if result["success"]:
                status = f"score={result['score']} [{timing}]"
            else:
                reason = _categorize_failure(result.get("error", ""))
                result["failure_reason"] = reason
                status = f"FAILED ({reason}) [{timing}]"
            if telemetry:
                telemetry.complete_agent(
                    agent_id=agent_id,
                    iteration=iteration,
                    success=result["success"],
                    score=result["score"],
                    llm_time_seconds=result.get("llm_time", 0.0),
                    exec_time_seconds=result.get("exec_time", 0.0),
                    runtime_seconds=elapsed,
                    failure_reason=result.get("failure_reason"),
                    instance_scores=result.get("instance_scores"),
                    instance_errors=result.get("instance_errors"),
                    approach=result.get("approach"),
                )
                telemetry.emit_event(
                    "agent_completed",
                    message=f"Agent {agent_id} {'succeeded' if result['success'] else 'failed'}",
                    level="info" if result["success"] else "warning",
                    agent_id=agent_id,
                    iteration=iteration,
                    success=result["success"],
                    score=result["score"],
                    failure_reason=result.get("failure_reason"),
                )
            print(f"    Agent {agent_id:2d} done: {status}")
            return result

    tasks = [
        run_with_limit(assignment.agent_id, assignment, endpoint)
        for assignment, endpoint in scheduled
    ]

    return await asyncio.gather(*tasks)


def _register_launcher_process(telemetry: RunTelemetry) -> None:
    """Register an auto-started local backend process when launched by setup_run.py."""
    raw_pid = os.environ.get("SWARMLLM_LOCAL_SERVER_PID")
    if not raw_pid:
        return
    try:
        pid = int(raw_pid)
    except ValueError:
        return
    kind = os.environ.get("SWARMLLM_LOCAL_SERVER_KIND", "backend")
    log_path = os.environ.get("SWARMLLM_LOCAL_SERVER_LOG_PATH", "")
    label = f"{kind} server"
    telemetry.register_process(
        pid=pid,
        label=label,
        kind="backend",
        role="model-server",
        metadata={"log_path": log_path, "kind": kind},
    )
    telemetry.emit_event(
        "local_backend_registered",
        message=f"Tracking local {kind} server (pid {pid})",
        pid=pid,
        kind=kind,
        log_path=log_path,
    )


def _truncate_analysis(analysis: str, limit: int = 220) -> str:
    """Shorten coordinator analysis for the recent-events feed."""
    compact = " ".join(analysis.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 1]}..."


def _categorize_failure(error: str) -> str:
    """Categorize a failure error string into an actionable reason for the coordinator."""
    if "timed out" in error.lower():
        return "execution timed out — code took too long, needs a faster algorithm or early termination"
    elif "not allowed" in error:
        # Extract which module was blocked
        module = _extract_between(error, "Import of '", "'")
        if module:
            return f"blocked import '{module}' — agent tried to use a forbidden module"
        return "blocked import — agent tried to use a forbidden module"
    elif "ModuleNotFoundError" in error or "ImportError" in error:
        module = _extract_between(error, "No module named '", "'") or _extract_between(error, "cannot import name '", "'")
        if module:
            return f"missing module '{module}' — library not available in sandbox"
        return "import error — required library not available"
    elif "IndentationError" in error:
        return "indentation error — code has broken indentation, likely malformed output"
    elif "SyntaxError" in error:
        detail = _extract_between(error, "SyntaxError: ", "\n") or ""
        return f"syntax error — invalid Python syntax{': ' + detail if detail else ''}"
    elif "NameError" in error:
        name = _extract_between(error, "name '", "'")
        if name:
            return f"undefined variable '{name}' — code references something not defined"
        return "undefined variable — code references something not defined"
    elif "TypeError" in error:
        detail = _extract_last_line_with(error, "TypeError:")
        return f"type error — {detail}" if detail else "type error — wrong argument types or bad function call"
    elif "IndexError" in error:
        return "index error — code accessed a list/array out of bounds"
    elif "KeyError" in error:
        key = _extract_between(error, "KeyError: ", "\n")
        return f"key error — missing key {key}" if key else "key error — accessed missing dictionary key"
    elif "ValueError" in error:
        detail = _extract_last_line_with(error, "ValueError:")
        return f"value error — {detail}" if detail else "value error — invalid value in computation"
    elif "ZeroDivisionError" in error:
        return "division by zero — code divided by zero, needs a guard"
    elif "MemoryError" in error or "memory" in error.lower():
        return "out of memory — algorithm used too much RAM, needs a lighter approach"
    elif "RecursionError" in error:
        return "recursion limit — infinite or too-deep recursion"
    elif "No valid Python code" in error:
        return "no code generated — LLM response had no valid python code block"
    elif "LLM error" in error or "Failed after" in error:
        return "LLM unreachable — could not get a response from the model"
    elif "Invalid permutation" in error:
        missing = _extract_between(error, "Missing: {", "}")
        extra = _extract_between(error, "Extra: {", "}")
        parts = []
        if missing:
            parts.append(f"missing jobs: {{{missing}}}")
        if extra:
            parts.append(f"extra/duplicate jobs: {{{extra}}}")
        detail = ", ".join(parts) if parts else "not a valid permutation of all job IDs"
        return f"invalid schedule — {detail}"
    else:
        # Extract the actual exception type from the traceback
        exc_type = _extract_exception_type(error)
        if exc_type:
            detail = _extract_last_line_with(error, exc_type + ":")
            if detail:
                return f"{exc_type} — {detail}"
            return f"{exc_type} — unexpected error during execution"
        return f"runtime error — {error[:100]}"


def _extract_between(text: str, start: str, end: str) -> str:
    """Extract text between two markers."""
    try:
        s = text.index(start) + len(start)
        e = text.index(end, s)
        return text[s:e].strip()
    except ValueError:
        return ""


def _extract_last_line_with(text: str, marker: str) -> str:
    """Extract the content after the last occurrence of marker in the text."""
    for line in reversed(text.split("\n")):
        if marker in line:
            return line.split(marker, 1)[1].strip()[:120]
    return ""


def _extract_exception_type(error: str) -> str:
    """Extract the Python exception type name from a traceback string."""
    import re
    # Match common pattern: "ExceptionType: message" at start of a line
    match = re.search(r"^(\w*Error|\w*Exception|\w*Warning):", error, re.MULTILINE)
    if match:
        return match.group(1)
    return ""


def _compute_baselines(problem: ProblemInstance) -> dict[str, float]:
    """Compute baseline scores for reference."""
    results = {}
    for name, fn in [("FIFO", baseline_fifo), ("EDD", baseline_edd), ("SPT", baseline_spt)]:
        schedule = fn(problem)
        eval_result = evaluate_schedule(problem, schedule)
        results[name] = eval_result["total_tardiness"]
    return results


def _format_baselines(all_baselines: dict[str, dict], agg_baselines: dict[str, float]) -> str:
    lines = ["## Baseline Scores (for reference)", ""]
    for inst_name in all_baselines:
        lines.append(f"### {inst_name}")
        for name, score in all_baselines[inst_name].items():
            lines.append(f"- **{name}:** {score}")
        lines.append("")
    lines.append("### Aggregate (sum across all instances)")
    for name, score in agg_baselines.items():
        lines.append(f"- **{name}:** {score}")
    lines.append("")
    lines.append("Agents should aim to beat these baselines across all instances.")
    return "\n".join(lines)


def _print_iteration_summary(results: list[dict], instance_names: list[str]):
    """Print a compact summary of iteration results."""
    scores = [r["score"] for r in results if r["success"] and r["score"] is not None]
    if scores:
        scores.sort()
        print(f"  Aggregate scores: min={scores[0]}, median={scores[len(scores)//2]}, "
              f"max={scores[-1]} ({len(scores)} successful)")
        # Per-instance breakdown
        for name in instance_names:
            inst_scores = [
                r["instance_scores"][name]
                for r in results
                if r.get("instance_scores") and name in r["instance_scores"]
            ]
            if inst_scores:
                inst_scores.sort()
                print(f"    {name}: min={inst_scores[0]}, "
                      f"median={inst_scores[len(inst_scores)//2]}, "
                      f"max={inst_scores[-1]}")


def _print_final_summary(log, all_baselines, agg_baselines, best_score, best_approach,
                         best_instance_scores, iteration_stats, config,
                         output_dir, token_tracker: TokenTracker = None):
    """Print the final summary after all iterations and save to file."""
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    out()
    out("=" * 60)
    out("  FINAL RESULTS")
    out("=" * 60)

    # Run configuration
    out()
    out("  RUN CONFIGURATION:")
    out(f"    Backend kind:      {config.llm.backend_kind}")
    out(f"    Coordinator model: {config.llm.coordinator_model}")
    out(f"    Agent model:       {config.llm.agent_model}")
    out(f"    Coord endpoint:    {config.llm.coordinator_endpoints[0].base_url}")
    out(f"    Worker endpoints:  {', '.join(e.base_url for e in config.llm.worker_endpoints)}")
    out(f"    Agents per iter:   {config.swarm.num_agents}")
    out(f"    Max concurrent:    {config.swarm.max_concurrent_agents}")
    out(f"    Iterations:        {config.swarm.num_iterations}")
    out(f"    Explore ratio:     {config.swarm.explore_ratio}")
    out(f"    Code timeout:      {config.sandbox.timeout}s")
    out(f"    Seed:              {config.problem.seed}")

    # Baselines
    out()
    out("  BASELINES:")
    for inst_name in all_baselines:
        best_bl = min(all_baselines[inst_name], key=all_baselines[inst_name].get)
        scores = ", ".join(f"{n}: {s}" for n, s in all_baselines[inst_name].items())
        out(f"    {inst_name}: {scores} (best: {best_bl})")
    out(f"    Aggregate: {', '.join(f'{n}: {s}' for n, s in agg_baselines.items())}")
    best_baseline = min(agg_baselines.values())
    best_bl_name = min(agg_baselines, key=agg_baselines.get)
    out(f"    Target to beat: {best_baseline} ({best_bl_name})")

    # Best result
    out()
    out("  BEST RESULT:")
    out(f"    Aggregate score: {best_score}")
    if best_instance_scores:
        parts = [f"{name}={score}" for name, score in sorted(best_instance_scores.items())]
        out(f"    Per-instance:    {', '.join(parts)}")
    out(f"    Approach:        {best_approach}")
    out()
    if best_score < best_baseline:
        improvement = ((best_baseline - best_score) / best_baseline) * 100
        out(f"    >>> Swarm BEAT baselines by {improvement:.1f}% <<<")
    elif best_score == best_baseline:
        out(f"    --- Swarm MATCHED best baseline ---")
    else:
        gap = ((best_score - best_baseline) / best_baseline) * 100
        out(f"    Swarm did NOT beat baselines (gap: +{gap:.1f}%)")

    # Iteration progression
    out()
    out("  ITERATION PROGRESSION:")
    out(f"    {'Iter':>4} | {'Success':>7} | {'Failed':>6} | {'Best(iter)':>10} | {'Best(all)':>10} | {'Wall Time':>9} | {'Avg LLM':>8} | {'Avg Exec':>8}")
    out(f"    {'-'*4}-+-{'-'*7}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*8}")
    total_successful = 0
    total_failed = 0
    total_wall = 0
    for s in iteration_stats:
        iter_best = str(s["best_score_this_iter"]) if s["best_score_this_iter"] is not None else "—"
        out(f"    {s['iteration']:>4} | "
            f"{s['successful']:>3}/{s['num_agents']:<3} | "
            f"{s['failed']:>6} | "
            f"{iter_best:>10} | "
            f"{s['best_score_overall']:>10} | "
            f"{s['wall_time']:>8.1f}s | "
            f"{s['avg_llm_time']:>7.1f}s | "
            f"{s['avg_exec_time']:>7.1f}s")
        total_successful += s["successful"]
        total_failed += s["failed"]
        total_wall += s["wall_time"]

    # Collect all per-agent times across iterations
    all_llm_times = []
    all_exec_times = []
    for s in iteration_stats:
        all_llm_times.extend(s["llm_times"])
        all_exec_times.extend(s["exec_times"])
    total_llm_time = sum(all_llm_times)
    total_exec_time = sum(all_exec_times)

    # Overall agent stats
    total_agents = total_successful + total_failed
    success_rate = (total_successful / total_agents * 100) if total_agents > 0 else 0
    out()
    out("  AGENT STATS:")
    out(f"    Total agent runs:  {total_agents}")
    out(f"    Successful:        {total_successful} ({success_rate:.1f}%)")
    out(f"    Failed:            {total_failed} ({100 - success_rate:.1f}%)")
    out()
    out("  TIMING:")
    out(f"    Total wall time:     {total_wall:.1f}s ({total_wall/60:.1f}m)")
    out(f"    Total LLM time:      {total_llm_time:.1f}s ({total_llm_time/60:.1f}m) — waiting for model responses")
    out(f"    Total exec time:     {total_exec_time:.1f}s ({total_exec_time/60:.1f}m) — running agent code in sandbox")
    if all_llm_times:
        out(f"    Avg LLM per agent:   {sum(all_llm_times)/len(all_llm_times):.1f}s "
            f"(min: {min(all_llm_times):.1f}s, max: {max(all_llm_times):.1f}s)")
    if all_exec_times:
        out(f"    Avg exec per agent:  {sum(all_exec_times)/len(all_exec_times):.1f}s "
            f"(min: {min(all_exec_times):.1f}s, max: {max(all_exec_times):.1f}s)")

    # Failure breakdown
    all_failures = {}
    for s in iteration_stats:
        for reason, count in s["failure_counts"].items():
            all_failures[reason] = all_failures.get(reason, 0) + count
    if all_failures:
        out()
        out("  FAILURE BREAKDOWN:")
        for reason, count in sorted(all_failures.items(), key=lambda x: -x[1]):
            out(f"    {count:>4}x  {reason}")

    # Token usage
    if token_tracker:
        out()
        out("  TOKEN USAGE:")
        out(f"    Total tokens:        {token_tracker.total_tokens:>12,}")
        out(f"      Prompt tokens:     {token_tracker.total_prompt:>12,}")
        out(f"      Completion tokens: {token_tracker.total_completion:>12,}")
        if token_tracker.total_thinking:
            out(f"      Thinking tokens:   {token_tracker.total_thinking:>12,}  (subset of completion)")
        out()
        agent_calls = len([c for c in token_tracker._calls if c["role"] == "agent"])
        coord_calls = len([c for c in token_tracker._calls if c["role"] == "coordinator"])
        out(f"    Agent tokens:        {token_tracker.agent_total:>12,} ({agent_calls} calls)")
        out(f"      Prompt:            {token_tracker.agent_prompt:>12,}")
        out(f"      Completion:        {token_tracker.agent_completion:>12,}")
        if token_tracker.agent_thinking:
            out(f"      Thinking:          {token_tracker.agent_thinking:>12,}  (subset of completion)")
        if agent_calls > 0:
            out(f"      Avg per call:      {token_tracker.agent_total // agent_calls:>12,}")
        out(f"    Coordinator tokens:  {token_tracker.coordinator_total:>12,} ({coord_calls} calls)")
        out(f"      Prompt:            {token_tracker.coordinator_prompt:>12,}")
        out(f"      Completion:        {token_tracker.coordinator_completion:>12,}")
        if token_tracker.coordinator_thinking:
            out(f"      Thinking:          {token_tracker.coordinator_thinking:>12,}  (subset of completion)")
        if coord_calls > 0:
            out(f"      Avg per call:      {token_tracker.coordinator_total // coord_calls:>12,}")
        out()
        out("    Per iteration:")
        for it in sorted(token_tracker._iteration_totals.keys()):
            s = token_tracker._iteration_totals[it]
            thinking_str = f", thinking: {s['thinking_tokens']:,}" if s.get("thinking_tokens") else ""
            out(f"      Iter {it:>2}: {s['total_tokens']:>10,} tokens "
                f"(prompt: {s['prompt_tokens']:,}, completion: {s['completion_tokens']:,}{thinking_str}) "
                f"| {s['agent_calls']} agents + {s['coordinator_calls']} coordinator")
        token_tracker.save(output_dir)

    out()
    out(f"  Full results log: {log.path}")
    out()

    # Save summary to file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary saved to: {summary_path}")
