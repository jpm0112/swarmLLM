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
import json
import os
import time
from dataclasses import asdict

from config import Config
from problem import (
    generate_instance,
    save_instance,
    ProblemInstance,
    evaluate_schedule,
    baseline_fifo,
    baseline_edd,
    baseline_spt,
)
from agent import run_agent
from coordinator import get_initial_directions, get_next_directions
from logger import SharedLog
from prompt_logger import PromptLogger
from token_tracker import TokenTracker


async def run_swarm(config: Config, output_dir: str = "."):
    """Run the full swarm optimization loop."""
    print("=" * 60)
    print("  SwarmLLM — Multi-Agent Optimization")
    print("=" * 60)
    print()

    # 1. Generate problem instances (one per profile)
    profiles = config.problem.instances
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
    for problem, profile in zip(problems, profiles):
        baselines = _compute_baselines(problem)
        all_baselines[profile.name] = baselines
        print(f"  {profile.name} ({profile.num_jobs} jobs, tightness={profile.due_date_tightness}, "
              f"pt={profile.min_processing_time}-{profile.max_processing_time}) — "
              f"total_pt={problem.total_processing_time}")
        for name, score in baselines.items():
            print(f"    {name}: {score}")
    print()

    # Aggregate baseline scores (sum across all instances, per strategy)
    agg_baselines = {}
    for name in ["FIFO", "EDD", "SPT"]:
        agg_baselines[name] = sum(all_baselines[n][name] for n in instance_names)

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
    best_instance_scores = {}  # per-instance scores for best solution
    # Track top solutions (score, approach, code) for exploit agents
    top_solutions = []

    for iteration in range(1, config.swarm.num_iterations + 1):
        print(f"{'=' * 60}")
        print(f"  Iteration {iteration}/{config.swarm.num_iterations}")
        print(f"{'=' * 60}")

        # Get directions
        if iteration == 1:
            print("  Getting initial directions...")
            directions = await get_initial_directions(config)
        else:
            print("  Coordinator analyzing results...")
            log_content = log.read()
            analysis, directions, coord_tokens = await get_next_directions(
                iteration=iteration,
                log_content=log_content,
                config=config,
                prompt_logger=prompt_logger,
                top_solutions=top_solutions,
            )
            if coord_tokens:
                token_tracker.record("coordinator", iteration, None,
                                     config.llm.coordinator_model, coord_tokens)
            print(f"  Coordinator analysis: {analysis[:200]}...")
            log.append_coordinator_summary(iteration, analysis)

        # Run agents with concurrency limit
        print(f"  Running {config.swarm.num_agents} agents "
              f"(max {config.swarm.max_concurrent_agents} concurrent)...")
        print(f"  Testing on {len(profiles)} instances: {', '.join(instance_names)}")
        start_time = time.time()

        # Build (name, problem) tuples for agents
        named_problems = list(zip(instance_names, problems))

        results = await _run_agents_parallel(
            directions=directions,
            problems=named_problems,
            config=config,
            iteration=iteration,
            prompt_logger=prompt_logger,
            top_solutions=top_solutions,
        )

        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s")

        # Log results, track progress, record tokens
        successful = 0
        for result in results:
            if result.get("token_usage"):
                token_tracker.record("agent", iteration, result["agent_id"],
                                     config.llm.agent_model, result["token_usage"])
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

        print(f"  Results: {successful}/{config.swarm.num_agents} successful, "
              f"best aggregate: {best_score}")
        _print_iteration_summary(results, instance_names)
        token_tracker.print_iteration_tokens(iteration)
        token_tracker.print_running_total()
        print()

    # 4. Final summary
    _print_final_summary(log, all_baselines, agg_baselines, best_score, best_approach,
                         best_instance_scores, output_dir, token_tracker)

    return best_score, best_approach


async def _run_agents_parallel(
    directions: list[str],
    problems: list[tuple[str, ProblemInstance]],
    config: Config,
    iteration: int = 0,
    prompt_logger: PromptLogger | None = None,
    top_solutions: list[dict] | None = None,
) -> list[dict]:
    """Run all agents with a concurrency semaphore."""
    semaphore = asyncio.Semaphore(config.swarm.max_concurrent_agents)

    async def run_with_limit(agent_id, direction):
        async with semaphore:
            print(f"    Agent {agent_id:2d}: {direction[:100]}...")
            agent_start = time.time()
            result = await run_agent(
                agent_id, direction, problems, config,
                iteration=iteration, prompt_logger=prompt_logger,
                top_solutions=top_solutions,
            )
            elapsed = time.time() - agent_start
            result["runtime"] = elapsed
            if result["success"]:
                status = f"score={result['score']} ({elapsed:.1f}s)"
            else:
                reason = _categorize_failure(result.get("error", ""))
                result["failure_reason"] = reason
                status = f"FAILED ({reason}) ({elapsed:.1f}s)"
            print(f"    Agent {agent_id:2d} done: {status}")
            return result

    tasks = [
        run_with_limit(i, directions[i])
        for i in range(len(directions))
    ]

    return await asyncio.gather(*tasks)


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
                         best_instance_scores, output_dir, token_tracker: TokenTracker = None):
    """Print the final summary after all iterations and save to file."""
    lines = []

    def out(msg=""):
        print(msg)
        lines.append(msg)

    out()
    out("=" * 60)
    out("  FINAL RESULTS")
    out("=" * 60)
    out()
    out("  Baselines (per instance):")
    for inst_name in all_baselines:
        out(f"    {inst_name}:")
        for name, score in all_baselines[inst_name].items():
            out(f"      {name}: {score}")
    out()
    out("  Baselines (aggregate):")
    for name, score in agg_baselines.items():
        out(f"    {name}: {score}")
    out()
    out(f"  Best swarm aggregate score: {best_score}")
    if best_instance_scores:
        parts = [f"{name}={score}" for name, score in sorted(best_instance_scores.items())]
        out(f"  Best per-instance scores: {', '.join(parts)}")
    out(f"  Best approach: {best_approach}")
    out()

    best_baseline = min(agg_baselines.values())
    if best_score < best_baseline:
        improvement = ((best_baseline - best_score) / best_baseline) * 100
        out(f"  Swarm BEAT baselines by {improvement:.1f}%")
    elif best_score == best_baseline:
        out(f"  Swarm MATCHED best baseline")
    else:
        out(f"  Swarm did NOT beat baselines (best aggregate baseline: {best_baseline})")

    out()
    if token_tracker:
        token_lines = token_tracker.print_final_summary()
        lines.extend(token_lines)
        token_tracker.save(output_dir)
    out()
    out(f"  Full results log: {log.path}")
    out()

    # Save summary to file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary saved to: {summary_path}")
