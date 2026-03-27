from __future__ import annotations

"""
Worker Agent

Each agent receives a direction from the coordinator,
proposes a solution approach, writes code, and tests it.
"""

import re
import time
from config import Config
from problem import ProblemInstance, evaluate_schedule
from sandbox import execute_agent_code
from llm_client import chat_completion
from prompt_logger import PromptLogger


FIX_PROMPT = """\
Your code failed when tested.

## Error:
```
{error}
```

## Your original code:
```python
{code}
```

Fix the code so it works correctly. The function must return a valid permutation
of ALL job IDs — every job exactly once, no duplicates, no missing.
Your code will be tested on multiple instances of different sizes.

Output your fixed code in this exact format:

APPROACH: <one-line description of your approach>

```python
<your fixed code here>
```

NOTES: <what you fixed>
"""


AGENT_SYSTEM_PROMPT = """\
You are an optimization coder working on a job scheduling problem.
You will be given a problem description and a research direction to explore.
Your job is to write a Python function that produces a good schedule.

RULES:
1. You MUST define a function: def schedule(jobs: list[dict]) -> list[int]
2. Each job dict has keys: "id", "processing_time", "due_date"
3. Return a list of job IDs (a permutation of all job IDs) — the order jobs should be processed
4. The goal is to MINIMIZE total tardiness (lower is better)
5. Prefer standard library and already-installed packages when possible. You may import any pip package (it will be auto-installed). Already installed: {pip_packages}. Blocked: os, sys, subprocess, socket, and other system/network modules.
6. Your code has a {timeout}s time limit.
7. Be creative and try novel approaches based on your assigned direction

Output your response in this exact format:

APPROACH: <one-line description of your approach>

```python
<your complete code here, must define schedule(jobs) function>
```

NOTES: <brief notes on why this might work or any caveats>
"""


async def run_agent(
    agent_id: int,
    direction: str,
    problems: list[tuple[str, ProblemInstance]],
    config: Config,
    iteration: int = 0,
    prompt_logger: PromptLogger | None = None,
    top_solutions: list[dict] | None = None,
) -> dict:
    """
    Run a single worker agent.

    problems: list of (instance_name, ProblemInstance) tuples.
    Tests the generated code on all problem instances.
    Returns a dict with: agent_id, direction, approach, code, score, success, error, notes,
                         instance_scores, instance_errors
    """
    # Show smallest instance as example, note that code will be tested on diverse instances
    example_name, example_problem = problems[0]  # first (smallest) as example
    instance_desc = ", ".join(f"{name} ({len(p.jobs)} jobs)" for name, p in problems)

    prompt = f"""{example_problem.to_description()}

Your code will be tested on {len(problems)} diverse instances: {instance_desc}.
They vary in size, deadline tightness, and processing time ranges.
Write a general algorithm — do not hardcode for a specific instance.

## Your Research Direction

{direction}

Think carefully about this direction. Design a scheduling algorithm that follows
this approach. Write clean, working Python code.
"""

    pip_list = ", ".join(config.sandbox.pip_packages) if config.sandbox.pip_packages else "none"
    system_prompt = AGENT_SYSTEM_PROMPT.format(timeout=config.sandbox.timeout, pip_packages=pip_list)

    # Ask the LLM to generate a solution
    token_usage = None
    llm_time = 0.0
    try:
        llm_start = time.time()
        response, token_usage = await chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            config=config.llm,
            model=config.llm.agent_model,
            temperature=config.llm.temperature_worker,
            max_tokens=config.llm.max_tokens_worker,
        )
        llm_time = time.time() - llm_start
    except Exception as e:
        if prompt_logger:
            prompt_logger.log("agent", agent_id, iteration, system_prompt, prompt, "", str(e))
        return {
            "agent_id": agent_id,
            "direction": direction,
            "approach": "Failed to get LLM response",
            "code": "",
            "score": None,
            "success": False,
            "error": f"LLM error: {str(e)}",
            "notes": "",
            "instance_scores": {},
            "instance_errors": {},
            "token_usage": None,
            "llm_time": 0.0,
            "exec_time": 0.0,
            "retries_used": 0,
        }

    # Log the prompt and response
    if prompt_logger:
        prompt_logger.log("agent", agent_id, iteration, system_prompt, prompt, response)

    # Parse the response
    approach, code, notes = _parse_response(response)

    if not code:
        return {
            "agent_id": agent_id,
            "direction": direction,
            "approach": approach or "Could not parse response",
            "code": response[:1000],
            "score": None,
            "success": False,
            "error": "No valid Python code block found in response",
            "notes": notes,
            "instance_scores": {},
            "instance_errors": {},
            "token_usage": token_usage,
            "llm_time": llm_time,
            "exec_time": 0.0,
            "retries_used": 0,
        }

    # Pre-test on all instances — if any fails, give LLM retries with error feedback
    def _pre_test(test_code):
        """Test code on all instances, return first error or None."""
        for inst_name, problem in problems:
            job_data = [
                {"id": j.id, "processing_time": j.processing_time, "due_date": j.due_date}
                for j in problem.jobs
            ]
            result = execute_agent_code(test_code, job_data, config.sandbox)
            if not result["success"]:
                return f"[{inst_name}] {result['error']}"
            eval_check = evaluate_schedule(problem, result["schedule"])
            if not eval_check["valid"]:
                return f"[{inst_name}] {eval_check['error']}"
        return None

    pre_error = _pre_test(code)

    retries_used = 0
    for retry in range(config.swarm.agent_retries):
        if pre_error is None:
            break  # pre-test passed, no retry needed

        retries_used += 1

        # Feed the error back to the LLM
        fix_prompt = FIX_PROMPT.format(
            error=pre_error[:1000],
            code=code,
        )
        try:
            fix_start = time.time()
            fix_response, fix_tokens = await chat_completion(
                prompt=fix_prompt,
                system_prompt=system_prompt,
                config=config.llm,
                model=config.llm.agent_model,
                temperature=config.llm.temperature_worker,
                max_tokens=config.llm.max_tokens_worker,
            )
            llm_time += time.time() - fix_start
            if fix_tokens and token_usage:
                token_usage.prompt_tokens += fix_tokens.prompt_tokens
                token_usage.completion_tokens += fix_tokens.completion_tokens
                token_usage.total_tokens += fix_tokens.total_tokens

            if prompt_logger:
                prompt_logger.log(f"agent_fix{retry+1}", agent_id, iteration,
                                  system_prompt, fix_prompt, fix_response)

            fix_approach, fix_code, fix_notes = _parse_response(fix_response)
            if fix_code:
                code = fix_code
                if fix_approach:
                    approach = fix_approach
                if fix_notes:
                    notes = fix_notes + f" (fixed after pre-test retry {retry+1})"

                # Re-test on all instances
                pre_error = _pre_test(code)
            else:
                break  # couldn't parse fixed code, stop retrying
        except Exception:
            break  # LLM call failed, stop retrying

    # Test on all instances
    exec_start = time.time()
    instance_scores = {}
    instance_errors = {}
    first_error = None

    for inst_name, problem in problems:
        job_data = [
            {"id": j.id, "processing_time": j.processing_time, "due_date": j.due_date}
            for j in problem.jobs
        ]

        exec_result = execute_agent_code(code, job_data, config.sandbox)

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

    # Determine overall success and aggregate score
    # Only count as fully successful if ALL instances passed
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
            "retries_used": retries_used,
        }

    aggregate_score = sum(instance_scores.values())
    notes_parts = [f"Aggregate={aggregate_score}"]
    for name in sorted(instance_scores.keys()):
        notes_parts.append(f"{name}={instance_scores[name]}")
    if instance_errors:
        failed_names = sorted(instance_errors.keys())
        reasons = []
        for fn in failed_names:
            err = instance_errors[fn]
            # Extract exception type from traceback
            lines = err.strip().splitlines()
            last = lines[-1] if lines else err[:80]
            reasons.append(f"{fn}: {last[:80]}")
        notes_parts.append(f"Failed: {'; '.join(reasons)}")
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
        "retries_used": retries_used,
    }


def _parse_response(response: str) -> tuple[str, str, str]:
    """
    Parse agent LLM response to extract approach, code, and notes.

    Returns (approach, code, notes).
    """
    # Extract approach
    approach = ""
    approach_match = re.search(r"APPROACH:\s*(.+)", response)
    if approach_match:
        approach = approach_match.group(1).strip()

    # Extract code block
    code = ""
    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    # Extract notes
    notes = ""
    notes_match = re.search(r"NOTES:\s*(.+?)(?:\n\n|$)", response, re.DOTALL)
    if notes_match:
        notes = notes_match.group(1).strip()

    return approach, code, notes
