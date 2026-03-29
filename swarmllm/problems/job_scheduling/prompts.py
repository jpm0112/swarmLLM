from __future__ import annotations

"""
Job Scheduling Problem — Prompts

All LLM prompts specific to the job scheduling (minimize total tardiness) problem.
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
8. Write proper multi-line Python code using newlines and indentation. Never use semicolons to join statements on one line. Always include a complete implementation — never return stubs, ellipsis (...), or placeholder comments.

Return structured output with:
- approach: a short one-line description of the algorithm
- code: the complete Python source defining schedule(jobs)
- notes: brief rationale, caveats, or what you changed on a retry

Return only the structured result. Do not emit XML tags, <tools> wrappers,
markdown fences, or any prose outside the final structured output.
"""


FIX_PROMPT = """\
Your previous draft failed when tested.

Full error traceback:
{error}

Previous code:
```python
{code}
```

Revise the implementation so it returns a valid permutation of all job IDs,
with no duplicates or omissions, while preserving the assigned research direction.
"""


PROBLEM_DESCRIPTION = """\
# Job Scheduling Problem

**Objective:** Minimize total tardiness.
**Tardiness** of job j = max(0, completion_time_j - due_date_j)
**Total tardiness** = sum of tardiness across all jobs. Lower is better.

You must return an ordering (permutation) of job IDs. Jobs are processed
sequentially in the order you return. Each job starts immediately after the
previous one finishes.

## Function Signature

`def schedule(jobs: list[dict]) -> list[int]`

**Input:** `jobs` — a list of dicts, each with:
- `"id"` (int): unique job identifier (0 to N-1)
- `"processing_time"` (int): how long the job takes to complete
- `"due_date"` (int): the deadline for the job

**Output:** a list of all job IDs (ints) in the order they should be processed.
Every job must appear exactly once (a valid permutation)."""


COORDINATOR_PROBLEM_DESCRIPTION = "a job scheduling problem (minimize total tardiness)"
