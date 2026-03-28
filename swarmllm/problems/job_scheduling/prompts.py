from __future__ import annotations

"""
Job Scheduling Problem — Prompts

All LLM prompts specific to the job scheduling (minimize total tardiness) problem.
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
