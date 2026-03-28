from __future__ import annotations

"""
Job Shop Scheduling Problem — Prompts

All LLM prompts specific to the Job Shop Scheduling Problem (minimize makespan).
"""

AGENT_SYSTEM_PROMPT = """\
You are an optimization coder working on the Job Shop Scheduling Problem (JSP).
You will be given a problem description and a research direction to explore.
Your job is to write a Python function that produces a good schedule.

RULES:
1. You MUST define a function: def solve(jobs: list[list[dict]]) -> list[list[dict]]
2. Input: a list of jobs. Each job is a list of operations (in order).
   Each operation is a dict with keys: "machine" (int), "duration" (int)
3. Output: a list of jobs, where each job is a list of dicts with keys:
   "machine" (int), "duration" (int), "start" (int)
   The operations MUST be in the same order as the input (job routing is fixed).
4. Constraints:
   - Each machine can process only ONE operation at a time (no overlap)
   - Operations within a job must be processed in the given order
   - No preemption (once started, an operation runs to completion)
5. The goal is to MINIMIZE makespan (the time when ALL jobs are finished). Lower is better.
6. Prefer standard library and already-installed packages when possible. You may import any pip package (it will be auto-installed). Already installed: {pip_packages}. Blocked: os, sys, subprocess, socket, and other system/network modules.
7. Your code has a {timeout}s time limit.
8. Be creative and try novel approaches based on your assigned direction

Output your response in this exact format:

APPROACH: <one-line description of your approach>

```python
<your complete code here, must define solve(jobs) function>
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

Fix the code so it works correctly. The function must return a valid schedule where:
- Each operation has a "start" time assigned
- Operations within each job maintain their order (start[i+1] >= start[i] + duration[i])
- No two operations on the same machine overlap
- Every operation from the input must appear in the output

Your code will be tested on multiple instances of different sizes.

Output your fixed code in this exact format:

APPROACH: <one-line description of your approach>

```python
<your fixed code here>
```

NOTES: <what you fixed>
"""


PROBLEM_DESCRIPTION = """\
# Job Shop Scheduling Problem (JSP)

**Objective:** Minimize makespan (total completion time of all jobs). Lower is better.

There are **N jobs** and **M machines**. Each job consists of a sequence of operations,
each assigned to a specific machine with a given processing duration. The operation
order within each job is fixed and must be respected.

**Constraints:**
- Each machine processes only one operation at a time (no overlap)
- Operations in a job must follow their given order (precedence)
- No preemption — once an operation starts, it runs to completion
- All jobs start at time 0 or later

## Function Signature

`def solve(jobs: list[list[dict]]) -> list[list[dict]]`

**Input:** `jobs` — a list of N jobs. Each job is a list of M operations (in order).
Each operation dict has:
- `"machine"` (int): which machine this operation runs on (0 to M-1)
- `"duration"` (int): how long this operation takes

**Output:** a list of N jobs, same structure but each operation dict also includes:
- `"start"` (int): the time this operation begins

The makespan is the maximum (start + duration) across all operations."""


COORDINATOR_PROBLEM_DESCRIPTION = "a Job Shop Scheduling Problem (minimize makespan)"
