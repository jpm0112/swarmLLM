from __future__ import annotations

"""
Job Scheduling Problem

N jobs, each with a processing time and a due date.
Find an ordering of jobs that minimizes total tardiness.

Tardiness of job j = max(0, completion_time_j - due_date_j)
Total tardiness = sum of all job tardinesses.

Lower is better. Zero is perfect (all jobs on time).
"""

import json
import os
import random
from dataclasses import dataclass


@dataclass
class Job:
    id: int
    processing_time: int
    due_date: int


@dataclass
class ProblemInstance:
    jobs: list[Job]
    total_processing_time: int
    optimal_lower_bound: float  # rough lower bound for reference

    def to_description(self) -> str:
        """Structure-only problem description for LLM agents (no instance data)."""
        return """# Job Scheduling Problem

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


def generate_instance(
    num_jobs: int = 20,
    seed: int = 42,
    min_pt: int = 1,
    max_pt: int = 20,
    due_date_tightness: float = 0.6,
) -> ProblemInstance:
    """Generate a random job scheduling problem instance."""
    rng = random.Random(seed)

    jobs = []
    total_pt = 0
    for i in range(num_jobs):
        pt = rng.randint(min_pt, max_pt)
        total_pt += pt
        jobs.append(Job(id=i, processing_time=pt, due_date=0))

    # Set due dates relative to total processing time
    # tightness controls how tight deadlines are (lower = tighter)
    for job in jobs:
        latest = int(total_pt * due_date_tightness)
        job.due_date = rng.randint(job.processing_time, max(job.processing_time, latest))

    return ProblemInstance(
        jobs=jobs,
        total_processing_time=total_pt,
        optimal_lower_bound=0.0,  # we don't know the true optimum
    )


def evaluate_schedule(instance: ProblemInstance, schedule: list[int]) -> dict:
    """
    Evaluate a schedule (list of job IDs) against the problem instance.

    Returns a dict with:
        - total_tardiness: the objective value (lower is better)
        - max_tardiness: worst single job tardiness
        - num_tardy: number of late jobs
        - details: per-job breakdown
        - valid: whether the schedule is a valid permutation
        - error: error message if invalid
    """
    job_map = {job.id: job for job in instance.jobs}
    all_ids = set(job_map.keys())

    # Validate schedule
    if not isinstance(schedule, list):
        return {"valid": False, "error": "Schedule must be a list of job IDs", "total_tardiness": float("inf")}

    if set(schedule) != all_ids or len(schedule) != len(all_ids):
        missing = all_ids - set(schedule)
        extra = set(schedule) - all_ids
        return {
            "valid": False,
            "error": f"Invalid permutation. Missing: {missing}, Extra: {extra}",
            "total_tardiness": float("inf"),
        }

    # Compute tardiness
    current_time = 0
    total_tardiness = 0
    max_tardiness = 0
    num_tardy = 0
    details = []

    for job_id in schedule:
        job = job_map[job_id]
        current_time += job.processing_time
        tardiness = max(0, current_time - job.due_date)
        total_tardiness += tardiness
        max_tardiness = max(max_tardiness, tardiness)
        if tardiness > 0:
            num_tardy += 1
        details.append({
            "job_id": job_id,
            "processing_time": job.processing_time,
            "due_date": job.due_date,
            "completion_time": current_time,
            "tardiness": tardiness,
        })

    return {
        "valid": True,
        "total_tardiness": total_tardiness,
        "max_tardiness": max_tardiness,
        "num_tardy": num_tardy,
        "num_jobs": len(schedule),
        "details": details,
    }


# --- Baseline schedules for reference ---

def baseline_fifo(instance: ProblemInstance) -> list[int]:
    """First-in-first-out: process in order of job ID."""
    return [job.id for job in instance.jobs]


def baseline_edd(instance: ProblemInstance) -> list[int]:
    """Earliest Due Date first."""
    return [job.id for job in sorted(instance.jobs, key=lambda j: j.due_date)]


def baseline_spt(instance: ProblemInstance) -> list[int]:
    """Shortest Processing Time first."""
    return [job.id for job in sorted(instance.jobs, key=lambda j: j.processing_time)]


def save_instance(instance: ProblemInstance, path: str, profile_name: str = "", profile_params: dict | None = None):
    """Save a problem instance to a JSON file."""
    data = {
        "profile_name": profile_name,
        "num_jobs": len(instance.jobs),
        "total_processing_time": instance.total_processing_time,
        "optimal_lower_bound": instance.optimal_lower_bound,
        "jobs": [
            {"id": j.id, "processing_time": j.processing_time, "due_date": j.due_date}
            for j in instance.jobs
        ],
    }
    if profile_params:
        data["profile_params"] = profile_params
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_instance(path: str) -> ProblemInstance:
    """Load a problem instance from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    jobs = [Job(id=j["id"], processing_time=j["processing_time"], due_date=j["due_date"]) for j in data["jobs"]]
    return ProblemInstance(
        jobs=jobs,
        total_processing_time=data["total_processing_time"],
        optimal_lower_bound=data.get("optimal_lower_bound", 0.0),
    )
