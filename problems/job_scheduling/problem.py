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
from typing import Any

from problems import ProblemBase, ProblemInstance, InstanceProfile
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    FIX_PROMPT,
    PROBLEM_DESCRIPTION,
    COORDINATOR_PROBLEM_DESCRIPTION,
)


@dataclass
class Job:
    id: int
    processing_time: int
    due_date: int


class JobSchedulingProblem(ProblemBase):
    """Job scheduling: minimize total tardiness."""

    @property
    def name(self) -> str:
        return "job_scheduling"

    @property
    def objective(self) -> str:
        return "minimize total tardiness (lower is better)"

    def generate_instance(self, profile: InstanceProfile, seed: int) -> ProblemInstance:
        """Generate a random job scheduling problem instance."""
        params = profile.params
        num_jobs = params.get("num_jobs", 20)
        min_pt = params.get("min_processing_time", 1)
        max_pt = params.get("max_processing_time", 20)
        due_date_tightness = params.get("due_date_tightness", 0.6)

        rng = random.Random(seed)

        jobs = []
        total_pt = 0
        for i in range(num_jobs):
            pt = rng.randint(min_pt, max_pt)
            total_pt += pt
            jobs.append(Job(id=i, processing_time=pt, due_date=0))

        # Set due dates relative to total processing time
        for job in jobs:
            latest = int(total_pt * due_date_tightness)
            job.due_date = rng.randint(job.processing_time, max(job.processing_time, latest))

        return ProblemInstance(
            data=jobs,
            metadata={
                "total_processing_time": total_pt,
                "optimal_lower_bound": 0.0,
            },
        )

    def evaluate(self, instance: ProblemInstance, solution: Any) -> dict:
        """Evaluate a schedule (list of job IDs) against the problem instance.

        Returns dict with: valid, score, error, plus extra details.
        """
        jobs = instance.data
        job_map = {job.id: job for job in jobs}
        all_ids = set(job_map.keys())

        schedule = solution

        # Validate schedule
        if not isinstance(schedule, list):
            return {"valid": False, "score": float("inf"), "error": "Schedule must be a list of job IDs"}

        if set(schedule) != all_ids or len(schedule) != len(all_ids):
            missing = all_ids - set(schedule)
            extra = set(schedule) - all_ids
            return {
                "valid": False,
                "score": float("inf"),
                "error": f"Invalid permutation. Missing: {missing}, Extra: {extra}",
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
            "score": total_tardiness,
            "error": None,
            "max_tardiness": max_tardiness,
            "num_tardy": num_tardy,
            "num_jobs": len(schedule),
            "details": details,
        }

    def get_baselines(self, instance: ProblemInstance) -> dict[str, float]:
        """Compute baseline scores: FIFO, EDD, SPT."""
        jobs = instance.data
        results = {}
        for label, key_fn in [
            ("FIFO", lambda j: j.id),
            ("EDD", lambda j: j.due_date),
            ("SPT", lambda j: j.processing_time),
        ]:
            ordered = [j.id for j in sorted(jobs, key=key_fn)]
            ev = self.evaluate(instance, ordered)
            results[label] = ev["score"]
        return results

    def get_agent_system_prompt(self, timeout: int, pip_packages: str) -> str:
        return AGENT_SYSTEM_PROMPT.format(timeout=timeout, pip_packages=pip_packages)

    def get_agent_user_prompt(self, instance: ProblemInstance, instances_desc: str) -> str:
        return PROBLEM_DESCRIPTION

    def get_fix_prompt(self, error: str, code: str) -> str:
        return FIX_PROMPT.format(error=error[:1000], code=code)

    def get_coordinator_problem_description(self) -> str:
        return COORDINATOR_PROBLEM_DESCRIPTION

    def get_function_name(self) -> str:
        return "schedule"

    def prepare_input(self, instance: ProblemInstance) -> list[dict]:
        """Convert jobs to list of dicts for agent code."""
        return [
            {"id": j.id, "processing_time": j.processing_time, "due_date": j.due_date}
            for j in instance.data
        ]

    def extract_solution(self, exec_result: dict) -> Any:
        """Extract schedule from sandbox execution result."""
        return exec_result.get("result")

    def save_instance(self, instance: ProblemInstance, path: str,
                      profile_name: str = "", profile_params: dict | None = None) -> None:
        """Save a problem instance to a JSON file."""
        jobs = instance.data
        data = {
            "profile_name": profile_name,
            "num_jobs": len(jobs),
            "total_processing_time": instance.metadata.get("total_processing_time", 0),
            "optimal_lower_bound": instance.metadata.get("optimal_lower_bound", 0.0),
            "jobs": [
                {"id": j.id, "processing_time": j.processing_time, "due_date": j.due_date}
                for j in jobs
            ],
        }
        if profile_params:
            data["profile_params"] = profile_params
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_instance(self, path: str) -> ProblemInstance:
        """Load a problem instance from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        jobs = [Job(id=j["id"], processing_time=j["processing_time"], due_date=j["due_date"])
                for j in data["jobs"]]
        return ProblemInstance(
            data=jobs,
            metadata={
                "total_processing_time": data["total_processing_time"],
                "optimal_lower_bound": data.get("optimal_lower_bound", 0.0),
            },
        )

    def get_instance_profiles(self, sizes_str: str, seed: int) -> list[InstanceProfile]:
        """Parse '20,50,100' into InstanceProfiles with varied parameters."""
        sizes = [int(s.strip()) for s in sizes_str.split(",")]
        tightness_values = [0.4, 0.6, 0.8, 0.5, 0.7]
        pt_ranges = [(1, 15), (1, 20), (5, 30), (1, 25), (3, 20)]
        profiles = []
        for i, size in enumerate(sizes):
            t = tightness_values[i % len(tightness_values)]
            min_pt, max_pt = pt_ranges[i % len(pt_ranges)]
            profiles.append(InstanceProfile(
                name=f"{size}jobs_t{t}",
                params={
                    "num_jobs": size,
                    "min_processing_time": min_pt,
                    "max_processing_time": max_pt,
                    "due_date_tightness": t,
                },
            ))
        return profiles

    def get_default_profiles(self) -> list[InstanceProfile]:
        """Return the 3 default job scheduling profiles."""
        return [
            InstanceProfile(
                name="small_tight",
                params={
                    "num_jobs": 20,
                    "min_processing_time": 1,
                    "max_processing_time": 15,
                    "due_date_tightness": 0.4,
                },
            ),
            InstanceProfile(
                name="medium_mixed",
                params={
                    "num_jobs": 50,
                    "min_processing_time": 1,
                    "max_processing_time": 20,
                    "due_date_tightness": 0.6,
                },
            ),
            InstanceProfile(
                name="large_loose",
                params={
                    "num_jobs": 100,
                    "min_processing_time": 5,
                    "max_processing_time": 30,
                    "due_date_tightness": 0.8,
                },
            ),
        ]

    def format_instance_info(self, instance: ProblemInstance, profile: InstanceProfile) -> str:
        """Format job scheduling instance info for display."""
        p = profile.params
        total_pt = instance.metadata.get("total_processing_time", 0)
        return (f"{p.get('num_jobs', '?')} jobs | "
                f"tightness={p.get('due_date_tightness', '?')} | "
                f"processing time={p.get('min_processing_time', '?')}-{p.get('max_processing_time', '?')} | "
                f"total PT={total_pt}")
