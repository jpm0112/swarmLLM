from __future__ import annotations

"""
Job Shop Scheduling Problem (JSP)

N jobs, M machines. Each job has a fixed sequence of operations (one per machine).
Each operation has a machine assignment and a processing duration.
Find start times for all operations that minimize the makespan (total completion time).

Constraints:
- Operations within a job must follow their given order
- Each machine processes one operation at a time
- No preemption

This uses the classic JSPLIB benchmark instances.
"""

import json
import os
import random
from typing import Any

from swarmllm.problems import ProblemBase, ProblemInstance, InstanceProfile
from .prompts import (
    AGENT_SYSTEM_PROMPT,
    FIX_PROMPT,
    PROBLEM_DESCRIPTION,
    COORDINATOR_PROBLEM_DESCRIPTION,
)

# Directory containing JSPLIB instance files
INSTANCES_DIR = os.path.join(os.path.dirname(__file__), "instances")
INSTANCES_JSON = os.path.join(os.path.dirname(__file__), "instances.json")

# Curated instance selections by difficulty
# Each tuple: (instance_name, num_jobs, num_machines, known_optimum_or_upper_bound)
DIFFICULTY_PRESETS = {
    "easy": [
        ("ft06", 6, 6, 55),
        ("la01", 10, 5, 666),
        ("la02", 10, 5, 655),
    ],
    "medium": [
        ("ft10", 10, 10, 930),
        ("la16", 10, 10, 945),
        ("orb01", 10, 10, 1059),
    ],
    "hard": [
        ("ft20", 20, 5, 1165),
        ("abz7", 20, 15, 656),
        ("la21", 15, 10, 1046),
    ],
    "very_hard": [
        ("swv01", 20, 10, 1407),
        ("ta01", 15, 15, 1231),
        ("yn1", 20, 20, 885),
    ],
}


def parse_jsplib_file(filepath: str) -> tuple[int, int, list[list[dict]]]:
    """Parse a JSPLIB instance file.

    Returns (num_jobs, num_machines, jobs) where jobs is a list of lists of
    operation dicts with keys 'machine' and 'duration'.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # First line: num_jobs num_machines
    parts = lines[0].split()
    num_jobs = int(parts[0])
    num_machines = int(parts[1])

    jobs = []
    for i in range(1, num_jobs + 1):
        values = list(map(int, lines[i].split()))
        operations = []
        for j in range(0, len(values), 2):
            operations.append({
                "machine": values[j],
                "duration": values[j + 1],
            })
        jobs.append(operations)

    return num_jobs, num_machines, jobs


def load_instances_metadata() -> dict:
    """Load the instances.json metadata file."""
    if os.path.exists(INSTANCES_JSON):
        with open(INSTANCES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {entry["name"]: entry for entry in data}
    return {}


class JobShopSchedulingProblem(ProblemBase):
    """Job Shop Scheduling: minimize makespan using JSPLIB benchmarks."""

    def __init__(self):
        self._metadata = load_instances_metadata()

    @property
    def name(self) -> str:
        return "job_shop_scheduling"

    @property
    def objective(self) -> str:
        return "minimize makespan (lower is better)"

    def generate_instance(self, profile: InstanceProfile, seed: int) -> ProblemInstance:
        """Load a JSPLIB instance by name (not randomly generated)."""
        params = profile.params
        instance_name = params.get("instance_name", "ft06")

        filepath = os.path.join(INSTANCES_DIR, instance_name)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSPLIB instance not found: {filepath}")

        num_jobs, num_machines, jobs = parse_jsplib_file(filepath)

        # Get known optimum or bounds from metadata
        meta = self._metadata.get(instance_name, {})
        optimum = meta.get("optimum")
        bounds = meta.get("bounds", {}) or {}
        best_known = optimum or bounds.get("upper")

        return ProblemInstance(
            data=jobs,
            metadata={
                "instance_name": instance_name,
                "num_jobs": num_jobs,
                "num_machines": num_machines,
                "optimum": optimum,
                "upper_bound": bounds.get("upper"),
                "lower_bound": bounds.get("lower") or optimum,
                "best_known": best_known,
            },
        )

    def evaluate(self, instance: ProblemInstance, solution: Any) -> dict:
        """Evaluate a JSP solution (list of jobs with start times assigned).

        Validates constraints and computes makespan.
        """
        jobs_input = instance.data
        num_jobs = instance.metadata["num_jobs"]
        num_machines = instance.metadata["num_machines"]

        if not isinstance(solution, list):
            return {"valid": False, "score": float("inf"),
                    "error": "Solution must be a list of jobs"}

        if len(solution) != num_jobs:
            return {"valid": False, "score": float("inf"),
                    "error": f"Expected {num_jobs} jobs, got {len(solution)}"}

        makespan = 0
        # Track machine usage: list of (start, end) intervals per machine
        machine_intervals = {m: [] for m in range(num_machines)}

        for job_idx, job_sol in enumerate(solution):
            if not isinstance(job_sol, list):
                return {"valid": False, "score": float("inf"),
                        "error": f"Job {job_idx} must be a list of operations"}

            job_input = jobs_input[job_idx]
            if len(job_sol) != len(job_input):
                return {"valid": False, "score": float("inf"),
                        "error": f"Job {job_idx}: expected {len(job_input)} ops, got {len(job_sol)}"}

            prev_end = 0
            for op_idx, op in enumerate(job_sol):
                if not isinstance(op, dict):
                    return {"valid": False, "score": float("inf"),
                            "error": f"Job {job_idx} op {op_idx}: must be a dict"}

                if "start" not in op:
                    return {"valid": False, "score": float("inf"),
                            "error": f"Job {job_idx} op {op_idx}: missing 'start' key"}

                expected_machine = job_input[op_idx]["machine"]
                expected_duration = job_input[op_idx]["duration"]

                machine = op.get("machine", expected_machine)
                duration = op.get("duration", expected_duration)
                start = op["start"]

                if not isinstance(start, (int, float)) or start < 0:
                    return {"valid": False, "score": float("inf"),
                            "error": f"Job {job_idx} op {op_idx}: invalid start time {start}"}

                start = int(start)

                # Check machine assignment matches
                if machine != expected_machine:
                    return {"valid": False, "score": float("inf"),
                            "error": f"Job {job_idx} op {op_idx}: wrong machine {machine}, expected {expected_machine}"}

                # Check duration matches
                if duration != expected_duration:
                    return {"valid": False, "score": float("inf"),
                            "error": f"Job {job_idx} op {op_idx}: wrong duration {duration}, expected {expected_duration}"}

                # Check precedence: operation must start after previous op finishes
                if start < prev_end:
                    return {"valid": False, "score": float("inf"),
                            "error": f"Job {job_idx} op {op_idx}: starts at {start} but previous op ends at {prev_end} (precedence violation)"}

                end = start + duration
                prev_end = end

                # Record this interval on the machine
                machine_intervals[machine].append((start, end, job_idx, op_idx))

                makespan = max(makespan, end)

        # Check machine constraints: no overlapping operations on any machine
        for machine, intervals in machine_intervals.items():
            intervals.sort()  # sort by start time
            for i in range(len(intervals) - 1):
                _, end1, j1, o1 = intervals[i]
                start2, _, j2, o2 = intervals[i + 1]
                if start2 < end1:
                    return {"valid": False, "score": float("inf"),
                            "error": f"Machine {machine}: overlap between job {j1} op {o1} (ends {end1}) and job {j2} op {o2} (starts {start2})"}

        best_known = instance.metadata.get("best_known")
        gap = None
        if best_known:
            gap = (makespan - best_known) / best_known * 100

        return {
            "valid": True,
            "score": makespan,
            "error": None,
            "makespan": makespan,
            "best_known": best_known,
            "gap_percent": gap,
            "num_jobs": num_jobs,
            "num_machines": num_machines,
        }

    def get_baselines(self, instance: ProblemInstance) -> dict[str, float]:
        """Compute baseline schedules using simple dispatching rules."""
        jobs_input = instance.data
        num_jobs = instance.metadata["num_jobs"]
        num_machines = instance.metadata["num_machines"]

        results = {}
        for rule_name, rule_fn in [
            ("SPT", self._dispatch_spt),
            ("LPT", self._dispatch_lpt),
            ("FIFO", self._dispatch_fifo),
        ]:
            solution = rule_fn(jobs_input, num_jobs, num_machines)
            ev = self.evaluate(instance, solution)
            if ev["valid"]:
                results[rule_name] = ev["score"]
            else:
                results[rule_name] = float("inf")

        # Add known optimum as reference
        best_known = instance.metadata.get("best_known")
        if best_known:
            results["OPTIMAL"] = best_known

        return results

    def _dispatch_schedule(self, jobs_input, num_jobs, num_machines, priority_fn):
        """Generic dispatching rule scheduler.

        At each step, pick the ready operation with highest priority (lowest value).
        """
        # Track next operation index for each job
        next_op = [0] * num_jobs
        # Track when each job's last operation finishes
        job_available = [0] * num_jobs
        # Track when each machine becomes free
        machine_available = [0] * num_machines
        # Build output
        output = [[None] * len(jobs_input[j]) for j in range(num_jobs)]

        total_ops = sum(len(j) for j in jobs_input)
        scheduled = 0

        while scheduled < total_ops:
            # Find all ready operations (next unscheduled op for each job)
            candidates = []
            for j in range(num_jobs):
                if next_op[j] < len(jobs_input[j]):
                    op = jobs_input[j][next_op[j]]
                    machine = op["machine"]
                    earliest = max(job_available[j], machine_available[machine])
                    candidates.append((j, next_op[j], machine, op["duration"], earliest))

            if not candidates:
                break

            # Apply priority function to select next operation
            candidates.sort(key=lambda c: (c[4], priority_fn(c)))  # earliest start, then priority
            # Pick the one that can start earliest, break ties with priority
            best = min(candidates, key=lambda c: (c[4], priority_fn(c)))

            j, op_idx, machine, duration, start = best

            output[j][op_idx] = {
                "machine": machine,
                "duration": duration,
                "start": start,
            }

            machine_available[machine] = start + duration
            job_available[j] = start + duration
            next_op[j] = op_idx + 1
            scheduled += 1

        return output

    def _dispatch_spt(self, jobs_input, num_jobs, num_machines):
        """Shortest Processing Time first."""
        return self._dispatch_schedule(jobs_input, num_jobs, num_machines,
                                       lambda c: c[3])  # sort by duration

    def _dispatch_lpt(self, jobs_input, num_jobs, num_machines):
        """Longest Processing Time first."""
        return self._dispatch_schedule(jobs_input, num_jobs, num_machines,
                                       lambda c: -c[3])  # sort by -duration

    def _dispatch_fifo(self, jobs_input, num_jobs, num_machines):
        """First In First Out (by job index)."""
        return self._dispatch_schedule(jobs_input, num_jobs, num_machines,
                                       lambda c: c[0])  # sort by job index

    def get_agent_system_prompt(self, timeout: int, pip_packages: str) -> str:
        return AGENT_SYSTEM_PROMPT.format(timeout=timeout, pip_packages=pip_packages)

    def get_agent_user_prompt(self, instance: ProblemInstance, instances_desc: str) -> str:
        return PROBLEM_DESCRIPTION

    def get_fix_prompt(self, error: str, code: str) -> str:
        return FIX_PROMPT.format(error=error[:1000], code=code)

    def get_coordinator_problem_description(self) -> str:
        return COORDINATOR_PROBLEM_DESCRIPTION

    def get_function_name(self) -> str:
        return "solve"

    def prepare_input(self, instance: ProblemInstance) -> list[list[dict]]:
        """Convert jobs to the format passed to agent code."""
        return instance.data  # already in the right format

    def extract_solution(self, exec_result: dict) -> Any:
        """Extract schedule from sandbox execution result."""
        return exec_result.get("result")

    def save_instance(self, instance: ProblemInstance, path: str,
                      profile_name: str = "", profile_params: dict | None = None) -> None:
        """Save a problem instance to a JSON file."""
        data = {
            "profile_name": profile_name,
            "instance_name": instance.metadata.get("instance_name", ""),
            "num_jobs": instance.metadata["num_jobs"],
            "num_machines": instance.metadata["num_machines"],
            "optimum": instance.metadata.get("optimum"),
            "best_known": instance.metadata.get("best_known"),
            "jobs": instance.data,
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
        return ProblemInstance(
            data=data["jobs"],
            metadata={
                "instance_name": data.get("instance_name", ""),
                "num_jobs": data["num_jobs"],
                "num_machines": data["num_machines"],
                "optimum": data.get("optimum"),
                "best_known": data.get("best_known"),
                "upper_bound": data.get("best_known"),
                "lower_bound": data.get("optimum"),
            },
        )

    def get_instance_profiles(self, sizes_str: str, seed: int) -> list[InstanceProfile]:
        """Parse instance names or difficulty levels.

        Accepts:
        - Difficulty levels: 'easy', 'medium', 'hard', 'very_hard'
        - Specific instance names: 'ft06,ft10,abz7'
        - Mix: 'easy,abz7,ta01'
        """
        profiles = []
        for item in sizes_str.split(","):
            item = item.strip()
            if item in DIFFICULTY_PRESETS:
                # Pick one representative instance from the difficulty level
                instances = DIFFICULTY_PRESETS[item]
                rng = random.Random(seed)
                chosen = rng.choice(instances)
                profiles.append(InstanceProfile(
                    name=f"{item}_{chosen[0]}",
                    params={"instance_name": chosen[0], "difficulty": item},
                ))
            else:
                # Assume it's a specific instance name
                profiles.append(InstanceProfile(
                    name=item,
                    params={"instance_name": item},
                ))
        return profiles

    def get_default_profiles(self) -> list[InstanceProfile]:
        """Return default instances: one easy, one medium, one hard."""
        return [
            InstanceProfile(
                name="easy_ft06",
                params={"instance_name": "ft06", "difficulty": "easy"},
            ),
            InstanceProfile(
                name="medium_ft10",
                params={"instance_name": "ft10", "difficulty": "medium"},
            ),
            InstanceProfile(
                name="hard_abz7",
                params={"instance_name": "abz7", "difficulty": "hard"},
            ),
        ]

    def format_instance_info(self, instance: ProblemInstance, profile: InstanceProfile) -> str:
        """Format JSP instance info for display."""
        m = instance.metadata
        name = m.get("instance_name", "?")
        best = m.get("best_known")
        opt_str = f" | optimal={best}" if best else ""
        return (f"{name} ({m.get('num_jobs', '?')} jobs × {m.get('num_machines', '?')} machines"
                f"{opt_str})")
