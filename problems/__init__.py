from __future__ import annotations

"""
Problem Base Classes

Abstract interface that all optimization problems must implement.
Each problem lives in its own sub-package (e.g. problems/job_scheduling/).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstanceProfile:
    """Generic problem instance profile.

    Each problem type defines what 'params' means.
    For job scheduling: num_jobs, min_processing_time, max_processing_time, due_date_tightness.
    """
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class ProblemInstance:
    """Generic container for a problem instance.

    Holds the raw data in a problem-specific format plus metadata.
    """
    data: Any              # problem-specific data (e.g. list of Job dicts)
    metadata: dict = field(default_factory=dict)  # e.g. total_processing_time, optimal_lower_bound

    def get(self, key: str, default: Any = None) -> Any:
        """Get a metadata value."""
        return self.metadata.get(key, default)


class ProblemBase(ABC):
    """Abstract base class for optimization problems."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Problem name (e.g. 'job_scheduling')."""
        ...

    @property
    @abstractmethod
    def objective(self) -> str:
        """Objective description (e.g. 'minimize total tardiness (lower is better)')."""
        ...

    @abstractmethod
    def generate_instance(self, profile: InstanceProfile, seed: int) -> ProblemInstance:
        """Generate a random problem instance from a profile."""
        ...

    @abstractmethod
    def evaluate(self, instance: ProblemInstance, solution: Any) -> dict:
        """Evaluate a solution.

        Returns dict with at least:
            - 'valid': bool
            - 'score': float (generic objective value, lower is better)
            - 'error': str or None
        """
        ...

    @abstractmethod
    def get_baselines(self, instance: ProblemInstance) -> dict[str, float]:
        """Compute baseline scores for an instance.

        Returns dict mapping baseline name -> score.
        """
        ...

    @abstractmethod
    def get_agent_system_prompt(self, timeout: int, pip_packages: str) -> str:
        """Full system prompt for worker agents."""
        ...

    @abstractmethod
    def get_agent_user_prompt(self, instance: ProblemInstance, instances_desc: str) -> str:
        """Problem description / user prompt for agents (structure only, no instance data)."""
        ...

    @abstractmethod
    def get_fix_prompt(self, error: str, code: str) -> str:
        """Prompt for fixing broken agent code."""
        ...

    @abstractmethod
    def get_coordinator_problem_description(self) -> str:
        """Short problem description for coordinator system prompt."""
        ...

    @abstractmethod
    def get_function_name(self) -> str:
        """Name of the function agents must define (e.g. 'schedule')."""
        ...

    @abstractmethod
    def prepare_input(self, instance: ProblemInstance) -> list[dict]:
        """Convert a problem instance to the dict format passed to agent code."""
        ...

    @abstractmethod
    def extract_solution(self, exec_result: dict) -> Any:
        """Extract the solution from sandbox execution result."""
        ...

    @abstractmethod
    def save_instance(self, instance: ProblemInstance, path: str,
                      profile_name: str = "", profile_params: dict | None = None) -> None:
        """Save a problem instance to a JSON file."""
        ...

    @abstractmethod
    def load_instance(self, path: str) -> ProblemInstance:
        """Load a problem instance from a JSON file."""
        ...

    @abstractmethod
    def get_instance_profiles(self, sizes_str: str, seed: int) -> list[InstanceProfile]:
        """Parse a sizes string (e.g. '20,50,100') into instance profiles."""
        ...

    @abstractmethod
    def get_default_profiles(self) -> list[InstanceProfile]:
        """Return default instance profiles for this problem."""
        ...

    def format_instance_info(self, instance: ProblemInstance, profile: InstanceProfile) -> str:
        """Format instance info for display. Override for problem-specific formatting."""
        return f"{profile.name}: {profile.params}"


def load_problem(problem_type: str) -> ProblemBase:
    """Load a problem class by type name."""
    if problem_type == "job_scheduling":
        from problems.job_scheduling import JobSchedulingProblem
        return JobSchedulingProblem()
    elif problem_type == "job_shop_scheduling":
        from problems.job_shop_scheduling import JobShopSchedulingProblem
        return JobShopSchedulingProblem()
    else:
        raise ValueError(f"Unknown problem type: {problem_type}. Available: job_scheduling, job_shop_scheduling")
