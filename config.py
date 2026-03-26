from __future__ import annotations

"""
SwarmLLM Configuration

All tunable parameters in one place. Edit this file to change
the behavior of the swarm without touching any other code.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LLMConfig:
    """Configuration for Ollama LLM connections."""
    base_urls: list[str] = field(default_factory=lambda: ["http://localhost:11434"])
    coordinator_model: str = "qwen2.5-coder:14b"
    agent_model: str = "qwen2.5-coder:14b"
    temperature_worker: float = 0.7
    temperature_coordinator: float = 0.4
    max_tokens_worker: int = 4096
    max_tokens_coordinator: int = 8192
    request_timeout: int = 300  # seconds


@dataclass
class SwarmConfig:
    """Configuration for the swarm."""
    num_agents: int = 20  # should be a multiple of max_concurrent_agents
    num_iterations: int = 5
    explore_ratio: float = 0.5  # fraction of agents that explore (vs exploit)
    max_concurrent_agents: int = 5  # overridden by setup_run.py based on model + GPUs
    agent_retries: int = 1  # retries per agent if code fails pre-test on smallest instance


@dataclass
class SandboxConfig:
    """Configuration for code execution sandbox."""
    timeout: int = 120  # seconds per execution
    max_memory_mb: int = 512
    # Block dangerous modules, allow everything else (stdlib + installed packages)
    blocked_imports: list[str] = field(default_factory=lambda: [
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests",
        "ftplib", "smtplib", "imaplib", "poplib",
        "webbrowser", "antigravity",
        "ctypes", "multiprocessing", "threading",
        "signal", "pickle", "shelve", "marshal",
        "code", "codeop", "compile", "compileall",
        "importlib", "runpy", "ensurepip", "pip",
        "venv", "distutils", "setuptools",
    ])
    # Extra pip packages to install and make available to agents
    pip_packages: list[str] = field(default_factory=lambda: [
        "numpy", "scipy", "networkx",
    ])


@dataclass
class InstanceProfile:
    """Configuration for a single problem instance."""
    name: str                       # human-readable label (e.g. "small_tight")
    num_jobs: int = 20
    min_processing_time: int = 1
    max_processing_time: int = 20
    due_date_tightness: float = 0.6  # lower = tighter deadlines


# Default profiles: vary size, deadline pressure, and processing time distribution
DEFAULT_INSTANCES = [
    InstanceProfile(
        name="small_tight",
        num_jobs=20,
        min_processing_time=1,
        max_processing_time=15,
        due_date_tightness=0.4,     # tight deadlines — forces smart ordering
    ),
    InstanceProfile(
        name="medium_mixed",
        num_jobs=50,
        min_processing_time=1,
        max_processing_time=20,
        due_date_tightness=0.6,     # moderate deadlines — balanced challenge
    ),
    InstanceProfile(
        name="large_loose",
        num_jobs=100,
        min_processing_time=5,
        max_processing_time=30,
        due_date_tightness=0.8,     # loose deadlines but long jobs — tests scalability
    ),
]


@dataclass
class ProblemConfig:
    """Configuration for the job scheduling problem."""
    instances: list[InstanceProfile] = field(default_factory=lambda: list(DEFAULT_INSTANCES))
    seed: int = 1048596  # fixed seed for reproducibility


@dataclass
class LogConfig:
    """Configuration for the shared results log."""
    log_file: str = "results_log.md"
    format: Literal["flat_markdown"] = "flat_markdown"


@dataclass
class Config:
    """Top-level configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    log: LogConfig = field(default_factory=LogConfig)
