from swarmllm.config import SandboxConfig
from swarmllm.sandbox.executor import execute_agent_code


def test_execute_agent_code_runs_simple_schedule():
    code = """
def schedule(jobs):
    return [job["id"] for job in sorted(jobs, key=lambda job: (job["due_date"], job["id"]))]
"""
    jobs = [
        {"id": 0, "processing_time": 3, "due_date": 10},
        {"id": 1, "processing_time": 1, "due_date": 5},
        {"id": 2, "processing_time": 2, "due_date": 7},
    ]

    result = execute_agent_code(code, jobs, SandboxConfig(timeout=5))

    assert result["success"] is True
    assert result["schedule"] == [1, 2, 0]
    assert result["error"] is None


def test_execute_agent_code_blocks_dangerous_imports():
    code = """
import os

def schedule(jobs):
    return [job["id"] for job in jobs]
"""
    jobs = [{"id": 0, "processing_time": 1, "due_date": 1}]

    result = execute_agent_code(code, jobs, SandboxConfig(timeout=5))

    assert result["success"] is False
    assert "ImportError" in result["error"] or "blocked" in result["error"]
