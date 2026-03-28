import asyncio
import threading

import pytest

from swarmllm.config import SandboxConfig
from swarmllm.sandbox.executor import execute_agent_code, execute_agent_code_async


def _simple_jobs() -> list[dict]:
    return [
        {"id": 0, "processing_time": 3, "due_date": 10},
        {"id": 1, "processing_time": 1, "due_date": 5},
        {"id": 2, "processing_time": 2, "due_date": 7},
    ]


def _sorting_code() -> str:
    return """
def schedule(jobs):
    return [job["id"] for job in sorted(jobs, key=lambda job: (job["due_date"], job["id"]))]
"""


def test_execute_agent_code_runs_simple_schedule():
    result = execute_agent_code(_sorting_code(), _simple_jobs(), SandboxConfig(timeout=5))

    assert result["success"] is True
    assert result["result"] == [1, 2, 0]
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


@pytest.mark.anyio
async def test_execute_agent_code_async_matches_sync_result():
    config = SandboxConfig(timeout=5)

    sync_result = execute_agent_code(_sorting_code(), _simple_jobs(), config)
    async_result = await execute_agent_code_async(_sorting_code(), _simple_jobs(), config)

    assert async_result == sync_result


@pytest.mark.anyio
async def test_execute_agent_code_async_blocks_dangerous_imports():
    code = """
import os

def schedule(jobs):
    return [job["id"] for job in jobs]
"""
    jobs = [{"id": 0, "processing_time": 1, "due_date": 1}]

    result = await execute_agent_code_async(code, jobs, SandboxConfig(timeout=5))

    assert result["success"] is False
    assert "ImportError" in result["error"] or "blocked" in result["error"]


@pytest.mark.anyio
async def test_execute_agent_code_async_preserves_timeout_error():
    code = """
import time

def schedule(jobs):
    time.sleep(0.2)
    return [job["id"] for job in jobs]
"""

    result = await execute_agent_code_async(code, _simple_jobs(), SandboxConfig(timeout=0.05))

    assert result["success"] is False
    assert result["error"] == "Execution timed out after 0.05 seconds"


@pytest.mark.anyio
async def test_execute_agent_code_async_can_overlap(monkeypatch):
    active_calls = 0
    max_active_calls = 0
    lock = threading.Lock()
    both_entered = threading.Event()
    release = threading.Event()

    def blocking_execute(
        code: str,
        job_data: list[dict],
        config: SandboxConfig,
        function_name: str = "schedule",
        telemetry=None,
        process_label=None,
        process_metadata=None,
    ) -> dict:
        del code, config, telemetry, process_label, process_metadata
        nonlocal active_calls, max_active_calls
        with lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            if active_calls == 2:
                both_entered.set()
        try:
            if not release.wait(timeout=1.0):
                raise AssertionError("sandbox release was not triggered")
            return {
                "success": True,
                "schedule": [job["id"] for job in job_data],
                "error": None,
                "stdout": "",
            }
        finally:
            with lock:
                active_calls -= 1

    monkeypatch.setattr("swarmllm.sandbox.executor.execute_agent_code", blocking_execute)

    task_a = asyncio.create_task(execute_agent_code_async("def schedule(jobs): return []", _simple_jobs(), SandboxConfig()))
    task_b = asyncio.create_task(execute_agent_code_async("def schedule(jobs): return []", _simple_jobs(), SandboxConfig()))

    assert await asyncio.to_thread(both_entered.wait, 1.0)
    assert max_active_calls == 2

    release.set()
    results = await asyncio.gather(task_a, task_b)

    assert [result["success"] for result in results] == [True, True]
