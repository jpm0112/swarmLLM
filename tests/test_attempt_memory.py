"""Tests for the AttemptMemory tracker."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from toon import decode as toon_decode

from swarmllm.tracking.attempt_memory import AttemptMemory, MEMORY_ROOT


@pytest.fixture
def memory(tmp_path):
    """Create an AttemptMemory that writes into a temp directory."""
    with patch("swarmllm.tracking.attempt_memory.MEMORY_ROOT", tmp_path):
        yield AttemptMemory("test_run")


@pytest.fixture
def sample_result():
    return {
        "approach": "greedy with penalty",
        "code": "def schedule(jobs):\n    return sorted(range(len(jobs)))",
        "score": 1500.0,
        "success": True,
        "error": None,
        "failure_reason": None,
        "notes": "Aggregate=1500 | small=500 | medium=1000",
        "instance_scores": {"small": 500, "medium": 1000},
        "instance_errors": {},
    }


@pytest.fixture
def failed_result():
    return {
        "approach": "random shuffle",
        "code": "def schedule(jobs):\n    raise ValueError('bad')",
        "score": None,
        "success": False,
        "error": "ValueError: bad",
        "failure_reason": "runtime error",
        "notes": "",
        "instance_scores": {},
        "instance_errors": {"small": "ValueError: bad"},
    }


class FakeAssignment:
    def __init__(self, agent_id, mode, direction, source_refs=None):
        self.agent_id = agent_id
        self.mode = mode
        self.direction = direction
        self.source_refs = source_refs or []


class FakeSourceRef:
    def __init__(self, agent_id, iteration):
        self.agent_id = agent_id
        self.iteration = iteration


# --- Coder records ---


def test_record_and_flush_coders(memory, sample_result, tmp_path):
    memory.record_attempt(1, 0, "explore", "try greedy", [], sample_result)
    memory.record_attempt(1, 1, "explore", "try random", [], sample_result)
    memory.flush_iteration_coders(1)

    path = tmp_path / "test_run" / "iteration1_coders.toon"
    assert path.exists()
    records = toon_decode(path.read_text(encoding="utf-8"))
    assert len(records) == 2
    assert records[0]["agent_id"] == 0
    assert records[1]["agent_id"] == 1


def test_coder_record_fields(memory, sample_result, tmp_path):
    memory.record_attempt(1, 0, "explore", "try greedy", [], sample_result)
    memory.flush_iteration_coders(1)

    records = toon_decode((tmp_path / "test_run" / "iteration1_coders.toon").read_text(encoding="utf-8"))
    r = records[0]
    expected_fields = {
        "record_type", "iteration", "agent_id", "mode", "direction",
        "source_refs", "approach", "code", "score", "success",
        "error", "failure_reason", "notes", "instance_scores", "instance_errors",
    }
    assert set(r.keys()) == expected_fields
    assert r["record_type"] == "agent_attempt"
    assert r["score"] == 1500.0
    assert r["success"] is True
    assert r["instance_scores"] == {"small": 500, "medium": 1000}


def test_failed_result_serialization(memory, failed_result, tmp_path):
    memory.record_attempt(1, 0, "explore", "try random", [], failed_result)
    memory.flush_iteration_coders(1)

    records = toon_decode((tmp_path / "test_run" / "iteration1_coders.toon").read_text(encoding="utf-8"))
    r = records[0]
    assert r["score"] is None
    assert r["success"] is False
    assert r["error"] == "ValueError: bad"
    assert r["instance_errors"] == {"small": "ValueError: bad"}


def test_flush_empty_iteration(memory, tmp_path):
    memory.flush_iteration_coders(99)
    path = tmp_path / "test_run" / "iteration99_coders.toon"
    assert not path.exists()


def test_read_coders(memory, sample_result):
    memory.record_attempt(1, 0, "explore", "try greedy", [], sample_result)
    memory.flush_iteration_coders(1)

    records = memory.read_coders(1)
    assert len(records) == 1
    assert records[0]["agent_id"] == 0


def test_read_coders_nonexistent(memory):
    assert memory.read_coders(99) == []


# --- Coordinator records ---


def test_coordinator_decision(memory, tmp_path):
    assignments = [
        FakeAssignment(0, "explore", "try greedy"),
        FakeAssignment(1, "exploit", "refine agent 0", [FakeSourceRef(0, 1)]),
    ]
    memory.record_coordinator_decision(1, assignments)

    path = tmp_path / "test_run" / "iteration1_coordinator.toon"
    assert path.exists()
    record = toon_decode(path.read_text(encoding="utf-8"))
    assert record["record_type"] == "coordinator_decision"
    assert record["iteration"] == 1
    assert record["analysis"] is None
    assert len(record["assignments"]) == 2
    assert record["assignments"][1]["source_refs"] == [{"agent_id": 0, "iteration": 1}]


def test_coordinator_with_analysis(memory, tmp_path):
    assignments = [FakeAssignment(0, "explore", "new direction")]
    memory.record_coordinator_decision(2, assignments, analysis="Agent 0 did well, try variations")

    record = toon_decode((tmp_path / "test_run" / "iteration2_coordinator.toon").read_text(encoding="utf-8"))
    assert record["analysis"] == "Agent 0 did well, try variations"


def test_read_coordinator(memory):
    assignments = [FakeAssignment(0, "explore", "try something")]
    memory.record_coordinator_decision(1, assignments)

    record = memory.read_coordinator(1)
    assert record is not None
    assert record["iteration"] == 1


def test_read_coordinator_nonexistent(memory):
    assert memory.read_coordinator(99) is None


# --- Folder creation ---


def test_run_folder_created(tmp_path):
    with patch("swarmllm.tracking.attempt_memory.MEMORY_ROOT", tmp_path):
        AttemptMemory("my_run_2026")
    assert (tmp_path / "my_run_2026").is_dir()


# --- Backward compatibility with legacy JSON ---


def test_read_coders_json_fallback(memory, tmp_path):
    """Old .json files should still be readable."""
    legacy = [{"record_type": "agent_attempt", "agent_id": 0, "score": 100}]
    path = tmp_path / "test_run" / "iteration5_coders.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    records = memory.read_coders(5)
    assert len(records) == 1
    assert records[0]["agent_id"] == 0


def test_read_coordinator_json_fallback(memory, tmp_path):
    """Old .json coordinator files should still be readable."""
    legacy = {"record_type": "coordinator_decision", "iteration": 5, "analysis": "test"}
    path = tmp_path / "test_run" / "iteration5_coordinator.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    record = memory.read_coordinator(5)
    assert record is not None
    assert record["analysis"] == "test"


# --- Round-trip with complex data ---


def test_roundtrip_multiline_code(memory):
    """Multi-line Python code and tracebacks should survive TOON round-trip."""
    result = {
        "approach": "GA with crossover",
        "code": "def schedule(jobs):\n    for j in jobs:\n        if j['due'] < 10:\n            pass\n    return [j['id'] for j in jobs]",
        "score": 42.0,
        "success": True,
        "error": None,
        "failure_reason": None,
        "notes": "multi-line test",
        "instance_scores": {"inst_a": 20, "inst_b": 22},
        "instance_errors": {},
    }
    memory.record_attempt(1, 0, "explore", "test direction", [], result)
    memory.flush_iteration_coders(1)

    records = memory.read_coders(1)
    assert len(records) == 1
    assert records[0]["code"] == result["code"]
    assert records[0]["instance_scores"] == {"inst_a": 20, "inst_b": 22}


def test_roundtrip_multiline_error(memory):
    """Multi-line traceback errors should survive TOON round-trip."""
    result = {
        "approach": "broken",
        "code": "def schedule(jobs): raise ValueError('bad')",
        "score": None,
        "success": False,
        "error": 'Traceback (most recent call last):\n  File "test.py", line 1\n    raise ValueError("bad")\nValueError: bad',
        "failure_reason": "runtime error",
        "notes": "",
        "instance_scores": {},
        "instance_errors": {"small": 'File "test.py", line 1\nValueError: bad'},
    }
    memory.record_attempt(1, 0, "explore", "test", [], result)
    memory.flush_iteration_coders(1)

    records = memory.read_coders(1)
    assert records[0]["error"] == result["error"]
    assert records[0]["instance_errors"]["small"] == result["instance_errors"]["small"]
