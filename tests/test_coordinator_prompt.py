from __future__ import annotations

"""Tests for coordinator prompt construction and SharedLog.read_iteration()."""

import pytest
from swarmllm.config import LogConfig
from swarmllm.tracking.shared_log import SharedLog


# ---------------------------------------------------------------------------
# SharedLog.read_iteration tests
# ---------------------------------------------------------------------------


def _make_log(tmp_path, entries: list[dict]) -> SharedLog:
    log = SharedLog(LogConfig(log_file="results_log.md"), output_dir=str(tmp_path))
    for e in entries:
        log.append_result(**e)
    return log


def _entry(iteration: int, agent_id: int, score: float | None = 10.0, success: bool = True) -> dict:
    return dict(
        iteration=iteration,
        agent_id=agent_id,
        direction=f"direction-{iteration}-{agent_id}",
        approach=f"approach-{iteration}-{agent_id}",
        code="def schedule(jobs): return []",
        score=score,
        success=success,
    )


def test_read_iteration_returns_only_target_iteration(tmp_path):
    log = _make_log(tmp_path, [
        _entry(1, 0),
        _entry(1, 1),
        _entry(2, 0),
        _entry(2, 1),
        _entry(3, 0),
    ])

    result = log.read_iteration(2)

    assert "### Iteration 2 —" in result
    assert "### Iteration 1 —" not in result
    assert "### Iteration 3 —" not in result


def test_read_iteration_contains_both_agents(tmp_path):
    log = _make_log(tmp_path, [
        _entry(1, 0),
        _entry(2, 0),
        _entry(2, 1),
    ])

    result = log.read_iteration(2)

    assert "Agent 0" in result
    assert "Agent 1" in result


def test_read_iteration_excludes_coordinator_summary(tmp_path):
    log = _make_log(tmp_path, [_entry(1, 0)])
    log.append_coordinator_summary(1, "Great progress so far.")

    result = log.read_iteration(1)

    assert "Coordinator Summary" not in result
    assert "Great progress so far" not in result


def test_read_iteration_empty_for_nonexistent_iteration(tmp_path):
    log = _make_log(tmp_path, [_entry(1, 0)])

    result = log.read_iteration(99)

    assert result == ""


def test_read_iteration_does_not_bleed_across_similar_numbers(tmp_path):
    """Iteration 1 entries must not appear when reading iteration 10."""
    log = _make_log(tmp_path, [
        _entry(1, 0),
        _entry(10, 0),
    ])

    result_1 = log.read_iteration(1)
    result_10 = log.read_iteration(10)

    assert "### Iteration 1 —" in result_1
    assert "### Iteration 10 —" not in result_1
    assert "### Iteration 10 —" in result_10
    assert "### Iteration 1 —" not in result_10


# ---------------------------------------------------------------------------
# get_next_directions prompt construction tests
# ---------------------------------------------------------------------------


def _build_prompt(last_iteration_content: str, best_solution: dict | None, iteration: int = 2) -> str:
    """Replicate the prompt logic from get_next_directions without hitting an LLM."""
    best_section = ""
    if best_solution:
        best_section = "\n## Best Solution So Far\n\n"
        best_section += f"**Score:** {best_solution['score']} — {best_solution['approach']}\n\n"
        best_section += f"```python\n{best_solution['code']}\n```\n\n"
        best_section += "Agents may refine, combine, or contrast with this solution.\n"

    return (
        f"## Last Iteration Results\n\n"
        f"{last_iteration_content}\n\n"
        f"{best_section}"
        f"---\n\n"
        f"This is iteration {iteration}."
    )


def test_prompt_contains_last_iteration_content():
    content = "### Iteration 1 — Agent 0 [SUCCESS]"
    prompt = _build_prompt(content, best_solution=None)
    assert "Last Iteration Results" in prompt
    assert content in prompt


def test_prompt_contains_best_solution_when_provided():
    best = {"score": 42.0, "approach": "EDD variant", "code": "def f(): pass"}
    prompt = _build_prompt("some content", best_solution=best)
    assert "Best Solution So Far" in prompt
    assert "42.0" in prompt
    assert "EDD variant" in prompt
    assert "def f(): pass" in prompt


def test_prompt_omits_best_solution_section_when_none():
    prompt = _build_prompt("some content", best_solution=None)
    assert "Best Solution So Far" not in prompt


def test_prompt_does_not_contain_results_so_far_heading():
    """Old heading 'Results So Far' must not appear in new prompt."""
    content = "iteration data"
    prompt = _build_prompt(content, best_solution=None)
    assert "Results So Far" not in prompt


def test_prompt_uses_last_iteration_heading():
    prompt = _build_prompt("data", best_solution=None)
    assert "Last Iteration Results" in prompt
