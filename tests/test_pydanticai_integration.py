from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent, ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from swarmllm.config import Config, LLMEndpoint
from swarmllm.core.agent import run_agent
from swarmllm.core.coordinator import get_next_directions
from swarmllm.llm.factory import build_coordinator_agent, build_worker_agent
from swarmllm.llm.schemas import CoordinatorRoundPlan, WorkerDraft
from swarmllm.problems.scheduling import generate_instance


def test_worker_agent_supports_typed_output_with_test_model():
    config = Config()
    endpoint = LLMEndpoint(base_url="http://worker/v1", api_key="test")
    agent = build_worker_agent(config, endpoint, "worker system prompt")

    with agent.override(
        model=TestModel(
            custom_output_args={
                "approach": "EDD with tie-breaking",
                "code": "def schedule(jobs):\n    return [job['id'] for job in jobs]",
                "notes": "typed output works",
            }
        )
    ):
        result = agent.run_sync("write scheduling code")

    assert isinstance(result.output, WorkerDraft)
    assert result.output.approach == "EDD with tie-breaking"


def test_coordinator_agent_supports_typed_output_with_test_model():
    config = Config()
    endpoint = LLMEndpoint(base_url="http://coord/v1", api_key="test")
    agent = build_coordinator_agent(config, endpoint, "coord system prompt")

    with agent.override(
        model=TestModel(
            custom_output_args={
                "analysis": "EDD-style heuristics are promising.",
                "directions": [
                    {"agent_id": 0, "mode": "explore", "direction": "Try EDD variants"},
                    {"agent_id": 1, "mode": "exploit", "direction": "Refine local search on the best result"},
                ],
            }
        )
    ):
        result = agent.run_sync("plan the next round")

    assert isinstance(result.output, CoordinatorRoundPlan)
    assert len(result.output.directions) == 2


@pytest.mark.anyio
async def test_run_agent_retries_after_pretest_failure(monkeypatch):
    config = Config()
    config.swarm.agent_retries = 1
    endpoint = LLMEndpoint(base_url="http://worker/v1", api_key="test")
    problems = [
        ("small", generate_instance(num_jobs=4, seed=1, min_pt=1, max_pt=3)),
        ("medium", generate_instance(num_jobs=5, seed=2, min_pt=1, max_pt=4)),
    ]
    calls = {"count": 0}

    def function_model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        calls["count"] += 1
        if calls["count"] == 1:
            args = {
                "approach": "Broken draft",
                "code": "def not_schedule(jobs):\n    return []",
                "notes": "first pass is intentionally broken",
            }
        else:
            args = {
                "approach": "FIFO repair",
                "code": "def schedule(jobs):\n    return [job['id'] for job in jobs]",
                "notes": "fixed by restoring the schedule function",
            }
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args=args,
                    tool_call_id=f"worker-{calls['count']}",
                )
            ]
        )

    test_agent = Agent(FunctionModel(function_model), output_type=WorkerDraft)
    monkeypatch.setattr("swarmllm.core.agent.build_worker_agent", lambda *args, **kwargs: test_agent)

    result = await run_agent(
        agent_id=0,
        direction="Try FIFO as a sanity check",
        problems=problems,
        config=config,
        endpoint=endpoint,
        iteration=1,
        prompt_logger=None,
    )

    assert result["success"] is True
    assert result["score"] is not None
    assert "fixed after pre-test retry 1" in result["notes"]
    assert result["token_usage"] is not None
    assert result["token_usage"].total_tokens > 0


@pytest.mark.anyio
async def test_run_agent_fails_cleanly_for_malformed_code(monkeypatch):
    config = Config()
    config.swarm.agent_retries = 0
    endpoint = LLMEndpoint(base_url="http://worker/v1", api_key="test")
    problems = [("small", generate_instance(num_jobs=4, seed=3, min_pt=1, max_pt=3))]

    def function_model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "approach": "Syntax error draft",
                        "code": "def schedule(jobs)\n    return []",
                        "notes": "missing colon",
                    },
                    tool_call_id="worker-malformed",
                )
            ]
        )

    test_agent = Agent(FunctionModel(function_model), output_type=WorkerDraft)
    monkeypatch.setattr("swarmllm.core.agent.build_worker_agent", lambda *args, **kwargs: test_agent)

    result = await run_agent(
        agent_id=1,
        direction="Produce malformed code",
        problems=problems,
        config=config,
        endpoint=endpoint,
        iteration=1,
        prompt_logger=None,
    )

    assert result["success"] is False
    assert "SyntaxError" in result["error"]
    assert result["token_usage"] is not None


@pytest.mark.anyio
async def test_get_next_directions_fills_missing_assignments(monkeypatch):
    config = Config()
    config.swarm.num_agents = 3
    endpoint = LLMEndpoint(base_url="http://coord/v1", api_key="test")

    def function_model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "analysis": "Need more diversity.",
                        "directions": [
                            {"agent_id": 0, "mode": "explore", "direction": "Try EDD with stochastic tie-breaks"}
                        ],
                    },
                    tool_call_id="coord-partial",
                )
            ]
        )

    test_agent = Agent(FunctionModel(function_model), output_type=CoordinatorRoundPlan)
    monkeypatch.setattr("swarmllm.core.coordinator.build_coordinator_agent", lambda *args, **kwargs: test_agent)

    analysis, directions, usage = await get_next_directions(
        iteration=2,
        log_content="prior results",
        config=config,
        endpoint=endpoint,
        prompt_logger=None,
        top_solutions=[],
    )

    assert analysis == "Need more diversity."
    assert len(directions) == 3
    assert directions[0] == "Try EDD with stochastic tie-breaks"
    assert directions[1] != ""
    assert usage is not None
