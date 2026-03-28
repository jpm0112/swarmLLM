from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any

import httpx
import pytest
from pydantic_ai import Agent, ModelResponse, ToolCallPart
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.test import TestModel

from swarmllm.config import Config, InstanceProfile, LLMEndpoint
from swarmllm.core.agent import run_agent
from swarmllm.core.coordinator import get_next_directions
from swarmllm.core.orchestrator import _run_agents_parallel, run_swarm
from swarmllm.llm.factory import build_coordinator_agent, build_worker_agent, clear_caches
from swarmllm.llm.routing import EndpointRouter
from swarmllm.llm.health import validate_backend_or_raise
from swarmllm.llm.schemas import CoordinatorRoundPlan, DirectionAssignment, WorkerDraft
from swarmllm.problems.scheduling import generate_instance


def _make_valid_worker_agent() -> Agent:
    def function_model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "approach": "FIFO sanity check",
                        "code": "def schedule(jobs):\n    return [job['id'] for job in jobs]",
                        "notes": "valid test worker draft",
                    },
                    tool_call_id="worker-valid",
                )
            ]
        )

    return Agent(FunctionModel(function_model), output_type=WorkerDraft)


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


def test_coordinator_agent_retries_after_malformed_text_output():
    clear_caches()
    config = Config()
    endpoint = LLMEndpoint(base_url="http://coord-retry/v1", api_key="test")
    agent = build_coordinator_agent(config, endpoint, "coord retry system prompt")
    calls = {"count": 0}

    def function_model(messages: list[Any], info: Any) -> ModelResponse:
        del messages, info
        calls["count"] += 1
        if calls["count"] == 1:
            return ModelResponse(
                parts=[
                    TextPart(
                        content=textwrap.dedent(
                            """\
                            <tools>
                            {"name": "final_result", "arguments": {"analysis": "bad", "directions": []}}
                            </tools>
                            """
                        ).strip()
                    )
                ]
            )

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={
                        "analysis": "Recovered after retry.",
                        "directions": [
                            {"agent_id": 0, "mode": "explore", "direction": "Try EDD variants"}
                        ],
                    },
                    tool_call_id="coord-retry",
                )
            ]
        )

    try:
        with agent.override(model=FunctionModel(function_model)):
            result = agent.run_sync("plan the next round")
    finally:
        clear_caches()

    assert result.output.analysis == "Recovered after retry."
    assert len(result.output.directions) == 1
    assert calls["count"] == 2


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
async def test_run_agent_reports_partial_benchmark_failures(monkeypatch):
    config = Config()
    config.swarm.agent_retries = 0
    endpoint = LLMEndpoint(base_url="http://worker/v1", api_key="test")
    problems = [
        ("small", generate_instance(num_jobs=4, seed=4, min_pt=1, max_pt=3)),
        ("medium", generate_instance(num_jobs=5, seed=5, min_pt=1, max_pt=4)),
    ]

    monkeypatch.setattr("swarmllm.core.agent.build_worker_agent", lambda *args, **kwargs: _make_valid_worker_agent())

    async def fake_execute(code, job_data, config, telemetry=None, process_label=None, process_metadata=None):
        del code, config, telemetry, process_label
        schedule = [job["id"] for job in job_data]
        if process_metadata and process_metadata.get("stage") == "precheck":
            return {"success": True, "schedule": schedule, "error": None, "stdout": ""}
        if process_metadata and process_metadata.get("instance") == "small":
            return {"success": True, "schedule": schedule, "error": None, "stdout": ""}
        return {
            "success": False,
            "schedule": None,
            "error": "RuntimeError: medium benchmark failed",
            "stdout": "",
        }

    monkeypatch.setattr("swarmllm.core.agent.execute_agent_code_async", fake_execute)

    result = await run_agent(
        agent_id=2,
        direction="Try FIFO and report failures cleanly",
        problems=problems,
        config=config,
        endpoint=endpoint,
        iteration=1,
        prompt_logger=None,
    )

    assert result["success"] is False
    assert result["score"] is None
    assert result["error"] == "RuntimeError: medium benchmark failed"
    assert result["failure_reason"] == "passed 1/2 instances"
    assert list(result["instance_scores"]) == ["small"]
    assert result["instance_errors"] == {"medium": "RuntimeError: medium benchmark failed"}


@pytest.mark.anyio
async def test_run_agent_allows_parallel_sandbox_execution(monkeypatch):
    config = Config()
    config.swarm.agent_retries = 0
    endpoint = LLMEndpoint(base_url="http://worker/v1", api_key="test")
    problems = [("tiny", generate_instance(num_jobs=4, seed=6, min_pt=1, max_pt=3))]
    entered_agents: list[int] = []
    both_entered = asyncio.Event()
    release = asyncio.Event()

    monkeypatch.setattr("swarmllm.core.agent.build_worker_agent", lambda *args, **kwargs: _make_valid_worker_agent())

    async def fake_execute(code, job_data, config, telemetry=None, process_label=None, process_metadata=None):
        del code, config, telemetry, process_label
        schedule = [job["id"] for job in job_data]
        if process_metadata and process_metadata.get("stage") == "precheck":
            entered_agents.append(process_metadata["agent_id"])
            if len(set(entered_agents)) == 2:
                both_entered.set()
            await release.wait()
        return {"success": True, "schedule": schedule, "error": None, "stdout": ""}

    monkeypatch.setattr("swarmllm.core.agent.execute_agent_code_async", fake_execute)

    task_a = asyncio.create_task(
        run_agent(
            agent_id=0,
            direction="Agent 0 test direction",
            problems=problems,
            config=config,
            endpoint=endpoint,
            iteration=1,
            prompt_logger=None,
        )
    )
    task_b = asyncio.create_task(
        run_agent(
            agent_id=1,
            direction="Agent 1 test direction",
            problems=problems,
            config=config,
            endpoint=endpoint,
            iteration=1,
            prompt_logger=None,
        )
    )

    await asyncio.wait_for(both_entered.wait(), timeout=1.0)
    assert set(entered_agents) == {0, 1}

    release.set()
    results = await asyncio.gather(task_a, task_b)

    assert all(result["success"] for result in results)


@pytest.mark.anyio
async def test_run_agents_parallel_respects_agent_semaphore_during_sandbox_execution(monkeypatch):
    config = Config()
    config.swarm.max_concurrent_agents = 1
    config.swarm.agent_retries = 0
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://worker/v1", api_key="test", label="worker-a")]
    router = EndpointRouter(config.llm)
    problems = [("tiny", generate_instance(num_jobs=4, seed=7, min_pt=1, max_pt=3))]
    assignments = [
        DirectionAssignment(agent_id=0, mode="explore", direction="First direction"),
        DirectionAssignment(agent_id=1, mode="explore", direction="Second direction"),
    ]
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release = asyncio.Event()
    first_agent_id: dict[str, int | None] = {"value": None}

    monkeypatch.setattr("swarmllm.core.agent.build_worker_agent", lambda *args, **kwargs: _make_valid_worker_agent())

    async def fake_execute(code, job_data, config, telemetry=None, process_label=None, process_metadata=None):
        del code, config, telemetry, process_label
        schedule = [job["id"] for job in job_data]
        if process_metadata and process_metadata.get("stage") == "precheck":
            agent_id = process_metadata["agent_id"]
            if first_agent_id["value"] is None:
                first_agent_id["value"] = agent_id
                first_entered.set()
                await release.wait()
            else:
                second_entered.set()
        return {"success": True, "schedule": schedule, "error": None, "stdout": ""}

    monkeypatch.setattr("swarmllm.core.agent.execute_agent_code_async", fake_execute)

    task = asyncio.create_task(
        _run_agents_parallel(
            assignments=assignments,
            problems=problems,
            config=config,
            router=router,
            iteration=1,
            prompt_logger=None,
            telemetry=None,
        )
    )

    await asyncio.wait_for(first_entered.wait(), timeout=1.0)
    await asyncio.sleep(0.05)
    assert second_entered.is_set() is False

    release.set()
    results = await asyncio.wait_for(task, timeout=1.0)

    assert second_entered.is_set() is True
    assert all(result["success"] for result in results)


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
        last_iteration_content="prior results",
        config=config,
        endpoint=endpoint,
        prompt_logger=None,
        best_solution=None,
    )

    assert analysis == "Need more diversity."
    assert len(directions) == 3
    assert directions[0].direction == "Try EDD with stochastic tie-breaks"
    assert directions[0].mode == "explore"
    assert directions[1].direction != ""
    assert usage is not None


@pytest.mark.anyio
async def test_run_swarm_smoke_uses_resolved_loopback_endpoint(monkeypatch, tmp_path):
    config = Config()
    config.llm.backend_kind = "ollama"
    config.llm.coordinator_endpoints = [LLMEndpoint(base_url="http://127.0.0.1:11434/v1", api_key="ollama")]
    config.llm.worker_endpoints = [LLMEndpoint(base_url="http://127.0.0.1:11434/v1", api_key="ollama")]
    config.swarm.num_agents = 1
    config.swarm.num_iterations = 1
    config.swarm.max_concurrent_agents = 1
    config.swarm.agent_retries = 0
    config.problem.instances = [
        InstanceProfile(
            name="tiny",
            num_jobs=4,
            min_processing_time=1,
            max_processing_time=3,
            due_date_tightness=0.6,
        )
    ]

    seen = {"models": 0, "coordinator": 0, "worker": 0, "chat_hosts": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            raise httpx.ConnectError("connection refused", request=request)

        if request.url.path == "/v1/models":
            seen["models"] += 1
            return httpx.Response(
                200,
                json={"data": [{"id": config.llm.coordinator_model}, {"id": config.llm.agent_model}]},
            )

        if request.url.path == "/v1/chat/completions":
            seen["chat_hosts"].append(request.url.host)
            body = json.loads(request.content)
            system_prompt = body["messages"][0]["content"]
            if "coordinator of a swarm" in system_prompt:
                seen["coordinator"] += 1
                payload = {
                    "analysis": "Start with a simple FIFO sanity check.",
                    "directions": [
                        {"agent_id": 0, "mode": "explore", "direction": "Try FIFO as a sanity check"}
                    ],
                }
            else:
                seen["worker"] += 1
                payload = {
                    "approach": "FIFO sanity check",
                    "code": "def schedule(jobs):\n    return [job['id'] for job in jobs]",
                    "notes": "valid smoke-test worker draft",
                }

            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1710000000,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "final_result",
                                            "arguments": json.dumps(payload),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            )

        raise AssertionError(f"Unexpected request path: {request.url.path}")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, trust_env=False)

    async def validate_with_mock(llm_config):
        await validate_backend_or_raise(llm_config, transport=transport)

    clear_caches()
    monkeypatch.setattr("swarmllm.core.orchestrator.validate_backend_or_raise", validate_with_mock)
    monkeypatch.setattr("swarmllm.llm.factory._get_http_client", lambda *args, **kwargs: client)

    try:
        with override_allow_model_requests(True):
            await run_swarm(config, output_dir=str(tmp_path), dashboard_mode="plain")
    finally:
        clear_caches()
        await client.aclose()

    assert config.llm.coordinator_endpoints[0].base_url == "http://localhost:11434/v1"
    assert config.llm.worker_endpoints[0].base_url == "http://localhost:11434/v1"
    assert seen["models"] == 2
    assert seen["coordinator"] == 1
    assert seen["worker"] == 1
    assert seen["chat_hosts"] == ["localhost", "localhost"]
    assert (tmp_path / "results_log.md").exists()
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "live_state.json").exists()
