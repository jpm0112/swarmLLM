from __future__ import annotations

import textwrap

import pytest

from scripts.run import build_config_from_args, build_parser
from swarmllm.config import Config
from swarmllm.llm.profiles import apply_backend_profile, load_backend_profile


def test_load_backend_profile_normalizes_urls(tmp_path):
    profile_path = tmp_path / "backend.toml"
    profile_path.write_text(
        textwrap.dedent(
            """
            name = "test-backend"
            kind = "vllm"
            request_timeout = 120
            default_max_concurrent_agents = 9

            [coordinator]
            model = "coord-model"
            temperature = 0.3
            max_tokens = 9000

            [worker]
            model = "worker-model"
            temperature = 0.8
            max_tokens = 4000

            [[coordinator_endpoints]]
            base_url = "http://localhost:8000"
            api_key_env = "TEST_KEY"

            [[worker_endpoints]]
            base_url = "http://localhost:8001/"
            api_key = "token"
            weight = 2
            """
        ),
        encoding="utf-8",
    )

    profile = load_backend_profile(profile_path)

    assert profile.kind == "vllm"
    assert profile.coordinator_endpoints[0].base_url == "http://localhost:8000/v1"
    assert profile.worker_endpoints[0].base_url == "http://localhost:8001/v1"


def test_apply_backend_profile_updates_runtime_config(tmp_path):
    profile_path = tmp_path / "backend.toml"
    profile_path.write_text(
        textwrap.dedent(
            """
            name = "ollama-local"
            kind = "ollama"
            request_timeout = 200
            default_max_concurrent_agents = 7

            [coordinator]
            model = "coord-model"
            temperature = 0.4
            max_tokens = 8100

            [worker]
            model = "worker-model"
            temperature = 0.6
            max_tokens = 4200

            [[coordinator_endpoints]]
            base_url = "http://127.0.0.1:11434/v1"
            api_key = "ollama"

            [[worker_endpoints]]
            base_url = "http://127.0.0.1:11434/v1"
            api_key = "ollama"
            weight = 1
            """
        ),
        encoding="utf-8",
    )

    config = Config()
    apply_backend_profile(config, profile_path)

    assert config.llm.backend_kind == "ollama"
    assert config.llm.backend_profile_name == "ollama-local"
    assert config.llm.coordinator_model == "coord-model"
    assert config.llm.agent_model == "worker-model"
    assert config.swarm.max_concurrent_agents == 7


def test_cli_overrides_win_over_backend_profile(tmp_path):
    profile_path = tmp_path / "backend.toml"
    profile_path.write_text(
        textwrap.dedent(
            """
            name = "vllm-local"
            kind = "vllm"
            request_timeout = 200
            default_max_concurrent_agents = 7

            [coordinator]
            model = "coord-model"
            temperature = 0.4
            max_tokens = 8100

            [worker]
            model = "worker-model"
            temperature = 0.6
            max_tokens = 4200

            [[coordinator_endpoints]]
            base_url = "http://127.0.0.1:8000/v1"

            [[worker_endpoints]]
            base_url = "http://127.0.0.1:8000/v1"
            weight = 1
            """
        ),
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "--backend-profile",
            str(profile_path),
            "--coordinator-model",
            "override-coord",
            "--agent-model",
            "override-worker",
            "--max-concurrent",
            "3",
        ]
    )

    config = build_config_from_args(args)

    assert config.llm.coordinator_model == "override-coord"
    assert config.llm.agent_model == "override-worker"
    assert config.swarm.max_concurrent_agents == 3


def test_invalid_backend_profile_requires_worker_endpoints(tmp_path):
    profile_path = tmp_path / "backend.toml"
    profile_path.write_text(
        textwrap.dedent(
            """
            name = "broken"
            kind = "vllm"

            [coordinator]
            model = "coord-model"

            [worker]
            model = "worker-model"

            [[coordinator_endpoints]]
            base_url = "http://127.0.0.1:8000/v1"
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_backend_profile(profile_path)
