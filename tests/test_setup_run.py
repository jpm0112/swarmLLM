from __future__ import annotations

import os

from scripts.setup_run import (
    build_local_vllm_server_spec,
    build_vllm_serve_command,
    choose_ollama_models,
    parse_flat_yaml,
    supported_backends_for_platform,
)


def test_supported_backends_for_windows():
    assert supported_backends_for_platform("Windows", "AMD64") == ["ollama"]


def test_supported_backends_for_apple_silicon():
    assert supported_backends_for_platform("Darwin", "arm64") == ["ollama", "vllm-metal", "mlx-lm"]


def test_supported_backends_for_linux():
    assert supported_backends_for_platform("Linux", "x86_64") == ["ollama", "vllm"]


def test_choose_ollama_models_from_available_list(monkeypatch):
    monkeypatch.setattr(
        "scripts.setup_run.fetch_available_models",
        lambda *args, **kwargs: ["llama3.2:3b", "qwen2.5-coder:14b"],
    )
    monkeypatch.setattr(
        "scripts.setup_run.fetch_ollama_model_sizes",
        lambda: {"qwen2.5-coder:14b": 9.0},
    )
    answers = iter(["2", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    coordinator, worker, sizes = choose_ollama_models(
        "http://127.0.0.1:11434/v1",
        "ollama",
        "llama3.2:3b",
        "llama3.2:3b",
    )

    assert coordinator == "qwen2.5-coder:14b"
    assert worker == "llama3.2:3b"
    assert sizes.get("qwen2.5-coder:14b") == 9.0


def test_parse_flat_yaml_and_build_vllm_command(tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: Qwen/Qwen2.5-Coder-14B-Instruct",
                "served-model-name: qwen2.5-coder-14b",
                'host: "127.0.0.1"',
                "port: 8000",
                'api-key: "token-abc123"',
                "dtype: auto",
                "max-model-len: 16384",
                "gpu-memory-utilization: 0.9",
                "max-num-seqs: 16",
                "enable-prefix-caching: true",
                "tensor-parallel-size: 1",
                "pipeline-parallel-size: 1",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_flat_yaml(str(config_path))
    command = build_vllm_serve_command("~/bin/vllm", parsed)

    assert command[:3] == [os.path.expanduser("~/bin/vllm"), "serve", "Qwen/Qwen2.5-Coder-14B-Instruct"]
    assert "--served-model-name" in command
    assert "--host" in command
    assert "--port" in command
    assert "--api-key" in command
    assert "--enable-prefix-caching" in command


def test_build_local_vllm_server_spec_uses_profile_alias_and_updates_env(monkeypatch, tmp_path):
    config_path = tmp_path / "serve.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model: Qwen/Qwen2.5-Coder-14B-Instruct",
                "served-model-name: placeholder",
                'host: "0.0.0.0"',
                "port: 9999",
                'api-key: "token-abc123"',
            ]
        ),
        encoding="utf-8",
    )

    answers = iter([str(config_path), "~/custom-vllm/bin/vllm", "secret-key"])
    monkeypatch.setattr("scripts.setup_run.ask", lambda prompt, default, explanation="": next(answers))

    env: dict[str, str] = {}
    spec = build_local_vllm_server_spec(
        "vllm-metal",
        "configs/backends/vllm-metal.local.example.toml",
        str(tmp_path),
        env,
    )

    assert spec.command[0] == os.path.expanduser("~/custom-vllm/bin/vllm")
    assert spec.base_url == "http://127.0.0.1:8000/v1"
    assert spec.command[spec.command.index("--served-model-name") + 1] == "qwen2.5-coder-14b"
    assert spec.command[spec.command.index("--host") + 1] == "127.0.0.1"
    assert spec.command[spec.command.index("--port") + 1] == "8000"
    assert spec.command[spec.command.index("--api-key") + 1] == "secret-key"
    assert env["SWARMLLM_VLLM_METAL_API_KEY"] == "secret-key"
