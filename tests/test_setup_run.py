from __future__ import annotations

from scripts.setup_run import supported_backends_for_platform


def test_supported_backends_for_windows():
    assert supported_backends_for_platform("Windows", "AMD64") == ["ollama"]


def test_supported_backends_for_apple_silicon():
    assert supported_backends_for_platform("Darwin", "arm64") == ["ollama", "vllm-metal"]


def test_supported_backends_for_linux():
    assert supported_backends_for_platform("Linux", "x86_64") == ["ollama", "vllm"]
