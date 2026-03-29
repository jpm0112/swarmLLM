"""
Interactive setup script that chooses a backend profile and launches the swarm.

This script is backend-aware and can bootstrap local backends for common
single-node workflows. Runtime validation still happens inside `scripts/run.py`
via `/v1/models` checks before the swarm starts.
"""

from __future__ import annotations

import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarmllm.llm.profiles import load_backend_profile, loopback_base_url_candidates, resolve_api_key


DEFAULT_PROFILE_BY_BACKEND = {
    "ollama": os.path.join("configs", "backends", "ollama.local.example.toml"),
    "ollama-cloud": os.path.join("configs", "backends", "ollama.cloud.example.toml"),
    "vllm-metal": os.path.join("configs", "backends", "vllm-metal.local.example.toml"),
    "vllm": os.path.join("configs", "backends", "vllm.single-node.example.toml"),
    "mlx-lm": os.path.join("configs", "backends", "mlx-lm.local.example.toml"),
}

DEFAULT_VLLM_CONFIG_BY_BACKEND = {
    "vllm-metal": os.path.join("configs", "vllm", "serve.single-node.example.yaml"),
    "vllm": os.path.join("configs", "vllm", "serve.single-node.example.yaml"),
}

DEFAULT_VLLM_EXECUTABLE_BY_BACKEND = {
    "vllm-metal": os.path.join("~", ".venv-vllm-metal", "bin", "vllm"),
    "vllm": "vllm",
}

DEFAULT_MLX_LM_CONFIG = os.path.join("configs", "mlx-lm", "serve.example.yaml")
DEFAULT_MLX_LM_EXECUTABLE = "mlx_lm.server"


@dataclass
class LocalServerSpec:
    executable: str
    command: list[str]
    api_key_env: str | None
    api_key: str
    base_url: str
    log_path: str


@dataclass
class LocalServerState:
    process: subprocess.Popen[str] | None
    log_path: str | None
    started: bool


def supported_backends_for_platform(system_name: str, machine: str) -> list[str]:
    """Return the backend kinds supported by the current OS."""
    system_name = system_name.lower()
    machine = machine.lower()
    if system_name == "windows":
        return ["ollama", "ollama-cloud", "vllm"]
    if system_name == "darwin":
        backends = ["ollama", "ollama-cloud"]
        if machine in {"arm64", "aarch64"}:
            backends.append("vllm-metal")
            backends.append("mlx-lm")
        return backends
    return ["ollama", "ollama-cloud", "vllm"]


def ask(prompt: str, default: str, explanation: str = "") -> str:
    """Ask for a parameter with a default value and optional explanation."""
    if explanation:
        print(f"    {explanation}")
    value = input(f"  {prompt} [{default}]: ").strip()
    return value if value else default


def ask_yes_no(prompt: str, default: bool = True, explanation: str = "") -> bool:
    """Ask a yes/no question with a default."""
    if explanation:
        print(f"    {explanation}")
    default_label = "Y/n" if default else "y/N"
    while True:
        value = input(f"  {prompt} [{default_label}]: ").strip().lower()
        if value == "":
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("  Invalid choice, try again.")


def pick_option(prompt: str, options: list[str], default: str | None = None) -> str:
    """Pick a single option from a numbered list."""
    if not options:
        raise ValueError("pick_option requires at least one option")
    default_index = options.index(default) + 1 if default in options else 1
    print()
    for idx, option in enumerate(options, start=1):
        marker = " (default)" if idx == default_index else ""
        print(f"    {idx}) {option}{marker}")
    while True:
        choice = input(f"  {prompt} [{default_index}]: ").strip()
        if choice == "":
            return options[default_index - 1]
        try:
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def pick_backend(options: list[str]) -> str:
    print("\n  Available backends:")
    print("  ----------------------------------------")
    for idx, backend in enumerate(options, start=1):
        print(f"    {idx}) {backend}")
    print("  ----------------------------------------")
    default_idx = 1
    while True:
        choice = input(f"  Pick a backend [{default_idx}]: ").strip()
        if choice == "":
            return options[default_idx - 1]
        try:
            index = int(choice)
            if 1 <= index <= len(options):
                return options[index - 1]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def backend_notes(backend: str) -> str:
    if backend == "ollama":
        return "Expect a running Ollama server exposing its OpenAI-compatible API."
    if backend == "ollama-cloud":
        return (
            "Uses the Ollama cloud API. Requires an API key set via OLLAMA_API_KEY "
            "or entered during setup."
        )
    if backend == "vllm-metal":
        return (
            "Can auto-start an Apple Silicon vLLM Metal server, typically from "
            "~/.venv-vllm-metal/bin/vllm, using the example single-node settings."
        )
    if backend == "mlx-lm":
        return (
            "Can auto-start a local mlx-lm server (`mlx_lm.server`) on Apple Silicon. "
            "Useful for models that run better with the MLX framework."
        )
    return (
        "Can auto-start a local vLLM server for single-node use. Cluster or cloud "
        "profiles should still point at remote endpoints."
    )


def is_local_base_url(base_url: str) -> bool:
    """Return whether the base URL points at a loopback or wildcard local address."""
    hostname = urlsplit(base_url).hostname
    return hostname in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def parse_base_url(base_url: str) -> tuple[str, int]:
    """Extract host and port from an OpenAI-compatible base URL."""
    parts = urlsplit(base_url)
    if parts.hostname is None or parts.port is None:
        raise ValueError(f"Base URL must include host and port: {base_url}")
    return parts.hostname, parts.port


def parse_flat_yaml(path: str) -> dict[str, Any]:
    """Parse the repo's simple single-level YAML templates into a dict."""
    parsed: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = parse_yaml_scalar(value.strip())
    return parsed


def parse_yaml_scalar(value: str) -> Any:
    """Parse a scalar value from the simple YAML config templates."""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def fetch_available_models(base_url: str, api_key: str, timeout: float = 3.0) -> list[str]:
    """Fetch available model ids from an OpenAI-compatible `/v1/models` endpoint."""
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error: Exception | None = None
    for candidate in loopback_base_url_candidates(base_url):
        request = Request(f"{candidate}/models", headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            return sorted(item["id"] for item in payload.get("data", []) if "id" in item)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"Failed to query {base_url}/models: {last_error}") from last_error
    return []


def fetch_ollama_model_sizes() -> dict[str, float]:
    """Run ``ollama list`` and parse model sizes in GB.

    Returns a mapping from model name to size in GB.
    """
    sizes: dict[str, float] = {}
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return sizes
        for line in result.stdout.strip().split("\n")[1:]:  # skip header
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            for i, p in enumerate(parts):
                if p in ("GB", "MB"):
                    try:
                        size_gb = float(parts[i - 1])
                        if p == "MB":
                            size_gb /= 1024
                        sizes[name] = size_gb
                    except (ValueError, IndexError):
                        pass
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return sizes


def estimate_parallel(model_size_gb: float, vram_gb: float = 24) -> int:
    """Estimate max parallel requests based on model size and VRAM.

    Model VRAM usage is roughly 1.2x the file size (weights + overhead).
    Each parallel slot needs ~2-3GB for KV cache at 4K context.
    Conservative estimate to avoid OOM / Ollama timeouts.
    """
    model_vram = model_size_gb * 1.2  # actual VRAM when loaded
    free = vram_gb - model_vram - 2  # 2GB system overhead
    per_slot = 2.5  # ~2.5GB KV cache per parallel slot
    slots = max(1, int(free / per_slot))
    return min(slots, 8)  # cap at 8


def pick_model_with_sizes(
    models: list[str],
    sizes: dict[str, float],
    label: str,
    preferred: str = "",
) -> str:
    """Show numbered menu with sizes and let user pick a model."""
    default_idx = 1
    for i, m in enumerate(models, 1):
        if m == preferred:
            default_idx = i
            break

    print(f"\n  {label}:")
    print("  ----------------------------------------")
    for i, m in enumerate(models, 1):
        size_str = f" ({sizes[m]:.1f} GB)" if m in sizes else ""
        marker = " *" if i == default_idx else ""
        print(f"    {i}) {m}{size_str}{marker}")
    print("  ----------------------------------------")
    while True:
        choice = input(f"  Pick a number [{default_idx}]: ").strip()
        if choice == "":
            return models[default_idx - 1]
        try:
            idx = int(choice)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def choose_ollama_models(base_url: str, api_key: str, default_coordinator: str, default_worker: str) -> tuple[str, str, dict[str, float]]:
    """Fetch and choose coordinator/worker models from a running Ollama server.

    Returns (coordinator_model, worker_model, model_sizes_dict).
    """
    sizes = fetch_ollama_model_sizes()

    try:
        available_models = fetch_available_models(base_url, api_key)
    except RuntimeError as exc:
        print()
        print(f"  Warning: could not read Ollama models automatically: {exc}")
        coordinator = ask("Coordinator model", default_coordinator)
        worker = ask("Worker model", default_worker)
        return coordinator, worker, sizes

    if not available_models:
        print()
        print("  Warning: Ollama returned no models.")
        coordinator = ask("Coordinator model", default_coordinator)
        worker = ask("Worker model", default_worker)
        return coordinator, worker, sizes

    if sizes:
        print()
        print("  The COORDINATOR reads all results and assigns research")
        print("  directions. Bigger model = smarter strategy.")
        coordinator = pick_model_with_sizes(available_models, sizes, "Coordinator model", default_coordinator)
        print(f"  -> {coordinator}")

        print()
        print("  The AGENT model writes optimization code. Runs N times")
        print("  per iteration, so speed matters. Can be smaller/faster.")
        worker = pick_model_with_sizes(available_models, sizes, "Agent model", default_worker)
        print(f"  -> {worker}")
    else:
        print()
        print("  Available Ollama models:")
        coordinator = pick_option("Coordinator model", available_models, default=default_coordinator)
        worker = pick_option("Worker model", available_models, default=default_worker)

    return coordinator, worker, sizes


def pick_problem() -> str:
    """Let user pick a problem type."""
    available = ["job_scheduling", "job_shop_scheduling"]
    print("\n  Problem type:")
    print("  ----------------------------------------")
    for i, p in enumerate(available, 1):
        marker = " *" if i == 1 else ""
        print(f"    {i}) {p}{marker}")
    print("  ----------------------------------------")
    while True:
        choice = input("  Pick a number [1]: ").strip()
        if choice == "":
            return available[0]
        try:
            idx = int(choice)
            if 1 <= idx <= len(available):
                return available[idx - 1]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def prompt_backend_api_key(api_key_env: str | None, default_api_key: str, env: dict[str, str]) -> str:
    """Prompt for an API key and keep the backend env var in sync."""
    api_key = ask(
        "Backend API key",
        default_api_key,
        "This must match the API key passed to the local server and the backend profile env var.",
    )
    if api_key_env:
        env[api_key_env] = api_key
    return api_key


def build_vllm_serve_command(executable: str, config_values: dict[str, Any]) -> list[str]:
    """Build a `vllm serve ...` command from a flat template mapping."""
    if "model" not in config_values:
        raise ValueError("vLLM config must define `model`.")

    command = [os.path.expanduser(executable), "serve", str(config_values["model"])]
    flag_order = [
        "served-model-name",
        "host",
        "port",
        "api-key",
        "dtype",
        "max-model-len",
        "gpu-memory-utilization",
        "max-num-seqs",
        "tensor-parallel-size",
        "pipeline-parallel-size",
        "tool-call-parser",
    ]
    for key in flag_order:
        if key in config_values:
            command.extend([f"--{key}", str(config_values[key])])
    if config_values.get("enable-prefix-caching"):
        command.append("--enable-prefix-caching")
    if config_values.get("enable-auto-tool-choice"):
        command.append("--enable-auto-tool-choice")
    return command


def build_local_vllm_server_spec(
    backend: str,
    profile_path: str,
    output_dir: str,
    env: dict[str, str],
    model_override: str | None = None,
) -> LocalServerSpec:
    """Build the launch command and environment for a local vLLM-style backend."""
    profile = load_backend_profile(profile_path)
    endpoint = profile.coordinator_endpoints[0]
    base_url = endpoint.base_url
    host, port = parse_base_url(base_url)
    api_key_env = endpoint.api_key_env

    config_path = DEFAULT_VLLM_CONFIG_BY_BACKEND[backend]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"vLLM config not found: {config_path}")

    executable = DEFAULT_VLLM_EXECUTABLE_BY_BACKEND[backend]

    config_values = parse_flat_yaml(config_path)
    if model_override:
        config_values["model"] = model_override
        config_values["served-model-name"] = model_override
    else:
        config_values["served-model-name"] = profile.coordinator.model
    config_values["host"] = host
    config_values["port"] = port

    api_key = env.get(api_key_env, str(config_values.get("api-key", "token-abc123"))) if api_key_env else str(
        config_values.get("api-key", "token-abc123")
    )
    if api_key_env:
        env[api_key_env] = api_key
    config_values["api-key"] = api_key

    command = build_vllm_serve_command(executable, config_values)
    log_path = os.path.join(output_dir, f"{backend}_server.log")
    return LocalServerSpec(
        executable=os.path.expanduser(executable),
        command=command,
        api_key_env=api_key_env,
        api_key=api_key,
        base_url=base_url,
        log_path=log_path,
    )


def build_mlx_lm_serve_command(executable: str, config_values: dict[str, Any]) -> list[str]:
    """Build an `mlx_lm.server ...` command from a flat template mapping."""
    if "model" not in config_values:
        raise ValueError("mlx-lm config must define `model`.")
    command = [os.path.expanduser(executable), "--model", str(config_values["model"])]
    for key in ["host", "port", "api-key", "log-level", "max-tokens"]:
        if key in config_values:
            command.extend([f"--{key}", str(config_values[key])])
    return command


def build_local_mlx_lm_server_spec(
    profile_path: str,
    output_dir: str,
    env: dict[str, str],
) -> LocalServerSpec:
    """Build the launch command and environment for a local mlx-lm backend."""
    profile = load_backend_profile(profile_path)
    endpoint = profile.coordinator_endpoints[0]
    base_url = endpoint.base_url
    host, port = parse_base_url(base_url)
    api_key_env = endpoint.api_key_env

    config_path = ask(
        "mlx-lm config path",
        DEFAULT_MLX_LM_CONFIG,
        "Use a simple YAML template to seed the launch command.",
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"mlx-lm config not found: {config_path}")

    executable = ask(
        "mlx_lm.server executable",
        DEFAULT_MLX_LM_EXECUTABLE,
        "Path to the `mlx_lm.server` executable (or just `mlx_lm.server` if on PATH).",
    )

    config_values = parse_flat_yaml(config_path)
    config_values["host"] = host
    config_values["port"] = port

    api_key = resolve_api_key(endpoint, profile.kind)
    if api_key and api_key not in {"EMPTY", ""}:
        config_values["api-key"] = api_key
    if api_key_env:
        env[api_key_env] = api_key

    command = build_mlx_lm_serve_command(executable, config_values)
    log_path = os.path.join(output_dir, "mlx-lm_server.log")
    return LocalServerSpec(
        executable=os.path.expanduser(executable),
        command=command,
        api_key_env=api_key_env,
        api_key=api_key,
        base_url=base_url,
        log_path=log_path,
    )


def wait_for_backend_ready(
    base_url: str,
    api_key: str,
    timeout_seconds: float,
    process: subprocess.Popen[str] | None = None,
) -> bool:
    """Poll `/v1/models` until a backend is ready or the process exits."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            fetch_available_models(base_url, api_key, timeout=2.0)
            return True
        except RuntimeError:
            pass
        if process is not None and process.poll() is not None:
            return False
        time.sleep(1.0)
    return False


def tail_file(path: str, num_lines: int = 20) -> list[str]:
    """Return the last few lines from a text file."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-num_lines:]]


def stop_local_server(process: subprocess.Popen[str]) -> None:
    """Best-effort shutdown for a local model server process."""
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            # On Windows with WSL, also kill the vllm process inside WSL
            try:
                subprocess.run(
                    ["wsl", "-d", "Ubuntu", "--", "pkill", "-f", "vllm serve"],
                    timeout=5, capture_output=True,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            process.terminate()
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()


def ensure_local_vllm_server(
    backend: str,
    profile_path: str,
    output_dir: str,
    env: dict[str, str],
    model_override: str | None = None,
) -> LocalServerState:
    """Start a local vLLM-style backend when the profile points at loopback and nothing is listening."""
    profile = load_backend_profile(profile_path)
    endpoint = profile.coordinator_endpoints[0]
    base_url = endpoint.base_url

    if not is_local_base_url(base_url):
        print("  Remote backend profile detected; launcher will not auto-start a local server.")
        return LocalServerState(process=None, log_path=None, started=False)

    api_key = resolve_api_key(endpoint, profile.kind)
    if endpoint.api_key_env:
        env.setdefault(endpoint.api_key_env, api_key or "token-abc123")
        api_key = env[endpoint.api_key_env]

    try:
        models = fetch_available_models(base_url, api_key)
        print(f"  Reusing existing {backend} server at {base_url} ({len(models)} model(s) visible).")
        return LocalServerState(process=None, log_path=None, started=False)
    except RuntimeError:
        pass

    if backend == "mlx-lm":
        spec = build_local_mlx_lm_server_spec(profile_path, output_dir, env)
    else:
        spec = build_local_vllm_server_spec(backend, profile_path, output_dir, env,
                                            model_override=model_override)

    launch_command = spec.command
    # On Windows, vLLM must run inside WSL2 (requires CUDA/Linux).
    # Wrap the command so it activates the venv and launches inside WSL.
    if os.name == "nt" and backend in {"vllm"}:
        vllm_venv = "~/vllm-env"
        # Build the inner shell command that runs inside WSL
        inner_parts = [
            f"source {vllm_venv}/bin/activate",
            "CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID "
            + " ".join(spec.command),
        ]
        inner_cmd = " && ".join(inner_parts)
        launch_command = ["wsl", "-d", "Ubuntu", "--", "bash", "-lc", inner_cmd]

    log_handle = open(spec.log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        launch_command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=(os.name != "nt"),
    )
    log_handle.close()

    print(f"  Starting {backend} server (via WSL2)..." if os.name == "nt" else f"  Starting {backend} server...")
    print(f"    Command: {' '.join(launch_command)}")
    print(f"    Log:     {spec.log_path}")

    # vLLM on WSL needs extra time to load model weights and compile CUDA graphs
    startup_timeout = 300 if backend in {"vllm", "vllm-metal"} else 90
    if not wait_for_backend_ready(spec.base_url, spec.api_key, timeout_seconds=startup_timeout, process=process):
        print(f"ERROR: {backend} server did not become ready at {spec.base_url}")
        tail = tail_file(spec.log_path)
        if tail:
            print("  Last log lines:")
            for line in tail:
                print(f"    {line}")
        stop_local_server(process)
        raise RuntimeError(f"{backend} startup failed")

    print(f"  {backend} server is ready.")
    return LocalServerState(process=process, log_path=spec.log_path, started=True)


def main():
    print("=" * 60)
    print("  SwarmLLM — Backend-Aware Setup")
    print("=" * 60)

    system_name = platform.system()
    machine = platform.machine()
    supported_backends = supported_backends_for_platform(system_name, machine)

    print(f"\n  Platform detected: {system_name} ({machine})")
    backend = pick_backend(supported_backends)
    print(f"  -> {backend}")
    print(f"    {backend_notes(backend)}")

    profile_default = DEFAULT_PROFILE_BY_BACKEND[backend]
    profile_path = ask(
        "Backend profile path",
        profile_default,
        "Use a TOML backend profile that defines model aliases, endpoint pools, and request defaults.",
    )
    if not os.path.exists(profile_path):
        print(f"ERROR: Backend profile not found: {profile_path}")
        sys.exit(1)
    profile = load_backend_profile(profile_path)

    # Pick problem type
    problem_type = pick_problem()
    print(f"  -> {problem_type}")

    coordinator_model = profile.coordinator.model
    worker_model = profile.worker.model
    model_sizes: dict[str, float] = {}
    recommended_parallel = 4  # sensible default

    if backend in {"ollama", "ollama-cloud"}:
        if backend == "ollama-cloud" and not os.environ.get("OLLAMA_API_KEY"):
            print()
            api_key_input = input("  OLLAMA_API_KEY: ").strip()
            if api_key_input:
                os.environ["OLLAMA_API_KEY"] = api_key_input
        api_key = resolve_api_key(profile.coordinator_endpoints[0], profile.kind)
        coordinator_model, worker_model, model_sizes = choose_ollama_models(
            profile.coordinator_endpoints[0].base_url,
            api_key,
            coordinator_model,
            worker_model,
        )

        # GPU VRAM estimation for parallel slots
        if model_sizes and worker_model in model_sizes:
            agent_size = model_sizes[worker_model]
            GPU_0_VRAM = 24  # RTX 3090 — TODO: detect automatically
            parallel_gpu0 = estimate_parallel(agent_size, GPU_0_VRAM)
            recommended_parallel = parallel_gpu0
            print(f"\n  Agent model ~{agent_size:.1f} GB")
            print(f"    Estimated ~{parallel_gpu0} parallel slots (assuming {GPU_0_VRAM} GB VRAM)")

    elif backend in {"vllm", "vllm-metal", "mlx-lm"}:
        coordinator_model = ask(
            "Coordinator model",
            coordinator_model,
            "HuggingFace model ID for the coordinator (reads results, assigns directions).",
        )
        worker_model = ask(
            "Worker/agent model",
            worker_model,
            "HuggingFace model ID for agents (writes optimization code). Can be the same as coordinator.",
        )

    concurrent = ask(
        "Max concurrent agents",
        str(recommended_parallel),
        "How many agents run at the same time. Based on model size and GPU VRAM.",
    )
    concurrent_int = int(concurrent)

    agents = ask(
        "Number of agents",
        str(concurrent_int),
        f"How many worker agents per iteration.",
    )
    iterations = ask(
        "Number of iterations",
        "2",
        "How many coordinator/worker rounds to run.",
    )
    if problem_type == "job_shop_scheduling":
        instance_default = "easy,medium,hard"
        instance_help = ("JSPLIB instances to test on. Use difficulty levels (easy,medium,hard,very_hard) "
                         "or specific instance names (ft06,ft10,abz7). E.g. easy,medium,hard")
    else:
        instance_default = "20,50,100"
        instance_help = "Problem sizes to test on. Each gets different characteristics. E.g. 20,50,100"
    instance_sizes = ask(
        "Instance sizes (comma-separated)",
        instance_default,
        instance_help,
    )
    explore_ratio = ask(
        "Explore ratio (0.0-1.0)",
        "0.5",
        "Fraction of agents exploring new ideas versus exploiting current best results.",
    )
    timeout = ask(
        "Code execution timeout (seconds)",
        "120",
        "Max time each agent's generated code may run in the sandbox.",
    )
    retries = ask(
        "Agent retries",
        "1",
        "How many fix-up attempts a worker gets if the smallest-instance pre-test fails.",
    )
    seed = ask(
        "Random seed",
        "1048596",
        "Same seed means the same benchmark instances.",
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_coord = re.sub(r'[^a-zA-Z0-9._-]', '_', coordinator_model)
    safe_agent = re.sub(r'[^a-zA-Z0-9._-]', '_', worker_model)
    output_dir = os.path.join(
        "runs",
        f"{timestamp}_{problem_type}_coord-{safe_coord}_agent-{safe_agent}_{agents}agents_{iterations}iter",
    )
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  Configuration:")
    print(f"    Problem type:   {problem_type}")
    print(f"    Backend:        {backend}")
    print(f"    Profile:        {profile_path}")
    print(f"    Coord model:    {coordinator_model}")
    print(f"    Worker model:   {worker_model}")
    if model_sizes and worker_model in model_sizes:
        print(f"    Worker size:    {model_sizes[worker_model]:.1f} GB")
    print(f"    Agents:         {agents}")
    print(f"    Iterations:     {iterations}")
    print(f"    Concurrent:     {concurrent}")
    print(f"    Instances:      {instance_sizes}")
    print(f"    Explore ratio:  {explore_ratio}")
    print(f"    Timeout:        {timeout}s")
    print(f"    Seed:           {seed}")
    print(f"    Output folder:  {output_dir}")
    print("=" * 60)
    print()

    confirm = input("  Start the swarm? (Y/n): ").strip()
    if confirm.lower() == "n":
        print("  Cancelled.")
        return

    run_env = os.environ.copy()
    server_state = LocalServerState(process=None, log_path=None, started=False)
    if backend in {"vllm-metal", "vllm", "mlx-lm"}:
        try:
            server_state = ensure_local_vllm_server(
                backend, profile_path, output_dir, run_env,
                model_override=worker_model,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    for env_key in [
        "SWARMLLM_LOCAL_SERVER_PID",
        "SWARMLLM_LOCAL_SERVER_KIND",
        "SWARMLLM_LOCAL_SERVER_LOG_PATH",
    ]:
        run_env.pop(env_key, None)
    if server_state.started and server_state.process is not None:
        run_env["SWARMLLM_LOCAL_SERVER_PID"] = str(server_state.process.pid)
        run_env["SWARMLLM_LOCAL_SERVER_KIND"] = backend
        if server_state.log_path:
            run_env["SWARMLLM_LOCAL_SERVER_LOG_PATH"] = server_state.log_path

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run.py"),
        "--problem",
        problem_type,
        "--backend-profile",
        profile_path,
        "--coordinator-model",
        coordinator_model,
        "--agent-model",
        worker_model,
        "--agents",
        agents,
        "--iterations",
        iterations,
        "--max-concurrent",
        concurrent,
        "--instance-sizes",
        instance_sizes,
        "--explore-ratio",
        explore_ratio,
        "--seed",
        seed,
        "--timeout",
        timeout,
        "--agent-retries",
        retries,
        "--output-dir",
        output_dir,
    ]
    try:
        subprocess.run(cmd, env=run_env)
    finally:
        if server_state.process is not None:
            print()
            print("  Stopping local model server...")
            stop_local_server(server_state.process)

    # Print the summary so it's visible in the terminal after the run
    summary_path = os.path.join(output_dir, "summary.txt")
    if os.path.exists(summary_path):
        print()
        with open(summary_path, "r", encoding="utf-8") as f:
            print(f.read())
        print(f"  Output folder: {output_dir}")
        print()

    input("  Press Enter to exit...")


if __name__ == "__main__":
    main()
