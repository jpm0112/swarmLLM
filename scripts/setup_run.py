"""
Interactive setup script that chooses a backend profile and launches the swarm.

This script is backend-aware but intentionally lightweight: local or remote model
servers must already be running. Runtime validation happens inside `scripts/run.py`
via `/v1/models` checks before the swarm starts.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_PROFILE_BY_BACKEND = {
    "ollama": os.path.join("configs", "backends", "ollama.local.example.toml"),
    "vllm-metal": os.path.join("configs", "backends", "vllm-metal.local.example.toml"),
    "vllm": os.path.join("configs", "backends", "vllm.single-node.example.toml"),
}


def supported_backends_for_platform(system_name: str, machine: str) -> list[str]:
    """Return the backend kinds supported by the current OS."""
    system_name = system_name.lower()
    machine = machine.lower()
    if system_name == "windows":
        return ["ollama"]
    if system_name == "darwin":
        backends = ["ollama"]
        if machine in {"arm64", "aarch64"}:
            backends.append("vllm-metal")
        return backends
    return ["ollama", "vllm"]


def ask(prompt: str, default: str, explanation: str = "") -> str:
    """Ask for a parameter with a default value and optional explanation."""
    if explanation:
        print(f"    {explanation}")
    value = input(f"  {prompt} [{default}]: ").strip()
    return value if value else default


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
    if backend == "vllm-metal":
        return (
            "Expect a running Apple Silicon vLLM Metal server, typically from "
            "~/.venv-vllm-metal/bin/vllm serve ... using the example YAML template."
        )
    return (
        "Expect a running vLLM server. Cluster or cloud profiles should point at "
        "remote endpoints; this launcher does not provision infrastructure."
    )


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

    agents = ask(
        "Number of agents",
        "20",
        "How many worker agents to run per iteration.",
    )
    iterations = ask(
        "Number of iterations",
        "5",
        "How many coordinator/worker rounds to run.",
    )
    instance_sizes = ask(
        "Instance sizes (comma-separated)",
        "20,50,100",
        "Problem sizes to test on. Example: 20,50,100",
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
    output_dir = os.path.join("runs", f"{timestamp}_{backend}_{agents}agents_{iterations}iter")
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("  Configuration:")
    print(f"    Backend:        {backend}")
    print(f"    Profile:        {profile_path}")
    print(f"    Agents:         {agents}")
    print(f"    Iterations:     {iterations}")
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

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run.py"),
        "--backend-profile",
        profile_path,
        "--agents",
        agents,
        "--iterations",
        iterations,
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
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
