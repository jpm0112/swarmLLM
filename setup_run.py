"""
Interactive setup script — asks for parameters and launches the swarm.
Called by run.bat after venv activation.
"""

import subprocess
import sys
import os
import re
from datetime import datetime


def get_models():
    """Get list of available Ollama models with sizes."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: Ollama is not running. Start it first.")
        sys.exit(1)
    models = []
    sizes = {}
    for line in result.stdout.strip().split("\n")[1:]:  # skip header
        parts = line.split()
        if parts:
            name = parts[0]
            models.append(name)
            # Parse size (e.g., "9.0 GB", "1.9 GB")
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
    return models, sizes


def estimate_parallel(model_size_gb, vram_gb=24):
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


def pick_model(models, sizes, label, preferred="qwen2.5-coder:14b"):
    """Show numbered menu with sizes and let user pick a model."""
    # Find default index (preferred model, or first)
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


def ask(prompt, default, explanation=""):
    """Ask for a parameter with a default value and optional explanation."""
    if explanation:
        print(f"    {explanation}")
    val = input(f"  {prompt} [{default}]: ").strip()
    return val if val else str(default)


def main():
    print("=" * 60)
    print("  SwarmLLM — Run Setup")
    print("=" * 60)

    models, sizes = get_models()
    if not models:
        print("ERROR: No models found. Pull a model first (ollama pull <model>).")
        sys.exit(1)

    print("\n  The COORDINATOR reads all results and assigns research")
    print("  directions. Bigger model = smarter strategy.")
    coord_model = pick_model(models, sizes, "Coordinator model")
    print(f"  -> {coord_model}")

    print("\n  The AGENT model writes optimization code. Runs N times")
    print("  per iteration, so speed matters. Can be smaller/faster.")
    agent_model = pick_model(models, sizes, "Agent model")
    print(f"  -> {agent_model}")

    # Estimate best parallelism for the agent model
    agent_size = sizes.get(agent_model, 10)
    recommended_parallel = estimate_parallel(agent_size)
    print(f"\n  Agent model ~{agent_size:.1f}GB -> recommended max parallel: {recommended_parallel} (on 24GB VRAM)")

    print()
    agents = ask(
        "Number of agents", recommended_parallel,
        f"How many agents work on the problem each iteration. Default matches max parallel ({recommended_parallel})."
    )
    iterations = ask(
        "Number of iterations", 5,
        "How many rounds of generate -> evaluate -> coordinate."
    )
    concurrent = ask(
        "Max concurrent agents", min(int(agents), recommended_parallel),
        f"How many agents run at the same time. Max recommended: {recommended_parallel} (based on your VRAM)."
    )
    instance_sizes = ask(
        "Instance sizes (comma-separated)", "20,50,100",
        "Problem sizes to test on. Each gets different characteristics (tightness, PT range). E.g. 20,50,100"
    )
    explore = ask(
        "Explore ratio (0.0-1.0)", 0.5,
        "Fraction of agents that try new ideas vs refine the best ones. 0.5 = balanced."
    )
    timeout = ask(
        "Code execution timeout (seconds)", 60,
        "Max time each agent's code can run. Same limit for all agents."
    )
    seed = ask(
        "Random seed", 1048596,
        "Fixed seed for reproducibility. Same seed = same problem instance."
    )

    # Build output folder name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_coord = re.sub(r'[^a-zA-Z0-9._-]', '_', coord_model)
    safe_agent = re.sub(r'[^a-zA-Z0-9._-]', '_', agent_model)
    outdir = os.path.join("runs", f"{timestamp}_coord-{safe_coord}_agent-{safe_agent}_{agents}agents_{iterations}iter")
    os.makedirs(outdir, exist_ok=True)

    print()
    print("=" * 60)
    print("  Configuration:")
    print(f"    Coordinator:   {coord_model}")
    print(f"    Agent model:   {agent_model}")
    print(f"    Agents:        {agents}")
    print(f"    Iterations:    {iterations}")
    print(f"    Concurrent:    {concurrent}")
    print(f"    Instances:     {instance_sizes} jobs")
    print(f"    Explore ratio: {explore}")
    print(f"    Timeout:       {timeout}s")
    print(f"    Seed:          {seed}")
    print(f"    Output folder: {outdir}")
    print("=" * 60)
    print()

    confirm = input("  Start the swarm? (Y/n): ").strip()
    if confirm.lower() == "n":
        print("  Cancelled.")
        return

    # Install extra pip packages for agents
    from config import SandboxConfig
    extra_pkgs = SandboxConfig().pip_packages
    if extra_pkgs:
        print(f"\n  Installing agent packages: {', '.join(extra_pkgs)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + extra_pkgs,
            capture_output=True,
        )

    # Restart Ollama with OLLAMA_NUM_PARALLEL so it actually takes effect
    print()
    print(f"  Restarting Ollama with OLLAMA_NUM_PARALLEL={concurrent}...")
    subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], capture_output=True)
    subprocess.run(["taskkill", "/f", "/im", "ollama_runners.exe"], capture_output=True)
    import time as _time
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = concurrent
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _time.sleep(3)  # wait for Ollama to start

    print("  Starting SwarmLLM...")
    print()

    cmd = [
        sys.executable, "run.py",
        "--coordinator-model", coord_model,
        "--agent-model", agent_model,
        "--agents", agents,
        "--iterations", iterations,
        "--max-concurrent", concurrent,
        "--instance-sizes", instance_sizes,
        "--explore-ratio", explore,
        "--seed", seed,
        "--timeout", timeout,
        "--output-dir", outdir,
    ]

    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
