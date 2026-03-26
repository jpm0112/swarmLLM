"""
SwarmLLM Entry Point

Usage:
    python run.py
    python run.py --agents 10 --iterations 3
    python run.py --coordinator-model qwen2.5-coder:14b --agent-model qwen2.5:3b
"""

import argparse
import asyncio
import os
import sys

from config import Config
from orchestrator import run_swarm


def main():
    parser = argparse.ArgumentParser(description="SwarmLLM — Multi-Agent Optimization")
    parser.add_argument("--coordinator-model", type=str, default=None, help="Ollama model for the coordinator")
    parser.add_argument("--agent-model", type=str, default=None, help="Ollama model for worker agents")
    parser.add_argument("--agents", type=int, default=None, help="Number of agents (default: 20)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations (default: 5)")
    parser.add_argument("--explore-ratio", type=float, default=None, help="Explore/exploit ratio (default: 0.5)")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Max concurrent agents (default: 5)")
    parser.add_argument("--instance-sizes", type=str, default=None,
                        help="Comma-separated instance sizes, e.g. '20,50,100'. "
                             "Overrides default profiles with auto-generated ones.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for problem generation")
    parser.add_argument("--timeout", type=int, default=None, help="Code execution timeout in seconds (default: 30)")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output files")
    args = parser.parse_args()

    config = Config()

    # Apply overrides
    if args.coordinator_model is not None:
        config.llm.coordinator_model = args.coordinator_model
    if args.agent_model is not None:
        config.llm.agent_model = args.agent_model
    if args.agents is not None:
        config.swarm.num_agents = args.agents
    if args.iterations is not None:
        config.swarm.num_iterations = args.iterations
    if args.explore_ratio is not None:
        config.swarm.explore_ratio = args.explore_ratio
    if args.max_concurrent is not None:
        config.swarm.max_concurrent_agents = args.max_concurrent
    if args.instance_sizes is not None:
        from config import InstanceProfile
        sizes = [int(s.strip()) for s in args.instance_sizes.split(",")]
        # Auto-generate diverse profiles from sizes
        tightness_values = [0.4, 0.6, 0.8, 0.5, 0.7]  # cycle through if more than 3
        pt_ranges = [(1, 15), (1, 20), (5, 30), (1, 25), (3, 20)]
        config.problem.instances = []
        for i, size in enumerate(sizes):
            t = tightness_values[i % len(tightness_values)]
            min_pt, max_pt = pt_ranges[i % len(pt_ranges)]
            config.problem.instances.append(InstanceProfile(
                name=f"{size}jobs_t{t}",
                num_jobs=size,
                min_processing_time=min_pt,
                max_processing_time=max_pt,
                due_date_tightness=t,
            ))
    if args.seed is not None:
        config.problem.seed = args.seed
    if args.timeout is not None:
        config.sandbox.timeout = args.timeout

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print config
    print(f"Config:")
    print(f"  Coordinator model: {config.llm.coordinator_model}")
    print(f"  Agent model: {config.llm.agent_model}")
    print(f"  Agents: {config.swarm.num_agents}")
    print(f"  Iterations: {config.swarm.num_iterations}")
    print(f"  Explore ratio: {config.swarm.explore_ratio}")
    print(f"  Max concurrent: {config.swarm.max_concurrent_agents}")
    print(f"  Instances: {len(config.problem.instances)}")
    for inst in config.problem.instances:
        print(f"    {inst.name}: {inst.num_jobs} jobs, tightness={inst.due_date_tightness}, "
              f"pt={inst.min_processing_time}-{inst.max_processing_time}")
    print(f"  Seed: {config.problem.seed}")
    print()

    # Run
    asyncio.run(run_swarm(config, args.output_dir))


if __name__ == "__main__":
    main()
