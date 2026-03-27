"""
SwarmLLM Entry Point

Usage:
    python run.py
    python run.py --agents 10 --iterations 3
    python run.py --problem job_scheduling --instance-sizes 20,50,100
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
    parser.add_argument("--problem", type=str, default=None,
                        help="Problem type (default: job_scheduling). Available: job_scheduling")
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
    parser.add_argument("--timeout", type=int, default=None, help="Code execution timeout in seconds (default: 120)")
    parser.add_argument("--agent-retries", type=int, default=None, help="Retries per agent if code fails pre-test (default: 1)")
    parser.add_argument("--base-urls", type=str, default=None, help="Comma-separated Ollama base URLs (e.g. 'http://localhost:11434,http://localhost:11435')")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output files")
    args = parser.parse_args()

    config = Config()

    # Apply problem type override first (affects how instance-sizes is parsed)
    if args.problem is not None:
        config.problem.problem_type = args.problem

    # Load the problem module for profile parsing
    from problems import load_problem
    problem = load_problem(config.problem.problem_type)

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
        config.problem.instance_profiles = problem.get_instance_profiles(
            args.instance_sizes, config.problem.seed
        )
    if args.seed is not None:
        config.problem.seed = args.seed
    if args.timeout is not None:
        config.sandbox.timeout = args.timeout
    if args.agent_retries is not None:
        config.swarm.agent_retries = args.agent_retries
    if args.base_urls is not None:
        config.llm.base_urls = [u.strip() for u in args.base_urls.split(",")]

    # If no profiles set, use problem defaults
    if not config.problem.instance_profiles:
        config.problem.instance_profiles = problem.get_default_profiles()

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Print config
    print(f"Config:")
    print(f"  Problem type:      {config.problem.problem_type}")
    print(f"  Coordinator model: {config.llm.coordinator_model}")
    print(f"  Agent model: {config.llm.agent_model}")
    print(f"  Agents: {config.swarm.num_agents}")
    print(f"  Iterations: {config.swarm.num_iterations}")
    print(f"  Explore ratio: {config.swarm.explore_ratio}")
    print(f"  Max concurrent: {config.swarm.max_concurrent_agents}")
    print(f"  Agent retries: {config.swarm.agent_retries}")
    print(f"  Instances: {len(config.problem.instance_profiles)}")
    for inst in config.problem.instance_profiles:
        print(f"    {inst.name}: {inst.params}")
    print(f"  Ollama URLs: {config.llm.base_urls}")
    print(f"  Seed: {config.problem.seed}")
    print()

    # Run
    asyncio.run(run_swarm(config, args.output_dir))


if __name__ == "__main__":
    main()
