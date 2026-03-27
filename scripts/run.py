"""
SwarmLLM Entry Point

Usage:
    python -m scripts.run --backend-profile configs/backends/ollama.local.example.toml
    python -m scripts.run --backend-profile configs/backends/vllm.single-node.example.toml --agents 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Ensure project root is on sys.path so `swarmllm` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarmllm.config import Config, InstanceProfile, LLMEndpoint
from swarmllm.core.orchestrator import run_swarm
from swarmllm.llm.profiles import apply_backend_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SwarmLLM — Multi-Agent Optimization")
    parser.add_argument(
        "--backend-profile",
        type=str,
        default=None,
        help="Path to a backend TOML profile. This is the primary backend selector.",
    )
    parser.add_argument("--coordinator-model", type=str, default=None, help="Coordinator model alias override")
    parser.add_argument("--agent-model", type=str, default=None, help="Worker model alias override")
    parser.add_argument("--agents", type=int, default=None, help="Number of agents (default: 20)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations (default: 5)")
    parser.add_argument("--explore-ratio", type=float, default=None, help="Explore/exploit ratio (default: 0.5)")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Max concurrent agents")
    parser.add_argument(
        "--instance-sizes",
        type=str,
        default=None,
        help="Comma-separated instance sizes, e.g. '20,50,100'. Overrides default profiles.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for problem generation")
    parser.add_argument("--timeout", type=int, default=None, help="Code execution timeout in seconds")
    parser.add_argument("--agent-retries", type=int, default=None, help="Retries per agent if code fails pre-test")
    parser.add_argument(
        "--base-urls",
        type=str,
        default=None,
        help="Comma-separated OpenAI-compatible base URLs. Overrides profile endpoint pools.",
    )
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for output files")
    return parser


def build_config_from_args(args: argparse.Namespace) -> Config:
    config = Config()
    if args.backend_profile is not None:
        apply_backend_profile(config, args.backend_profile)

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
        sizes = [int(s.strip()) for s in args.instance_sizes.split(",")]
        tightness_values = [0.4, 0.6, 0.8, 0.5, 0.7]
        pt_ranges = [(1, 15), (1, 20), (5, 30), (1, 25), (3, 20)]
        config.problem.instances = []
        for i, size in enumerate(sizes):
            tightness = tightness_values[i % len(tightness_values)]
            min_pt, max_pt = pt_ranges[i % len(pt_ranges)]
            config.problem.instances.append(
                InstanceProfile(
                    name=f"{size}jobs_t{tightness}",
                    num_jobs=size,
                    min_processing_time=min_pt,
                    max_processing_time=max_pt,
                    due_date_tightness=tightness,
                )
            )
    if args.seed is not None:
        config.problem.seed = args.seed
    if args.timeout is not None:
        config.sandbox.timeout = args.timeout
    if args.agent_retries is not None:
        config.swarm.agent_retries = args.agent_retries
    if args.base_urls is not None:
        _apply_base_url_override(config, [url.strip() for url in args.base_urls.split(",") if url.strip()])
    return config


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Config:")
    print(f"  Backend kind: {config.llm.backend_kind}")
    print(f"  Backend profile: {config.llm.backend_profile_path or '(default config)'}")
    print(f"  Coordinator model: {config.llm.coordinator_model}")
    print(f"  Agent model: {config.llm.agent_model}")
    print(f"  Coordinator endpoints: {[endpoint.base_url for endpoint in config.llm.coordinator_endpoints]}")
    print(f"  Worker endpoints: {[endpoint.base_url for endpoint in config.llm.worker_endpoints]}")
    print(f"  Agents: {config.swarm.num_agents}")
    print(f"  Iterations: {config.swarm.num_iterations}")
    print(f"  Explore ratio: {config.swarm.explore_ratio}")
    print(f"  Max concurrent: {config.swarm.max_concurrent_agents}")
    print(f"  Agent retries: {config.swarm.agent_retries}")
    print(f"  Instances: {len(config.problem.instances)}")
    for inst in config.problem.instances:
        print(
            f"    {inst.name}: {inst.num_jobs} jobs, tightness={inst.due_date_tightness}, "
            f"pt={inst.min_processing_time}-{inst.max_processing_time}"
        )
    print(f"  Seed: {config.problem.seed}")
    print()

    asyncio.run(run_swarm(config, args.output_dir))


def _apply_base_url_override(config: Config, base_urls: list[str]) -> None:
    if not base_urls:
        return
    coordinator_template = config.llm.coordinator_endpoints[0]
    worker_template = config.llm.worker_endpoints[0]
    config.llm.coordinator_endpoints = [
        LLMEndpoint(
            base_url=base_urls[0],
            api_key_env=coordinator_template.api_key_env,
            api_key=coordinator_template.api_key,
            label=coordinator_template.label,
        )
    ]
    config.llm.worker_endpoints = [
        LLMEndpoint(
            base_url=url,
            api_key_env=worker_template.api_key_env,
            api_key=worker_template.api_key,
            label=f"override-{idx}",
        )
        for idx, url in enumerate(base_urls)
    ]


if __name__ == "__main__":
    main()
