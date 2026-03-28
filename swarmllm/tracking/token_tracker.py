from __future__ import annotations

"""
Token Tracker

Tracks prompt/completion/total token usage across all LLM calls.
Provides per-call, per-iteration, and per-role (agent vs coordinator) breakdowns.
"""

import json
import os
from dataclasses import dataclass

from pydantic_ai.usage import RunUsage


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0  # subset of completion_tokens (reasoning/thinking)

    @classmethod
    def from_run_usage(cls, usage: RunUsage | None) -> "TokenUsage | None":
        """Convert a PydanticAI RunUsage object into tracker-friendly totals."""
        if usage is None:
            return None
        prompt_tokens = (
            usage.input_tokens
            + usage.cache_write_tokens
            + usage.cache_read_tokens
            + usage.input_audio_tokens
            + usage.cache_audio_read_tokens
        )
        completion_tokens = usage.output_tokens + usage.output_audio_tokens
        # reasoning_tokens is a subset of output_tokens tracked in details by the OpenAI provider
        thinking_tokens = usage.details.get("reasoning_tokens", 0)
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            thinking_tokens=thinking_tokens,
        )

    @classmethod
    def from_usage_delta(cls, before: "UsageSnapshot", after: "UsageSnapshot") -> "TokenUsage":
        """Convert a usage delta into tracker-friendly totals."""
        prompt_tokens = (
            (after.input_tokens - before.input_tokens)
            + (after.cache_write_tokens - before.cache_write_tokens)
            + (after.cache_read_tokens - before.cache_read_tokens)
            + (after.input_audio_tokens - before.input_audio_tokens)
            + (after.cache_audio_read_tokens - before.cache_audio_read_tokens)
        )
        completion_tokens = (
            (after.output_tokens - before.output_tokens)
            + (after.output_audio_tokens - before.output_audio_tokens)
        )
        thinking_tokens = after.details.get("reasoning_tokens", 0) - before.details.get("reasoning_tokens", 0)
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            thinking_tokens=thinking_tokens,
        )


@dataclass(frozen=False)
class UsageSnapshot:
    """Point-in-time counters from a PydanticAI RunUsage object."""

    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    input_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0
    output_tokens: int = 0
    output_audio_tokens: int = 0
    details: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    @classmethod
    def capture(cls, usage: RunUsage | None) -> "UsageSnapshot":
        """Capture the current token counters from a usage object."""
        if usage is None:
            return cls()
        return cls(
            input_tokens=usage.input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            input_audio_tokens=usage.input_audio_tokens,
            cache_audio_read_tokens=usage.cache_audio_read_tokens,
            output_tokens=usage.output_tokens,
            output_audio_tokens=usage.output_audio_tokens,
            details=dict(usage.details),
        )


class TokenTracker:
    """Aggregates token usage across the entire swarm run."""

    def __init__(self):
        # Per-call records: list of dicts
        self._calls: list[dict] = []
        # Running totals
        self.total_prompt: int = 0
        self.total_completion: int = 0
        self.total_tokens: int = 0
        self.total_thinking: int = 0
        # Per-role totals
        self.agent_prompt: int = 0
        self.agent_completion: int = 0
        self.agent_total: int = 0
        self.agent_thinking: int = 0
        self.coordinator_prompt: int = 0
        self.coordinator_completion: int = 0
        self.coordinator_total: int = 0
        self.coordinator_thinking: int = 0
        # Per-iteration totals
        self._iteration_totals: dict[int, dict] = {}

    def record(
        self,
        role: str,
        iteration: int,
        agent_id: int | None,
        model: str,
        usage: TokenUsage,
        duration_seconds: float | None = None,
    ):
        """Record a single LLM call's token usage."""
        self._calls.append({
            "role": role,
            "iteration": iteration,
            "agent_id": agent_id,
            "model": model,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "thinking_tokens": usage.thinking_tokens,
            "duration_seconds": duration_seconds,
        })

        # Running totals
        self.total_prompt += usage.prompt_tokens
        self.total_completion += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.total_thinking += usage.thinking_tokens

        # Per-role
        if role == "coordinator":
            self.coordinator_prompt += usage.prompt_tokens
            self.coordinator_completion += usage.completion_tokens
            self.coordinator_total += usage.total_tokens
            self.coordinator_thinking += usage.thinking_tokens
        else:
            self.agent_prompt += usage.prompt_tokens
            self.agent_completion += usage.completion_tokens
            self.agent_total += usage.total_tokens
            self.agent_thinking += usage.thinking_tokens

        # Per-iteration
        if iteration not in self._iteration_totals:
            self._iteration_totals[iteration] = {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "thinking_tokens": 0, "agent_calls": 0, "coordinator_calls": 0,
            }
        it = self._iteration_totals[iteration]
        it["prompt_tokens"] += usage.prompt_tokens
        it["completion_tokens"] += usage.completion_tokens
        it["total_tokens"] += usage.total_tokens
        it["thinking_tokens"] += usage.thinking_tokens
        if role == "coordinator":
            it["coordinator_calls"] += 1
        else:
            it["agent_calls"] += 1

    def get_iteration_summary(self, iteration: int) -> dict:
        """Get token summary for a specific iteration."""
        return self._iteration_totals.get(iteration, {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "agent_calls": 0, "coordinator_calls": 0,
        })

    def print_iteration_tokens(self, iteration: int):
        """Print token usage for an iteration."""
        s = self.get_iteration_summary(iteration)
        thinking_str = f", {s['thinking_tokens']:,} thinking" if s.get("thinking_tokens") else ""
        print(f"  Tokens this iteration: "
              f"{s['total_tokens']:,} total "
              f"({s['prompt_tokens']:,} prompt + {s['completion_tokens']:,} completion{thinking_str}) "
              f"| {s['agent_calls']} agent calls"
              + (f" + {s['coordinator_calls']} coordinator" if s['coordinator_calls'] > 0 else ""))

    def print_running_total(self):
        """Print running totals."""
        print(f"  Running total: {self.total_tokens:,} tokens "
              f"(agents: {self.agent_total:,}, coordinator: {self.coordinator_total:,})")

    def print_final_summary(self) -> list[str]:
        """Print and return final token summary lines."""
        lines = []

        def out(msg=""):
            print(msg)
            lines.append(msg)

        out("  Token Usage Summary:")
        out(f"    Total tokens:       {self.total_tokens:>12,}")
        out(f"      Prompt tokens:    {self.total_prompt:>12,}")
        out(f"      Completion tokens:{self.total_completion:>12,}")
        out()
        out(f"    Agent tokens:       {self.agent_total:>12,} "
            f"({len([c for c in self._calls if c['role'] == 'agent'])} calls)")
        out(f"    Coordinator tokens: {self.coordinator_total:>12,} "
            f"({len([c for c in self._calls if c['role'] == 'coordinator'])} calls)")
        out()
        out("    Per iteration:")
        for it in sorted(self._iteration_totals.keys()):
            s = self._iteration_totals[it]
            out(f"      Iteration {it:>2}: {s['total_tokens']:>10,} tokens "
                f"({s['agent_calls']} agents + {s['coordinator_calls']} coordinator)")

        return lines

    def save(self, output_dir: str):
        """Save detailed token log to file."""
        path = os.path.join(output_dir, "token_usage.json")
        data = {
            "summary": {
                "total_prompt_tokens": self.total_prompt,
                "total_completion_tokens": self.total_completion,
                "total_thinking_tokens": self.total_thinking,
                "total_tokens": self.total_tokens,
                "agent_tokens": self.agent_total,
                "agent_thinking_tokens": self.agent_thinking,
                "coordinator_tokens": self.coordinator_total,
                "coordinator_thinking_tokens": self.coordinator_thinking,
                "total_calls": len(self._calls),
            },
            "per_iteration": self._iteration_totals,
            "calls": self._calls,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
