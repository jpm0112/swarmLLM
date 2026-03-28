from __future__ import annotations

"""
Shared Log Manager

Manages the flat markdown log file where all agents record their results.
The coordinator reads this log to decide on next directions.
"""

import os
from datetime import datetime

from swarmllm.config import LogConfig


class SharedLog:
    """Manages the shared results log file (flat markdown)."""

    def __init__(self, config: LogConfig, output_dir: str = "."):
        self.path = os.path.join(output_dir, config.log_file)
        self._ensure_exists()

    def _ensure_exists(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("# SwarmLLM Results Log\n\n")
                f.write(f"Started: {datetime.now().isoformat()}\n\n")
                f.write("---\n\n")

    def read(self) -> str:
        """Read the full log contents."""
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()

    def append_result(
        self,
        iteration: int,
        agent_id: int,
        direction: str,
        approach: str,
        code: str,
        score: float | None,
        success: bool,
        error: str | None = None,
        notes: str = "",
        runtime: float | None = None,
        failure_reason: str | None = None,
        instance_scores: dict | None = None,
        instance_errors: dict | None = None,
        llm_time: float | None = None,
        exec_time: float | None = None,
        retries_used: int = 0,
    ):
        """Append an agent's result to the log."""
        entry = _format_entry(
            iteration=iteration,
            agent_id=agent_id,
            direction=direction,
            approach=approach,
            code=code,
            score=score,
            success=success,
            error=error,
            notes=notes,
            runtime=runtime,
            failure_reason=failure_reason,
            instance_scores=instance_scores,
            instance_errors=instance_errors,
            llm_time=llm_time,
            exec_time=exec_time,
            retries_used=retries_used,
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry)

    def append_coordinator_summary(self, iteration: int, summary: str):
        """Append coordinator's analysis after a round."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"## Coordinator Summary — Iteration {iteration}\n\n")
            f.write(summary)
            f.write("\n\n---\n\n")

    def get_best_score(self) -> float | None:
        """Parse log to find the best (lowest) score so far."""
        content = self.read()
        best = None
        for line in content.split("\n"):
            if line.startswith("- **Aggregate Score:**"):
                try:
                    score_str = line.split("**Aggregate Score:**")[1].strip()
                    score = float(score_str)
                    if best is None or score < best:
                        best = score
                except (ValueError, IndexError):
                    continue
        return best


def _format_entry(
    iteration: int,
    agent_id: int,
    direction: str,
    approach: str,
    code: str,
    score: float | None,
    success: bool,
    error: str | None,
    notes: str,
    runtime: float | None = None,
    failure_reason: str | None = None,
    instance_scores: dict | None = None,
    instance_errors: dict | None = None,
    llm_time: float | None = None,
    exec_time: float | None = None,
    retries_used: int = 0,
) -> str:
    """Format a single agent result entry."""
    status = "SUCCESS" if success else "FAILED"
    score_str = str(score) if score is not None else "N/A"
    runtime_str = f"{runtime:.1f}s" if runtime is not None else "N/A"
    llm_str = f"{llm_time:.1f}s" if llm_time is not None else "N/A"
    exec_str = f"{exec_time:.1f}s" if exec_time is not None else "N/A"
    retry_str = f", retries: {retries_used}" if retries_used > 0 else ""

    lines = [
        f"### Iteration {iteration} — Agent {agent_id} [{status}] ({runtime_str})",
        f"",
        f"- **Direction:** {direction}",
        f"- **Approach:** {approach}",
        f"- **Aggregate Score:** {score_str}",
        f"- **Runtime:** {runtime_str} (LLM: {llm_str}, execution: {exec_str}{retry_str})",
    ]

    # Per-instance scores
    if instance_scores:
        for size in sorted(instance_scores.keys()):
            lines.append(f"- **Score ({size} jobs):** {instance_scores[size]}")

    # Per-instance errors
    if instance_errors:
        for size in sorted(instance_errors.keys()):
            lines.append(f"- **Error ({size} jobs):** {instance_errors[size]}")

    if failure_reason:
        lines.append(f"- **Failure Reason:** {failure_reason}")

    if error:
        lines.append(f"- **Error:** `{error[:500]}`")

    if notes:
        lines.append(f"- **Notes:** {notes}")

    lines.append(f"")
    lines.append(f"<details><summary>Code</summary>")
    lines.append(f"")
    lines.append(f"```python")
    lines.append(code)
    lines.append(f"```")
    lines.append(f"")
    lines.append(f"</details>")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    return "\n".join(lines)
