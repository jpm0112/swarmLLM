from __future__ import annotations

"""
Prompt Logger

Logs every LLM call as individual markdown files, organized by iteration.

Structure:
    prompts/
        iter_1/
            agent_00.md
            agent_01.md
            ...
        iter_2/
            coordinator.md
            agent_00.md
            ...
"""

import json
import os
from datetime import datetime


class PromptLogger:
    """Saves each LLM interaction as a readable markdown file."""

    def __init__(self, output_dir: str = "."):
        self.prompts_dir = os.path.join(output_dir, "prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)

    def log(
        self,
        role: str,            # "agent" or "coordinator"
        agent_id: int | None,
        iteration: int,
        system_prompt: str,
        user_prompt: str,
        response: str,
        error: str | None = None,
    ):
        # Create iteration folder
        iter_dir = os.path.join(self.prompts_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)

        filename = self._filename(role, agent_id)

        filepath = os.path.join(iter_dir, filename)

        # Write readable markdown
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {role.capitalize()}")
            if agent_id is not None:
                f.write(f" {agent_id}")
            f.write(f" — Iteration {iteration}\n\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")

            if error:
                f.write(f"**Error:** {error}\n\n")

            f.write(f"---\n\n")
            f.write(f"## System Prompt\n\n")
            f.write(f"{system_prompt}\n\n")

            f.write(f"---\n\n")
            f.write(f"## User Prompt\n\n")
            f.write(f"{user_prompt}\n\n")

            f.write(f"---\n\n")
            f.write(f"## Response\n\n")
            f.write(f"{response}\n")

    def log_structured(
        self,
        role: str,
        agent_id: int | None,
        iteration: int,
        system_prompt: str,
        user_prompt: str,
        output: object,
        messages_json: bytes | str,
        error: str | None = None,
    ):
        """Log a structured PydanticAI run with raw message history and output."""
        iter_dir = os.path.join(self.prompts_dir, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        filepath = os.path.join(iter_dir, self._filename(role, agent_id))

        if isinstance(messages_json, bytes):
            messages_text = messages_json.decode("utf-8")
        else:
            messages_text = messages_json

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {role.capitalize()}")
            if agent_id is not None:
                f.write(f" {agent_id}")
            f.write(f" — Iteration {iteration}\n\n")
            f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")

            if error:
                f.write(f"**Error:** {error}\n\n")

            f.write("---\n\n")
            f.write("## System Prompt\n\n")
            f.write(f"{system_prompt}\n\n")

            f.write("---\n\n")
            f.write("## User Prompt\n\n")
            f.write(f"{user_prompt}\n\n")

            f.write("---\n\n")
            f.write("## Structured Output\n\n")
            f.write("```json\n")
            f.write(f"{self._serialize_output(output)}\n")
            f.write("```\n\n")

            f.write("---\n\n")
            f.write("## Message History\n\n")
            f.write("```json\n")
            f.write(f"{messages_text}\n")
            f.write("```\n")

    def _filename(self, role: str, agent_id: int | None) -> str:
        if role == "coordinator":
            return "coordinator.md"
        if role.startswith("coordinator_"):
            suffix = role.split("_", 1)[1]
            return f"coordinator_{suffix}.md"
        if agent_id is None:
            return f"{role}.md"
        if role == "agent":
            return f"agent_{agent_id:02d}.md"
        suffix = role.replace("agent_", "", 1)
        return f"agent_{agent_id:02d}_{suffix}.md"

    def _serialize_output(self, output: object) -> str:
        if hasattr(output, "model_dump"):
            payload = output.model_dump()
        else:
            payload = output
        return json.dumps(payload, indent=2, ensure_ascii=True, default=str)
