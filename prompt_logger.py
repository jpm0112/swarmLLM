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

        # Name the file
        if role == "coordinator":
            filename = "coordinator.md"
        else:
            filename = f"agent_{agent_id:02d}.md"

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
