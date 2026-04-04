"""Structured per-attempt memory for swarm runs.

Writes separate TOON files per iteration:
  - iterationN_coders.toon   — array of all coder attempt records
  - iterationN_coordinator.toon — single coordinator decision record

Files are stored under a central ``memory/`` folder at the project root,
with one subfolder per run.  Legacy ``.json`` files are still readable
for backward compatibility with older runs.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from toon import decode as toon_decode, encode as toon_encode


# Resolve project-root ``memory/`` directory (two levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_ROOT = _PROJECT_ROOT / "memory"


def _sanitize_for_toon(obj: Any) -> Any:
    """Recursively convert non-primitive types to strings (like json.dump default=str)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_toon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_toon(v) for v in obj]
    return str(obj)


class AttemptMemory:
    """Append-only structured memory for coder attempts and coordinator decisions."""

    def __init__(self, run_name: str) -> None:
        self._run_dir = MEMORY_ROOT / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Buffer coder records per iteration so we can write one file per iteration.
        self._coder_buffers: dict[int, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def record_attempt(
        self,
        iteration: int,
        agent_id: int,
        mode: str,
        direction: str,
        source_refs: list[dict],
        result: dict,
    ) -> None:
        """Buffer a single coder attempt record for *iteration*."""
        record = {
            "record_type": "agent_attempt",
            "iteration": iteration,
            "agent_id": agent_id,
            "mode": mode,
            "direction": direction,
            "source_refs": source_refs,
            "approach": result.get("approach", ""),
            "code": result.get("code", ""),
            "score": result.get("score"),
            "success": result.get("success", False),
            "error": result.get("error"),
            "failure_reason": result.get("failure_reason"),
            "notes": result.get("notes", ""),
            "instance_scores": result.get("instance_scores", {}),
            "instance_errors": result.get("instance_errors", {}),
        }
        with self._lock:
            self._coder_buffers.setdefault(iteration, []).append(record)

    def flush_iteration_coders(self, iteration: int) -> None:
        """Write all buffered coder records for *iteration* to disk."""
        with self._lock:
            records = self._coder_buffers.pop(iteration, [])
        if not records:
            return
        path = self._run_dir / f"iteration{iteration}_coders.toon"
        with open(path, "w", encoding="utf-8") as f:
            f.write(toon_encode(_sanitize_for_toon(records)))

    def record_coordinator_decision(
        self,
        iteration: int,
        assignments: list,
        analysis: str | None = None,
    ) -> None:
        """Write a coordinator decision record for *iteration* immediately."""
        serialized = []
        for a in assignments:
            entry: dict[str, Any] = {
                "agent_id": getattr(a, "agent_id", None),
                "mode": getattr(a, "mode", None),
                "direction": getattr(a, "direction", None),
            }
            refs = getattr(a, "source_refs", [])
            entry["source_refs"] = [
                {"agent_id": getattr(r, "agent_id", None), "iteration": getattr(r, "iteration", None)}
                for r in refs
            ]
            serialized.append(entry)

        record = {
            "record_type": "coordinator_decision",
            "iteration": iteration,
            "analysis": analysis,
            "assignments": serialized,
        }
        path = self._run_dir / f"iteration{iteration}_coordinator.toon"
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                f.write(toon_encode(_sanitize_for_toon(record)))

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def read_coders(self, iteration: int) -> list[dict]:
        """Read coder records for a given iteration (.toon, with .json fallback)."""
        toon_path = self._run_dir / f"iteration{iteration}_coders.toon"
        if toon_path.exists():
            with open(toon_path, encoding="utf-8") as f:
                return toon_decode(f.read())
        json_path = self._run_dir / f"iteration{iteration}_coders.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def read_coordinator(self, iteration: int) -> dict | None:
        """Read the coordinator decision for a given iteration (.toon, with .json fallback)."""
        toon_path = self._run_dir / f"iteration{iteration}_coordinator.toon"
        if toon_path.exists():
            with open(toon_path, encoding="utf-8") as f:
                return toon_decode(f.read())
        json_path = self._run_dir / f"iteration{iteration}_coordinator.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    def read_knowledge(self) -> str:
        """Read the current knowledge document, or return empty string."""
        path = self._run_dir / "knowledge.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def write_knowledge(self, content: str) -> None:
        """Write the knowledge document (overwrites previous version)."""
        path = self._run_dir / "knowledge.md"
        with self._lock:
            path.write_text(content, encoding="utf-8")

    def write_knowledge_iteration(self, iteration: int, content: str) -> None:
        """Save a snapshot of the knowledge document for a specific iteration."""
        path = self._run_dir / f"iteration{iteration}_knowledge.md"
        with self._lock:
            path.write_text(content, encoding="utf-8")
