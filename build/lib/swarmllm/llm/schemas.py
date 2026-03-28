from __future__ import annotations

"""Typed LLM outputs used by coordinator and worker agents."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class WorkerDraft(BaseModel):
    """Structured worker output for an agent-generated scheduling attempt."""

    approach: str = Field(min_length=1)
    code: str = Field(min_length=1)
    notes: str = ""

    @field_validator("approach", "code", "notes", mode="before")
    @classmethod
    def _strip_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class DirectionAssignment(BaseModel):
    """A coordinator-assigned research direction for a specific agent."""

    agent_id: int = Field(ge=0)
    mode: Literal["explore", "exploit"] = "explore"
    direction: str = Field(min_length=1)

    @field_validator("direction", mode="before")
    @classmethod
    def _strip_direction(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class CoordinatorRoundPlan(BaseModel):
    """Structured coordinator output for a swarm iteration."""

    analysis: str = ""
    directions: list[DirectionAssignment] = Field(default_factory=list)

    @field_validator("analysis", mode="before")
    @classmethod
    def _strip_analysis(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
