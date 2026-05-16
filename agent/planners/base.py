from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class PlannerAction(BaseModel):
    thought: str = ""
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    risk: str | None = None
    needs_user_confirmation: bool = False
    new_facts: dict[str, Any] = Field(default_factory=dict)
    native_tool_call_id: str | None = None
    raw: Any | None = None

    def to_action_dict(self, include_raw: bool = False) -> dict[str, Any]:
        data = self.model_dump(exclude={"raw"} if not include_raw else set(), exclude_none=True)
        if data.get("risk") is None:
            data["risk"] = "low"
        return data


class BasePlanner(ABC):
    @abstractmethod
    def plan(self, memory_payload: dict[str, Any]) -> PlannerAction:
        raise NotImplementedError

    def append_tool_result(self, action: PlannerAction, result: dict[str, Any]) -> None:
        return None
