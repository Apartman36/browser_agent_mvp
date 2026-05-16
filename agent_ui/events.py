from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


EventType = Literal[
    "run_started",
    "observation",
    "tool_call",
    "tool_result",
    "safety_prompt",
    "screenshot",
    "done",
    "error",
    "cancelled",
]

TERMINAL_EVENT_TYPES = {"done", "error", "cancelled"}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentUIEvent:
    event_type: EventType | str
    data: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    created_at: str = field(default_factory=utc_timestamp)

    def with_run_id(self, run_id: str) -> "AgentUIEvent":
        return AgentUIEvent(
            event_type=self.event_type,
            data=dict(self.data),
            run_id=run_id,
            created_at=self.created_at,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = dict(self.data)
        payload["timestamp"] = self.created_at
        if self.run_id:
            payload["run_id"] = self.run_id
        return payload


def make_event(event_type: EventType | str, data: dict[str, Any] | None = None) -> AgentUIEvent:
    return AgentUIEvent(event_type=event_type, data=data or {})
