from __future__ import annotations

from collections import deque
from typing import Any


class Memory:
    def __init__(self, goal: str) -> None:
        self.goal = goal
        self.facts: dict[str, str] = {}
        self.history: deque[dict[str, Any]] = deque(maxlen=8)
        self.current_obs: dict[str, Any] | None = None
        self.step = 0

    def update_observation(self, obs: dict[str, Any]) -> None:
        self.current_obs = obs

    def add_action(self, action: dict[str, Any], result: dict[str, Any]) -> None:
        self.step += 1
        result_data = result.get("data", {}) if isinstance(result, dict) else {}
        self.history.append(
            {
                "step": self.step,
                "thought": self._truncate(str(action.get("thought", "")), 500),
                "tool": action.get("tool"),
                "args": self._compact_args(action.get("args", {})),
                "result_ok": bool(result.get("ok", False)) if isinstance(result, dict) else False,
                "result_message": self._truncate(str(result.get("message", "")), 500)
                if isinstance(result, dict)
                else "",
                "result_data_excerpt": self._truncate(str(result_data), 1000),
            }
        )

    def merge_facts(self, new_facts: dict[str, Any] | None) -> None:
        if not new_facts:
            return
        for key, value in new_facts.items():
            self.facts[str(key)] = self._truncate(str(value), 1000)

    def to_prompt_payload(self) -> dict[str, Any]:
        obs = self.current_obs or {}
        return {
            "goal": self.goal,
            "facts": self.facts,
            "recent_history": list(self.history),
            "current_page": {
                "url": obs.get("url", ""),
                "title": obs.get("title", ""),
                "snapshot_yaml": self._truncate(obs.get("snapshot_yaml", ""), 12000),
                "body_text_excerpt": self._truncate(obs.get("body_text", ""), 6000),
                "observation_error": obs.get("error"),
            },
        }

    @classmethod
    def _compact_args(cls, args: Any) -> Any:
        if not isinstance(args, dict):
            return cls._truncate(str(args), 1000)
        compact: dict[str, Any] = {}
        for key, value in args.items():
            compact[str(key)] = cls._truncate(str(value), 1200)
        return compact

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + "...[truncated]"

