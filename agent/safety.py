from __future__ import annotations

import json
from typing import Any

from agent.planners.base import PlannerAction
from agent.safety_engine import DANGEROUS_LABEL_RE, SafetyEngine
from agent.tool_registry import TOOL_REGISTRY


DANGEROUS_RE = DANGEROUS_LABEL_RE


def is_high_risk(action: dict[str, Any], current_obs: dict[str, Any]) -> tuple[bool, str]:
    """Backward-compatible wrapper around the structured safety policy."""
    if action.get("needs_user_confirmation") is True:
        return True, "planner requested user confirmation"

    tool = str(action.get("tool", ""))
    try:
        spec = TOOL_REGISTRY.get(tool)
        planner_action = PlannerAction.model_validate(action)
        decision = SafetyEngine().evaluate(planner_action, spec, current_obs, memory=None)
    except Exception:
        action_text = json.dumps(action, ensure_ascii=False, default=str)
        if DANGEROUS_RE.search(action_text):
            return True, "action text matches high-risk heuristic"
        return False, ""
    if decision.requires_confirmation or decision.blocked:
        return True, decision.reason
    return False, ""


def user_confirmed(answer: str) -> bool:
    return answer.strip().lower().startswith(("y", "yes", "д", "да", "Рґ", "РґР°"))


def _snapshot_line_for_ref(snapshot_yaml: str, ref: str) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"
    for line in snapshot_yaml.splitlines():
        if token in line:
            return line
    return ""


def _snapshot_context_for_ref(snapshot_yaml: str, ref: str, radius: int = 2) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"
    lines = snapshot_yaml.splitlines()
    for index, line in enumerate(lines):
        if token in line:
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            return "\n".join(lines[start:end])
    return ""
