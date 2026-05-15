from __future__ import annotations

import re
from typing import Any



DANGEROUS_RE = re.compile(
    r"(?i)(apply|pay|send|confirm|delete|remove|buy|checkout|submit application|отправ|отклик|подтверд|удал|оплат|купить|заказ)"
)


def _action_contains_dangerous_text(obj: Any) -> bool:
    if isinstance(obj, str):
        return bool(DANGEROUS_RE.search(obj))
    elif isinstance(obj, dict):
        return any(_action_contains_dangerous_text(k) or _action_contains_dangerous_text(v) for k, v in obj.items())
    elif isinstance(obj, list):
        return any(_action_contains_dangerous_text(item) for item in obj)
    return False


def is_high_risk(action: dict[str, Any], current_obs: dict[str, Any]) -> tuple[bool, str]:
    if action.get("needs_user_confirmation") is True:
        return True, "planner requested user confirmation"

    tool = action.get("tool")
    args = action.get("args") or {}

    if tool == "click_element":
        ref = str(args.get("ref", ""))
        line = _snapshot_line_for_ref(current_obs.get("snapshot_yaml", ""), ref)
        if DANGEROUS_RE.search(line):
            return True, f'click on "{line.strip()}"'

    if tool == "type_text" and bool(args.get("submit", False)):
        ref = str(args.get("ref", ""))
        context = _snapshot_context_for_ref(current_obs.get("snapshot_yaml", ""), ref)
        if DANGEROUS_RE.search(context):
            return True, "typing text and pressing Enter may submit a high-risk form"

    if tool == "press_key" and str(args.get("key", "")).lower() == "enter":
        if _action_contains_dangerous_text(action):
            return True, "pressing Enter appears related to a high-risk action"

    return False, ""


def user_confirmed(answer: str) -> bool:
    return answer.strip().lower().startswith(("y", "yes", "д", "да"))


def _snapshot_line_for_ref(snapshot_yaml: str, ref: str) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"

    idx = snapshot_yaml.find(token)
    if idx == -1:
        return ""

    start = snapshot_yaml.rfind('\n', 0, idx)
    start = start + 1 if start != -1 else 0

    end = snapshot_yaml.find('\n', idx)
    end = end if end != -1 else len(snapshot_yaml)

    return snapshot_yaml[start:end]


def _snapshot_context_for_ref(snapshot_yaml: str, ref: str, radius: int = 2) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"

    idx = snapshot_yaml.find(token)
    if idx == -1:
        return ""

    # Find start
    start = idx
    for _ in range(radius + 1):
        start = snapshot_yaml.rfind('\n', 0, start)
        if start == -1:
            start = 0
            break

    if start != 0:
        start += 1

    # Find end
    end = idx
    for _ in range(radius + 1):
        next_end = snapshot_yaml.find('\n', end + 1) if end < len(snapshot_yaml) else -1
        if next_end == -1:
            end = len(snapshot_yaml)
            break
        end = next_end

    return snapshot_yaml[start:end]
