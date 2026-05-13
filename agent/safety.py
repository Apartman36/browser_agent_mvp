from __future__ import annotations

import json
import re
from typing import Any


DANGEROUS_RE = re.compile(
    r"(?i)(submit|apply|pay|send|confirm|delete|remove|buy|checkout|отправ|отклик|подтверд|удал|оплат|купить|заказ)"
)


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
        action_text = json.dumps(action, ensure_ascii=False)
        ref = str(args.get("ref", ""))
        line = _snapshot_line_for_ref(current_obs.get("snapshot_yaml", ""), ref)
        if DANGEROUS_RE.search(action_text) or DANGEROUS_RE.search(line):
            return True, "typing text and pressing Enter may submit a high-risk form"

    if tool == "press_key" and str(args.get("key", "")).lower() == "enter":
        action_text = json.dumps(action, ensure_ascii=False)
        if DANGEROUS_RE.search(action_text):
            return True, "pressing Enter appears related to a high-risk action"

    return False, ""


def user_confirmed(answer: str) -> bool:
    return answer.strip().lower().startswith(("y", "yes", "д", "да"))


def _snapshot_line_for_ref(snapshot_yaml: str, ref: str) -> str:
    if not ref:
        return ""
    token = f"[ref={ref}]"
    for line in snapshot_yaml.splitlines():
        if token in line:
            return line
    return ""

