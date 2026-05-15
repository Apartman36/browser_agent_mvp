from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.planners.base import PlannerAction
from agent.safety_engine import SafetyDecision


AUDIT_PATH = Path("logs") / "safety_audit.jsonl"
SECRET_KEYS_RE = re.compile(r"(?i)(password|passwd|token|api[_-]?key|secret|cookie|authorization)")
SECRET_VALUE_RE = re.compile(r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]{8,}|[0-9]{13,19})")


def append_safety_audit(
    *,
    step: int,
    action: PlannerAction,
    decision: SafetyDecision,
    observation: dict[str, Any] | None,
    user_decision: str | None = None,
    model: str | None = None,
) -> None:
    obs = observation or {}
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "step": step,
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
        "tool": action.tool,
        "sanitized_args": sanitize_args(action.args),
        "category": decision.category.value,
        "risk": decision.risk.value,
        "allowed": decision.allowed,
        "requires_confirmation": decision.requires_confirmation,
        "blocked": decision.blocked,
        "reason": decision.reason,
        "policy_rule": decision.policy_rule,
        "user_decision": user_decision,
        "model": model,
        "native_tool_call_id": action.native_tool_call_id,
    }
    with AUDIT_PATH.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in args.items():
        key_text = str(key)
        if SECRET_KEYS_RE.search(key_text):
            sanitized[key_text] = "[redacted]"
            continue
        if key_text == "text" and SECRET_VALUE_RE.search(str(value)):
            sanitized[key_text] = "[redacted]"
            continue
        sanitized[key_text] = value
    return sanitized


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)
