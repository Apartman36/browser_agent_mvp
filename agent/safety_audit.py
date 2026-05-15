from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.planners.base import PlannerAction
from agent.safety_engine import SafetyDecision
from agent.tool_registry import RiskLevel


AUDIT_PATH = Path("logs") / "safety_audit.jsonl"
SECRET_KEYS_RE = re.compile(r"(?i)(password|passwd|passcode|otp|2fa|cvv|card|token|api[_-]?key|secret|cookie|authorization)")
SECRET_VALUE_RE = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]{8,}|api[_-]?key\s*[:=]\s*\S+|\b[0-9][0-9 -]{11,23}[0-9]\b)"
)
SENSITIVE_TARGET_RE = re.compile(
    r"(?i)(password|passwd|passcode|otp|2fa|cvv|card|credit card|token|api[_ -]?key|secret|\u043f\u0430\u0440\u043e\u043b\u044c)"
)


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
        "sanitized_args": sanitize_args(action.args, action=action, decision=decision, observation=obs),
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


def sanitize_args(
    args: dict[str, Any],
    *,
    action: PlannerAction | None = None,
    decision: SafetyDecision | None = None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in args.items():
        key_text = str(key)
        if SECRET_KEYS_RE.search(key_text):
            sanitized[key_text] = "[redacted]"
            continue
        if key_text == "text" and _should_redact_text(
            str(value),
            args=args,
            action=action,
            decision=decision,
            observation=observation,
        ):
            sanitized[key_text] = "[redacted]"
            continue
        sanitized[key_text] = value
    return sanitized


def _should_redact_text(
    text: str,
    *,
    args: dict[str, Any],
    action: PlannerAction | None,
    decision: SafetyDecision | None,
    observation: dict[str, Any] | None,
) -> bool:
    is_type_text = action is not None and action.tool == "type_text"
    if decision is not None and decision.policy_rule == "sensitive_text":
        return True
    if is_type_text and decision is not None and decision.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        return True
    if _target_context_looks_sensitive(args=args, observation=observation):
        return True
    if _value_looks_sensitive(text):
        return True
    return False


def _target_context_looks_sensitive(*, args: dict[str, Any], observation: dict[str, Any] | None) -> bool:
    obs = observation or {}
    ref = str(args.get("ref", ""))
    context = _snapshot_context_for_ref(str(obs.get("snapshot_yaml", "")), ref)
    return bool(SENSITIVE_TARGET_RE.search(context))


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


def _value_looks_sensitive(text: str) -> bool:
    if SECRET_VALUE_RE.search(text):
        return True
    stripped = text.strip()
    if re.fullmatch(r"[A-Za-z0-9._~+/=-]{24,}", stripped):
        has_digit = bool(re.search(r"\d", stripped))
        has_symbol = bool(re.search(r"[._~+/=-]", stripped))
        has_mixed_case = bool(re.search(r"[a-z]", stripped)) and bool(re.search(r"[A-Z]", stripped))
        return sum([has_digit, has_symbol, has_mixed_case]) >= 2
    return False


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)
