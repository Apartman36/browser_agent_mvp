from __future__ import annotations

from agent.planners.base import PlannerAction
from agent.safety_audit import sanitize_args
from agent.safety_engine import SafetyEngine
from agent.tool_registry import TOOL_REGISTRY


def _decision(action: PlannerAction, obs: dict[str, object]):
    return SafetyEngine().evaluate(action, TOOL_REGISTRY.get(action.tool), obs)


def test_type_text_password_context_redacts_text() -> None:
    obs = {"snapshot_yaml": '- textbox "Password" [ref=p1]', "body_text": ""}
    action = PlannerAction(tool="type_text", args={"ref": "p1", "text": "Hunter2!", "submit": False})
    decision = _decision(action, obs)

    sanitized = sanitize_args(action.args, action=action, decision=decision, observation=obs)

    assert sanitized["text"] == "[redacted]"
    assert sanitized["ref"] == "p1"


def test_type_text_api_key_value_redacts_text() -> None:
    obs = {"snapshot_yaml": '- textbox "Search" [ref=s1]', "body_text": ""}
    action = PlannerAction(tool="type_text", args={"ref": "s1", "text": "sk-testsecret123456789", "submit": False})
    decision = _decision(action, obs)

    sanitized = sanitize_args(action.args, action=action, decision=decision, observation=obs)

    assert sanitized["text"] == "[redacted]"


def test_type_text_normal_search_query_is_not_redacted() -> None:
    obs = {"snapshot_yaml": '- textbox "Search jobs" [ref=s1]', "body_text": ""}
    action = PlannerAction(tool="type_text", args={"ref": "s1", "text": "AI engineer", "submit": True})
    decision = _decision(action, obs)

    sanitized = sanitize_args(action.args, action=action, decision=decision, observation=obs)

    assert sanitized["text"] == "AI engineer"
