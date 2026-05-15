from __future__ import annotations

import json
from types import SimpleNamespace

from agent.planners.base import PlannerAction
from agent.planners.json_mode import JsonModePlanner
from agent.tool_registry import TOOL_REGISTRY


class ProviderError(Exception):
    pass


class FakeLLM:
    provider_unavailable_error_type = ProviderError

    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []
        self.client = object()
        self.model = "primary"

    def _candidate_models(self, primary_model: str) -> list[str]:
        return [primary_model]

    def _chat_completion_with_retries(self, model: str, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def _print_fallback_message(self, models: list[str], index: int) -> None:
        return None

    @staticmethod
    def _missing_key_action() -> dict[str, object]:
        return {
            "thought": "missing key",
            "tool": "ask_user",
            "args": {"question": "missing key"},
            "risk": "medium",
            "needs_user_confirmation": False,
            "new_facts": {},
        }

    @staticmethod
    def _provider_unavailable_action() -> dict[str, object]:
        return {
            "thought": "provider unavailable",
            "tool": "ask_user",
            "args": {"question": "provider unavailable"},
            "risk": "medium",
            "needs_user_confirmation": True,
            "new_facts": {},
        }


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _payload() -> dict[str, object]:
    return {"goal": "test", "facts": {}, "recent_history": [], "current_page": {"snapshot_yaml": ""}}


def test_json_mode_planner_parses_valid_action() -> None:
    content = json.dumps(
        {
            "thought": "Observe first.",
            "tool": "observe",
            "args": {},
            "risk": "low",
            "needs_user_confirmation": False,
            "new_facts": {},
        }
    )
    llm = FakeLLM([_response(content)])
    planner = JsonModePlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "observe"
    assert llm.calls[0]["response_format"] == {"type": "json_object"}
    assert "click_element" in planner.last_messages[0]["content"]


def test_json_mode_planner_invalid_json_returns_safe_fallback() -> None:
    llm = FakeLLM([_response("not json"), _response("still bad"), _response("{")])
    planner = JsonModePlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "ask_user"
    assert action.needs_user_confirmation is True
    assert "last_parser_error" in action.new_facts


def test_json_mode_planner_append_tool_result_is_noop() -> None:
    content = json.dumps(
        {
            "thought": "Observe first.",
            "tool": "observe",
            "args": {},
            "risk": "low",
            "needs_user_confirmation": False,
            "new_facts": {},
        }
    )
    llm = FakeLLM([_response(content)])
    planner = JsonModePlanner(llm_client=llm, registry=TOOL_REGISTRY)

    planner.append_tool_result(
        PlannerAction(tool="observe", args={}, native_tool_call_id="call_json"),
        {"ok": True, "message": "observed", "data": {}},
    )
    action = planner.plan(_payload())

    assert action.tool == "observe"
    assert len(llm.calls) == 1
