from __future__ import annotations

from types import SimpleNamespace

from agent.planners.native_tools import NativeToolPlanner
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
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

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


def _response(*, content: str = "", tool_calls: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tool_call(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def _payload() -> dict[str, object]:
    return {
        "goal": "test",
        "facts": {},
        "recent_history": [],
        "current_page": {
            "url": "https://example.test",
            "title": "Example",
            "snapshot_yaml": '- button "Search" [ref=e1]',
            "body_text_excerpt": "Search",
        },
    }


def test_native_tool_planner_returns_click_action() -> None:
    llm = FakeLLM([_response(tool_calls=[_tool_call("click_element", '{"ref":"e1"}')])])
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "click_element"
    assert action.args == {"ref": "e1"}
    assert action.native_tool_call_id == "call_1"
    assert llm.calls[0]["tools"] == TOOL_REGISTRY.openai_tools()
    assert llm.calls[0]["tool_choice"] == "auto"
    assert llm.calls[0]["parallel_tool_calls"] is False
    assert planner._native_history[-1]["role"] == "assistant"
    assert planner._native_history[-1]["tool_calls"][0]["id"] == "call_1"


def test_native_tool_planner_appends_tool_result_with_matching_call_id() -> None:
    llm = FakeLLM([_response(tool_calls=[_tool_call("click_element", '{"ref":"e1"}')])])
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)
    action = planner.plan(_payload())

    planner.append_tool_result(action, {"ok": True, "message": "clicked element", "data": {"ref": "e1"}})

    tool_message = planner._native_history[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert "clicked element" in tool_message["content"]


def test_native_tool_planner_next_request_includes_prior_tool_result() -> None:
    llm = FakeLLM(
        [
            _response(tool_calls=[_tool_call("click_element", '{"ref":"e1"}', call_id="call_1")]),
            _response(tool_calls=[_tool_call("observe", "{}", call_id="call_2")]),
        ]
    )
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)
    first_action = planner.plan(_payload())
    planner.append_tool_result(first_action, {"ok": True, "message": "clicked element", "data": {"ref": "e1"}})

    second_action = planner.plan(_payload())

    second_messages = llm.calls[1]["messages"]
    prior_tool_messages = [message for message in second_messages if message.get("role") == "tool"]
    assert second_action.native_tool_call_id == "call_2"
    assert prior_tool_messages
    assert prior_tool_messages[-1]["tool_call_id"] == "call_1"
    assert second_messages[-1]["role"] == "user"


def test_native_tool_planner_records_blocked_and_denied_tool_results() -> None:
    llm = FakeLLM(
        [
            _response(tool_calls=[_tool_call("click_element", '{"ref":"e1"}', call_id="call_blocked")]),
            _response(tool_calls=[_tool_call("click_element", '{"ref":"e1"}', call_id="call_denied")]),
        ]
    )
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    blocked_action = planner.plan(_payload())
    planner.append_tool_result(
        blocked_action,
        {"ok": False, "message": "blocked by safety policy", "data": {"reason": "same action repeated"}},
    )
    assert planner._native_history[-1]["content"] == "BLOCKED: same action repeated"

    denied_action = planner.plan(_payload())
    planner.append_tool_result(
        denied_action,
        {"ok": False, "message": "user declined high-risk action", "data": {"reason": "target is Apply"}},
    )
    assert planner._native_history[-1]["content"] == "USER_DENIED: target is Apply"


def test_native_tool_planner_validates_type_text_args() -> None:
    llm = FakeLLM(
        [_response(tool_calls=[_tool_call("type_text", '{"ref":"e2","text":"hello","submit":true,"clear":false}')])]
    )
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "type_text"
    assert action.args == {"ref": "e2", "text": "hello", "submit": True, "clear": False}


def test_native_tool_planner_repairs_malformed_arguments_once() -> None:
    llm = FakeLLM(
        [
            _response(tool_calls=[_tool_call("click_element", "{bad")]),
            _response(tool_calls=[_tool_call("click_element", '{"ref":"e1"}')]),
        ]
    )
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "click_element"
    assert len(llm.calls) == 2


def test_native_tool_planner_unknown_tool_returns_safe_ask_user() -> None:
    llm = FakeLLM(
        [
            _response(tool_calls=[_tool_call("unknown", "{}")]),
            _response(tool_calls=[_tool_call("unknown", "{}")]),
        ]
    )
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "ask_user"
    assert action.needs_user_confirmation is True


def test_native_tool_planner_no_tool_call_final_content_becomes_done() -> None:
    llm = FakeLLM([_response(content="The task is complete.")])
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "done"
    assert action.args["status"] == "success"


def test_native_tool_planner_provider_exception_returns_safe_fallback() -> None:
    llm = FakeLLM([ProviderError("provider failed")])
    planner = NativeToolPlanner(llm_client=llm, registry=TOOL_REGISTRY)

    action = planner.plan(_payload())

    assert action.tool == "ask_user"
    assert action.needs_user_confirmation is True
