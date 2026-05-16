from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.tool_registry import TOOL_REGISTRY


EXPECTED_TOOLS = {
    "goto",
    "observe",
    "query_page",
    "click_element",
    "type_text",
    "press_key",
    "scroll",
    "wait",
    "screenshot",
    "extract_text",
    "ask_user",
    "done",
}


def test_registry_contains_all_browser_tools() -> None:
    assert {spec.name for spec in TOOL_REGISTRY.all()} == EXPECTED_TOOLS


def test_openai_tools_are_chat_completion_compatible() -> None:
    tools = TOOL_REGISTRY.openai_tools()

    assert len(tools) == 12
    click_tool = next(tool for tool in tools if tool["function"]["name"] == "click_element")
    assert click_tool["type"] == "function"
    assert "ARIA ref" in click_tool["function"]["description"]
    assert click_tool["function"]["parameters"]["type"] == "object"
    assert click_tool["function"]["parameters"]["additionalProperties"] is False
    assert "ref" in click_tool["function"]["parameters"]["properties"]


def test_prompt_block_includes_registered_tools() -> None:
    block = TOOL_REGISTRY.prompt_block()

    assert "click_element" in block
    assert "type_text" in block
    assert "current ARIA ref" in block


def test_validate_args_normalizes_extract_text_empty_refs() -> None:
    args = TOOL_REGISTRY.validate_args("extract_text", {"ref": "null"})

    assert args.model_dump() == {"ref": None}


def test_invalid_args_fail_validation() -> None:
    with pytest.raises(ValidationError):
        TOOL_REGISTRY.validate_args("scroll", {"direction": "left"})
