from __future__ import annotations

import json

from agent_ui.sse import format_sse


def test_format_sse_returns_valid_frame() -> None:
    frame = format_sse("tool_result", {"ok": True, "message": "done"})

    assert frame.startswith("event: tool_result\n")
    assert frame.endswith("\n\n")
    data_line = frame.splitlines()[1]
    assert data_line.startswith("data: ")
    assert json.loads(data_line.removeprefix("data: ")) == {"ok": True, "message": "done"}


def test_format_sse_json_encodes_newlines_safely() -> None:
    frame = format_sse("observation", {"text": "hello\n\nevent: injected"})

    assert "\nevent: injected" not in frame
    data = json.loads(frame.splitlines()[1].removeprefix("data: "))
    assert data["text"] == "hello\n\nevent: injected"


def test_format_sse_strips_newlines_from_event_type() -> None:
    frame = format_sse("done\nretry", {"ok": True})

    assert frame.startswith("event: doneretry\n")
