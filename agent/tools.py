from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


TOOL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "goto": {
        "kind": "mutating",
        "args": '{"url": "string"}',
        "description": "Navigate the current tab to a URL.",
    },
    "observe": {
        "kind": "read-only",
        "args": "{}",
        "description": "Refresh the current page observation.",
    },
    "query_page": {
        "kind": "read-only",
        "args": '{"question": "string"}',
        "description": "Ask a DOM/page analyst sub-agent a compact question about the current page.",
    },
    "click_element": {
        "kind": "mutating",
        "args": '{"ref": "string"}',
        "description": "Click one element by its current Playwright ARIA ref.",
    },
    "type_text": {
        "kind": "mutating",
        "args": '{"ref": "string", "text": "string", "submit": "boolean", "clear": "boolean"}',
        "description": "Fill or type text into an element by current ARIA ref, optionally pressing Enter.",
    },
    "press_key": {
        "kind": "mutating",
        "args": '{"key": "string"}',
        "description": "Press a keyboard key in the current browser context.",
    },
    "scroll": {
        "kind": "mutating",
        "args": '{"direction": "up|down"}',
        "description": "Scroll the visible page up or down.",
    },
    "wait": {
        "kind": "read-only",
        "args": '{"ms": "integer"}',
        "description": "Wait for a bounded number of milliseconds.",
    },
    "screenshot": {
        "kind": "read-only",
        "args": '{"full_page": "boolean"}',
        "description": "Save a screenshot under logs/screenshots.",
    },
    "extract_text": {
        "kind": "read-only",
        "args": '{"ref": "string|null"}',
        "description": "Extract visible text from a specific current ref or from the full page.",
    },

    "extract_dom": {
        "kind": "read-only",
        "args": '{"selector": "string"}',
        "description": "Extract raw DOM/HTML for a given CSS selector.",
    },
    "extract_css": {
        "kind": "read-only",
        "args": '{"selector": "string", "property": "string"}',
        "description": "Extract computed CSS property value for a given CSS selector.",
    },
    "dismiss_popup": {
        "kind": "mutating",
        "args": '{}',
        "description": "Attempt to auto-dismiss common pop-ups or overlays.",
    },
    "ask_user": {
        "kind": "read-only",
        "args": '{"question": "string"}',
        "description": "Ask the human for missing information or explicit confirmation.",
    },
    "done": {
        "kind": "read-only",
        "args": '{"summary": "string", "status": "success|failed|stopped_by_user"}',
        "description": "Finish the run and produce the final report.",
    },
}


def format_tool_descriptions() -> str:
    lines = []
    for name, meta in TOOL_DESCRIPTIONS.items():
        lines.append(f"- {name} ({meta['kind']}): args {meta['args']}. {meta['description']}")
    return "\n".join(lines)


class ToolDispatcher:
    def __init__(self, browser: Any, llm_client: Any, console: Console | None = None) -> None:
        self.browser = browser
        self.llm_client = llm_client
        self.console = console or Console()
        self.current_obs: dict[str, Any] | None = None

    def set_observation(self, obs: dict[str, Any]) -> None:
        self.current_obs = obs

    def dispatch(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = action.get("tool")
        args = action.get("args") or {}

        try:
            if tool == "goto":
                return self.browser.goto(str(args.get("url", "")))
            if tool == "observe":
                obs = self.browser.observe()
                self.current_obs = obs
                return {"ok": obs.get("ok", False), "message": "observed page", "data": obs}
            if tool == "query_page":
                return self.query_page(str(args.get("question", "")))
            if tool == "click_element":
                return self.browser.click_element(str(args.get("ref", "")))
            if tool == "type_text":
                return self.browser.type_text(
                    ref=str(args.get("ref", "")),
                    text=str(args.get("text", "")),
                    submit=bool(args.get("submit", False)),
                    clear=bool(args.get("clear", True)),
                )
            if tool == "press_key":
                return self.browser.press_key(str(args.get("key", "")))
            if tool == "scroll":
                return self.browser.scroll(str(args.get("direction", "down")))
            if tool == "wait":
                return self.browser.wait(int(args.get("ms", 1000)))
            if tool == "screenshot":
                return self.browser.screenshot(bool(args.get("full_page", False)))
            if tool == "extract_text":
                return self.browser.extract_text(args.get("ref"))

            if tool == "extract_dom":
                return self.browser.extract_dom(str(args.get("selector", "body")))
            if tool == "extract_css":
                return self.browser.extract_css(str(args.get("selector", "")), str(args.get("property", "")))
            if tool == "dismiss_popup":
                return self.browser.dismiss_popup()
            if tool == "ask_user":
                return self.ask_user(str(args.get("question", "Continue?")))
            if tool == "done":
                return {
                    "ok": True,
                    "message": "agent finished",
                    "data": {
                        "summary": str(args.get("summary", "")),
                        "status": str(args.get("status", "success")),
                    },
                }
            return {"ok": False, "message": f"unknown tool: {tool}", "data": {}}
        except Exception as exc:
            return {"ok": False, "message": f"tool dispatch failed: {tool}", "data": {"error": str(exc)}}

    def query_page(self, question: str) -> dict[str, Any]:
        self.console.print("[bold cyan]DOM Sub-agent:[/bold cyan] Processing query...")
        self.console.print(f"[dim]Question:[/dim] {escape(question)}")
        if not self.current_obs:
            return {"ok": False, "message": "no current observation available", "data": {}}
        answer = self.llm_client.query_page(self.current_obs, question)
        if answer.get("ok"):
            self.console.print(f"[dim]Answer:[/dim] {escape(str(answer['data'].get('answer', '')))}")
        return answer

    def ask_user(self, question: str) -> dict[str, Any]:
        # input() doesn't support rich markup, but we print it via console first to be safe and consistent
        self.console.print(f"[bold]Question:[/bold] {escape(question)}")
        from agent.core import wait_for_user_input
        answer = wait_for_user_input(question)
        return {"ok": True, "message": "user answered", "data": {"answer": answer}}





def get_tools_schema() -> list[dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": "execute_action",
            "description": "Execute the next browser action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "One short sentence explaining the next action"
                    },
                    "tool": {
                        "type": "string",
                        "enum": list(TOOL_DESCRIPTIONS.keys())
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments for the tool (values must be strings, booleans, or integers as required)"
                    },
                    "risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"]
                    },
                    "needs_user_confirmation": {
                        "type": "boolean"
                    },
                    "new_facts": {
                        "type": "object",
                        "description": "Any new facts discovered"
                    }
                },
                "required": ["thought", "tool", "args", "risk", "needs_user_confirmation", "new_facts"],
                "additionalProperties": False
            }
        }
    }]
