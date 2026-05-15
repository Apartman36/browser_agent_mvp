from __future__ import annotations

import json
from typing import Any

from rich.console import Console

from agent.tool_registry import TOOL_REGISTRY


TOOL_DESCRIPTIONS: dict[str, dict[str, str]] = {
    spec.name: {
        "kind": "read-only" if spec.read_only else spec.category.value,
        "args": json.dumps(spec.args_model.model_json_schema().get("properties", {}), ensure_ascii=False),
        "description": spec.description,
    }
    for spec in TOOL_REGISTRY.all()
}


def format_tool_descriptions() -> str:
    return TOOL_REGISTRY.prompt_block()


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
        self.console.print(f"[dim]Question:[/dim] {question}")
        if not self.current_obs:
            return {"ok": False, "message": "no current observation available", "data": {}}
        answer = self.llm_client.query_page(self.current_obs, question)
        if answer.get("ok"):
            self.console.print(f"[dim]Answer:[/dim] {answer['data'].get('answer', '')}")
        return answer

    @staticmethod
    def ask_user(question: str) -> dict[str, Any]:
        answer = input(f"{question}\n> ")
        return {"ok": True, "message": "user answered", "data": {"answer": answer}}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)
