from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from agent.prompts import SUBAGENT_PROMPT, SYSTEM_PROMPT
from agent.tools import TOOL_DESCRIPTIONS


ToolName = Literal[
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
]


class PlannerAction(BaseModel):
    thought: str
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"
    needs_user_confirmation: bool = False
    new_facts: dict[str, Any] = Field(default_factory=dict)


class LLMClient:
    def __init__(self) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = os.getenv("MODEL", "openai/gpt-oss-120b:free")
        self.verifier_model = os.getenv("MODEL_VERIFIER", self.model) or self.model
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-OpenRouter-Title": "browser-agent-mvp",
                },
            )

    def plan(self, prompt_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.client:
            return self._missing_key_action()

        user_prompt = (
            "Current compact memory and page observation JSON:\n"
            f"{json.dumps(prompt_payload, ensure_ascii=False)}\n\n"
            "Choose the single next action. Return JSON only."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_error = ""
        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            try:
                parsed = self._parse_action(content)
                return parsed.model_dump()
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Invalid JSON or invalid schema. "
                            f"Parser error: {last_error}. Return corrected strict JSON only."
                        ),
                    }
                )

        return {
            "thought": "The model returned invalid JSON and the agent cannot safely continue.",
            "tool": "ask_user",
            "args": {"question": "The LLM returned invalid JSON twice. What should I do next?"},
            "risk": "medium",
            "needs_user_confirmation": True,
            "new_facts": {"last_parser_error": last_error},
        }

    def query_page(self, observation: dict[str, Any], question: str) -> dict[str, Any]:
        if not self.client:
            return {
                "ok": False,
                "message": "OpenRouter API key is not configured.",
                "data": {"answer": "Set OPENROUTER_API_KEY in .env to enable query_page."},
            }

        payload = {
            "url": observation.get("url", ""),
            "title": observation.get("title", ""),
            "snapshot_yaml": observation.get("snapshot_yaml", ""),
            "body_text_excerpt": observation.get("body_text", "")[:6000],
            "question": question,
        }
        response = self.client.chat.completions.create(
            model=self.verifier_model,
            messages=[
                {"role": "system", "content": SUBAGENT_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content or ""
        return {"ok": True, "message": "page query answered", "data": {"answer": answer.strip()}}

    @staticmethod
    def _parse_action(content: str) -> PlannerAction:
        text = strip_json_fences(content)
        data = json.loads(text)
        if data.get("tool") not in TOOL_DESCRIPTIONS:
            raise ValueError(f"Unknown tool: {data.get('tool')}")
        return PlannerAction.model_validate(data)

    @staticmethod
    def _missing_key_action() -> dict[str, Any]:
        return {
            "thought": "OpenRouter is not configured, so I need the user to add an API key before planning.",
            "tool": "ask_user",
            "args": {"question": "OPENROUTER_API_KEY is missing in .env. Add it, then rerun the agent."},
            "risk": "medium",
            "needs_user_confirmation": False,
            "new_facts": {},
        }


def strip_json_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text

