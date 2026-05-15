from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
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

DEFAULT_MODEL = "google/gemma-4-31b-it:free"
PROVIDER_ERROR_TYPES = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)
RETRY_BACKOFF_SECONDS = (2.0, 5.0)
MAX_RETRY_AFTER_SECONDS = 10.0


class PlannerAction(BaseModel):
    thought: str
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    risk: Literal["low", "medium", "high"] = "low"
    needs_user_confirmation: bool = False
    new_facts: dict[str, Any] = Field(default_factory=dict)


class ProviderUnavailableError(Exception):
    def __init__(self, model: str, original: Exception) -> None:
        super().__init__(str(original))
        self.model = model
        self.original = original


class LLMClient:
    def __init__(self) -> None:
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model = os.getenv("MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        self.model_fallbacks = self._parse_model_fallbacks(
            os.getenv("MODEL_FALLBACKS", "")
        )
        self.verifier_model = os.getenv("MODEL_VERIFIER", self.model) or self.model
        self._sleep = time.sleep
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

        models = self._candidate_models(self.model)
        for index, model in enumerate(models):
            try:
                return self._plan_with_model(model, messages)
            except ProviderUnavailableError:
                self._print_fallback_message(models, index)

        return self._provider_unavailable_action()

    def query_page(self, observation: dict[str, Any], question: str) -> dict[str, Any]:
        if not self.client:
            return {
                "ok": False,
                "message": "OpenRouter API key is not configured.",
                "data": {
                    "answer": "Set OPENROUTER_API_KEY in .env to enable query_page."
                },
            }

        payload = {
            "url": observation.get("url", ""),
            "title": observation.get("title", ""),
            "snapshot_yaml": observation.get("snapshot_yaml", ""),
            "body_text_excerpt": observation.get("body_text", "")[:6000],
            "question": question,
        }
        messages = [
            {"role": "system", "content": SUBAGENT_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        models = self._candidate_models(self.verifier_model)
        for index, model in enumerate(models):
            try:
                response = self._chat_completion_with_retries(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
                answer = response.choices[0].message.content or ""
                return {
                    "ok": True,
                    "message": "page query answered",
                    "data": {"answer": answer.strip()},
                }
            except ProviderUnavailableError:
                self._print_fallback_message(models, index)

        return {
            "ok": False,
            "message": "LLM provider temporarily unavailable",
            "data": {
                "answer": (
                    "LLM provider error or rate limit after retries. "
                    "Try again, or switch MODEL/MODEL_VERIFIER in .env and rerun."
                )
            },
        }

    def _plan_with_model(
        self, model: str, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        model_messages = [dict(message) for message in messages]
        last_error = ""
        for _ in range(3):
            response = self._chat_completion_with_retries(
                model=model,
                messages=model_messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            try:
                parsed = self._parse_action(content)
                return parsed.model_dump()
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)
                model_messages.append({"role": "assistant", "content": content})
                model_messages.append(
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
            "args": {
                "question": "The LLM returned invalid JSON twice. What should I do next?"
            },
            "risk": "medium",
            "needs_user_confirmation": True,
            "new_facts": {"last_parser_error": last_error},
        }

    def _chat_completion_with_retries(self, model: str, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        max_attempts = len(RETRY_BACKOFF_SECONDS) + 1

        for attempt in range(max_attempts):
            try:
                return self.client.chat.completions.create(model=model, **kwargs)
            except PROVIDER_ERROR_TYPES as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    break
                delay = self._retry_delay_seconds(exc, attempt)
                print(
                    f"LLM provider warning for {model}: "
                    f"{self._format_provider_error(exc)}; retrying in {delay:g}s..."
                )
                self._sleep(delay)

        assert last_error is not None
        print(
            f"LLM provider failed for {model}: {self._format_provider_error(last_error)}"
        )
        raise ProviderUnavailableError(model, last_error)

    def _candidate_models(self, primary_model: str) -> list[str]:
        models: list[str] = []
        seen: set[str] = set()
        for model in [primary_model, *self.model_fallbacks]:
            model = model.strip()
            if model and model not in seen:
                models.append(model)
                seen.add(model)
        return models

    def _print_fallback_message(self, models: list[str], index: int) -> None:
        if index + 1 < len(models):
            print(
                f"LLM provider failed for {models[index]}, trying fallback {models[index + 1]}..."
            )

    @staticmethod
    def _parse_model_fallbacks(raw_value: str) -> list[str]:
        return [model.strip() for model in raw_value.split(",") if model.strip()]

    @staticmethod
    def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
        retry_after = LLMClient._retry_after_seconds(exc)
        if retry_after is not None:
            return retry_after
        return RETRY_BACKOFF_SECONDS[attempt]

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None

        value = headers.get("retry-after")
        if not value:
            return None

        try:
            seconds = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()

        return max(0.0, min(seconds, MAX_RETRY_AFTER_SECONDS))

    @staticmethod
    def _format_provider_error(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        status_code = getattr(exc, "status_code", None) or getattr(
            response, "status_code", None
        )
        message = " ".join(str(exc).split())
        if len(message) > 220:
            message = message[:217] + "..."
        if status_code:
            return f"{exc.__class__.__name__} {status_code}: {message}"
        return f"{exc.__class__.__name__}: {message}"

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
            "args": {
                "question": "OPENROUTER_API_KEY is missing in .env. Add it, then rerun the agent."
            },
            "risk": "medium",
            "needs_user_confirmation": False,
            "new_facts": {},
        }

    @staticmethod
    def _provider_unavailable_action() -> dict[str, Any]:
        return {
            "thought": "The LLM provider is temporarily unavailable, so I need user guidance instead of crashing.",
            "tool": "ask_user",
            "args": {
                "question": (
                    "The LLM provider returned an error or rate limit. Type 'retry' to try again, "
                    "'stop' to stop, or switch MODEL in .env and rerun."
                )
            },
            "risk": "medium",
            "needs_user_confirmation": True,
            "new_facts": {},
        }


def strip_json_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text[3:]
        i = 0
        while i < len(text) and (text[i].isalnum() or text[i] in "_-"):
            i += 1
        text = text[i:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
        text = text.strip()

    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return text
