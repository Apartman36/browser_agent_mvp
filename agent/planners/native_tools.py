from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from agent.prompts import NATIVE_SYSTEM_PROMPT
from agent.tool_registry import TOOL_REGISTRY, ToolRegistry

from .base import BasePlanner, PlannerAction


class NativeToolPlanner(BasePlanner):
    MAX_NATIVE_HISTORY_MESSAGES = 8
    MAX_TOOL_RESULT_CHARS = 1200
    SECRET_KEY_RE = re.compile(r"(?i)(password|passwd|passcode|otp|2fa|cvv|card|token|api[_-]?key|secret|cookie|authorization)")
    SECRET_VALUE_RE = re.compile(
        r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]{8,}|api[_-]?key\s*[:=]\s*\S+|\b[0-9][0-9 -]{11,23}[0-9]\b)"
    )

    def __init__(
        self,
        llm_client: Any,
        registry: ToolRegistry | None = None,
        fallback_planner: BasePlanner | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.registry = registry or TOOL_REGISTRY
        self.fallback_planner = fallback_planner
        self.last_messages: list[dict[str, Any]] = []
        self._native_history: list[dict[str, Any]] = []

    def plan(self, memory_payload: dict[str, Any]) -> PlannerAction:
        if not self.llm_client.client:
            return PlannerAction.model_validate(self.llm_client._missing_key_action())

        current_user_message = self._build_user_message(memory_payload)
        messages = self._build_messages(current_user_message)
        self.last_messages = [dict(message) for message in messages]
        models = self.llm_client._candidate_models(self.llm_client.model)

        for index, model in enumerate(models):
            try:
                return self._plan_with_model(model, messages, current_user_message)
            except self.llm_client.provider_unavailable_error_type:
                self.llm_client._print_fallback_message(models, index)

        if self.fallback_planner is not None:
            return self.fallback_planner.plan(memory_payload)

        return PlannerAction.model_validate(self.llm_client._provider_unavailable_action())

    def append_tool_result(self, action: PlannerAction, result: dict[str, Any]) -> None:
        if not action.native_tool_call_id:
            return None

        self._native_history.append(
            {
                "role": "tool",
                "tool_call_id": action.native_tool_call_id,
                "content": self._serialize_tool_result(result),
            }
        )
        self._trim_native_history()
        return None

    def _build_user_message(self, memory_payload: dict[str, Any]) -> dict[str, str]:
        current_page = memory_payload.get("current_page", {}) or {}
        trusted_payload = {
            "goal": memory_payload.get("goal", ""),
            "facts": memory_payload.get("facts", {}),
            "recent_history": memory_payload.get("recent_history", []),
        }
        page_content = {
            "url": current_page.get("url", ""),
            "title": current_page.get("title", ""),
            "snapshot_yaml": current_page.get("snapshot_yaml", ""),
            "body_text_excerpt": current_page.get("body_text_excerpt", ""),
            "observation_error": current_page.get("observation_error"),
        }
        user_prompt = (
            "Trusted task state JSON:\n"
            f"{json.dumps(trusted_payload, ensure_ascii=False)}\n\n"
            '<page_content untrusted="true">\n'
            f"{json.dumps(page_content, ensure_ascii=False)}\n"
            "</page_content>\n\n"
            "Choose exactly one tool call for the next step."
        )
        return {"role": "user", "content": user_prompt}

    def _build_messages(self, current_user_message: dict[str, str]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
            *[dict(message) for message in self._native_history],
            current_user_message,
        ]

    def _plan_with_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        current_user_message: dict[str, str],
    ) -> PlannerAction:
        model_messages = [dict(message) for message in messages]
        last_error = ""
        for attempt in range(2):
            response = self.llm_client._chat_completion_with_retries(
                model=model,
                messages=model_messages,
                tools=self.registry.openai_tools(),
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=0.2,
            )
            message = response.choices[0].message
            try:
                return self._action_from_message(message, current_user_message)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                if attempt == 0:
                    model_messages.append({"role": "assistant", "content": self._message_content(message)})
                    model_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous tool call was invalid. "
                                f"Parser error: {last_error}. Choose one valid registered tool call."
                            ),
                        }
                    )
                    continue

        return PlannerAction(
            thought="The native tool call was invalid, so I need user guidance.",
            tool="ask_user",
            args={"question": "The LLM returned an invalid native tool call. What should I do next?"},
            risk="medium",
            needs_user_confirmation=True,
            new_facts={"last_native_tool_error": last_error},
        )

    def _action_from_message(self, message: Any, current_user_message: dict[str, str]) -> PlannerAction:
        tool_calls = list(self._get(message, "tool_calls") or [])
        content = self._message_content(message)
        if not tool_calls:
            return self._fallback_action_from_content(content)

        tool_call = tool_calls[0]
        function = self._get(tool_call, "function")
        if function is None:
            raise ValueError("Native tool call has no function payload.")
        tool_name = str(self._get(function, "name") or "")
        raw_arguments = self._get(function, "arguments")
        arguments = self._parse_arguments(raw_arguments)
        self.registry.get(tool_name)
        validated_args = self.registry.validate_args(tool_name, arguments)
        thought = content.strip() or f"Model selected tool {tool_name}."
        new_facts: dict[str, Any] = {}
        if len(tool_calls) > 1:
            new_facts["deferred_tool_calls"] = len(tool_calls) - 1
        self._append_assistant_tool_call_message(
            current_user_message=current_user_message,
            content=content,
            tool_call=tool_call,
        )
        return PlannerAction(
            thought=thought,
            tool=tool_name,
            args=validated_args.model_dump(),
            risk=None,
            needs_user_confirmation=False,
            new_facts=new_facts,
            native_tool_call_id=str(self._get(tool_call, "id") or ""),
            raw=tool_call,
        )

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> dict[str, Any]:
        if raw_arguments is None or raw_arguments == "":
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if not isinstance(raw_arguments, str):
            raise TypeError("Native tool arguments must be a JSON string or object.")
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise TypeError("Native tool arguments must decode to an object.")
        return parsed

    @staticmethod
    def _fallback_action_from_content(content: str) -> PlannerAction:
        text = content.strip()
        lowered = text.lower()
        if text and any(marker in lowered for marker in ("done", "complete", "finished", "success")):
            return PlannerAction(
                thought="The model reported the task is complete without a tool call.",
                tool="done",
                args={"status": "success", "summary": text},
                risk="low",
                needs_user_confirmation=False,
                new_facts={},
            )
        return PlannerAction(
            thought="The model did not return a native tool call.",
            tool="ask_user",
            args={"question": "The LLM did not choose a tool. What should I do next?"},
            risk="medium",
            needs_user_confirmation=True,
            new_facts={"native_content": text[:500]},
        )

    @staticmethod
    def _message_content(message: Any) -> str:
        return str(NativeToolPlanner._get(message, "content") or "")

    @staticmethod
    def _get(value: Any, name: str) -> Any:
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    def _append_assistant_tool_call_message(
        self,
        *,
        current_user_message: dict[str, str],
        content: str,
        tool_call: Any,
    ) -> None:
        self._native_history.append(dict(current_user_message))
        self._native_history.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [self._tool_call_to_dict(tool_call)],
            }
        )
        self._trim_native_history()

    def _trim_native_history(self) -> None:
        while len(self._native_history) > self.MAX_NATIVE_HISTORY_MESSAGES:
            if len(self._native_history) >= 3 and [message.get("role") for message in self._native_history[:3]] == [
                "user",
                "assistant",
                "tool",
            ]:
                del self._native_history[:3]
            else:
                del self._native_history[0]

    def _tool_call_to_dict(self, tool_call: Any) -> dict[str, Any]:
        function = self._get(tool_call, "function") or {}
        raw_arguments = self._get(function, "arguments")
        if isinstance(raw_arguments, str):
            arguments = raw_arguments
        else:
            arguments = json.dumps(raw_arguments or {}, ensure_ascii=False)
        return {
            "id": str(self._get(tool_call, "id") or ""),
            "type": str(self._get(tool_call, "type") or "function"),
            "function": {
                "name": str(self._get(function, "name") or ""),
                "arguments": arguments,
            },
        }

    def _serialize_tool_result(self, result: dict[str, Any]) -> str:
        reason = str((result.get("data") or {}).get("reason") or result.get("message") or "").strip()
        message = str(result.get("message") or "")
        if message == "blocked by safety policy":
            return self._truncate(f"BLOCKED: {reason}")
        if message == "user declined high-risk action":
            return self._truncate(f"USER_DENIED: {reason}")

        compact = {
            "ok": bool(result.get("ok")),
            "message": result.get("message", ""),
            "data": self._compact_result_data(result.get("data", {})),
        }
        return self._truncate(json.dumps(compact, ensure_ascii=False, default=str, separators=(",", ":")))

    def _compact_result_data(self, value: Any) -> Any:
        if isinstance(value, dict):
            compact: dict[str, Any] = {}
            for key, child in value.items():
                key_text = str(key)
                if self.SECRET_KEY_RE.search(key_text):
                    compact[key_text] = "[redacted]"
                elif key_text in {"snapshot_yaml", "body_text", "body_text_excerpt"}:
                    compact[key_text] = self._truncate(str(child), limit=500)
                else:
                    compact[key_text] = self._compact_result_data(child)
            return compact
        if isinstance(value, list):
            return [self._compact_result_data(item) for item in value[:10]]
        if isinstance(value, str):
            if self._string_looks_sensitive(value):
                return "[redacted]"
            return self._truncate(value, limit=500)
        return value

    @classmethod
    def _string_looks_sensitive(cls, value: str) -> bool:
        if cls.SECRET_VALUE_RE.search(value):
            return True
        stripped = value.strip()
        if re.fullmatch(r"[A-Za-z0-9._~+/=-]{24,}", stripped):
            has_digit = bool(re.search(r"\d", stripped))
            has_symbol = bool(re.search(r"[._~+/=-]", stripped))
            has_mixed_case = bool(re.search(r"[a-z]", stripped)) and bool(re.search(r"[A-Z]", stripped))
            return sum([has_digit, has_symbol, has_mixed_case]) >= 2
        return False

    @classmethod
    def _truncate(cls, value: str, limit: int | None = None) -> str:
        max_chars = limit or cls.MAX_TOOL_RESULT_CHARS
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 15] + "...[truncated]"
