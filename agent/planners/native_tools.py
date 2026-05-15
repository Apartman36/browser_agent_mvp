from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from agent.prompts import NATIVE_SYSTEM_PROMPT
from agent.tool_registry import TOOL_REGISTRY, ToolRegistry

from .base import BasePlanner, PlannerAction


class NativeToolPlanner(BasePlanner):
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

    def plan(self, memory_payload: dict[str, Any]) -> PlannerAction:
        if not self.llm_client.client:
            return PlannerAction.model_validate(self.llm_client._missing_key_action())

        messages = self._build_messages(memory_payload)
        self.last_messages = [dict(message) for message in messages]
        models = self.llm_client._candidate_models(self.llm_client.model)

        for index, model in enumerate(models):
            try:
                return self._plan_with_model(model, messages)
            except self.llm_client.provider_unavailable_error_type:
                self.llm_client._print_fallback_message(models, index)

        if self.fallback_planner is not None:
            return self.fallback_planner.plan(memory_payload)

        return PlannerAction.model_validate(self.llm_client._provider_unavailable_action())

    def _build_messages(self, memory_payload: dict[str, Any]) -> list[dict[str, Any]]:
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
        return [
            {"role": "system", "content": NATIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _plan_with_model(self, model: str, messages: list[dict[str, Any]]) -> PlannerAction:
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
                return self._action_from_message(message)
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

    def _action_from_message(self, message: Any) -> PlannerAction:
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
