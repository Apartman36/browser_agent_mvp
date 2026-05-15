from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from agent.prompts import build_json_system_prompt
from agent.tool_registry import TOOL_REGISTRY, ToolRegistry

from .base import BasePlanner, PlannerAction


class JsonModePlanner(BasePlanner):
    def __init__(self, llm_client: Any, registry: ToolRegistry | None = None) -> None:
        self.llm_client = llm_client
        self.registry = registry or TOOL_REGISTRY
        self.last_messages: list[dict[str, str]] = []

    def plan(self, memory_payload: dict[str, Any]) -> PlannerAction:
        if not self.llm_client.client:
            return PlannerAction.model_validate(self.llm_client._missing_key_action())

        user_prompt = (
            "Current compact memory and page observation JSON:\n"
            f"{json.dumps(memory_payload, ensure_ascii=False)}\n\n"
            "Choose the single next action. Return JSON only."
        )
        messages = [
            {"role": "system", "content": build_json_system_prompt(self.registry)},
            {"role": "user", "content": user_prompt},
        ]
        self.last_messages = [dict(message) for message in messages]

        models = self.llm_client._candidate_models(self.llm_client.model)
        for index, model in enumerate(models):
            try:
                return self._plan_with_model(model, messages)
            except self.llm_client.provider_unavailable_error_type:
                self.llm_client._print_fallback_message(models, index)

        return PlannerAction.model_validate(self.llm_client._provider_unavailable_action())

    def _plan_with_model(self, model: str, messages: list[dict[str, str]]) -> PlannerAction:
        model_messages = [dict(message) for message in messages]
        last_error = ""
        for _ in range(3):
            response = self.llm_client._chat_completion_with_retries(
                model=model,
                messages=model_messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            try:
                return self._parse_action(content)
            except (json.JSONDecodeError, ValidationError, TypeError, ValueError, KeyError) as exc:
                last_error = str(exc)
                model_messages.append({"role": "assistant", "content": content})
                model_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Invalid JSON or invalid tool arguments. "
                            f"Parser error: {last_error}. Return corrected strict JSON only."
                        ),
                    }
                )

        return PlannerAction(
            thought="The model returned invalid JSON or invalid tool arguments, so I need user guidance.",
            tool="ask_user",
            args={"question": "The LLM returned invalid JSON or invalid tool arguments. What should I do next?"},
            risk="medium",
            needs_user_confirmation=True,
            new_facts={"last_parser_error": last_error},
        )

    def _parse_action(self, content: str) -> PlannerAction:
        from agent.llm import strip_json_fences

        text = strip_json_fences(content)
        data = json.loads(text)
        tool_name = str(data.get("tool", ""))
        self.registry.get(tool_name)
        validated_args = self.registry.validate_args(tool_name, data.get("args") or {})
        data["args"] = validated_args.model_dump()
        return PlannerAction.model_validate(data)
