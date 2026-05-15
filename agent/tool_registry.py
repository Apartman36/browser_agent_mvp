from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ActionCategory(str, Enum):
    READ_ONLY = "read_only"
    NAVIGATION = "navigation"
    REVERSIBLE = "reversible"
    MUTATING = "mutating"
    IRREVERSIBLE = "irreversible"
    SENSITIVE = "sensitive"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StrictArgsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GotoArgs(StrictArgsModel):
    url: str = Field(..., description="HTTP or HTTPS URL to open in the current tab.")


class ObserveArgs(StrictArgsModel):
    pass


class QueryPageArgs(StrictArgsModel):
    question: str = Field(..., description="Compact question about the current page observation.")


class ClickElementArgs(StrictArgsModel):
    ref: str = Field(..., description="Current ARIA ref from the latest observation, for example e7.")


class TypeTextArgs(StrictArgsModel):
    ref: str = Field(..., description="Current ARIA ref from the latest observation.")
    text: str = Field(..., description="Text to enter into the element.")
    submit: bool = Field(False, description="Press Enter after typing when true.")
    clear: bool = Field(True, description="Clear the field before typing when true.")


class PressKeyArgs(StrictArgsModel):
    key: str = Field(..., description="Keyboard key name supported by Playwright, for example Enter.")


class ScrollArgs(StrictArgsModel):
    direction: Literal["up", "down"]


class WaitArgs(StrictArgsModel):
    ms: int = Field(..., ge=0, le=30000, description="Milliseconds to wait, bounded to 30 seconds.")


class ScreenshotArgs(StrictArgsModel):
    full_page: bool = False


class ExtractTextArgs(StrictArgsModel):
    ref: str | None = Field(
        None,
        description="Current ARIA ref to extract from, or null to extract visible page text.",
    )

    @field_validator("ref", mode="before")
    @classmethod
    def normalize_empty_ref(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        return text


class AskUserArgs(StrictArgsModel):
    question: str


class DoneArgs(StrictArgsModel):
    status: Literal["success", "failed", "stopped_by_user"]
    summary: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    category: ActionCategory
    default_risk: RiskLevel
    read_only: bool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": self._strict_schema(spec.args_model),
                },
            }
            for spec in self.all()
        ]

    def prompt_block(self) -> str:
        lines: list[str] = []
        for spec in self.all():
            schema = self._strict_schema(spec.args_model)
            compact_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            access = "read-only" if spec.read_only else "state-changing"
            lines.append(
                f"- {spec.name} ({spec.category.value}, default risk {spec.default_risk.value}, {access}): "
                f"{spec.description} Args schema: {compact_schema}"
            )
        return "\n".join(lines)

    def validate_args(self, tool_name: str, raw_args: Any) -> BaseModel:
        spec = self.get(tool_name)
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise TypeError(f"Args for {tool_name} must be an object.")
        return spec.args_model.model_validate(raw_args)

    @staticmethod
    def _strict_schema(model: type[BaseModel]) -> dict[str, Any]:
        schema = model.model_json_schema()
        return _disallow_additional_properties(schema)


def _disallow_additional_properties(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "object":
            value.setdefault("additionalProperties", False)
        for child in value.values():
            _disallow_additional_properties(child)
    elif isinstance(value, list):
        for child in value:
            _disallow_additional_properties(child)
    return value


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for spec in [
        ToolSpec(
            name="goto",
            description="Navigate the current tab to an HTTP or HTTPS URL. Do not use javascript, file, or data URLs.",
            args_model=GotoArgs,
            category=ActionCategory.NAVIGATION,
            default_risk=RiskLevel.MEDIUM,
            read_only=False,
        ),
        ToolSpec(
            name="observe",
            description="Refresh the current Playwright ARIA accessibility observation.",
            args_model=ObserveArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
        ToolSpec(
            name="query_page",
            description="Ask a compact page-analysis sub-agent question about the current ARIA snapshot.",
            args_model=QueryPageArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
        ToolSpec(
            name="click_element",
            description=(
                "Click one element by its current ARIA ref from the latest observation only. "
                "Never invent refs or reuse stale refs."
            ),
            args_model=ClickElementArgs,
            category=ActionCategory.REVERSIBLE,
            default_risk=RiskLevel.MEDIUM,
            read_only=False,
        ),
        ToolSpec(
            name="type_text",
            description=(
                "Fill or type text into one element by its current ARIA ref from the latest observation only. "
                "Set submit=true only when pressing Enter is intended."
            ),
            args_model=TypeTextArgs,
            category=ActionCategory.REVERSIBLE,
            default_risk=RiskLevel.MEDIUM,
            read_only=False,
        ),
        ToolSpec(
            name="press_key",
            description="Press a keyboard key in the current browser context.",
            args_model=PressKeyArgs,
            category=ActionCategory.REVERSIBLE,
            default_risk=RiskLevel.MEDIUM,
            read_only=False,
        ),
        ToolSpec(
            name="scroll",
            description="Scroll the visible page up or down.",
            args_model=ScrollArgs,
            category=ActionCategory.NAVIGATION,
            default_risk=RiskLevel.LOW,
            read_only=False,
        ),
        ToolSpec(
            name="wait",
            description="Wait for a bounded number of milliseconds.",
            args_model=WaitArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
        ToolSpec(
            name="screenshot",
            description="Save a screenshot under logs/screenshots.",
            args_model=ScreenshotArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
        ToolSpec(
            name="extract_text",
            description=(
                "Extract visible text from a current ARIA ref, or from the full page when ref is null. "
                "Refs must come from the latest observation."
            ),
            args_model=ExtractTextArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
        ToolSpec(
            name="ask_user",
            description="Ask the human for missing information, credentials, or explicit confirmation.",
            args_model=AskUserArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
        ToolSpec(
            name="done",
            description="Finish the run and produce the final report.",
            args_model=DoneArgs,
            category=ActionCategory.READ_ONLY,
            default_risk=RiskLevel.LOW,
            read_only=True,
        ),
    ]:
        registry.register(spec)
    return registry


TOOL_REGISTRY = default_tool_registry()
