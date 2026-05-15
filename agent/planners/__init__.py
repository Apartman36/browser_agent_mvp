from __future__ import annotations

from typing import Any

from agent.config import AgentConfig
from agent.tool_registry import ToolRegistry

from .base import BasePlanner, PlannerAction
from .json_mode import JsonModePlanner
from .native_tools import NativeToolPlanner


def create_planner(config: AgentConfig, llm_client: Any, registry: ToolRegistry) -> BasePlanner:
    json_planner = JsonModePlanner(llm_client=llm_client, registry=registry)
    if config.planner_mode == "json" or config.native_tool_mode_disabled:
        return json_planner
    if config.planner_mode == "native_tools":
        return NativeToolPlanner(llm_client=llm_client, registry=registry)
    return NativeToolPlanner(llm_client=llm_client, registry=registry, fallback_planner=json_planner)


__all__ = [
    "BasePlanner",
    "JsonModePlanner",
    "NativeToolPlanner",
    "PlannerAction",
    "create_planner",
]
