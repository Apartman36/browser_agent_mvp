from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_PLANNER_MODE = "auto"


@dataclass(frozen=True)
class AgentConfig:
    openrouter_api_key: str
    model: str
    model_fallbacks: list[str]
    paid_fallback_model: str
    verifier_model: str
    planner_mode: str
    use_llm_risk_classifier: bool
    max_steps: int
    start_url: str
    native_tool_mode_disabled: bool


def load_config() -> AgentConfig:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=True)

    model = os.getenv("MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    planner_mode = (os.getenv("PLANNER_MODE", DEFAULT_PLANNER_MODE) or DEFAULT_PLANNER_MODE).strip().lower()
    if planner_mode not in {"auto", "native_tools", "json"}:
        planner_mode = DEFAULT_PLANNER_MODE

    return AgentConfig(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model=model,
        model_fallbacks=_parse_csv(os.getenv("MODEL_FALLBACKS", "")),
        paid_fallback_model=os.getenv("PAID_FALLBACK_MODEL", "").strip(),
        verifier_model=os.getenv("MODEL_VERIFIER", model) or model,
        planner_mode=planner_mode,
        use_llm_risk_classifier=_parse_bool(os.getenv("USE_LLM_RISK_CLASSIFIER", "false")),
        max_steps=_parse_int(os.getenv("MAX_STEPS", "25"), default=25),
        start_url=os.getenv("START_URL", ""),
        native_tool_mode_disabled=_parse_bool(os.getenv("NATIVE_TOOL_MODE_DISABLED", "false")),
    )


def _parse_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(raw_value: str, default: int) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default
