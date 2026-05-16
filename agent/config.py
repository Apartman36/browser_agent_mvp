from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = ""
DEFAULT_PLANNER_MODE = "auto"

BROWSER_MODE_CHROMIUM = "chromium"
BROWSER_MODE_CHROME_PROFILE = "chrome_profile"
ALLOWED_BROWSER_MODES = frozenset({BROWSER_MODE_CHROMIUM, BROWSER_MODE_CHROME_PROFILE})

DEFAULT_BROWSER_MODE = BROWSER_MODE_CHROMIUM
DEFAULT_CHROME_CHANNEL = "chrome"
DEFAULT_CHROMIUM_USER_DATA_DIR = ".pw_profile"
DEFAULT_CHROME_USER_DATA_DIR = ".chrome_agent_profile"
DEFAULT_RUN_LOG_ROOT = "logs/runs"


@dataclass(frozen=True)
class BrowserRuntimeConfig:
    """Browser runtime knobs read from env. Defaults preserve current behavior."""

    mode: str = DEFAULT_BROWSER_MODE
    chrome_channel: str = DEFAULT_CHROME_CHANNEL
    chromium_user_data_dir: str = DEFAULT_CHROMIUM_USER_DATA_DIR
    chrome_user_data_dir: str = DEFAULT_CHROME_USER_DATA_DIR
    slow_mo_ms: int = 0
    action_min_delay_ms: int = 0
    action_max_delay_ms: int = 0

    def active_user_data_dir(self) -> str:
        if self.mode == BROWSER_MODE_CHROME_PROFILE:
            return self.chrome_user_data_dir
        return self.chromium_user_data_dir

    def active_channel(self) -> str | None:
        if self.mode == BROWSER_MODE_CHROME_PROFILE:
            return self.chrome_channel
        return None


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
    browser: BrowserRuntimeConfig = field(default_factory=BrowserRuntimeConfig)
    run_log_root: str = DEFAULT_RUN_LOG_ROOT


def load_config() -> AgentConfig:
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

    model = (os.getenv("MODEL", DEFAULT_MODEL) or DEFAULT_MODEL).strip()
    planner_mode = (os.getenv("PLANNER_MODE", DEFAULT_PLANNER_MODE) or DEFAULT_PLANNER_MODE).strip().lower()
    if planner_mode not in {"auto", "native_tools", "json"}:
        planner_mode = DEFAULT_PLANNER_MODE

    return AgentConfig(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        model=model,
        model_fallbacks=_parse_csv(os.getenv("MODEL_FALLBACKS", "")),
        paid_fallback_model=os.getenv("PAID_FALLBACK_MODEL", "").strip(),
        verifier_model=(os.getenv("MODEL_VERIFIER", model) or model).strip(),
        planner_mode=planner_mode,
        use_llm_risk_classifier=_parse_bool(os.getenv("USE_LLM_RISK_CLASSIFIER", "false")),
        max_steps=_parse_int(os.getenv("MAX_STEPS", "25"), default=25),
        start_url=os.getenv("START_URL", ""),
        native_tool_mode_disabled=_parse_bool(os.getenv("NATIVE_TOOL_MODE_DISABLED", "false")),
        browser=load_browser_runtime_config(),
        run_log_root=(os.getenv("RUN_LOG_ROOT", DEFAULT_RUN_LOG_ROOT) or DEFAULT_RUN_LOG_ROOT).strip(),
    )


def load_browser_runtime_config() -> BrowserRuntimeConfig:
    """Read browser runtime env vars. Invalid mode falls back to chromium with a clear error."""

    raw_mode = (os.getenv("BROWSER_MODE", DEFAULT_BROWSER_MODE) or DEFAULT_BROWSER_MODE).strip().lower()
    if raw_mode not in ALLOWED_BROWSER_MODES:
        raise ValueError(
            f"Invalid BROWSER_MODE={raw_mode!r}. Allowed values: {sorted(ALLOWED_BROWSER_MODES)}."
        )

    chrome_channel = (os.getenv("CHROME_CHANNEL", DEFAULT_CHROME_CHANNEL) or DEFAULT_CHROME_CHANNEL).strip()
    chrome_user_data_dir = (
        os.getenv("CHROME_USER_DATA_DIR", DEFAULT_CHROME_USER_DATA_DIR) or DEFAULT_CHROME_USER_DATA_DIR
    ).strip()
    chromium_user_data_dir = (
        os.getenv("CHROMIUM_USER_DATA_DIR", DEFAULT_CHROMIUM_USER_DATA_DIR) or DEFAULT_CHROMIUM_USER_DATA_DIR
    ).strip()

    slow_mo = _parse_non_negative_int(os.getenv("BROWSER_SLOW_MO_MS", "0"))
    min_delay = _parse_non_negative_int(os.getenv("ACTION_MIN_DELAY_MS", "0"))
    max_delay = _parse_non_negative_int(os.getenv("ACTION_MAX_DELAY_MS", "0"))

    if max_delay < min_delay:
        raise ValueError(
            f"ACTION_MAX_DELAY_MS ({max_delay}) must be >= ACTION_MIN_DELAY_MS ({min_delay})."
        )

    return BrowserRuntimeConfig(
        mode=raw_mode,
        chrome_channel=chrome_channel,
        chromium_user_data_dir=chromium_user_data_dir,
        chrome_user_data_dir=chrome_user_data_dir,
        slow_mo_ms=slow_mo,
        action_min_delay_ms=min_delay,
        action_max_delay_ms=max_delay,
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


def _parse_non_negative_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return max(0, value)
