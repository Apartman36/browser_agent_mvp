from __future__ import annotations

import pytest

from agent.config import (
    BROWSER_MODE_CHROME_PROFILE,
    BROWSER_MODE_CHROMIUM,
    BrowserRuntimeConfig,
    load_browser_runtime_config,
    load_config,
)


@pytest.fixture(autouse=True)
def _clear_browser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "BROWSER_MODE",
        "CHROME_CHANNEL",
        "CHROME_USER_DATA_DIR",
        "CHROMIUM_USER_DATA_DIR",
        "BROWSER_SLOW_MO_MS",
        "ACTION_MIN_DELAY_MS",
        "ACTION_MAX_DELAY_MS",
        "RUN_LOG_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)


def test_default_browser_mode_is_chromium() -> None:
    runtime = load_browser_runtime_config()

    assert runtime.mode == BROWSER_MODE_CHROMIUM
    assert runtime.active_channel() is None
    assert runtime.active_user_data_dir() == ".pw_profile"
    assert runtime.slow_mo_ms == 0
    assert runtime.action_min_delay_ms == 0
    assert runtime.action_max_delay_ms == 0


def test_chrome_profile_mode_uses_chrome_channel_and_user_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_MODE", "chrome_profile")
    monkeypatch.setenv("CHROME_USER_DATA_DIR", "C:/tmp/my-agent-profile")
    monkeypatch.setenv("CHROME_CHANNEL", "chrome")

    runtime = load_browser_runtime_config()

    assert runtime.mode == BROWSER_MODE_CHROME_PROFILE
    assert runtime.active_channel() == "chrome"
    assert runtime.active_user_data_dir() == "C:/tmp/my-agent-profile"


def test_invalid_browser_mode_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_MODE", "headless_evil_mode")

    with pytest.raises(ValueError, match="Invalid BROWSER_MODE"):
        load_browser_runtime_config()


def test_action_delay_max_less_than_min_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTION_MIN_DELAY_MS", "1000")
    monkeypatch.setenv("ACTION_MAX_DELAY_MS", "500")

    with pytest.raises(ValueError, match="ACTION_MAX_DELAY_MS"):
        load_browser_runtime_config()


def test_action_delay_parses_non_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTION_MIN_DELAY_MS", "-50")
    monkeypatch.setenv("ACTION_MAX_DELAY_MS", "-10")
    monkeypatch.setenv("BROWSER_SLOW_MO_MS", "not-a-number")

    runtime = load_browser_runtime_config()

    assert runtime.action_min_delay_ms == 0
    assert runtime.action_max_delay_ms == 0
    assert runtime.slow_mo_ms == 0


def test_chrome_profile_dir_does_not_default_to_default_chrome_profile() -> None:
    runtime = BrowserRuntimeConfig(mode=BROWSER_MODE_CHROME_PROFILE)

    # The default agent profile dir must be a local, separate directory.
    assert runtime.active_user_data_dir() == ".chrome_agent_profile"
    assert "Google" not in runtime.active_user_data_dir()
    assert "AppData" not in runtime.active_user_data_dir()


def test_process_env_overrides_dotenv(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "BROWSER_MODE=chromium",
                "CHROME_USER_DATA_DIR=.chrome_agent_profile",
                "BROWSER_SLOW_MO_MS=0",
                "ACTION_MIN_DELAY_MS=0",
                "ACTION_MAX_DELAY_MS=0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BROWSER_MODE", "chrome_profile")
    monkeypatch.setenv(
        "CHROME_USER_DATA_DIR", "C:/Users/hustlePC/browser-agent-chrome-profile"
    )
    monkeypatch.setenv("BROWSER_SLOW_MO_MS", "150")
    monkeypatch.setenv("ACTION_MIN_DELAY_MS", "500")
    monkeypatch.setenv("ACTION_MAX_DELAY_MS", "1500")

    config = load_config()

    assert config.browser.mode == "chrome_profile"
    assert (
        config.browser.chrome_user_data_dir
        == "C:/Users/hustlePC/browser-agent-chrome-profile"
    )
    assert config.browser.slow_mo_ms == 150
    assert config.browser.action_min_delay_ms == 500
    assert config.browser.action_max_delay_ms == 1500


def test_dotenv_used_when_process_env_absent(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "BROWSER_MODE",
        "CHROME_USER_DATA_DIR",
        "CHROME_CHANNEL",
        "BROWSER_SLOW_MO_MS",
        "ACTION_MIN_DELAY_MS",
        "ACTION_MAX_DELAY_MS",
        "CHROMIUM_USER_DATA_DIR",
        "RUN_LOG_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "BROWSER_MODE=chrome_profile",
                "CHROME_USER_DATA_DIR=C:/Users/hustlePC/browser-agent-chrome-profile",
                "BROWSER_SLOW_MO_MS=150",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config()

    assert config.browser.mode == "chrome_profile"
    assert (
        config.browser.chrome_user_data_dir
        == "C:/Users/hustlePC/browser-agent-chrome-profile"
    )
    assert config.browser.slow_mo_ms == 150


def test_load_config_exposes_browser_runtime(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BROWSER_MODE", "chrome_profile")
    monkeypatch.setenv("CHROME_USER_DATA_DIR", ".my_agent_chrome")
    monkeypatch.setenv("RUN_LOG_ROOT", "logs/myruns")

    config = load_config()

    assert config.browser.mode == BROWSER_MODE_CHROME_PROFILE
    assert config.browser.active_user_data_dir() == ".my_agent_chrome"
    assert config.run_log_root == "logs/myruns"
