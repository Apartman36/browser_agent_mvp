from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent import browser as browser_module
from agent.browser import Browser
from agent.config import (
    BROWSER_MODE_CHROME_PROFILE,
    BROWSER_MODE_CHROMIUM,
    BrowserRuntimeConfig,
)


class FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"

    def set_default_timeout(self, ms: int) -> None:
        self.timeout = ms


class FakeContext:
    def __init__(self) -> None:
        self.pages: list[FakePage] = []
        self.closed = False
        self.timeout: int | None = None

    def set_default_timeout(self, ms: int) -> None:
        self.timeout = ms

    def new_page(self) -> FakePage:
        page = FakePage()
        self.pages.append(page)
        return page

    def close(self) -> None:
        self.closed = True


class FakeChromiumLauncher:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def launch_persistent_context(self, **kwargs: Any) -> FakeContext:
        self.last_kwargs = kwargs
        return FakeContext()


class FakePlaywright:
    def __init__(self) -> None:
        self.chromium = FakeChromiumLauncher()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class FakeSyncPlaywright:
    def __init__(self) -> None:
        self.instance = FakePlaywright()

    def start(self) -> FakePlaywright:
        return self.instance


@pytest.fixture
def fake_pw(monkeypatch: pytest.MonkeyPatch) -> FakeSyncPlaywright:
    fake = FakeSyncPlaywright()
    monkeypatch.setattr(browser_module, "sync_playwright", lambda: fake)
    return fake


def test_chromium_mode_launches_without_channel(tmp_path: Path, fake_pw: FakeSyncPlaywright) -> None:
    profile = tmp_path / "pw_profile"
    runtime = BrowserRuntimeConfig(
        mode=BROWSER_MODE_CHROMIUM,
        chromium_user_data_dir=str(profile),
    )
    browser = Browser(runtime=runtime)
    browser.start()

    kwargs = fake_pw.instance.chromium.last_kwargs
    assert kwargs is not None
    assert "channel" not in kwargs
    assert kwargs["user_data_dir"] == str(profile)
    assert kwargs["headless"] is False
    assert "slow_mo" not in kwargs
    assert profile.exists()


def test_chrome_profile_mode_launches_with_channel_and_separate_profile(
    tmp_path: Path, fake_pw: FakeSyncPlaywright
) -> None:
    agent_profile = tmp_path / "chrome_agent_profile"
    runtime = BrowserRuntimeConfig(
        mode=BROWSER_MODE_CHROME_PROFILE,
        chrome_channel="chrome",
        chrome_user_data_dir=str(agent_profile),
    )
    browser = Browser(runtime=runtime)
    browser.start()

    kwargs = fake_pw.instance.chromium.last_kwargs
    assert kwargs is not None
    assert kwargs["channel"] == "chrome"
    assert kwargs["user_data_dir"] == str(agent_profile)
    assert agent_profile.exists()


def test_chrome_profile_mode_never_uses_default_chrome_profile(
    tmp_path: Path, fake_pw: FakeSyncPlaywright
) -> None:
    runtime = BrowserRuntimeConfig(mode=BROWSER_MODE_CHROME_PROFILE)
    browser = Browser(runtime=runtime)
    browser.start()

    kwargs = fake_pw.instance.chromium.last_kwargs
    assert kwargs is not None
    user_data_dir = str(kwargs["user_data_dir"])
    # Default agent profile must be a local relative dir, not Chrome's real profile path.
    assert user_data_dir == ".chrome_agent_profile"
    forbidden = ("Google\\Chrome\\User Data", "Google/Chrome/User Data", "AppData")
    assert all(needle not in user_data_dir for needle in forbidden)


def test_slow_mo_only_passed_when_positive(tmp_path: Path, fake_pw: FakeSyncPlaywright) -> None:
    runtime = BrowserRuntimeConfig(
        mode=BROWSER_MODE_CHROMIUM,
        chromium_user_data_dir=str(tmp_path / "p"),
        slow_mo_ms=250,
    )
    browser = Browser(runtime=runtime)
    browser.start()

    kwargs = fake_pw.instance.chromium.last_kwargs
    assert kwargs is not None
    assert kwargs["slow_mo"] == 250


def test_explicit_user_data_dir_overrides_runtime(tmp_path: Path, fake_pw: FakeSyncPlaywright) -> None:
    runtime = BrowserRuntimeConfig(
        mode=BROWSER_MODE_CHROMIUM,
        chromium_user_data_dir=str(tmp_path / "ignored"),
    )
    override = tmp_path / "explicit"
    browser = Browser(user_data_dir=str(override), runtime=runtime)
    browser.start()

    kwargs = fake_pw.instance.chromium.last_kwargs
    assert kwargs is not None
    assert kwargs["user_data_dir"] == str(override)
