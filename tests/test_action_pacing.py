from __future__ import annotations

from typing import Any

import pytest

from agent import browser as browser_module
from agent.browser import Browser
from agent.config import BROWSER_MODE_CHROMIUM, BrowserRuntimeConfig


class StubPage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.calls: list[tuple[str, Any]] = []

    def goto(self, url: str, **kwargs: Any) -> None:
        self.calls.append(("goto", url))
        self.url = url

    def wait_for_load_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def wait_for_timeout(self, ms: int) -> None:
        return None

    def title(self) -> str:
        return "stub"


def _browser_with_runtime(runtime: BrowserRuntimeConfig) -> Browser:
    browser = Browser(runtime=runtime)
    browser.page = StubPage()
    return browser


def test_no_delay_when_min_and_max_are_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = BrowserRuntimeConfig(mode=BROWSER_MODE_CHROMIUM)
    browser = _browser_with_runtime(runtime)
    sleep_calls: list[float] = []
    monkeypatch.setattr(browser_module.time, "sleep", lambda s: sleep_calls.append(s))

    browser.goto("https://example.com")

    assert sleep_calls == []


def test_delay_applied_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = BrowserRuntimeConfig(
        mode=BROWSER_MODE_CHROMIUM,
        action_min_delay_ms=200,
        action_max_delay_ms=500,
    )
    browser = _browser_with_runtime(runtime)
    sleep_calls: list[float] = []
    monkeypatch.setattr(browser_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(browser_module.random, "randint", lambda lo, hi: 300)

    browser.goto("https://example.com")

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(0.300, rel=1e-6)


def test_delay_uses_value_in_configured_range(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = BrowserRuntimeConfig(
        mode=BROWSER_MODE_CHROMIUM,
        action_min_delay_ms=100,
        action_max_delay_ms=400,
    )
    browser = _browser_with_runtime(runtime)
    randint_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(browser_module.random, "randint", lambda lo, hi: randint_calls.append((lo, hi)) or hi)
    monkeypatch.setattr(browser_module.time, "sleep", lambda _: None)

    browser.goto("https://example.com")

    assert randint_calls == [(100, 400)]
