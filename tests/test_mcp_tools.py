from __future__ import annotations

from typing import Any

from agent.mcp_tools import MCPBrowserTools
from agent.planners.base import PlannerAction
from agent.safety_audit import sanitize_args
from agent.safety_engine import SafetyDecision, SafetyEngine
from agent.tool_registry import RiskLevel, ToolSpec


class FakePage:
    def __init__(self) -> None:
        self.url = "about:blank"


class FakeBrowser:
    def __init__(self, snapshot: str = '- button "Run" [ref=e1]') -> None:
        self.snapshot = snapshot
        self.page = FakePage()
        self.started = False
        self.closed = False
        self.calls: list[tuple[Any, ...]] = []

    def start(self) -> None:
        self.started = True
        self.calls.append(("start",))

    def close(self) -> None:
        self.closed = True
        self.calls.append(("close",))

    def observe(self) -> dict[str, Any]:
        self.calls.append(("observe",))
        return {
            "ok": True,
            "url": self.page.url,
            "title": "Fake Title",
            "snapshot_yaml": self.snapshot,
            "body_text": "Run",
            "error": None,
        }

    def goto(self, url: str) -> dict[str, Any]:
        self.calls.append(("goto", url))
        self.page.url = url
        return {"ok": True, "message": "successfully navigated", "data": {"url": url}}

    def click_element(self, ref: str) -> dict[str, Any]:
        self.calls.append(("click_element", ref))
        return {"ok": True, "message": "clicked element", "data": {"ref": ref, "url": self.page.url}}

    def type_text(self, ref: str, text: str, submit: bool = False, clear: bool = True) -> dict[str, Any]:
        self.calls.append(("type_text", ref, text, submit, clear))
        return {
            "ok": True,
            "message": "typed text",
            "data": {"ref": ref, "submitted": submit, "chars": len(text), "url": self.page.url},
        }

    def extract_text(self, ref: str | None = None) -> dict[str, Any]:
        self.calls.append(("extract_text", ref))
        return {"ok": True, "message": "extracted text", "data": {"ref": ref, "text": "Page text", "chars": 9}}

    def screenshot(self, full_page: bool = False) -> dict[str, Any]:
        self.calls.append(("screenshot", full_page))
        return {"ok": True, "message": "saved screenshot", "data": {"path": "logs/screenshots/fake.png"}}

    def scroll(self, direction: str) -> dict[str, Any]:
        self.calls.append(("scroll", direction))
        return {"ok": True, "message": "scrolled page", "data": {"direction": direction}}

    def wait(self, ms: int) -> dict[str, Any]:
        self.calls.append(("wait", ms))
        return {"ok": True, "message": "waited", "data": {"ms": ms}}

    def _safe_title(self) -> str:
        return "Fake Title"


class RecordingSafety(SafetyEngine):
    def __init__(
        self,
        events: list[tuple[Any, ...]],
        *,
        confirmation_tools: set[str] | None = None,
    ) -> None:
        self.events = events
        self.confirmation_tools = confirmation_tools or set()

    def evaluate(
        self,
        action: PlannerAction,
        tool_spec: ToolSpec,
        observation: dict[str, Any],
        memory: Any = None,
    ) -> SafetyDecision:
        self.events.append(("safety", action.tool))
        if action.tool in self.confirmation_tools:
            return SafetyDecision(
                allowed=True,
                requires_confirmation=True,
                blocked=False,
                risk=RiskLevel.HIGH,
                category=tool_spec.category,
                reason="test confirmation required",
                policy_rule="test_confirmation",
            )
        return SafetyDecision(
            allowed=True,
            requires_confirmation=False,
            blocked=False,
            risk=tool_spec.default_risk,
            category=tool_spec.category,
            reason="test allowed",
            policy_rule="test_allowed",
        )


def test_browser_goto_rejects_unsafe_url_without_starting_browser() -> None:
    browser = FakeBrowser()
    tools = MCPBrowserTools(browser_factory=lambda: browser, audit_writer=None)

    result = tools.browser_goto("javascript:alert(1)")

    assert result["ok"] is False
    assert result["blocked"] is True
    assert "non-http URL scheme" in result["reason"]
    assert browser.started is False
    assert browser.calls == []


def test_browser_goto_accepts_https_and_dispatches() -> None:
    browser = FakeBrowser()
    audit_records: list[dict[str, Any]] = []
    tools = MCPBrowserTools(browser_factory=lambda: browser, audit_writer=lambda **kwargs: audit_records.append(kwargs))

    result = tools.browser_goto("https://example.com")

    assert result["ok"] is True
    assert result["url"] == "https://example.com"
    assert ("goto", "https://example.com") in browser.calls
    assert audit_records[-1]["action"].tool == "goto"


def test_browser_observe_returns_structured_snapshot_result() -> None:
    browser = FakeBrowser(snapshot='- button "Run" [ref=e1]\n- link "Docs" [ref=e2]')
    tools = MCPBrowserTools(browser_factory=lambda: browser, audit_writer=None)

    result = tools.browser_observe()

    assert result["ok"] is True
    assert result["snapshot_id"] == "mcp-snapshot-1"
    assert result["snapshot"] == browser.snapshot
    assert result["refs_count"] == 2


def test_browser_click_element_runs_safety_before_dispatch() -> None:
    events: list[tuple[Any, ...]] = []
    browser = FakeBrowser()

    original_click = browser.click_element

    def click_with_event(ref: str) -> dict[str, Any]:
        events.append(("browser", "click_element", ref))
        return original_click(ref)

    browser.click_element = click_with_event  # type: ignore[method-assign]
    tools = MCPBrowserTools(
        browser_factory=lambda: browser,
        safety_engine=RecordingSafety(events),
        audit_writer=None,
    )
    tools.browser_observe()
    events.clear()

    result = tools.browser_click_element("e1")

    assert result["ok"] is True
    assert events[:2] == [("safety", "click_element"), ("browser", "click_element", "e1")]


def test_browser_click_element_does_not_dispatch_when_confirmation_required() -> None:
    events: list[tuple[Any, ...]] = []
    browser = FakeBrowser()
    tools = MCPBrowserTools(
        browser_factory=lambda: browser,
        safety_engine=RecordingSafety(events, confirmation_tools={"click_element"}),
        audit_writer=None,
    )
    tools.browser_observe()
    events.clear()

    result = tools.browser_click_element("e1")

    assert result["ok"] is False
    assert result["blocked"] is True
    assert result["requires_confirmation"] is True
    assert ("click_element", "e1") not in browser.calls
    assert events == [("safety", "click_element")]


def test_browser_type_text_sensitive_text_is_redacted_in_audit_path() -> None:
    browser = FakeBrowser(snapshot='- textbox "Search" [ref=e1]')
    audit_args: list[dict[str, Any]] = []

    def audit_writer(**kwargs: Any) -> None:
        audit_args.append(
            sanitize_args(
                kwargs["action"].args,
                action=kwargs["action"],
                decision=kwargs["decision"],
                observation=kwargs["observation"],
            )
        )

    tools = MCPBrowserTools(browser_factory=lambda: browser, audit_writer=audit_writer)
    tools.browser_observe()

    result = tools.browser_type_text("e1", "sk-testsecret123456789")

    assert result["ok"] is False
    assert result["requires_confirmation"] is True
    assert audit_args[-1]["text"] == "[redacted]"
    assert not any(call[0] == "type_text" for call in browser.calls)


def test_browser_extract_text_normalizes_null_like_refs() -> None:
    browser = FakeBrowser()
    tools = MCPBrowserTools(browser_factory=lambda: browser, audit_writer=None)

    result = tools.browser_extract_text("null")

    assert result["ok"] is True
    assert result["text"] == "Page text"
    assert ("extract_text", None) in browser.calls
