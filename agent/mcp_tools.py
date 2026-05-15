from __future__ import annotations

import sys
from contextlib import redirect_stdout
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Literal

from rich.console import Console

from agent.browser import Browser
from agent.planners.base import PlannerAction
from agent.safety_audit import append_safety_audit
from agent.safety_engine import SafetyDecision, SafetyEngine
from agent.tool_registry import TOOL_REGISTRY
from agent.tools import ToolDispatcher


MCP_TOOL_NAMES: tuple[str, ...] = (
    "browser_observe",
    "browser_goto",
    "browser_click_element",
    "browser_type_text",
    "browser_extract_text",
    "browser_screenshot",
    "browser_scroll",
    "browser_wait",
)

FORBIDDEN_MCP_TOOL_NAMES: tuple[str, ...] = ("ask_user", "done", "query_page")

REFRESH_WARNING = "Refs are ephemeral; call browser_observe after navigation or page mutation."

AuditWriter = Callable[..., None]


class _NullLLMClient:
    def query_page(self, observation: dict[str, Any], question: str) -> dict[str, Any]:
        return {"ok": False, "message": "query_page is not exposed by the MCP server", "data": {}}


class MCPBrowserTools:
    """Thin MCP-facing adapter over the existing browser, dispatcher, and safety code."""

    def __init__(
        self,
        *,
        browser_factory: Callable[[], Any] | None = None,
        safety_engine: SafetyEngine | None = None,
        audit_writer: AuditWriter | None = append_safety_audit,
    ) -> None:
        self._browser_factory = browser_factory or Browser
        self._browser: Any | None = None
        self._dispatcher: ToolDispatcher | None = None
        self._safety_engine = safety_engine or SafetyEngine()
        self._audit_writer = audit_writer
        self._lock = RLock()
        self._current_obs: dict[str, Any] | None = None
        self._snapshot_counter = 0
        self._snapshot_id: str | None = None
        self._step = 0

    def close(self) -> None:
        with self._lock:
            if self._browser is None:
                return
            try:
                with redirect_stdout(sys.stderr):
                    self._browser.close()
            finally:
                self._browser = None
                self._dispatcher = None

    def browser_observe(self) -> dict[str, Any]:
        with self._lock:
            try:
                action = PlannerAction(
                    thought="MCP browser tool call",
                    tool="observe",
                    args={},
                    risk=TOOL_REGISTRY.get("observe").default_risk.value,
                )
                decision = self._evaluate_and_audit(action)
                blocked = self._blocked_result_if_needed(decision)
                if blocked is not None:
                    return blocked
                browser = self._ensure_browser()
                with redirect_stdout(sys.stderr):
                    obs = browser.observe()
                self._current_obs = obs
                self._snapshot_counter += 1
                self._snapshot_id = f"mcp-snapshot-{self._snapshot_counter}"
                snapshot = str(obs.get("snapshot_yaml", ""))
                return {
                    "ok": bool(obs.get("ok", False)),
                    "message": "observed page" if obs.get("ok", False) else str(obs.get("error", "observe failed")),
                    "url": obs.get("url"),
                    "title": obs.get("title"),
                    "snapshot_id": self._snapshot_id,
                    "snapshot": snapshot,
                    "refs_count": snapshot.count("[ref="),
                    "text": None,
                    "artifact_path": None,
                    "warning": REFRESH_WARNING,
                }
            except Exception as exc:
                return self._failure("browser_observe failed", exc)

    def browser_goto(self, url: str) -> dict[str, Any]:
        with self._lock:
            valid_url = str(url).strip()
            return self._execute_tool("goto", {"url": valid_url})

    def browser_click_element(self, ref: str) -> dict[str, Any]:
        with self._lock:
            normalized_ref = self._normalize_required_ref(ref)
            if normalized_ref is None:
                return self._invalid_ref_result(ref)
            stale_result = self._validate_current_ref(normalized_ref)
            if stale_result is not None:
                return stale_result
            return self._execute_tool("click_element", {"ref": normalized_ref}, mutation_warning=True)

    def browser_type_text(
        self,
        ref: str,
        text: str,
        submit: bool = False,
        clear: bool = True,
    ) -> dict[str, Any]:
        with self._lock:
            normalized_ref = self._normalize_required_ref(ref)
            if normalized_ref is None:
                return self._invalid_ref_result(ref)
            stale_result = self._validate_current_ref(normalized_ref)
            if stale_result is not None:
                return stale_result
            return self._execute_tool(
                "type_text",
                {"ref": normalized_ref, "text": str(text), "submit": bool(submit), "clear": bool(clear)},
                mutation_warning=True,
            )

    def browser_extract_text(self, ref: str | None = None) -> dict[str, Any]:
        with self._lock:
            normalized_ref = self._normalize_optional_ref(ref)
            if normalized_ref is not None:
                stale_result = self._validate_current_ref(normalized_ref)
                if stale_result is not None:
                    return stale_result
            return self._execute_tool("extract_text", {"ref": normalized_ref})

    def browser_screenshot(self, full_page: bool = False) -> dict[str, Any]:
        with self._lock:
            return self._execute_tool("screenshot", {"full_page": bool(full_page)})

    def browser_scroll(self, direction: Literal["up", "down"]) -> dict[str, Any]:
        with self._lock:
            return self._execute_tool("scroll", {"direction": direction}, mutation_warning=True)

    def browser_wait(self, ms: int) -> dict[str, Any]:
        with self._lock:
            return self._execute_tool("wait", {"ms": int(ms)})

    def _ensure_browser(self) -> Any:
        if self._browser is None:
            browser = self._browser_factory()
            with redirect_stdout(sys.stderr):
                browser.start()
            self._browser = browser
            self._dispatcher = ToolDispatcher(
                browser=browser,
                llm_client=_NullLLMClient(),
                console=Console(file=sys.stderr, force_terminal=False, color_system=None),
            )
        return self._browser

    def _execute_tool(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        *,
        mutation_warning: bool = False,
    ) -> dict[str, Any]:
        try:
            args_model = TOOL_REGISTRY.validate_args(tool_name, raw_args)
            args = args_model.model_dump()
            action = PlannerAction(
                thought="MCP browser tool call",
                tool=tool_name,
                args=args,
                risk=TOOL_REGISTRY.get(tool_name).default_risk.value,
            )
            decision = self._evaluate_and_audit(action)
            blocked = self._blocked_result_if_needed(decision)
            if blocked is not None:
                return blocked

            dispatcher = self._ensure_dispatcher()
            with redirect_stdout(sys.stderr):
                result = dispatcher.dispatch(action.to_action_dict())
            if tool_name == "observe":
                self._current_obs = result.get("data") if isinstance(result.get("data"), dict) else None
            return self._format_result(
                tool_name=tool_name,
                result=result,
                warning=REFRESH_WARNING if mutation_warning else None,
            )
        except Exception as exc:
            return self._failure(f"{self._mcp_name(tool_name)} failed", exc)

    def _evaluate_and_audit(self, action: PlannerAction) -> SafetyDecision:
        self._step += 1
        spec = TOOL_REGISTRY.get(action.tool)
        observation = self._current_obs or {}
        decision = self._safety_engine.evaluate(action, spec, observation, memory=None)
        user_decision = None
        if decision.blocked:
            user_decision = "blocked_by_policy"
        elif decision.requires_confirmation:
            user_decision = "confirmation_required_not_executed"
        if self._audit_writer is not None:
            self._audit_writer(
                step=self._step,
                action=action,
                decision=decision,
                observation=observation,
                user_decision=user_decision,
                model="mcp-server",
            )
        return decision

    @staticmethod
    def _blocked_result_if_needed(decision: SafetyDecision) -> dict[str, Any] | None:
        if not decision.blocked and not decision.requires_confirmation:
            return None
        return {
            "ok": False,
            "blocked": True,
            "requires_confirmation": decision.requires_confirmation,
            "risk": decision.risk.value,
            "reason": decision.reason,
            "policy_rule": decision.policy_rule,
            "message": (
                "This action requires human confirmation and was not executed by the MCP server."
                if decision.requires_confirmation
                else "This action was blocked by the safety policy and was not executed."
            ),
            "url": None,
            "title": None,
            "snapshot_id": None,
            "snapshot": None,
            "text": None,
            "artifact_path": None,
            "warning": "Use the CLI agent for HITL actions that require explicit human approval."
            if decision.requires_confirmation
            else None,
        }

    def _ensure_dispatcher(self) -> ToolDispatcher:
        self._ensure_browser()
        if self._dispatcher is None:
            raise RuntimeError("Tool dispatcher was not initialized.")
        return self._dispatcher

    def _format_result(self, *, tool_name: str, result: dict[str, Any], warning: str | None = None) -> dict[str, Any]:
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        url, title = self._page_metadata(data)
        artifact_path = data.get("path")
        if artifact_path:
            artifact_path = str(Path(str(artifact_path)).resolve())
        text = data.get("text")
        formatted = {
            "ok": bool(result.get("ok", False)),
            "message": str(result.get("message", "")),
            "url": url,
            "title": title,
            "snapshot_id": self._snapshot_id,
            "snapshot": None,
            "text": text if isinstance(text, str) else None,
            "artifact_path": artifact_path,
            "warning": warning,
        }
        if tool_name == "extract_text":
            formatted["ref"] = data.get("ref")
            formatted["chars"] = data.get("chars")
        if tool_name == "screenshot":
            formatted["full_page"] = data.get("full_page")
        if tool_name in {"click_element", "type_text"}:
            formatted["ref"] = data.get("ref")
        if tool_name == "type_text":
            formatted["submitted"] = data.get("submitted")
            formatted["chars"] = data.get("chars")
        if tool_name == "scroll":
            formatted["direction"] = data.get("direction")
        if tool_name == "wait":
            formatted["ms"] = data.get("ms")
        if not formatted["ok"] and "error" in data:
            formatted["reason"] = data.get("error")
        return formatted

    def _page_metadata(self, data: dict[str, Any]) -> tuple[str | None, str | None]:
        url = data.get("url")
        title = None
        browser = self._browser
        if browser is not None:
            page = getattr(browser, "page", None)
            if url is None and page is not None:
                url = getattr(page, "url", None)
            safe_title = getattr(browser, "_safe_title", None)
            if callable(safe_title):
                try:
                    title = safe_title()
                except Exception:
                    title = None
        return (str(url) if url is not None else None, str(title) if title is not None else None)

    @staticmethod
    def _normalize_optional_ref(ref: str | None) -> str | None:
        if ref is None:
            return None
        value = str(ref).strip()
        if value == "" or value.lower() in {"none", "null"}:
            return None
        return value

    @classmethod
    def _normalize_required_ref(cls, ref: str) -> str | None:
        return cls._normalize_optional_ref(ref)

    @staticmethod
    def _mcp_name(tool_name: str) -> str:
        return f"browser_{tool_name}"

    def _validate_current_ref(self, ref: str) -> dict[str, Any] | None:
        snapshot = str((self._current_obs or {}).get("snapshot_yaml", ""))
        if not snapshot:
            return {
                "ok": False,
                "blocked": True,
                "requires_confirmation": False,
                "message": "Call browser_observe before using ARIA refs.",
                "reason": "No current MCP snapshot is available.",
                "url": None,
                "title": None,
                "snapshot_id": self._snapshot_id,
                "snapshot": None,
                "text": None,
                "artifact_path": None,
                "warning": REFRESH_WARNING,
            }
        if f"[ref={ref}]" not in snapshot:
            return {
                "ok": False,
                "blocked": True,
                "requires_confirmation": False,
                "message": "The ref is not present in the current MCP snapshot.",
                "reason": "Refs must come from the latest browser_observe result.",
                "url": None,
                "title": None,
                "snapshot_id": self._snapshot_id,
                "snapshot": None,
                "text": None,
                "artifact_path": None,
                "warning": REFRESH_WARNING,
            }
        return None

    def _invalid_ref_result(self, ref: object) -> dict[str, Any]:
        return {
            "ok": False,
            "blocked": True,
            "requires_confirmation": False,
            "message": "A non-empty ARIA ref is required.",
            "reason": f"Invalid ref: {ref!r}",
            "url": None,
            "title": None,
            "snapshot_id": self._snapshot_id,
            "snapshot": None,
            "text": None,
            "artifact_path": None,
            "warning": REFRESH_WARNING,
        }

    def _failure(self, message: str, exc: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "message": message,
            "reason": str(exc),
            "url": None,
            "title": None,
            "snapshot_id": self._snapshot_id,
            "snapshot": None,
            "text": None,
            "artifact_path": None,
            "warning": None,
        }
