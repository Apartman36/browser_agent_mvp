from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from agent.config import load_config
from agent.llm import LLMClient
from agent.logging_utils import append_action_log, write_final_report
from agent.memory import Memory
from agent.planners import create_planner
from agent.planners.base import PlannerAction
from agent.safety import user_confirmed
from agent.safety_audit import append_safety_audit
from agent.safety_engine import SafetyEngine
from agent.tool_registry import TOOL_REGISTRY, RiskLevel
from agent.tools import ToolDispatcher, compact_json


LLM_FALLBACK_RETRY_ANSWERS = {"retry", "r", "повтор", "повтори"}
LLM_FALLBACK_STOP_ANSWERS = {"stop", "s", "стоп"}


EventSink = Callable[[str, dict[str, Any]], None]
CancelChecker = Callable[[], bool]
SafetyConfirmation = Callable[[dict[str, Any]], str]
SENSITIVE_EVENT_KEYS = {"password", "passwd", "token", "secret", "api_key", "apikey", "authorization"}


def run_agent(
    goal: str,
    browser: Any,
    max_steps: int = 25,
    llm_client: LLMClient | None = None,
    event_sink: EventSink | None = None,
    should_cancel: CancelChecker | None = None,
    safety_confirmation: SafetyConfirmation | None = None,
) -> dict[str, Any]:
    console = Console()
    config = load_config()
    registry = TOOL_REGISTRY
    llm = llm_client or LLMClient()
    planner = create_planner(config=config, llm_client=llm, registry=registry)
    safety_engine = SafetyEngine()
    memory = Memory(goal)
    tools = ToolDispatcher(browser=browser, llm_client=llm, console=console)

    console.print(Panel.fit(goal, title="User task", border_style="cyan"))
    _emit_event(event_sink, "run_started", {"task": goal, "max_steps": max_steps})

    try:
        for step in range(1, max_steps + 1):
            if _is_cancel_requested(should_cancel):
                return _cancelled_result(goal, console, event_sink)

            obs = browser.observe()
            memory.update_observation(obs)
            tools.set_observation(obs)
            ref_count = obs.get("snapshot_yaml", "").count("[ref=")
            _emit_event(
                event_sink,
                "observation",
                {
                    "step": step,
                    "url": obs.get("url", ""),
                    "title": obs.get("title", ""),
                    "refs_count": ref_count,
                    "ok": bool(obs.get("ok", False)),
                    "error": obs.get("error"),
                },
            )
            console.print(
                f"\n[bold]Step {step}/{max_steps}[/bold] "
                f"[dim]URL:[/dim] {obs.get('url', '')} "
                f"[dim]Title:[/dim] {obs.get('title', '')} "
                f"[dim]Refs:[/dim] {ref_count}"
            )

            action_model = _validate_planned_action(planner.plan(memory.to_prompt_payload()), registry)
            action = action_model.to_action_dict()
            console.print(f"[bold green]Assistant:[/bold green] {action_model.thought}")
            console.print(f"[bold blue]Using tool:[/bold blue] {action_model.tool}")
            console.print(f"[dim]Input:[/dim] {compact_json(action_model.args)}")
            _emit_event(
                event_sink,
                "tool_call",
                {
                    "step": step,
                    "tool": action_model.tool,
                    "thought": action_model.thought,
                    "args": _sanitize_event_value(action_model.args),
                    "risk": action_model.risk,
                    "needs_user_confirmation": action_model.needs_user_confirmation,
                },
            )

            result = _execute_with_safety(
                step=step,
                action=action_model,
                obs=obs,
                memory=memory,
                tools=tools,
                console=console,
                safety_engine=safety_engine,
                model=llm.model,
                safety_confirmation=safety_confirmation,
            )
            planner.append_tool_result(action_model, result)
            _emit_tool_result_event(event_sink, step, action_model.tool, result)

            console.print(
                f"[bold]Result:[/bold] {'OK' if result.get('ok') else 'ERROR'} - "
                f"{result.get('message', '')}"
            )
            memory.merge_facts(action.get("new_facts", {}))
            memory.add_action(action, result)
            append_action_log(step, action, result, obs)

            if _is_llm_provider_fallback_action(action):
                decision = _llm_provider_fallback_decision(result)
                if decision == "retry":
                    console.print("[yellow]Retry requested after LLM provider error.[/yellow]")
                    continue
                if decision == "stop":
                    summary = "Stopped by user after the LLM provider returned an error or rate limit."
                    report_path = write_final_report(goal, "stopped_by_user", summary)
                    console.print(Panel(summary, title="Final report: stopped_by_user", border_style="yellow"))
                    console.print(f"[dim]Saved final report:[/dim] {report_path}")
                    final = {
                        "ok": False,
                        "status": "stopped_by_user",
                        "summary": summary,
                        "report_path": str(report_path),
                    }
                    _emit_event(event_sink, "done", final)
                    return final
                console.print("[yellow]Continuing after LLM provider fallback answer.[/yellow]")
                continue

            if action_model.tool == "done":
                status = str(action_model.args.get("status", result.get("data", {}).get("status", "success")))
                summary = str(action_model.args.get("summary", result.get("data", {}).get("summary", "")))
                report_path = write_final_report(goal, status, summary)
                console.print(Panel(summary, title=f"Final report: {status}", border_style="green"))
                console.print(f"[dim]Saved final report:[/dim] {report_path}")
                final = {"ok": status == "success", "status": status, "summary": summary, "report_path": str(report_path)}
                _emit_event(event_sink, "done", final)
                return final

        summary = f"Stopped after reaching max_steps={max_steps} before the task was completed."
        report_path = write_final_report(goal, "failed", summary)
        console.print(Panel(summary, title="Final report: failed", border_style="red"))
        final = {"ok": False, "status": "failed", "summary": summary, "report_path": str(report_path)}
        _emit_event(event_sink, "done", final)
        return final

    except KeyboardInterrupt:
        summary = "Stopped by user with KeyboardInterrupt."
        report_path = write_final_report(goal, "stopped_by_user", summary)
        console.print("\n[yellow]Stopped by user.[/yellow]")
        final = {"ok": False, "status": "stopped_by_user", "summary": summary, "report_path": str(report_path)}
        _emit_event(event_sink, "cancelled", final)
        return final


def _execute_with_safety(
    step: int,
    action: PlannerAction,
    obs: dict[str, Any],
    memory: Memory,
    tools: ToolDispatcher,
    console: Console,
    safety_engine: SafetyEngine,
    model: str | None = None,
    safety_confirmation: SafetyConfirmation | None = None,
) -> dict[str, Any]:
    spec = TOOL_REGISTRY.get(action.tool)
    decision = safety_engine.evaluate(action, spec, obs, memory=memory)
    if decision.requires_confirmation or decision.blocked or decision.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        console.print(
            f"[bold yellow]Safety:[/bold yellow] {decision.risk.value} "
            f"({decision.policy_rule}) - {decision.reason}"
        )

    if decision.blocked:
        append_safety_audit(
            step=step,
            action=action,
            decision=decision,
            observation=obs,
            user_decision="blocked_by_policy",
            model=model,
        )
        return {
            "ok": False,
            "message": "blocked by safety policy",
            "data": {"reason": decision.reason, "policy_rule": decision.policy_rule},
        }

    if decision.requires_confirmation:
        prompt = {
            "step": step,
            "tool": action.tool,
            "args": _sanitize_event_value(action.args),
            "risk": decision.risk.value,
            "reason": decision.reason,
            "policy_rule": decision.policy_rule,
        }
        answer = safety_confirmation(prompt) if safety_confirmation else input("Execute? [y/N]\n> ")
        if not user_confirmed(answer):
            append_safety_audit(
                step=step,
                action=action,
                decision=decision,
                observation=obs,
                user_decision="denied",
                model=model,
            )
            return {
                "ok": False,
                "message": "user declined high-risk action",
                "data": {"reason": decision.reason, "policy_rule": decision.policy_rule},
            }
        append_safety_audit(
            step=step,
            action=action,
            decision=decision,
            observation=obs,
            user_decision="approved",
            model=model,
        )
    else:
        append_safety_audit(step=step, action=action, decision=decision, observation=obs, model=model)

    return tools.dispatch(action.to_action_dict())


def _validate_planned_action(action: PlannerAction, registry: Any) -> PlannerAction:
    try:
        if not isinstance(action, PlannerAction):
            action = PlannerAction.model_validate(action)
        registry.get(action.tool)
        validated_args = registry.validate_args(action.tool, action.args)
        action.args = validated_args.model_dump()
        return action
    except (KeyError, TypeError, ValidationError, ValueError) as exc:
        return PlannerAction(
            thought="The planned tool call failed validation, so I need user guidance.",
            tool="ask_user",
            args={"question": f"The planned tool call was invalid: {exc}. What should I do next?"},
            risk="medium",
            needs_user_confirmation=True,
            new_facts={"last_tool_validation_error": str(exc)},
        )


def pretty_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, indent=2, default=str)


def _is_llm_provider_fallback_action(action: dict[str, Any]) -> bool:
    if action.get("tool") != "ask_user":
        return False
    question = str((action.get("args") or {}).get("question", ""))
    return "LLM provider returned an error or rate limit" in question


def _llm_provider_fallback_decision(result: dict[str, Any]) -> str:
    answer = str(result.get("data", {}).get("answer", "")).strip().lower()
    if answer in LLM_FALLBACK_RETRY_ANSWERS:
        return "retry"
    if answer in LLM_FALLBACK_STOP_ANSWERS:
        return "stop"
    return "continue"


def _emit_event(event_sink: EventSink | None, event_type: str, data: dict[str, Any]) -> None:
    if event_sink is None:
        return
    try:
        event_sink(event_type, data)
    except Exception:
        # UI telemetry must never alter CLI or MCP agent behavior.
        return


def _emit_tool_result_event(
    event_sink: EventSink | None,
    step: int,
    tool: str,
    result: dict[str, Any],
) -> None:
    data = {
        "step": step,
        "tool": tool,
        "ok": bool(result.get("ok", False)),
        "message": str(result.get("message", ""))[:1000],
        "data": _sanitize_event_value(result.get("data", {}), max_string=1200),
    }
    _emit_event(event_sink, "tool_result", data)

    if tool == "screenshot" and result.get("ok"):
        path = (result.get("data") or {}).get("path")
        if path:
            _emit_event(
                event_sink,
                "screenshot",
                {
                    "step": step,
                    "path": str(path),
                    "full_page": bool((result.get("data") or {}).get("full_page", False)),
                },
            )


def _sanitize_event_value(value: Any, max_string: int = 1000) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(secret in key_text.lower() for secret in SENSITIVE_EVENT_KEYS):
                sanitized[key_text] = "[redacted]"
            else:
                sanitized[key_text] = _sanitize_event_value(item, max_string=max_string)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_event_value(item, max_string=max_string) for item in value[:50]]
    if isinstance(value, tuple):
        return [_sanitize_event_value(item, max_string=max_string) for item in value[:50]]
    if isinstance(value, str):
        if len(value) <= max_string:
            return value
        return value[:max_string] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:max_string]


def _is_cancel_requested(should_cancel: CancelChecker | None) -> bool:
    if should_cancel is None:
        return False
    try:
        return bool(should_cancel())
    except Exception:
        return False


def _cancelled_result(goal: str, console: Console, event_sink: EventSink | None) -> dict[str, Any]:
    summary = "Cancelled by user."
    report_path = write_final_report(goal, "cancelled", summary)
    console.print(Panel(summary, title="Final report: cancelled", border_style="yellow"))
    final = {"ok": False, "status": "cancelled", "summary": summary, "report_path": str(report_path)}
    _emit_event(event_sink, "cancelled", final)
    return final
