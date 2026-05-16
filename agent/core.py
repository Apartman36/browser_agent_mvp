from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from agent.config import AgentConfig, load_config
from agent.llm import LLMClient
from agent.logging_utils import append_action_log, write_final_report
from agent.memory import Memory
from agent.planners import create_planner
from agent.planners.base import PlannerAction
from agent.run_context import RunContext, create_run_context
from agent.safety import user_confirmed
from agent.safety_audit import append_safety_audit
from agent.safety_engine import SafetyEngine
from agent.tool_registry import TOOL_REGISTRY, RiskLevel
from agent.tools import ToolDispatcher, compact_json


LLM_FALLBACK_RETRY_ANSWERS = {"retry", "r", "повтор", "повтори"}
LLM_FALLBACK_STOP_ANSWERS = {"stop", "s", "стоп"}


def run_agent(
    goal: str,
    browser: Any,
    max_steps: int = 25,
    llm_client: LLMClient | None = None,
    *,
    run_context: RunContext | None = None,
    start_url: str | None = None,
) -> dict[str, Any]:
    console = Console()
    config = load_config()
    registry = TOOL_REGISTRY
    llm = llm_client or LLMClient()
    planner = create_planner(config=config, llm_client=llm, registry=registry)
    safety_engine = SafetyEngine()
    memory = Memory(goal)
    tools = ToolDispatcher(browser=browser, llm_client=llm, console=console)

    run_ctx = run_context or create_run_context(config.run_log_root)
    _initialize_run_metadata(run_ctx, config, llm, goal, max_steps, start_url)
    set_screenshot_dir = getattr(browser, "set_screenshot_dir", None)
    if callable(set_screenshot_dir):
        set_screenshot_dir(run_ctx.screenshots_dir)

    console.print(Panel.fit(goal, title="User task", border_style="cyan"))
    console.print(f"[dim]Run id:[/dim] {run_ctx.run_id}  [dim]Logs:[/dim] {run_ctx.run_dir}")

    final_status: str | None = None
    final_summary: str = ""
    final_ok: bool = False
    final_report_path: str | None = None

    try:
        for step in range(1, max_steps + 1):
            obs = browser.observe()
            memory.update_observation(obs)
            tools.set_observation(obs)
            ref_count = obs.get("snapshot_yaml", "").count("[ref=")
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

            result = _execute_with_safety(
                step=step,
                action=action_model,
                obs=obs,
                memory=memory,
                tools=tools,
                console=console,
                safety_engine=safety_engine,
                model=llm.model,
                run_ctx=run_ctx,
                browser_mode=config.browser.mode,
            )
            planner.append_tool_result(action_model, result)

            console.print(
                f"[bold]Result:[/bold] {'OK' if result.get('ok') else 'ERROR'} - "
                f"{result.get('message', '')}"
            )
            memory.merge_facts(action.get("new_facts", {}))
            memory.add_action(action, result)
            append_action_log(step, action, result, obs, log_path=run_ctx.actions_log_path)

            if _is_llm_provider_fallback_action(action):
                decision = _llm_provider_fallback_decision(result)
                if decision == "retry":
                    console.print("[yellow]Retry requested after LLM provider error.[/yellow]")
                    continue
                if decision == "stop":
                    final_summary = "Stopped by user after the LLM provider returned an error or rate limit."
                    final_status = "stopped_by_user"
                    final_ok = False
                    report_path = write_final_report(
                        goal, final_status, final_summary, report_path=run_ctx.final_report_path
                    )
                    final_report_path = str(report_path)
                    console.print(Panel(final_summary, title="Final report: stopped_by_user", border_style="yellow"))
                    console.print(f"[dim]Saved final report:[/dim] {report_path}")
                    console.print(f"[dim]Saved run logs:[/dim] {run_ctx.run_dir}")
                    _finalize_run_metadata(run_ctx, final_status, final_summary, final_ok)
                    return {
                        "ok": False,
                        "status": final_status,
                        "summary": final_summary,
                        "report_path": final_report_path,
                        "run_id": run_ctx.run_id,
                        "run_dir": str(run_ctx.run_dir),
                    }
                console.print("[yellow]Continuing after LLM provider fallback answer.[/yellow]")
                continue

            if action_model.tool == "done":
                status = str(action_model.args.get("status", result.get("data", {}).get("status", "success")))
                summary = str(action_model.args.get("summary", result.get("data", {}).get("summary", "")))
                final_status = status
                final_summary = summary
                final_ok = status == "success"
                report_path = write_final_report(goal, status, summary, report_path=run_ctx.final_report_path)
                final_report_path = str(report_path)
                console.print(Panel(summary, title=f"Final report: {status}", border_style="green"))
                console.print(f"[dim]Saved final report:[/dim] {report_path}")
                console.print(f"[dim]Saved run logs:[/dim] {run_ctx.run_dir}")
                _finalize_run_metadata(run_ctx, final_status, final_summary, final_ok)
                return {
                    "ok": final_ok,
                    "status": status,
                    "summary": summary,
                    "report_path": final_report_path,
                    "run_id": run_ctx.run_id,
                    "run_dir": str(run_ctx.run_dir),
                }

        final_summary = f"Stopped after reaching max_steps={max_steps} before the task was completed."
        final_status = "failed"
        final_ok = False
        report_path = write_final_report(goal, final_status, final_summary, report_path=run_ctx.final_report_path)
        final_report_path = str(report_path)
        console.print(Panel(final_summary, title="Final report: failed", border_style="red"))
        console.print(f"[dim]Saved run logs:[/dim] {run_ctx.run_dir}")
        _finalize_run_metadata(run_ctx, final_status, final_summary, final_ok)
        return {
            "ok": False,
            "status": final_status,
            "summary": final_summary,
            "report_path": final_report_path,
            "run_id": run_ctx.run_id,
            "run_dir": str(run_ctx.run_dir),
        }

    except KeyboardInterrupt:
        final_summary = "Stopped by user with KeyboardInterrupt."
        final_status = "stopped_by_user"
        final_ok = False
        report_path = write_final_report(goal, final_status, final_summary, report_path=run_ctx.final_report_path)
        final_report_path = str(report_path)
        console.print("\n[yellow]Stopped by user.[/yellow]")
        console.print(f"[dim]Saved run logs:[/dim] {run_ctx.run_dir}")
        _finalize_run_metadata(run_ctx, final_status, final_summary, final_ok)
        return {
            "ok": False,
            "status": final_status,
            "summary": final_summary,
            "report_path": final_report_path,
            "run_id": run_ctx.run_id,
            "run_dir": str(run_ctx.run_dir),
        }


def _execute_with_safety(
    step: int,
    action: PlannerAction,
    obs: dict[str, Any],
    memory: Memory,
    tools: ToolDispatcher,
    console: Console,
    safety_engine: SafetyEngine,
    model: str | None = None,
    run_ctx: RunContext | None = None,
    browser_mode: str | None = None,
) -> dict[str, Any]:
    spec = TOOL_REGISTRY.get(action.tool)
    decision = safety_engine.evaluate(action, spec, obs, memory=memory)
    if decision.requires_confirmation or decision.blocked or decision.risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
        console.print(
            f"[bold yellow]Safety:[/bold yellow] {decision.risk.value} "
            f"({decision.policy_rule}) - {decision.reason}"
        )

    audit_kwargs = {
        "audit_path": run_ctx.safety_audit_path if run_ctx else None,
        "run_id": run_ctx.run_id if run_ctx else None,
        "browser_mode": browser_mode,
    }

    if decision.blocked:
        append_safety_audit(
            step=step,
            action=action,
            decision=decision,
            observation=obs,
            user_decision="blocked_by_policy",
            model=model,
            **audit_kwargs,
        )
        return {
            "ok": False,
            "message": "blocked by safety policy",
            "data": {"reason": decision.reason, "policy_rule": decision.policy_rule},
        }

    if decision.requires_confirmation:
        answer = input("Execute? [y/N]\n> ")
        if not user_confirmed(answer):
            append_safety_audit(
                step=step,
                action=action,
                decision=decision,
                observation=obs,
                user_decision="denied",
                model=model,
                **audit_kwargs,
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
            **audit_kwargs,
        )
    else:
        append_safety_audit(step=step, action=action, decision=decision, observation=obs, model=model, **audit_kwargs)

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


def _initialize_run_metadata(
    run_ctx: RunContext,
    config: AgentConfig,
    llm: LLMClient,
    goal: str,
    max_steps: int,
    start_url: str | None,
) -> None:
    browser = config.browser
    run_ctx.write_initial_metadata(
        {
            "run_id": run_ctx.run_id,
            "started_at": run_ctx.started_at,
            "status": "running",
            "goal": goal,
            "max_steps": max_steps,
            "start_url": start_url or config.start_url or "",
            "planner_mode": config.planner_mode,
            "model": getattr(llm, "model", "") or "",
            "browser_mode": browser.mode,
            "browser_channel": browser.active_channel(),
            "browser_profile_dir": str(browser.active_user_data_dir()),
            "browser_slow_mo_ms": browser.slow_mo_ms,
            "action_min_delay_ms": browser.action_min_delay_ms,
            "action_max_delay_ms": browser.action_max_delay_ms,
        }
    )


def _finalize_run_metadata(
    run_ctx: RunContext,
    status: str | None,
    summary: str,
    ok: bool,
) -> None:
    run_ctx.update_metadata(
        {
            "ended_at": datetime.now().replace(microsecond=0).isoformat(),
            "status": status or "unknown",
            "ok": ok,
            "summary_excerpt": summary[:240],
        }
    )
