from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from agent.llm import LLMClient
from agent.logging_utils import append_action_log, write_final_report
from agent.memory import Memory
from agent.safety import is_high_risk, user_confirmed
from agent.tools import ToolDispatcher, compact_json


LLM_FALLBACK_RETRY_ANSWERS = {"retry", "r", "повтор", "повтори"}
LLM_FALLBACK_STOP_ANSWERS = {"stop", "s", "стоп"}


def run_agent(goal: str, browser: Any, max_steps: int = 25, llm_client: LLMClient | None = None) -> dict[str, Any]:
    console = Console()
    llm = llm_client or LLMClient()
    memory = Memory(goal)
    tools = ToolDispatcher(browser=browser, llm_client=llm, console=console)

    console.print(Panel.fit(escape(goal), title="User task", border_style="cyan"))

    try:
        for step in range(1, max_steps + 1):
            obs = browser.observe()
            memory.update_observation(obs)
            tools.set_observation(obs)
            ref_count = obs.get("snapshot_yaml", "").count("[ref=")
            console.print(
                f"\n[bold]Step {step}/{max_steps}[/bold] "
                f"[dim]URL:[/dim] {escape(obs.get('url', ''))} "
                f"[dim]Title:[/dim] {escape(obs.get('title', ''))} "
                f"[dim]Refs:[/dim] {ref_count}"
            )

            action = llm.plan(memory.to_prompt_payload())
            console.print(f"[bold green]Assistant:[/bold green] {escape(str(action.get('thought', '')))}")
            console.print(f"[bold blue]Using tool:[/bold blue] {escape(str(action.get('tool', '')))}")
            console.print(f"[dim]Input:[/dim] {escape(compact_json(action.get('args', {})))}")

            result = _execute_with_safety(action, obs, tools, console)

            console.print(
                f"[bold]Result:[/bold] {'OK' if result.get('ok') else 'ERROR'} - "
                f"{escape(str(result.get('message', '')))}"
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
                    console.print(Panel(escape(summary), title="Final report: stopped_by_user", border_style="yellow"))
                    console.print(f"[dim]Saved final report:[/dim] {report_path}")
                    return {
                        "ok": False,
                        "status": "stopped_by_user",
                        "summary": summary,
                        "report_path": str(report_path),
                    }
                console.print("[yellow]Continuing after LLM provider fallback answer.[/yellow]")
                continue

            if action.get("tool") == "done":
                status = str(action.get("args", {}).get("status", result.get("data", {}).get("status", "success")))
                summary = str(action.get("args", {}).get("summary", result.get("data", {}).get("summary", "")))
                report_path = write_final_report(goal, status, summary)
                console.print(Panel(escape(summary), title=escape(f"Final report: {status}"), border_style="green"))
                console.print(f"[dim]Saved final report:[/dim] {report_path}")
                return {"ok": status == "success", "status": status, "summary": summary, "report_path": str(report_path)}

        summary = f"Stopped after reaching max_steps={max_steps} before the task was completed."
        report_path = write_final_report(goal, "failed", summary)
        console.print(Panel(escape(summary), title="Final report: failed", border_style="red"))
        return {"ok": False, "status": "failed", "summary": summary, "report_path": str(report_path)}

    except KeyboardInterrupt:
        summary = "Stopped by user with KeyboardInterrupt."
        report_path = write_final_report(goal, "stopped_by_user", summary)
        console.print("\n[yellow]Stopped by user.[/yellow]")
        return {"ok": False, "status": "stopped_by_user", "summary": summary, "report_path": str(report_path)}


def _execute_with_safety(
    action: dict[str, Any],
    obs: dict[str, Any],
    tools: ToolDispatcher,
    console: Console,
) -> dict[str, Any]:
    if action.get("tool") in {"ask_user", "done"}:
        return tools.dispatch(action)

    high_risk, reason = is_high_risk(action, obs)
    if high_risk:
        console.print(f"[bold yellow]High-risk action detected:[/bold yellow] {escape(str(reason))}")
        answer = input("Execute? [y/N]\n> ")
        if not user_confirmed(answer):
            return {
                "ok": False,
                "message": "user declined high-risk action",
                "data": {"reason": reason},
            }

    return tools.dispatch(action)


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
