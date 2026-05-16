from __future__ import annotations

import json
import threading
import time
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from agent.llm import LLMClient
from agent.logging_utils import append_action_log
from agent.memory import Memory
from agent.safety import is_high_risk, user_confirmed
from agent.tools import ToolDispatcher

LLM_FALLBACK_RETRY_ANSWERS = {"retry", "r", "повтор", "повтори"}
LLM_FALLBACK_STOP_ANSWERS = {"stop", "s", "стоп"}

interrupt_event = threading.Event()
user_input_queue = []
ui_logs = []

def ui_print(msg):
    Console().print(msg)
    ui_logs.append(str(msg))

def wait_for_user_input(prompt: str) -> str:
    ui_print(f"[bold yellow]WAITING FOR USER:[/bold yellow] {prompt}")
    user_input_queue.clear()
    while not user_input_queue:
        time.sleep(0.5)
    return user_input_queue.pop(0)

def handle_interrupt(memory: Memory) -> bool:
    if interrupt_event.is_set():
        interrupt_event.clear()
        ans = wait_for_user_input("[bold red]Execution Paused.[/bold red] Enter new instructions to inject, or 'stop' to abort:")
        if ans.lower().strip() == 'stop':
            return True
        ui_print("[bold green]Injecting instruction and resuming.[/bold green]")
        memory.merge_facts({"injected_instruction": ans})
    return False

def run_agent(goal: str, browser: Any, max_steps: int = 25, llm_client: LLMClient | None = None) -> dict[str, Any]:
    console = Console()
    llm = llm_client or LLMClient()

    ui_print("[bold yellow]Planner Stage: Analyzing task...[/bold yellow]")
    sub_goals = _orchestrate_goals(goal, llm)
    if not sub_goals:
        return {"ok": False, "status": "stopped_by_user", "summary": "Task aborted during planning.", "report_path": ""}

    ui_print(f"[bold green]Final Planned Goals:[/bold green] {sub_goals}")

    final_status = "success"
    final_summary = "All goals completed."

    for current_goal in sub_goals:
        ui_print(Panel.fit(escape(current_goal), title="Current Sub-Goal", border_style="cyan"))
        memory = Memory(current_goal, db_path="memory.db")
        tools = ToolDispatcher(browser=browser, llm_client=llm, console=console)
        tools.ask_user = lambda q: {"ok": True, "message": "user answered", "data": {"answer": wait_for_user_input(q)}}

        try:
            goal_completed = False

            while not goal_completed:
                steps_taken = 0


                for step in range(1, max_steps + 1):
                    steps_taken = step

                    if handle_interrupt(memory):
                        return {"ok": False, "status": "stopped_by_user", "summary": "Interrupted", "report_path": ""}

                    obs = browser.observe()
                    memory.update_observation(obs)
                    tools.set_observation(obs)
                    ref_count = obs.get("snapshot_yaml", "").count("[ref=")
                    ui_print(
                        f"\n[bold]Step {step}/{max_steps}[/bold] "
                        f"[dim]URL:[/dim] {escape(obs.get('url', ''))} "
                        f"[dim]Refs:[/dim] {ref_count}"
                    )

                    action = llm.plan(memory.to_prompt_payload(), memory.get_current_screenshot())
                    ui_print(f"[bold green]Assistant:[/bold green] {escape(str(action.get('thought', '')))}")
                    ui_print(f"[bold blue]Using tool:[/bold blue] {escape(str(action.get('tool', '')))}")

                    result = _execute_with_safety(action, obs, tools, console)

                    memory.merge_facts(action.get("new_facts", {}))
                    memory.add_action(action, result)
                    append_action_log(step, action, result, obs)

                    if _is_llm_provider_fallback_action(action):
                        decision = _llm_provider_fallback_decision(result)
                        if decision == "retry":
                            continue
                        if decision == "stop":
                            return {"ok": False, "status": "stopped_by_user", "summary": "Stopped.", "report_path": ""}
                        continue

                    if action.get("tool") == "done":
                        goal_completed = True
                        break

                if not goal_completed and steps_taken >= max_steps:
                    ui_print(f"[bold red]Stuck Detection:[/bold red] Goal exceeded {max_steps} steps.")
                    ans = wait_for_user_input("Agent is stuck. Enter new instructions to reset and continue, or 'stop' to abort:")
                    if ans.lower().strip() == 'stop':
                        final_status = "failed"
                        final_summary = f"Stopped after {max_steps} steps."
                        goal_completed = True # Break outer while
                    else:
                        ui_print("[bold green]Resetting step limit and continuing with new instructions.[/bold green]")
                        memory.goal = memory.goal + " | Additional user instructions: " + ans

        except Exception as e:
            ui_print(f"[red]Error:[/red] {e}")
            return {"ok": False, "status": "error", "summary": str(e), "report_path": ""}

        if final_status != "success":
            break

    ui_print(f"\n[bold green]Final report[/bold green]\n\n- Status: {final_status}\n- Goal: {goal}\n\n## Summary\n\n{final_summary}")
    return {"ok": final_status == "success", "status": final_status, "summary": final_summary, "report_path": ""}

def _orchestrate_goals(goal: str, llm: LLMClient) -> list[str]:
    context = goal
    while True:
        messages = [
            {"role": "system", "content": "You are a planner. Either ask the user a clarifying question starting with 'QUESTION: ', or if the task is clear, output ONLY a JSON list of 1-3 sequential strings representing sub-goals."},
            {"role": "user", "content": f"Task/Context: {context}"}
        ]
        try:
            models = llm._candidate_models(llm.model)
            for model in models:
                res = llm._chat_completion_with_retries(model=model, messages=messages, temperature=0.1)
                content = res.choices[0].message.content or ""

                if content.strip().startswith("QUESTION:"):
                    ans = wait_for_user_input(content.strip())
                    if ans.lower().strip() == 'stop':
                        return []
                    context += f"\nQ: {content}\nA: {ans}"
                    break # Go to next while iteration

                # Try extract JSON
                start = content.find("[")
                end = content.rfind("]")
                if start != -1 and end != -1:
                    return json.loads(content[start:end+1])
        except Exception:
            pass

        # Fallback to simple split if LLM fails
        if " and " in goal:
            return goal.split(" and ")
        return [goal]

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
        ans = wait_for_user_input(f"High risk action: {reason}. Execute? [y/N]")
        if not user_confirmed(ans):
            return {"ok": False, "message": "declined", "data": {}}

    return tools.dispatch(action)

def pretty_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, indent=2, default=str)

def _is_llm_provider_fallback_action(action: dict[str, Any]) -> bool:
    return action.get("tool") == "ask_user" and "LLM provider" in str(action.get("args", {}).get("question", ""))

def _llm_provider_fallback_decision(result: dict[str, Any]) -> str:
    answer = str(result.get("data", {}).get("answer", "")).strip().lower()
    if answer in LLM_FALLBACK_RETRY_ANSWERS:
        return "retry"
    if answer in LLM_FALLBACK_STOP_ANSWERS:
        return "stop"
    return "continue"
