from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_ui.events import AgentUIEvent, TERMINAL_EVENT_TYPES, make_event


EventSink = Callable[[str, dict[str, Any]], None]
CancelChecker = Callable[[], bool]
SafetyConfirmation = Callable[[dict[str, Any]], str]
AgentRunner = Callable[
    [str, str | None, int, EventSink, CancelChecker, SafetyConfirmation],
    dict[str, Any],
]


class RunAlreadyActiveError(RuntimeError):
    pass


class RunNotFoundError(KeyError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Run:
    run_id: str
    task: str
    status: str
    started_at: datetime
    start_url: str | None = None
    max_steps: int = 20
    ended_at: datetime | None = None
    final_report: str | None = None
    report_path: str | None = None
    queue: asyncio.Queue[AgentUIEvent] = field(default_factory=asyncio.Queue)
    background_task: asyncio.Task[None] | None = None
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    events: list[AgentUIEvent] = field(default_factory=list)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal_event_sent: bool = False
    pending_prompts: dict[str, Future[str]] = field(default_factory=dict)
    prompt_lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status,
            "start_url": self.start_url,
            "max_steps": self.max_steps,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "final_report": self.final_report,
            "report_path": self.report_path,
        }


class RunManager:
    def __init__(
        self,
        runner: AgentRunner | None = None,
        safety_timeout_seconds: float = 300.0,
    ) -> None:
        self._runs: dict[str, Run] = {}
        self._active_run_id: str | None = None
        self._runner = runner or run_browser_agent
        self._safety_timeout_seconds = safety_timeout_seconds

    def start_run(self, task: str, start_url: str | None = None, max_steps: int = 20) -> Run:
        task = task.strip()
        if not task:
            raise ValueError("task is required")
        self._clear_finished_active_run()
        if self._active_run_id is not None:
            raise RunAlreadyActiveError("another run is already active")

        run = Run(
            run_id=uuid.uuid4().hex,
            task=task,
            status="running",
            started_at=_utc_now(),
            start_url=start_url or None,
            max_steps=max_steps,
        )
        self._runs[run.run_id] = run
        self._active_run_id = run.run_id

        loop = asyncio.get_running_loop()
        run.background_task = loop.create_task(self._execute_run(run, loop))
        return run

    def get_run(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        return run

    async def cancel_run(self, run_id: str) -> Run:
        run = self.get_run(run_id)
        run.cancel_requested.set()
        self._answer_pending_prompts(run, "n")
        if run.status == "running":
            await self.publish(
                run_id,
                make_event("cancelled", {"message": "Cancellation requested."}),
            )
        return run

    async def publish(self, run_id: str, event: AgentUIEvent) -> None:
        run = self.get_run(run_id)
        event = event.with_run_id(run_id)
        self._apply_event_to_state(run, event)
        run.events.append(event)
        await run.queue.put(event)
        async with run.condition:
            run.condition.notify_all()

    async def event_stream(self, run_id: str) -> AsyncIterator[AgentUIEvent]:
        run = self.get_run(run_id)
        index = 0
        while True:
            while index < len(run.events):
                event = run.events[index]
                index += 1
                yield event
                if event.event_type in TERMINAL_EVENT_TYPES:
                    return
            if run.terminal_event_sent:
                return
            async with run.condition:
                await run.condition.wait()

    def respond_to_safety_prompt(self, run_id: str, prompt_id: str, answer: str) -> bool:
        run = self.get_run(run_id)
        with run.prompt_lock:
            pending = run.pending_prompts.get(prompt_id)
        if pending is None or pending.done():
            return False
        pending.set_result(answer)
        return True

    async def _execute_run(self, run: Run, loop: asyncio.AbstractEventLoop) -> None:
        event_sink = self._make_event_sink(run, loop)
        safety_confirmation = self._make_safety_confirmation(run, event_sink)
        try:
            result = await asyncio.to_thread(
                self._runner,
                run.task,
                run.start_url,
                run.max_steps,
                event_sink,
                run.cancel_requested.is_set,
                safety_confirmation,
            )
        except Exception as exc:
            if not run.terminal_event_sent:
                await self.publish(run.run_id, make_event("error", {"message": str(exc)}))
        else:
            if run.terminal_event_sent:
                return
            if run.cancel_requested.is_set() or result.get("status") == "cancelled":
                await self.publish(run.run_id, make_event("cancelled", result))
            else:
                await self.publish(run.run_id, make_event("done", result))
        finally:
            self._answer_pending_prompts(run, "n")
            if self._active_run_id == run.run_id:
                self._active_run_id = None

    def _make_event_sink(self, run: Run, loop: asyncio.AbstractEventLoop) -> EventSink:
        def publish_from_thread(event_type: str, data: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(
                self.publish(run.run_id, make_event(event_type, data)),
                loop,
            )
            try:
                future.result(timeout=5)
            except Exception:
                return

        return publish_from_thread

    def _make_safety_confirmation(self, run: Run, event_sink: EventSink) -> SafetyConfirmation:
        def confirm(prompt: dict[str, Any]) -> str:
            prompt_id = uuid.uuid4().hex
            future: Future[str] = Future()
            with run.prompt_lock:
                run.pending_prompts[prompt_id] = future
            payload = dict(prompt)
            payload["prompt_id"] = prompt_id
            event_sink("safety_prompt", payload)
            try:
                return future.result(timeout=self._safety_timeout_seconds)
            except TimeoutError:
                return "n"
            finally:
                with run.prompt_lock:
                    run.pending_prompts.pop(prompt_id, None)

        return confirm

    def _answer_pending_prompts(self, run: Run, answer: str) -> None:
        with run.prompt_lock:
            pending = list(run.pending_prompts.values())
        for future in pending:
            if not future.done():
                future.set_result(answer)

    def _apply_event_to_state(self, run: Run, event: AgentUIEvent) -> None:
        data = event.data
        if event.event_type == "done":
            run.status = "done"
            run.ended_at = run.ended_at or _utc_now()
            run.final_report = data.get("summary") or data.get("final_report")
            run.report_path = data.get("report_path")
            run.terminal_event_sent = True
        elif event.event_type == "error":
            run.status = "error"
            run.ended_at = run.ended_at or _utc_now()
            run.final_report = data.get("message")
            run.terminal_event_sent = True
        elif event.event_type == "cancelled":
            run.status = "cancelled"
            run.ended_at = run.ended_at or _utc_now()
            run.final_report = data.get("summary") or data.get("message")
            run.report_path = data.get("report_path")
            run.terminal_event_sent = True

    def _clear_finished_active_run(self) -> None:
        if self._active_run_id is None:
            return
        active = self._runs.get(self._active_run_id)
        if active is None:
            self._active_run_id = None
            return
        if active.background_task is not None and active.background_task.done():
            self._active_run_id = None


def run_browser_agent(
    task: str,
    start_url: str | None,
    max_steps: int,
    event_sink: EventSink,
    should_cancel: CancelChecker,
    safety_confirmation: SafetyConfirmation,
) -> dict[str, Any]:
    from agent.browser import Browser
    from agent.core import run_agent

    with Browser() as browser:
        if start_url:
            event_sink("tool_call", {"step": 0, "tool": "goto", "args": {"url": start_url}})
            result = browser.goto(start_url)
            event_sink(
                "tool_result",
                {
                    "step": 0,
                    "tool": "goto",
                    "ok": bool(result.get("ok", False)),
                    "message": str(result.get("message", "")),
                    "data": result.get("data", {}),
                },
            )
        if should_cancel():
            return {"ok": False, "status": "cancelled", "summary": "Cancelled before agent start."}
        return run_agent(
            goal=task,
            browser=browser,
            max_steps=max_steps,
            event_sink=event_sink,
            should_cancel=should_cancel,
            safety_confirmation=safety_confirmation,
        )
