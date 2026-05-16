from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from agent_ui.events import make_event
from agent_ui.run_manager import RunAlreadyActiveError, RunManager


def test_start_run_creates_running_run_and_rejects_second_active_run() -> None:
    release = threading.Event()

    def runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        release.wait(timeout=2)
        return {"ok": True, "status": "success", "summary": "done"}

    async def scenario() -> None:
        manager = RunManager(runner=runner)
        run = manager.start_run("Do one thing", max_steps=3)

        assert run.run_id
        assert run.status == "running"
        assert run.max_steps == 3
        with pytest.raises(RunAlreadyActiveError):
            manager.start_run("Do another thing")

        release.set()
        await asyncio.wait_for(run.background_task, timeout=3)

    asyncio.run(scenario())

def test_publish_and_event_stream_return_events_in_order_and_store_final_report() -> None:
    release = threading.Event()

    def runner(*args: Any, **kwargs: Any) -> dict[str, Any]:
        release.wait(timeout=2)
        return {"ok": True, "status": "success", "summary": "runner done"}

    async def scenario() -> None:
        manager = RunManager(runner=runner)
        run = manager.start_run("Watch events")
        await manager.publish(run.run_id, make_event("observation", {"step": 1}))
        await manager.publish(run.run_id, make_event("tool_call", {"tool": "done"}))
        await manager.publish(run.run_id, make_event("done", {"summary": "final text"}))

        event_types = []
        async for event in manager.event_stream(run.run_id):
            event_types.append(event.event_type)

        assert event_types == ["observation", "tool_call", "done"]
        assert run.status == "done"
        assert run.final_report == "final text"

        release.set()
        await asyncio.wait_for(run.background_task, timeout=3)

    asyncio.run(scenario())


def test_cancel_transitions_state_and_releases_runner() -> None:
    def runner(
        task: str,
        start_url: str | None,
        max_steps: int,
        event_sink: Any,
        should_cancel: Any,
        safety_confirmation: Any,
    ) -> dict[str, Any]:
        while not should_cancel():
            time.sleep(0.01)
        return {"ok": False, "status": "cancelled", "summary": "cancelled"}

    async def scenario() -> None:
        manager = RunManager(runner=runner)
        run = manager.start_run("Cancel me")

        cancelled = await manager.cancel_run(run.run_id)
        assert cancelled.status == "cancelled"
        assert cancelled.final_report == "Cancellation requested."

        event = await asyncio.wait_for(run.queue.get(), timeout=1)
        assert event.event_type == "cancelled"
        await asyncio.wait_for(run.background_task, timeout=3)

    asyncio.run(scenario())
