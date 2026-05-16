from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient

from agent_ui.app import create_app
from agent_ui.run_manager import RunManager


TOKEN = "test-token"


def fake_runner(
    task: str,
    start_url: str | None,
    max_steps: int,
    event_sink: Any,
    should_cancel: Any,
    safety_confirmation: Any,
) -> dict[str, Any]:
    event_sink("run_started", {"task": task})
    event_sink(
        "observation",
        {"step": 1, "url": start_url or "", "title": "Mock page", "refs_count": 0},
    )
    return {"ok": True, "status": "success", "summary": "fake final", "report_path": "logs/final.md"}


def make_client(runner: Any = fake_runner) -> TestClient:
    return TestClient(create_app(manager=RunManager(runner=runner), token=TOKEN))


def test_healthz_works_without_token() -> None:
    with make_client() as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_token_required_for_protected_routes() -> None:
    with make_client() as client:
        response = client.post("/runs", json={"task": "Do it"})

    assert response.status_code == 401


def test_post_run_and_get_status() -> None:
    with make_client() as client:
        response = client.post(
            f"/runs?token={TOKEN}",
            json={"task": "Do it", "start_url": "https://example.com", "max_steps": 2},
        )
        assert response.status_code == 200
        run_id = response.json()["run_id"]

        status = client.get(f"/runs/{run_id}?token={TOKEN}")
    assert status.status_code == 200
    assert status.json()["run_id"] == run_id


def test_events_streams_mocked_run_events() -> None:
    with make_client() as client:
        response = client.post(f"/runs?token={TOKEN}", json={"task": "Do it"})
        run_id = response.json()["run_id"]

        with client.stream("GET", f"/events?run_id={run_id}&token={TOKEN}") as stream:
            body = "".join(stream.iter_text())

    assert "event: run_started" in body
    assert "event: observation" in body
    assert "event: done" in body
    assert "fake final" in body


def test_cancel_changes_status() -> None:
    def cancelable_runner(
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

    with make_client(runner=cancelable_runner) as client:
        response = client.post(f"/runs?token={TOKEN}", json={"task": "Cancel it"})
        run_id = response.json()["run_id"]

        cancelled = client.post(f"/runs/{run_id}/cancel?token={TOKEN}")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
