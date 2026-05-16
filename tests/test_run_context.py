from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from agent.logging_utils import append_action_log, write_final_report
from agent.run_context import create_run_context


def test_create_run_context_makes_unique_timestamped_dir(tmp_path: Path) -> None:
    moment = datetime(2026, 5, 16, 21, 43, 12)
    first = create_run_context(tmp_path, now=moment)
    second = create_run_context(tmp_path, now=moment)

    assert first.run_id == "2026-05-16_21-43-12"
    assert second.run_id.startswith("2026-05-16_21-43-12_")
    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()
    assert first.screenshots_dir.exists()


def test_metadata_includes_required_fields(tmp_path: Path) -> None:
    ctx = create_run_context(tmp_path, now=datetime(2026, 5, 16, 12, 0, 0))
    ctx.write_initial_metadata(
        {
            "run_id": ctx.run_id,
            "started_at": ctx.started_at,
            "status": "running",
            "goal": "Open example.com and stop.",
            "max_steps": 3,
            "start_url": "https://example.com",
            "planner_mode": "auto",
            "model": "openrouter/some-model",
            "browser_mode": "chrome_profile",
            "browser_channel": "chrome",
            "browser_profile_dir": str(tmp_path / "agent-profile"),
            "browser_slow_mo_ms": 150,
            "action_min_delay_ms": 500,
            "action_max_delay_ms": 1500,
        }
    )

    data = json.loads(ctx.metadata_path.read_text(encoding="utf-8"))
    assert data["run_id"] == ctx.run_id
    assert data["browser_mode"] == "chrome_profile"
    assert data["browser_channel"] == "chrome"
    assert data["browser_profile_dir"].endswith("agent-profile")
    assert data["action_min_delay_ms"] == 500
    assert data["action_max_delay_ms"] == 1500
    assert data["planner_mode"] == "auto"
    assert data["max_steps"] == 3
    assert data["start_url"] == "https://example.com"


def test_metadata_does_not_leak_api_keys(tmp_path: Path) -> None:
    ctx = create_run_context(tmp_path, now=datetime(2026, 5, 16, 12, 0, 0))
    ctx.write_initial_metadata(
        {
            "goal": "test",
            "openrouter_api_key": "sk-should-never-be-saved",
            "API_KEY": "also-secret",
            "anthropic_token": "another-secret",
            "model": "ok-model",
        }
    )

    raw = ctx.metadata_path.read_text(encoding="utf-8")
    assert "sk-should-never-be-saved" not in raw
    assert "also-secret" not in raw
    assert "another-secret" not in raw
    assert "ok-model" in raw


def test_update_metadata_merges_end_fields(tmp_path: Path) -> None:
    ctx = create_run_context(tmp_path, now=datetime(2026, 5, 16, 12, 0, 0))
    ctx.write_initial_metadata({"goal": "task", "status": "running"})
    ctx.update_metadata({"status": "success", "ended_at": "2026-05-16T12:00:30"})

    data = json.loads(ctx.metadata_path.read_text(encoding="utf-8"))
    assert data["goal"] == "task"
    assert data["status"] == "success"
    assert data["ended_at"] == "2026-05-16T12:00:30"


def test_append_action_log_writes_under_run_dir(tmp_path: Path) -> None:
    ctx = create_run_context(tmp_path, now=datetime(2026, 5, 16, 12, 0, 0))

    append_action_log(
        step=1,
        action={"tool": "observe", "args": {}},
        result={"ok": True, "message": "observed", "data": {}},
        observation={"url": "https://example.com", "title": "Example"},
        log_path=ctx.actions_log_path,
    )

    lines = ctx.actions_log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["step"] == 1
    assert record["action"]["tool"] == "observe"
    assert record["url"] == "https://example.com"


def test_write_final_report_writes_under_run_dir(tmp_path: Path) -> None:
    ctx = create_run_context(tmp_path, now=datetime(2026, 5, 16, 12, 0, 0))

    path = write_final_report("the goal", "success", "all good", report_path=ctx.final_report_path)

    assert path == ctx.final_report_path
    text = path.read_text(encoding="utf-8")
    assert "the goal" in text
    assert "success" in text
    assert "all good" in text
