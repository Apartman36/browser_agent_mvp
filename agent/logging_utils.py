from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_DIR = Path("logs")


def ensure_logs_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def append_action_log(
    step: int,
    action: dict[str, Any],
    result: dict[str, Any],
    observation: dict[str, Any] | None,
) -> None:
    ensure_logs_dir()
    obs = observation or {}
    record = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "action": action,
        "result": result,
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
    }
    # newline="\n" prevents Windows from rewriting "\n" as "\r\n" inside a JSONL line.
    with (LOG_DIR / "actions.jsonl").open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_final_report(goal: str, status: str, summary: str) -> Path:
    ensure_logs_dir()
    path = LOG_DIR / "final_report.md"
    content = (
        "# Final report\n\n"
        f"- Status: {status}\n"
        f"- Goal: {goal}\n"
        f"- Finished at: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        "## Summary\n\n"
        f"{summary.strip()}\n"
    )
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def read_action_log() -> list[dict[str, Any]]:
    path = LOG_DIR / "actions.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

