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
    *,
    log_path: Path | None = None,
) -> None:
    obs = observation or {}
    record = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "action": action,
        "result": result,
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
    }
    serialized = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    target = Path(log_path) if log_path is not None else (ensure_logs_dir() / "actions.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" prevents Windows from rewriting "\n" as "\r\n" inside a JSONL line.
    with target.open("a", encoding="utf-8", newline="\n") as file:
        file.write(serialized)


def write_final_report(
    goal: str,
    status: str,
    summary: str,
    *,
    report_path: Path | None = None,
) -> Path:
    content = (
        "# Final report\n\n"
        f"- Status: {status}\n"
        f"- Goal: {goal}\n"
        f"- Finished at: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        "## Summary\n\n"
        f"{summary.strip()}\n"
    )
    target = Path(report_path) if report_path is not None else (ensure_logs_dir() / "final_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")
    return target


def read_action_log(path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path is not None else (LOG_DIR / "actions.jsonl")
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
