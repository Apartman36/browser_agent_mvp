from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_ID_FORMAT = "%Y-%m-%d_%H-%M-%S"


@dataclass
class RunContext:
    """Owns paths for a single agent run."""

    run_id: str
    run_dir: Path
    started_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def actions_log_path(self) -> Path:
        return self.run_dir / "actions.jsonl"

    @property
    def safety_audit_path(self) -> Path:
        return self.run_dir / "safety_audit.jsonl"

    @property
    def final_report_path(self) -> Path:
        return self.run_dir / "final_report.md"

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def screenshots_dir(self) -> Path:
        return self.run_dir / "screenshots"

    def write_initial_metadata(self, fields: dict[str, Any]) -> None:
        sanitized = _sanitize_metadata(fields)
        sanitized.setdefault("run_id", self.run_id)
        sanitized.setdefault("started_at", self.started_at)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def update_metadata(self, fields: dict[str, Any]) -> None:
        existing: dict[str, Any] = {}
        if self.metadata_path.exists():
            try:
                existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        existing.update(_sanitize_metadata(fields))
        self.metadata_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def create_run_context(
    root: str | Path = "logs/runs",
    *,
    now: datetime | None = None,
) -> RunContext:
    """Create a unique timestamped run directory under ``root``.

    Defaults preserve current behavior: no environment-specific data is read here.
    """

    moment = now or datetime.now()
    base = moment.strftime(RUN_ID_FORMAT)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    candidate = root_path / base
    counter = 1
    while candidate.exists():
        candidate = root_path / f"{base}_{counter:03d}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=True)
    (candidate / "screenshots").mkdir(parents=True, exist_ok=True)
    return RunContext(
        run_id=candidate.name,
        run_dir=candidate,
        started_at=moment.replace(microsecond=0).isoformat(),
    )


_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "openrouter_api_key",
        "api_key",
        "apikey",
        "openai_api_key",
        "cookies",
        "cookie",
        "authorization",
        "secret",
        "password",
        "token",
        "env",
        "environ",
    }
)


def _sanitize_metadata(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop obviously sensitive keys before writing metadata."""

    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        if not isinstance(key, str):
            continue
        if key.lower() in _FORBIDDEN_METADATA_KEYS:
            continue
        if "api_key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
            continue
        sanitized[key] = value
    return sanitized
