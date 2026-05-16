from __future__ import annotations

from pathlib import Path


def test_readme_has_no_demo_todo_placeholders() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "Video: TODO" not in text
    assert "GitHub: TODO" not in text
