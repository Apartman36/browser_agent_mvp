from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _force_utf8_io() -> None:
    """Force stdin/stdout/stderr to UTF-8 so logs and reports don't get mojibake on Windows.

    Why: on Windows the default for `sys.stdin/stdout/stderr` is the system ANSI code page
    (e.g. cp1251), so any Russian/Unicode text passing through `input()` or `print()` can be
    re-encoded incorrectly before reaching disk. Forcing UTF-8 here keeps both the terminal
    and the files written under `logs/` consistent.
    """
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_io()

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402

from agent.browser import Browser  # noqa: E402
from agent.core import run_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the browser automation AI agent.")
    parser.add_argument("task", nargs="+", help="Natural-language task for the agent.")
    parser.add_argument("--start-url", help="URL to open before starting the agent.")
    parser.add_argument("--max-steps", type=int, help="Maximum agent loop steps.")
    parser.add_argument("--login-wait", action="store_true", help="Pause for manual login before agent starts.")
    parser.add_argument("--dry-run", action="store_true", help="Open browser, observe once, and exit without LLM calls.")
    return parser.parse_args()


def main() -> int:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    args = parse_args()
    console = Console()

    goal = " ".join(args.task)
    start_url = args.start_url or os.getenv("START_URL", "")
    max_steps = args.max_steps or int(os.getenv("MAX_STEPS", "25"))

    with Browser() as browser:
        if start_url:
            console.print(f"[bold blue]Opening:[/bold blue] {start_url}")
            result = browser.goto(start_url)
            console.print(f"[dim]{result.get('message')}[/dim]")

        if args.login_wait:
            input("Log in manually if needed, then press Enter\n> ")

        if args.dry_run:
            obs = browser.observe()
            summary = (
                f"URL: {obs.get('url', '')}\n"
                f"Title: {obs.get('title', '')}\n"
                f"Refs: {obs.get('snapshot_yaml', '').count('[ref=')}\n"
                "No OpenRouter call was made."
            )
            console.print(Panel(summary, title="Dry run observation", border_style="cyan"))
            return 0

        result = run_agent(goal=goal, browser=browser, max_steps=max_steps)
        return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

