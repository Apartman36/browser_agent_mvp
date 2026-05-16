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
from agent.config import load_config  # noqa: E402
from agent.core import run_agent  # noqa: E402
from agent.run_context import create_run_context  # noqa: E402


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
    config = load_config()
    start_url = args.start_url or config.start_url
    max_steps = args.max_steps or config.max_steps

    run_ctx = create_run_context(config.run_log_root)
    console.print(f"[bold]Browser mode:[/bold] {config.browser.mode}")
    console.print(f"[bold]Profile dir:[/bold] {config.browser.active_user_data_dir()}")
    if config.browser.active_channel():
        console.print(f"[bold]Chrome channel:[/bold] {config.browser.active_channel()}")
    console.print(f"[bold]Run dir:[/bold] {run_ctx.run_dir}")

    with Browser(runtime=config.browser, screenshot_dir=run_ctx.screenshots_dir) as browser:
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
            console.print(f"[dim]Saved run logs:[/dim] {run_ctx.run_dir}")
            return 0

        result = run_agent(
            goal=goal,
            browser=browser,
            max_steps=max_steps,
            run_context=run_ctx,
            start_url=start_url,
        )
        return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
