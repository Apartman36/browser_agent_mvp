from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from agent.browser import Browser
from agent.core import run_agent


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

