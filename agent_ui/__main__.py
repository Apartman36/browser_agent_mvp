from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

from agent_ui.app import DEFAULT_HOST, DEFAULT_PORT, create_app, generate_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experimental browser agent local web UI.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local port to bind on 127.0.0.1.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    args = parse_args()
    token = generate_token()
    app = create_app(token=token)

    if hasattr(asyncio, "WindowsProactorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    print("Experimental browser agent UI")
    print("Localhost only: http://127.0.0.1 traffic stays on this machine.")
    print(f"Open: http://{DEFAULT_HOST}:{args.port}/?token={token}")
    uvicorn.run(app, host=DEFAULT_HOST, port=args.port)


if __name__ == "__main__":
    main()
