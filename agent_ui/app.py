from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from agent_ui.run_manager import RunAlreadyActiveError, RunManager, RunNotFoundError
from agent_ui.sse import format_sse


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"


class StartRunRequest(BaseModel):
    task: str = Field(min_length=1)
    start_url: str | None = None
    max_steps: int = Field(default=20, ge=1, le=100)

    @field_validator("task")
    @classmethod
    def _strip_task(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("task is required")
        return value

    @field_validator("start_url")
    @classmethod
    def _strip_start_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class SafetyResponseRequest(BaseModel):
    prompt_id: str = Field(min_length=1)
    answer: str = Field(min_length=1)


def generate_token() -> str:
    return os.getenv("AGENT_UI_TOKEN") or secrets.token_urlsafe(32)


def create_app(manager: RunManager | None = None, token: str | None = None) -> FastAPI:
    app = FastAPI(title="Browser Agent Local UI")
    app.state.run_manager = manager or RunManager()
    app.state.agent_ui_token = token or generate_token()

    def require_token(
        request: Request,
        token_query: Annotated[str | None, Query(alias="token")] = None,
        x_agent_token: Annotated[str | None, Header(alias="X-Agent-Token")] = None,
    ) -> None:
        supplied = token_query or x_agent_token or ""
        expected = request.app.state.agent_ui_token
        if not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    @app.get("/healthz")
    async def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/runs", dependencies=[Depends(require_token)])
    async def start_run(request: StartRunRequest) -> dict[str, object]:
        try:
            run = app.state.run_manager.start_run(
                task=request.task,
                start_url=request.start_url,
                max_steps=request.max_steps,
            )
        except RunAlreadyActiveError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return run.to_dict()

    @app.get("/runs/{run_id}", dependencies=[Depends(require_token)])
    async def get_run(run_id: str) -> dict[str, object]:
        try:
            return app.state.run_manager.get_run(run_id).to_dict()
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.post("/runs/{run_id}/cancel", dependencies=[Depends(require_token)])
    async def cancel_run(run_id: str) -> dict[str, object]:
        try:
            run = await app.state.run_manager.cancel_run(run_id)
            return run.to_dict()
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/events", dependencies=[Depends(require_token)])
    async def events(run_id: str) -> StreamingResponse:
        try:
            app.state.run_manager.get_run(run_id)
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

        async def stream():
            async for event in app.state.run_manager.event_stream(run_id):
                yield format_sse(event.event_type, event.to_payload())

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/runs/{run_id}/respond", dependencies=[Depends(require_token)])
    async def respond_to_safety_prompt(run_id: str, request: SafetyResponseRequest) -> dict[str, bool]:
        try:
            accepted = app.state.run_manager.respond_to_safety_prompt(
                run_id=run_id,
                prompt_id=request.prompt_id,
                answer=request.answer,
            )
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        if not accepted:
            raise HTTPException(status_code=404, detail="safety prompt not found")
        return {"ok": True}

    @app.get("/screenshots/{filename}", dependencies=[Depends(require_token)])
    async def screenshot(filename: str) -> FileResponse:
        if filename != Path(filename).name:
            raise HTTPException(status_code=404, detail="screenshot not found")
        screenshot_dir = (Path.cwd() / "logs" / "screenshots").resolve()
        path = (screenshot_dir / filename).resolve()
        if screenshot_dir not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="screenshot not found")
        return FileResponse(path)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
