"""Authenticated UI and OpenAI-compatible API for supervised desktop tasks."""

import argparse
import asyncio
import base64
import html
import io
import json
import platform
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from gui_agents.s3.screens import monitors
from gui_agents.s3.task_manager import ACTIVE_STATES, TaskConflict, TaskManager
from gui_agents.s3.ui_config import (
    AgentConfig,
    ConfigStore,
    PROVIDERS,
    backend,
    presets,
    request_target,
    server_token,
)

UI_DIR = Path(__file__).resolve().parent / "ui"


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    instruction: str = Field(min_length=1, max_length=8192)
    config: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    step: int = Field(ge=1, le=60)
    approval_token: str = Field(min_length=32, max_length=32)
    approved: bool


class ConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    aspect: Literal["planner", "grounding"] = "planner"
    provider: str = ""
    base_url: str = ""
    api_key: str = Field(default="", max_length=4096, repr=False)
    model: str = ""
    verify_vision: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    model: str = ""
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=256)
    stream: bool = False
    agent: bool = False
    max_steps: int | None = Field(default=None, ge=1, le=60)


def create_app(
    config_path=None, token=None, manager=None, allowed_hosts=None, transport=None
):
    store = ConfigStore(config_path)
    tasks = manager or TaskManager()
    auth_token = token if token is not None else server_token()
    if not auth_token:
        raise ValueError("The UI server requires a nonempty authentication token")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(
            transport=transport, follow_redirects=False, trust_env=False
        ) as client:
            app.state.client = client
            yield
        await asyncio.to_thread(tasks.shutdown)

    app = FastAPI(title="Agent S3 supervised desktop API", lifespan=lifespan)
    app.state.store, app.state.tasks, app.state.token = store, tasks, auth_token
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts or ["localhost", "127.0.0.1", "[::1]"],
    )

    @app.middleware("http")
    async def request_boundaries(request, call_next):
        origin = request.headers.get("origin")
        own_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin and origin != own_origin:
            return JSONResponse(
                {"detail": "Cross-origin requests are not allowed"}, status_code=403
            )
        if request.method in ("POST", "PUT", "PATCH"):
            limit = (
                8 * 1024 * 1024 if request.url.path.startswith("/v1/") else 128 * 1024
            )
            chunks, length = [], 0
            async for chunk in request.stream():
                length += len(chunk)
                if length > limit:
                    return JSONResponse(
                        {"detail": "Request body is too large"}, status_code=413
                    )
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    def authorize(request: Request):
        value = request.headers.get("authorization", "")
        supplied = value[7:] if value.lower().startswith("bearer ") else ""
        if not secrets.compare_digest(supplied, auth_token):
            raise HTTPException(401, "A valid Agent S server token is required")

    protected = [Depends(authorize)]

    @app.exception_handler(ValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            {"detail": json.loads(exc.json(include_input=False, include_url=False))},
            status_code=422,
        )

    @app.exception_handler(TaskConflict)
    async def task_conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing_task(request, exc):
        return JSONResponse({"detail": "Unknown task"}, status_code=404)

    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = (UI_DIR / "index.html").read_text(encoding="utf-8")
        response = HTMLResponse(
            page.replace(
                "<!--AGENT_S_AUTH-->",
                f'<meta name="agent-s-token" content="{html.escape(auth_token, quote=True)}" />',
            )
        )
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/api/status", dependencies=protected)
    def status():
        try:
            choices = monitors()
            screen = next(
                (m for m in choices if m["id"] == store.snapshot().monitor), choices[0]
            )
        except Exception:
            choices, screen = [], {"width": 0, "height": 0}
        return {
            "platform": platform.system().lower(),
            "screen": screen,
            "monitors": choices,
            "active_tasks": tasks.active(),
            "server_time": time.time(),
        }

    @app.get("/api/providers", dependencies=protected)
    def providers():
        return {"engine_types": PROVIDERS, "presets": presets()}

    @app.get("/api/config", dependencies=protected)
    def get_config():
        return store.snapshot().public()

    @app.post("/api/config", dependencies=protected)
    def update_config(body: dict[str, Any]):
        return store.update(body)

    async def discover(body):
        if body.provider and body.provider not in PROVIDERS:
            raise HTTPException(422, "Unsupported provider")
        try:
            resolved = backend(store.snapshot(), body.aspect, body.model_dump())
            url, headers = request_target(resolved, "models")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        async with app.state.client.stream(
            "GET", url, headers=headers, timeout=store.snapshot().inference_timeout
        ) as response:
            response.raise_for_status()
            payload = json.loads(await read_limited(response, 2 * 1024 * 1024))
        entries = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(entries, list) or len(entries) > 10000:
            raise ValueError("Backend returned an invalid model list")
        ids = [
            (
                entry
                if isinstance(entry, str)
                else entry.get("id") if isinstance(entry, dict) else None
            )
            for entry in entries
        ]
        if any(not isinstance(value, str) for value in ids):
            raise ValueError("Backend model IDs must be strings")
        return resolved, ids

    @app.post("/api/models/list", dependencies=protected)
    async def list_models(body: ConnectionRequest):
        try:
            _, ids = await discover(body)
            return {"ok": True, "models": ids, "raw": [{"id": value} for value in ids]}
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, f"Model discovery failed: {exc}") from exc

    @app.post("/api/connection/test", dependencies=protected)
    async def test_connection(body: ConnectionRequest):
        try:
            effective = backend(store.snapshot(), body.aspect, body.model_dump())
            if effective["provider"] == "azure" and body.verify_vision:
                resolved, ids = effective, [effective["model"]]
            else:
                resolved, ids = await discover(body)
            accepted = False
            if body.verify_vision:
                if not resolved["model"]:
                    raise ValueError("Pick a model ID before testing vision")
                from PIL import Image

                output = io.BytesIO()
                Image.new("RGB", (32, 32), "blue").save(output, "PNG")
                image = (
                    "data:image/png;base64,"
                    + base64.b64encode(output.getvalue()).decode()
                )
                payload = {
                    "model": resolved["model"],
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe the color of this image in one word.",
                                },
                                {"type": "image_url", "image_url": {"url": image}},
                            ],
                        }
                    ],
                    "max_tokens": 16,
                }
                url, headers = request_target(resolved, "chat/completions")
                response = await app.state.client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=store.snapshot().inference_timeout,
                )
                response.raise_for_status()
                accepted = bool(response.json().get("choices"))
            return {
                "ok": True,
                "aspect": body.aspect,
                "models": ids,
                "count": len(ids),
                "vision_request_accepted": accepted,
                "hint": "Discovery/vision acceptance does not measure grounding accuracy. Verify UI-TARS weights, mmproj, and coordinates separately.",
            }
        except (httpx.HTTPError, ValueError) as exc:
            return {"ok": False, "aspect": body.aspect, "error": str(exc)[:1000]}

    def task_config(overrides):
        data = store.snapshot().model_dump()
        for provider, url, key in (
            ("provider", "model_url", "model_api_key"),
            ("ground_provider", "ground_url", "ground_api_key"),
        ):
            if provider in overrides and overrides[provider] != data[provider]:
                from gui_agents.s3.ui_config import DEFAULT_URLS

                data[url], data[key] = DEFAULT_URLS.get(overrides[provider], ""), ""
        data.update(overrides)
        return AgentConfig.model_validate(data)

    def launch(instruction, config, request):
        instruction = instruction.strip()
        if not instruction:
            raise HTTPException(422, "Instruction cannot be empty")
        for aspect in ("planner", "grounding"):
            try:
                resolved = backend(config, aspect)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            target = urlsplit(resolved["base_url"])
            own_port = request.url.port or (
                443 if request.url.scheme == "https" else 80
            )
            if (
                target.hostname
                in ("localhost", "127.0.0.1", "::1", request.url.hostname)
                and (target.port or (443 if target.scheme == "https" else 80))
                == own_port
            ):
                raise HTTPException(
                    422, "A desktop backend cannot point to this UI server"
                )
        return tasks.launch(instruction, config)

    @app.post("/api/tasks", dependencies=protected)
    def create_task(body: TaskRequest, request: Request):
        try:
            return launch(body.instruction, task_config(body.config), request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/tasks", dependencies=protected)
    def list_tasks():
        return {"tasks": tasks.list()}

    @app.get("/api/tasks/{task_id}", dependencies=protected)
    def get_task(task_id: str, summary: bool = False):
        return tasks.public(task_id, detail="summary" if summary else "full")

    @app.delete("/api/tasks/{task_id}", dependencies=protected)
    def delete_task(task_id: str):
        tasks.delete(task_id)
        return {"id": task_id, "deleted": True}

    @app.get("/api/tasks/{task_id}/screenshot", dependencies=protected)
    def screenshot(task_id: str):
        image = tasks.screenshot(task_id)
        if image is None:
            raise HTTPException(404, "No screenshot yet")
        return Response(
            image, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @app.post("/api/tasks/{task_id}/stop", dependencies=protected)
    def stop_task(task_id: str):
        return tasks.stop(task_id)

    @app.post("/api/tasks/{task_id}/approve", dependencies=protected)
    def approve_task(task_id: str, body: ApprovalRequest):
        return tasks.approve(task_id, body.step, body.approval_token, body.approved)

    @app.get("/v1/models", dependencies=protected)
    def models():
        config = store.snapshot()
        ids = list(dict.fromkeys([config.model or "planner", "agent-s3"]))
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "agent-s3",
                }
                for name in ids
            ],
        }

    def instruction_for(body):
        for message in reversed(body.messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        part["text"]
                        for part in content
                        if isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                    )
                if isinstance(content, str) and 0 < len(content.strip()) <= 8192:
                    return content.strip()
        raise HTTPException(422, "Provide a bounded user text instruction")

    async def agent_response(body, request):
        config = store.snapshot()
        if config.approval == "manual":
            raise HTTPException(
                409,
                "Manual tasks require /api/tasks so a task/step approval token is available",
            )
        if body.model not in ("", "agent-s3", "agent_s3", "agent-s"):
            config.model = body.model
        if body.max_steps is not None:
            config.max_steps = body.max_steps
        task = await asyncio.to_thread(launch, instruction_for(body), config, request)
        task_id = task["id"]
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        def chunk(content, final=False):
            return {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "agent-s3",
                "choices": [
                    {
                        "index": 0,
                        "delta": {} if final else {"content": content},
                        "finish_reason": "stop" if final else None,
                    }
                ],
            }

        async def stream():
            version = -1
            seen_steps = {}
            completed = False
            try:
                yield "data: " + json.dumps(chunk(f"Task {task_id} started\n")) + "\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    state = tasks.public(task_id)
                    if state["version"] != version:
                        version = state["version"]
                        for step in state["steps"]:
                            key = (step["n"], step["status"])
                            if key not in seen_steps:
                                seen_steps[key] = True
                                yield "data: " + json.dumps(
                                    chunk(
                                        f"Step {step['n']} [{step['status']}]: {step['exec_code']}\n"
                                    )
                                ) + "\n\n"
                    if state["status"] not in ACTIVE_STATES:
                        yield "data: " + json.dumps(
                            chunk(
                                f"Task finished: {state['done_reason']}. {state['error'] or ''}\n"
                            )
                        ) + "\n\n"
                        yield "data: " + json.dumps(chunk("", final=True)) + "\n\n"
                        yield "data: [DONE]\n\n"
                        completed = True
                        break
                    await asyncio.sleep(0.1)
            finally:
                if not completed:
                    tasks.stop(task_id)

        if body.stream:
            return StreamingResponse(
                stream(),
                media_type="text/event-stream",
                headers={"X-Agent-S-Task-ID": task_id, "Cache-Control": "no-store"},
            )
        try:
            while True:
                if await request.is_disconnected():
                    tasks.stop(task_id)
                    raise HTTPException(499, "Client disconnected; task stopping")
                state = tasks.public(task_id)
                if state["status"] not in ACTIVE_STATES:
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            tasks.stop(task_id)
            raise
        summary = f"Task {task_id} finished ({state['done_reason']}) after {len(state['steps'])} steps."
        if state["error"]:
            summary += " Error: " + state["error"]
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": "agent-s3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": summary},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
            headers={"X-Agent-S-Task-ID": task_id},
        )

    @app.post("/v1/chat/completions", dependencies=protected)
    async def chat(body: ChatRequest, request: Request):
        if body.agent or body.model in ("agent-s3", "agent_s3", "agent-s"):
            return await agent_response(body, request)
        config = store.snapshot()
        try:
            resolved = backend(config)
            payload = body.model_dump(exclude={"agent", "max_steps"}, exclude_none=True)
            payload["model"] = payload["model"] or resolved["model"]
            url, headers = request_target(
                resolved, "chat/completions", payload["model"]
            )
            target = urlsplit(url)
            own_port = request.url.port or (
                443 if request.url.scheme == "https" else 80
            )
            if (
                target.hostname
                in ("localhost", "127.0.0.1", "::1", request.url.hostname)
                and (target.port or (443 if target.scheme == "https" else 80))
                == own_port
            ):
                raise ValueError("Planner proxy cannot point back to this UI server")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            upstream = await app.state.client.send(
                app.state.client.build_request(
                    "POST",
                    url,
                    headers=headers,
                    json=payload,
                    timeout=config.inference_timeout,
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Backend connection failed: {exc}") from exc
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key in ("retry-after", "x-request-id")
        }
        if body.stream and upstream.is_success:

            async def forward():
                try:
                    iterator = upstream.aiter_bytes().__aiter__()
                    deadline = time.monotonic() + config.task_timeout
                    while True:
                        try:
                            data = await asyncio.wait_for(
                                iterator.__anext__(),
                                timeout=max(0.01, deadline - time.monotonic()),
                            )
                        except StopAsyncIteration:
                            break
                        yield data
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                forward(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-store", **response_headers},
            )
        try:
            content = await asyncio.wait_for(
                read_limited(upstream, 8 * 1024 * 1024), timeout=config.task_timeout
            )
            return Response(
                content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json"),
                headers=response_headers,
            )
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
            raise HTTPException(502, f"Backend response failed: {exc}") from exc
        finally:
            await upstream.aclose()

    return app


async def read_limited(response, limit):
    chunks, length = [], 0
    async for chunk in response.aiter_bytes():
        length += len(chunk)
        if length > limit:
            raise ValueError("Backend response exceeded its size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    app = create_app(allowed_hosts=["localhost", "127.0.0.1", "[::1]", args.host])
    print(
        f"Agent S UI: http://{args.host}:{args.port}; API token stored beside the UI config"
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
