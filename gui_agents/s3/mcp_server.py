"""Agent S3 as an MCP server (stdio).

Lets a coding agent (Claude Code, opencode, ...) launch and supervise
desktop tasks through this app::

    agent_s_mcp
    # or: python -m gui_agents.s3.mcp_server

Claude Code example (``.mcp.json``)::

    {"mcpServers": {"agent-s3": {"command": "agent_s_mcp"}}}

The MCP tools are thin wrappers over the UI server's HTTP API
(``agent_s_ui``, default http://127.0.0.1:8000). If the UI server is not
running, it is spawned automatically (override with AGENT_S_UI_URL, or set
AGENT_S_UI_AUTOSTART=0 to disable).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from gui_agents.s3.core.openai_compatible import is_local_url
from gui_agents.s3.ui_config import server_token, state_directory
from gui_agents.s3.task_manager import ACTIVE_STATES

mcp = MCPServer("agent-s3")

UI_URL = os.getenv("AGENT_S_UI_URL", "http://127.0.0.1:8000").rstrip("/")
AUTOSTART = os.getenv("AGENT_S_UI_AUTOSTART", "1") == "1"
_startup_lock = threading.Lock()


def _headers():
    if not is_local_url(UI_URL) and not os.getenv("AGENT_S_UI_TOKEN"):
        raise ToolError("Remote UI URLs require an explicit AGENT_S_UI_TOKEN")
    token = server_token()
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _http(
    method: str, path: str, body: Optional[Dict] = None, timeout: int = 30
) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        UI_URL + path,
        data=data,
        headers=_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode()) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:500]
        raise ToolError(f"UI server {exc.code}: {detail}") from exc


def _screenshot_bytes(path: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(UI_URL + path, headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise ToolError(f"UI server {exc.code}: {exc.read().decode()[:300]}") from exc


def ensure_server(wait_seconds: int = 30) -> Dict[str, Any]:
    """Serialize local startup; authentication errors never spawn another server."""
    with _startup_lock:
        return _ensure_server(wait_seconds)


def _ensure_server(wait_seconds):
    try:
        status = _http("GET", "/api/status", timeout=3)
        if not isinstance(status, dict) or "active_tasks" not in status:
            raise ToolError("The configured URL is not an Agent S UI backend")
        return status
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        pass
    if not AUTOSTART:
        raise ToolError(
            f"UI server not reachable at {UI_URL}. Start it with `agent_s_ui` "
            "(or set AGENT_S_UI_URL / AGENT_S_UI_AUTOSTART=1)."
        )
    parts = urlparse(UI_URL)
    if (
        not is_local_url(UI_URL)
        or parts.scheme != "http"
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
        or parts.username
    ):
        raise ToolError(
            "Autostart supports only a local HTTP URL without a path; start remote/HTTPS backends yourself"
        )
    host, port = parts.hostname or "127.0.0.1", parts.port or 8000
    directory = state_directory()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "ui-startup.log"
    with log_path.open("ab") as log:
        if os.name != "nt":
            os.chmod(log_path, 0o600)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gui_agents.s3.ui_server",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
        )
    deadline = time.monotonic() + wait_seconds
    last_err = "unknown"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ToolError(
                f"UI server exited (code {proc.returncode}); diagnostics: {log_path}"
            )
        try:
            return _http("GET", "/api/status", timeout=3)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = str(exc)
            time.sleep(1.0)
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    raise ToolError(f"UI startup timed out: {last_err}; diagnostics: {log_path}")


@mcp.tool()
def server_status() -> Dict[str, Any]:
    """Check the UI backend status (platform, screen size, active tasks)."""
    ensure_server()
    return _http("GET", "/api/status")


@mcp.tool()
def get_config() -> Dict[str, Any]:
    """Get the current planner/grounding/agent configuration."""
    ensure_server()
    return _http("GET", "/api/config")


@mcp.tool()
def update_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration (provider, model, model_url, ground_*, max_steps,
    max_trajectory_length, enable_reflection, approval, overlay, ...)."""
    ensure_server()
    return _http("POST", "/api/config", config)


@mcp.tool()
def list_models(base_url: str, api_key: str = "") -> List[str]:
    """List model ids on an OpenAI-compatible server (e.g. LM Studio)."""
    ensure_server()
    result = _http(
        "POST", "/api/models/list", {"base_url": base_url, "api_key": api_key}
    )
    return [
        m.get("id", m) if isinstance(m, dict) else m
        for m in result.get("raw", result.get("models", []))
    ]


@mcp.tool()
def test_connection(
    aspect: str = "planner", base_url: str = "", api_key: str = ""
) -> Dict[str, Any]:
    """Test planner or grounding endpoint. aspect is 'planner'/'grounding';
    empty base_url tests the currently configured endpoint."""
    ensure_server()
    if aspect not in ("planner", "grounding"):
        raise ToolError("aspect must be planner or grounding")
    if not base_url:
        # The HTTP backend resolves saved credentials; secrets are never returned.
        return _http(
            "POST", "/api/connection/test", {"aspect": aspect, "api_key": api_key}
        )
    return _http(
        "POST",
        "/api/connection/test",
        {"base_url": base_url, "api_key": api_key, "aspect": aspect},
    )


@mcp.tool()
def launch_task(
    instruction: str, max_steps: int = 0, approval: str = ""
) -> Dict[str, Any]:
    """Launch Agent S3 on this machine to perform a desktop task.
    Returns {"id": task_id}. Only one task runs at a time.
    Poll with task_status / wait_task. Set approval to
    "manual"/"dry_run" to override the configured mode."""
    ensure_server()
    body: Dict[str, Any] = {"instruction": instruction, "config": {}}
    if max_steps:
        body["config"]["max_steps"] = max_steps
    if approval:
        body["config"]["approval"] = approval
    return _http("POST", "/api/tasks", body, timeout=60)


def _trim(task: Dict[str, Any]) -> Dict[str, Any]:
    steps = task.get("steps", [])
    last = steps[-1] if steps else {}
    return {
        "id": task["id"],
        "status": task["status"],
        "current_step": task.get("current_step", 0),
        "max_steps": task.get("max_steps"),
        "done_reason": task.get("done_reason"),
        "error": (task.get("error") or "")[:500],
        "awaiting_approval": task.get("awaiting_approval", False),
        "pending_approval": task.get("pending_approval"),
        "overlay": task.get("overlay", "off"),
        "last_step": (
            {
                "n": last.get("n"),
                "status": last.get("status"),
                "plan": (last.get("plan") or "")[:500],
                "exec_code": (last.get("exec_code") or "")[:300],
                "error": (last.get("error") or "")[:300],
            }
            if last
            else None
        ),
        "recent_logs": task.get("logs", [])[-10:],
    }


@mcp.tool()
def task_status(task_id: str) -> Dict[str, Any]:
    """Get trimmed task state (status, current step, last plan, recent logs)."""
    ensure_server()
    return _trim(_http("GET", f"/api/tasks/{task_id}?summary=true"))


@mcp.tool()
def task_steps(task_id: str) -> List[Dict[str, Any]]:
    """Get full per-step detail (plan, executed code, reflection, errors)."""
    ensure_server()
    return _http("GET", f"/api/tasks/{task_id}").get("steps", [])


@mcp.tool()
def wait_task(
    task_id: str, timeout_seconds: int = 600, poll_seconds: int = 5
) -> Dict[str, Any]:
    """Block until the task reaches done/stopped/error (or timeout)."""
    ensure_server()
    if not 0 <= timeout_seconds <= 3600 or not 1 <= poll_seconds <= 60:
        raise ToolError("timeout_seconds must be 0..3600 and poll_seconds 1..60")
    deadline = time.monotonic() + timeout_seconds
    last = _trim(_http("GET", f"/api/tasks/{task_id}?summary=true"))
    while last["status"] in ACTIVE_STATES and time.monotonic() < deadline:
        time.sleep(poll_seconds)
        last = _trim(_http("GET", f"/api/tasks/{task_id}?summary=true"))
    if last["status"] in ACTIVE_STATES:
        last["note"] = f"still running after {timeout_seconds}s; call wait_task again"
    return last


@mcp.tool()
def task_screenshot(task_id: str) -> Image:
    """Get the task's latest desktop screenshot as an image."""
    ensure_server()
    return Image(
        data=_screenshot_bytes(f"/api/tasks/{task_id}/screenshot"), format="jpeg"
    )


@mcp.tool()
def approve_step(
    task_id: str, step: int, approval_token: str, approved: bool = True
) -> Dict[str, Any]:
    """Approve (or reject) the step awaiting approval in manual mode."""
    ensure_server()
    return _http(
        "POST",
        f"/api/tasks/{task_id}/approve",
        {"approved": approved, "step": step, "approval_token": approval_token},
    )


@mcp.tool()
def stop_task(task_id: str) -> Dict[str, Any]:
    """Stop a running task."""
    ensure_server()
    return _http("POST", f"/api/tasks/{task_id}/stop")


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
