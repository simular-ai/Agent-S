#!/usr/bin/env python3
"""Local web panel for Agent-S3 — start/stop a task, stream its log live.

FastAPI + uvicorn, already present in the venv (no new dependency added).
Runs entirely on localhost; not exposed to the network.
"""
import asyncio
import json
import os
import re
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from dotenv import dotenv_values

from gui_agents.s3.mcp_client import MCPClient, MCPToolError
from gui_agents.s3.orchestrator import Orchestrator
from gui_agents.s3.coordination.peer_lock import PeerLock

AGENT_S_DIR = Path(__file__).resolve().parent.parent
VENV_AGENT_S = AGENT_S_DIR / "venv" / "bin" / "agent_s"
ENV_FILE = AGENT_S_DIR / ".env"
MEMORY_FILE = AGENT_S_DIR / "memory" / "darwin" / "narrative_memory.json"

CUBEFLOW_DIR = Path.home() / "Documents" / "Projetos 2026" / "CubeFlow"
BRIDGE_CMD = ["node", str(CUBEFLOW_DIR / "src" / "mcp" / "cubeflow-full-mcp.js")]
KEYWORDS = re.compile(r"\b(bateria|drum|quantize|hitpoint)\b", re.IGNORECASE)


def _is_cubeflow_task(task: str) -> bool:
    return bool(KEYWORDS.search(task or ""))

FIXED_ARGS = [
    "--provider", "anthropic",
    "--model", "claude-sonnet-5",
    "--ground_provider", "anthropic",
    "--ground_model", "claude-sonnet-5",
    "--ground_url", "unused",
    "--grounding_width", "1728",
    "--grounding_height", "1117",
]

app = FastAPI()

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_log_lines: list[str] = []
_status = "idle"  # idle | running | completed | error


def _reset_for_new_run():
    global _log_lines, _status
    _log_lines = []
    _status = "running"


def _reader_thread(proc: subprocess.Popen):
    global _status
    for raw_line in iter(proc.stdout.readline, ""):
        if not raw_line:
            break
        with _lock:
            _log_lines.append(raw_line.rstrip("\n"))
    proc.stdout.close()
    returncode = proc.wait()
    with _lock:
        if _status == "running":  # not manually stopped in the meantime
            _status = "completed" if returncode == 0 else "error"
        _log_lines.append(f"[panel] process exited with code {returncode}")


@app.post("/api/start")
async def start(payload: dict):
    global _proc
    task = (payload.get("task") or "").strip()
    if not task:
        return JSONResponse({"ok": False, "error": "tarefa vazia"}, status_code=400)

    with _lock:
        if _status == "running":
            return JSONResponse({"ok": False, "error": "já tem uma tarefa rodando"}, status_code=409)
        _reset_for_new_run()

    env = os.environ.copy()
    if ENV_FILE.exists():
        env.update({k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None})
    # Without this, Python block-buffers stdout when it's a pipe (not a TTY),
    # so the log would arrive in delayed chunks instead of live line-by-line.
    env["PYTHONUNBUFFERED"] = "1"

    if _is_cubeflow_task(task):
        threading.Thread(target=_run_orchestrator, args=(task, env), daemon=True).start()
        return {"ok": True}

    cmd = [str(VENV_AGENT_S), *FIXED_ARGS, "--task", task]
    proc = subprocess.Popen(
        cmd,
        cwd=str(AGENT_S_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with _lock:
        _proc = proc
        _log_lines.append(f"[panel] started: {' '.join(cmd)}")

    threading.Thread(target=_reader_thread, args=(proc,), daemon=True).start()
    return {"ok": True}


def _run_orchestrator(task: str, env: dict):
    global _status
    with _lock:
        _log_lines.append(f"[panel] orchestrator dispatch: {task}")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    peer_lock = None
    hb_stop = None
    try:
        bridge_env = {
            **env,
            "CUBEFLOW_ALLOW_REAL_AUTOMATION": "true",
            "CUBEFLOW_PEER": "agent_s",
            "CUBEFLOW_COORDINATION_DB": str(CUBEFLOW_DIR / "src" / "database" / "peer-coordination.db"),
        }
        client = MCPClient(BRIDGE_CMD, env=bridge_env)
        loop.run_until_complete(client.start())
        peer_lock = PeerLock(bridge_env["CUBEFLOW_COORDINATION_DB"])
        acquired = peer_lock.acquire("agent_s", task, timeout_ms=5000)
        if not acquired.get("ok"):
            with _lock:
                _log_lines.append(f"[panel] peer-lock bloqueado: {acquired.get('reason')}")
            return
        hb_stop = threading.Event()
        threading.Thread(target=_heartbeat, args=(peer_lock, hb_stop), daemon=True).start()
        orch = Orchestrator(
            mcp=client,
            anthropic_key=env.get("ANTHROPIC_API_KEY", ""),
            peer_lock=peer_lock,
            peer_name="agent_s",
        )
        # session default — bateria 120bpm 1/16; user pode estender depois
        session = {"tracks": [], "bpm": 120, "grid": "1/16", "toleranceMs": 10}

        def emit(line: str):
            with _lock:
                _log_lines.append(line)

        orch.logger.info = lambda *a, **k: emit(str(a))
        res = loop.run_until_complete(orch.run(task, session))
        with _lock:
            for ln in res["report"].splitlines():
                _log_lines.append(ln)
            _status = "completed" if res["perfect"] else "error"
    except Exception as e:
        with _lock:
            _log_lines.append(f"[panel] orchestrator error: {e}")
            _status = "error"
    finally:
        if hb_stop is not None:
            hb_stop.set()
        if peer_lock is not None:
            try:
                peer_lock.release("agent_s")
            except Exception:
                pass
        loop.close()


def _heartbeat(peer_lock, stop_event):
    import time
    while not stop_event.is_set():
        try:
            peer_lock.heartbeat("agent_s")
        except Exception:
            pass
        time.sleep(5)


@app.post("/api/stop")
async def stop():
    global _status
    with _lock:
        proc = _proc
        if proc is None or proc.poll() is not None:
            return JSONResponse({"ok": False, "error": "nada rodando"}, status_code=409)
        _status = "idle"
        _log_lines.append("[panel] stop requested by user")

    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    return {"ok": True}


def _memory_count() -> int:
    try:
        data = json.loads(MEMORY_FILE.read_text())
        return len(data)
    except Exception:
        return 0


@app.get("/api/status")
async def status():
    with _lock:
        return {"status": _status, "memory_count": _memory_count()}


@app.get("/api/stream")
async def stream():
    async def event_gen():
        last_idx = 0
        while True:
            with _lock:
                new_lines = _log_lines[last_idx:]
                last_idx = len(_log_lines)
                current_status = _status
            for line in new_lines:
                yield f"data: {json.dumps(line)}\n\n"
            yield f"event: status\ndata: {json.dumps({'status': current_status, 'memory_count': _memory_count()})}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5151, log_level="warning")
