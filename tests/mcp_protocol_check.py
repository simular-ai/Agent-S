"""Exercise every MCP tool over real stdio with a synthetic HTTP service."""

import asyncio
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def check():
    token = "synthetic-mcp-token"
    config = {"model_url": "http://stub/v1", "ground_url": "http://stub/v1"}
    task = {
        "id": "stub",
        "status": "done",
        "current_step": 1,
        "steps": [{"n": 1, "status": "finished", "exec_code": "DONE"}],
        "logs": [],
        "overlay": "off",
    }
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, "JPEG")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def authorized(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(401)
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path.endswith("/screenshot"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(image.getvalue())
            elif self.path == "/api/status":
                self.respond({"active_tasks": 0, "platform": "windows"})
            elif self.path == "/api/config":
                self.respond(config)
            else:
                self.respond(task)

        def do_POST(self):
            if not self.authorized():
                return
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            )
            if self.path == "/api/config":
                config.update(body)
                self.respond(config)
            elif self.path == "/api/models/list":
                self.respond({"ok": True, "models": ["stub"], "raw": [{"id": "stub"}]})
            elif self.path == "/api/connection/test":
                self.respond({"ok": True, "aspect": body["aspect"]})
            elif self.path == "/api/tasks":
                assert body["config"]["approval"] == "dry_run"
                self.respond({"id": "stub"})
            elif self.path.endswith("/approve"):
                assert body["step"] == 1 and body["approval_token"] == "a" * 32
                self.respond({"approved": body["approved"]})
            else:
                self.respond(task)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "gui_agents.s3.mcp_server"],
            cwd=str(Path(__file__).resolve().parents[1]),
            env={
                "AGENT_S_UI_URL": f"http://127.0.0.1:{server.server_port}",
                "AGENT_S_UI_AUTOSTART": "0",
                "AGENT_S_UI_TOKEN": token,
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert len(tools.tools) == 12
                inputs = {
                    "server_status": {},
                    "get_config": {},
                    "update_config": {"config": {"overlay": False}},
                    "list_models": {"base_url": "http://stub/v1"},
                    "test_connection": {},
                    "launch_task": {"instruction": "synthetic", "approval": "dry_run"},
                    "task_status": {"task_id": "stub"},
                    "task_steps": {"task_id": "stub"},
                    "wait_task": {"task_id": "stub", "timeout_seconds": 1},
                    "task_screenshot": {"task_id": "stub"},
                    "approve_step": {
                        "task_id": "stub",
                        "step": 1,
                        "approval_token": "a" * 32,
                        "approved": False,
                    },
                    "stop_task": {"task_id": "stub"},
                }
                for name, arguments in inputs.items():
                    result = await session.call_tool(name, arguments)
                    assert not result.is_error, (name, result)
                    if name == "task_screenshot":
                        assert (
                            result.content[0].type == "image"
                            and result.content[0].mime_type == "image/jpeg"
                        )
                invalid = await session.call_tool(
                    "test_connection", {"aspect": "invalid"}
                )
                assert invalid.is_error and "aspect" in str(invalid.content)
        print(
            "MCP stdio: 12 authenticated tools and image content passed; useful error preserved."
        )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    asyncio.run(check())
