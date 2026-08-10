"""MCP client — fala JSON-RPC over stdio com o bridge Node do CubeFlow."""
import asyncio
import json
import logging

class MCPToolError(Exception):
    def __init__(self, tool, message):
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message

class MCPClient:
    def __init__(self, bridge_cmd, env, logger=None):
        self.bridge_cmd = list(bridge_cmd)
        self.env = dict(env)
        self.logger = logger or logging.getLogger(__name__)
        self._proc = None
        self._next_id = 1
        self._pending = {}      # id -> Future
        self._tools = []        # lista em schema Anthropic
        self._reader_task = None
        self._stderr_task = None

    async def start(self):
        self._proc = await asyncio.create_subprocess_exec(
            *self.bridge_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        # initialize
        init = await self._request("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "agent_s", "version": "1.0.0"},
        })
        # notificação initialized (sem id, sem resposta)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        # tools/list
        listing = await self._request("tools/list", {})
        self._tools = [
            {"name": t["name"], "description": t.get("description", ""),
             "input_schema": t.get("inputSchema", {"type": "object", "properties": {}})}
            for t in listing.get("tools", [])
        ]

    async def tools(self):
        return list(self._tools)

    async def call(self, name, arguments):
        res = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        if res.get("isError"):
            text = res.get("content", [{}])[0].get("text", "")
            try: payload = json.loads(text)
            except Exception: payload = {"error": text}
            raise MCPToolError(name, payload.get("error", text))
        text = res.get("content", [{}])[0].get("text", "")
        try:
            return json.loads(text)
        except Exception:
            return text

    async def close(self):
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try: await asyncio.wait_for(self._proc.wait(), timeout=3)
            except asyncio.TimeoutError:
                self._proc.kill()
        for t in (self._reader_task, self._stderr_task):
            if t: t.cancel()

    def _send(self, obj):
        line = (json.dumps(obj) + "\n").encode("utf-8")
        self._proc.stdin.write(line)

    async def _request(self, method, params):
        fut = asyncio.get_running_loop().create_future()
        msg_id = self._next_id
        self._next_id += 1
        self._pending[msg_id] = fut
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
        return await asyncio.wait_for(fut, timeout=120)

    async def _read_loop(self):
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.warning("mcp: linha não-JSON: %r", line[:200])
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    continue  # notificação
                fut = self._pending.pop(msg_id, None)
                if fut and not fut.done():
                    if "error" in msg:
                        fut.set_exception(MCPToolError(msg.get("method", "?"), json.dumps(msg["error"])))
                    else:
                        fut.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(MCPToolError("bridge", "conexão encerrada"))

    async def _drain_stderr(self):
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line: break
                self.logger.info("bridge: %s", line.decode("utf-8", errors="replace").strip())
        except asyncio.CancelledError:
            pass