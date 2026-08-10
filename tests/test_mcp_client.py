# tests/test_mcp_client.py
import asyncio, json, os, sys, tempfile, unittest
from unittest.mock import MagicMock, AsyncMock, patch
from gui_agents.s3.mcp_client import MCPClient, MCPToolError

class _EmptyStream:
    """Stream vazio para stderr — readline sempre retorna b'' (encerra _drain_stderr)."""
    async def readline(self):
        await asyncio.sleep(0)
        return b""

class FakeProc:
    """Subprocess fake: lê commands de uma fila e responde."""
    def __init__(self):
        self._responses = []
        self.sent = []
        self.terminated = False
        # A impl chama proc.stdout.readline / proc.stdin.write / proc.stderr.readline
        # e proc.returncode / proc.terminate / proc.wait. Exponha si mesmo como
        # stdin/stdout; stderr é um stream vazio separado para _drain_stderr sair
        # imediatamente sem competir pela fila de respostas.
        self.stdin = self
        self.stdout = self
        self.stderr = _EmptyStream()
        self.returncode = None
    def push(self, obj): self._responses.append((json.dumps(obj) + "\n").encode("utf-8"))
    async def readline(self):
        await asyncio.sleep(0)
        return self._responses.pop(0) if self._responses else b""
    def write(self, s): self.sent.append(s)
    def terminate(self): self.terminated = True; self.returncode = 0
    async def wait(self): return self.returncode

class TestMCPClient(unittest.IsolatedAsyncioTestCase):
    async def test_start_lists_tools(self):
        proc = FakeProc()
        proc.push({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{"listChanged":False}},"serverInfo":{"name":"cubeflow","version":"1"}}})
        proc.push({"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"verify_quality","description":"dq","inputSchema":{"type":"object","properties":{"tracks":{"type":"array"}}}}]}})
        client = MCPClient(["node","fake"], env={}, logger=MagicMock())
        with patch("gui_agents.s3.mcp_client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            await client.start()
        names = [t["name"] for t in client._tools]
        self.assertIn("verify_quality", names)
        # schema MCP inputSchema vira input_schema Anthropic
        vq = [t for t in client._tools if t["name"]=="verify_quality"][0]
        self.assertEqual(vq["input_schema"]["properties"]["tracks"]["type"], "array")

    async def test_call_returns_parsed_content(self):
        proc = FakeProc()
        proc.push({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"cubeflow","version":"1"}}})
        proc.push({"jsonrpc":"2.0","id":2,"result":{"tools":[]}})
        proc.push({"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"perfect\":true}"}],"isError":False}})
        client = MCPClient(["node","fake"], env={}, logger=MagicMock())
        with patch("gui_agents.s3.mcp_client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            await client.start()
            res = await client.call("verify_quality", {"tracks":[]})
        self.assertEqual(res, {"perfect": True})

    async def test_call_raises_on_isError(self):
        proc = FakeProc()
        proc.push({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"cubeflow","version":"1"}}})
        proc.push({"jsonrpc":"2.0","id":2,"result":{"tools":[]}})
        proc.push({"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\"error\":\"bloqueado\"}"}],"isError":True}})
        client = MCPClient(["node","fake"], env={}, logger=MagicMock())
        with patch("gui_agents.s3.mcp_client.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
            await client.start()
            with self.assertRaises(MCPToolError):
                await client.call("quantize", {"action":"quantize_all_tracks"})

if __name__ == "__main__":
    unittest.main()