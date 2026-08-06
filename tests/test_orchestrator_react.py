# tests/test_orchestrator_react.py
import asyncio, unittest
from unittest.mock import AsyncMock, MagicMock, patch
from gui_agents.s3.orchestrator import Orchestrator
from gui_agents.s3.mcp_client import MCPToolError

def make_mcp(perfect_sequence):
    """MCP mock que retorna verify_quality com sequência de perfects/issues."""
    calls = {"verify": 0, "tool": 0}
    seq = list(perfect_sequence)
    mcp = MagicMock()
    mcp.tools = AsyncMock(return_value=[
        {"name":"verify_quality","description":"dq","input_schema":{"type":"object","properties":{"tracks":{"type":"array"}}}},
        {"name":"smart_quantize","description":"q","input_schema":{"type":"object","properties":{}}},
        {"name":"correct_off_grid","description":"c","input_schema":{"type":"object","properties":{}}},
    ])
    async def call(name, args):
        if name == "verify_quality":
            i = min(calls["verify"], len(seq)-1)
            r = seq[i]; calls["verify"] += 1
            return r
        calls["tool"] += 1
        return {"ok": True}
    mcp.call = call
    return mcp, calls

class TestOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def test_stops_immediately_if_already_perfect(self):
        mcp, calls = make_mcp([{"perfect":True,"issues":[],"sections":{},"summary":"ok"}])
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None)
        res = await orch.run("deixar bateria perfeita", {"tracks":[{"name":"kick","file":"/a.wav"}],"bpm":120,"grid":"1/16"})
        self.assertTrue(res["perfect"])
        self.assertEqual(calls["verify"], 1)

    async def test_retries_until_perfect(self):
        mcp, calls = make_mcp([
            {"perfect":False,"issues":["kick off-grid"],"sections":{},"summary":"ruim"},
            {"perfect":False,"issues":["kick off-grid"],"sections":{},"summary":"ruim"},
            {"perfect":True,"issues":[],"sections":{},"summary":"ok"},
        ])
        # Anthropic mock: cada turno devolve 1 tool_use e depois end_turn (1 tool/ciclo).
        # AJUSTE: o mock original devolvia stop_reason="tool_use" sempre, fazendo o
        # react-loop rodar 8 iterações por ciclo e consumir entradas de verify_quality
        # como tool, o que colapsava cycles para 1. O mock agora alterna tool_use ->
        # end_turn para refletir o comportamento real do LLM (1 tool, depois fim do
        # turno). As asserções (perfect + cycles==2) são preservadas.
        tool_idx = {"i":0}
        tools_seq = ["smart_quantize","correct_off_grid","correct_off_grid"]
        def messages_create(**kw):
            class Block:
                def __init__(self, name, inp): self.type="tool_use"; self.name=name; self.input=inp; self.id=f"t{tool_idx['i']}"
            i = tool_idx["i"]; tool_idx["i"] += 1
            if i % 2 == 0:
                tool_name = tools_seq[i // 2] if (i // 2) < len(tools_seq) else "verify_quality"
                return MagicMock(content=[Block(tool_name, {})], stop_reason="tool_use")
            return MagicMock(content=[], stop_reason="end_turn")
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None, max_cycles=3)
        with patch.object(orch._client.messages, "create", side_effect=messages_create):
            res = await orch.run("bateria", {"tracks":[{"name":"k","file":"/a.wav"}],"bpm":120,"grid":"1/16"})
        self.assertTrue(res["perfect"])
        self.assertEqual(res["cycles"], 2)  # 2 ciclos de refaz até perfect

    async def test_budget_cap_stops_loop(self):
        mcp, calls = make_mcp([{"perfect":False,"issues":["x"],"sections":{},"summary":""}])
        def messages_create(**kw):
            class B:
                def __init__(self): self.type="tool_use"; self.name="smart_quantize"; self.input={}; self.id="x"
            return MagicMock(content=[B()], stop_reason="tool_use")
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None, max_cycles=1, max_api_calls=2)
        with patch.object(orch._client.messages, "create", side_effect=messages_create):
            res = await orch.run("bateria", {"tracks":[{"name":"k","file":"/a.wav"}],"bpm":120,"grid":"1/16"})
        self.assertFalse(res["perfect"])
        self.assertLessEqual(res["api_calls"], 2)

    async def test_blocked_tool_emits_error_after_retries(self):
        """Peer-lock bloqueado 3x → tool_result de erro, mcp.call não chamado, run completa."""
        mcp, calls = make_mcp([{"perfect":False,"issues":["x"],"sections":{},"summary":""}])
        called_names = []
        orig_call = mcp.call
        async def tracking_call(name, args):
            called_names.append(name)
            return await orig_call(name, args)
        mcp.call = tracking_call
        def messages_create(**kw):
            class B:
                def __init__(self): self.type="tool_use"; self.name="smart_quantize"; self.input={}; self.id="x"
            return MagicMock(content=[B()], stop_reason="tool_use")
        peer_lock = MagicMock()
        peer_lock.is_held_by_other = MagicMock(return_value=True)  # sempre ocupado
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=peer_lock,
                            peer_name="agent_s", max_cycles=1, max_api_calls=2)
        with patch.object(orch._client.messages, "create", side_effect=messages_create), \
             patch("gui_agents.s3.orchestrator.asyncio.sleep", new=AsyncMock(return_value=None)):
            res = await orch.run("bateria", {"tracks":[{"name":"k","file":"/a.wav"}],"bpm":120,"grid":"1/16"})
        # tool destrutiva bloqueada nunca foi chamada
        self.assertNotIn("smart_quantize", called_names)
        # run completa (não perfect)
        self.assertFalse(res["perfect"])
        # peer-lock foi re-checado pelo menos 3x (1 checagem inicial + 3 dentro do loop)
        self.assertGreaterEqual(peer_lock.is_held_by_other.call_count, 4)
        # métrica de blocked registrada
        self.assertGreaterEqual(res["metrics"]["blocked"], 1)

if __name__ == "__main__":
    unittest.main()