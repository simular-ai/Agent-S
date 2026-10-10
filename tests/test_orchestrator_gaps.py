# tests/test_orchestrator_gaps.py
"""Agent-S3 — testes de gap da auditoria 2026-08-08 (Camada 2.4).

Quatro lacunas identificadas (evidência em gui_agents/s3/orchestrator.py,
taskstore.py, panel/server.py):

  A. discovery acha 0 tracks → orquestrador procedia com tracks=[] e chamava
     PROBE sobre vazio (nada a quantizar). Desejado: abortar com reason clara,
     sem PROBE. (bug — RED até C3.2)
  B. signature bpm=None + 0 tracks colide entre projetos
     ("None:1/16:0:[]") → replay cross-project. Desejado: signature inclui
     identificador de projeto. (bug — RED até C3.2)
  C. API key vazia → Anthropic client constrói OK, erro só explode no 1º
     messages.create dentro do ReAct (sync em async, sem try/except) →
     opaque, desperdiça discovery+PROBE. Desejado: validar cedo, falhar
     gracioso, 0 MCP calls. (bug — RED até C3.2)
  D. escalate path (agent_s_operate quando ciclos esgotam) — cobertura de
     comportamento atual (não é bug). GREEN.
"""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gui_agents.s3.orchestrator import Orchestrator
from gui_agents.s3.mcp_client import MCPToolError
from gui_agents.s3 import taskstore


class TestDiscoveryZeroTracks(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "0"

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem

    async def test_A_discovery_zero_tracks_aborts_without_probe(self):
        """Discovery acha 0 tracks → aborta com reason clara, NÃO chama
        verify_quality (PROBE sobre vazio não tem sentido)."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[])
        probe_called = {"v": False}
        discover_called = {"v": False}

        async def call(name, args):
            if name == "list_project_tracks":
                discover_called["v"] = True
                return {"ok": True, "tracks": [], "cpr": "/a/p.cpr",
                        "audioDir": "/a/Audio",
                        "diagnostics": {"matched": 0, "wavCount": 0, "filter": "bateria"}}
            if name == "verify_quality":
                probe_called["v"] = True
                return {"perfect": True, "issues": [], "summary": "ok"}
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None)
        res = await orch.run("bateria", {"tracks": [], "filter": "bateria", "bpm": 120})
        self.assertTrue(discover_called["v"], "list_project_tracks deve ser chamado")
        self.assertFalse(res["perfect"])
        self.assertFalse(probe_called["v"],
                         "verify_quality NÃO deve rodar com 0 tracks descobertas")
        # reason clara no relatório de saída
        self.assertEqual(res["metrics"]["after"].get("error"), "no_tracks_found")


class TestEmptyApiKey(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "0"
        # garante que nenhuma env key mascare o teste
        self._prev_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem
        if self._prev_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key

    async def test_C_empty_api_key_fails_fast_without_mcp_calls(self):
        """key vazia → run falha gracioso ANTES de qualquer MCP call (discovery,
        PROBE, replay). Sem crash, reason clara."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[])
        any_call = {"v": False}

        async def call(name, args):
            any_call["v"] = True
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="", peer_lock=None)
        res = await orch.run("bateria",
                             {"tracks": [{"name": "k", "file": "/a.wav"}],
                              "bpm": 120, "grid": "1/16"})
        self.assertFalse(res["perfect"])
        self.assertFalse(any_call["v"],
                         "nenhum MCP call deve acontecer com API key vazia")
        self.assertEqual(res["metrics"]["after"].get("error"),
                         "missing_anthropic_api_key")


class TestEscalatePath(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "0"

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem

    async def test_D_escalate_calls_agent_s_operate_when_cycles_exhausted(self):
        """Ciclos esgotam sem perfect → escale para agent_s_operate (fallback
        GUI). Comportamento atual (não é bug) — cobertura de regressão."""
        seq = [{"perfect": False, "issues": ["x"], "sections": {}, "summary": ""}]
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[
            {"name": "verify_quality", "description": "dq",
             "input_schema": {"type": "object"}},
            {"name": "smart_quantize", "description": "q",
             "input_schema": {"type": "object"}},
        ])
        escalated = {"v": False}
        vcount = {"i": 0}

        async def call(name, args):
            if name == "verify_quality":
                i = min(vcount["i"], len(seq) - 1)
                vcount["i"] += 1
                return seq[i]
            if name == "agent_s_operate":
                escalated["v"] = True
                return {"ok": True}
            return {"ok": True}
        mcp.call = call

        def messages_create(**kw):
            class B:
                def __init__(self):
                    self.type = "tool_use"; self.name = "smart_quantize"
                    self.input = {}; self.id = "x"
            return MagicMock(content=[B()], stop_reason="tool_use")
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None,
                            max_cycles=1, max_api_calls=2)
        with patch.object(orch._client.messages, "create",
                          side_effect=messages_create):
            res = await orch.run("bateria",
                                 {"tracks": [{"name": "k", "file": "/a.wav"}],
                                  "bpm": 120, "grid": "1/16"})
        self.assertFalse(res["perfect"])
        self.assertTrue(escalated["v"],
                        "agent_s_operate deve ser chamado quando os ciclos esgotam")


class TestSignatureProjectDisambiguator(unittest.TestCase):
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "1"

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem

    def test_B_signature_includes_project_to_avoid_cross_project_replay(self):
        """bpm=None + 0 tracks + projetos diferentes → signatures diferentes.
        Antes colidia em "None:1/16:0:[]" → replay cross-project."""
        a = taskstore.signature(None, "1/16", [], project="/projA/drums.cpr")
        b = taskstore.signature(None, "1/16", [], project="/projB/drums.cpr")
        self.assertNotEqual(a, b)

    def test_B_signature_same_project_same_inputs_still_stable(self):
        """Mesmo projeto + mesmas inputs → mesma signature (replay válido)."""
        s1 = taskstore.signature(120, "1/16",
                                 [{"name": "KICK"}, {"name": "SNARE"}],
                                 project="/projA/drums.cpr")
        s2 = taskstore.signature(120, "1/16",
                                 [{"name": "SNARE"}, {"name": "KICK"}],
                                 project="/projA/drums.cpr")
        self.assertEqual(s1, s2)

    def test_B_signature_legacy_without_project_still_string(self):
        """Callers legados (sem project) continuam funcionando — signature é
        str, não quebra."""
        s = taskstore.signature(None, "1/16", [])
        self.assertIsInstance(s, str)


if __name__ == "__main__":
    unittest.main()