# tests/test_orchestrator_adversarial.py
"""Agent-S3 — testes adversariais / edge-case do orquestrador ReAct.

Expõe: API key whitespace-only (derrota o gate de key vazia do C3.2),
discovery retorna res não-dict, tracks parcialmente resolvidas (name/file
ausente filtrados), max_cycles=0 escala direto, max_api_calls=0 não entra
no ReAct, memory replay com step malformado (sem name) cai gracioso no
ReAct, atalho PROBE-perfect (0 ciclos), peer_lock bloqueia tool destrutiva.
"""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gui_agents.s3.orchestrator import Orchestrator
from gui_agents.s3.mcp_client import MCPToolError


def _mem_off():
    os.environ["AGENT_S_USE_MEMORY"] = "0"


class _MemOffMixin:
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        _mem_off()
        self._prev_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem
        if self._prev_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key


class TestWhitespaceApiKey(_MemOffMixin, unittest.IsolatedAsyncioTestCase):
    async def test_whitespace_api_key_fails_fast_without_mcp_calls(self):
        """key="  " (whitespace) é truthy → gate `if not self._api_key` deixa
        passar → orquestrador procede p/ discovery+PROBE+ReAct e falha opaco
        no 1º messages.create. C3.2 queria falha graciosa em key ruim; whitespace
        derrota o gate. Desejado: strip → key vazia → abort gracioso, 0 MCP."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[])
        any_call = {"v": False}

        async def call(name, args):
            any_call["v"] = True
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="   ", peer_lock=None)
        # Protege contra chamada de rede real enquanto o bug existir (RED):
        # se o gate falhar e chegar no ReAct, messages.create mockado quebra.
        def no_net(**kw):
            return MagicMock(content=[], stop_reason="end_turn")
        # Robusto aos dois estados: RED (client construído, patch aplicado,
        # abort não acontece → any_call True → falha) e GREEN (client None
        # pós-strip → não há o que patchear, run aborta sozinho).
        if orch._client is not None:
            with patch.object(orch._client.messages, "create", side_effect=no_net):
                res = await orch.run("bateria",
                                     {"tracks": [{"name": "k", "file": "/a.wav"}],
                                      "bpm": 120, "grid": "1/16"})
        else:
            res = await orch.run("bateria",
                                 {"tracks": [{"name": "k", "file": "/a.wav"}],
                                  "bpm": 120, "grid": "1/16"})
        self.assertFalse(res["perfect"])
        self.assertFalse(any_call["v"], "whitespace key deve abortar antes de qualquer MCP call")
        self.assertEqual(res["metrics"]["after"].get("error"), "missing_anthropic_api_key")


class TestDiscoveryMalformedRes(_MemOffMixin, unittest.IsolatedAsyncioTestCase):
    async def test_discovery_non_dict_res_treated_as_empty(self):
        """list_project_tracks retorna list/não-dict → tracks=[] → 0 achadas
        → abort no_tracks_found (não crash em res.get)."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[])

        async def call(name, args):
            if name == "list_project_tracks":
                return [["track1"], {"weird": "res"}]  # não-dict
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None)
        res = await orch.run("bateria", {"tracks": [], "bpm": 120, "filter": "bateria"})
        self.assertFalse(res["perfect"])
        self.assertEqual(res["metrics"]["after"].get("error"), "no_tracks_found")

    async def test_discovery_tracks_missing_name_or_file_filtered(self):
        """Tracks com name ou file ausente são filtradas (só válidas entram).
        Se TODAS inválidas → 0 → abort no_tracks_found (não prossegue com lixo)."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[])

        async def call(name, args):
            if name == "list_project_tracks":
                return {"tracks": [
                    {"name": "KICK", "file": ""},          # file vazio → filtra
                    {"name": "", "file": "/a.wav"},        # name vazio → filtra
                    {"file": "/b.wav"},                    # sem name → filtra
                    {"name": "SNARE"},                     # sem file → filtra
                ], "cpr": "/p.cpr"}
            if name == "verify_quality":
                return {"perfect": True, "issues": [], "summary": ""}
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None)
        res = await orch.run("bateria", {"tracks": [], "bpm": 120})
        self.assertFalse(res["perfect"])
        self.assertEqual(res["metrics"]["after"].get("error"), "no_tracks_found")


class TestMaxCyclesZero(_MemOffMixin, unittest.IsolatedAsyncioTestCase):
    async def test_max_cycles_zero_skips_react_and_escalates(self):
        """max_cycles=0 → range(1,1) vazio, sem ReAct → escala direto p/
        agent_s_operate. Comportamento de borda, não crash."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[
            {"name": "verify_quality", "description": "q",
             "input_schema": {"type": "object"}}])
        escalated = {"v": False}

        async def call(name, args):
            if name == "verify_quality":
                return {"perfect": False, "issues": ["x"], "summary": ""}
            if name == "agent_s_operate":
                escalated["v"] = True
                return {"ok": True}
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None,
                            max_cycles=0, max_api_calls=5)
        with patch.object(orch._client.messages, "create",
                          return_value=MagicMock(content=[], stop_reason="end_turn")):
            res = await orch.run("bateria",
                                 {"tracks": [{"name": "k", "file": "/a.wav"}],
                                  "bpm": 120, "grid": "1/16"})
        self.assertFalse(res["perfect"])
        self.assertTrue(escalated["v"], "max_cycles=0 escala direto sem ReAct")


class TestMaxApiCallsZero(_MemOffMixin, unittest.IsolatedAsyncioTestCase):
    async def test_max_api_calls_zero_skips_react(self):
        """max_api_calls=0 → `if self._api_calls >= 0` true no início do ciclo
        → break imediato, sem ReAct → escala."""
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[
            {"name": "verify_quality", "description": "q",
             "input_schema": {"type": "object"}}])
        escalated = {"v": False}
        api_called = {"v": False}

        async def call(name, args):
            if name == "verify_quality":
                return {"perfect": False, "issues": ["x"], "summary": ""}
            if name == "agent_s_operate":
                escalated["v"] = True
                return {"ok": True}
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None,
                            max_cycles=3, max_api_calls=0)

        def no_api(**kw):
            api_called["v"] = True
            return MagicMock(content=[], stop_reason="end_turn")
        with patch.object(orch._client.messages, "create", side_effect=no_api):
            res = await orch.run("bateria",
                                 {"tracks": [{"name": "k", "file": "/a.wav"}],
                                  "bpm": 120, "grid": "1/16"})
        self.assertFalse(api_called["v"], "max_api_calls=0 não chama Anthropic")
        self.assertTrue(escalated["v"])


class TestProbePerfectShortcut(unittest.IsolatedAsyncioTestCase):
    """PROBE já perfeito → atalho: grava sequência vazia, 0 ciclos, perfect."""
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "0"
        self._prev_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem
        if self._prev_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key

    async def test_probe_perfect_returns_immediately_zero_cycles(self):
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[])
        any_other = {"v": False}

        async def call(name, args):
            if name == "verify_quality":
                return {"perfect": True, "issues": [], "summary": "ok"}
            any_other["v"] = True
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=None)
        res = await orch.run("bateria",
                             {"tracks": [{"name": "k", "file": "/a.wav"}],
                              "bpm": 120, "grid": "1/16"})
        self.assertTrue(res["perfect"])
        self.assertEqual(res["cycles"], 0)
        self.assertFalse(any_other["v"], "PROBE perfeito não chama outras tools")


class TestPeerLockBlocksDestructive(unittest.IsolatedAsyncioTestCase):
    """peer_lock held by other → tool destrutiva bloqueada, reportada como
    is_error, e orquestrador continua (não trava)."""
    def setUp(self):
        self._prev_mem = os.environ.get("AGENT_S_USE_MEMORY")
        os.environ["AGENT_S_USE_MEMORY"] = "0"
        self._prev_key = os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self):
        if self._prev_mem is None:
            os.environ.pop("AGENT_S_USE_MEMORY", None)
        else:
            os.environ["AGENT_S_USE_MEMORY"] = self._prev_mem
        if self._prev_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._prev_key

    async def test_peer_lock_blocks_smart_quantize_then_releases(self):
        mcp = MagicMock()
        mcp.tools = AsyncMock(return_value=[
            {"name": "verify_quality", "description": "q",
             "input_schema": {"type": "object"}},
            {"name": "smart_quantize", "description": "q",
             "input_schema": {"type": "object"}},
        ])
        # is_held_by_other True só na 1ª checagem (libera rápido — 1 sleep de 2s,
        # não 3x2s) simulando outro peer soltando o lock.
        class FakeLock:
            def __init__(self):
                self.n = 0
            def is_held_by_other(self, peer):
                self.n += 1
                return self.n <= 1
        verify = {"i": 0}

        async def call(name, args):
            if name == "verify_quality":
                verify["i"] += 1
                return {"perfect": verify["i"] >= 2, "issues": [], "summary": ""}
            return {"ok": True}
        mcp.call = call
        orch = Orchestrator(mcp=mcp, anthropic_key="k", peer_lock=FakeLock(),
                            peer_name="agent_s", max_cycles=1, max_api_calls=2)

        class B:
            def __init__(self):
                self.type = "tool_use"; self.name = "smart_quantize"
                self.input = {}; self.id = "b1"

        def mk(**kw):
            return MagicMock(content=[B()], stop_reason="tool_use")
        with patch.object(orch._client.messages, "create", side_effect=mk):
            res = await orch.run("bateria",
                                 {"tracks": [{"name": "k", "file": "/a.wav"}],
                                  "bpm": 120, "grid": "1/16"})
        self.assertGreaterEqual(res["metrics"]["blocked"], 1,
                               "tool bloqueada contabilizada em metrics[blocked]")


if __name__ == "__main__":
    unittest.main()