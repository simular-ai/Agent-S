# tests/test_field_real_mcp.py
"""Agent-S3 — TESTE REAL DE CAMPO da integração com o CubeFlow.

Spawna o bridge MCP real do CubeFlow (cubeflow-full-mcp.js) via MCPClient (o
mesmo transporte stdio JSON-RPC que o orquestrador usa) e chama tools reais
NÃO-destrutivas contra arquivos scratch. Prova o wiring de campo real:
MCPClient.start (initialize+tools/list), list_project_tracks resolvendo um
.cpr+Audio/ reais (não mockado), peer-lock, sem tocar Cubase GUI.

NÃO é E2E destrutivo: nenhum keystroke/osascript/cliclick, nenhum WAV do
Beto sobrescrito. Só tools read-only contra tmp.
"""
import asyncio
import os
import struct
import tempfile
import unittest

from gui_agents.s3.mcp_client import MCPClient

CUBEFLOW_DIR = os.path.expanduser("~/Documents/Projetos 2026/CubeFlow")
BRIDGE_CMD = ["node", os.path.join(CUBEFLOW_DIR, "src", "mcp", "cubeflow-full-mcp.js")]


def _build_fake_cpr(names):
    """.cpr FAKE válido p/ o parser real: 4 bytes UInt32BE (len+4) + name + BOM."""
    out = bytearray()
    for name in names:
        nb = name.encode("ascii")
        out += struct.pack(">I", len(nb) + 4)
        out += nb
        out += b"\x00\xef\xbb\xbf"
    return bytes(out)


def _build_scratch_project(names):
    proj_dir = tempfile.mkdtemp(prefix="cf_field_py_")
    cpr_path = os.path.join(proj_dir, "FIELD-PY.cpr")
    with open(cpr_path, "wb") as f:
        f.write(_build_fake_cpr(names))
    audio_dir = os.path.join(proj_dir, "Audio")
    os.makedirs(audio_dir, exist_ok=True)
    for name in names:
        with open(os.path.join(audio_dir, name + ".wav"), "wb") as f:
            f.write(b"RIFF\x24\x00\x00\x00WAVEfmt " + name.encode())
    return cpr_path, audio_dir, proj_dir


class TestFieldRealMCP(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(BRIDGE_CMD[1]):
            raise unittest.SkipTest(f"CubeFlow bridge não encontrado em {BRIDGE_CMD[1]}")

    async def _make_client(self):
        app_data = tempfile.mkdtemp(prefix="cf_field_appdata_")
        coord_db = os.path.join(tempfile.mkdtemp(prefix="cf_field_coord_"), "peer.db")
        env = dict(os.environ)
        env["CUBEFLOW_APP_DATA"] = app_data
        env["CUBEFLOW_COORDINATION_DB"] = coord_db
        client = MCPClient(BRIDGE_CMD, env=env)
        await client.start()
        return client

    async def test_mcpclient_start_lists_real_tools(self):
        """MCPClient.start spawna o bridge real e popula tools (initialize +
        tools/list sobre stdio de verdade). Não mockado."""
        client = await self._make_client()
        try:
            tools = await client.tools()
            self.assertGreater(len(tools), 5, "bridge real expõe um registry real")
            names = [t["name"] for t in tools]
            self.assertIn("list_project_tracks", names)
            self.assertIn("smart_quantize", names)
            self.assertIn("verify_quality", names)
        finally:
            await client.close()

    async def test_list_project_tracks_resolves_real_scratch(self):
        """Discovery REAL end-to-end: MCPClient → bridge → resolver lendo .cpr
        + Audio/ scratch. Core do fix C3.2 (orquestrador chama exatamente isto).
        Bateria-only: BASS filtrado, 4 drums retornadas com file real no disco."""
        names = ["KICK IN", "SNARE TOP", "HIHAT", "TOM 1", "BASS"]
        cpr_path, audio_dir, proj_dir = _build_scratch_project(names)
        client = await self._make_client()
        try:
            res = await client.call("list_project_tracks",
                                    {"projectPath": cpr_path, "filter": "bateria"})
            self.assertTrue(res.get("ok"), f"resolver falhou: {res}")
            self.assertEqual(res["cpr"], cpr_path)
            self.assertEqual(res["audioDir"], audio_dir)
            got = [t["name"] for t in res["tracks"]]
            for drum in ["KICK IN", "SNARE TOP", "HIHAT", "TOM 1"]:
                self.assertIn(drum, got, f"{drum} deve resolver da pasta Audio/ real")
            self.assertNotIn("BASS", got, "BASS filtrado (não-bateria)")
            for t in res["tracks"]:
                self.assertTrue(t["file"] and os.path.exists(t["file"]),
                                f"WAV real existe: {t['name']}")
        finally:
            await client.close()
            import shutil
            shutil.rmtree(proj_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()