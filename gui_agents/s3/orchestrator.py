"""Orquestrador ReAct — claude-sonnet-5 tool-use dirigindo o CubeFlow via MCP."""
import asyncio
import json
import time
import uuid
from pathlib import Path

import anthropic

from gui_agents.s3.mcp_client import MCPToolError

SYSTEM_PROMPT = """Você é o orquestrador de edição de áudio do CubeFlow. Objetivo: editar a bateria
até verify_quality.perfect=true. Ferramentas: smart_quantize, correct_off_grid,
verify_quality, place_real_hitpoints, quantize, agent_s_operate.
Regra: PROBE antes de mexer. Não repita quantize se já aplicou — use correct_off_grid
para residuais. Se 3 ciclos não convergirem, escale para agent_s_operate (GUI).
Sempre termine chamando verify_quality. Responda só com tool_use."""


class Orchestrator:
    def __init__(self, mcp, anthropic_key, model="claude-sonnet-5",
                 logger=None, max_cycles=3, max_api_calls=20, peer_lock=None, peer_name="agent_s"):
        self.mcp = mcp
        self.model = model
        self.max_cycles = max_cycles
        self.max_api_calls = max_api_calls
        self.peer_lock = peer_lock
        self.peer_name = peer_name
        self.logger = logger or __import__("logging").getLogger(__name__)
        self._client = anthropic.Anthropic(api_key=anthropic_key)
        self._api_calls = 0
        self._telemetry = []

    async def run(self, task, session):
        run_id = uuid.uuid4().hex[:8]
        t0 = time.time()
        metrics = {"before": None, "after": None, "tool_calls": 0, "blocked": 0}
        # 1. PROBE
        probe = await self.mcp.call("verify_quality", self._verify_args(session))
        metrics["before"] = probe
        self._log(run_id, "probe", probe)
        if probe.get("perfect"):
            return self._report(run_id, True, 0, probe, probe, metrics, t0)
        last_quality = probe
        for cycle in range(1, self.max_cycles + 1):
            if self._api_calls >= self.max_api_calls:
                break
            # 2-3. PLAN + EXECUTE via Anthropic tool-use loop
            await self._react_loop(task, session, last_quality, run_id, metrics)
            if self._api_calls >= self.max_api_calls:
                break
            # 4. VERIFY
            q = await self.mcp.call("verify_quality", self._verify_args(session))
            metrics["after"] = q
            self._log(run_id, "verify", {"cycle": cycle, "quality": q})
            last_quality = q
            if q.get("perfect"):
                return self._report(run_id, True, cycle, probe, q, metrics, t0)
        # 8. ESCALATE — ciclos falham
        self._log(run_id, "escalate", {"reason": "ciclos esgotados"})
        try:
            await self.mcp.call("agent_s_operate", {"task": task})
        except MCPToolError as e:
            self._log(run_id, "escalate_failed", {"err": str(e)})
        return self._report(run_id, False, self.max_cycles, probe, last_quality, metrics, t0)

    async def _react_loop(self, task, session, quality, run_id, metrics):
        messages = [{
            "role": "user",
            "content": (
                f"TAREFA: {task}\nSESSÃO: {json.dumps(session)}\n"
                f"PROBLEMAS ATUAIS: {json.dumps(quality.get('issues', []))}\n"
                f"Resumo: {quality.get('summary','')}\n"
                "Escolha a próxima tool."
            ),
        }]
        tools = await self.mcp.tools()
        for _ in range(8):  # no máx 8 tool_uses por ciclo
            if self._api_calls >= self.max_api_calls:
                break
            resp = self._client.messages.create(
                model=self.model, max_tokens=4096, system=SYSTEM_PROMPT,
                tools=tools, messages=messages,
            )
            self._api_calls += 1
            if not resp.content or resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            any_tool = False
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                any_tool = True
                name, args = block.name, block.input or {}
                # gate destrutivo via peer-lock — espera 2s e re-checa (até 3x)
                if self.peer_lock and self.peer_lock.is_held_by_other(self.peer_name):
                    metrics["blocked"] += 1
                    self._log(run_id, "blocked", {"tool": name, "holder": "other"})
                    blocked = True
                    for _ in range(3):
                        await asyncio.sleep(2)
                        if not self.peer_lock.is_held_by_other(self.peer_name):
                            blocked = False
                            break
                    if blocked:
                        tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps({"error": "peer-lock ocupado por outro peer", "blocked": True}),
                            "is_error": True})
                        continue
                try:
                    result = await self.mcp.call(name, args)
                    metrics["tool_calls"] += 1
                    self._log(run_id, "tool", {"name": name, "args": args, "result": result})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps(result)})
                except Exception as e:
                    self._log(run_id, "tool_error", {"name": name, "err": str(e)})
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                        "content": json.dumps({"error": str(e)}), "is_error": True})
            if not any_tool:
                break
            messages.append({"role": "user", "content": tool_results})

    def _verify_args(self, session):
        args = {"tracks": session.get("tracks", []), "bpm": session.get("bpm")}
        if "phasePairs" in session: args["phasePairs"] = session["phasePairs"]
        args["toleranceMs"] = session.get("toleranceMs", 10)
        args["toleranceDb"] = session.get("toleranceDb", 3)
        return args

    def _report(self, run_id, perfect, cycles, before, after, metrics, t0):
        cost = self._api_calls * 0.003 + metrics["tool_calls"] * 0.0001  # estimativa rough
        report = (
            f"=== Relatório Agent-S3 [{run_id}] ===\n"
            f"Perfect: {perfect}\nCiclos: {cycles}\nAPI calls: {self._api_calls}\n"
            f"Tool calls: {metrics['tool_calls']}\nBlocked: {metrics['blocked']}\n"
            f"Antes: {json.dumps(before.get('issues', []))}\n"
            f"Depois: {json.dumps(after.get('issues', []))}\n"
            f"Custo est.: ${cost:.4f}\nTempo: {time.time()-t0:.1f}s"
        )
        return {"perfect": perfect, "cycles": cycles, "api_calls": self._api_calls,
                "report": report, "metrics": metrics, "cost_usd": cost}

    def _log(self, run_id, event, data):
        entry = {"ts": time.time(), "run": run_id, "event": event, "data": data}
        self._telemetry.append(entry)
        try:
            Path(f"/tmp/cubeflow-agent-{run_id}.jsonl").open("a").write(
                json.dumps(entry) + "\n")
        except Exception:
            pass