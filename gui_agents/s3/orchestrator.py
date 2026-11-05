"""Orquestrador ReAct — claude-sonnet-5 tool-use dirigindo o CubeFlow via MCP."""
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import anthropic

from gui_agents.s3.mcp_client import MCPToolError
from gui_agents.s3 import taskstore

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
        # Guarda a key e só constrói o client Anthropic se ela existir. Antes
        # `anthropic.Anthropic(api_key="")` podia levantar TypeError opaco na
        # construção (sem ANTHROPIC_API_KEY em env) — o erro explodia em _run_or
        # chestrator (server.py) como "orchestrator error" genérico. Agora a
        # ausência vira self._client=None e run() falha gracioso cedo (0 MCP
        # calls) com reason clara: missing_anthropic_api_key.
        # Strip: key com só whitespace ("  ") é truthy e derrotava o gate
        # `if not self._api_key` — o orquestrador procedia p/ discovery+PROBE+
        # ReAct e falhava opaco no 1º messages.create (misconfig comum de .env
        # `ANTHROPIC_API_KEY= ` com espaço). Strip fecha a brecha: whitespace
        # vira "" → None → abort gracioso, igual à key vazia.
        key = (anthropic_key or "").strip()
        self._api_key = key or None
        self._client = anthropic.Anthropic(api_key=key) if key else None
        self._api_calls = 0
        self._telemetry = []

    async def run(self, task, session):
        run_id = uuid.uuid4().hex[:8]
        t0 = time.time()
        metrics = {"before": None, "after": None, "tool_calls": 0, "blocked": 0, "tool_sequence": []}
        # 0a. VALIDA API KEY ANTES de qualquer coisa — sem key, nem adianta
        # descobrir tracks ou PROBEar: o ReAct vai falhar opaco no 1º messages.
        # Falha gracioso aqui (0 MCP calls) com reason clara.
        if not self._api_key:
            self._log(run_id, "abort", {"reason": "missing_anthropic_api_key"})
            return self._record_and_report(run_id, task, None, t0, False, 0,
                {"perfect": False, "error": "missing_anthropic_api_key"},
                {"perfect": False, "error": "missing_anthropic_api_key"}, metrics)
        # 0b. AUTO-DISCOVERY — se session não trouxer tracks, descobre do projeto
        # Cubase aberto via list_project_tracks (.cpr + pasta Audio/). Sem isso o
        # PROBE recebe tracks=[] e nada a quantizar. Discovery não é destrutivo.
        if not session.get("tracks"):
            disc = await self._discover_tracks(session, run_id)
            if disc is None:
                # erro já logado — reporta com reason clara, não crash
                return self._record_and_report(run_id, task, None, t0, False, 0,
                    {"perfect": False, "error": "discovery_failed"},
                    {"perfect": False, "error": "discovery_failed"}, metrics)
            if disc is False:
                # discovery rodou mas achou 0 tracks — nada a quantizar. Antes
                # prosseguia com tracks=[] e PROBEava sobre vazio. Aborta com
                # reason clara, sem PROBE.
                self._log(run_id, "abort", {"reason": "no_tracks_found"})
                return self._record_and_report(run_id, task, None, t0, False, 0,
                    {"perfect": False, "error": "no_tracks_found"},
                    {"perfect": False, "error": "no_tracks_found"}, metrics)
        # `project` (cpr) disambigua signatures entre projetos distintos com
        # mesmas inputs (ex. bpm=None) — evita cross-project replay da memória.
        sig = taskstore.signature(session.get("bpm"), session.get("grid", "1/16"),
                                  session.get("tracks", []),
                                  project=session.get("cpr"))
        # 1. PROBE
        probe = await self.mcp.call("verify_quality", self._verify_args(session))
        metrics["before"] = probe
        self._log(run_id, "probe", probe)
        if probe.get("perfect"):
            # já perfeito antes de mexer — grava sequência vazia (atalho: nada a fazer)
            return self._record_and_report(run_id, task, sig, t0, True, 0, probe, probe, metrics)
        # 1b. MEMÓRIA PROCEDURAL — sequência vencedora gravada pra essa signature?
        # Replaya direto (pula PLAN Anthropic), só VERIFY. Não perfect → cai no ReAct.
        last_quality = probe
        if session.get("use_memory", True) and taskstore.enabled():
            winning = taskstore.lookup_winning_sequence(sig)
            if winning:
                self._log(run_id, "memory_replay", {"sig": sig, "steps": len(winning)})
                try:
                    await self._replay_sequence(winning, run_id, metrics)
                except Exception as e:
                    self._log(run_id, "memory_replay_aborted", {"err": str(e)})
                else:
                    q = await self.mcp.call("verify_quality", self._verify_args(session))
                    metrics["after"] = q
                    self._log(run_id, "verify", {"cycle": "memory", "quality": q})
                    last_quality = q
                    if q.get("perfect"):
                        return self._record_and_report(run_id, task, sig, t0, True, 0, probe, q, metrics)
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
                return self._record_and_report(run_id, task, sig, t0, True, cycle, probe, q, metrics)
        # 8. ESCALATE — ciclos falham
        self._log(run_id, "escalate", {"reason": "ciclos esgotados"})
        try:
            await self.mcp.call("agent_s_operate", {"task": task})
        except MCPToolError as e:
            self._log(run_id, "escalate_failed", {"err": str(e)})
        return self._record_and_report(run_id, task, sig, t0, False, self.max_cycles, probe, last_quality, metrics)

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
                    metrics.setdefault("tool_sequence", []).append({"name": name, "args": args})
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

    async def _replay_sequence(self, seq, run_id, metrics):
        """Replaya uma sequência vencedora gravada [{name,args}]. Tools
        destrutivas AINDA passam pelo peer-lock gate — memória não bypassa
        segurança. Levanta em erro de tool (chamador cai no ReAct normal)."""
        for step in seq:
            name = step.get("name")
            args = step.get("args") or {}
            if self.peer_lock and self.peer_lock.is_held_by_other(self.peer_name):
                self._log(run_id, "replay_blocked", {"tool": name})
                metrics["blocked"] += 1
                continue
            result = await self.mcp.call(name, args)
            metrics["tool_calls"] += 1
            metrics.setdefault("tool_sequence", []).append({"name": name, "args": args})
            self._log(run_id, "replay_tool", {"name": name, "args": args, "result": result})

    def _record_and_report(self, run_id, task, sig, t0, perfect, cycles, before, after, metrics):
        """Grava a run no TaskStore (se habilitado) e devolve o relatório."""
        # Persiste before/after no dict de métricas em TODOS os caminhos (antes
        # só o verify-path setava metrics["after"]). Assim aborts (discovery
        # falhou/0 tracks, key vazia) expõem a reason em res["metrics"]["after"].
        metrics["before"] = before
        metrics["after"] = after
        if sig and taskstore.enabled():
            try:
                cost = self._api_calls * 0.003 + metrics["tool_calls"] * 0.0001
                taskstore.record_run(run_id, task, sig, t0, time.time(), cycles,
                                     self._api_calls, metrics["tool_calls"], cost,
                                     perfect, before, after,
                                     metrics.get("tool_sequence", []))
            except Exception as e:
                self.logger.warning("taskstore record falhou: %s", e)
        return self._report(run_id, perfect, cycles, before, after, metrics, t0)

    async def _discover_tracks(self, session, run_id):
        """Descobre tracks do projeto Cubase aberto via list_project_tracks.
        Popula session['tracks'] com [{name, file}] (só os que têm WAV).
        Retorna True se achou tracks, False se vazio, None se erro (já logado)."""
        flt = session.get("filter", "bateria")
        try:
            res = await self.mcp.call("list_project_tracks",
                                      {"filter": flt})
        except MCPToolError as e:
            self._log(run_id, "discovery_failed", {"err": str(e), "tool": "list_project_tracks"})
            self.logger.error("discovery list_project_tracks falhou: %s", e)
            return None
        tracks = (res or {}).get("tracks", []) if isinstance(res, dict) else []
        # só tracks com arquivo de áudio válido
        resolved = [{"name": t["name"], "file": t["file"]}
                    for t in tracks if t.get("name") and t.get("file")]
        session["tracks"] = resolved
        # guarda o cpr do projeto para disambiguar a signature (evita
        # cross-project replay quando bpm=None ou tracks homônimas).
        if isinstance(res, dict) and res.get("cpr"):
            session["cpr"] = res["cpr"]
        self._log(run_id, "discovery", {"filter": flt, "found": len(resolved),
                                         "diag": (res or {}).get("diagnostics") if isinstance(res, dict) else None})
        self.logger.info("discovery: %d tracks (filter=%s)", len(resolved), flt)
        return True if resolved else False

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