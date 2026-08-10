"""ChatCLI — painel interativo Rich + cmd (TIER 5).

Loop de input estilo chat que orquestra a arquitetura multi-agente cognitiva
(o1-like: planejar → executar → criticar):

    usuário digita tarefa
        │
        ▼
    1. VectorMemory ── consulta experiência passada (semântica)
        │
        ▼
    2. CriticAgent.research ── pesquisa docs na web se precisar
        │
        ▼
    3. Executor (LLM) ── gera o script (python/bash) dado task + contexto
        │
        ▼
    4. CriticAgent.review_code ── revisa bugs/paths/imports ANTES de rodar
        │
        ▼
    5. DockerExecutor ── roda o código corrigido num sandbox isolado
        │  (fallback subprocess se docker indisponível)
        ▼
    6. VectorMemory.save_success_trajectory ── grava trajetória vencedora

Env-gates (default OFF = behavior inalterado, mesmo padrão do resto do S3):
- ``AGENT_S3_USE_MEMORY=1``  → steps 1 e 6 ativos (VectorMemory/ChromaDB)
- ``AGENT_S3_USE_DOCKER=1``  → step 5 via DockerExecutor; senão subprocess host
- ``AGENT_S3_CRITIC_PROVIDER`` → "openai"|"anthropic" (default openai)

Degradación graciosa: qualquer peça faltante (API key, SDK, daemon, chromadb)
é reportada num Panel amarelo e o fluxo continua sem aquela peça — nunca
explode no rosto do usuário.

Uso::

    python -m gui_agents.s3.cli.chat_cli
    # ou no console_scripts: agent-s3-chat
"""
from __future__ import annotations

import cmd
import logging
import os
import sys
from typing import Any, List, Optional

logger = logging.getLogger("desktopenv.agent.cli")

# ───────────────────────────────────────────────────────────── rich (opcional)
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover — dep opcional
    _HAS_RICH = False
    Console = None  # type: ignore

# ─────────────────────────────────────────────────── peças do framework
from gui_agents.s3.cognition.critic_agent import CriticAgent, ReviewResult

try:
    from gui_agents.s3.execution.docker_executor import (
        DockerExecutor,
        ExecutionResult,
    )

    _HAS_DOCKER_EXEC = True
except Exception:  # import-guard: docker SDK pode faltar
    _HAS_DOCKER_EXEC = False
    DockerExecutor = None  # type: ignore
    ExecutionResult = None  # type: ignore

try:
    from gui_agents.s3.memory.vector_memory import (
        MemoryEntry,
        get_vector_memory,
    )

    _HAS_VECTOR_MEM = True
except Exception:  # import-guard: chromadb/sentence-transformers podem faltar
    _HAS_VECTOR_MEM = False
    MemoryEntry = None  # type: ignore
    get_vector_memory = None  # type: ignore

try:
    from gui_agents.s3.utils.local_env import LocalController

    _HAS_LOCAL_ENV = True
except Exception:  # pragma: no cover
    _HAS_LOCAL_ENV = False
    LocalController = None  # type: ignore


_CODEGEN_SYSTEM = (
    "You are an expert code generator. Given a task and supporting context, "
    "write a single self-contained, runnable {language} script that solves it. "
    "Output ONLY the script code — no markdown fences, no explanation. Use only "
    "stdlib + commonly-available libraries. Prefer relative paths. Fail loudly "
    "(raise on error) rather than silently."
)

_BASH_HINTS = ("bash", "shell", "terminal", "apt", "sed ", "grep ", "awk ", "chmod")


class ChatCLI(cmd.Cmd):
    """Loop de chat Rich + cmd que orquestra a cadeia cognitiva."""

    intro = "Agent-S3 Chat — digite uma tarefa (ex: 'quantize o audio xyz.mp3'). /help para comandos, /exit para sair."
    prompt = "agent-s3> "

    def __init__(self, critic: Optional[CriticAgent] = None) -> None:
        super().__init__()
        self.console = Console() if _HAS_RICH else _PlainConsole()
        self._critic: Optional[CriticAgent] = critic
        self._docker: Any = None  # DockerExecutor lazy (daemon pode estar off)
        self._local: Any = None  # LocalController fallback lazy
        self._mem: Any = None  # VectorMemory lazy
        self._use_memory = os.environ.get("AGENT_S3_USE_MEMORY") == "1"
        self._use_docker = os.environ.get("AGENT_S3_USE_DOCKER") == "1"
        self._print_banner()

    # ─────────────────────────────────────────────────────────── setup lazy
    @property
    def critic(self) -> CriticAgent:
        if self._critic is None:
            provider = os.environ.get("AGENT_S3_CRITIC_PROVIDER", "openai")
            try:
                self._critic = CriticAgent(provider=provider)
            except Exception as exc:
                self._warn(f"CriticAgent indisponível: {exc}\nDefina OPENAI_API_KEY/ANTHROPIC_API_KEY.")
                raise
        return self._critic

    def _memory(self) -> Any:
        if not self._use_memory or not _HAS_VECTOR_MEM:
            return None
        if self._mem is None:
            try:
                self._mem = get_vector_memory()
            except Exception as exc:
                self._warn(f"VectorMemory indisponível: {exc}")
                self._mem = False  # sentinel: desativada
        return self._mem if self._mem is not False else None

    def _executor(self) -> Any:
        """Devolve DockerExecutor (se habilitado+disponível) ou LocalController."""
        if self._use_docker and _HAS_DOCKER_EXEC and self._docker is not False:
            if self._docker is None:
                try:
                    self._docker = DockerExecutor()
                except Exception as exc:
                    self._warn(
                        f"Docker indisponível ({exc}) — fallback p/ subprocess host."
                    )
                    self._docker = False  # sentinel: não tenta de novo
        if self._docker and self._docker is not False:
            return ("docker", self._docker)
        # fallback subprocess
        if _HAS_LOCAL_ENV:
            if self._local is None:
                self._local = LocalController()
            return ("local", self._local)
        return (None, None)

    # ─────────────────────────────────────────────────────────── fluxo
    def run_task(self, user_input: str) -> None:
        """Cadeia cognitiva completa: memória → research → gen → review → run → save."""
        task = user_input.strip()
        if not task:
            return
        language = self._detect_language(task)
        self.console.print(Panel(f"[bold]{task}[/bold]\nlinguagem: {language}", title="Tarefa"))

        # 1. VectorMemory — experiência passada
        memory_block = self._step_memory(task)

        # 2. CriticAgent.research — docs web
        docs_block = self._step_research(task)

        # 3. Executor (LLM) — gera código
        code = self._step_generate(task, language, memory_block, docs_block)
        if not code:
            self._warn("Geração de código falhou (resposta vazia).")
            return
        self._show_code(code, language, title="Código gerado")

        # 4. CriticAgent.review_code — revisão pré-execução
        reviewed = self._step_review(code, language, task, docs_block)
        final_code = reviewed.corrected_code
        if reviewed.changed:
            self._show_code(final_code, language, title="Código revisado pelo Crítico")
            self._show_issues(reviewed.issues)
        else:
            self.console.print("[dim]Crítico: sem alterações.[/dim]")

        # 5. DockerExecutor / subprocess — roda
        result = self._step_run(final_code, language)
        if result is None:
            return  # erro já reportado

        # 6. VectorMemory — salva trajetória vencedora
        if result.get("success"):
            self._step_save(task, final_code, language, result)
            self.console.print("[bold green]✓ Tarefa concluída com sucesso.[/bold green]")
        else:
            self.console.print(
                f"[bold red]✗ Execução falhou.[/bold red]\n"
                f"exit: {result.get('exit_code')} stderr: {result.get('stderr', '')[:500]}"
            )

    # ───────────────────────────────────────────── steps individuais
    def _step_memory(self, task: str) -> str:
        mem = self._memory()
        if mem is None:
            return ""
        try:
            with self.console.status("[cyan]consultando memória…[/cyan]"):
                hits: List[Any] = mem.query_similar_experience(task, top_k=1)
        except Exception as exc:
            self._warn(f"query memória falhou: {exc}")
            return ""
        if not hits:
            self.console.print("[dim]memória: sem experiência similar.[/dim]")
            return ""
        top = hits[0]
        score = getattr(top, "score", 0.0)
        steps = getattr(top, "steps_taken", "")
        self.console.print(f"[dim]memória: experiência similar (score={score:.2f}).[/dim]")
        return f"Past successful trajectory (score={score:.2f}):\n{steps}\n"

    def _step_research(self, task: str) -> str:
        try:
            with self.console.status("[cyan]CriticAgent pesquisando docs…[/cyan]"):
                docs = self.critic.research(task)
        except Exception:
            return ""  # critic indisponível já avisado; sem docs
        if docs:
            self.console.print(f"[dim]research: {len(docs.splitlines())} linhas de docs.[/dim]")
        return docs

    def _step_generate(
        self, task: str, language: str, memory_block: str, docs_block: str
    ) -> str:
        context_parts = []
        if memory_block:
            context_parts.append(memory_block)
        if docs_block:
            context_parts.append(docs_block)
        context = "\n".join(context_parts)
        prompt = f"Task: {task}\n\n{context}\n\nWrite a single self-contained {language} script that solves the task. Output ONLY the code."
        try:
            with self.console.status(f"[cyan]Executor gerando código {language}…[/cyan]"):
                raw = self.critic.complete(prompt, system=_CODEGEN_SYSTEM.format(language=language))
        except Exception as exc:
            self._warn(f"geração falhou: {exc}")
            return ""
        return CriticAgent._strip_fences(raw).strip()

    def _step_review(
        self, code: str, language: str, task: str, docs_block: str
    ) -> ReviewResult:
        try:
            with self.console.status("[cyan]Crítico revisando código…[/cyan]"):
                return self.critic.review_code(
                    code, language=language, task=task, docs_context=docs_block
                )
        except Exception as exc:
            self._warn(f"revisão falhou: {exc}")
            return ReviewResult(corrected_code=code)

    def _step_run(self, code: str, language: str) -> Optional[dict]:
        kind, ex = self._executor()
        if ex is None:
            self._warn("Nenhum executor disponível (docker off + local_env ausente).")
            return None
        try:
            with self.console.status("[cyan]rodando no sandbox…[/cyan]"):
                if kind == "docker":
                    r = ex.run_python(code) if language == "python" else ex.run_bash(code)
                    return {
                        "success": r.success,
                        "exit_code": r.exit_code,
                        "stdout": r.stdout,
                        "stderr": r.stderr,
                        "duration_s": r.duration_s,
                        "timed_out": r.timed_out,
                    }
                # local subprocess
                if language == "python":
                    r = ex.run_python_script(code, timeout=30)
                else:
                    r = ex.run_bash_script(code, timeout=30)
                return {
                    "success": r.get("status") == "ok" or r.get("return_code") == 0,
                    "exit_code": r.get("return_code", r.get("exit_code")),
                    "stdout": r.get("output", ""),
                    "stderr": r.get("error", ""),
                    "duration_s": None,
                    "timed_out": "Timeout" in str(r.get("error", "")),
                }
        except Exception as exc:
            self._warn(f"execução falhou: {exc}")
            return None

    def _step_save(self, task: str, code: str, language: str, result: dict) -> None:
        mem = self._memory()
        if mem is None:
            return
        try:
            mem.save_success_trajectory(
                task_description=task,
                steps_taken=code,
                metadata={
                    "language": language,
                    "exit_code": result.get("exit_code"),
                    "via": "docker" if self._use_docker and self._docker else "local",
                },
            )
            self.console.print("[dim]memória: trajetória salva.[/dim]")
        except Exception as exc:
            self._warn(f"save memória falhou: {exc}")

    # ─────────────────────────────────────────────────────── helpers UI
    def _show_code(self, code: str, language: str, title: str) -> None:
        if _HAS_RICH:
            self.console.print(
                Panel(Syntax(code, language, theme="monokai", line_numbers=False), title=title)
            )
        else:
            print(f"--- {title} ---\n{code}\n")

    def _show_issues(self, issues: List[str]) -> None:
        if not issues:
            return
        if _HAS_RICH:
            t = Table(title="Issues do Crítico", show_header=True)
            t.add_column("#", style="dim")
            t.add_column("Issue")
            for i, issue in enumerate(issues, 1):
                t.add_row(str(i), issue)
            self.console.print(t)
        else:
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

    def _warn(self, msg: str) -> None:
        if _HAS_RICH:
            self.console.print(Panel(msg, title="⚠ Aviso", border_style="yellow"))
        else:
            print(f"[warn] {msg}")

    def _print_banner(self) -> None:
        flags = []
        flags.append(f"memory={'ON' if self._use_memory else 'off'}")
        flags.append(f"docker={'ON' if self._use_docker else 'off'}")
        flags.append(f"rich={'ON' if _HAS_RICH else 'off'}")
        if _HAS_RICH:
            self.console.print(Panel(" · ".join(flags), title="Agent-S3 Chat — flags"))

    @staticmethod
    def _detect_language(task: str) -> str:
        lowered = task.lower()
        if any(h in lowered for h in _BASH_HINTS):
            return "bash"
        return "python"

    # ─────────────────────────────────────────────────────────── cmd loop
    def default(self, line: str) -> bool:  # noqa: D401
        """Qualquer linha que não é comando vira tarefa."""
        try:
            self.run_task(line)
        except KeyboardInterrupt:
            self.console.print("\n[dim]interrompido.[/dim]")
        except Exception as exc:  # noqa: BLE001 — loop nunca morre
            self._warn(f"erro inesperado: {exc}")
        return False

    def emptyline(self) -> bool:
        return False

    def do_exit(self, _arg: str) -> bool:
        return True

    do_quit = do_exit
    do_EOF = do_exit

    def do_help(self, _arg: str) -> None:  # type: ignore[override]
        if _HAS_RICH:
            self.console.print(
                Panel(
                    "Digite qualquer frase como tarefa (ex: quantize o audio xyz.mp3)\n"
                    "/memory  — status da memória semântica\n"
                    "/exit    — sair",
                    title="Comandos",
                )
            )
        else:
            print(self.intro)

    def do_memory(self, _arg: str) -> None:
        mem = self._memory()
        if mem is None:
            self._warn("memória desativada (AGENT_S3_USE_MEMORY=1 p/ ativar).")
            return
        try:
            n = mem.count()
            self.console.print(f"[dim]memória: {n} trajetórias armazenadas.[/dim]")
        except Exception as exc:
            self._warn(f"count memória falhou: {exc}")


# ───────────────────────────────────────────────────────────── fallback UI
class _PlainConsole:
    """Fallback mínimo se rich não estiver instalado — keep CLI usável."""

    def print(self, *args: Any, **kwargs: Any) -> None:
        # rich Markup tokens ([bold], [dim]) — strip simples p/ texto puro.
        import re

        def _strip(s: Any) -> str:
            return re.sub(r"\[/?[a-zA-Z0-9 _=#]+/?\]", "", str(s)) if isinstance(s, str) else str(s)

        print(*(_strip(a) for a in args))

    def status(self, _msg: str) -> "_NoopCtx":
        return _NoopCtx()


class _NoopCtx:
    def __enter__(self) -> "_NoopCtx":
        return self

    def __exit__(self, *_: Any) -> None:
        pass


def main() -> None:
    """Entry point: ``python -m gui_agents.s3.cli.chat_cli`` ou console_script."""
    ChatCLI().cmdloop()


if __name__ == "__main__":
    main()