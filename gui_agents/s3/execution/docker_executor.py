"""DockerExecutor — sandbox de código em contêiner efêmero (TIER 4 MÓDULO 1).

O agente NÃO roda Python/Bash gerado por LLM no host (risco de alucinação
destrutiva). Todo código é executado num contêiner Docker isolado, com:

- volume temporário bind-mount (escrita só no work_dir)
- rede desabilitada por padrão (código sandboxed sem internet)
- cap_drop=ALL + no-new-privileges (mínimos privilégios)
- timeout com kill forçado
- destruição automática (GC) do contêiner + do volume após cada execução

Requer: ``pip install docker`` + daemon Docker rodando.
**Import-guard:** o módulo carrega mesmo sem o SDK / sem daemon. O erro
explícito só aparece ao instanciar :class:`DockerExecutor` — mesmo padrão
usado por ``orchestration/scheduler.py`` (APScheduler) e
``observability/metrics.py`` (prometheus), para não quebrar import em
ambientes sem a dep opcional.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("desktopenv.agent.docker")

# ───────────────────────────────────────────────────── import-guard: docker SDK
try:
    import docker as _docker  # type: ignore
    from docker.errors import APIError as _DockerAPIError  # type: ignore
    from docker.errors import ImageNotFound as _DockerImageNotFound  # type: ignore

    _HAS_DOCKER = True
except ImportError:  # pragma: no cover — ambiente sem SDK
    _docker = None  # type: ignore
    _DockerAPIError = Exception  # type: ignore
    _DockerImageNotFound = Exception  # type: ignore
    _HAS_DOCKER = False


def _require_docker() -> Any:
    """Devolve um cliente Docker validado OU levanta ``RuntimeError`` explicativo.

    Fail-fast: chamar no ``__init__`` para não esperar o primeiro ``run_*``
    descobrir que o daemon está morto.
    """
    if not _HAS_DOCKER:
        raise RuntimeError(
            "docker SDK ausente — instale com: pip install docker"
        )
    try:
        client = _docker.from_env()  # type: ignore[union-attr]
        client.ping()  # daemon reachável?
        return client
    except Exception as exc:  # docker.errors.DockerException etc.
        raise RuntimeError(
            f"daemon Docker inacessível ({exc}). Inicie o Docker Desktop/daemon."
        ) from exc


# ─────────────────────────────────────────────────────────────────── resultado
@dataclass
class ExecutionResult:
    """Snapshot imutável de uma execução sandboxed."""

    success: bool
    exit_code: Optional[int]
    stdout: str
    stderr: str
    duration_s: float
    container_id: str
    command: List[str] = field(default_factory=list)

    @property
    def timed_out(self) -> bool:
        """True se a execução foi morta por estourar o timeout (exit_code == -1)."""
        return self.exit_code == -1

    def __bool__(self) -> bool:
        return self.success


# ─────────────────────────────────────────────────────────────────── executor
class DockerExecutor:
    """Sandbox de execução via contêiner Docker efêmero.

    Defaults de hardening (produção):
      * ``network_disabled=True``  — código sandboxed sem internet.
      * ``auto_remove=True``       — contêiner destruído após execução.
      * ``cap_drop=["ALL"]`` + ``no-new-privileges`` — mínimos privilégios.
      * ``mem_limit``              — cota de RAM (evita OOM no host).
      * ``timeout``                — kill forçado se exceder.

    Uso típico::

        executor = DockerExecutor()
        r = executor.run_python("print(2+2)")
        assert r.success and r.stdout.strip() == "4"
    """

    DEFAULT_IMAGE = os.environ.get("AGENT_S3_SANDBOX_IMAGE", "python:3.11-slim")

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        work_dir: str = "/workspace",
        network_disabled: bool = True,
        timeout: float = 120.0,
        auto_remove: bool = True,
        mem_limit: str = "512m",
    ) -> None:
        self.image = image
        self.work_dir = work_dir
        self.network_disabled = network_disabled
        self.timeout = timeout
        self.auto_remove = auto_remove
        self.mem_limit = mem_limit
        # Fail-fast: valida SDK + daemon no construtor.
        self._client = _require_docker()
        self._ensure_image()
        logger.info(
            "docker_executor_ready",
            extra={
                "image": image,
                "network_disabled": network_disabled,
                "timeout": timeout,
                "mem_limit": mem_limit,
            },
        )

    # ─────────────────────────────────────────────────────────── API pública
    def run_python(
        self,
        script: str,
        files: Optional[Dict[str, bytes]] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Executa ``script`` (Python) no contêiner.

        Args:
            script: código Python fonte.
            files: mapa ``nome → bytes`` gravado no work_dir antes de rodar.
            timeout: sobrescreve o timeout do construtor p/ esta chamada.
            env: variáveis de ambiente injetadas no contêiner (ex.: context_id).
        """
        return self._run_script(script, files, "python", timeout, env)

    def run_bash(
        self,
        script: str,
        files: Optional[Dict[str, bytes]] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Executa ``script`` (Bash) no contêiner (requer ``bash`` na imagem)."""
        return self._run_script(script, files, "bash", timeout, env)

    # ───────────────────────────────────────────────────────────── internos
    def _ensure_image(self) -> None:
        """Puxa a imagem se não estiver em cache local (evita ImageNotFound no run)."""
        try:
            self._client.images.get(self.image)
        except _DockerImageNotFound:
            logger.info("docker_image_pull", extra={"image": self.image})
            self._client.images.pull(self.image)

    def _run_script(
        self,
        script: str,
        files: Optional[Dict[str, bytes]],
        interpreter: str,
        timeout: Optional[float],
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        host_dir = Path(tempfile.mkdtemp(prefix="agent_s3_sandbox_"))
        try:
            # Marker c/ PID p/ o reaper limpar dirs órfãos sob SIGKILL/OOM (#2).
            (host_dir / ".agent_s3_pid").write_text(str(os.getpid()), encoding="utf-8")
            # Escreve script + arquivos no volume host (bind-mount → work_dir).
            if interpreter == "python":
                (host_dir / "script.py").write_text(script, encoding="utf-8")
                command = ["python", f"{self.work_dir}/script.py"]
            else:
                (host_dir / "script.sh").write_text(script, encoding="utf-8")
                command = ["bash", f"{self.work_dir}/script.sh"]
            for name, data in (files or {}).items():
                # Sanitiza nome: sem path traversal p/ fora do host_dir.
                safe = Path(name).name
                (host_dir / safe).write_bytes(data)
            return self._execute(command, host_dir, timeout, env)
        except Exception as exc:
            logger.error(
                "sandbox_setup_error",
                extra={"error": str(exc), "interpreter": interpreter},
            )
            return ExecutionResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=f"setup error: {exc}",
                duration_s=0.0,
                container_id="",
                command=[],
            )
        finally:
            shutil.rmtree(host_dir, ignore_errors=True)

    def _execute(
        self,
        command: List[str],
        host_dir: Path,
        timeout: Optional[float],
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        to = self.timeout if timeout is None else timeout
        run_kwargs: Dict[str, Any] = dict(
            image=self.image,
            command=command,
            volumes={
                str(host_dir.resolve()): {"bind": self.work_dir, "mode": "rw"}
            },
            working_dir=self.work_dir,
            detach=True,  # detached p/ controlar timeout + GC manual
            network_disabled=self.network_disabled,
            mem_limit=self.mem_limit,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            # auto_remove tratado no GC p/ garantir logs mesmo em kill.
            auto_remove=False,
            # Label p/ o reaper (reap_orphans) encontrar órfãos sob SIGKILL/OOM.
            labels={
                "agent_s3.owner": f"pid{os.getpid()}",
                "agent_s3.managed": "1",
            },
        )
        # #10: injeta env vars (ex.: AGENT_S3_CONTEXT_ID) p/ correlação de logs.
        if env:
            run_kwargs["environment"] = env
        container = None
        start = time.time()
        try:
            container = self._client.containers.run(**run_kwargs)
            cid = container.id[:12]
            logger.info(
                "sandbox_started",
                extra={"cid": cid, "cmd": command},
            )

            # Espera com timeout; mata se estourar.
            exit_code: Optional[int]
            try:
                res = container.wait(timeout=to)
                exit_code = int(res.get("StatusCode", -1))
            except Exception:  # ReadTimeout / APIError → mata e marca timeout
                try:
                    container.kill()
                except Exception:
                    pass
                logger.warning(
                    "sandbox_timeout",
                    extra={"cid": cid, "timeout": to},
                )
                exit_code = -1

            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )
            if exit_code == -1 and not stderr:
                stderr = f"execution timed out after {to}s"

            duration = round(time.time() - start, 3)
            logger.info(
                "sandbox_done",
                extra={"cid": cid, "exit": exit_code, "duration": duration},
            )
            return ExecutionResult(
                success=(exit_code == 0),
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
                container_id=container.id,
                command=command,
            )
        except Exception as exc:
            logger.error(
                "sandbox_error",
                extra={"error": str(exc), "cmd": command},
            )
            return ExecutionResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                duration_s=round(time.time() - start, 3),
                container_id=container.id if container else "",
                command=command,
            )
        finally:
            self._cleanup(container)

    def _cleanup(self, container: Any) -> None:
        """GC: força remoção do contêiner (auto_remove=False → manual)."""
        if container is None:
            return
        for op in ("kill", "remove"):
            try:
                getattr(container, op)(force=True)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────── reaper (orphan GC)
    def reap_orphans(self) -> int:
        """Remove containers ``agent_s3.managed`` órfãos de PIDs mortos.

        Sob SIGKILL/OOM do processo Python, o ``finally`` que chama
        :meth:`_cleanup` NÃO roda → containers ``detach=True`` + volumes
        ``mkdtemp`` vazam. Este reaper (chamar no startup do executor e
        opcionalmente no lifespan da API) remove os órfãos por label.

        Retorna a quantidade removida.
        """
        if not _HAS_DOCKER:
            return 0
        try:
            client = _require_docker()
            alive_pids = _alive_pids()
            alive = {f"pid{p}" for p in alive_pids}
            n = 0
            for c in client.containers.list(
                all=True, filters={"label": "agent_s3.managed=1"}
            ):
                owner = c.labels.get("agent_s3.owner", "")
                if owner and owner not in alive:
                    try:
                        c.remove(force=True)
                        n += 1
                    except Exception:  # noqa: BLE001
                        pass
            # #2: limpa dirs temp órfãos (mkdtemp vazia sob SIGKILL/OOM).
            dirs = _reap_temp_dirs(alive_pids)
            if n or dirs:
                logger.info(
                    "docker_reaped_orphans",
                    extra={"containers": n, "temp_dirs": dirs},
                )
            return n
        except Exception:  # noqa: BLE001
            return 0


def _alive_pids() -> List[int]:
    """PIDs vivos relevantes p/ o reaper. Default: só o atual.

    Em produção multi-process (ex.: gunicorn workers), expandir p/ varrer
    ``/proc`` (Linux) ou ``ps`` — aqui mantido mínimo (Simplicity First).
    """
    import os as _os

    return [_os.getpid()]


def _reap_temp_dirs(alive_pids: List[int]) -> int:
    """Remove dirs ``agent_s3_sandbox_*`` órfãos em /tmp cujo PID-marker morreu.

    Sob SIGKILL/OOM, o ``finally`` que faz ``shutil.rmtree(host_dir)`` não
    roda → dirs temp vazam. O marker ``.agent_s3_pid`` (escrito em
    :meth:`_run_script`) permite ao reaper identificar órfãos por PID.
    """
    import glob
    import shutil as _shutil

    alive = set(alive_pids)
    n = 0
    for d in glob.glob(str(Path(tempfile.gettempdir()) / "agent_s3_sandbox_*")):
        marker = Path(d) / ".agent_s3_pid"
        if not marker.exists():
            continue
        try:
            pid = int(marker.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            continue
        if pid not in alive:
            _shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n