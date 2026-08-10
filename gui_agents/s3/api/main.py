# gui_agents/s3/api/main.py
"""FastAPI — plano de controle do Agent-S3.

Endpoints:
    POST /tasks         — cria e inicia tarefa (pool background)
    GET  /tasks/{id}    — status/estado da tarefa
    GET  /tasks         — lista tarefas (query ?status=&limit=)
    POST /workflows     — submete um DAG (nós referenciam handlers registrados)
    GET  /health        — health check básico

Integra: TaskStore (FASE 1), JSON logging com context_id (FASE 1),
observability track_task (FASE 3), DAGExecutor (FASE 2). Idempotência via
header ``X-Idempotency-Key``. Shutdown gracoso via lifespan (drain +
cancela PENDING).
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Header, Query
from pydantic import BaseModel, Field

from gui_agents.s3.logging_utils.structured_logger import (
    bind_context_id,
    configure_logging,
    get_logger,
    new_context_id,
    reset_context_id,
)
from gui_agents.s3.observability.metrics import track_task
from gui_agents.s3.orchestration.dag_executor import DAGExecutor, NodeStatus
from gui_agents.s3.persistence.task_store import TaskStatus, TaskStore

configure_logging()
logger = get_logger("desktopenv.agent.api")

# Handler plugável: assinatura (task_id, instruction, metadata) -> Any.
TaskHandler = Callable[[str, str, dict], Any]
# Action p/ workflows: recebe o context do DAG, devolve Any.
ActionFn = Callable[[dict], Any]

# Pool bounded — evita explosão de threads sob burst de POST /tasks.
MAX_WORKERS = int(os.environ.get("AGENT_S3_WORKERS", "8"))


class TaskCreateRequest(BaseModel):
    instruction: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    schedule_cron: Optional[str] = None  # se fornecido, agenda (cron) em vez de rodar já


class TaskResponse(BaseModel):
    id: str
    status: str
    instruction: str
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int
    created_at: float
    updated_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    tasks_total: int
    tasks_running: int
    tasks_failed: int
    orphans_recovered: int
    uptime_s: float


# ----------------------------------------------------------- workflow models
class WorkflowNodeSpec(BaseModel):
    id: str
    handler: str                       # nome de action registrada via register_action
    dependencies: list[str] = Field(default_factory=list)
    retry: bool = False
    max_attempts: int = 3
    backoff_base: float = 2.0


class WorkflowRequest(BaseModel):
    nodes: list[WorkflowNodeSpec]
    context: dict[str, Any] = Field(default_factory=dict)


class WorkflowResponse(BaseModel):
    workflow_id: str
    status: str
    summary: dict[str, int]
    context: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------- app state
def _maybe_scheduler():
    """Cria TaskScheduler se apscheduler estiver instalado; senão None."""
    try:
        from gui_agents.s3.orchestration.scheduler import TaskScheduler
        return TaskScheduler()
    except ImportError:
        return None


def create_app(
    task_store: Optional[TaskStore] = None,
    handler: Optional[TaskHandler] = None,
    scheduler=None,
    *,
    recover_orphans: bool = True,
) -> FastAPI:
    """Factory — permite injetar store, handler e scheduler (testes/prod).

    Args:
        recover_orphans: no startup, marca RUNNING/RETRYING órfãos (de crash
            anterior) como FAILED. Evita tasks presas p/ sempre no DB.
    """
    from contextlib import asynccontextmanager

    store = task_store or TaskStore()
    state: dict[str, Any] = {
        "store": store,
        "handler": handler,
        "scheduler": scheduler,  # None → criado no lifespan se apscheduler ok
        "pool": ThreadPoolExecutor(max_workers=MAX_WORKERS,
                                   thread_name_prefix="task"),
        "actions": {},            # registro nome → ActionFn p/ workflows
        "shutting_down": False,   # flag p/ shutdown gracioso
        "started_at": time.time(),
        "orphans_recovered": 0,
    }

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Reaper de startup: RUNNING/RETRYING órfãos → FAILED.
        if recover_orphans:
            n = store.recover_orphans(requeue=False)
            state["orphans_recovered"] = n
            if n:
                logger.warning(
                    "orphans_recovered", extra={"count": n}
                )
        # Scheduler criado AQUI (não lazy no request handler) → sem race.
        sched = state["scheduler"] or _maybe_scheduler()
        state["scheduler"] = sched
        if sched is not None:
            sched.start()
        yield
        # ---- shutdown gracoso ----
        # 1. Sinaliza _run_task p/ não iniciar novas PENDING.
        state["shutting_down"] = True
        # 2. Scheduler para de disparar novos jobs.
        if sched is not None:
            sched.shutdown(wait=False)
        # 3. Marca PENDING (na fila, ainda não começaram) como CANCELLED.
        cancelled = store.cancel_pending(reason="shutdown")
        if cancelled:
            logger.info("shutdown_cancelled_pending", extra={"count": cancelled})
        # 4. Pool: espera tarefas em voo terminarem; cancela as que ainda
        #    estão na fila (já marcadas CANCELLED no store, então _run_task
        #    que eventualmente pegá-las verá o status e skip).
        state["pool"].shutdown(wait=True, cancel_futures=True)

    app = FastAPI(
        title="Agent-S3 Control API",
        version="0.3.0",
        lifespan=_lifespan,
    )

    def _default_handler(task_id: str, instruction: str, meta: dict) -> Any:
        # Stub — registra que rodou sem plugar o Worker de verdade.
        logger.info(
            "task_handler_stub",
            extra={"task_id": task_id, "instruction": instruction},
        )
        return {"stub": True, "instruction": instruction}

    def _get_handler() -> TaskHandler:
        return state["handler"] or _default_handler

    def _run_task(task_id: str, instruction: str, meta: dict) -> None:
        """Pool worker — executa handler e atualiza o store + observability.
        Usado por POST /tasks (pool) e por jobs cron do scheduler.
        """
        token = bind_context_id(task_id)
        s: TaskStore = state["store"]
        # Shutdown em andamento: cancela e não executa (não estava RUNNING ainda).
        if state["shutting_down"]:
            s.cancel(task_id)
            track_task("cancelled")
            logger.info("task_cancelled_shutdown", extra={"task_id": task_id})
            reset_context_id(token)
            return
        s.mark_running(task_id)
        try:
            result = _get_handler()(task_id, instruction, meta)
            s.set_result(task_id, result)
            track_task("completed")
            logger.info("task_completed", extra={"task_id": task_id})
        except Exception as exc:  # noqa: BLE001
            s.set_error(task_id, str(exc))
            # alert_on_fail=True dispara Slack se webhook configurado.
            track_task("failed", alert_on_fail=True)
            logger.error(
                "task_failed",
                extra={"task_id": task_id, "error": str(exc)},
            )
        finally:
            reset_context_id(token)

    def _cron_fire(instruction: str, meta: dict) -> None:
        """Cada disparo cron = NOVA tarefa (não reusa id do agendamento).

        O rec.id do agendamento é só p/ nomear o job; o histórico de cada
        fire é uma linha própria no store.
        """
        fire_meta = {**meta, "trigger": "cron"}
        rec = state["store"].create(instruction, metadata=fire_meta)
        state["pool"].submit(_run_task, rec.id, instruction, fire_meta)

    # ---------------------------------------------------------------- routes
    @app.post("/tasks", response_model=TaskResponse, status_code=202)
    def create_task(
        req: TaskCreateRequest,
        idem: Optional[str] = Header(None, alias="X-Idempotency-Key"),
    ) -> TaskResponse:
        s: TaskStore = state["store"]

        if req.schedule_cron:
            # Modo agendado: cria PENDING (template) + registra cron que
            # gera nova task a cada fire. Não roda agora.
            sched = state["scheduler"]
            if sched is None:
                raise HTTPException(
                    status_code=400,
                    detail="schedule_cron requer apscheduler. "
                           "Rode: pip install apscheduler",
                )
            meta = {**req.metadata, "schedule_cron": req.schedule_cron}
            rec = s.create(req.instruction, metadata=meta, idempotency_key=idem)
            sched.add_cron(
                _cron_fire,
                job_id=f"cron:{rec.id}",
                cron=req.schedule_cron,
                args=(req.instruction, meta),
            )
            logger.info(
                "task_scheduled",
                extra={"task_id": rec.id, "cron": req.schedule_cron},
            )
            return _to_response(rec)

        # Modo imediato: submete ao pool (bounded, não Thread cru).
        rec = s.create(req.instruction, metadata=req.metadata, idempotency_key=idem)
        # Se idempotência devolveu task pré-existente, NÃO re-submete (já rodando/rodou).
        if idem and rec.status != TaskStatus.PENDING:
            logger.info(
                "task_idempotent_hit",
                extra={"task_id": rec.id, "status": rec.status.value},
            )
            return _to_response(rec)
        state["pool"].submit(
            _run_task, rec.id, req.instruction, req.metadata
        )
        logger.info(
            "task_accepted",
            extra={"task_id": rec.id, "instruction": req.instruction},
        )
        return _to_response(rec)

    @app.get("/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> TaskResponse:
        store: TaskStore = state["store"]
        rec = store.get(task_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="task not found")
        return _to_response(rec)

    @app.get("/tasks", response_model=list[TaskResponse])
    def list_tasks(
        status: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=1000),
    ) -> list[TaskResponse]:
        store: TaskStore = state["store"]
        st = TaskStatus(status) if status else None
        return [_to_response(r) for r in store.list(status=st, limit=limit)]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        store: TaskStore = state["store"]
        return HealthResponse(
            status="ok",
            tasks_total=store.count(),
            tasks_running=store.count(TaskStatus.RUNNING),
            tasks_failed=store.count(TaskStatus.FAILED),
            orphans_recovered=state["orphans_recovered"],
            uptime_s=round(time.time() - state["started_at"], 3),
        )

    # ------------------------------------------------------------- workflows
    def _execute_workflow(
        wf_task_id: str, nodes: list[WorkflowNodeSpec], context: dict[str, Any]
    ) -> None:
        """Pool worker — roda um DAG de handlers registrados.

        Cada nó referencia um handler pelo nome no registro ``state["actions"]``.
        Dependências respeitadas; falha de nó skipa dependentes (fail_fast=False).
        Resultado do workflow é armazenado na task wf_task_id.
        """
        token = bind_context_id(wf_task_id)
        s: TaskStore = state["store"]
        if state["shutting_down"]:
            s.cancel(wf_task_id)
            track_task("cancelled")
            reset_context_id(token)
            return
        s.mark_running(wf_task_id)
        try:
            actions: dict[str, ActionFn] = state["actions"]
            dag = DAGExecutor(task_store=s, fail_fast=False)
            for n in nodes:
                fn = actions.get(n.handler)
                if fn is None:
                    raise KeyError(
                        f"handler '{n.handler}' não registrado. "
                        f"Disponíveis: {sorted(actions)}"
                    )
                dag.add_node(
                    n.id,
                    executor_func=fn,
                    dependencies=n.dependencies,
                    retry=n.retry,
                    max_attempts=n.max_attempts,
                    backoff_base=n.backoff_base,
                )
            # execute() devolve o context MUTADO (copia internamente e popula
            # com resultados de cada nó). Usar o retorno, não a var original.
            out_ctx = dag.execute(context)
            summary = dag._summary()["summary"]
            result = {
                "summary": summary,
                "context": out_ctx,
            }
            s.set_result(wf_task_id, result)
            track_task("completed")
            logger.info("workflow_completed", extra={"task_id": wf_task_id,
                                                       "summary": summary})
        except Exception as exc:  # noqa: BLE001
            s.set_error(wf_task_id, str(exc))
            track_task("failed", alert_on_fail=True)
            logger.error("workflow_failed", extra={"task_id": wf_task_id,
                                                    "error": str(exc)})
        finally:
            reset_context_id(token)

    @app.post("/workflows", response_model=WorkflowResponse, status_code=202)
    def submit_workflow(req: WorkflowRequest) -> WorkflowResponse:
        actions: dict[str, ActionFn] = state["actions"]
        # Validação eagerly: handlers existem? DAG válido?
        missing = [n.handler for n in req.nodes if n.handler not in actions]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"handlers não registrados: {sorted(set(missing))}. "
                       f"Disponíveis: {sorted(actions)}",
            )
        s: TaskStore = state["store"]
        rec = s.create(
            f"workflow: {len(req.nodes)} nodes",
            metadata={"type": "workflow", "nodes": [n.id for n in req.nodes]},
        )
        state["pool"].submit(_execute_workflow, rec.id, req.nodes, dict(req.context))
        logger.info(
            "workflow_accepted",
            extra={"task_id": rec.id, "node_count": len(req.nodes)},
        )
        return WorkflowResponse(
            workflow_id=rec.id,
            status=rec.status.value,
            summary={"total": len(req.nodes)},
            context=dict(req.context),
        )

    # Permite trocar handler depois de criado (prod pluga Worker).
    app.state.set_handler = lambda h: state.__setitem__("handler", h)
    # Registro de actions p/ workflows: app.state.register_action("foo", fn)
    def _register_action(name: str, fn: ActionFn) -> None:
        state["actions"][name] = fn
    app.state.register_action = _register_action
    app.state.store = state["store"]
    return app


def _to_response(rec) -> TaskResponse:
    return TaskResponse(
        id=rec.id,
        status=rec.status.value,
        instruction=rec.instruction,
        result=rec.result,
        error=rec.error,
        attempts=rec.attempts,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
        started_at=rec.started_at,
        completed_at=rec.completed_at,
        metadata=rec.metadata,
    )


# Instância default p/ ``uvicorn gui_agents.s3.api.main:app``.
app = create_app()


def main() -> None:
    """Entry point: ``python -m gui_agents.s3.api.main``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")


if __name__ == "__main__":
    main()