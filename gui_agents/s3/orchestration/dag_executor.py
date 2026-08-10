# gui_agents/s3/orchestration/dag_executor.py
"""DAGExecutor — motor de workflows (Grafo Direcionado Acíclico).

Melhorias sobre o draft original:
- Estado real por nó (PENDING/RUNNING/COMPLETED/FAILED/SKIPPED) — nó falho
  NÃO é marcado como executado; dependentes são SKIPPED, não rodados cegos.
- Detecção de ciclo DFS antes de executar (erro explícito, não loop morto).
- Dependência faltante = erro (não ignorado silenciosamente).
- Integra FASE 1: TaskStore (persiste lifecycle de cada nó), decorator
  @retry_with_backoff (opcional por nó), logging JSON com context_id.
- Executor async concorrente por camada (nós prontos independentes rodam
  em paralelo via asyncio.gather).
- Política de falha configurável: fail_fast (raise na 1a falha) ou
  continue (skip dependentes, segue). Default: continue.
"""
from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Union

from gui_agents.s3.logging_utils.structured_logger import (
    bind_context_id,
    get_logger,
    new_context_id,
    reset_context_id,
)
from gui_agents.s3.persistence.task_store import TaskStore, TaskStatus
from gui_agents.s3.retry.retry_decorator import retry_with_backoff

logger = get_logger("desktopenv.agent.dag")

# Fn pode ser sync ou async; recebe context e devolve resultado.
NodeFn = Callable[[dict[str, Any]], Any]


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # dependência falhou/foi skipped


class DAGCycleError(Exception):
    """O grafo contém um ciclo."""


class DAGNode:
    def __init__(
        self,
        task_id: str,
        executor_func: NodeFn,
        dependencies: Optional[list[str]] = None,
        *,
        retry: bool = False,
        max_attempts: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self.task_id = task_id
        self.executor_func = executor_func
        self.dependencies = list(dependencies or [])
        self.retry = retry
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.status = NodeStatus.PENDING
        self.result: Any = None
        self.error: Optional[str] = None

    @property
    def is_async(self) -> bool:
        return asyncio.iscoroutinefunction(self.executor_func)

    def _wrap_retry(self) -> NodeFn:
        """Aplica @retry_with_backoff se habilitado (sync ou async)."""
        if not self.retry:
            return self.executor_func
        return retry_with_backoff(
            max_attempts=self.max_attempts,
            backoff_base=self.backoff_base,
        )(self.executor_func)


class DAGExecutor:
    """Executor de DAG com persistência e tolerância a falhas."""

    def __init__(
        self,
        *,
        task_store: Optional[TaskStore] = None,
        fail_fast: bool = False,
    ) -> None:
        self.nodes: dict[str, DAGNode] = {}
        self.task_store = task_store  # Opcional — sem store = só in-memory.
        self.fail_fast = fail_fast

    # ------------------------------------------------------------------ build
    def add_node(
        self,
        task_id: str,
        executor_func: NodeFn,
        dependencies: Optional[list[str]] = None,
        *,
        retry: bool = False,
        max_attempts: int = 3,
        backoff_base: float = 2.0,
    ) -> DAGNode:
        if task_id in self.nodes:
            raise ValueError(f"Node {task_id} já existe no DAG.")
        node = DAGNode(
            task_id,
            executor_func,
            dependencies,
            retry=retry,
            max_attempts=max_attempts,
            backoff_base=backoff_base,
        )
        # Valida deps referenciam nós que já existem OU falham depois no
        # _validate. Aqui só rejeita self-loop óbvio.
        if task_id in node.dependencies:
            raise DAGCycleError(f"Self-loop: {task_id} depende de si mesmo.")
        self.nodes[task_id] = node
        logger.info(
            "dag_node_added",
            extra={"node": task_id, "deps": node.dependencies, "retry": retry},
        )
        return node

    # --------------------------------------------------------------- validate
    def _validate(self) -> None:
        """Dep faltante + ciclo (DFS com cores)."""
        for node in self.nodes.values():
            missing = [d for d in node.dependencies if d not in self.nodes]
            if missing:
                raise ValueError(
                    f"Node {node.task_id} deps faltantes: {missing}"
                )
        # DFS 3-cores: WHITE(não visto) → GRAY(pilha) → BLACK(pronto).
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {nid: WHITE for nid in self.nodes}

        def visit(nid: str, path: list[str]) -> None:
            if color[nid] == GRAY:
                raise DAGCycleError(
                    f"Ciclo detectado: {' -> '.join(path + [nid])}"
                )
            if color[nid] == BLACK:
                return
            color[nid] = GRAY
            for dep in self.nodes[nid].dependencies:
                visit(dep, path + [nid])
            color[nid] = BLACK

        for nid in self.nodes:
            if color[nid] == WHITE:
                visit(nid, [])

    # ------------------------------------------------------------------ ready
    def _ready_nodes(self) -> list[str]:
        """Prontos: PENDING cujas deps são todas COMPLETED."""
        ready = []
        for nid, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            if all(
                self.nodes[dep].status == NodeStatus.COMPLETED
                for dep in node.dependencies
            ):
                ready.append(nid)
        return ready

    def _dependents_blocked(self) -> list[str]:
        """PENDING cuja alguma dep é FAILED/SKIPPED → marcar SKIPPED."""
        blocked = []
        for nid, node in self.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            if any(
                self.nodes[dep].status in (NodeStatus.FAILED, NodeStatus.SKIPPED)
                for dep in node.dependencies
            ):
                blocked.append(nid)
        return blocked

    # --------------------------------------------------------------- persist
    def _persist(self, node: DAGNode, status: TaskStatus, run_cid: str) -> None:
        if self.task_store is None:
            return
        # #17: store_id composto por run → isola runs no store (re-run do mesmo
        # DAG não reusa/sobrescreve rows do run anterior). idempotency_key
        # garante que re-chamadas dentro do MESMO run sejam idempotentes.
        store_id = f"{run_cid}:{node.task_id}"
        rec = self.task_store.get(store_id)
        if rec is None:
            self.task_store.create(
                node.task_id,  # instrução = id do nó (executor opaco)
                task_id=store_id,
                metadata={"dag": True, "run_cid": run_cid},
                idempotency_key=store_id,
            )
        if status == TaskStatus.RUNNING:
            self.task_store.mark_running(store_id)
        elif status == TaskStatus.COMPLETED:
            self.task_store.set_result(store_id, node.result)
        elif status == TaskStatus.FAILED:
            self.task_store.set_error(store_id, node.error or "unknown")

    # ------------------------------------------------------------- exec sync
    def execute(self, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Executa o DAG sincrono, respeitando dependências."""
        self._validate()
        context = dict(context or {})
        run_cid = new_context_id()
        logger.info("dag_execute_start", extra={"nodes": len(self.nodes)})

        pending = len(self.nodes)
        while pending > 0:
            # 1. Skip nós bloqueados por falha upstream.
            for nid in self._dependents_blocked():
                node = self.nodes[nid]
                node.status = NodeStatus.SKIPPED
                logger.warning("dag_node_skipped", extra={"node": nid})
                pending -= 1

            ready = self._ready_nodes()
            if not ready:
                # Só sobraram blocked/skipped já tratados, ou deadlock.
                if all(
                    n.status != NodeStatus.PENDING for n in self.nodes.values()
                ):
                    break
                raise RuntimeError("Deadlock: nós PENDING sem deps resolvidos.")

            for nid in ready:
                node = self.nodes[nid]
                self._run_node_sync(node, context, run_cid)
                if node.status == NodeStatus.FAILED and self.fail_fast:
                    raise RuntimeError(
                        f"fail_fast: nó {nid} falhou: {node.error}"
                    )
                pending -= 1

        logger.info(
            "dag_execute_done",
            extra=self._summary(),
        )
        return context

    def _run_node_sync(
        self, node: DAGNode, context: dict[str, Any], run_cid: str
    ) -> None:
        token = bind_context_id(f"{run_cid}:{node.task_id}")
        node.status = NodeStatus.RUNNING
        self._persist(node, TaskStatus.RUNNING, run_cid)
        logger.info("dag_node_start", extra={"node": node.task_id})
        try:
            fn = node._wrap_retry()
            result = fn(context)
            node.result = result
            context[node.task_id] = result
            node.status = NodeStatus.COMPLETED
            self._persist(node, TaskStatus.COMPLETED, run_cid)
            logger.info(
                "dag_node_done",
                extra={"node": node.task_id, "status": "completed"},
            )
        except Exception as exc:  # noqa: BLE001 — error msg pro store/log
            node.error = str(exc)
            node.status = NodeStatus.FAILED
            context[node.task_id] = {"error": str(exc)}
            self._persist(node, TaskStatus.FAILED, run_cid)
            logger.error(
                "dag_node_failed",
                extra={"node": node.task_id, "error": str(exc)},
            )
        finally:
            reset_context_id(token)

    # ------------------------------------------------------------ exec async
    async def execute_async(
        self, context: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Executa o DAG async — nós prontos independentes em paralelo."""
        self._validate()
        context = dict(context or {})
        run_cid = new_context_id()
        logger.info("dag_execute_async_start", extra={"nodes": len(self.nodes)})

        pending = len(self.nodes)
        while pending > 0:
            for nid in self._dependents_blocked():
                self.nodes[nid].status = NodeStatus.SKIPPED
                logger.warning("dag_node_skipped", extra={"node": nid})
                pending -= 1

            ready = self._ready_nodes()
            if not ready:
                if all(
                    n.status != NodeStatus.PENDING for n in self.nodes.values()
                ):
                    break
                raise RuntimeError("Deadlock: nós PENDING sem deps resolvidos.")

            # Camada pronta em paralelo.
            await asyncio.gather(
                *(self._run_node(self.nodes[n], context, run_cid) for n in ready)
            )
            for nid in ready:
                if (
                    self.nodes[nid].status == NodeStatus.FAILED
                    and self.fail_fast
                ):
                    raise RuntimeError(
                        f"fail_fast: nó {nid} falhou: {self.nodes[nid].error}"
                    )
                pending -= 1

        logger.info("dag_execute_async_done", extra=self._summary())
        return context

    async def _run_node(
        self, node: DAGNode, context: dict[str, Any], run_cid: str
    ) -> None:
        token = bind_context_id(f"{run_cid}:{node.task_id}")
        node.status = NodeStatus.RUNNING
        self._persist(node, TaskStatus.RUNNING, run_cid)
        logger.info("dag_node_start", extra={"node": node.task_id})
        try:
            fn = node._wrap_retry()
            if node.is_async:
                result = await fn(context)
            else:
                result = fn(context)
            node.result = result
            context[node.task_id] = result
            node.status = NodeStatus.COMPLETED
            self._persist(node, TaskStatus.COMPLETED, run_cid)
            logger.info(
                "dag_node_done",
                extra={"node": node.task_id, "status": "completed"},
            )
        except Exception as exc:  # noqa: BLE001
            node.error = str(exc)
            node.status = NodeStatus.FAILED
            context[node.task_id] = {"error": str(exc)}
            self._persist(node, TaskStatus.FAILED, run_cid)
            logger.error(
                "dag_node_failed",
                extra={"node": node.task_id, "error": str(exc)},
            )
        finally:
            reset_context_id(token)

    # -------------------------------------------------------------- summary
    def _summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.status.value] = counts.get(n.status.value, 0) + 1
        return {"summary": counts, "nodes": len(self.nodes)}