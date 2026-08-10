# gui_agents/s3/observability/metrics.py
"""Observability — métricas Prometheus + alertas Slack.

Correções sobre o draft original:
- Singleton: ``track_task`` NÃO instancia ObservabilityManager por chamada
  (draft criava instância a cada falha → ``start_http_server`` repetido →
  "address already in use"). Métricas são module-level; servidor sobe 1x.
- ``start_http_server`` só no ``start()`` explícito (não no import, não por track).
- Decorator ``measure_duration`` com ``functools.wraps`` + suporte async.
- JSON logging (FASE 1), não f-string.
- Slack webhook via env ``AGENT_S3_SLACK_WEBHOOK`` (não hardcode).
- ``prometheus_client`` ausente → métricas viram no-op shim (módulo carrega,
  contam in-memory, não exportam). Instalar p/ expor: ``pip install
  prometheus_client``.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from gui_agents.s3.logging_utils.structured_logger import get_logger

logger = get_logger("desktopenv.agent.metrics")

# --------------------------------------------------------- prometheus shim
try:
    from prometheus_client import (
        Counter as _PromCounter,
        Histogram as _PromHistogram,
        start_http_server as _start_http_server,
    )
    _HAS_PROM = True
except ImportError:  # pragma: no cover — dep gate
    _HAS_PROM = False

    class _PromCounter:
        def __init__(self, *a, **k): self._labels = {}
        def labels(self, **kw): return self
        def inc(self, amount=1): pass

    class _PromHistogram:
        def __init__(self, *a, **k): pass
        def labels(self, **kw): return self
        def observe(self, amount): pass

    def _start_http_server(port):  # no-op
        logger.info("prometheus_server_skipped", extra={"reason": "prometheus_client missing", "port": port})


# Métricas module-level — singleton conceitual, criadas 1x.
TASKS_TOTAL: _PromCounter = _PromCounter(
    "agent_tasks_total", "Total tasks processed", ["status"]
)
TASK_DURATION: _PromHistogram = _PromHistogram(
    "agent_task_duration_seconds", "Time spent processing tasks"
)
ACTIONS_TOTAL: _PromCounter = _PromCounter(
    "agent_actions_total", "Total GUI/Code actions executed", ["type", "status"]
)


# ------------------------------------------------------------- manager
class ObservabilityManager:
    """Singleton de observability. Instancie 1x no entrypoint; chame start()."""

    _instance: Optional["ObservabilityManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        metrics_port: int = 8001,
        slack_webhook_url: Optional[str] = None,
    ) -> None:
        with self._lock:
            if getattr(self, "_initialized", False):
                return
            self.metrics_port = metrics_port
            # Env fallback — não loga o valor (segredo).
            self.slack_webhook_url = (
                slack_webhook_url
                or os.environ.get("AGENT_S3_SLACK_WEBHOOK")
            )
            self._server_started = False
            self._initialized = True

    def start(self) -> None:
        """Sobe servidor de métricas Prometheus (1x). Idempotente."""
        if self._server_started:
            return
        _start_http_server(self.metrics_port)
        self._server_started = True
        logger.info(
            "prometheus_server_started",
            extra={"port": self.metrics_port, "prom_installed": _HAS_PROM},
        )

    def send_alert(self, message: str) -> bool:
        """Envia alerta p/ Slack se webhook configurado. Devolve success."""
        if not self.slack_webhook_url:
            return False
        try:
            import requests  # dep já presente; fallback urllib seria redundância
            payload = {"text": f"🚨 [Agent-S3 Alert]: {message}"}
            requests.post(self.slack_webhook_url, json=payload, timeout=5)
            logger.info("slack_alert_sent", extra={"message": message})
            return True
        except Exception as exc:  # noqa: BLE001 — alerta não derruba fluxo
            logger.error("slack_alert_failed", extra={"error": str(exc)})
            return False


# ----------------------------------------------------- tracking functions
def track_task(status: str, *, alert_on_fail: bool = False) -> None:
    """Incrementa contador de tarefas. Opcionalmente alerta em falha."""
    TASKS_TOTAL.labels(status=status).inc()
    logger.info("task_tracked", extra={"status": status})
    if status == "failed" and alert_on_fail:
        ObservabilityManager().send_alert("Tarefa do Agent-S3 falhou.")


def track_action(action_type: str, status: str) -> None:
    """Incrementa contador de ações (click, type, script, hotkey...)."""
    ACTIONS_TOTAL.labels(type=action_type, status=status).inc()
    logger.info(
        "action_tracked", extra={"type": action_type, "status": status}
    )


def measure_duration() -> Callable[[Callable], Callable]:
    """Decorator: mede duração e observa no histograma. Sync + async.

    Preserva wraps; funciona com sync e coroutines.
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    TASK_DURATION.observe(time.time() - start)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                TASK_DURATION.observe(time.time() - start)

        return sync_wrapper

    return decorator