# gui_agents/s3/logging_utils/structured_logger.py
"""Logging estruturado em JSON com ``context_id`` p/ correlacionar tarefas.

Zero deps novas — stdlib ``logging`` + ``contextvars`` + ``json``. Substitui
os logs básicos do project (``logging.getLogger("desktopenv.agent")``) por
linhas JSON parseáveis por qualquer coletor (Loki, Datadog, jq).

Cada registro emitido inclui:
    - ts (ISO8601 UTC), level, logger, msg
    - context_id (correlação de tarefa, via contextvars — propaga em async)
    - event, extra fields passados via logger.info(msg, extra={...})

Uso:
    configure_logging(level="INFO")           # 1x no entrypoint
    token = bind_context_id("task-abc-123")   # ou new_context_id()
    log = get_logger("desktopenv.agent")
    log.info("task_started", extra={"instruction": "abra o Cubase"})
    # ... ao fim do contexto:
    reset_context_id(token)

``structlog``/``python-json-logger`` são upgrade opcional; este módulo
entrega o mesmo contrato JSON sem instalá-los.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from typing import Any, Optional

# contextvar propaga automaticamente em asyncio tasks (same task = same id).
_CONTEXT_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "agent_s3_context_id", default=None
)

_ROOT_LOGGER = "desktopenv.agent"


def context_id() -> Optional[str]:
    """ID de contexto corrente (ou None se fora de qualquer bind)."""
    return _CONTEXT_ID.get()


def new_context_id() -> str:
    """Gera um novo context_id e faz bind no contexto atual."""
    cid = str(uuid.uuid4())
    _CONTEXT_ID.set(cid)
    return cid


def bind_context_id(cid: str):
    """Faz bind de um context_id existente. Retorna token p/ restaurar.

    Uso:
        token = bind_context_id(task.id)
        try: ...
        finally: _CONTEXT_ID.reset(token)
    """
    return _CONTEXT_ID.set(cid)


def reset_context_id(token) -> None:
    """Restaura context_id ao estado anterior ao bind."""
    _CONTEXT_ID.reset(token)


class _JsonFormatter(logging.Formatter):
    """Formatter JSON. Inclui context_id + extras do record."""

    _RESERVED = frozenset(
        {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "taskName",
            "message",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "context_id": _CONTEXT_ID.get(),
        }
        # Extras definidos pelo caller viram chaves top-level.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str | int = "INFO",
    *,
    logger_name: str = _ROOT_LOGGER,
    json_output: Optional[bool] = None,
) -> logging.Logger:
    """Configura logging root do agent. Idempotente (limpa handlers antigos).

    Args:
        level: nível (str ou int).
        logger_name: logger alvo (default ``desktopenv.agent``).
        json_output: força JSON ou texto. Default: JSON se não for TTY,
            texto legível se TTY (dev).
    """
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    # Idempotente: remove handlers anteriores pra não duplicar linhas.
    for h in list(log.handlers):
        log.removeHandler(h)

    if json_output is None:
        json_output = not _is_tty()

    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] "
                "ctx=%(context_id)s %(message)s"
            )
        )
    log.addHandler(handler)
    log.propagate = False
    return log


def get_logger(name: str = _ROOT_LOGGER) -> logging.Logger:
    """Retorna logger configurado. Cria com defaults se não configurado."""
    log = logging.getLogger(name)
    if not log.handlers:
        configure_logging(logger_name=name)
    return log


def _is_tty() -> bool:
    try:
        return os.isatty(2)
    except Exception:
        return False