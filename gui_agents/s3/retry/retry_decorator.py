# gui_agents/s3/retry/retry_decorator.py
"""@retry_with_backoff — decorator de retentativa com backoff exponencial.

Lida com falhas transitórias (API, subprocesso, IO). Sincrono e assíncrono.
Zero deps — stdlib apenas (project já tem lib ``backoff``, mas este decorator
é auto-contido pra ser reusável em camadas que não dependem do engine).

Uso:
    @retry_with_backoff(max_attempts=3, backoff_base=2.0)
    def call_api(...): ...

    @retry_with_backoff(exceptions=(HTTPError, TimeoutError), jitter=0.3)
    async def fetch(...): ...

Backoff: wait = backoff_base ** (attempt-1) + jitter uniforme[0, jitter).
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Awaitable, Callable, TypeVar, Union

logger = logging.getLogger("desktopenv.agent")

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

# Exceções transitórias padrão — callers podem sobrescrever via ``exceptions``.
DEFAULT_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)


class RetryExhausted(Exception):
    """Todas as tentativas falharam. Encapsula a última exceção."""

    def __init__(self, last_exc: BaseException, attempts: int) -> None:
        self.last_exc = last_exc
        self.attempts = attempts
        super().__init__(f"retry exhausted after {attempts} attempts: {last_exc!r}")


def _wait_seconds(attempt: int, backoff_base: float, jitter: float) -> float:
    """Backoff exponencial + jitter. attempt é 1-based."""
    base = backoff_base ** (attempt - 1)
    return base + random.uniform(0, jitter) if jitter > 0 else base


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    *,
    exceptions: tuple[type[BaseException], ...] = DEFAULT_EXCEPTIONS,
    jitter: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[F], F]:
    """Decora fn sync/async com retentativa.

    Args:
        max_attempts: tentativas totais (inclui a primeira).
        backoff_base: base do expoente. wait = base**(attempt-1).
        exceptions: tupla de exceções que disparam retry. Outras propagam já.
        jitter: segundos extras uniformes [0, jitter) por tentativa.
        sleep: injetável p/ testes (default time.sleep).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if backoff_base <= 0:
        raise ValueError("backoff_base must be > 0")
    if jitter < 0:
        raise ValueError("jitter must be >= 0")

    def decorator(fn: F) -> F:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exc: BaseException | None = None
                for attempt in range(1, max_attempts + 1):
                    try:
                        return await fn(*args, **kwargs)
                    except exceptions as exc:
                        last_exc = exc
                        if attempt >= max_attempts:
                            break
                        wait = _wait_seconds(attempt, backoff_base, jitter)
                        logger.warning(
                            "retry async %s attempt %d/%d failed: %r — "
                            "backing off %.2fs",
                            fn.__qualname__,
                            attempt,
                            max_attempts,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)
                assert last_exc is not None
                logger.error(
                    "retry async %s exhausted after %d attempts",
                    fn.__qualname__,
                    max_attempts,
                )
                raise RetryExhausted(last_exc, max_attempts) from last_exc

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    wait = _wait_seconds(attempt, backoff_base, jitter)
                    logger.warning(
                        "retry %s attempt %d/%d failed: %r — backing off %.2fs",
                        fn.__qualname__,
                        attempt,
                        max_attempts,
                        exc,
                        wait,
                    )
                    sleep(wait)
            assert last_exc is not None
            logger.error(
                "retry %s exhausted after %d attempts",
                fn.__qualname__,
                max_attempts,
            )
            raise RetryExhausted(last_exc, max_attempts) from last_exc

        return sync_wrapper  # type: ignore[return-value]

    return decorator