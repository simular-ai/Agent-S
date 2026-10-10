# gui_agents/s3/orchestration/fallback.py
"""FallbackManager — tenta estratégias alternativas quando a primária falha.

Caso de uso (Worker Agent): se click por coordenada falhar N vezes, cair p/
hotkey; se hotkey falhar, cair p/ descrição textual. Estratégias = lista
ordenada de prioridade.

Correções sobre o draft:
- Estratégias nomeadas (lambdas são todas ``<lambda>`` — colidiam).
- Sucesso = ausência de exceção (não ``result is not False``, que tratava
  ``None``/``0``/``""`` como sucesso).
- Logging JSON (FASE 1), não f-string.
- Suporte sync + async.
- Integra observability (``track_action``) opcional.
- Devolve qual estratégia venceu + resultado.
- Contador de falhas resetável.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from gui_agents.s3.logging_utils.structured_logger import get_logger

logger = get_logger("desktopenv.agent.fallback")

# Estratégia = (nome, fn). fn recebe context, devolve Any (ou levanta p/ falhar).
StrategyFn = Callable[[Any], Any]


@dataclass
class Strategy:
    name: str
    fn: StrategyFn

    @property
    def is_async(self) -> bool:
        return asyncio.iscoroutinefunction(self.fn)


@dataclass
class FallbackResult:
    strategy: str
    result: Any
    attempts: dict[str, int] = field(default_factory=dict)


class FallbackManager:
    """Tenta estratégias em ordem, com N tentativas cada uma.

    Contrato de retry (importante p/ evitar explosão de tentativas):
    - Este loop = retry de ESTRATÉGIA (semântica "Estratégia A falhou 2x →
      ir pra B"). Mantém ``max_retries_per_strategy`` conforme spec original.
    - ``@retry_with_backoff`` (FASE 1) = retry de TRANSIENTE (rede/API/HTTP)
      num nível mais baixo (ex: a chamada LLM dentro da estratégia).
    - NÃO decore a MESMA fn com ``@retry_with_backoff`` E use-a aqui com
      ``max_retries>1`` — multiplica tentativas (max_attempts × max_retries)
      e explode latência p/ falhas persistentes. Regra: retry de transiente
      dentro da estratégia; retry de abordagem aqui. Camadas distintas OK.
    """

    def __init__(
        self,
        max_retries_per_strategy: int = 2,
        *,
        track: bool = False,
    ) -> None:
        self.max_retries = max_retries_per_strategy
        self.track = track  # se True, chama observability.track_action
        self.strategy_failures: dict[str, int] = {}

    def reset(self) -> None:
        self.strategy_failures = {}

    # estratégias podem ser Strategy, (name, fn), ou fn (nome = fn.__name__)
    @staticmethod
    def _normalize(strategies) -> list[Strategy]:
        out: list[Strategy] = []
        for s in strategies:
            if isinstance(s, Strategy):
                out.append(s)
            elif isinstance(s, tuple) and len(s) == 2:
                out.append(Strategy(name=s[0], fn=s[1]))
            else:
                out.append(Strategy(name=getattr(s, "__name__", "strategy"), fn=s))
        return out

    def _track(self, action_type: str, status: str) -> None:
        if not self.track:
            return
        try:
            from gui_agents.s3.observability.metrics import track_action
            track_action(action_type, status)
        except Exception:  # noqa: BLE001 — observability nunca derruba fluxo
            pass

    # ------------------------------------------------------------- sync
    def execute_with_fallback(self, strategies, context: Any = None) -> FallbackResult:
        plans = self._normalize(strategies)
        attempts_log: dict[str, int] = {}
        for idx, strat in enumerate(plans):
            attempts = 0
            while attempts < self.max_retries:
                attempts += 1
                attempts_log[strat.name] = attempts
                logger.info(
                    "fallback_attempt",
                    extra={"strategy": strat.name, "attempt": attempts,
                           "index": idx},
                )
                try:
                    result = strat.fn(context)
                    logger.info(
                        "fallback_success",
                        extra={"strategy": strat.name, "attempt": attempts},
                    )
                    self._track(strat.name, "ok")
                    return FallbackResult(strat.name, result, attempts_log)
                except Exception as exc:  # noqa: BLE001 — falha controlada
                    logger.warning(
                        "fallback_failed",
                        extra={"strategy": strat.name, "attempt": attempts,
                               "error": str(exc)},
                    )
                    self.strategy_failures[strat.name] = (
                        self.strategy_failures.get(strat.name, 0) + 1
                    )
            self._track(strat.name, "fail")
            logger.error(
                "fallback_strategy_exhausted",
                extra={"strategy": strat.name, "max": self.max_retries},
            )
        raise RuntimeError("Todas as estratégias de fallback falharam.")

    # ------------------------------------------------------------- async
    async def execute_with_fallback_async(
        self, strategies, context: Any = None
    ) -> FallbackResult:
        plans = self._normalize(strategies)
        attempts_log: dict[str, int] = {}
        for idx, strat in enumerate(plans):
            attempts = 0
            while attempts < self.max_retries:
                attempts += 1
                attempts_log[strat.name] = attempts
                logger.info(
                    "fallback_attempt_async",
                    extra={"strategy": strat.name, "attempt": attempts,
                           "index": idx},
                )
                try:
                    if strat.is_async:
                        result = await strat.fn(context)
                    else:
                        result = strat.fn(context)
                    logger.info(
                        "fallback_success_async",
                        extra={"strategy": strat.name, "attempt": attempts},
                    )
                    self._track(strat.name, "ok")
                    return FallbackResult(strat.name, result, attempts_log)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "fallback_failed_async",
                        extra={"strategy": strat.name, "attempt": attempts,
                               "error": str(exc)},
                    )
                    self.strategy_failures[strat.name] = (
                        self.strategy_failures.get(strat.name, 0) + 1
                    )
            self._track(strat.name, "fail")
            logger.error(
                "fallback_strategy_exhausted_async",
                extra={"strategy": strat.name, "max": self.max_retries},
            )
        raise RuntimeError("Todas as estratégias de fallback falharam.")