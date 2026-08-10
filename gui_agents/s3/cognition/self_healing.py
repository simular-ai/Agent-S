"""SelfHealingEngine — auto-correção causal via VLM (TIER 4 MÓDULO 3).

Quando uma ação do agente falha, o engine feeda o VLM (GPT-4o / Claude) com:
  - a ação tentada + a exceção/mensagem de erro
  - screenshot ANTES da ação
  - screenshot DEPOIS da ação
...e pede um diagnóstico estruturado: o que mudou entre as telas, causa raiz
provável, e a próxima ação lógica — retornada como JSON consumível pelo Worker.

Fluxo integrado (Worker / cli_app)::

    try:
        exec(code[0])
    except Exception as e:
        shot_after = capture_screenshot()
        heal = SelfHealingEngine(provider="openai")
        result = heal.diagnose(action_desc, str(e), shot_before, shot_after)
        if result.action:
            exec(build_code_from_action(result.action))  # ação corrigida

**Import-guard:** módulo carrega sem openai/anthropic SDK e sem API key; o erro
explícito só aparece ao instanciar/diagnosticar — mesmo padrão de
``DockerExecutor`` e ``VectorMemory``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("desktopenv.agent.cognition")

# ───────────────────────────────────────────────────────────── import guards
try:
    from openai import OpenAI as _OpenAIClient  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    _OpenAIClient = None  # type: ignore
    _HAS_OPENAI = False

try:
    from anthropic import Anthropic as _AnthropicClient  # type: ignore

    _HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover
    _AnthropicClient = None  # type: ignore
    _HAS_ANTHROPIC = False


# ───────────────────────────────────────────────────────────── resultado
@dataclass
class HealingResult:
    """Diagnóstico de auto-correção produzido pelo VLM."""

    action: Dict[str, Any]  # próxima ação sugerida (ex.: type/coords/keys/text)
    root_cause: str  # causa raiz provável da falha
    what_changed: str  # diff visual entre antes/depois
    confidence: float  # 0.0–1.0 (auto-avaliada pelo VLM)
    raw_response: str = ""  # resposta crua p/ debug

    @property
    def has_recovery(self) -> bool:
        return bool(self.action)


# ───────────────────────────────────────────────────────────── prompt
_PROMPT_TEMPLATE = """\
You are a GUI-agent self-healing module. An action just FAILED.

ACTION ATTEMPTED:
{action}

ERROR:
{error}

You are given two screenshots: image 1 = BEFORE the action, image 2 = AFTER.
Compare them carefully and reason about the failure.

Respond as STRICT JSON with this schema:
{{
  "what_changed": "<what visually changed between screenshot 1 and 2, or 'no visible change'>",
  "root_cause": "<most likely cause of the failure>",
  "confidence": <float 0.0-1.0>,
  "action": {{
    "action_type": "click" | "type" | "hotkey" | "scroll" | "wait" | "done" | "fail",
    "coordinates": [x, y],
    "text": "<text to type, if action_type=type>",
    "keys": "<hotkey combo, if action_type=hotkey>",
    "reasoning": "<why this next action should succeed>"
  }}
}}

Rules:
- coordinates are absolute pixel coords in the screenshot space.
- If the failure is unrecoverable, set action_type to "fail".
- Output ONLY the JSON object, no prose."""


# ───────────────────────────────────────────────────────────── engine
class SelfHealingEngine:
    """Diagnóstico causal de falhas via VLM (visão + linguagem).

    Args:
        provider: "openai" (GPT-4o, default) ou "anthropic" (Claude).
        model: nome do modelo (defaults sensatos por provider).
        api_key: chave; se None, lê ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``.
        max_tokens: teto de tokens da resposta.
    """

    DEFAULT_MODEL_OPENAI = os.environ.get(
        "AGENT_S3_HEAL_MODEL_OPENAI", "gpt-4o"
    )
    DEFAULT_MODEL_ANTHROPIC = os.environ.get(
        "AGENT_S3_HEAL_MODEL_ANTHROPIC", "claude-sonnet-5-20251022"
    )

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        vlm_timeout: float = 30.0,
    ) -> None:
        if provider not in ("openai", "anthropic"):
            raise ValueError(
                f"provider inválido: {provider!r} (use openai|anthropic)"
            )
        self.provider = provider
        self.model = model or (
            self.DEFAULT_MODEL_OPENAI
            if provider == "openai"
            else self.DEFAULT_MODEL_ANTHROPIC
        )
        self.max_tokens = max_tokens
        # #7: timeout HTTP p/ chamada VLM — VLM hang não bloqueia step do agente.
        self.vlm_timeout = vlm_timeout
        self._client = self._build_client(provider, api_key)
        logger.info(
            "self_healing_ready",
            extra={"provider": provider, "model": self.model},
        )

    # ─────────────────────────────────────────────────────────── cliente
    def _build_client(self, provider: str, api_key: Optional[str]) -> Any:
        if provider == "openai":
            if not _HAS_OPENAI:
                raise RuntimeError(
                    "openai SDK ausente — instale com: pip install openai"
                )
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY ausente — defina a env ou passe api_key="
                )
            return _OpenAIClient(api_key=key, timeout=self.vlm_timeout)  # type: ignore[misc]
        # anthropic
        if not _HAS_ANTHROPIC:
            raise RuntimeError(
                "anthropic SDK ausente — instale com: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY ausente — defina a env ou passe api_key="
            )
        return _AnthropicClient(api_key=key, timeout=self.vlm_timeout)  # type: ignore[misc]

    # ───────────────────────────────────────────────────────── API pública
    def diagnose(
        self,
        action: str,
        error: str,
        screenshot_before: Optional[bytes] = None,
        screenshot_after: Optional[bytes] = None,
    ) -> HealingResult:
        """Diagnostica a falha e sugere a próxima ação.

        Args:
            action: descrição/código da ação tentada.
            error: mensagem de erro / exceção.
            screenshot_before: PNG bytes da tela antes da ação (ou None).
            screenshot_after: PNG bytes da tela depois da ação (ou None).
        """
        prompt = _PROMPT_TEMPLATE.format(action=action, error=error)
        try:
            raw = (
                self._call_openai(prompt, screenshot_before, screenshot_after)
                if self.provider == "openai"
                else self._call_anthropic(
                    prompt, screenshot_before, screenshot_after
                )
            )
        except Exception as exc:
            logger.error(
                "healing_call_error",
                extra={"error": str(exc), "provider": self.provider},
            )
            return HealingResult(
                action={},
                root_cause=f"VLM call failed: {exc}",
                what_changed="",
                confidence=0.0,
            )
        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning(
                "healing_parse_failed",
                extra={"raw_head": raw[:200]},
            )
            return HealingResult(
                action={},
                root_cause="VLM response not valid JSON",
                what_changed="",
                confidence=0.0,
                raw_response=raw,
            )
        conf = parsed.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        result = HealingResult(
            action=parsed.get("action", {}) or {},
            root_cause=str(parsed.get("root_cause", "")),
            what_changed=str(parsed.get("what_changed", "")),
            confidence=max(0.0, min(1.0, conf)),
            raw_response=raw,
        )
        logger.info(
            "healing_diagnosis",
            extra={
                "action_type": (result.action or {}).get("action_type"),
                "confidence": result.confidence,
                "root_cause": result.root_cause[:80],
            },
        )
        return result

    # ─────────────────────────────────────────────────────── VLM backends
    def _call_openai(
        self, prompt: str, before: Optional[bytes], after: Optional[bytes]
    ) -> str:
        content: list = [{"type": "text", "text": prompt}]
        for img in (before, after):
            if img:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _png_data_uri(img),
                            "detail": "low",
                        },
                    }
                )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(
        self, prompt: str, before: Optional[bytes], after: Optional[bytes]
    ) -> str:
        content: list = [{"type": "text", "text": prompt}]
        for img in (before, after):
            if img:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": _png_b64(img),
                        },
                    }
                )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        # content = list de blocks; juntar text blocks
        return "".join(
            block.text  # type: ignore[attr-defined]
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    # ─────────────────────────────────────────────────────────── parse
    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Extrai JSON object da resposta (tolera ```json fences / prose)."""
        if not raw:
            return None
        # fence ```json ... ```
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        candidate = m.group(1) if m else raw
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            # fallback: primeiro { ... } balanceado
            start = candidate.find("{")
            if start < 0:
                return None
            depth = 0
            for i in range(start, len(candidate)):
                if candidate[i] == "{":
                    depth += 1
                elif candidate[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(candidate[start : i + 1])
                            break
                        except json.JSONDecodeError:
                            return None
            else:
                return None
        if not isinstance(obj, dict):
            return None
        return obj


# ───────────────────────────────────────────────────────────── helpers
def _png_b64(img: bytes) -> str:
    return base64.b64encode(img).decode("ascii")


def _png_data_uri(img: bytes) -> str:
    return f"data:image/png;base64,{_png_b64(img)}"