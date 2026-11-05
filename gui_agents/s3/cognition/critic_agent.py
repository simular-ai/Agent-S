"""CriticAgent — o "3º cérebro": Planejador & Revisor (TIER 5).

Dois papéis num agente só, inspirado no padrão o1 (planejar → executar →
criticar) e no self-consistency de cadeias longas:

1. **Research / Planning** — antes de gerar código, pesquisa documentação
   na web (DuckDuckGo via ``duckduckgo_search``; Tavily como fallback
   opcional se ``AGENT_S3_TAVILY_KEY`` estiver setada). Só pesquisa quando
   a tarefa menciona algo específico (biblioteca, versão, "how to") —
   gate heurístico evita buscas inúteis em tarefas genéricas.

2. **Pre-Execution Review** — recebe o script Python/Bash gerado pelo
   Executor (LLM) e analisa procurando bugs lógicos, caminhos de arquivo
   hardcoded, imports faltantes e uso de APIs inexistentes. Devolve o
   código corrigido + a lista de issues. Roda ANTES do DockerExecutor,
   então um bug pegue aqui economiza um ciclo de sandbox inteiro.

Arquitetura (espelha :mod:`self_healing`):
- provider pluggable (``openai`` | ``anthropic``), mesmo padrão de
  import-guard e fail-fast no construtor.
- ``_parse_json`` reusa ``json.JSONDecoder().raw_decode`` (H3 — brace-counter
  quebra com ``}`` dentro de strings; raw_decode é um parser JSON real).
- Chamadas web/LLM envolvidas em try/except: falha de API nunca derruba o
  ciclo do agente — research vira "" e review devolve o código original.

**Import-guard:** módulo carrega sem openai/anthropic/duckduckgo_search;
o erro explícito só aparece ao instanciar. Mesmo padrão de
``DockerExecutor``, ``VectorMemory`` e ``SelfHealingEngine``.

Uso típico::

    critic = CriticAgent(provider="openai")
    docs = critic.research("parse PDF with pdfplumber 0.11")
    reviewed = critic.review_code(script, language="python",
                                  task="parse PDF", docs_context=docs)
    if reviewed.changed:
        run_in_sandbox(reviewed.corrected_code)
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("desktopenv.agent.cognition")

# ───────────────────────────────────────────────────────────── import guards
try:
    from openai import OpenAI as _OpenAIClient  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover — ambiente sem SDK
    _OpenAIClient = None  # type: ignore
    _HAS_OPENAI = False

try:
    from anthropic import Anthropic as _AnthropicClient  # type: ignore

    _HAS_ANTHROPIC = True
except ImportError:  # pragma: no cover
    _AnthropicClient = None  # type: ignore
    _HAS_ANTHROPIC = False

try:
    from duckduckgo_search import DDGS as _DDGS  # type: ignore

    _HAS_DDGS = True
except ImportError:  # pragma: no cover — dep opcional
    _DDGS = None  # type: ignore
    _HAS_DDGS = False


# ───────────────────────────────────────────────────────────── resultados
@dataclass
class SearchResult:
    """Resultado individual de busca web."""

    title: str
    url: str
    snippet: str


@dataclass
class ReviewResult:
    """Saída da revisão pré-execução de código.

    ``corrected_code`` é sempre seguro de rodar: se a LLM não sugeriu
    mudanças (ou se a chamada falhou), devolve o código original ipsis.
    """

    corrected_code: str
    issues: List[str] = field(default_factory=list)
    changed: bool = False
    raw_response: str = ""


# ───────────────────────────────────────────────────────────── prompts
_RESEARCH_GATE_KEYWORDS = (
    "library",
    "libraries",
    "package",
    "pip install",
    "npm install",
    "version",
    "docs",
    "documentation",
    "how to",
    "how do i",
    "api",
    "sdk",
    "framework",
    "module",
    "import ",
    "using ",
    "upgrade",
    "migrate",
)

# Padrão "palavra-versão" (ex.: pdfplumber 0.11, Python 3.12, React 18).
# Dot-group opcional: aceita major-int (React 18) e dotted (Python 3.12).
# Grupo 1 = nome da lib (mantido no sub); só o número da versão é removido.
# Gate de research prefere recall — falso-positivo só dispara uma busca vazia
# (barato); falso-negativo perde docs relevantes (caro).
_VERSION_RE = re.compile(r"(\b[a-z][a-z0-9_.-]+)\s+v?\d+(\.\d+)*\b", re.IGNORECASE)

_REVIEW_PROMPT = """\
You are a senior code reviewer doing a PRE-EXECUTION review of a script
that will run inside a sandboxed executor. The script has NOT run yet —
your job is to catch problems before it does, saving a failed run.

Analyze the {language} script below for, in priority order:
1. Missing imports (used but not imported) or wrong import names.
2. Hardcoded absolute file paths that won't exist in the sandbox
   (e.g. /Users/.../file.mp3, C:\\\\...). Flag them and replace with a
   relative path or a placeholder variable the caller must fill.
3. Logic bugs: wrong operator, off-by-one, undefined variable, wrong
   function signature, missing await, unreachable code.
4. Calls to APIs/functions that don't exist on the stated library version.
5. Silent failures (bare except, swallowed errors) that would hide a crash.

Task the script is meant to solve: {task}
{docs_block}
Return ONLY a JSON object (no prose, no markdown fences) with this shape:
{{
  "issues": ["short description of each problem found, or [] if clean"],
  "corrected_code": "the full corrected script. If no changes needed,
                      repeat the original script verbatim here.",
  "changed": true or false
}}

SCRIPT:
```{language}
{code}
```
"""

_CODEGEN_SYSTEM = (
    "You are an expert code generator. Given a task and supporting context, "
    "write a single self-contained, runnable {language} script that solves it. "
    "Output ONLY the script code — no markdown fences, no explanation. Use only "
    "stdlib + commonly-available libraries. Prefer relative paths. Fail loudly "
    "(raise on error) rather than silently."
)


# ───────────────────────────────────────────────────────────── agente
class CriticAgent:
    """Planejador & Revisor cognitivo (o "3º cérebro").

    Args:
        provider: ``"openai"`` (default) ou ``"anthropic"``.
        model: nome do modelo (defaults sensatos por provider).
        api_key: chave; se None, lê ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``.
        max_tokens: teto de tokens da resposta da LLM.
        timeout: timeout HTTP da chamada LLM (não derruba o ciclo se hang).
        search_max_results: nº de resultados de web search por query.
    """

    DEFAULT_MODEL_OPENAI = os.environ.get(
        "AGENT_S3_CRITIC_MODEL_OPENAI", "gpt-4o"
    )
    DEFAULT_MODEL_ANTHROPIC = os.environ.get(
        "AGENT_S3_CRITIC_MODEL_ANTHROPIC", "claude-sonnet-5-20251022"
    )

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        search_max_results: int = 5,
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
        self.timeout = timeout
        self.search_max_results = search_max_results
        self._client = self._build_client(provider, api_key)
        logger.info(
            "critic_agent_ready",
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
            return _OpenAIClient(api_key=key, timeout=self.timeout)  # type: ignore[misc]
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
        return _AnthropicClient(api_key=key, timeout=self.timeout)  # type: ignore[misc]

    # ───────────────────────────────────────────────────────── API pública
    def research(self, task_description: str, max_results: Optional[int] = None) -> str:
        """Pesquisa documentação na web quando a tarefa pede algo específico.

        Gate heurístico: só busca se a tarefa menciona biblioteca/versão/
        "how to"/docs. Caso contrário retorna "" (sem desperdiçar calls).

        Retorna um bloco de texto formatado com títulos + snippets p/ o
        prompt de geração/revisão. Em qualquer falha de API, retorna "" —
        nunca derruba o ciclo.
        """
        if not task_description or not task_description.strip():
            return ""
        if not self._needs_research(task_description):
            logger.info("critic_research_skipped", extra={"reason": "no_specific_topic"})
            return ""
        n = max_results or self.search_max_results
        query = self._build_query(task_description)
        results = self._web_search(query, n)
        if not results:
            logger.info("critic_research_empty", extra={"query": query[:80]})
            return ""
        logger.info(
            "critic_research_done",
            extra={"query": query[:80], "hits": len(results)},
        )
        lines = ["Relevant docs found by the CriticAgent (use as needed):"]
        for r in results:
            lines.append(f"- {r.title}\n  {r.url}\n  {r.snippet}")
        return "\n".join(lines)

    def review_code(
        self,
        code: str,
        language: str = "python",
        task: str = "",
        docs_context: str = "",
    ) -> ReviewResult:
        """Revisão pré-execução do script gerado pelo Executor.

        Devolve :class:`ReviewResult`. Se a LLM falhar ou retornar JSON
        inválido, devolve o código original inalterado (``changed=False``)
        — o fluxo nunca trava por causa do crítico.
        """
        if not code or not code.strip():
            return ReviewResult(corrected_code=code)
        docs_block = (
            f"Relevant documentation:\n{docs_context}\n" if docs_context else ""
        )
        prompt = _REVIEW_PROMPT.format(
            language=language,
            task=task or "(not specified)",
            docs_block=docs_block,
            code=code,
        )
        try:
            raw = self._complete(prompt, system=None)
        except Exception as exc:
            logger.error(
                "critic_review_call_error",
                extra={"error": str(exc), "provider": self.provider},
            )
            return ReviewResult(corrected_code=code, issues=[f"review LLM failed: {exc}"])
        parsed = self._parse_json(raw)
        if parsed is None:
            # Fallback: a LLM pode ter devolvido só o código sem JSON wrapper.
            stripped = self._strip_fences(raw)
            if stripped and stripped.strip() != code.strip():
                logger.info("critic_review_raw_code_fallback", extra={})
                return ReviewResult(
                    corrected_code=stripped,
                    issues=["review returned raw code instead of JSON"],
                    changed=True,
                    raw_response=raw,
                )
            logger.warning("critic_review_parse_failed", extra={"raw_head": raw[:200]})
            return ReviewResult(corrected_code=code, raw_response=raw)
        corrected = str(parsed.get("corrected_code", "")).strip()
        issues = parsed.get("issues", []) or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        else:
            issues = [str(i) for i in issues]
        changed = bool(parsed.get("changed", False)) or corrected != code.strip()
        # Se a LLM disse que corrigiu mas devolveu vazio, não descarta o código.
        if not corrected:
            corrected = code
            changed = False
        result = ReviewResult(
            corrected_code=corrected,
            issues=issues,
            changed=changed,
            raw_response=raw,
        )
        logger.info(
            "critic_review_done",
            extra={"changed": changed, "issues": len(issues)},
        )
        return result

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        """Chamada LLM texto genérica (pública — usada pelo Executor no CLI).

        Não usa vision (o crítico trabalha com texto/código, não screenshots).
        Envolve timeout + try/except no caller; aqui só encapsula o backend.
        """
        return self._complete(prompt, system)

    # ─────────────────────────────────────────────────────── LLM backends
    def _complete(self, prompt: str, system: Optional[str]) -> str:
        if self.provider == "openai":
            messages: list = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content or ""
        # anthropic
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(
            block.text  # type: ignore[attr-defined]
            for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    # ─────────────────────────────────────────────────────── web search
    def _web_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Busca web: DuckDuckGo (default) → Tavily (fallback se key setada).

        Ambos os backends são envolvidos em try/except — falha de API
        retorna [] (caller decide se prossegue sem docs).
        """
        results = self._ddgs_search(query, max_results)
        if results:
            return results
        tavily_key = os.environ.get("AGENT_S3_TAVILY_KEY")
        if tavily_key:
            results = self._tavily_search(query, max_results, tavily_key)
        return results

    def _ddgs_search(self, query: str, max_results: int) -> List[SearchResult]:
        if not _HAS_DDGS:
            logger.info("critic_search_ddgs_unavailable", extra={})
            return []
        try:
            out: List[SearchResult] = []
            # v8.x: a forma context-manager (``with DDGS() as d``) devolve
            # sistematicamente 0 resultados — usar a instanciação direta.
            ddgs = _DDGS()  # type: ignore[misc]
            for r in ddgs.text(query, max_results=max_results):
                out.append(
                    SearchResult(
                        title=str(r.get("title", "")),
                        # DDGS usa 'href' ou 'url' conforme a versão.
                        url=str(r.get("href") or r.get("url") or ""),
                        snippet=str(r.get("body") or r.get("snippet") or ""),
                    )
                )
            return out
        except Exception as exc:  # noqa: BLE001 — API web nunca derruba o ciclo
            logger.warning(
                "critic_search_ddgs_failed",
                extra={"error": str(exc), "query": query[:80]},
            )
            return []

    def _tavily_search(
        self, query: str, max_results: int, api_key: str
    ) -> List[SearchResult]:
        """Fallback Tavily (requer AGENT_S3_TAVILY_KEY). Dep: requests/urllib."""
        try:
            import urllib.request  # stdlib — evita dep extra de requests
            import json as _json

            payload = _json.dumps(
                {"api_key": api_key, "query": query, "max_results": max_results}
            ).encode("utf-8")
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                data = _json.loads(resp.read().decode("utf-8"))
            return [
                SearchResult(
                    title=str(r.get("title", "")),
                    url=str(r.get("url", "")),
                    snippet=str(r.get("content", "")),
                )
                for r in data.get("results", [])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "critic_search_tavily_failed",
                extra={"error": str(exc), "query": query[:80]},
            )
            return []

    # ─────────────────────────────────────────────────────── heurísticas
    @staticmethod
    def _needs_research(task: str) -> bool:
        lowered = task.lower()
        if any(kw in lowered for kw in _RESEARCH_GATE_KEYWORDS):
            return True
        if _VERSION_RE.search(task):
            return True
        return False

    @staticmethod
    def _build_query(task: str) -> str:
        # DDGS prefere queries curtas (2-6 termos). Queries longas ou com
        # version numbers ("pdfplumber 0.11 ...") devolvem 0 resultados.
        # Strip: version patterns + gate keywords/filler; pega os termos
        # mais distintivos (cap 6 palavras).
        q = _VERSION_RE.sub(r"\1", task)  # drop "0.11", "3.12" — keep lib name
        q = re.sub(
            r"\b(library|libraries|documentation|docs?|how to|how do i|"
            r"using|use|with|package|pip install|npm install|please|the|a|an)\b",
            " ",
            q,
            flags=re.IGNORECASE,
        )
        words = [w for w in q.split() if len(w) > 1]
        return " ".join(words[:6])

    # ─────────────────────────────────────────────────────────── parse
    @staticmethod
    def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
        """Extrai JSON object da resposta (tolera ```json fences / prose).

        H3 (mesmo fix do self_healing): ``json.JSONDecoder().raw_decode`` em
        vez de brace-counting — o contador quebra com ``}`` dentro de strings
        (ex.: valor ``"click({x:100})"``). raw_decode é um parser JSON real.
        """
        if not raw:
            return None
        decoder = json.JSONDecoder()
        start = raw.find("{")
        while start >= 0:
            try:
                obj, _ = decoder.raw_decode(raw[start:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            start = raw.find("{", start + 1)
        return None

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove ```lang ... ``` fences se a LLM devolveu só código."""
        s = raw.strip()
        if s.startswith("```"):
            # drop primeira linha (```python) e último ```
            lines = s.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines)
        return s