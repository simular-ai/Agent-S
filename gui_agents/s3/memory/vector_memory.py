"""VectorMemory — memória semântica persistente via ChromaDB (TIER 4 MÓDULO 2).

O agente recorda trajetórias de sucesso passadas e reusa experiência quando
uma tarefa nova é semanticamente parecida (approach tipo case-based reasoning).

- ``save_success_trajectory(task_description, steps_taken)`` — grava uma
  trajetória vencedora como documento embeddado na coleção ``agent_memory``.
- ``query_similar_experience(task_description, top_k=1)`` — recupera as
  trajetórias passadas mais similares à tarefa atual.

Persistência: ChromaDB ``PersistentClient`` em disco (default
``~/Agent-S/data/vector_memory/``) — sobrevive entre sessões, sem servidor.

Embeddings (pluggable):
- ``provider="local"``  (default) → sentence-transformers ``all-MiniLM-L6-v2``
  (roda offline, sem API key). Requer ``pip install sentence-transformers``.
- ``provider="openai"`` → ``text-embedding-3-small``. Requer ``OPENAI_API_KEY``
  + ``pip install openai``.

**Import-guard:** módulo carrega sem chromadb/sentence-transformers; o erro
explícito só aparece ao instanciar — mesmo padrão de ``DockerExecutor`` e
``scheduler.py``.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("desktopenv.agent.memory")

# ───────────────────────────────────────────────────────────── import guards
try:
    import chromadb as _chroma  # type: ignore

    _HAS_CHROMA = True
except ImportError:  # pragma: no cover
    _chroma = None  # type: ignore
    _HAS_CHROMA = False


def _require_chroma() -> Any:
    """Devolve o módulo chromadb OU levanta ``RuntimeError`` explicativo."""
    if not _HAS_CHROMA:
        raise RuntimeError(
            "chromadb ausente — instale com: pip install chromadb"
        )
    return _chroma


@dataclass
class MemoryEntry:
    """Trajetória recuperada da memória semântica."""

    id: str
    task_description: str
    steps_taken: str
    score: float  # similaridade em [0,1] — 1 = idêntico (convertido da distância)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────── embeddings
class _LocalEmbedding:
    """Wrapper sentence-transformers — carrega lazy (modelo ~90MB no 1º uso)."""

    def __init__(self, model_name: str) -> None:
        try:
            from chromadb.utils import embedding_functions  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "chromadb.utils.embedding_functions indisponível — "
                "instale chromadb"
            ) from exc
        self._fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self._name = f"sentence-transformers/{model_name}"

    # ChromaDB exige `name()` callable no embedding function (metadata coleção).
    def name(self) -> str:
        return self._name

    # ChromaDB ≥0.5 chama embed_query/embed_documents (não __call__).
    def embed_query(self, input: str) -> List[float]:
        return self._fn.embed_query(input)  # type: ignore[no-any-return]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._fn.embed_documents(input)  # type: ignore[no-any-return]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._fn(input)  # type: ignore[no-any-return]


class _OpenAIEmbedding:
    """Wrapper OpenAI text-embedding-3-small."""

    def __init__(self, model_name: str, api_key: Optional[str]) -> None:
        try:
            from chromadb.utils import embedding_functions  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "chromadb.utils.embedding_functions indisponível"
            ) from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY ausente — defina a env ou use provider='local'"
            )
        self._fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=key, model_name=model_name
        )
        self._name = f"openai/{model_name}"

    def name(self) -> str:
        return self._name

    def embed_query(self, input: str) -> List[float]:
        return self._fn.embed_query(input)  # type: ignore[no-any-return]

    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self._fn.embed_documents(input)  # type: ignore[no-any-return]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._fn(input)  # type: ignore[no-any-return]


# ───────────────────────────────────────────────────────────── executor
class VectorMemory:
    """Memória semântica persistente (ChromaDB).

    Uso típico::

        mem = VectorMemory()                      # local, offline
        mem.save_success_trajectory(
            "abrir Cubase e importar áudio",
            "1. launch Cubase\\n2. File > Import > Audio",
        )
        hits = mem.query_similar_experience("abrir DAW e carregar áudio")
        if hits:
            print(hits[0].steps_taken)  # reusa trajetória vencedora
    """

    DEFAULT_DIR = os.environ.get(
        "AGENT_S3_VECTOR_DIR",
        str(Path.home() / "Agent-S" / "data" / "vector_memory"),
    )
    DEFAULT_COLLECTION = "agent_memory"
    DEFAULT_LOCAL_MODEL = os.environ.get(
        "AGENT_S3_EMBED_MODEL", "all-MiniLM-L6-v2"
    )
    DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        persist_dir: str = DEFAULT_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        provider: str = "local",
        embed_model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        # #5: provider="remote" (ou env AGENT_S3_CHROMA_URL set) → HttpClient
        # conecta a um servidor ChromaDB (`chroma run`), seguro p/ multi-process.
        # Default: PersistentClient local (single-writer DuckDB, 1 processo).
        chroma_url = os.environ.get("AGENT_S3_CHROMA_URL")
        if provider not in ("local", "openai", "remote"):
            raise ValueError(
                f"provider inválido: {provider!r} (use local|openai|remote)"
            )
        if provider != "remote" and chroma_url:
            provider = "remote"  # auto-promove se env set
        self.provider = provider
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = _require_chroma()
        if provider == "remote":
            if not chroma_url:
                raise RuntimeError(
                    "provider='remote' requer AGENT_S3_CHROMA_URL "
                    "(ex.: http://localhost:8000) — rode `chroma run`"
                )
            self._client = chroma.HttpClient(url=chroma_url)  # type: ignore[union-attr]
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chroma.PersistentClient(path=persist_dir)

        # Embedding function (decidida aqui p/ fail-fast no construtor).
        if provider == "local":
            self._embed = _LocalEmbedding(embed_model or self.DEFAULT_LOCAL_MODEL)
        else:
            self._embed = _OpenAIEmbedding(
                embed_model or self.DEFAULT_OPENAI_MODEL, openai_api_key
            )

        # get_or_create — idempotente entre sessões.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_memory_ready",
            extra={
                "persist_dir": persist_dir,
                "collection": collection_name,
                "provider": provider,
                "model": embed_model
                or (self.DEFAULT_LOCAL_MODEL if provider == "local"
                    else self.DEFAULT_OPENAI_MODEL),
            },
        )

    # ───────────────────────────────────────────────────────── API pública
    def save_success_trajectory(
        self,
        task_description: str,
        steps_taken: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Grava uma trajetória vencedora. Retorna o id gerado.

        Args:
            task_description: descrição da tarefa resolvida.
            steps_taken: passos executados (texto livre / código / plano).
            metadata: dict arbitrário serializável (ex.: platform, model, ts).
        """
        if not task_description.strip():
            raise ValueError("task_description não pode ser vazio")
        entry_id = str(uuid.uuid4())
        # Documento embeddado combina tarefa + passos — o embedding captura
        # ambos, então queries semânticas casam por intenção E solução.
        document = f"TASK: {task_description}\nSTEPS:\n{steps_taken}"
        meta = dict(metadata or {})
        meta.setdefault("task_description", task_description)
        meta.setdefault("steps_taken", steps_taken)
        try:
            self._collection.add(
                ids=[entry_id],
                documents=[document],
                metadatas=[meta],
            )
            logger.info(
                "trajectory_saved",
                extra={"id": entry_id, "task": task_description[:80]},
            )
        except Exception as exc:
            logger.error(
                "trajectory_save_error",
                extra={"error": str(exc), "task": task_description[:80]},
            )
            raise
        return entry_id

    def query_similar_experience(
        self,
        task_description: str,
        top_k: int = 1,
    ) -> List[MemoryEntry]:
        """Recupera as trajetórias passadas mais similares à tarefa atual.

        Retorna lista vazia se a memória estiver vazia ou a query falhar.
        """
        if top_k < 1:
            raise ValueError("top_k deve ser >= 1")
        try:
            count = self._collection.count()
            if count == 0:
                logger.info("memory_empty", extra={"task": task_description[:80]})
                return []
            res = self._collection.query(
                query_texts=[task_description],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error(
                "memory_query_error",
                extra={"error": str(exc), "task": task_description[:80]},
            )
            return []

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        entries: List[MemoryEntry] = []
        for eid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance ∈ [0,2]; similaridade = 1 - dist/2 (clamp [0,1]).
            score = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
            entries.append(
                MemoryEntry(
                    id=str(eid),
                    task_description=str((meta or {}).get(
                        "task_description", "")),
                    steps_taken=str((meta or {}).get("steps_taken", doc)),
                    score=score,
                    metadata=dict(meta or {}),
                )
            )
        logger.info(
            "memory_recall",
            extra={
                "task": task_description[:80],
                "hits": len(entries),
                "top_score": entries[0].score if entries else 0.0,
            },
        )
        return entries

    # ─────────────────────────────────────────────────────────── util
    def count(self) -> int:
        """Número de trajetórias armazenadas."""
        try:
            return int(self._collection.count())
        except Exception:
            return 0


# ───────────────────────────────────────────────────────────── singleton
_VM_INSTANCE: Optional["VectorMemory"] = None
_VM_LOCK = threading.Lock()


def get_vector_memory(**kwargs: Any) -> "VectorMemory":
    """Singleton thread-safe — reusa PersistentClient + modelo de embedding.

    Evita recarregar ``all-MiniLM-L6-v2`` (~90MB) a cada chamada e múltiplos
    PersistentClient contendendo o lock single-writer do DuckDB. Use este em
    vez de ``VectorMemory()`` direto em loops/hot paths.
    """
    global _VM_INSTANCE
    with _VM_LOCK:
        if _VM_INSTANCE is None:
            _VM_INSTANCE = VectorMemory(**kwargs)
        return _VM_INSTANCE