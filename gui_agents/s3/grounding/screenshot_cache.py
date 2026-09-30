# gui_agents/s3/grounding/screenshot_cache.py
"""ScreenshotCache — cache por hash p/ não reprocessar a mesma tela.

Quando o grounding analisa um screenshot (set-of-marks, OCR, elements p/ o
LLM), reprocessar a mesma tela é caro e redundante. Hash estável (sha256 dos
bytes PNG) como chave → se a tela não mudou, devolve o resultado cacheado
(em vez de chamar LLM/OCR de novo).

- Backend disk: ``~/Agent-S/data/screenshots_cache/{hash}.json`` (+ .png opc).
- LRU: evict pelo entry mais antigo quando count > max_entries.
- TTL: entries mais velhos que ttl_seconds expirados no acesso.
- Thread-safe (lock). Suporta bytes PNG ou PIL.Image.
- ``get_or_compute(hash, fn)`` — retorna cache ou chama fn, armazena, retorna.
- Métricas: cache_hit/cache_miss via observability (opcional, ``track=True``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from gui_agents.s3.logging_utils.structured_logger import get_logger

logger = get_logger("desktopenv.agent.cache")

DEFAULT_DIR = os.path.expanduser("~/Agent-S/data/screenshots_cache")

# PIL é opcional — só preciso p/ converter Image em bytes PNG estáveis.
try:
    from PIL import Image as _PILImage  # type: ignore
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


def _to_png_bytes(image: Union[bytes, "object"]) -> bytes:
    """Aceita bytes PNG brutos ou PIL.Image → bytes PNG estáveis p/ hash."""
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    if _HAS_PIL and isinstance(image, _PILImage.Image):
        import io
        buf = io.BytesIO()
        # RGB garante estabilidade (RGBA muda se alpha diferir).
        rgb = image.convert("RGB") if image.mode != "RGB" else image
        rgb.save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(
        f"snapshot deve ser bytes PNG ou PIL.Image, got {type(image)!r}"
    )


def hash_screenshot(image: Union[bytes, "object"]) -> str:
    """sha256 hex dos bytes PNG da tela."""
    return hashlib.sha256(_to_png_bytes(image)).hexdigest()


class ScreenshotCache:
    """Cache disk de resultados de grounding por hash de screenshot."""

    def __init__(
        self,
        cache_dir: str = DEFAULT_DIR,
        *,
        max_entries: int = 500,
        ttl_seconds: Optional[float] = 3600.0,
        track: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.track = track
        self._lock = threading.Lock()

    # --------------------------------------------------------------- helpers
    def _path(self, key: str, suffix: str = ".json") -> Path:
        return self.cache_dir / f"{key}{suffix}"

    def _entry_path(self, key: str) -> Path:
        return self._path(key, ".json")

    def _png_path(self, key: str) -> Path:
        return self._path(key, ".png")

    def _expired(self, path: Path) -> bool:
        if self.ttl_seconds is None:
            return False
        age = time.time() - path.stat().st_mtime
        return age > self.ttl_seconds

    def _track_metric(self, name: str) -> None:
        if not self.track:
            return
        try:
            from gui_agents.s3.observability.metrics import ACTIONS_TOTAL
            ACTIONS_TOTAL.labels(type="screenshot_cache", status=name).inc()
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------- get
    def get(self, key: str) -> Optional[Any]:
        """Retorna resultado cacheado se existir e não expirado, senão None."""
        path = self._entry_path(key)
        with self._lock:
            if not path.exists() or self._expired(path):
                if path.exists():
                    # lazy eviction do expirado
                    path.unlink(missing_ok=True)
                    self._png_path(key).unlink(missing_ok=True)
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # touch mtime = LRU mais recente
                path.touch()
                self._track_metric("hit")
                logger.info("cache_hit", extra={"key": key[:12]})
                return data.get("result")
            except (OSError, ValueError) as exc:
                logger.warning("cache_corrupt", extra={"key": key[:12], "error": str(exc)})
                path.unlink(missing_ok=True)
                return None

    # ----------------------------------------------------------------- put
    def put(
        self,
        key: str,
        result: Any,
        *,
        screenshot: Optional[Union[bytes, "object"]] = None,
    ) -> None:
        """Armazena resultado (e opcionalmente o PNG p/ debug). Evict LRU se cheio."""
        with self._lock:
            # Eviction por LRU antes de escrever.
            self._evict_if_needed()
            payload = {
                "result": result,
                "stored_at": time.time(),
                "key": key,
            }
            self._entry_path(key).write_text(
                json.dumps(payload, default=str), encoding="utf-8"
            )
            if screenshot is not None:
                try:
                    self._png_path(key).write_bytes(_to_png_bytes(screenshot))
                except Exception as exc:  # noqa: BLE001 — PNG é opcional
                    logger.warning("cache_png_skip", extra={"error": str(exc)})
            self._track_metric("miss")
            logger.info("cache_put", extra={"key": key[:12]})

    # ------------------------------------------------------- get_or_compute
    def get_or_compute(
        self,
        key_or_image: Union[str, bytes, "object"],
        compute: Callable[[], Any],
        *,
        screenshot: Optional[Union[bytes, "object"]] = None,
    ) -> Any:
        """Conveniência: devolve cache ou chama ``compute``, armazena e devolve.

        ``key_or_image`` pode ser hash (str) já calculado ou a imagem (bytes/
        PIL) — neste caso calcula o hash aqui.
        """
        key = (
            key_or_image
            if isinstance(key_or_image, str)
            else hash_screenshot(key_or_image)
        )
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute()
        self.put(key, result, screenshot=screenshot or (
            key_or_image if not isinstance(key_or_image, str) else None
        ))
        return result

    # -------------------------------------------------------------- eviction
    def _evict_if_needed(self) -> None:
        entries = list(self.cache_dir.glob("*.json"))
        if len(entries) < self.max_entries:
            return
        # Remove expirados primeiro.
        for p in entries:
            if self._expired(p):
                self._remove_entry(p)
        # Ainda cheio? Remove os mais antigos (LRU) até caber.
        entries = sorted(
            self.cache_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(entries) - (self.max_entries - 1)  # -1: vai entrar 1 novo
        for p in entries[:max(0, excess)]:
            self._remove_entry(p)

    def _remove_entry(self, json_path: Path) -> None:
        json_path.unlink(missing_ok=True)
        png = json_path.with_suffix(".png")
        png.unlink(missing_ok=True)

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self.cache_dir.glob("*.json"))
            pngs = list(self.cache_dir.glob("*.png"))
            total_bytes = sum(p.stat().st_size for p in entries + pngs if p.exists())
        return {
            "entries": len(entries),
            "pngs": len(pngs),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "bytes": total_bytes,
        }

    def clear(self) -> int:
        """Remove tudo. Devolve qtd de entries removidos."""
        with self._lock:
            entries = list(self.cache_dir.glob("*.json"))
            for p in entries:
                self._remove_entry(p)
            return len(entries)