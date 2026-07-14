"""In-memory task service for the observation-only MCP surface."""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Callable

from .capture import CapturedObservation, MSSCapture
from .planner import AgentSObserverPlanner, OpenAIChatClient, TaskHistory


DEFAULT_MAIN_MODEL = "gpt-5-2025-08-07"
DEFAULT_BUILD_METADATA_PATH = "/opt/agent-s/observer-build.json"


def _build_identity() -> dict[str, object]:
    path = Path(os.environ.get("AGENT_S_BUILD_METADATA", DEFAULT_BUILD_METADATA_PATH))
    fallback: dict[str, object] = {
        "status": "development",
        "source_commit": "unknown",
        "source_dirty": None,
        "built_at": None,
        "source_archive_sha256": None,
        "requirements_lock_sha256": None,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return {
        "status": "sealed",
        "source_commit": payload.get("source_commit", "unknown"),
        "source_dirty": payload.get("source_dirty"),
        "built_at": payload.get("built_at"),
        "source_archive_sha256": payload.get("source_archive_sha256"),
        "requirements_lock_sha256": payload.get("requirements_lock_sha256"),
    }


class ObserverService:
    def __init__(
        self,
        *,
        capture: MSSCapture | None = None,
        planner_factory: Callable[[], AgentSObserverPlanner] | None = None,
    ):
        self.capture_backend = capture or MSSCapture()
        self.planner_factory = planner_factory or self._planner_from_environment
        self._planner: AgentSObserverPlanner | None = None
        self._active_task: TaskHistory | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _planner_from_environment() -> AgentSObserverPlanner:
        openai_key = os.environ.get("OPENAI_API_KEY")
        hf_token = os.environ.get("HF_TOKEN")
        hf_url = os.environ.get("HF_ENDPOINT_URL")
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", openai_key),
                ("HF_TOKEN", hf_token),
                ("HF_ENDPOINT_URL", hf_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing observer configuration: {', '.join(missing)}")
        main_client = OpenAIChatClient(
            model=os.environ.get("AGENT_S_MAIN_MODEL", DEFAULT_MAIN_MODEL),
            api_key=openai_key,
        )
        grounding_client = OpenAIChatClient(
            model=os.environ.get("AGENT_S_GROUND_MODEL", "tgi"),
            api_key=hf_token,
            base_url=hf_url,
        )
        return AgentSObserverPlanner(main_client, grounding_client)

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": "observation_only",
                "desktop_actions_exposed": False,
                "display": os.environ.get("DISPLAY", ""),
                "expected_resolution": {"width": 1920, "height": 1080},
                "main_model": os.environ.get("AGENT_S_MAIN_MODEL", DEFAULT_MAIN_MODEL),
                "openai_key_configured": bool(os.environ.get("OPENAI_API_KEY")),
                "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
                "hf_endpoint_configured": bool(os.environ.get("HF_ENDPOINT_URL")),
                "build": _build_identity(),
                "active_task_id": (
                    self._active_task.task_id if self._active_task else None
                ),
                "proposal_count": (
                    len(self._active_task.responses) if self._active_task else 0
                ),
            }

    def observe(self) -> CapturedObservation:
        return self.capture_backend.capture()

    def start_task(self, instruction: str) -> dict[str, object]:
        instruction = " ".join(instruction.split())
        if not 1 <= len(instruction) <= 2000:
            raise ValueError("instruction must contain 1-2000 characters")
        with self._lock:
            task_id = str(uuid.uuid4())
            self._active_task = TaskHistory(task_id=task_id, instruction=instruction)
            self._planner = None
            return {"task_id": task_id, "mode": "observation_only", "proposal_limit": 5}

    def propose_next(self, task_id: str) -> dict[str, object]:
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task_id:
                raise ValueError("task_id is not the active observer task")
            if self._planner is None:
                self._planner = self.planner_factory()
            observation = self.capture_backend.capture()
            return self._planner.propose(self._active_task, observation).to_dict()

    def reset_task(self, task_id: str) -> dict[str, object]:
        with self._lock:
            if self._active_task is None or self._active_task.task_id != task_id:
                raise ValueError("task_id is not the active observer task")
            self._active_task = None
            self._planner = None
            return {"reset": True, "task_id": task_id}
