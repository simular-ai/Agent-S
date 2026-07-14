"""Hosted-model planner that returns proposals and has no action executor."""

from __future__ import annotations

import base64
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from openai import OpenAI

from .actions import ActionCall, ActionProposal, extract_action_code, parse_action_call
from .capture import CapturedObservation


SYSTEM_PROMPT = """You are Agent S3 running in OBSERVATION-ONLY mode inside an isolated test VM.
You inspect screenshots and propose exactly one next action, but no action will be executed.
Never request passwords, sudo, shell commands, code execution, package installation, filesystem
access, app launching, purchases, deletion, account changes, or authentication approval.

Available proposal methods:
  agent.click(element_description: str, num_clicks: int = 1, button_type: str = "left", hold_keys: list = [])
  agent.type(element_description: str | None = None, text: str = "", overwrite: bool = False, enter: bool = False)
  agent.scroll(element_description: str, clicks: int, shift: bool = False)
  agent.hotkey(keys: list[str])
  agent.wait(time: float)
  agent.done()
  agent.fail()

Respond with three short sections named (Screenshot Analysis), (Next Action), and
(Grounded Action). The Grounded Action must contain exactly one python fenced code block
and exactly one agent method call using literal arguments. Do not include other code.
"""

_COORDINATE_PATTERNS = (
    re.compile(r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>"),
    re.compile(r"\((\d+),\s*(\d+)\)"),
)


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int) -> str:
        """Return one assistant message."""


class OpenAIChatClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 90.0,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned an empty response")
        return content


def _image_url(png: bytes) -> dict[str, Any]:
    encoded = base64.b64encode(png).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
    }


def _summary(response: str) -> str:
    marker = "(Next Action)"
    if marker in response:
        value = response.split(marker, 1)[1].split("(Grounded Action)", 1)[0].strip()
    else:
        value = "Model produced a validated action proposal."
    return " ".join(value.split())[:512]


def parse_grounding_coordinates(
    response: str, width: int, height: int
) -> tuple[int, int]:
    for pattern in _COORDINATE_PATTERNS:
        match = pattern.search(response)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            if 0 <= x < width and 0 <= y < height:
                return x, y
            raise ValueError(f"Grounding coordinates out of bounds: ({x}, {y})")
    raise ValueError("Grounding model did not return one coordinate pair")


def _target_description(action: ActionCall) -> str | None:
    if action.kind in {"click", "scroll"}:
        return action.arguments["element_description"]
    if action.kind == "type":
        return action.arguments["element_description"]
    return None


@dataclass
class TaskHistory:
    task_id: str
    instruction: str
    responses: list[str] = field(default_factory=list)
    screenshots: list[bytes] = field(default_factory=list)


class AgentSObserverPlanner:
    """Stateful Agent S-style planner limited to five non-executing proposals."""

    def __init__(
        self,
        main_client: ChatClient,
        grounding_client: ChatClient,
        *,
        max_steps: int = 5,
        max_trajectory: int = 4,
    ):
        self.main_client = main_client
        self.grounding_client = grounding_client
        self.max_steps = max_steps
        self.max_trajectory = max_trajectory

    def propose(
        self, history: TaskHistory, observation: CapturedObservation
    ) -> ActionProposal:
        if len(history.responses) >= self.max_steps:
            raise RuntimeError(f"Task reached the {self.max_steps}-proposal limit")
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]
        start = max(0, len(history.responses) - self.max_trajectory)
        for screenshot, response in zip(
            history.screenshots[start:], history.responses[start:]
        ):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Task: {history.instruction}"},
                        _image_url(screenshot),
                    ],
                }
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Task: {history.instruction}\n"
                            "Inspect the current screenshot and propose one next action."
                        ),
                    },
                    _image_url(observation.png),
                ],
            }
        )
        response = self.main_client.complete(messages, max_tokens=1200)
        action = parse_action_call(extract_action_code(response))
        target = None
        description = _target_description(action)
        if description:
            grounding_messages = [
                {
                    "role": "user",
                    "content": [
                        _image_url(observation.png),
                        {
                            "type": "text",
                            "text": (
                                f"Query:{description}\n"
                                "Output only the coordinate of one point in your response."
                            ),
                        },
                    ],
                }
            ]
            grounding_response = self.grounding_client.complete(
                grounding_messages, max_tokens=80
            )
            x, y = parse_grounding_coordinates(
                grounding_response, observation.width, observation.height
            )
            target = {"x": x, "y": y}
        history.responses.append(response)
        history.screenshots.append(observation.png)
        return ActionProposal(
            proposal_id=str(uuid.uuid4()),
            task_id=history.task_id,
            step=len(history.responses),
            action=action,
            summary=_summary(response),
            screenshot_sha256=observation.sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
            target=target,
        )
