"""Codex MCP server for observation-only Agent S."""

from __future__ import annotations

import base64
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent, ToolAnnotations

from .service import ObserverService


INSTRUCTIONS = (
    "Observation-only Agent S VM. No tool can click, type, scroll, launch apps, or "
    "change the desktop. Call start_task before propose_next. propose_next sends the "
    "current VM screenshot to configured OpenAI and Hugging Face endpoints and returns "
    "one reviewable proposal. Never claim a proposal was executed."
)

mcp = FastMCP("agent-s-observer", instructions=INSTRUCTIONS, log_level="WARNING")
service = ObserverService()


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def status() -> dict[str, object]:
    """Return observer configuration and active-task health without calling a model."""

    return service.status()


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def observe() -> list[TextContent | ImageContent]:
    """Capture the isolated VM's single monitor and return metadata plus PNG content."""

    observation = service.observe()
    return [
        TextContent(
            type="text", text=json.dumps(observation.metadata(), sort_keys=True)
        ),
        ImageContent(
            type="image",
            data=base64.b64encode(observation.png).decode("ascii"),
            mimeType="image/png",
        ),
    ]


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    )
)
def start_task(instruction: str) -> dict[str, object]:
    """Start one in-memory observation task; this does not call a model or change the UI."""

    return service.start_task(instruction)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
)
def propose_next(task_id: str) -> dict[str, object]:
    """Call hosted models and return one typed proposal without executing it."""

    return service.propose_next(task_id)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
)
def reset_task(task_id: str) -> dict[str, object]:
    """Discard an in-memory observer task without changing the desktop."""

    return service.reset_task(task_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
