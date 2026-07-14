#!/usr/bin/env python3
"""Exercise the observer MCP through the restricted SSH bridge."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import ImageContent, TextContent


BRIDGE = Path(__file__).with_name("codex-mcp-bridge.sh")
EXPECTED_TOOLS = {
    "observe",
    "propose_next",
    "reset_task",
    "start_task",
    "status",
}


async def smoke(
    screenshot_path: Path | None, *, health_only: bool = False
) -> dict[str, object]:
    parameters = StdioServerParameters(command=str(BRIDGE))
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tool_names = {tool.name for tool in (await session.list_tools()).tools}
            if tool_names != EXPECTED_TOOLS:
                raise RuntimeError(f"Unexpected MCP tools: {sorted(tool_names)}")

            status_result = await session.call_tool("status", {})
            status_payload = status_result.structuredContent
            if not isinstance(status_payload, dict):
                raise RuntimeError("status did not return structured content")

            observe_result = await session.call_tool("observe", {})
            metadata: dict[str, object] | None = None
            png: bytes | None = None
            for item in observe_result.content:
                if isinstance(item, TextContent):
                    metadata = json.loads(item.text)
                elif isinstance(item, ImageContent):
                    png = base64.b64decode(item.data, validate=True)
            if metadata is None or png is None:
                raise RuntimeError("observe did not return metadata and a PNG")
            if (metadata.get("width"), metadata.get("height")) != (1920, 1080):
                raise RuntimeError(f"Unexpected display metadata: {metadata}")
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError("observe returned invalid PNG data")
            if screenshot_path is not None:
                screenshot_path.write_bytes(png)

            task_lifecycle = "skipped"
            if not health_only:
                task_result = await session.call_tool(
                    "start_task", {"instruction": "Identify the test page heading."}
                )
                task_payload = task_result.structuredContent
                if not isinstance(task_payload, dict) or "task_id" not in task_payload:
                    raise RuntimeError("start_task did not return a task_id")
                reset_result = await session.call_tool(
                    "reset_task", {"task_id": task_payload["task_id"]}
                )
                if reset_result.isError:
                    raise RuntimeError("reset_task failed")
                task_lifecycle = "pass"

            return {
                "ok": True,
                "tools": sorted(tool_names),
                "status": status_payload,
                "observation": metadata,
                "png_bytes": len(png),
                "task_lifecycle": task_lifecycle,
            }


def _error_messages(error: BaseException) -> list[str]:
    nested = getattr(error, "exceptions", None)
    if nested:
        messages: list[str] = []
        for item in nested:
            messages.extend(_error_messages(item))
        return messages
    message = " ".join(str(error).split())
    return [message or type(error).__name__]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Skip the in-memory start/reset task lifecycle check.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show a full traceback instead of a concise JSON error.",
    )
    args = parser.parse_args()
    try:
        result = asyncio.run(smoke(args.screenshot, health_only=args.health_only))
    except BaseException as error:
        if args.verbose:
            raise
        messages = list(dict.fromkeys(_error_messages(error)))
        print(
            json.dumps(
                {"ok": False, "error": "MCP canary failed", "details": messages},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
