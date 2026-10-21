"""Shared CLI helpers for Prime Agent Python skills."""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any, Callable

import tyro


async def run_cli(func: Callable[..., Any], prog: str | None = None) -> None:
    """Parse CLI arguments for a skill function and print a non-None result."""
    result = tyro.cli(func, prog=prog)
    if inspect.isawaitable(result):
        result = await result
    if result is not None:
        print(result)


def cli() -> None:
    """Run `<skill>.run` for a console script named exactly after the skill import."""
    prog = Path(sys.argv[0]).stem
    try:
        module = __import__(prog)
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import Python skill module {prog!r}. "
            "The console-script name must match the skill import name exactly; "
            "use underscores instead of dashes."
        ) from exc
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError(f"{prog} does not expose a callable run()")
    asyncio.run(run_cli(run, prog=prog))
