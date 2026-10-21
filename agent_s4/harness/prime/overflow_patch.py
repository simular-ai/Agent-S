"""Apply the OSWorld compatibility overlay to Prime's overflow detector.

CoreWeave's vLLM backend reports context overflow as "decoder prompt ... is
longer than the maximum model length".  The pinned Prime release does not
recognize that wording, so it treats the response as a generic retryable error
instead of invoking its native compact-and-retry recovery.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

OVERFLOW_MODULE = (
    "/opt/prime-agent/node_modules/@earendil-works/pi-ai/dist/utils/overflow.js"
)
ANCHOR = "    /maximum context length is \\d+ tokens/i, // OpenRouter (all backends)"
COREWEAVE_PATTERN = (
    "    /decoder prompt.*longer than the maximum model length/i, " "// CoreWeave/vLLM"
)


def patch_overflow_module(path: Path) -> bool:
    """Patch one pinned module, returning True only when it changed."""
    source = path.read_text(encoding="utf-8")
    if COREWEAVE_PATTERN in source:
        return False
    if source.count(ANCHOR) != 1:
        raise RuntimeError(
            "Prime overflow module does not match the pinned runtime; "
            "refusing an unsafe compatibility patch"
        )
    patched = source.replace(ANCHOR, f"{ANCHOR}\n{COREWEAVE_PATTERN}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary.write(patched)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(path.stat().st_mode)
    os.replace(temporary_path, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=OVERFLOW_MODULE)
    args = parser.parse_args()
    patch_overflow_module(Path(args.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
