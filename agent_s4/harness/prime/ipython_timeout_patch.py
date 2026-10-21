"""Add a retriable 60-second cell limit to the pinned Prime IPython tool.

The pinned Prime runtime already knows how to interrupt an executing kernel,
but ordinary IPython tool calls do not install a deadline.  This overlay keeps
the episode alive, interrupts only the overlong cell, and returns an explicit
tool error asking the model to retry with smaller work.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

IPYTHON_MODULE = "/opt/prime-agent/node_modules/prime-agent/dist/core/tools/ipython.js"
IPYTHON_DIST_ROOT = "/opt/prime-agent/node_modules/prime-agent/dist"
TIMEOUT_MS = 60_000
TIMEOUT_CONSTANT = f"const IPYTHON_CELL_TIMEOUT_MS = {TIMEOUT_MS};"
TIMEOUT_NOTICE = (
    'const IPYTHON_TIMEOUT_NOTICE = "<ipython_timeout>\\nIPython cell exceeded '
    "60 seconds and was interrupted. Retry with a smaller, bounded cell; split "
    "long work into multiple calls and inspect progress between calls.\\n"
    '</ipython_timeout>";'
)
CONSTANT_ANCHOR = "const KERNEL_RESTART_NOTICE = ["
BUNDLE_CONSTANT_ANCHOR = "var KERNEL_RESTART_NOTICE = ["

EXECUTE_START = (
    "async function executeWithBusyKernelChoice(provisioner, "
    "reportStartupProgress, toolCallId, code, signal, onStream, "
    "onWorkingMessage, onLateSentAgentMessage, ctx) {"
)
EXECUTE_END = "\n}\n/** Turn kernel image attachments"
BUNDLE_EXECUTE_END = "\n}\nfunction imageBlocksFromAttachments"

PATCHED_EXECUTE = f"""async function executeWithBusyKernelChoice(provisioner, reportStartupProgress, toolCallId, code, signal, onStream, onWorkingMessage, onLateSentAgentMessage, ctx) {{
    let kernelRestarted = false;
    while (true) {{
        const m = await provisioner.ensure(reportStartupProgress, signal);
        const timeoutController = new AbortController();
        let timedOut = false;
        const timeout = globalThis.setTimeout(() => {{
            timedOut = true;
            timeoutController.abort();
        }}, IPYTHON_CELL_TIMEOUT_MS);
        timeout.unref?.();
        const linked = createLinkedAbortSignal([signal, timeoutController.signal]);
        try {{
            const result = await m.execute(code, {{
                signal: linked.signal,
                onStream,
                onLateSentAgentMessage: onLateSentAgentMessage
                    ? (message) => onLateSentAgentMessage(toolCallId, message)
                    : undefined,
            }});
            if (timedOut && m.activeExecution) {{
                onWorkingMessage("Restarting IPython kernel after timed-out cell...");
                await provisioner.kill();
                kernelRestarted = true;
            }}
            return {{ result, kernelRestarted, timedOut }};
        }}
        catch (error) {{
            if (!(error instanceof KernelBusyAfterInterruptError) || signal?.aborted) {{
                throw error;
            }}
            const action = await chooseBusyKernelAction(ctx, signal);
            if (action === "wait") {{
                onWorkingMessage("Waiting for IPython kernel...");
                continue;
            }}
            if (action === "kill") {{
                onWorkingMessage("Restarting IPython kernel...");
                await provisioner.kill();
                kernelRestarted = true;
                continue;
            }}
            throw error;
        }}
        finally {{
            globalThis.clearTimeout(timeout);
            linked.cleanup();
        }}
    }}
}}"""

BUNDLE_PATCHED_EXECUTE = f"""async function executeWithBusyKernelChoice(provisioner, reportStartupProgress, toolCallId, code, signal, onStream, onWorkingMessage, onLateSentAgentMessage, ctx) {{
  let kernelRestarted = false;
  while (true) {{
    const m2 = await provisioner.ensure(reportStartupProgress, signal);
    const timeoutController = new AbortController();
    let timedOut = false;
    const timeout = globalThis.setTimeout(() => {{
      timedOut = true;
      timeoutController.abort();
    }}, IPYTHON_CELL_TIMEOUT_MS);
    timeout.unref?.();
    const linked = createLinkedAbortSignal([signal, timeoutController.signal]);
    try {{
      const result = await m2.execute(code, {{
        signal: linked.signal,
        onStream,
        onLateSentAgentMessage: onLateSentAgentMessage ? (message) => onLateSentAgentMessage(toolCallId, message) : void 0
      }});
      if (timedOut && m2.activeExecution) {{
        onWorkingMessage("Restarting IPython kernel after timed-out cell...");
        await provisioner.kill();
        kernelRestarted = true;
      }}
      return {{ result, kernelRestarted, timedOut }};
    }} catch (error) {{
      if (!(error instanceof KernelBusyAfterInterruptError) || signal?.aborted) {{
        throw error;
      }}
      const action = await chooseBusyKernelAction(ctx, signal);
      if (action === "wait") {{
        onWorkingMessage("Waiting for IPython kernel...");
        continue;
      }}
      if (action === "kill") {{
        onWorkingMessage("Restarting IPython kernel...");
        await provisioner.kill();
        kernelRestarted = true;
        continue;
      }}
      throw error;
    }} finally {{
      globalThis.clearTimeout(timeout);
      linked.cleanup();
    }}
  }}
}}"""

DESTRUCTURE_ANCHOR = (
    "const { result: r, kernelRestarted } = await " "executeWithBusyKernelChoice("
)
DESTRUCTURE_PATCH = (
    "const { result: r, kernelRestarted, timedOut } = await "
    "executeWithBusyKernelChoice("
)
ERROR_TEXT_ANCHOR = """                if (r.status === "error" && r.error) {
                    text += (text ? "\\n" : "") + r.error.traceback.join("\\n");
                }
"""
ERROR_TEXT_PATCH = ERROR_TEXT_ANCHOR + """                if (timedOut) {
                    text += (text ? "\\n" : "") + IPYTHON_TIMEOUT_NOTICE;
                }
"""
BUNDLE_ERROR_TEXT_ANCHOR = """        if (r.status === "error" && r.error) {
          text += (text ? "\\n" : "") + r.error.traceback.join("\\n");
        }
"""
BUNDLE_ERROR_TEXT_PATCH = BUNDLE_ERROR_TEXT_ANCHOR + """        if (timedOut) {
          text += (text ? "\\n" : "") + IPYTHON_TIMEOUT_NOTICE;
        }
"""
STATUS_ANCHOR = "                        status: r.status,"
STATUS_PATCH = """                        status: timedOut ? "timeout" : r.status,
                        timedOut,
                        timeoutMs: IPYTHON_CELL_TIMEOUT_MS,"""
BUNDLE_STATUS_ANCHOR = "            status: r.status,"
BUNDLE_STATUS_PATCH = """            status: timedOut ? "timeout" : r.status,
            timedOut,
            timeoutMs: IPYTHON_CELL_TIMEOUT_MS,"""
IS_ERROR_ANCHOR = (
    '                    isError: r.status === "error" || r.status === "aborted",'
)
IS_ERROR_PATCH = (
    '                    isError: timedOut || r.status === "error" || '
    'r.status === "aborted",'
)
BUNDLE_IS_ERROR_ANCHOR = (
    '          isError: r.status === "error" || r.status === "aborted"'
)
BUNDLE_IS_ERROR_PATCH = (
    '          isError: timedOut || r.status === "error" || ' 'r.status === "aborted"'
)
DESCRIPTION_ANCHOR = (
    'description: "Execute Python scratchpad code and `%%bash` shell cells in a '
    "persistent IPython kernel."
)
DESCRIPTION_PATCH = (
    'description: "Execute Python scratchpad code and `%%bash` shell cells in a '
    "persistent IPython kernel. Each call is limited to 60 seconds; a timed-out "
    "cell is interrupted and should be retried as smaller bounded work."
)


def _replace_once(source: str, anchor: str, replacement: str, label: str) -> str:
    if source.count(anchor) != 1:
        raise RuntimeError(
            f"Prime IPython module does not match the pinned runtime at {label}; "
            "refusing an unsafe timeout patch"
        )
    return source.replace(anchor, replacement)


def patch_ipython_module(path: Path) -> bool:
    """Patch one pinned module, returning True only when it changed."""
    source = path.read_text(encoding="utf-8")
    if TIMEOUT_CONSTANT in source:
        return False

    bundled = BUNDLE_CONSTANT_ANCHOR in source
    constant_anchor = BUNDLE_CONSTANT_ANCHOR if bundled else CONSTANT_ANCHOR
    execute_end = BUNDLE_EXECUTE_END if bundled else EXECUTE_END
    patched_execute = BUNDLE_PATCHED_EXECUTE if bundled else PATCHED_EXECUTE
    error_anchor = BUNDLE_ERROR_TEXT_ANCHOR if bundled else ERROR_TEXT_ANCHOR
    error_patch = BUNDLE_ERROR_TEXT_PATCH if bundled else ERROR_TEXT_PATCH
    status_anchor = BUNDLE_STATUS_ANCHOR if bundled else STATUS_ANCHOR
    status_patch = BUNDLE_STATUS_PATCH if bundled else STATUS_PATCH
    is_error_anchor = BUNDLE_IS_ERROR_ANCHOR if bundled else IS_ERROR_ANCHOR
    is_error_patch = BUNDLE_IS_ERROR_PATCH if bundled else IS_ERROR_PATCH

    source = _replace_once(
        source,
        constant_anchor,
        f"{TIMEOUT_CONSTANT}\n{TIMEOUT_NOTICE}\n{constant_anchor}",
        "timeout constants",
    )
    start = source.find(EXECUTE_START)
    end = source.find(execute_end, start)
    if start < 0 or end < 0:
        raise RuntimeError(
            "Prime IPython module does not match the pinned execute helper; "
            "refusing an unsafe timeout patch"
        )
    source = source[:start] + patched_execute + source[end + 2 :]
    source = _replace_once(
        source,
        DESTRUCTURE_ANCHOR,
        DESTRUCTURE_PATCH,
        "timeout result",
    )
    source = _replace_once(
        source,
        error_anchor,
        error_patch,
        "model timeout notice",
    )
    source = _replace_once(source, status_anchor, status_patch, "timeout details")
    source = _replace_once(source, is_error_anchor, is_error_patch, "timeout error")
    source = _replace_once(
        source,
        DESCRIPTION_ANCHOR,
        DESCRIPTION_PATCH,
        "tool description",
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary.write(source)
        temporary_path = Path(temporary.name)
    temporary_path.chmod(path.stat().st_mode)
    os.replace(temporary_path, path)
    return True


def patch_ipython_runtime(path: Path) -> list[Path]:
    """Patch every executable copy of the pinned IPython tool.

    Prime publishes both the source-shaped ``dist/core/tools/ipython.js`` and
    a separately bundled CLI under ``dist/bundle``.  The ``prime-agent``
    executable resolves to the bundle, so patching only the core copy is a
    silent no-op at runtime even though an installation probe can see the
    timeout marker.  Discover by the pinned function/description anchors so a
    future artifact layout change fails closed instead of patching an
    unrelated JavaScript file.
    """
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = []
        for candidate in sorted(path.rglob("*.js")):
            try:
                source = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if EXECUTE_START in source and DESCRIPTION_ANCHOR in source:
                candidates.append(candidate)
    else:
        raise FileNotFoundError(path)

    if not candidates:
        raise RuntimeError(
            f"No pinned Prime IPython module found under {path}; refusing an "
            "unverified timeout installation"
        )
    for candidate in candidates:
        patch_ipython_module(candidate)
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=IPYTHON_DIST_ROOT)
    args = parser.parse_args()
    patched = patch_ipython_runtime(Path(args.path))
    for path in patched:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
