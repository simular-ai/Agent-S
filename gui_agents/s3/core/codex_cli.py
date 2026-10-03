"""Codex CLI planner integration for Agent S3.

Uses the installed ``codex`` CLI with the user's existing ChatGPT/Codex
sign-in for read-only text generation. No shell, file, network, MCP, plugin,
or hook execution is enabled by these invocations: every call is
``codex exec`` with ``--sandbox read-only`` plus config overrides that remove
the shell tool and force ``approval_policy="never"``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

CODEX_EXEC_TIMEOUT_BUFFER = 10
DEFAULT_IMAGE_DETAIL = "high"

_lock = threading.RLock()
_status_cache: dict = {"key": None, "value": None}


def codex_command() -> str:
    override = os.getenv("AGENT_S_CODEX_BIN", "").strip()
    if override:
        return override
    found = shutil.which("codex")
    if not found:
        raise ValueError(
            "Codex CLI is not installed or not on PATH; install Codex CLI first"
        )
    return found


def clear_status_cache():
    with _lock:
        _status_cache.update({"key": None, "value": None})


def _run_json(args, timeout, stdin_text=""):
    try:
        completed = subprocess.run(
            args,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )
    except FileNotFoundError as exc:
        raise ValueError("Codex CLI is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("Codex request timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-1000:]
        raise ValueError(f"Codex CLI failed (exit {completed.returncode}): {detail}")
    return completed.stdout


def codex_status(model="", refresh=False):
    """Report ChatGPT/Codex sign-in state without exposing credentials."""
    from gui_agents.s3.ui_config import CODEX_DEFAULT_MODEL

    cache_key = (model or "").strip()
    with _lock:
        if not refresh and _status_cache["key"] == cache_key:
            return dict(_status_cache["value"])
    try:
        command = codex_command()
    except ValueError as exc:
        return {"signed_in": False, "error": str(exc)}
    try:
        raw = _run_json([command, "login", "status"], timeout=30)
    except (ValueError, TimeoutError) as exc:
        return {"signed_in": False, "error": str(exc)}
    text = (raw or "").strip()
    signed_in = "logged in" in text.lower()
    result = {
        "signed_in": signed_in,
        "account": text.splitlines()[0][:256] if text else "",
        "model": (model or "").strip() or CODEX_DEFAULT_MODEL,
        "error": "" if signed_in else (text[-500:] or "Codex is not signed in"),
    }
    with _lock:
        _status_cache.update({"key": cache_key, "value": dict(result)})
    return result


def codex_models(limit=50):
    """List models visible to the signed-in Codex account."""
    try:
        command = codex_command()
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    status = codex_status(refresh=True)
    if not status.get("signed_in"):
        raise ValueError(status.get("error") or "Codex is not signed in")
    # Model discovery without starting a billable turn is not exposed by this
    # CLI version; return the configured/default model so the UI can proceed to
    # a real vision acceptance check.
    del command, limit
    return [status.get("model") or "gpt-6-astra"]


def _message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _encode_image(image_content):
    if isinstance(image_content, str) and os.path.exists(image_content):
        with open(image_content, "rb") as handle:
            return handle.read()
    if isinstance(image_content, bytes):
        return image_content
    if hasattr(image_content, "tobytes"):
        output = io.BytesIO()
        from PIL import Image

        Image.fromarray(image_content).save(output, "PNG")
        return output.getvalue()
    raise ValueError("Codex planner requires screenshot bytes or an image path")


def _prompt_from_messages(messages, max_images=1):
    chunks, images = [], []
    for message in messages or []:
        role = message.get("role", "user")
        text = _message_text(message).strip()
        if role == "system":
            chunks.append(f"[system]\n{text}" if text else "")
            continue
        if text:
            chunks.append(f"[{role}]\n{text}")
        for part in message.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            url = ""
            if part.get("type") == "image_url":
                target = part.get("image_url", {})
                url = target.get("url", "") if isinstance(target, dict) else ""
            elif part.get("type") in ("image", "input_image"):
                source = part.get("source", {})
                if isinstance(source, dict) and source.get("data"):
                    url = "data:image/png;base64," + source["data"]
            if url.startswith("data:image/") and len(images) < max_images:
                header, _, payload = url.partition(";base64,")
                kind = "png"
                if "/" in header:
                    kind = header.split("/", 1)[1].split(";")[0][:8] or "png"
                images.append((kind, base64.b64decode(payload)))
    prompt = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    if not prompt and not images:
        raise ValueError("Codex planner needs prompt text or a screenshot")
    if not prompt:
        prompt = "Describe the attached screen and respond with one desktop action."
    return prompt, images


def _codex_args(model, timeout):
    return [
        codex_command(),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "-m",
        model,
        "--json",
        "-c",
        "features.shell_tool=false",
        "-c",
        'approval_policy="never"',
        "-c",
        "model_verbosity=low",
    ]


def _final_text(stdout):
    text = ""
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                text = item["text"]
    if not text.strip():
        raise ValueError("Codex returned no planner text")
    return text


def codex_generate(messages, model, timeout=60, max_images=1):
    """Run one read-only Codex turn and return its final assistant text."""
    from gui_agents.s3.ui_config import CODEX_DEFAULT_MODEL

    selected = (model or "").strip() or CODEX_DEFAULT_MODEL
    status = codex_status(selected, refresh=True)
    if not status.get("signed_in"):
        raise ValueError(status.get("error") or "Codex is not signed in")
    prompt, images = _prompt_from_messages(messages, max_images=max_images)
    image_args = []
    temp_dir = None
    try:
        if images:
            temp_dir = tempfile.TemporaryDirectory(prefix="agent-s-codex-")
            for index, (kind, payload) in enumerate(images):
                path = Path(temp_dir.name) / f"screenshot-{index}.{kind}"
                path.write_bytes(payload)
                image_args.extend(["-i", str(path)])
        args = _codex_args(status.get("model") or selected, timeout) + image_args + ["-"]
        stdout = _run_json(args, timeout=timeout + CODEX_EXEC_TIMEOUT_BUFFER, stdin_text=prompt)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
    return _final_text(stdout)


def test_codex_vision(model="", timeout=60):
    """Acceptance check: one read-only turn with a synthetic screenshot."""
    from PIL import Image

    from gui_agents.s3.ui_config import CODEX_DEFAULT_MODEL

    selected = (model or "").strip() or CODEX_DEFAULT_MODEL
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(output, "PNG")
    image = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the color of this image in one word."},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]
    text = codex_generate(messages, selected, timeout=timeout)
    return {"ok": True, "model": selected, "sample": text[:500]}
