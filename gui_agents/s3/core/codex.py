"""Codex SDK integration for multimodal planner generation and account access."""

from __future__ import annotations

import base64
import io
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from openai_codex import (
    ApprovalMode,
    Codex,
    CodexConfig,
    ImageInput,
    InputItem,
    RunInput,
    Sandbox,
    TextInput,
)

from gui_agents.s3.core.lmm_engine import LMMEngine
from gui_agents.s3.core.messages import LMMMessage, TextMessage, normalize_messages


def _config() -> CodexConfig:
    return CodexConfig(
        client_name="agent_s3",
        client_title="Agent-S3",
        config_overrides=(
            'forced_login_method="chatgpt"',
            "features.shell_tool=false",
        ),
    )


def _selected_model(model: str | None) -> str:
    from gui_agents.s3.ui_config import CODEX_DEFAULT_MODEL

    return (model or "").strip() or CODEX_DEFAULT_MODEL


def encode_image(image_content: str | bytes | np.ndarray) -> bytes:
    """Encode a screenshot supplied as a path, bytes, or NumPy array."""
    if isinstance(image_content, str):
        return Path(image_content).read_bytes()
    if isinstance(image_content, bytes):
        return image_content
    if isinstance(image_content, np.ndarray):
        from PIL import Image

        output = io.BytesIO()
        Image.fromarray(image_content).save(output, "PNG")
        return output.getvalue()
    raise ValueError(
        "Codex planner requires screenshot bytes, an array, or an image path"
    )


def messages_to_run_input(messages: list[LMMMessage]) -> tuple[str | None, RunInput]:
    """Separate system instructions and preserve role-labelled history and images.

    RunInput has content items but no chat roles. Role markers retain historical
    user/assistant boundaries without replaying old messages as additional turns.
    """
    instructions: list[str] = []
    inputs: list[InputItem] = []
    for message in messages:
        if message["role"] == "system":
            for part in message["content"]:
                if part["type"] != "text":
                    raise ValueError("Codex system instructions must contain text only")
                instructions.append(part["text"])
            continue
        if message["content"]:
            inputs.append(TextInput(text=f"[{message['role']}]"))
        for part in message["content"]:
            if part["type"] == "text":
                inputs.append(TextInput(text=part["text"]))
            elif part["type"] == "image_url":
                inputs.append(ImageInput(url=part["image_url"]["url"]))
            elif part["type"] == "image":
                source = part["source"]
                inputs.append(
                    ImageInput(
                        url=f"data:{source['media_type']};base64,{source['data']}"
                    )
                )
            else:
                raise ValueError(f"Unsupported Codex content type: {part['type']}")
    if not inputs:
        raise ValueError("Codex planner needs a user message or screenshot")
    return "\n\n".join(instructions) or None, inputs


class LMMEngineCodex(LMMEngine):
    """Planner backed by the Codex SDK and a ChatGPT account."""

    def __init__(
        self, model: str | None = None, timeout: float | None = None, **kwargs
    ):
        self.model = _selected_model(model)
        self.timeout = timeout if timeout is not None else 60
        self.config = _config()

    def login(self) -> None:
        with Codex(config=self.config) as codex:
            login = codex.login_chatgpt()
            if not webbrowser.open(login.auth_url):
                login.cancel()
                raise ValueError("Could not open the ChatGPT sign-in page")
            if not login.wait().success:
                raise ValueError("ChatGPT sign-in failed")

    def check_login(self) -> None:
        with Codex(config=self.config) as codex:
            _check_login(codex)

    def get_models(self) -> list[str]:
        return codex_models()

    def generate(
        self,
        messages: list[LMMMessage],
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> str:
        instructions, inputs = messages_to_run_input(messages)
        timed_out = threading.Event()
        with Codex(config=self.config) as codex:

            def expire():
                timed_out.set()
                codex.close()

            timer = threading.Timer(self.timeout, expire)
            timer.daemon = True
            timer.start()
            try:
                _check_login(codex)
                thread = codex.thread_start(
                    model=self.model,
                    developer_instructions=instructions,
                    ephemeral=True,
                    approval_mode=ApprovalMode.deny_all,
                    sandbox=Sandbox.read_only,
                )
                result = thread.run(inputs, **kwargs)
                if timed_out.is_set():
                    raise TimeoutError("Codex request timed out")
                if not result.final_response or not result.final_response.strip():
                    raise ValueError("Codex returned no planner text")
                return result.final_response
            except Exception as exc:
                if timed_out.is_set():
                    raise TimeoutError("Codex request timed out") from exc
                raise ValueError(f"Codex generation failed: {exc}") from exc
            finally:
                timer.cancel()


def _check_login(codex: Codex) -> None:
    account = codex.account().account
    if account is None or account.root.type != "chatgpt":
        raise ValueError("Sign in with your ChatGPT account; use Login with Codex")


def codex_status(model: str = "", refresh: bool = False) -> dict:
    selected = _selected_model(model)
    try:
        with Codex(config=_config()) as codex:
            account = codex.account(refresh_token=refresh).account
            signed_in = account is not None and account.root.type == "chatgpt"
            return {
                "signed_in": signed_in,
                "account": getattr(account.root, "email", "") if account else "",
                "model": selected,
                "error": "" if signed_in else "Codex is not signed in",
            }
    except Exception as exc:
        return {"signed_in": False, "model": selected, "error": str(exc)}


def codex_models(limit: int = 50) -> list[str]:
    try:
        with Codex(config=_config()) as codex:
            _check_login(codex)
            return [model.id for model in codex.models().data][:limit]
    except Exception as exc:
        raise ValueError(f"Codex model discovery failed: {exc}") from exc


def codex_login() -> dict:
    """Start browser sign-in and keep the SDK alive until login completes."""
    codex = Codex(config=_config())
    try:
        login = codex.login_chatgpt()
        if not webbrowser.open(login.auth_url):
            login.cancel()
            raise ValueError("Could not open the ChatGPT sign-in page")
    except Exception:
        codex.close()
        raise

    def finish_login():
        try:
            login.wait()
        finally:
            codex.close()

    threading.Thread(target=finish_login, daemon=True).start()
    return {"started": True, "output": "Complete ChatGPT sign-in in your browser."}


def codex_logout() -> None:
    with Codex(config=_config()) as codex:
        codex.logout()


def codex_generate(
    messages: Sequence[LMMMessage | TextMessage], model: str, timeout: float = 60
) -> str:
    return LMMEngineCodex(model=model, timeout=timeout).generate(
        normalize_messages(messages)
    )


def test_codex_vision(model: str = "", timeout: float = 60) -> dict:
    from PIL import Image

    selected = _selected_model(model)
    output = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(output, "PNG")
    image = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    messages: list[LMMMessage | TextMessage] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe the color of this image in one word.",
                },
                {"type": "image_url", "image_url": {"url": image}},
            ],
        }
    ]
    text = codex_generate(messages, selected, timeout=timeout)
    return {"ok": True, "model": selected, "sample": text[:500]}
