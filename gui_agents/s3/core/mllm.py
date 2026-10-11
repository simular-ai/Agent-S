import base64
from collections.abc import Sequence

import numpy as np

from gui_agents.s3.core.messages import (
    ImageDetail,
    LMMMessage,
    MessageRole,
    TextMessage,
    normalize_messages,
)
from gui_agents.s3.core.lmm_engine import LMMEngine

SingleImage = str | bytes | np.ndarray
ImageContent = SingleImage | list[SingleImage]

from gui_agents.s3.core.engine import (
    LMMEngineAnthropic,
    LMMEngineAzureOpenAI,
    LMMEngineCodex,
    LMMEngineHuggingFace,
    LMMEngineOpenAI,
    LMMEngineOpenRouter,
    LMMEngineParasail,
    LMMEnginevLLM,
    LMMEngineGemini,
)
from gui_agents.s3.core.openai_compatible import (
    OPENAI_COMPATIBLE_ALIASES,
    resolve_lmstudio_params,
)


class LMMAgent:
    def __init__(
        self,
        engine_params=None,
        system_prompt: str | None = None,
        engine: LMMEngine | None = None,
    ):
        self.engine: LMMEngine
        if engine is None:
            if engine_params is not None:
                engine_type = engine_params.get("engine_type")
                if engine_type in OPENAI_COMPATIBLE_ALIASES:
                    # LM Studio / any OpenAI-compatible local server.
                    engine_params = resolve_lmstudio_params(engine_params)
                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "openai":
                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "codex":
                    self.engine = LMMEngineCodex(
                        model=engine_params.get("model"),
                        timeout=engine_params.get("timeout"),
                    )
                elif engine_type == "anthropic":
                    self.engine = LMMEngineAnthropic(**engine_params)
                elif engine_type == "azure":
                    self.engine = LMMEngineAzureOpenAI(**engine_params)
                elif engine_type == "vllm":
                    self.engine = LMMEnginevLLM(**engine_params)
                elif engine_type == "huggingface":
                    self.engine = LMMEngineHuggingFace(**engine_params)
                elif engine_type == "gemini":
                    self.engine = LMMEngineGemini(**engine_params)
                elif engine_type == "open_router":
                    self.engine = LMMEngineOpenRouter(**engine_params)
                elif engine_type == "parasail":
                    self.engine = LMMEngineParasail(**engine_params)
                elif engine_type == "ollama":
                    # Reuse LMMEngineOpenAI for Ollama
                    if not engine_params.get("base_url"):
                        import os

                        base_url = os.getenv("OLLAMA_HOST")
                        if base_url:
                            if not base_url.endswith("/v1"):
                                base_url = base_url.rstrip("/") + "/v1"
                            engine_params["base_url"] = base_url
                        else:
                            # RAISE ERROR instead of default
                            raise ValueError(
                                "Ollama endpoint must be provided via 'base_url' parameter or 'OLLAMA_HOST' environment variable."
                            )
                    if not engine_params.get("api_key"):
                        engine_params["api_key"] = "ollama"
                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "deepseek":
                    if "base_url" not in engine_params:
                        import os

                        base_url = os.getenv("DEEPSEEK_ENDPOINT_URL")
                        if not base_url:
                            base_url = "https://api.deepseek.com"
                        if not base_url.endswith("/v1"):
                            base_url = base_url.rstrip("/") + "/v1"
                        engine_params["base_url"] = base_url

                    if not engine_params.get("api_key"):
                        import os

                        api_key = os.getenv("DEEPSEEK_API_KEY")
                        if not api_key:
                            raise ValueError(
                                "DeepSeek API key must be provided via 'api_key' parameter or 'DEEPSEEK_API_KEY' environment variable."
                            )
                        engine_params["api_key"] = api_key

                    self.engine = LMMEngineOpenAI(**engine_params)
                elif engine_type == "qwen":
                    if not engine_params.get("base_url"):
                        import os

                        base_url = os.getenv("QWEN_ENDPOINT_URL")
                        if not base_url:
                            base_url = (
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"
                            )
                        if not base_url.endswith("/v1"):
                            base_url = base_url.rstrip("/") + "/v1"
                        engine_params["base_url"] = base_url

                    if not engine_params.get("api_key"):
                        import os

                        api_key = os.getenv("QWEN_API_KEY")
                        if not api_key:
                            raise ValueError(
                                "Qwen API key must be provided via 'api_key' parameter or 'QWEN_API_KEY' environment variable."
                            )
                        engine_params["api_key"] = api_key
                    self.engine = LMMEngineOpenAI(**engine_params)
                else:
                    raise ValueError(f"engine_type '{engine_type}' is not supported")
            else:
                raise ValueError("engine_params must be provided")
        else:
            self.engine = engine

        self.messages: list[LMMMessage] = []

        if system_prompt:
            self.add_system_prompt(system_prompt)
        else:
            self.add_system_prompt("You are a helpful assistant.")

    def encode_image(self, image_content: SingleImage) -> str:
        # if image_content is a path to an image file, check type of the image_content to verify
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            image_bytes = (
                image_content.tobytes()
                if isinstance(image_content, np.ndarray)
                else image_content
            )
            return base64.b64encode(image_bytes).decode("utf-8")

    def _append_codex_image(
        self, message: LMMMessage, image_content: SingleImage
    ) -> None:
        from gui_agents.s3.core.codex import encode_image

        payload = base64.b64encode(encode_image(image_content)).decode("utf-8")
        message["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{payload}"},
            }
        )

    def reset(
        self,
    ):

        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        if len(self.messages) > 0:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def remove_message_at(self, index: int) -> None:
        """Remove a message at a given index"""
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(
        self,
        index: int,
        text_content: str,
        image_content: SingleImage | None = None,
        image_detail: ImageDetail = "high",
    ) -> None:
        """Replace a message at a given index"""
        if index < len(self.messages):
            self.messages[index] = {
                "role": self.messages[index]["role"],
                "content": [{"type": "text", "text": text_content}],
            }
            if isinstance(image_content, np.ndarray) or image_content:
                if isinstance(self.engine, LMMEngineCodex):
                    self._append_codex_image(self.messages[index], image_content)
                    return
                base64_image = self.encode_image(image_content)
                self.messages[index]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

    def add_message(
        self,
        text_content: str,
        image_content: ImageContent | None = None,
        role: MessageRole | None = None,
        image_detail: ImageDetail = "high",
        put_text_last: bool = False,
    ) -> None:
        """Add a new message to the list of messages"""

        # API-style inference from OpenAI and AzureOpenAI
        if isinstance(
            self.engine,
            (
                LMMEngineOpenAI,
                LMMEngineCodex,
                LMMEngineAzureOpenAI,
                LMMEngineHuggingFace,
                LMMEngineGemini,
                LMMEngineOpenRouter,
                LMMEngineParasail,
            ),
        ):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message: LMMMessage = {
                "role": role or "user",
                "content": [{"type": "text", "text": text_content}],
            }

            if isinstance(image_content, np.ndarray) or image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        if isinstance(self.engine, LMMEngineCodex):
                            self._append_codex_image(message, image)
                            continue
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    if isinstance(self.engine, LMMEngineCodex):
                        self._append_codex_image(message, image_content)
                    else:
                        base64_image = self.encode_image(image_content)
                        message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": image_detail,
                                },
                            }
                        )

            # Rotate text to be the last message if desired
            if put_text_last:
                text_part = message["content"].pop(0)
                message["content"].append(text_part)

            self.messages.append(message)

        # For API-style inference from Anthropic
        elif isinstance(self.engine, LMMEngineAnthropic):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role or "user",
                "content": [{"type": "text", "text": text_content}],
            }

            if image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image,
                                },
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        }
                    )
            self.messages.append(message)

        # Locally hosted vLLM model inference
        elif isinstance(self.engine, LMMEnginevLLM):
            # infer role from previous message
            if role != "user":
                if self.messages[-1]["role"] == "system":
                    role = "user"
                elif self.messages[-1]["role"] == "user":
                    role = "assistant"
                elif self.messages[-1]["role"] == "assistant":
                    role = "user"

            message = {
                "role": role or "user",
                "content": [{"type": "text", "text": text_content}],
            }

            if image_content:
                # Check if image_content is a list or a single image
                if isinstance(image_content, list):
                    # If image_content is a list of images, loop through each image
                    for image in image_content:
                        base64_image = self.encode_image(image)
                        message["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image;base64,{base64_image}"
                                },
                            }
                        )
                else:
                    # If image_content is a single image, handle it directly
                    base64_image = self.encode_image(image_content)
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image;base64,{base64_image}"},
                        }
                    )

            self.messages.append(message)
        else:
            raise ValueError("engine_type is not supported")

    def get_response(
        self,
        user_message: str | None = None,
        messages: Sequence[LMMMessage | TextMessage] | None = None,
        temperature: float = 0.0,
        max_new_tokens: int | None = None,
        use_thinking: bool = False,
        **kwargs,
    ):
        """Generate the next response based on previous messages"""
        response_messages = (
            self.messages if messages is None else normalize_messages(messages)
        )
        if user_message:
            response_messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            )

        # Regular generation
        if use_thinking:
            return self.engine.generate_with_thinking(
                response_messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        return self.engine.generate(
            response_messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
